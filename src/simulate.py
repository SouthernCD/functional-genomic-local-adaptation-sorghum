from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import gc
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from numba import njit, prange


import os
import json
import time
import hashlib
import shutil
import tempfile
import gzip
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .ld import prune_sites_by_ld
from .vcf import get_sample_list_from_vcf, get_genotype_df_from_vcf_by_site_df
from .gwas import GEMMA_JOB, plot_gwas_summary_panel
from .enrich import enrichment_permutation_analysis
from .sim.phenotype_simulator_wrap import simulate_phenotype_with_kinship

# ============================================================
# Numba core functions
# ============================================================


@njit
def _fill_undirected_csr(
    src,
    dst,
    weight,
    cursor,
    indices,
    data,
):
    """
    将每条无向边 u--v 写成两条 CSR adjacency：

        u -> v
        v -> u

    直接写入预分配数组，不构造：
        concatenate([src, dst])
        concatenate([dst, src])

    对几亿条边可以显著减少峰值内存。
    """

    m = src.shape[0]

    for e in range(m):

        u = int(src[e])
        v = int(dst[e])
        w = weight[e]

        # u -> v
        p = cursor[u]
        indices[p] = v
        data[p] = w
        cursor[u] += 1

        # v -> u
        p = cursor[v]
        indices[p] = u
        data[p] = w
        cursor[v] += 1


@njit(parallel=True)
def _calculate_initial_state(
    indptr,
    indices,
    data,
    labels,
):
    """
    一次性计算：

        score[i]
        C

    score[i] =
        same-label incident weight
        -
        different-label incident weight

    CSR 是对称的，因此 FG-BG edge 会被看到两次，
    最后 C 除以 2。
    """

    n = labels.shape[0]

    score = np.empty(n, dtype=np.float64)

    cut_twice = 0.0

    for i in prange(n):

        label_i = labels[i]

        local_score = 0.0
        local_cut = 0.0

        start = indptr[i]
        end = indptr[i + 1]

        for p in range(start, end):

            j = indices[p]
            w = data[p]

            if label_i == labels[j]:

                local_score += w

            else:

                local_score -= w
                local_cut += w

        score[i] = local_score

        cut_twice += local_cut

    C = 0.5 * cut_twice

    return score, C


@njit
def _edge_weight_sorted(
    indptr,
    indices,
    data,
    a,
    b,
):
    """
    查询 edge a--b 的 weight。

    CSR 每一行已经排序，因此使用 binary search。

    不存在则返回 0。
    """

    lo = indptr[a]
    hi = indptr[a + 1]

    while lo < hi:

        mid = (lo + hi) // 2

        x = indices[mid]

        if x < b:

            lo = mid + 1

        else:

            hi = mid

    end = indptr[a + 1]

    if lo < end and indices[lo] == b:

        return float(data[lo])

    return 0.0


@njit
def _batch_delta_C(
    indptr,
    indices,
    data,
    score,
    a_nodes,
    b_nodes,
):
    """
    同时计算一批 FG-BG swap proposal 的 ΔC。

    对异标签节点：

        ΔC = score[a] + score[b] + 2*w_ab
    """

    n = a_nodes.shape[0]

    delta = np.empty(n, dtype=np.float64)

    for i in range(n):

        a = a_nodes[i]
        b = b_nodes[i]

        wab = _edge_weight_sorted(
            indptr,
            indices,
            data,
            a,
            b,
        )

        delta[i] = (
            score[a]
            + score[b]
            + 2.0 * wab
        )

    return delta


@njit
def _apply_swap_inplace(
    indptr,
    indices,
    data,
    labels,
    score,
    a,
    b,
):
    """
    真正执行 FG/BG swap。

    只更新：

        a
        b
        neighbors(a)
        neighbors(b)

    不扫描整个 graph。

    返回：
        ΔC
    """

    label_a = labels[a]
    label_b = labels[b]

    if label_a == label_b:

        return 0.0

    # --------------------------------------------------------
    # a-b 是否直接相连
    # --------------------------------------------------------

    wab = _edge_weight_sorted(
        indptr,
        indices,
        data,
        a,
        b,
    )

    old_score_a = score[a]
    old_score_b = score[b]

    # --------------------------------------------------------
    # 预测 C 的变化
    # --------------------------------------------------------

    delta_C = (
        old_score_a
        + old_score_b
        + 2.0 * wab
    )

    # --------------------------------------------------------
    # 更新 a 的所有邻居
    # --------------------------------------------------------

    start = indptr[a]
    end = indptr[a + 1]

    for p in range(start, end):

        k = indices[p]

        # a-b edge 在两者同时 flip 后
        # FG-BG 状态并不会发生变化
        if k == b:
            continue

        w = data[p]

        # swap 前 a-k 是 same
        #
        # +w -> -w
        #
        # 所以 k 的 score -= 2w
        if labels[k] == label_a:

            score[k] -= 2.0 * w

        # swap 前 a-k 是 different
        #
        # -w -> +w
        #
        # score += 2w
        else:

            score[k] += 2.0 * w

    # --------------------------------------------------------
    # 更新 b 的所有邻居
    # --------------------------------------------------------

    start = indptr[b]
    end = indptr[b + 1]

    for p in range(start, end):

        k = indices[p]

        if k == a:
            continue

        w = data[p]

        if labels[k] == label_b:

            score[k] -= 2.0 * w

        else:

            score[k] += 2.0 * w

    # --------------------------------------------------------
    # 更新 a / b 自己的 score
    #
    # 除 a-b edge 外，其余 incident edges 的 same/different
    # 全部反转。
    #
    # a-b 原贡献 = -wab
    # swap 后仍然 = -wab
    #
    # 所以：
    #
    # new_score = -old_score - 2*wab
    # --------------------------------------------------------

    score[a] = (
        -old_score_a
        - 2.0 * wab
    )

    score[b] = (
        -old_score_b
        - 2.0 * wab
    )

    # --------------------------------------------------------
    # 最后再交换 labels
    # --------------------------------------------------------

    labels[a] = label_b
    labels[b] = label_a

    return delta_C


@njit(parallel=True)
def _calculate_expected_random_C_conditional(
    indptr,
    indices,
    data,
    labels,
    movable_mask,
    movable_fg_count,
):
    """
    在 fixed nodes 标签保持不变、movable nodes 中 FG 数量固定的条件下，
    精确计算随机重排 movable labels 时的 E[C]。

    edge 分为三类：

    1) fixed -- fixed
       cut 状态完全固定。

    2) fixed -- movable
       movable endpoint 在固定 FG quota 下随机取标签。

    3) movable -- movable
       在不放回、固定 FG 数量的随机排列下：

           P(labels differ)
           = 2*m*(M-m) / [M*(M-1)]

       其中：
           M = movable node 数
           m = movable FG 数

    CSR 是对称的，因此这里只统计 j > i 的 edge，一条无向边只算一次。
    """

    n = labels.shape[0]

    movable_count = 0
    for i in range(n):
        if movable_mask[i]:
            movable_count += 1

    if movable_count == 0:
        p_movable_fg = 0.0
        p_diff_movable = 0.0
    else:
        p_movable_fg = movable_fg_count / movable_count

        if movable_count >= 2:
            p_diff_movable = (
                2.0
                * movable_fg_count
                * (movable_count - movable_fg_count)
                / (
                    movable_count
                    * (movable_count - 1)
                )
            )
        else:
            p_diff_movable = 0.0

    # 使用 per-node partial sums，避免 Numba parallel reduction
    # 在某些版本下对 expected += local 的 reduction 编译失败。
    partial = np.zeros(n, dtype=np.float64)

    for i in prange(n):

        local = 0.0
        movable_i = movable_mask[i]

        start = indptr[i]
        end = indptr[i + 1]

        for p in range(start, end):

            j = indices[p]

            # 每条无向 edge 只统计一次
            if j <= i:
                continue

            w = data[p]
            movable_j = movable_mask[j]

            # fixed -- fixed
            if (not movable_i) and (not movable_j):

                if labels[i] != labels[j]:
                    local += w

            # movable -- movable
            elif movable_i and movable_j:

                local += w * p_diff_movable

            # fixed -- movable
            else:

                if movable_i:
                    fixed_label = labels[j]
                else:
                    fixed_label = labels[i]

                if fixed_label == 1:
                    # fixed=FG 时，different 要求 movable=BG
                    p_diff = 1.0 - p_movable_fg
                else:
                    # fixed=BG 时，different 要求 movable=FG
                    p_diff = p_movable_fg

                local += w * p_diff

        partial[i] = local

    expected = 0.0
    for i in range(n):
        expected += partial[i]

    return expected


# ============================================================
# Main simulator
# ============================================================


class LDCutSimulator:
    """
    LD-contamination simulator driven by random child-parent mismatch swaps.

    Core design
    -----------
    1. LD_PRUNING_KEEP=True nodes are fixed anchors; False nodes are movable
       children. Each child has one immutable pruning parent, LD_PRUNED_BY.

    2. To INCREASE contamination, randomly swap:

         current FG child with FG parent
             <->
         current BG child with BG parent

       Both children become parent-mismatched. The reverse swap decreases
       contamination. Candidate selection never inspects C, delta_C, node
       degree, or any other full-graph property.

    3. The full LD graph is used only to MEASURE the realized cross-class cut:

         C = sum_{(i,j) in E} w_ij * I(label_i != label_j)

       C_relative is therefore an outcome, not a control variable.

    4. The main control variable is contamination_level:

         (D - D_initial) / (E[D_random] - D_initial)

       where D is the child-parent mismatch count. Thus:
         0 -> initialized parent-cohesive state
         1 -> random exact-quota mismatch expectation
         >1 -> anti-inheritance / stress-test region

    5. Every swap preserves fixed-anchor labels and the exact movable FG quota.
    """

    def __init__(
        self,
        pruned_context_df,
        ld_df,
        source_col="INDEX_1",
        target_col="INDEX_2",
        weight_col="R2",
        weight_dtype=np.float32,
        verbose=True,
    ):
        """
        Parameters
        ----------
        pruned_context_df : pandas.DataFrame
            Must contain:
                LD_PRUNING_KEEP
                LD_PRUNED_BY
                foreground

            Node IDs are taken from pruned_context_df.index.

            The index must contain exactly:
                0, 1, ..., N-1

            Row order itself does NOT matter; the DataFrame is internally
            reordered by index before arrays are constructed.

            foreground:
                bool or 0/1
                True/1 = FG
                False/0 = BG

            LD_PRUNING_KEEP:
                bool or 0/1
                True  = fixed anchor
                False = movable child

            LD_PRUNED_BY:
                For every movable child, must contain the node ID/index of
                the retained fixed anchor that pruned it.
                Fixed anchors may contain NA.

        ld_df : pandas.DataFrame
            One undirected LD edge per row is recommended.

            Required columns:
                source_col
                target_col
                weight_col

            Node IDs must use the same 0..N-1 IDs as
            pruned_context_df.index.

        weight_col : str
            Edge weight used by global graph cut C.

            "R2" is supported.
            If the final overlap metric uses scaled R2, using an already
            prepared "R2_scale" column is usually preferable.

        weight_dtype :
            CSR edge-weight dtype. float32 is recommended for very large
            graphs. score and C remain float64.
        """

        # ====================================================
        # 1. Validate and normalize pruned_context_df
        # ====================================================

        if not isinstance(pruned_context_df, pd.DataFrame):
            raise TypeError(
                "pruned_context_df must be a pandas DataFrame."
            )

        required_context_cols = [
            "LD_PRUNING_KEEP",
            "LD_PRUNED_BY",
            "foreground",
        ]

        missing = [
            c
            for c in required_context_cols
            if c not in pruned_context_df.columns
        ]

        if missing:
            raise ValueError(
                "pruned_context_df is missing required columns: "
                + ", ".join(missing)
            )

        if not pruned_context_df.index.is_unique:
            raise ValueError(
                "pruned_context_df.index must be unique."
            )

        self.n_nodes = len(pruned_context_df)

        if self.n_nodes == 0:
            raise ValueError(
                "pruned_context_df contains no nodes."
            )

        # The graph implementation uses direct integer indexing.
        # Allow arbitrary row order, but require the index labels themselves
        # to be exactly 0..N-1.
        try:
            context_index_numeric = np.asarray(
                pruned_context_df.index,
                dtype=np.int64,
            )
        except Exception as exc:
            raise ValueError(
                "pruned_context_df.index must contain integer node IDs "
                "0..N-1."
            ) from exc

        expected_ids = np.arange(
            self.n_nodes,
            dtype=np.int64,
        )

        if (
            len(context_index_numeric) != self.n_nodes
            or not np.array_equal(
                np.sort(context_index_numeric),
                expected_ids,
            )
        ):
            raise ValueError(
                "pruned_context_df.index must contain exactly the node IDs "
                "0..N-1 used by ld_df."
            )

        # Reorder by node ID so array position == graph node ID.
        context = pruned_context_df.loc[
            expected_ids
        ]

        # Keep a lightweight reference to the normalized view for diagnostics.
        # The simulator's state is stored in NumPy arrays; external mutation of
        # the original DataFrame does not change simulator labels.
        self.pruned_context_df = context

        # ---------------------------
        # foreground -> int8 labels
        # ---------------------------

        foreground_raw = context[
            "foreground"
        ]

        if foreground_raw.isna().any():
            raise ValueError(
                "pruned_context_df['foreground'] contains missing values."
            )

        fg_values = set(
            pd.unique(
                foreground_raw
            )
        )

        if not fg_values <= {
            0, 1, False, True
        }:
            raise ValueError(
                "foreground must contain only True/False or 0/1."
            )

        labels = foreground_raw.astype(
            bool
        ).to_numpy(
            dtype=np.int8
        )

        # --------------------------------
        # LD_PRUNING_KEEP -> fixed/movable
        # --------------------------------

        keep_raw = context[
            "LD_PRUNING_KEEP"
        ]

        if keep_raw.isna().any():
            raise ValueError(
                "LD_PRUNING_KEEP contains missing values."
            )

        keep_values = set(
            pd.unique(
                keep_raw
            )
        )

        if not keep_values <= {
            0, 1, False, True
        }:
            raise ValueError(
                "LD_PRUNING_KEEP must contain only True/False or 0/1."
            )

        keep_mask = keep_raw.astype(
            bool
        ).to_numpy(
            dtype=bool
        )

        movable_mask = ~keep_mask

        if not keep_mask.any():
            raise ValueError(
                "No fixed anchors found: LD_PRUNING_KEEP has no True rows."
            )

        if not movable_mask.any():
            raise ValueError(
                "No movable children found: LD_PRUNING_KEEP has no False rows."
            )

        self.labels = labels.copy()
        self.movable_mask = movable_mask.copy()
        self.keep_mask = keep_mask.copy()

        self._fixed_node_ids = np.flatnonzero(
            self.keep_mask
        )

        self._movable_node_ids = np.flatnonzero(
            self.movable_mask
        )

        self._fixed_labels_reference = self.labels[
            self._fixed_node_ids
        ].copy()

        # ====================================================
        # 2. Parse LD_PRUNED_BY and establish latent parent
        # ====================================================

        parent_id = np.full(
            self.n_nodes,
            -1,
            dtype=np.int64,
        )

        parent_raw = context.loc[
            self.movable_mask,
            "LD_PRUNED_BY",
        ]

        if parent_raw.isna().any():
            bad_nodes = parent_raw.index[
                parent_raw.isna()
            ].to_numpy()

            preview = bad_nodes[:10].tolist()

            raise ValueError(
                "Every movable child must have LD_PRUNED_BY. "
                f"Missing for {len(bad_nodes):,} movable nodes; "
                f"examples: {preview}"
            )

        try:
            parent_numeric = pd.to_numeric(
                parent_raw,
                errors="raise",
            ).to_numpy(
                dtype=np.float64
            )
        except Exception as exc:
            raise ValueError(
                "LD_PRUNED_BY values for movable children must be integer "
                "node IDs from pruned_context_df.index."
            ) from exc

        if not np.all(
            np.isfinite(
                parent_numeric
            )
        ):
            raise ValueError(
                "LD_PRUNED_BY contains non-finite values for movable children."
            )

        if not np.all(
            parent_numeric
            == np.floor(
                parent_numeric
            )
        ):
            raise ValueError(
                "LD_PRUNED_BY must contain integer node IDs."
            )

        parent_numeric = parent_numeric.astype(
            np.int64
        )

        if (
            np.any(parent_numeric < 0)
            or np.any(
                parent_numeric
                >= self.n_nodes
            )
        ):
            raise ValueError(
                "LD_PRUNED_BY contains node IDs outside 0..N-1."
            )

        child_ids = self._movable_node_ids

        parent_id[
            child_ids
        ] = parent_numeric

        if np.any(
            parent_id[
                child_ids
            ]
            == child_ids
        ):
            bad = child_ids[
                parent_id[
                    child_ids
                ]
                == child_ids
            ]

            raise ValueError(
                "A movable child cannot be its own LD_PRUNED_BY parent. "
                f"Examples: {bad[:10].tolist()}"
            )

        parent_is_fixed = self.keep_mask[
            parent_id[
                child_ids
            ]
        ]

        if not np.all(
            parent_is_fixed
        ):
            bad_children = child_ids[
                ~parent_is_fixed
            ]

            bad_pairs = [
                (
                    int(i),
                    int(parent_id[i]),
                )
                for i in bad_children[:10]
            ]

            raise ValueError(
                "Every LD_PRUNED_BY parent must be a retained/fixed node "
                "(LD_PRUNING_KEEP=True). "
                f"Invalid child->parent examples: {bad_pairs}"
            )

        self.parent_id = parent_id

        # preferred_label is the immutable latent direction for each child.
        # -1 is used for fixed anchors because they do not need a preference.
        self.preferred_label = np.full(
            self.n_nodes,
            -1,
            dtype=np.int8,
        )

        self.preferred_label[
            child_ids
        ] = self.labels[
            parent_id[
                child_ids
            ]
        ]

        # Save initialization state after parent mapping is established.
        self.initial_labels = self.labels.copy()

        # ====================================================
        # 3. Build fixed undirected LD CSR graph
        # ====================================================

        if not isinstance(ld_df, pd.DataFrame):
            raise TypeError(
                "ld_df must be a pandas DataFrame."
            )

        for col in (
            source_col,
            target_col,
            weight_col,
        ):
            if col not in ld_df.columns:
                raise ValueError(
                    f"ld_df is missing required column: {col}"
                )

        if verbose:
            print(
                f"Nodes: {self.n_nodes:,}"
            )
            print(
                f"Fixed anchors: {len(self._fixed_node_ids):,}"
            )
            print(
                f"Movable children: {len(self._movable_node_ids):,}"
            )
            print(
                f"Input edges: {len(ld_df):,}"
            )

        src = ld_df[
            source_col
        ].to_numpy(
            copy=False
        )

        dst = ld_df[
            target_col
        ].to_numpy(
            copy=False
        )

        weight = ld_df[
            weight_col
        ].to_numpy(
            dtype=weight_dtype,
            copy=False,
        )

        if len(src) == 0:
            raise ValueError(
                "ld_df contains no edges."
            )

        min_node = min(
            int(src.min()),
            int(dst.min()),
        )

        max_node = max(
            int(src.max()),
            int(dst.max()),
        )

        if min_node < 0:
            raise ValueError(
                "ld_df node IDs must be >= 0."
            )

        if max_node >= self.n_nodes:
            raise ValueError(
                "ld_df contains node IDs outside pruned_context_df index."
            )

        if not np.all(
            np.isfinite(
                weight
            )
        ):
            raise ValueError(
                f"ld_df[{weight_col!r}] contains non-finite weights."
            )

        if np.any(
            weight < 0
        ):
            raise ValueError(
                f"ld_df[{weight_col!r}] contains negative weights."
            )

        if verbose:
            print(
                "Calculating node degrees..."
            )

        degree = np.bincount(
            src,
            minlength=self.n_nodes,
        ).astype(
            np.int64,
            copy=False,
        )

        degree += np.bincount(
            dst,
            minlength=self.n_nodes,
        )

        indptr = np.empty(
            self.n_nodes + 1,
            dtype=np.int64,
        )

        indptr[0] = 0

        np.cumsum(
            degree,
            out=indptr[1:],
        )

        nnz = int(
            indptr[-1]
        )

        if verbose:
            print(
                f"Undirected CSR entries: {nnz:,}"
            )

        del degree

        index_dtype = (
            np.int32
            if self.n_nodes
            < np.iinfo(
                np.int32
            ).max
            else np.int64
        )

        self.index_dtype = index_dtype

        indices = np.empty(
            nnz,
            dtype=index_dtype,
        )

        data = np.empty(
            nnz,
            dtype=weight_dtype,
        )

        cursor = indptr[:-1].copy()

        if verbose:
            print(
                "Building undirected CSR..."
            )

        _fill_undirected_csr(
            src,
            dst,
            weight,
            cursor,
            indices,
            data,
        )

        del cursor
        del src
        del dst
        del weight

        gc.collect()

        A = csr_matrix(
            (
                data,
                indices,
                indptr,
            ),
            shape=(
                self.n_nodes,
                self.n_nodes,
            ),
            copy=False,
        )

        # Duplicate input edges are collapsed exactly as in the previous
        # implementation.
        A.sum_duplicates()

        A.setdiag(0)
        A.eliminate_zeros()

        # Binary-search edge lookup requires sorted CSR rows.
        A.sort_indices()

        self.A = A
        self.indptr = A.indptr
        self.indices = A.indices
        self.data = A.data

        # ====================================================
        # 4. Initial global graph-cut state
        # ====================================================

        if verbose:
            print(
                "Calculating initial C and node scores..."
            )

        self.score, self.C = (
            _calculate_initial_state(
                self.indptr,
                self.indices,
                self.data,
                self.labels,
            )
        )

        self.total_weight = (
            0.5
            * self.data.sum(
                dtype=np.float64
            )
        )

        # Movable FG/BG candidate pools.
        self._rebuild_movable_pools()

        # Conditional random baseline for global C.
        self._refresh_expected_random_C()

        # Parent-guidance diagnostics.
        self._refresh_parent_state()

        # Fixed baseline for the intervention coordinate.
        self._initial_parent_mismatch_count = int(
            self.parent_mismatch_count
        )

        if verbose:
            print(
                f"Total FG nodes: "
                f"{int(np.sum(self.labels == 1)):,}"
            )
            print(
                f"Total BG nodes: "
                f"{int(np.sum(self.labels == 0)):,}"
            )
            print(
                f"Movable FG nodes: {len(self.fg_nodes):,}"
            )
            print(
                f"Movable BG nodes: {len(self.bg_nodes):,}"
            )
            print(
                f"C: {self.C:,.6f}"
            )
            print(
                f"C / total weight: "
                f"{self.C_fraction:.6f}"
            )
            print(
                f"C / E[C_random | fixed nodes]: "
                f"{self.C_relative:.6f}"
            )
            print(
                f"Parent match fraction: "
                f"{self.parent_match_fraction:.6f}"
            )
            print(
                f"Parent mismatch relative to random: "
                f"{self.parent_mismatch_relative:.6f}"
            )
            print(
                f"Contamination level: "
                f"{self.contamination_level:.6f}"
            )

    # ========================================================
    # movable-pool / parent-state helpers
    # ========================================================

    def _rebuild_movable_pools(
        self,
    ):
        """
        Rebuild current movable FG/BG pools.

        Fixed anchors never enter these pools.
        """

        self.fg_nodes = np.flatnonzero(
            (self.labels == 1)
            & self.movable_mask
        ).astype(
            self.index_dtype,
            copy=False,
        )

        self.bg_nodes = np.flatnonzero(
            (self.labels == 0)
            & self.movable_mask
        ).astype(
            self.index_dtype,
            copy=False,
        )

        self.pool_pos = np.full(
            self.n_nodes,
            -1,
            dtype=np.int64,
        )

        self.pool_pos[
            self.fg_nodes
        ] = np.arange(
            len(self.fg_nodes),
            dtype=np.int64,
        )

        self.pool_pos[
            self.bg_nodes
        ] = np.arange(
            len(self.bg_nodes),
            dtype=np.int64,
        )

    def _refresh_expected_random_C(
        self,
    ):
        """
        Exact E[C] when:
            - fixed-anchor labels are held fixed;
            - only movable labels are randomized;
            - movable FG count is held fixed.
        """

        self._expected_random_C = float(
            _calculate_expected_random_C_conditional(
                self.indptr,
                self.indices,
                self.data,
                self.labels,
                self.movable_mask,
                len(self.fg_nodes),
            )
        )

    def _refresh_preferred_labels_from_parents(
        self,
    ):
        """
        Recompute child preferred labels from current fixed-parent labels.

        Normal simulations never change fixed anchors, so this is normally
        unnecessary after initialization. It is used only when set_labels()
        is explicitly called with require_fixed_unchanged=False.
        """

        child_ids = self._movable_node_ids

        self.preferred_label[
            child_ids
        ] = self.labels[
            self.parent_id[
                child_ids
            ]
        ]

    def _calculate_parent_mismatch_count(
        self,
    ):
        child_ids = self._movable_node_ids

        return int(
            np.count_nonzero(
                self.labels[
                    child_ids
                ]
                != self.preferred_label[
                    child_ids
                ]
            )
        )

    def _refresh_expected_random_parent_mismatch(
        self,
    ):
        """
        Expected number of parent-tag mismatches under a random exact-quota
        assignment of movable FG labels.

        Let:
            M = number of movable children
            m = number of movable FG children
            P = number of children whose preferred(parent) label is FG

        Every movable node has marginal P(current FG)=m/M under a uniform
        exact-quota permutation, so:

            E[mismatch]
                = P * (1 - m/M)
                  + (M-P) * (m/M)
        """

        M = int(
            len(
                self._movable_node_ids
            )
        )

        if M == 0:
            self._expected_random_parent_mismatch = 0.0
            return

        m = int(
            len(
                self.fg_nodes
            )
        )

        child_pref = self.preferred_label[
            self._movable_node_ids
        ]

        P = int(
            np.count_nonzero(
                child_pref == 1
            )
        )

        p_fg = (
            m
            / M
        )

        expected = (
            P
            * (1.0 - p_fg)
            +
            (M - P)
            * p_fg
        )

        self._expected_random_parent_mismatch = float(
            expected
        )

    def _refresh_parent_state(
        self,
    ):
        self.parent_mismatch_count = (
            self._calculate_parent_mismatch_count()
        )

        self._refresh_expected_random_parent_mismatch()

    def _parent_guided_mask(
        self,
        fg_nodes,
        bg_nodes,
        direction,
    ):
        """
        Return legal LD_PRUNED_BY-guided swaps.

        For a current FG node a and current BG node b:

        decrease:
            preferred[a] == BG
            preferred[b] == FG

            Both nodes currently mismatch their own parents; after swapping,
            both match.

        increase:
            preferred[a] == FG
            preferred[b] == BG

            Both nodes currently match their own parents; after swapping,
            both mismatch.
        """

        if direction not in (
            "decrease",
            "increase",
        ):
            raise ValueError(
                "direction must be 'decrease' or 'increase'"
            )

        pref_fg = self.preferred_label[
            fg_nodes
        ]

        pref_bg = self.preferred_label[
            bg_nodes
        ]

        if direction == "decrease":
            return (
                (pref_fg == 0)
                & (pref_bg == 1)
            )

        return (
            (pref_fg == 1)
            & (pref_bg == 0)
        )

    # ========================================================
    # state replacement / reset
    # ========================================================

    def set_labels(
        self,
        labels,
        require_fixed_unchanged=True,
    ):
        """
        Replace the complete current 0/1 labeling without rebuilding the graph.

        By default, fixed anchors must remain exactly as they were at
        simulator initialization.
        """

        labels = np.asarray(
            labels,
            dtype=np.int8,
        )

        if (
            labels.ndim != 1
            or len(labels) != self.n_nodes
        ):
            raise ValueError(
                "labels must be a 1D array with length n_nodes."
            )

        unique_labels = np.unique(
            labels
        )

        if not np.all(
            np.isin(
                unique_labels,
                [0, 1],
            )
        ):
            raise ValueError(
                "labels must contain only 0 and 1."
            )

        if require_fixed_unchanged:
            if not np.array_equal(
                labels[
                    self._fixed_node_ids
                ],
                self._fixed_labels_reference,
            ):
                raise ValueError(
                    "Labels of fixed anchors cannot be changed."
                )

        fixed_changed = (
            not np.array_equal(
                labels[
                    self._fixed_node_ids
                ],
                self.labels[
                    self._fixed_node_ids
                ],
            )
        )

        self.labels[:] = labels

        self.score, self.C = (
            _calculate_initial_state(
                self.indptr,
                self.indices,
                self.data,
                self.labels,
            )
        )

        self._rebuild_movable_pools()
        self._refresh_expected_random_C()

        if fixed_changed:
            self._refresh_preferred_labels_from_parents()

        self._refresh_parent_state()

        if fixed_changed:
            self._initial_parent_mismatch_count = int(
                self.parent_mismatch_count
            )

        return self.C

    def reset_labels(
        self,
    ):
        """
        Restore simulator-construction labels.
        """

        return self.set_labels(
            self.initial_labels,
            require_fixed_unchanged=True,
        )

    def get_labels(
        self,
        copy=True,
    ):
        if copy:
            return self.labels.copy()

        return self.labels

    def get_context_df(
        self,
        copy=True,
        foreground_as_bool=True,
    ):
        """
        Return pruned_context_df with foreground replaced by current labels.

        This is only a convenience method; internal simulator state is kept in
        NumPy arrays.
        """

        if copy:
            out = self.pruned_context_df.copy()
        else:
            out = self.pruned_context_df

        if foreground_as_bool:
            out[
                "foreground"
            ] = self.labels.astype(
                bool
            )
        else:
            out[
                "foreground"
            ] = self.labels

        return out

    # ========================================================
    # metrics
    # ========================================================

    @property
    def C_fraction(
        self,
    ):
        if self.total_weight == 0:
            return np.nan

        return (
            self.C
            / self.total_weight
        )

    @property
    def expected_random_C(
        self,
    ):
        return self._expected_random_C

    @property
    def C_relative(
        self,
    ):
        expected = self.expected_random_C

        if expected == 0:
            return np.nan

        return (
            self.C
            / expected
        )

    @property
    def parent_match_count(
        self,
    ):
        return (
            len(
                self._movable_node_ids
            )
            - self.parent_mismatch_count
        )

    @property
    def parent_match_fraction(
        self,
    ):
        M = len(
            self._movable_node_ids
        )

        if M == 0:
            return np.nan

        return (
            self.parent_match_count
            / M
        )

    @property
    def parent_mismatch_fraction(
        self,
    ):
        M = len(
            self._movable_node_ids
        )

        if M == 0:
            return np.nan

        return (
            self.parent_mismatch_count
            / M
        )

    @property
    def expected_random_parent_mismatch(
        self,
    ):
        return (
            self._expected_random_parent_mismatch
        )

    @property
    def parent_mismatch_relative(
        self,
    ):
        expected = (
            self.expected_random_parent_mismatch
        )

        if expected == 0:
            return np.nan

        return (
            self.parent_mismatch_count
            / expected
        )

    @property
    def initial_parent_mismatch_count(
        self,
    ):
        return int(self._initial_parent_mismatch_count)

    @property
    def initial_parent_mismatch_relative(
        self,
    ):
        expected = self.expected_random_parent_mismatch
        if expected == 0:
            return np.nan
        return self.initial_parent_mismatch_count / expected

    @property
    def contamination_level(
        self,
    ):
        """
        Progress from initialization toward random child-parent association.

            0 : initialized state
            1 : random exact-quota mismatch expectation
           >1 : anti-inheritance / stress-test region
        """
        D0 = float(self.initial_parent_mismatch_count)
        Drandom = float(self.expected_random_parent_mismatch)
        denom = Drandom - D0

        if denom <= 0:
            return np.nan

        return (float(self.parent_mismatch_count) - D0) / denom

    # ========================================================
    # graph queries
    # ========================================================

    def neighbors(
        self,
        node,
    ):
        start = self.indptr[
            node
        ]

        end = self.indptr[
            node + 1
        ]

        return (
            self.indices[
                start:end
            ],
            self.data[
                start:end
            ],
        )

    def edge_weight(
        self,
        a,
        b,
    ):
        return (
            _edge_weight_sorted(
                self.indptr,
                self.indices,
                self.data,
                int(a),
                int(b),
            )
        )

    # ========================================================
    # proposal calculations
    # ========================================================

    def delta_C(
        self,
        a,
        b,
    ):
        """
        Predict global graph-cut change:
            ΔC = C_after - C_before
        """

        a = int(a)
        b = int(b)

        if not (
            0 <= a < self.n_nodes
            and 0 <= b < self.n_nodes
        ):
            raise IndexError(
                "Node ID out of range."
            )

        if (
            not self.movable_mask[a]
            or not self.movable_mask[b]
        ):
            raise ValueError(
                "delta_C is only defined for movable swap candidates."
            )

        if (
            self.labels[a]
            == self.labels[b]
        ):
            return 0.0

        wab = self.edge_weight(
            a,
            b,
        )

        return float(
            self.score[a]
            + self.score[b]
            + 2.0 * wab
        )

    def delta_parent_mismatch(
        self,
        a,
        b,
    ):
        """
        Predict change in the number of child-parent tag mismatches.

        Returns:
            mismatch_after - mismatch_before

        For parent-guided swaps this is exactly:
            -2 for decrease
            +2 for increase
        """

        a = int(a)
        b = int(b)

        if (
            not self.movable_mask[a]
            or not self.movable_mask[b]
        ):
            raise ValueError(
                "delta_parent_mismatch is only defined for movable nodes."
            )

        label_a = int(
            self.labels[a]
        )
        label_b = int(
            self.labels[b]
        )

        if label_a == label_b:
            return 0

        before = (
            int(
                label_a
                != self.preferred_label[a]
            )
            +
            int(
                label_b
                != self.preferred_label[b]
            )
        )

        after = (
            int(
                label_b
                != self.preferred_label[a]
            )
            +
            int(
                label_a
                != self.preferred_label[b]
            )
        )

        return int(
            after - before
        )

    def propose_random_parent_swap(
        self,
        direction="increase",
        rng=None,
        max_attempts=100_000,
    ):
        """
        Draw one parent-guided FG<->BG swap without consulting the LD graph.

        increase:
            FG child / FG parent <-> BG child / BG parent
            Both children become parent-mismatched (delta D = +2).

        decrease:
            FG child / BG parent <-> BG child / FG parent
            Both children become parent-matched (delta D = -2).

        Rejection sampling from the current FG and BG pools makes the accepted
        node uniform within each eligible category. No C or delta_C criterion
        is used.
        """
        if direction not in ("increase", "decrease"):
            raise ValueError("direction must be 'increase' or 'decrease'")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")

        if rng is None:
            rng = np.random.default_rng()

        if len(self.fg_nodes) == 0 or len(self.bg_nodes) == 0:
            return None, None

        if direction == "increase":
            wanted_fg_parent = 1
            wanted_bg_parent = 0
        else:
            wanted_fg_parent = 0
            wanted_bg_parent = 1

        a = None
        for _ in range(max_attempts):
            candidate = int(
                self.fg_nodes[rng.integers(0, len(self.fg_nodes))]
            )
            if self.preferred_label[candidate] == wanted_fg_parent:
                a = candidate
                break

        if a is None:
            return None, None

        b = None
        for _ in range(max_attempts):
            candidate = int(
                self.bg_nodes[rng.integers(0, len(self.bg_nodes))]
            )
            if self.preferred_label[candidate] == wanted_bg_parent:
                b = candidate
                break

        if b is None:
            return None, None

        return a, b

    # ========================================================
    # execute swap
    # ========================================================

    def swap(
        self,
        a,
        b,
    ):
        """
        Execute one movable FG <-> movable BG swap.

        Global C and parent-mismatch count are both updated incrementally.
        """

        a = int(a)
        b = int(b)

        if not (
            0 <= a < self.n_nodes
            and 0 <= b < self.n_nodes
        ):
            raise IndexError(
                "Node ID out of range."
            )

        if (
            not self.movable_mask[a]
            or not self.movable_mask[b]
        ):
            raise ValueError(
                "Only movable children may be swapped."
            )

        label_a = int(
            self.labels[a]
        )

        label_b = int(
            self.labels[b]
        )

        if label_a == label_b:
            return 0.0

        delta_parent = (
            self.delta_parent_mismatch(
                a,
                b,
            )
        )

        pos_a = int(
            self.pool_pos[a]
        )

        pos_b = int(
            self.pool_pos[b]
        )

        delta = (
            _apply_swap_inplace(
                self.indptr,
                self.indices,
                self.data,
                self.labels,
                self.score,
                a,
                b,
            )
        )

        self.C += delta

        self.parent_mismatch_count += (
            delta_parent
        )

        # O(1) update of movable FG/BG pools.
        if (
            label_a == 1
            and label_b == 0
        ):
            self.fg_nodes[
                pos_a
            ] = b

            self.bg_nodes[
                pos_b
            ] = a

            self.pool_pos[b] = pos_a
            self.pool_pos[a] = pos_b

        else:
            # a was BG and b was FG.
            self.bg_nodes[
                pos_a
            ] = b

            self.fg_nodes[
                pos_b
            ] = a

            self.pool_pos[b] = pos_a
            self.pool_pos[a] = pos_b

        return float(
            delta
        )

    # ========================================================
    # random parent-mismatch intervention
    # ========================================================

    def move_contamination(
        self,
        n_swaps,
        direction="increase",
        seed=None,
        max_attempts_per_swap=100_000,
        record_every=1,
        verbose_every=1_000,
    ):
        """
        Perform random parent-guided swaps.

        C and C_relative are updated and recorded, but never used to choose or
        accept a swap.
        """
        if direction not in ("increase", "decrease"):
            raise ValueError("direction must be 'increase' or 'decrease'")
        if n_swaps < 0:
            raise ValueError("n_swaps must be >= 0")

        rng = np.random.default_rng(seed)
        history = []
        accepted = 0

        while accepted < n_swaps:
            a, b = self.propose_random_parent_swap(
                direction=direction,
                rng=rng,
                max_attempts=max_attempts_per_swap,
            )
            if a is None:
                break

            actual_delta_C = self.swap(a, b)
            accepted += 1

            if record_every and accepted % record_every == 0:
                history.append((
                    accepted, a, b, actual_delta_C,
                    self.parent_mismatch_count,
                    self.parent_mismatch_relative,
                    self.contamination_level,
                    self.parent_match_fraction,
                    self.C, self.C_relative,
                ))

            if verbose_every and accepted % verbose_every == 0:
                print(
                    f"{accepted:,} swaps | "
                    f"contamination={self.contamination_level:.6f} | "
                    f"parent_mismatch_rel={self.parent_mismatch_relative:.6f} | "
                    f"C_rel={self.C_relative:.6f} | "
                    f"dC={actual_delta_C:,.6f}"
                )

        return pd.DataFrame(
            history,
            columns=[
                "n_swaps", "FG_node", "BG_node", "delta_C",
                "parent_mismatch_count", "parent_mismatch_relative",
                "contamination_level", "parent_match_fraction",
                "C", "C_relative",
            ],
        )

    def move_through_contamination_targets(
        self,
        targets,
        direction="increase",
        seed=None,
        max_swaps_per_target=1_000_000,
        max_attempts_per_swap=100_000,
        verbose_every=10_000,
    ):
        """
        Visit a sequence of contamination_level targets.

        contamination_level is the CONTROL variable:

            (D - D_initial) / (E[D_random] - D_initial)

        D is child-parent mismatch count. Therefore 0 is the initialized state,
        1 is the random exact-quota expectation, and values >1 are an optional
        anti-inheritance stress-test region.

        C_relative is an OUTCOME. It is not required to move monotonically and
        is never used for candidate selection.
        """
        targets = np.asarray(targets, dtype=np.float64)

        if targets.ndim != 1 or len(targets) == 0:
            raise ValueError("targets must be a non-empty 1D sequence")
        if not np.all(np.isfinite(targets)):
            raise ValueError("targets must contain only finite values")
        if direction not in ("increase", "decrease"):
            raise ValueError("direction must be 'increase' or 'decrease'")
        if max_swaps_per_target <= 0:
            raise ValueError("max_swaps_per_target must be > 0")

        current_level = float(self.contamination_level)
        if not np.isfinite(current_level):
            raise ValueError(
                "contamination_level is undefined: initial mismatch must be "
                "below the random expectation"
            )

        if direction == "increase":
            if np.any(np.diff(targets) < 0):
                raise ValueError("increase targets must be nondecreasing")
            if np.any(targets < current_level - 1e-12):
                raise ValueError(
                    "all increase targets must be >= current contamination_level"
                )
        else:
            if np.any(np.diff(targets) > 0):
                raise ValueError("decrease targets must be nonincreasing")
            if np.any(targets > current_level + 1e-12):
                raise ValueError(
                    "all decrease targets must be <= current contamination_level"
                )

        rng = np.random.default_rng(seed)
        labels_by_target = {}
        summary = []
        total_accepted = 0

        for target in targets:
            target = float(target)
            start_level = float(self.contamination_level)
            start_D = int(self.parent_mismatch_count)
            start_C = float(self.C)
            start_C_relative = float(self.C_relative)
            accepted = 0

            def reached():
                if direction == "increase":
                    return self.contamination_level >= target
                return self.contamination_level <= target

            while not reached() and accepted < max_swaps_per_target:
                a, b = self.propose_random_parent_swap(
                    direction=direction,
                    rng=rng,
                    max_attempts=max_attempts_per_swap,
                )
                if a is None:
                    break

                self.swap(a, b)
                accepted += 1
                total_accepted += 1

                if verbose_every and accepted % verbose_every == 0:
                    print(
                        f"target={target:.4f} | "
                        f"target swaps={accepted:,} | "
                        f"total swaps={total_accepted:,} | "
                        f"contamination={self.contamination_level:.6f} | "
                        f"parent_mismatch_rel={self.parent_mismatch_relative:.6f} | "
                        f"C_rel={self.C_relative:.6f}"
                    )

            achieved = float(self.contamination_level)
            reached_target = bool(reached())
            labels_by_target[target] = self.get_labels(copy=True)

            summary.append({
                "target_contamination": target,
                "direction": direction,
                "start_contamination": start_level,
                "achieved_contamination": achieved,
                "overshoot": (
                    achieved - target
                    if direction == "increase"
                    else target - achieved
                ),
                "start_parent_mismatch_count": start_D,
                "achieved_parent_mismatch_count": int(self.parent_mismatch_count),
                "parent_mismatch_relative": float(self.parent_mismatch_relative),
                "parent_match_fraction": float(self.parent_match_fraction),
                "start_C": start_C,
                "achieved_C": float(self.C),
                "start_C_relative": start_C_relative,
                "achieved_C_relative": float(self.C_relative),
                "accepted_swaps": accepted,
                "total_accepted_swaps": total_accepted,
                "reached": reached_target,
            })

            print(
                f"Target contamination {target:.4f} -> "
                f"achieved={achieved:.6f} | "
                f"parent_mismatch_rel={self.parent_mismatch_relative:.6f} | "
                f"C_relative={self.C_relative:.6f} | "
                f"swaps={accepted:,} | "
                f"{'reached' if reached_target else 'NOT reached'}"
            )

            if not reached_target:
                break

        return labels_by_target, pd.DataFrame(summary)

    # ========================================================
    # validation / recomputation
    # ========================================================

    def recompute_state(
        self,
    ):
        """
        Full graph rescan. Intended for validation, not frequent use.
        """

        score, C = (
            _calculate_initial_state(
                self.indptr,
                self.indices,
                self.data,
                self.labels,
            )
        )

        self.score[:] = score
        self.C = float(
            C
        )

        self._refresh_parent_state()

        return self.C

    def check_state(
        self,
        rtol=1e-9,
        atol=1e-6,
    ):
        """
        Expensive consistency check for incremental graph state and
        LD_PRUNED_BY-parent state.
        """

        old_C = float(
            self.C
        )

        old_parent_mismatch = int(
            self.parent_mismatch_count
        )

        new_score, new_C = (
            _calculate_initial_state(
                self.indptr,
                self.indices,
                self.data,
                self.labels,
            )
        )

        C_ok = np.isclose(
            old_C,
            new_C,
            rtol=rtol,
            atol=atol,
        )

        max_score_error = float(
            np.max(
                np.abs(
                    self.score
                    - new_score
                )
            )
        )

        new_parent_mismatch = (
            self._calculate_parent_mismatch_count()
        )

        fixed_labels_ok = np.array_equal(
            self.labels[
                self._fixed_node_ids
            ],
            self._fixed_labels_reference,
        )

        preferred_labels_ok = np.array_equal(
            self.preferred_label[
                self._movable_node_ids
            ],
            self.labels[
                self.parent_id[
                    self._movable_node_ids
                ]
            ],
        )

        return {
            "C_stored":
                old_C,

            "C_recomputed":
                float(
                    new_C
                ),

            "C_difference":
                float(
                    old_C
                    - new_C
                ),

            "C_ok":
                bool(
                    C_ok
                ),

            "max_score_error":
                max_score_error,

            "parent_mismatch_stored":
                old_parent_mismatch,

            "parent_mismatch_recomputed":
                int(
                    new_parent_mismatch
                ),

            "parent_mismatch_ok":
                bool(
                    old_parent_mismatch
                    == new_parent_mismatch
                ),

            "fixed_labels_ok":
                bool(
                    fixed_labels_ok
                ),

            "preferred_labels_match_parents":
                bool(
                    preferred_labels_ok
                ),
        }


if __name__ == "__main__":
    # ========================================================
    # Usage example
    # ========================================================
    #
    # sim = LDCutSimulator(
    #     pruned_context_df=pruned_context_df,
    #     ld_df=ld_df,
    #     source_col="INDEX_1",
    #     target_col="INDEX_2",
    #     weight_col="R2_scale",
    #     weight_dtype=np.float32,
    #     verbose=True,
    # )
    #
    # Main control variable: contamination_level
    #   0 = initialized parent-cohesive state
    #   1 = random exact-quota parent-mismatch expectation
    #
    # targets = np.linspace(0.0, 1.0, 11)
    #
    # labels_by_target, summary_df = (
    #     sim.move_through_contamination_targets(
    #         targets=targets,
    #         direction="increase",
    #         seed=42,
    #         max_swaps_per_target=1_000_000,
    #         max_attempts_per_swap=100_000,
    #         verbose_every=10_000,
    #     )
    # )
    #
    # C_relative is only measured; it is never targeted.
    # print(summary_df)
    # print(sim.check_state())
    pass


# import gc
# import numpy as np
# import pandas as pd

# from scipy.sparse import csr_matrix
# from numba import njit, prange


# # ============================================================
# # Numba core functions
# # ============================================================


# @njit
# def _fill_undirected_csr(
#     src,
#     dst,
#     weight,
#     cursor,
#     indices,
#     data,
# ):
#     """
#     将每条无向边 u--v 写成两条 CSR adjacency：

#         u -> v
#         v -> u

#     直接写入预分配数组，不构造：
#         concatenate([src, dst])
#         concatenate([dst, src])

#     对几亿条边可以显著减少峰值内存。
#     """

#     m = src.shape[0]

#     for e in range(m):

#         u = int(src[e])
#         v = int(dst[e])
#         w = weight[e]

#         # u -> v
#         p = cursor[u]
#         indices[p] = v
#         data[p] = w
#         cursor[u] += 1

#         # v -> u
#         p = cursor[v]
#         indices[p] = u
#         data[p] = w
#         cursor[v] += 1


# @njit(parallel=True)
# def _calculate_initial_state(
#     indptr,
#     indices,
#     data,
#     labels,
# ):
#     """
#     一次性计算：

#         score[i]
#         C

#     score[i] =
#         same-label incident weight
#         -
#         different-label incident weight

#     CSR 是对称的，因此 FG-BG edge 会被看到两次，
#     最后 C 除以 2。
#     """

#     n = labels.shape[0]

#     score = np.empty(n, dtype=np.float64)

#     cut_twice = 0.0

#     for i in prange(n):

#         label_i = labels[i]

#         local_score = 0.0
#         local_cut = 0.0

#         start = indptr[i]
#         end = indptr[i + 1]

#         for p in range(start, end):

#             j = indices[p]
#             w = data[p]

#             if label_i == labels[j]:

#                 local_score += w

#             else:

#                 local_score -= w
#                 local_cut += w

#         score[i] = local_score

#         cut_twice += local_cut

#     C = 0.5 * cut_twice

#     return score, C


# @njit
# def _edge_weight_sorted(
#     indptr,
#     indices,
#     data,
#     a,
#     b,
# ):
#     """
#     查询 edge a--b 的 weight。

#     CSR 每一行已经排序，因此使用 binary search。

#     不存在则返回 0。
#     """

#     lo = indptr[a]
#     hi = indptr[a + 1]

#     while lo < hi:

#         mid = (lo + hi) // 2

#         x = indices[mid]

#         if x < b:

#             lo = mid + 1

#         else:

#             hi = mid

#     end = indptr[a + 1]

#     if lo < end and indices[lo] == b:

#         return float(data[lo])

#     return 0.0


# @njit
# def _batch_delta_C(
#     indptr,
#     indices,
#     data,
#     score,
#     a_nodes,
#     b_nodes,
# ):
#     """
#     同时计算一批 FG-BG swap proposal 的 ΔC。

#     对异标签节点：

#         ΔC = score[a] + score[b] + 2*w_ab
#     """

#     n = a_nodes.shape[0]

#     delta = np.empty(n, dtype=np.float64)

#     for i in range(n):

#         a = a_nodes[i]
#         b = b_nodes[i]

#         wab = _edge_weight_sorted(
#             indptr,
#             indices,
#             data,
#             a,
#             b,
#         )

#         delta[i] = (
#             score[a]
#             + score[b]
#             + 2.0 * wab
#         )

#     return delta


# @njit
# def _apply_swap_inplace(
#     indptr,
#     indices,
#     data,
#     labels,
#     score,
#     a,
#     b,
# ):
#     """
#     真正执行 FG/BG swap。

#     只更新：

#         a
#         b
#         neighbors(a)
#         neighbors(b)

#     不扫描整个 graph。

#     返回：
#         ΔC
#     """

#     label_a = labels[a]
#     label_b = labels[b]

#     if label_a == label_b:

#         return 0.0

#     # --------------------------------------------------------
#     # a-b 是否直接相连
#     # --------------------------------------------------------

#     wab = _edge_weight_sorted(
#         indptr,
#         indices,
#         data,
#         a,
#         b,
#     )

#     old_score_a = score[a]
#     old_score_b = score[b]

#     # --------------------------------------------------------
#     # 预测 C 的变化
#     # --------------------------------------------------------

#     delta_C = (
#         old_score_a
#         + old_score_b
#         + 2.0 * wab
#     )

#     # --------------------------------------------------------
#     # 更新 a 的所有邻居
#     # --------------------------------------------------------

#     start = indptr[a]
#     end = indptr[a + 1]

#     for p in range(start, end):

#         k = indices[p]

#         # a-b edge 在两者同时 flip 后
#         # FG-BG 状态并不会发生变化
#         if k == b:
#             continue

#         w = data[p]

#         # swap 前 a-k 是 same
#         #
#         # +w -> -w
#         #
#         # 所以 k 的 score -= 2w
#         if labels[k] == label_a:

#             score[k] -= 2.0 * w

#         # swap 前 a-k 是 different
#         #
#         # -w -> +w
#         #
#         # score += 2w
#         else:

#             score[k] += 2.0 * w

#     # --------------------------------------------------------
#     # 更新 b 的所有邻居
#     # --------------------------------------------------------

#     start = indptr[b]
#     end = indptr[b + 1]

#     for p in range(start, end):

#         k = indices[p]

#         if k == a:
#             continue

#         w = data[p]

#         if labels[k] == label_b:

#             score[k] -= 2.0 * w

#         else:

#             score[k] += 2.0 * w

#     # --------------------------------------------------------
#     # 更新 a / b 自己的 score
#     #
#     # 除 a-b edge 外，其余 incident edges 的 same/different
#     # 全部反转。
#     #
#     # a-b 原贡献 = -wab
#     # swap 后仍然 = -wab
#     #
#     # 所以：
#     #
#     # new_score = -old_score - 2*wab
#     # --------------------------------------------------------

#     score[a] = (
#         -old_score_a
#         - 2.0 * wab
#     )

#     score[b] = (
#         -old_score_b
#         - 2.0 * wab
#     )

#     # --------------------------------------------------------
#     # 最后再交换 labels
#     # --------------------------------------------------------

#     labels[a] = label_b
#     labels[b] = label_a

#     return delta_C


# @njit(parallel=True)
# def _calculate_expected_random_C_conditional(
#     indptr,
#     indices,
#     data,
#     labels,
#     movable_mask,
#     movable_fg_count,
# ):
#     """
#     在 fixed nodes 标签保持不变、movable nodes 中 FG 数量固定的条件下，
#     精确计算随机重排 movable labels 时的 E[C]。

#     edge 分为三类：

#     1) fixed -- fixed
#        cut 状态完全固定。

#     2) fixed -- movable
#        movable endpoint 在固定 FG quota 下随机取标签。

#     3) movable -- movable
#        在不放回、固定 FG 数量的随机排列下：

#            P(labels differ)
#            = 2*m*(M-m) / [M*(M-1)]

#        其中：
#            M = movable node 数
#            m = movable FG 数

#     CSR 是对称的，因此这里只统计 j > i 的 edge，一条无向边只算一次。
#     """

#     n = labels.shape[0]

#     movable_count = 0
#     for i in range(n):
#         if movable_mask[i]:
#             movable_count += 1

#     if movable_count == 0:
#         p_movable_fg = 0.0
#         p_diff_movable = 0.0
#     else:
#         p_movable_fg = movable_fg_count / movable_count

#         if movable_count >= 2:
#             p_diff_movable = (
#                 2.0
#                 * movable_fg_count
#                 * (movable_count - movable_fg_count)
#                 / (
#                     movable_count
#                     * (movable_count - 1)
#                 )
#             )
#         else:
#             p_diff_movable = 0.0

#     # 使用 per-node partial sums，避免 Numba parallel reduction
#     # 在某些版本下对 expected += local 的 reduction 编译失败。
#     partial = np.zeros(n, dtype=np.float64)

#     for i in prange(n):

#         local = 0.0
#         movable_i = movable_mask[i]

#         start = indptr[i]
#         end = indptr[i + 1]

#         for p in range(start, end):

#             j = indices[p]

#             # 每条无向 edge 只统计一次
#             if j <= i:
#                 continue

#             w = data[p]
#             movable_j = movable_mask[j]

#             # fixed -- fixed
#             if (not movable_i) and (not movable_j):

#                 if labels[i] != labels[j]:
#                     local += w

#             # movable -- movable
#             elif movable_i and movable_j:

#                 local += w * p_diff_movable

#             # fixed -- movable
#             else:

#                 if movable_i:
#                     fixed_label = labels[j]
#                 else:
#                     fixed_label = labels[i]

#                 if fixed_label == 1:
#                     # fixed=FG 时，different 要求 movable=BG
#                     p_diff = 1.0 - p_movable_fg
#                 else:
#                     # fixed=BG 时，different 要求 movable=FG
#                     p_diff = p_movable_fg

#                 local += w * p_diff

#         partial[i] = local

#     expected = 0.0
#     for i in range(n):
#         expected += partial[i]

#     return expected


# # ============================================================
# # Main simulator
# # ============================================================


# class LDCutSimulator:
#     """
#     LD graph cut simulator with LD_PRUNED_BY-guided child swapping.

#     Core design
#     -----------
#     1. pruned_context_df defines the latent LD units:
#          LD_PRUNING_KEEP == True   -> fixed anchor
#          LD_PRUNING_KEEP == False  -> movable child

#     2. Every movable child has exactly one latent parent:
#          parent_id = LD_PRUNED_BY

#        The parent's fixed foreground tag defines the child's immutable
#        preferred_label.

#     3. The full LD graph still defines the global cross-class cut C:

#          C = sum_{(i,j) in E} w_ij * I(label_i != label_j)

#        Therefore:
#          parent defines direction / latent truth
#          full graph defines the realized LD contamination level

#     4. A parent-guided decrease swap is only:

#          current FG child whose parent is BG
#              <->
#          current BG child whose parent is FG

#        After the swap, BOTH children match their own parents.

#        A parent-guided increase swap is the exact reverse:
#        both children currently match their own parents and the swap makes
#        both children mismatch.

#     5. Every accepted swap preserves:
#          - all fixed-anchor labels
#          - total FG/BG counts
#          - movable FG/BG counts
#          - the LD graph itself
#     """

#     def __init__(
#         self,
#         pruned_context_df,
#         ld_df,
#         source_col="INDEX_1",
#         target_col="INDEX_2",
#         weight_col="R2",
#         weight_dtype=np.float32,
#         verbose=True,
#     ):
#         """
#         Parameters
#         ----------
#         pruned_context_df : pandas.DataFrame
#             Must contain:
#                 LD_PRUNING_KEEP
#                 LD_PRUNED_BY
#                 foreground

#             Node IDs are taken from pruned_context_df.index.

#             The index must contain exactly:
#                 0, 1, ..., N-1

#             Row order itself does NOT matter; the DataFrame is internally
#             reordered by index before arrays are constructed.

#             foreground:
#                 bool or 0/1
#                 True/1 = FG
#                 False/0 = BG

#             LD_PRUNING_KEEP:
#                 bool or 0/1
#                 True  = fixed anchor
#                 False = movable child

#             LD_PRUNED_BY:
#                 For every movable child, must contain the node ID/index of
#                 the retained fixed anchor that pruned it.
#                 Fixed anchors may contain NA.

#         ld_df : pandas.DataFrame
#             One undirected LD edge per row is recommended.

#             Required columns:
#                 source_col
#                 target_col
#                 weight_col

#             Node IDs must use the same 0..N-1 IDs as
#             pruned_context_df.index.

#         weight_col : str
#             Edge weight used by global graph cut C.

#             "R2" is supported.
#             If the final overlap metric uses scaled R2, using an already
#             prepared "R2_scale" column is usually preferable.

#         weight_dtype :
#             CSR edge-weight dtype. float32 is recommended for very large
#             graphs. score and C remain float64.
#         """

#         # ====================================================
#         # 1. Validate and normalize pruned_context_df
#         # ====================================================

#         if not isinstance(pruned_context_df, pd.DataFrame):
#             raise TypeError(
#                 "pruned_context_df must be a pandas DataFrame."
#             )

#         required_context_cols = [
#             "LD_PRUNING_KEEP",
#             "LD_PRUNED_BY",
#             "foreground",
#         ]

#         missing = [
#             c
#             for c in required_context_cols
#             if c not in pruned_context_df.columns
#         ]

#         if missing:
#             raise ValueError(
#                 "pruned_context_df is missing required columns: "
#                 + ", ".join(missing)
#             )

#         if not pruned_context_df.index.is_unique:
#             raise ValueError(
#                 "pruned_context_df.index must be unique."
#             )

#         self.n_nodes = len(pruned_context_df)

#         if self.n_nodes == 0:
#             raise ValueError(
#                 "pruned_context_df contains no nodes."
#             )

#         # The graph implementation uses direct integer indexing.
#         # Allow arbitrary row order, but require the index labels themselves
#         # to be exactly 0..N-1.
#         try:
#             context_index_numeric = np.asarray(
#                 pruned_context_df.index,
#                 dtype=np.int64,
#             )
#         except Exception as exc:
#             raise ValueError(
#                 "pruned_context_df.index must contain integer node IDs "
#                 "0..N-1."
#             ) from exc

#         expected_ids = np.arange(
#             self.n_nodes,
#             dtype=np.int64,
#         )

#         if (
#             len(context_index_numeric) != self.n_nodes
#             or not np.array_equal(
#                 np.sort(context_index_numeric),
#                 expected_ids,
#             )
#         ):
#             raise ValueError(
#                 "pruned_context_df.index must contain exactly the node IDs "
#                 "0..N-1 used by ld_df."
#             )

#         # Reorder by node ID so array position == graph node ID.
#         context = pruned_context_df.loc[
#             expected_ids
#         ]

#         # Keep a lightweight reference to the normalized view for diagnostics.
#         # The simulator's state is stored in NumPy arrays; external mutation of
#         # the original DataFrame does not change simulator labels.
#         self.pruned_context_df = context

#         # ---------------------------
#         # foreground -> int8 labels
#         # ---------------------------

#         foreground_raw = context[
#             "foreground"
#         ]

#         if foreground_raw.isna().any():
#             raise ValueError(
#                 "pruned_context_df['foreground'] contains missing values."
#             )

#         fg_values = set(
#             pd.unique(
#                 foreground_raw
#             )
#         )

#         if not fg_values <= {
#             0, 1, False, True
#         }:
#             raise ValueError(
#                 "foreground must contain only True/False or 0/1."
#             )

#         labels = foreground_raw.astype(
#             bool
#         ).to_numpy(
#             dtype=np.int8
#         )

#         # --------------------------------
#         # LD_PRUNING_KEEP -> fixed/movable
#         # --------------------------------

#         keep_raw = context[
#             "LD_PRUNING_KEEP"
#         ]

#         if keep_raw.isna().any():
#             raise ValueError(
#                 "LD_PRUNING_KEEP contains missing values."
#             )

#         keep_values = set(
#             pd.unique(
#                 keep_raw
#             )
#         )

#         if not keep_values <= {
#             0, 1, False, True
#         }:
#             raise ValueError(
#                 "LD_PRUNING_KEEP must contain only True/False or 0/1."
#             )

#         keep_mask = keep_raw.astype(
#             bool
#         ).to_numpy(
#             dtype=bool
#         )

#         movable_mask = ~keep_mask

#         if not keep_mask.any():
#             raise ValueError(
#                 "No fixed anchors found: LD_PRUNING_KEEP has no True rows."
#             )

#         if not movable_mask.any():
#             raise ValueError(
#                 "No movable children found: LD_PRUNING_KEEP has no False rows."
#             )

#         self.labels = labels.copy()
#         self.movable_mask = movable_mask.copy()
#         self.keep_mask = keep_mask.copy()

#         self._fixed_node_ids = np.flatnonzero(
#             self.keep_mask
#         )

#         self._movable_node_ids = np.flatnonzero(
#             self.movable_mask
#         )

#         self._fixed_labels_reference = self.labels[
#             self._fixed_node_ids
#         ].copy()

#         # ====================================================
#         # 2. Parse LD_PRUNED_BY and establish latent parent
#         # ====================================================

#         parent_id = np.full(
#             self.n_nodes,
#             -1,
#             dtype=np.int64,
#         )

#         parent_raw = context.loc[
#             self.movable_mask,
#             "LD_PRUNED_BY",
#         ]

#         if parent_raw.isna().any():
#             bad_nodes = parent_raw.index[
#                 parent_raw.isna()
#             ].to_numpy()

#             preview = bad_nodes[:10].tolist()

#             raise ValueError(
#                 "Every movable child must have LD_PRUNED_BY. "
#                 f"Missing for {len(bad_nodes):,} movable nodes; "
#                 f"examples: {preview}"
#             )

#         try:
#             parent_numeric = pd.to_numeric(
#                 parent_raw,
#                 errors="raise",
#             ).to_numpy(
#                 dtype=np.float64
#             )
#         except Exception as exc:
#             raise ValueError(
#                 "LD_PRUNED_BY values for movable children must be integer "
#                 "node IDs from pruned_context_df.index."
#             ) from exc

#         if not np.all(
#             np.isfinite(
#                 parent_numeric
#             )
#         ):
#             raise ValueError(
#                 "LD_PRUNED_BY contains non-finite values for movable children."
#             )

#         if not np.all(
#             parent_numeric
#             == np.floor(
#                 parent_numeric
#             )
#         ):
#             raise ValueError(
#                 "LD_PRUNED_BY must contain integer node IDs."
#             )

#         parent_numeric = parent_numeric.astype(
#             np.int64
#         )

#         if (
#             np.any(parent_numeric < 0)
#             or np.any(
#                 parent_numeric
#                 >= self.n_nodes
#             )
#         ):
#             raise ValueError(
#                 "LD_PRUNED_BY contains node IDs outside 0..N-1."
#             )

#         child_ids = self._movable_node_ids

#         parent_id[
#             child_ids
#         ] = parent_numeric

#         if np.any(
#             parent_id[
#                 child_ids
#             ]
#             == child_ids
#         ):
#             bad = child_ids[
#                 parent_id[
#                     child_ids
#                 ]
#                 == child_ids
#             ]

#             raise ValueError(
#                 "A movable child cannot be its own LD_PRUNED_BY parent. "
#                 f"Examples: {bad[:10].tolist()}"
#             )

#         parent_is_fixed = self.keep_mask[
#             parent_id[
#                 child_ids
#             ]
#         ]

#         if not np.all(
#             parent_is_fixed
#         ):
#             bad_children = child_ids[
#                 ~parent_is_fixed
#             ]

#             bad_pairs = [
#                 (
#                     int(i),
#                     int(parent_id[i]),
#                 )
#                 for i in bad_children[:10]
#             ]

#             raise ValueError(
#                 "Every LD_PRUNED_BY parent must be a retained/fixed node "
#                 "(LD_PRUNING_KEEP=True). "
#                 f"Invalid child->parent examples: {bad_pairs}"
#             )

#         self.parent_id = parent_id

#         # preferred_label is the immutable latent direction for each child.
#         # -1 is used for fixed anchors because they do not need a preference.
#         self.preferred_label = np.full(
#             self.n_nodes,
#             -1,
#             dtype=np.int8,
#         )

#         self.preferred_label[
#             child_ids
#         ] = self.labels[
#             parent_id[
#                 child_ids
#             ]
#         ]

#         # Save initialization state after parent mapping is established.
#         self.initial_labels = self.labels.copy()

#         # ====================================================
#         # 3. Build fixed undirected LD CSR graph
#         # ====================================================

#         if not isinstance(ld_df, pd.DataFrame):
#             raise TypeError(
#                 "ld_df must be a pandas DataFrame."
#             )

#         for col in (
#             source_col,
#             target_col,
#             weight_col,
#         ):
#             if col not in ld_df.columns:
#                 raise ValueError(
#                     f"ld_df is missing required column: {col}"
#                 )

#         if verbose:
#             print(
#                 f"Nodes: {self.n_nodes:,}"
#             )
#             print(
#                 f"Fixed anchors: {len(self._fixed_node_ids):,}"
#             )
#             print(
#                 f"Movable children: {len(self._movable_node_ids):,}"
#             )
#             print(
#                 f"Input edges: {len(ld_df):,}"
#             )

#         src = ld_df[
#             source_col
#         ].to_numpy(
#             copy=False
#         )

#         dst = ld_df[
#             target_col
#         ].to_numpy(
#             copy=False
#         )

#         weight = ld_df[
#             weight_col
#         ].to_numpy(
#             dtype=weight_dtype,
#             copy=False,
#         )

#         if len(src) == 0:
#             raise ValueError(
#                 "ld_df contains no edges."
#             )

#         min_node = min(
#             int(src.min()),
#             int(dst.min()),
#         )

#         max_node = max(
#             int(src.max()),
#             int(dst.max()),
#         )

#         if min_node < 0:
#             raise ValueError(
#                 "ld_df node IDs must be >= 0."
#             )

#         if max_node >= self.n_nodes:
#             raise ValueError(
#                 "ld_df contains node IDs outside pruned_context_df index."
#             )

#         if not np.all(
#             np.isfinite(
#                 weight
#             )
#         ):
#             raise ValueError(
#                 f"ld_df[{weight_col!r}] contains non-finite weights."
#             )

#         if np.any(
#             weight < 0
#         ):
#             raise ValueError(
#                 f"ld_df[{weight_col!r}] contains negative weights."
#             )

#         if verbose:
#             print(
#                 "Calculating node degrees..."
#             )

#         degree = np.bincount(
#             src,
#             minlength=self.n_nodes,
#         ).astype(
#             np.int64,
#             copy=False,
#         )

#         degree += np.bincount(
#             dst,
#             minlength=self.n_nodes,
#         )

#         indptr = np.empty(
#             self.n_nodes + 1,
#             dtype=np.int64,
#         )

#         indptr[0] = 0

#         np.cumsum(
#             degree,
#             out=indptr[1:],
#         )

#         nnz = int(
#             indptr[-1]
#         )

#         if verbose:
#             print(
#                 f"Undirected CSR entries: {nnz:,}"
#             )

#         del degree

#         index_dtype = (
#             np.int32
#             if self.n_nodes
#             < np.iinfo(
#                 np.int32
#             ).max
#             else np.int64
#         )

#         self.index_dtype = index_dtype

#         indices = np.empty(
#             nnz,
#             dtype=index_dtype,
#         )

#         data = np.empty(
#             nnz,
#             dtype=weight_dtype,
#         )

#         cursor = indptr[:-1].copy()

#         if verbose:
#             print(
#                 "Building undirected CSR..."
#             )

#         _fill_undirected_csr(
#             src,
#             dst,
#             weight,
#             cursor,
#             indices,
#             data,
#         )

#         del cursor
#         del src
#         del dst
#         del weight

#         gc.collect()

#         A = csr_matrix(
#             (
#                 data,
#                 indices,
#                 indptr,
#             ),
#             shape=(
#                 self.n_nodes,
#                 self.n_nodes,
#             ),
#             copy=False,
#         )

#         # Duplicate input edges are collapsed exactly as in the previous
#         # implementation.
#         A.sum_duplicates()

#         A.setdiag(0)
#         A.eliminate_zeros()

#         # Binary-search edge lookup requires sorted CSR rows.
#         A.sort_indices()

#         self.A = A
#         self.indptr = A.indptr
#         self.indices = A.indices
#         self.data = A.data

#         # ====================================================
#         # 4. Initial global graph-cut state
#         # ====================================================

#         if verbose:
#             print(
#                 "Calculating initial C and node scores..."
#             )

#         self.score, self.C = (
#             _calculate_initial_state(
#                 self.indptr,
#                 self.indices,
#                 self.data,
#                 self.labels,
#             )
#         )

#         self.total_weight = (
#             0.5
#             * self.data.sum(
#                 dtype=np.float64
#             )
#         )

#         # Movable FG/BG candidate pools.
#         self._rebuild_movable_pools()

#         # Conditional random baseline for global C.
#         self._refresh_expected_random_C()

#         # Parent-guidance diagnostics.
#         self._refresh_parent_state()

#         if verbose:
#             print(
#                 f"Total FG nodes: "
#                 f"{int(np.sum(self.labels == 1)):,}"
#             )
#             print(
#                 f"Total BG nodes: "
#                 f"{int(np.sum(self.labels == 0)):,}"
#             )
#             print(
#                 f"Movable FG nodes: {len(self.fg_nodes):,}"
#             )
#             print(
#                 f"Movable BG nodes: {len(self.bg_nodes):,}"
#             )
#             print(
#                 f"C: {self.C:,.6f}"
#             )
#             print(
#                 f"C / total weight: "
#                 f"{self.C_fraction:.6f}"
#             )
#             print(
#                 f"C / E[C_random | fixed nodes]: "
#                 f"{self.C_relative:.6f}"
#             )
#             print(
#                 f"Parent match fraction: "
#                 f"{self.parent_match_fraction:.6f}"
#             )
#             print(
#                 f"Parent mismatch relative to random: "
#                 f"{self.parent_mismatch_relative:.6f}"
#             )

#     # ========================================================
#     # movable-pool / parent-state helpers
#     # ========================================================

#     def _rebuild_movable_pools(
#         self,
#     ):
#         """
#         Rebuild current movable FG/BG pools.

#         Fixed anchors never enter these pools.
#         """

#         self.fg_nodes = np.flatnonzero(
#             (self.labels == 1)
#             & self.movable_mask
#         ).astype(
#             self.index_dtype,
#             copy=False,
#         )

#         self.bg_nodes = np.flatnonzero(
#             (self.labels == 0)
#             & self.movable_mask
#         ).astype(
#             self.index_dtype,
#             copy=False,
#         )

#         self.pool_pos = np.full(
#             self.n_nodes,
#             -1,
#             dtype=np.int64,
#         )

#         self.pool_pos[
#             self.fg_nodes
#         ] = np.arange(
#             len(self.fg_nodes),
#             dtype=np.int64,
#         )

#         self.pool_pos[
#             self.bg_nodes
#         ] = np.arange(
#             len(self.bg_nodes),
#             dtype=np.int64,
#         )

#     def _refresh_expected_random_C(
#         self,
#     ):
#         """
#         Exact E[C] when:
#             - fixed-anchor labels are held fixed;
#             - only movable labels are randomized;
#             - movable FG count is held fixed.
#         """

#         self._expected_random_C = float(
#             _calculate_expected_random_C_conditional(
#                 self.indptr,
#                 self.indices,
#                 self.data,
#                 self.labels,
#                 self.movable_mask,
#                 len(self.fg_nodes),
#             )
#         )

#     def _refresh_preferred_labels_from_parents(
#         self,
#     ):
#         """
#         Recompute child preferred labels from current fixed-parent labels.

#         Normal simulations never change fixed anchors, so this is normally
#         unnecessary after initialization. It is used only when set_labels()
#         is explicitly called with require_fixed_unchanged=False.
#         """

#         child_ids = self._movable_node_ids

#         self.preferred_label[
#             child_ids
#         ] = self.labels[
#             self.parent_id[
#                 child_ids
#             ]
#         ]

#     def _calculate_parent_mismatch_count(
#         self,
#     ):
#         child_ids = self._movable_node_ids

#         return int(
#             np.count_nonzero(
#                 self.labels[
#                     child_ids
#                 ]
#                 != self.preferred_label[
#                     child_ids
#                 ]
#             )
#         )

#     def _refresh_expected_random_parent_mismatch(
#         self,
#     ):
#         """
#         Expected number of parent-tag mismatches under a random exact-quota
#         assignment of movable FG labels.

#         Let:
#             M = number of movable children
#             m = number of movable FG children
#             P = number of children whose preferred(parent) label is FG

#         Every movable node has marginal P(current FG)=m/M under a uniform
#         exact-quota permutation, so:

#             E[mismatch]
#                 = P * (1 - m/M)
#                   + (M-P) * (m/M)
#         """

#         M = int(
#             len(
#                 self._movable_node_ids
#             )
#         )

#         if M == 0:
#             self._expected_random_parent_mismatch = 0.0
#             return

#         m = int(
#             len(
#                 self.fg_nodes
#             )
#         )

#         child_pref = self.preferred_label[
#             self._movable_node_ids
#         ]

#         P = int(
#             np.count_nonzero(
#                 child_pref == 1
#             )
#         )

#         p_fg = (
#             m
#             / M
#         )

#         expected = (
#             P
#             * (1.0 - p_fg)
#             +
#             (M - P)
#             * p_fg
#         )

#         self._expected_random_parent_mismatch = float(
#             expected
#         )

#     def _refresh_parent_state(
#         self,
#     ):
#         self.parent_mismatch_count = (
#             self._calculate_parent_mismatch_count()
#         )

#         self._refresh_expected_random_parent_mismatch()

#     def _parent_guided_mask(
#         self,
#         fg_nodes,
#         bg_nodes,
#         direction,
#     ):
#         """
#         Return legal LD_PRUNED_BY-guided swaps.

#         For a current FG node a and current BG node b:

#         decrease:
#             preferred[a] == BG
#             preferred[b] == FG

#             Both nodes currently mismatch their own parents; after swapping,
#             both match.

#         increase:
#             preferred[a] == FG
#             preferred[b] == BG

#             Both nodes currently match their own parents; after swapping,
#             both mismatch.
#         """

#         if direction not in (
#             "decrease",
#             "increase",
#         ):
#             raise ValueError(
#                 "direction must be 'decrease' or 'increase'"
#             )

#         pref_fg = self.preferred_label[
#             fg_nodes
#         ]

#         pref_bg = self.preferred_label[
#             bg_nodes
#         ]

#         if direction == "decrease":
#             return (
#                 (pref_fg == 0)
#                 & (pref_bg == 1)
#             )

#         return (
#             (pref_fg == 1)
#             & (pref_bg == 0)
#         )

#     # ========================================================
#     # state replacement / reset
#     # ========================================================

#     def set_labels(
#         self,
#         labels,
#         require_fixed_unchanged=True,
#     ):
#         """
#         Replace the complete current 0/1 labeling without rebuilding the graph.

#         By default, fixed anchors must remain exactly as they were at
#         simulator initialization.
#         """

#         labels = np.asarray(
#             labels,
#             dtype=np.int8,
#         )

#         if (
#             labels.ndim != 1
#             or len(labels) != self.n_nodes
#         ):
#             raise ValueError(
#                 "labels must be a 1D array with length n_nodes."
#             )

#         unique_labels = np.unique(
#             labels
#         )

#         if not np.all(
#             np.isin(
#                 unique_labels,
#                 [0, 1],
#             )
#         ):
#             raise ValueError(
#                 "labels must contain only 0 and 1."
#             )

#         if require_fixed_unchanged:
#             if not np.array_equal(
#                 labels[
#                     self._fixed_node_ids
#                 ],
#                 self._fixed_labels_reference,
#             ):
#                 raise ValueError(
#                     "Labels of fixed anchors cannot be changed."
#                 )

#         fixed_changed = (
#             not np.array_equal(
#                 labels[
#                     self._fixed_node_ids
#                 ],
#                 self.labels[
#                     self._fixed_node_ids
#                 ],
#             )
#         )

#         self.labels[:] = labels

#         self.score, self.C = (
#             _calculate_initial_state(
#                 self.indptr,
#                 self.indices,
#                 self.data,
#                 self.labels,
#             )
#         )

#         self._rebuild_movable_pools()
#         self._refresh_expected_random_C()

#         if fixed_changed:
#             self._refresh_preferred_labels_from_parents()

#         self._refresh_parent_state()

#         return self.C

#     def reset_labels(
#         self,
#     ):
#         """
#         Restore simulator-construction labels.
#         """

#         return self.set_labels(
#             self.initial_labels,
#             require_fixed_unchanged=True,
#         )

#     def get_labels(
#         self,
#         copy=True,
#     ):
#         if copy:
#             return self.labels.copy()

#         return self.labels

#     def get_context_df(
#         self,
#         copy=True,
#         foreground_as_bool=True,
#     ):
#         """
#         Return pruned_context_df with foreground replaced by current labels.

#         This is only a convenience method; internal simulator state is kept in
#         NumPy arrays.
#         """

#         if copy:
#             out = self.pruned_context_df.copy()
#         else:
#             out = self.pruned_context_df

#         if foreground_as_bool:
#             out[
#                 "foreground"
#             ] = self.labels.astype(
#                 bool
#             )
#         else:
#             out[
#                 "foreground"
#             ] = self.labels

#         return out

#     # ========================================================
#     # metrics
#     # ========================================================

#     @property
#     def C_fraction(
#         self,
#     ):
#         if self.total_weight == 0:
#             return np.nan

#         return (
#             self.C
#             / self.total_weight
#         )

#     @property
#     def expected_random_C(
#         self,
#     ):
#         return self._expected_random_C

#     @property
#     def C_relative(
#         self,
#     ):
#         expected = self.expected_random_C

#         if expected == 0:
#             return np.nan

#         return (
#             self.C
#             / expected
#         )

#     @property
#     def parent_match_count(
#         self,
#     ):
#         return (
#             len(
#                 self._movable_node_ids
#             )
#             - self.parent_mismatch_count
#         )

#     @property
#     def parent_match_fraction(
#         self,
#     ):
#         M = len(
#             self._movable_node_ids
#         )

#         if M == 0:
#             return np.nan

#         return (
#             self.parent_match_count
#             / M
#         )

#     @property
#     def parent_mismatch_fraction(
#         self,
#     ):
#         M = len(
#             self._movable_node_ids
#         )

#         if M == 0:
#             return np.nan

#         return (
#             self.parent_mismatch_count
#             / M
#         )

#     @property
#     def expected_random_parent_mismatch(
#         self,
#     ):
#         return (
#             self._expected_random_parent_mismatch
#         )

#     @property
#     def parent_mismatch_relative(
#         self,
#     ):
#         expected = (
#             self.expected_random_parent_mismatch
#         )

#         if expected == 0:
#             return np.nan

#         return (
#             self.parent_mismatch_count
#             / expected
#         )

#     # ========================================================
#     # graph queries
#     # ========================================================

#     def neighbors(
#         self,
#         node,
#     ):
#         start = self.indptr[
#             node
#         ]

#         end = self.indptr[
#             node + 1
#         ]

#         return (
#             self.indices[
#                 start:end
#             ],
#             self.data[
#                 start:end
#             ],
#         )

#     def edge_weight(
#         self,
#         a,
#         b,
#     ):
#         return (
#             _edge_weight_sorted(
#                 self.indptr,
#                 self.indices,
#                 self.data,
#                 int(a),
#                 int(b),
#             )
#         )

#     # ========================================================
#     # proposal calculations
#     # ========================================================

#     def delta_C(
#         self,
#         a,
#         b,
#     ):
#         """
#         Predict global graph-cut change:
#             ΔC = C_after - C_before
#         """

#         a = int(a)
#         b = int(b)

#         if not (
#             0 <= a < self.n_nodes
#             and 0 <= b < self.n_nodes
#         ):
#             raise IndexError(
#                 "Node ID out of range."
#             )

#         if (
#             not self.movable_mask[a]
#             or not self.movable_mask[b]
#         ):
#             raise ValueError(
#                 "delta_C is only defined for movable swap candidates."
#             )

#         if (
#             self.labels[a]
#             == self.labels[b]
#         ):
#             return 0.0

#         wab = self.edge_weight(
#             a,
#             b,
#         )

#         return float(
#             self.score[a]
#             + self.score[b]
#             + 2.0 * wab
#         )

#     def delta_parent_mismatch(
#         self,
#         a,
#         b,
#     ):
#         """
#         Predict change in the number of child-parent tag mismatches.

#         Returns:
#             mismatch_after - mismatch_before

#         For parent-guided swaps this is exactly:
#             -2 for decrease
#             +2 for increase
#         """

#         a = int(a)
#         b = int(b)

#         if (
#             not self.movable_mask[a]
#             or not self.movable_mask[b]
#         ):
#             raise ValueError(
#                 "delta_parent_mismatch is only defined for movable nodes."
#             )

#         label_a = int(
#             self.labels[a]
#         )
#         label_b = int(
#             self.labels[b]
#         )

#         if label_a == label_b:
#             return 0

#         before = (
#             int(
#                 label_a
#                 != self.preferred_label[a]
#             )
#             +
#             int(
#                 label_b
#                 != self.preferred_label[b]
#             )
#         )

#         after = (
#             int(
#                 label_b
#                 != self.preferred_label[a]
#             )
#             +
#             int(
#                 label_a
#                 != self.preferred_label[b]
#             )
#         )

#         return int(
#             after - before
#         )

#     def sample_candidates(
#         self,
#         n_candidates=10_000,
#         rng=None,
#         direction=None,
#         parent_guided=False,
#         # Backward-compatible alias from the previous implementation.
#         anchor_guided=None,
#         strict_anchor=None,
#         anchor_eps=None,
#     ):
#         """
#         Randomly propose movable FG <-> movable BG swaps.

#         parent_guided=False:
#             all movable FG/BG pairs can be proposed.

#         parent_guided=True:
#             filter proposals using LD_PRUNED_BY parent tags.

#             direction="decrease":
#                 FG child with BG parent
#                     <->
#                 BG child with FG parent

#             direction="increase":
#                 FG child with FG parent
#                     <->
#                 BG child with BG parent

#         Notes on backward compatibility
#         -------------------------------
#         `anchor_guided` is accepted as an alias for `parent_guided` so older
#         calling code does not immediately break. `strict_anchor` and
#         `anchor_eps` are accepted but ignored because LD_PRUNED_BY guidance is
#         categorical and inherently strict.
#         """

#         if anchor_guided is not None:
#             parent_guided = bool(
#                 anchor_guided
#             )

#         if rng is None:
#             rng = np.random.default_rng()

#         if (
#             len(self.fg_nodes) == 0
#             or len(self.bg_nodes) == 0
#         ):
#             raise ValueError(
#                 "Movable nodes must contain at least one FG and one BG."
#             )

#         if parent_guided:
#             if direction not in (
#                 "decrease",
#                 "increase",
#             ):
#                 raise ValueError(
#                     "direction must be provided when parent_guided=True."
#                 )

#         fg_idx = rng.integers(
#             0,
#             len(self.fg_nodes),
#             size=n_candidates,
#         )

#         bg_idx = rng.integers(
#             0,
#             len(self.bg_nodes),
#             size=n_candidates,
#         )

#         fg_candidates = self.fg_nodes[
#             fg_idx
#         ]

#         bg_candidates = self.bg_nodes[
#             bg_idx
#         ]

#         if parent_guided:
#             guided = self._parent_guided_mask(
#                 fg_candidates,
#                 bg_candidates,
#                 direction=direction,
#             )

#             fg_candidates = fg_candidates[
#                 guided
#             ]

#             bg_candidates = bg_candidates[
#                 guided
#             ]

#         if len(
#             fg_candidates
#         ) == 0:
#             return (
#                 fg_candidates,
#                 bg_candidates,
#                 np.empty(
#                     0,
#                     dtype=np.float64,
#                 ),
#             )

#         delta = _batch_delta_C(
#             self.indptr,
#             self.indices,
#             self.data,
#             self.score,
#             fg_candidates,
#             bg_candidates,
#         )

#         return (
#             fg_candidates,
#             bg_candidates,
#             delta,
#         )

#     def propose_best_swap(
#         self,
#         direction="decrease",
#         n_candidates=10_000,
#         rng=None,
#         parent_guided=True,
#         anchor_guided=None,
#         strict_anchor=None,
#         anchor_eps=None,
#     ):
#         """
#         From a random candidate batch choose the most extreme ΔC in the
#         requested direction.

#         By default candidate legality is determined by LD_PRUNED_BY parent
#         tags.
#         """

#         (
#             fg,
#             bg,
#             delta,
#         ) = self.sample_candidates(
#             n_candidates=n_candidates,
#             rng=rng,
#             direction=direction,
#             parent_guided=parent_guided,
#             anchor_guided=anchor_guided,
#             strict_anchor=strict_anchor,
#             anchor_eps=anchor_eps,
#         )

#         if len(
#             delta
#         ) == 0:
#             return (
#                 None,
#                 None,
#                 np.nan,
#             )

#         if direction == "decrease":
#             i = int(
#                 np.argmin(
#                     delta
#                 )
#             )

#         elif direction == "increase":
#             i = int(
#                 np.argmax(
#                     delta
#                 )
#             )

#         else:
#             raise ValueError(
#                 "direction must be 'decrease' or 'increase'"
#             )

#         return (
#             int(
#                 fg[i]
#             ),
#             int(
#                 bg[i]
#             ),
#             float(
#                 delta[i]
#             ),
#         )

#     def propose_random_useful_swap(
#         self,
#         direction="increase",
#         n_candidates=10_000,
#         rng=None,
#         parent_guided=True,
#         anchor_guided=None,
#         strict_anchor=None,
#         anchor_eps=None,
#     ):
#         """
#         Randomly choose ONE useful swap from a random candidate batch.

#         This is the non-greedy proposal used for the simulation trajectory.

#         With parent_guided=True:

#             increase:
#                 current FG child with FG parent
#                     <->
#                 current BG child with BG parent

#                 Both children are parent-matched before the swap and become
#                 parent-mismatched after the swap. Only candidates with dC > 0
#                 are eligible.

#             decrease:
#                 current FG child with BG parent
#                     <->
#                 current BG child with FG parent

#                 Both children are parent-mismatched before the swap and become
#                 parent-matched after the swap. Only candidates with dC < 0 are
#                 eligible.

#         Importantly, among useful candidates the chosen pair is RANDOM. The
#         magnitude of dC is not optimized, so graph topology does not determine
#         which useful pair is preferentially selected.
#         """

#         if anchor_guided is not None:
#             parent_guided = bool(anchor_guided)

#         if rng is None:
#             rng = np.random.default_rng()

#         if direction not in ("decrease", "increase"):
#             raise ValueError(
#                 "direction must be 'decrease' or 'increase'"
#             )

#         fg, bg, delta = self.sample_candidates(
#             n_candidates=n_candidates,
#             rng=rng,
#             direction=direction,
#             parent_guided=parent_guided,
#             strict_anchor=strict_anchor,
#             anchor_eps=anchor_eps,
#         )

#         if len(delta) == 0:
#             return None, None, np.nan

#         if direction == "increase":
#             useful = delta > 0
#         else:
#             useful = delta < 0

#         useful_idx = np.flatnonzero(useful)

#         if len(useful_idx) == 0:
#             return None, None, np.nan

#         i = int(rng.choice(useful_idx))

#         return (
#             int(fg[i]),
#             int(bg[i]),
#             float(delta[i]),
#         )

#     # ========================================================
#     # execute swap
#     # ========================================================

#     def swap(
#         self,
#         a,
#         b,
#     ):
#         """
#         Execute one movable FG <-> movable BG swap.

#         Global C and parent-mismatch count are both updated incrementally.
#         """

#         a = int(a)
#         b = int(b)

#         if not (
#             0 <= a < self.n_nodes
#             and 0 <= b < self.n_nodes
#         ):
#             raise IndexError(
#                 "Node ID out of range."
#             )

#         if (
#             not self.movable_mask[a]
#             or not self.movable_mask[b]
#         ):
#             raise ValueError(
#                 "Only movable children may be swapped."
#             )

#         label_a = int(
#             self.labels[a]
#         )

#         label_b = int(
#             self.labels[b]
#         )

#         if label_a == label_b:
#             return 0.0

#         delta_parent = (
#             self.delta_parent_mismatch(
#                 a,
#                 b,
#             )
#         )

#         pos_a = int(
#             self.pool_pos[a]
#         )

#         pos_b = int(
#             self.pool_pos[b]
#         )

#         delta = (
#             _apply_swap_inplace(
#                 self.indptr,
#                 self.indices,
#                 self.data,
#                 self.labels,
#                 self.score,
#                 a,
#                 b,
#             )
#         )

#         self.C += delta

#         self.parent_mismatch_count += (
#             delta_parent
#         )

#         # O(1) update of movable FG/BG pools.
#         if (
#             label_a == 1
#             and label_b == 0
#         ):
#             self.fg_nodes[
#                 pos_a
#             ] = b

#             self.bg_nodes[
#                 pos_b
#             ] = a

#             self.pool_pos[b] = pos_a
#             self.pool_pos[a] = pos_b

#         else:
#             # a was BG and b was FG.
#             self.bg_nodes[
#                 pos_a
#             ] = b

#             self.fg_nodes[
#                 pos_b
#             ] = a

#             self.pool_pos[b] = pos_a
#             self.pool_pos[a] = pos_b

#         return float(
#             delta
#         )

#     # ========================================================
#     # move C monotonically
#     # ========================================================

#     def move_C(
#         self,
#         direction,
#         n_swaps,
#         n_candidates=10_000,
#         seed=None,
#         max_failed_rounds=100,
#         record_every=1,
#         verbose_every=100,
#         parent_guided=True,
#         # backward-compatible old keyword:
#         anchor_guided=None,
#         strict_anchor=None,
#         anchor_eps=None,
#     ):
#         """
#         Move C monotonically using RANDOM useful parent-guided swaps.

#         Unlike the old greedy implementation, this method does NOT choose the
#         candidate with the largest |dC|.

#         increase:
#             parent-matched FG child <-> parent-matched BG child,
#             require dC > 0, then choose randomly among useful candidates.

#         decrease:
#             parent-mismatched FG child <-> parent-mismatched BG child,
#             require dC < 0, then choose randomly among useful candidates.

#         Every swap preserves fixed-anchor labels and the movable FG quota.
#         """

#         if anchor_guided is not None:
#             parent_guided = bool(anchor_guided)

#         if direction not in ("decrease", "increase"):
#             raise ValueError(
#                 "direction must be 'decrease' or 'increase'"
#             )

#         rng = np.random.default_rng(seed)

#         history = []
#         accepted = 0
#         failed_rounds = 0

#         while accepted < n_swaps:

#             a, b, delta = self.propose_random_useful_swap(
#                 direction=direction,
#                 n_candidates=n_candidates,
#                 rng=rng,
#                 parent_guided=parent_guided,
#                 strict_anchor=strict_anchor,
#                 anchor_eps=anchor_eps,
#             )

#             if a is None:
#                 failed_rounds += 1

#                 if failed_rounds >= max_failed_rounds:
#                     break

#                 continue

#             failed_rounds = 0

#             actual_delta = self.swap(a, b)
#             accepted += 1

#             if record_every and accepted % record_every == 0:
#                 history.append(
#                     (
#                         accepted,
#                         a,
#                         b,
#                         actual_delta,
#                         self.C,
#                         self.C_fraction,
#                         self.C_relative,
#                         self.parent_match_fraction,
#                         self.parent_mismatch_relative,
#                     )
#                 )

#             if verbose_every and accepted % verbose_every == 0:
#                 print(
#                     f"{accepted:,} swaps | "
#                     f"C={self.C:,.6f} | "
#                     f"C_rel={self.C_relative:.6f} | "
#                     f"parent_match={self.parent_match_fraction:.6f} | "
#                     f"parent_mismatch_rel={self.parent_mismatch_relative:.6f} | "
#                     f"dC={actual_delta:,.6f}"
#                 )

#         return pd.DataFrame(
#             history,
#             columns=[
#                 "n_swaps",
#                 "FG_node",
#                 "BG_node",
#                 "delta_C",
#                 "C",
#                 "C_fraction",
#                 "C_relative",
#                 "parent_match_fraction",
#                 "parent_mismatch_relative",
#             ],
#         )

#     # ========================================================
#     # move toward target C
#     # ========================================================

#     def move_to_C(
#         self,
#         target_C,
#         n_candidates=10_000,
#         max_swaps=100_000,
#         tolerance=0.0,
#         seed=None,
#         max_failed_rounds=100,
#         record_every=1,
#         verbose_every=100,
#         parent_guided=True,
#         # backward-compatible old keyword:
#         anchor_guided=None,
#         strict_anchor=None,
#         anchor_eps=None,
#     ):
#         """
#         Move toward target_C without greedily optimizing dC.

#         At each iteration, candidates are filtered to those that move C closer
#         to the target. One of those useful candidates is then chosen RANDOMLY.

#         For the main contamination simulation, prefer
#         move_through_C_relative_targets(), which saves the first state that
#         crosses each requested C_relative threshold.
#         """

#         if anchor_guided is not None:
#             parent_guided = bool(anchor_guided)

#         target_C = float(target_C)
#         rng = np.random.default_rng(seed)

#         history = []
#         accepted = 0
#         failed_rounds = 0

#         while accepted < max_swaps:

#             current_distance = abs(self.C - target_C)

#             if current_distance <= tolerance:
#                 break

#             if target_C < self.C:
#                 direction = "decrease"
#             else:
#                 direction = "increase"

#             fg, bg, delta = self.sample_candidates(
#                 n_candidates=n_candidates,
#                 rng=rng,
#                 direction=direction,
#                 parent_guided=parent_guided,
#             )

#             if len(delta) == 0:
#                 failed_rounds += 1
#                 if failed_rounds >= max_failed_rounds:
#                     break
#                 continue

#             new_C = self.C + delta
#             new_distance = np.abs(new_C - target_C)

#             # Must actually move closer to the requested target.
#             useful = new_distance < current_distance
#             useful_idx = np.flatnonzero(useful)

#             if len(useful_idx) == 0:
#                 failed_rounds += 1
#                 if failed_rounds >= max_failed_rounds:
#                     break
#                 continue

#             failed_rounds = 0

#             # Critical change: random useful candidate, NOT closest-to-target.
#             i = int(rng.choice(useful_idx))

#             a = int(fg[i])
#             b = int(bg[i])
#             actual_delta = self.swap(a, b)
#             accepted += 1

#             if record_every and accepted % record_every == 0:
#                 history.append(
#                     (
#                         accepted,
#                         a,
#                         b,
#                         actual_delta,
#                         self.C,
#                         self.C_fraction,
#                         self.C_relative,
#                         self.parent_match_fraction,
#                         self.parent_mismatch_relative,
#                     )
#                 )

#             if verbose_every and accepted % verbose_every == 0:
#                 print(
#                     f"{accepted:,} swaps | "
#                     f"C={self.C:,.6f} | "
#                     f"target={target_C:,.6f} | "
#                     f"C_rel={self.C_relative:.6f} | "
#                     f"parent_match={self.parent_match_fraction:.6f} | "
#                     f"parent_mismatch_rel={self.parent_mismatch_relative:.6f} | "
#                     f"distance={abs(self.C-target_C):,.6f}"
#                 )

#         return pd.DataFrame(
#             history,
#             columns=[
#                 "n_swaps",
#                 "FG_node",
#                 "BG_node",
#                 "delta_C",
#                 "C",
#                 "C_fraction",
#                 "C_relative",
#                 "parent_match_fraction",
#                 "parent_mismatch_relative",
#             ],
#         )

#     def move_to_C_relative(
#         self,
#         target_relative,
#         **kwargs,
#     ):
#         target_C = (
#             float(
#                 target_relative
#             )
#             * self.expected_random_C
#         )

#         return self.move_to_C(
#             target_C=target_C,
#             **kwargs,
#         )

#     # ========================================================
#     # validation / recomputation
#     # ========================================================

#     def recompute_state(
#         self,
#     ):
#         """
#         Full graph rescan. Intended for validation, not frequent use.
#         """

#         score, C = (
#             _calculate_initial_state(
#                 self.indptr,
#                 self.indices,
#                 self.data,
#                 self.labels,
#             )
#         )

#         self.score[:] = score
#         self.C = float(
#             C
#         )

#         self._refresh_parent_state()

#         return self.C

#     def check_state(
#         self,
#         rtol=1e-9,
#         atol=1e-6,
#     ):
#         """
#         Expensive consistency check for incremental graph state and
#         LD_PRUNED_BY-parent state.
#         """

#         old_C = float(
#             self.C
#         )

#         old_parent_mismatch = int(
#             self.parent_mismatch_count
#         )

#         new_score, new_C = (
#             _calculate_initial_state(
#                 self.indptr,
#                 self.indices,
#                 self.data,
#                 self.labels,
#             )
#         )

#         C_ok = np.isclose(
#             old_C,
#             new_C,
#             rtol=rtol,
#             atol=atol,
#         )

#         max_score_error = float(
#             np.max(
#                 np.abs(
#                     self.score
#                     - new_score
#                 )
#             )
#         )

#         new_parent_mismatch = (
#             self._calculate_parent_mismatch_count()
#         )

#         fixed_labels_ok = np.array_equal(
#             self.labels[
#                 self._fixed_node_ids
#             ],
#             self._fixed_labels_reference,
#         )

#         preferred_labels_ok = np.array_equal(
#             self.preferred_label[
#                 self._movable_node_ids
#             ],
#             self.labels[
#                 self.parent_id[
#                     self._movable_node_ids
#                 ]
#             ],
#         )

#         return {
#             "C_stored":
#                 old_C,

#             "C_recomputed":
#                 float(
#                     new_C
#                 ),

#             "C_difference":
#                 float(
#                     old_C
#                     - new_C
#                 ),

#             "C_ok":
#                 bool(
#                     C_ok
#                 ),

#             "max_score_error":
#                 max_score_error,

#             "parent_mismatch_stored":
#                 old_parent_mismatch,

#             "parent_mismatch_recomputed":
#                 int(
#                     new_parent_mismatch
#                 ),

#             "parent_mismatch_ok":
#                 bool(
#                     old_parent_mismatch
#                     == new_parent_mismatch
#                 ),

#             "fixed_labels_ok":
#                 bool(
#                     fixed_labels_ok
#                 ),

#             "preferred_labels_match_parents":
#                 bool(
#                     preferred_labels_ok
#                 ),
#         }

#     # ========================================================
#     # continuous trajectory through multiple C_relative targets
#     # ========================================================

#     def move_through_C_relative_targets(
#         self,
#         targets,
#         n_candidates=10_000,
#         max_swaps_per_target=100_000,
#         tolerance_relative=1e-4,
#         seed=None,
#         max_failed_rounds=100,
#         verbose_every=1_000,
#         parent_guided=True,
#         direction=None,
#         # backward-compatible old keywords:
#         anchor_guided=None,
#         strict_anchor=None,
#         anchor_eps=None,
#     ):
#         """
#         Traverse C_relative monotonically using RANDOM useful swaps and save
#         the first state that reaches/crosses each requested threshold.

#         This is intended for the contamination simulation:

#             1. Start from a low-cross-class-LD labeling where children tend to
#                inherit their LD_PRUNED_BY parent tag.
#             2. Use direction="increase".
#             3. Each accepted swap must be parent-guided and have dC > 0.
#             4. Among useful swaps, choose randomly; never maximize dC.
#             5. Save the first state that reaches/crosses each target.

#         Example
#         -------
#             targets = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#             labels_by_target, summary_df = (
#                 sim.move_through_C_relative_targets(
#                     targets,
#                     direction="increase",
#                     seed=42,
#                 )
#             )

#         Notes
#         -----
#         - For direction="increase", targets must be nondecreasing.
#         - For direction="decrease", targets must be nonincreasing.
#         - Exact target matching is NOT optimized. Small overshoot is expected
#           because C changes in discrete swaps.
#         - Fixed anchors and the movable FG quota remain unchanged.
#         """

#         if anchor_guided is not None:
#             parent_guided = bool(anchor_guided)

#         targets = np.asarray(targets, dtype=np.float64)

#         if targets.ndim != 1 or len(targets) == 0:
#             raise ValueError(
#                 "targets must be a non-empty 1D sequence."
#             )

#         if not np.all(np.isfinite(targets)):
#             raise ValueError(
#                 "targets must contain only finite values."
#             )

#         if tolerance_relative < 0:
#             raise ValueError(
#                 "tolerance_relative must be >= 0."
#             )

#         if n_candidates <= 0:
#             raise ValueError(
#                 "n_candidates must be > 0."
#             )

#         if max_swaps_per_target <= 0:
#             raise ValueError(
#                 "max_swaps_per_target must be > 0."
#             )

#         start_relative_global = float(self.C_relative)

#         # Infer direction only when it is unambiguous.
#         if direction is None:
#             if np.all(targets >= start_relative_global - tolerance_relative):
#                 direction = "increase"
#             elif np.all(targets <= start_relative_global + tolerance_relative):
#                 direction = "decrease"
#             else:
#                 raise ValueError(
#                     "Could not infer one monotonic direction from the current "
#                     "C_relative and targets. Please set direction explicitly."
#                 )

#         if direction not in ("increase", "decrease"):
#             raise ValueError(
#                 "direction must be 'increase' or 'decrease'"
#             )

#         if direction == "increase":
#             if np.any(np.diff(targets) < 0):
#                 raise ValueError(
#                     "For direction='increase', targets must be nondecreasing."
#                 )
#             if np.any(targets < start_relative_global - tolerance_relative):
#                 raise ValueError(
#                     "For direction='increase', every target must be at or "
#                     "above the starting C_relative."
#                 )
#         else:
#             if np.any(np.diff(targets) > 0):
#                 raise ValueError(
#                     "For direction='decrease', targets must be nonincreasing."
#                 )
#             if np.any(targets > start_relative_global + tolerance_relative):
#                 raise ValueError(
#                     "For direction='decrease', every target must be at or "
#                     "below the starting C_relative."
#                 )

#         rng = np.random.default_rng(seed)

#         labels_by_target = {}
#         summary = []
#         total_accepted = 0

#         for target_relative in targets:

#             target_relative = float(target_relative)

#             start_C = float(self.C)
#             start_relative = float(self.C_relative)
#             start_parent_match = float(self.parent_match_fraction)
#             start_parent_mismatch_relative = float(
#                 self.parent_mismatch_relative
#             )

#             accepted = 0
#             failed_rounds = 0

#             def target_reached():
#                 if direction == "increase":
#                     return (
#                         self.C_relative
#                         >= target_relative - tolerance_relative
#                     )
#                 return (
#                     self.C_relative
#                     <= target_relative + tolerance_relative
#                 )

#             while (
#                 not target_reached()
#                 and accepted < max_swaps_per_target
#             ):

#                 a, b, delta = self.propose_random_useful_swap(
#                     direction=direction,
#                     n_candidates=n_candidates,
#                     rng=rng,
#                     parent_guided=parent_guided,
#                     strict_anchor=strict_anchor,
#                     anchor_eps=anchor_eps,
#                 )

#                 if a is None:
#                     failed_rounds += 1

#                     if failed_rounds >= max_failed_rounds:
#                         break

#                     continue

#                 failed_rounds = 0

#                 self.swap(a, b)
#                 accepted += 1
#                 total_accepted += 1

#                 if verbose_every and accepted % verbose_every == 0:
#                     print(
#                         f"target={target_relative:.6f} | "
#                         f"target swaps={accepted:,} | "
#                         f"total swaps={total_accepted:,} | "
#                         f"C_rel={self.C_relative:.6f} | "
#                         f"parent_match={self.parent_match_fraction:.6f} | "
#                         f"parent_mismatch_rel={self.parent_mismatch_relative:.6f}"
#                     )

#             achieved_relative = float(self.C_relative)
#             achieved_C = float(self.C)
#             achieved_parent_match = float(self.parent_match_fraction)
#             achieved_parent_mismatch_relative = float(
#                 self.parent_mismatch_relative
#             )

#             reached = bool(target_reached())

#             labels_by_target[target_relative] = self.get_labels(copy=True)

#             if direction == "increase":
#                 overshoot_relative = achieved_relative - target_relative
#             else:
#                 overshoot_relative = target_relative - achieved_relative

#             summary.append({
#                 "target_relative": target_relative,
#                 "direction": direction,
#                 "start_C_relative": start_relative,
#                 "achieved_C_relative": achieved_relative,
#                 "overshoot_relative": overshoot_relative,
#                 "start_parent_match_fraction": start_parent_match,
#                 "achieved_parent_match_fraction": achieved_parent_match,
#                 "start_parent_mismatch_relative":
#                     start_parent_mismatch_relative,
#                 "achieved_parent_mismatch_relative":
#                     achieved_parent_mismatch_relative,
#                 "target_C": target_relative * self.expected_random_C,
#                 "start_C": start_C,
#                 "achieved_C": achieved_C,
#                 "distance_relative": abs(
#                     achieved_relative - target_relative
#                 ),
#                 "accepted_swaps": accepted,
#                 "total_accepted_swaps": total_accepted,
#                 "reached": reached,
#                 "failed_rounds": failed_rounds,
#             })

#             print(
#                 f"Target {target_relative:.4f} -> "
#                 f"C_relative={achieved_relative:.6f} | "
#                 f"parent_match={achieved_parent_match:.6f} | "
#                 f"parent_mismatch_rel={achieved_parent_mismatch_relative:.6f} | "
#                 f"swaps={accepted:,} | "
#                 f"{'reached/crossed' if reached else 'NOT reached'}"
#             )

#             if not reached:
#                 # The same constrained trajectory cannot reach any more extreme
#                 # target once useful swaps are exhausted.
#                 break

#         summary_df = pd.DataFrame(summary)

#         return labels_by_target, summary_df


# if __name__ == "__main__":
#     # ========================================================
#     # Usage example
#     # ========================================================
#     #
#     # Required input DataFrame:
#     #
#     #   pruned_context_df
#     #
#     # with columns:
#     #
#     #   LD_PRUNING_KEEP
#     #   LD_PRUNED_BY
#     #   foreground
#     #
#     # and index equal to the node IDs used in ld_df:
#     #
#     #   0 .. N-1
#     #
#     # --------------------------------------------------------
#     # Optional: use R2_scale as the graph-cut weight
#     # --------------------------------------------------------
#     #
#     # min_R2 = 0.5
#     #
#     # ld_df = ld_df.copy()
#     # ld_df["R2_scale"] = (
#     #     (ld_df["R2"] - min_R2)
#     #     / (1.0 - min_R2)
#     # ).clip(0, 1)
#     #
#     # --------------------------------------------------------
#     # Construct simulator
#     # --------------------------------------------------------
#     #
#     # sim = LDCutSimulator(
#     #     pruned_context_df=pruned_context_df,
#     #     ld_df=ld_df,
#     #     source_col="INDEX_1",
#     #     target_col="INDEX_2",
#     #     weight_col="R2_scale",
#     #     weight_dtype=np.float32,
#     #     verbose=True,
#     # )
#     #
#     # print("Initial C_relative:", sim.C_relative)
#     # print("Initial parent match:", sim.parent_match_fraction)
#     # print(
#     #     "Initial parent mismatch relative:",
#     #     sim.parent_mismatch_relative,
#     # )
#     #
#     # --------------------------------------------------------
#     # LOW cross-class-LD branch
#     # --------------------------------------------------------
#     #
#     # sim.reset_labels()
#     #
#     # labels_low, summary_low = (
#     #     sim.move_through_C_relative_targets(
#     #         targets=[0.9, 0.8, 0.7, 0.6],
#     #         n_candidates=10_000,
#     #         max_swaps_per_target=100_000,
#     #         tolerance_relative=1e-3,
#     #         seed=42,
#     #         max_failed_rounds=100,
#     #         verbose_every=1_000,
#     #     )
#     # )
#     #
#     # Expected direction:
#     #
#     #   C_relative decreases
#     #   parent_match_fraction increases
#     #   parent_mismatch_relative decreases
#     #
#     # --------------------------------------------------------
#     # HIGH cross-class-LD branch
#     # --------------------------------------------------------
#     #
#     # sim.reset_labels()
#     #
#     # labels_high, summary_high = (
#     #     sim.move_through_C_relative_targets(
#     #         targets=[1.1, 1.2, 1.3],
#     #         n_candidates=10_000,
#     #         max_swaps_per_target=100_000,
#     #         tolerance_relative=1e-3,
#     #         seed=42,
#     #         max_failed_rounds=100,
#     #         verbose_every=1_000,
#     #     )
#     # )
#     #
#     # Expected direction:
#     #
#     #   C_relative increases
#     #   parent_match_fraction decreases
#     #   parent_mismatch_relative increases
#     #
#     # --------------------------------------------------------
#     # Recover current labels or a current context DataFrame
#     # --------------------------------------------------------
#     #
#     # labels = sim.get_labels()
#     #
#     # context_now = sim.get_context_df(
#     #     foreground_as_bool=True,
#     # )
#     #
#     # --------------------------------------------------------
#     # Sanity check
#     # --------------------------------------------------------
#     #
#     # print(sim.check_state())
#     #
#     pass


def sample_causal_snps(
    df,
    causal_fraction=0.05,
    enrichment_ratio=1.0,
    maf_col="MAF",
    foreground_col="foreground",
    maf_bins=None,
    seed=None,
    causal_col="causal",
    causal_foreground_fraction=None,
):
    """
    Sample causal SNPs from an LD-pruned SNP DataFrame.

    enrichment_ratio is defined as:

        P(causal | foreground) / P(causal | background)

    Causal SNPs are stratified by MAF so that the MAF distribution of
    causal SNPs approximately follows that of the full SNP universe.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per LD-pruned SNP. Must contain foreground and MAF columns.

    causal_fraction : float
        Fraction of all SNPs to label causal, e.g. 0.05.

    enrichment_ratio : float
        Relative causal probability in foreground versus background.
        1.0 = no enrichment, 2.0 = 2-fold enrichment.

    causal_foreground_fraction : float, optional
        Target fraction of causal SNPs in the foreground, in [0, 1]. Set
        enrichment_ratio=None when using this mode. The requested foreground
        and background counts are rounded to whole SNPs, then sampled within
        each group's MAF bins in proportion to available sites.

    maf_bins : array-like, optional
        MAF bin edges. Default:
        [0, .05, .10, .15, ..., .50].

    seed : int, optional
        Random seed.

    Returns
    -------
    sampled_df : pandas.DataFrame
        Copy of df with a boolean causal column.

    summary_df : pandas.DataFrame
        Per-MAF-bin sampling summary.
    """
    if not 0 <= causal_fraction <= 1:
        raise ValueError("causal_fraction must be in [0, 1]")
    if (enrichment_ratio is None) == (causal_foreground_fraction is None):
        raise ValueError(
            "Specify exactly one of enrichment_ratio or causal_foreground_fraction"
        )
    if enrichment_ratio is not None:
        if isinstance(enrichment_ratio, (bool, np.bool_)):
            raise ValueError("enrichment_ratio must be a finite positive number")
        try:
            enrichment_ratio = float(enrichment_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError("enrichment_ratio must be a finite positive number") from exc
        if not np.isfinite(enrichment_ratio) or enrichment_ratio <= 0:
            raise ValueError("enrichment_ratio must be a finite positive number")
    else:
        if isinstance(causal_foreground_fraction, (bool, np.bool_)):
            raise ValueError("causal_foreground_fraction must be in [0, 1]")
        try:
            causal_foreground_fraction = float(causal_foreground_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("causal_foreground_fraction must be in [0, 1]") from exc
        if not np.isfinite(causal_foreground_fraction) or not 0 <= causal_foreground_fraction <= 1:
            raise ValueError("causal_foreground_fraction must be in [0, 1]")
    if maf_col not in df or foreground_col not in df:
        raise ValueError(f"df must contain {maf_col!r} and {foreground_col!r}")

    out = df.copy()
    maf = pd.to_numeric(out[maf_col], errors="coerce")
    if maf.isna().any() or ((maf < 0) | (maf > 0.5)).any():
        raise ValueError("MAF must contain finite values in [0, 0.5]")

    foreground = out[foreground_col]
    if not foreground.isin([True, False]).all():
        raise ValueError(f"{foreground_col!r} must contain only True/False")

    if maf_bins is None:
        maf_bins = np.linspace(0, 0.5, 11)

    maf_bins = np.asarray(maf_bins, dtype=float)
    if maf_bins.ndim != 1 or len(maf_bins) < 2:
        raise ValueError("maf_bins must contain at least two edges")
    if np.any(np.diff(maf_bins) <= 0):
        raise ValueError("maf_bins must be strictly increasing")
    if maf.min() < maf_bins[0] or maf.max() > maf_bins[-1]:
        raise ValueError("maf_bins do not cover all observed MAF values")

    rng = np.random.default_rng(seed)

    out["_maf_bin"] = pd.cut(
        maf,
        bins=maf_bins,
        include_lowest=True,
        right=True,
    )
    out[causal_col] = False

    # Exact genome-wide number of causal SNPs.
    n_total = len(out)
    n_causal_total = int(round(n_total * causal_fraction))

    bin_counts = out["_maf_bin"].value_counts(sort=False)
    if causal_foreground_fraction is None:
        # Allocate the exact total across MAF bins before weighting FG/BG.
        expected = n_causal_total * bin_counts / n_total
        causal_per_bin = np.floor(expected).astype(int)
        remainder = n_causal_total - causal_per_bin.sum()
        if remainder > 0:
            fractional = expected - causal_per_bin
            add_bins = fractional.sort_values(ascending=False).index[:remainder]
            causal_per_bin.loc[add_bins] += 1
    else:
        target_fg = int(round(n_causal_total * causal_foreground_fraction))
        target_bg = n_causal_total - target_fg
        n_fg_available = int(out[foreground_col].sum())
        n_bg_available = n_total - n_fg_available
        if target_fg > n_fg_available or target_bg > n_bg_available:
            raise ValueError(
                f"Requested causal_foreground_fraction is infeasible: "
                f"need {target_fg} foreground and {target_bg} background SNPs, "
                f"but only {n_fg_available} and {n_bg_available} are available"
            )
        fixed_counts = {
            maf_bin: {True: 0, False: 0} for maf_bin in bin_counts.index
        }
        for is_fg, target, available in (
            (True, target_fg, n_fg_available),
            (False, target_bg, n_bg_available),
        ):
            if target == 0:
                continue
            group_bins = out.loc[
                out[foreground_col] == is_fg, "_maf_bin"
            ].value_counts(sort=False)
            expected = target * group_bins / available
            allocated = np.floor(expected).astype(int)
            remainder = target - int(allocated.sum())
            if remainder:
                fractional = expected - allocated
                add_bins = fractional.sort_values(ascending=False).index[:remainder]
                allocated.loc[add_bins] += 1
            for maf_bin, count in allocated.items():
                fixed_counts[maf_bin][is_fg] = int(count)

    records = []

    for maf_bin, n_bin in bin_counts.items():
        subset = out["_maf_bin"] == maf_bin
        fg_idx = out.index[subset & out[foreground_col]]
        bg_idx = out.index[subset & ~out[foreground_col]]

        n_fg = len(fg_idx)
        n_bg = len(bg_idx)
        n_causal = (int(causal_per_bin.loc[maf_bin])
                    if causal_foreground_fraction is None else
                    fixed_counts[maf_bin][True] + fixed_counts[maf_bin][False])

        if n_causal == 0:
            records.append({
                "maf_bin": str(maf_bin),
                "n_total": n_bin,
                "n_fg": n_fg,
                "n_bg": n_bg,
                "n_causal": 0,
                "n_causal_fg": 0,
                "n_causal_bg": 0,
                "fg_rate": np.nan,
                "bg_rate": np.nan,
                "realized_enrichment": np.nan,
            })
            continue

        if causal_foreground_fraction is None:
            denom = enrichment_ratio * n_fg + n_bg
            if denom == 0:
                raise ValueError(f"No SNPs available in MAF bin {maf_bin}")
            # (C_F / N_F) / (C_B / N_B) = enrichment_ratio.
            expected_fg = n_causal * enrichment_ratio * n_fg / denom
            min_fg = max(0, n_causal - n_bg)
            max_fg = min(n_fg, n_causal)
            n_causal_fg = int(np.clip(round(expected_fg), min_fg, max_fg))
            n_causal_bg = n_causal - n_causal_fg
            q_bg = n_causal / denom
            q_fg = enrichment_ratio * q_bg
            if q_fg > 1 + 1e-12 or q_bg > 1 + 1e-12:
                raise ValueError(
                    f"Requested causal_fraction/enrichment_ratio is infeasible "
                    f"in MAF bin {maf_bin}: expected FG rate={q_fg:.3f}, "
                    f"BG rate={q_bg:.3f}"
                )
        else:
            n_causal_fg = fixed_counts[maf_bin][True]
            n_causal_bg = fixed_counts[maf_bin][False]

        if n_causal_fg:
            selected_fg = rng.choice(
                fg_idx.to_numpy(),
                size=n_causal_fg,
                replace=False,
            )
            out.loc[selected_fg, causal_col] = True

        if n_causal_bg:
            selected_bg = rng.choice(
                bg_idx.to_numpy(),
                size=n_causal_bg,
                replace=False,
            )
            out.loc[selected_bg, causal_col] = True

        fg_rate = n_causal_fg / n_fg if n_fg else np.nan
        bg_rate = n_causal_bg / n_bg if n_bg else np.nan
        realized_enrichment = (
            fg_rate / bg_rate
            if np.isfinite(fg_rate) and bg_rate > 0
            else np.nan
        )

        records.append({
            "maf_bin": str(maf_bin),
            "n_total": n_bin,
            "n_fg": n_fg,
            "n_bg": n_bg,
            "n_causal": n_causal,
            "n_causal_fg": n_causal_fg,
            "n_causal_bg": n_causal_bg,
            "fg_rate": fg_rate,
            "bg_rate": bg_rate,
            "realized_enrichment": realized_enrichment,
        })

    summary = pd.DataFrame(records)

    if out[causal_col].sum() != n_causal_total:
        raise RuntimeError(
            "Internal error: incorrect total number of causal SNPs")

    out = out.drop(columns="_maf_bin")

    return out, summary


def evaluate_causal_gwas_recovery(
    causal_effect_df,
    gwas_df,
    causal_id_col="SNP_ID",
    beta_col="beta_raw",
    gwas_id_col="ID",
    p_col="P_WALD",
    top_fractions=(0.001, 0.01, 0.05, 0.10),
):
    """
    Evaluate whether simulated causal SNPs are preferentially recovered by GWAS.

    Parameters
    ----------
    causal_effect_df : pandas.DataFrame
        Causal SNP table containing SNP ID and beta_raw.

    gwas_df : pandas.DataFrame
        Genome-wide GWAS result containing SNP ID and P_WALD.

    causal_id_col : str
        SNP ID column in causal_effect_df.

    beta_col : str
        Raw causal effect column.

    gwas_id_col : str
        SNP ID column in gwas_df.

    p_col : str
        GWAS P-value column.

    top_fractions : iterable of float
        GWAS top fractions used for recovery analysis.

    Returns
    -------
    result : dict
        Keys:
            summary
            recovery
            causal_table
    """
    causal = causal_effect_df[[causal_id_col, beta_col]].copy()
    gwas = gwas_df[[gwas_id_col, p_col]].copy()

    causal[causal_id_col] = causal[causal_id_col].astype(str)
    gwas[gwas_id_col] = gwas[gwas_id_col].astype(str)

    if causal[causal_id_col].duplicated().any():
        raise ValueError("Duplicated causal SNP IDs")
    if gwas[gwas_id_col].duplicated().any():
        raise ValueError("Duplicated GWAS SNP IDs")

    causal[beta_col] = pd.to_numeric(causal[beta_col], errors="coerce")
    gwas[p_col] = pd.to_numeric(gwas[p_col], errors="coerce")

    if causal[beta_col].isna().any():
        raise ValueError(f"{beta_col} contains missing/non-numeric values")
    if gwas[p_col].isna().any():
        raise ValueError(f"{p_col} contains missing/non-numeric values")
    if ((gwas[p_col] < 0) | (gwas[p_col] > 1)).any():
        raise ValueError(f"{p_col} must be in [0, 1]")

    causal_ids = set(causal[causal_id_col])
    gwas_ids = set(gwas[gwas_id_col])
    missing_ids = causal_ids - gwas_ids

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} causal SNPs are missing from GWAS results; "
            f"examples: {list(missing_ids)[:10]}"
        )

    tiny = np.finfo(float).tiny

    gwas["gwas_score"] = -np.log10(
        gwas[p_col].clip(lower=tiny)
    )
    gwas["is_causal"] = gwas[gwas_id_col].isin(causal_ids)

    # Smaller percentile = stronger GWAS signal.
    gwas["gwas_rank_percentile"] = (
        gwas[p_col]
        .rank(method="average", ascending=True)
        / len(gwas)
    )

    causal_table = causal.merge(
        gwas[
            [
                gwas_id_col,
                p_col,
                "gwas_score",
                "gwas_rank_percentile",
            ]
        ],
        left_on=causal_id_col,
        right_on=gwas_id_col,
        how="left",
        validate="one_to_one",
    )

    if gwas_id_col != causal_id_col:
        causal_table = causal_table.drop(columns=gwas_id_col)

    causal_table["beta2"] = causal_table[beta_col] ** 2

    beta2 = causal_table["beta2"].to_numpy()
    causal_score = causal_table["gwas_score"].to_numpy()
    noncausal_score = gwas.loc[
        ~gwas["is_causal"],
        "gwas_score",
    ].to_numpy()

    if np.std(beta2) > 0 and np.std(causal_score) > 0:
        pearson_r, pearson_p = pearsonr(beta2, causal_score)
        spearman_r, spearman_p = spearmanr(beta2, causal_score)
    else:
        pearson_r = pearson_p = np.nan
        spearman_r = spearman_p = np.nan

    if len(noncausal_score) > 0:
        mw = mannwhitneyu(
            causal_score,
            noncausal_score,
            alternative="greater",
        )

        # With "greater" and larger GWAS score = stronger signal,
        # U/(n1*n0) equals the probability that a random causal SNP
        # has a stronger score than a random noncausal SNP.
        auc = mw.statistic / (
            len(causal_score) * len(noncausal_score)
        )

        mw_u = mw.statistic
        mw_p = mw.pvalue
    else:
        auc = mw_u = mw_p = np.nan

    n_gwas = len(gwas)
    n_causal = len(causal)
    causal_fraction = n_causal / n_gwas

    recovery_records = []

    # Stable sort gives an exact number of top SNPs even if P-values tie.
    ranked = gwas.sort_values(
        p_col,
        ascending=True,
        kind="mergesort",
    )

    for fraction in top_fractions:
        if not 0 < fraction <= 1:
            raise ValueError(
                "All top_fractions must be in (0, 1]"
            )

        n_top = max(
            1,
            int(np.ceil(n_gwas * fraction)),
        )

        top = ranked.iloc[:n_top]
        n_causal_top = int(top["is_causal"].sum())

        recovery_rate = n_causal_top / n_causal
        expected_recovery = n_top / n_gwas
        recovery_lift = (
            recovery_rate / expected_recovery
        )

        precision = n_causal_top / n_top
        precision_lift = (
            precision / causal_fraction
        )

        recovery_records.append({
            "top_fraction": fraction,
            "n_top": n_top,
            "n_causal_top": n_causal_top,

            # Fraction of all true causal SNPs recovered.
            "causal_recovery_rate": recovery_rate,

            # Expected recovery if causal labels were random.
            "random_expected_recovery": expected_recovery,

            # >1 means causal SNPs are enriched among strong GWAS hits.
            "recovery_lift": recovery_lift,

            # Fraction of top GWAS SNPs that are causal.
            "precision": precision,

            # Baseline genome-wide causal fraction.
            "baseline_causal_fraction": causal_fraction,

            # >1 means top GWAS hits contain more causal SNPs than expected.
            "precision_lift": precision_lift,
        })

    recovery_df = pd.DataFrame(recovery_records)

    summary = {
        "n_gwas": n_gwas,
        "n_causal": n_causal,
        "causal_fraction": causal_fraction,

        "causal_gwas_score_mean": float(
            np.mean(causal_score)
        ),
        "causal_gwas_score_median": float(
            np.median(causal_score)
        ),

        "noncausal_gwas_score_mean": float(
            np.mean(noncausal_score)
        ),
        "noncausal_gwas_score_median": float(
            np.median(noncausal_score)
        ),

        "causal_rank_percentile_mean": float(
            causal_table["gwas_rank_percentile"].mean()
        ),
        "causal_rank_percentile_median": float(
            causal_table["gwas_rank_percentile"].median()
        ),

        "beta2_vs_gwas_score_pearson_r": float(
            pearson_r
        ),
        "beta2_vs_gwas_score_pearson_p": float(
            pearson_p
        ),

        "beta2_vs_gwas_score_spearman_r": float(
            spearman_r
        ),
        "beta2_vs_gwas_score_spearman_p": float(
            spearman_p
        ),

        "causal_vs_noncausal_auc": float(auc),
        "causal_vs_noncausal_mannwhitney_u": float(
            mw_u
        ),
        "causal_vs_noncausal_mannwhitney_p": float(
            mw_p
        ),
    }

    return {
        "summary": summary,
        "recovery": recovery_df,
        "causal_table": causal_table,
    }



class SIMULATION_JOB:
    """Run one reproducible causal-SNP phenotype simulation and GWAS.

    ``config`` is a dictionary with these required keys:

    * ``name``: readable run label; used in file names.
    * ``output_dir``: root directory for hash-addressed completed runs.
    * ``genome_fasta_file``: reference FASTA used by the GWAS plot.
    * ``site_info_df_h5``: HDF5 site annotation table, always loaded.
    * ``input_vcf_file``: VCF used for sample IDs, genotypes and GWAS.
    * ``kinship_ibs_file``: comma-separated, headerless kinship matrix in
      exactly the VCF sample order.
    * ``foreground_col``: boolean site-annotation column whose causal rate
      is enriched relative to the background.
    * Exactly one of ``causal_site_num`` (positive count) or
      ``causal_fraction`` (fraction of LD-pruned sites in [0, 1]).

    Optional keys and defaults:

    * ``site_info_key='df'``: HDF5 key for the annotation table.
    * ``maf_col='MAF'``: minor allele frequency column used for stratification.
    * ``enrichment_ratio=1.0``: target ratio of foreground to background
      causal probabilities within each MAF bin; 1 means no enrichment.
      Set to ``None`` when using ``causal_foreground_fraction``.
    * ``causal_foreground_fraction=None``: alternatively set the target
      fraction of causal SNPs in the foreground, from 0 to 1 inclusive.
      The foreground count is ``round(n_causal * fraction)``. SNPs are then
      sampled across MAF bins within each foreground/background group.
      The two controls cannot both be set or both be ``None``; infeasible
      counts raise an error.
    * ``ld_window_size=15000``: bp window for automatic LD pruning.
    * ``r2_threshold=0.5``: discard later SNP when pairwise r² exceeds this.
    * ``pool_num=40``: worker processes for automatic LD pruning.
    * ``pruned_site_parameters=None``: required provenance dictionary when
      ``pruned_site_df`` is supplied; record the external pruning method,
      window, threshold, seed and other parameters used to make that table.
    * ``pruned_site_ids_file=None``: path to a plain or gzip text file with
      one SNP ID per line. When set, reconstruct the pruned table from the
      site annotation HDF5 in file order and skip LD pruning. Missing or
      duplicated IDs are errors. Cannot be combined with ``pruned_site_df``.
    * ``effect_assignment={'method': 'random_sign_uniform', 'low': 0.5,
      'high': 1.5}``: rule for assigning raw causal SNP effects.
      ``normal`` draws each beta independently from a normal distribution;
      provide its mean ``beta_loc`` and positive SD ``scale``. The older
      nested key ``beta_scale`` is also accepted and normalized to ``scale``.
      ``{'method': 'constant', 'value': 1.0}`` assigns the same numeric
      ``beta_raw`` to every causal SNP. ``{'method': 'random_sign',
      'magnitude': 1.0}`` independently assigns +1 or -1 with equal
      probability to each site; ``magnitude`` must be positive.
      ``{'method': 'random_sign_uniform'}`` first samples independent +1/-1
      signs with equal probability, then samples magnitudes uniformly from
      ``[low, high)`` using the same RNG; defaults are ``low=0.5`` and
      ``high=1.5``. Both bounds must be positive and ``high > low``.
      ``{'method': 'vector', 'values': [0.5, -1.0, ...]}`` assigns a supplied
      one-dimensional numeric sequence positionally in ``causal_site_df['ID']``
      order. Its length must equal the number of sampled causal SNPs. The
      phenotype simulator subsequently rescales the joint causal component
      to ``causal_var``; these are raw, not final phenotype-scale effects.
    * ``causal_var=0.1``, ``kinship_var=0.6``, ``residual_var=0.3``:
      target realized sample variances of the three phenotype components;
      nonnegative values must sum to one.
    * ``seed=12345``: master seed; pruning, sampling, betas and phenotype
      use seed+1 through seed+4 respectively.
    * ``threshold_fdr=0.05``: FDR threshold for the GWAS summary plot.
    * ``lmm=1``: GEMMA linear mixed model mode (1=Wald, 4=all tests).
    * ``plot_bins=50``: number of bins in that plot.
    * ``enrichment={'top_percent': 0.05, 'pool_num': 40,
      'permutations': 1000, 'show_progress': True,
      'progress_interval': 2.0}``: after GWAS, compare the foreground and
      background top score quantiles using circular-shift permutations.
      The GWAS score is ``-log10(P_WALD)``; zero P-values are clipped to
      the smallest positive float. Results are saved as gzip JSON.

    Pass an optional ``pruned_site_df`` (also accepted by ``run``) to skip
    LD pruning. Every completed run stores ``pruned_site_ids_file`` in its
    result; pass that file as a config value to reuse the site selection in
    another run. The annotation HDF5 is always read. The exact effective
    config, input metadata and supplied table/ID-list content determine the
    run hash. Completed runs are returned from the manifest; a matching
    archived GWAS or unfinished GEMMA HDF can also be reused after the
    phenotype, VCF, kinship and LMM mode have been checked.
    """

    DEFAULTS = {
        "site_info_key": "df", "maf_col": "MAF", "causal_site_num": None,
        "causal_fraction": None, "enrichment_ratio": 1.0,
        "causal_foreground_fraction": None,
        "ld_window_size": 15000, "r2_threshold": 0.5, "pool_num": 40,
        "pruned_site_parameters": None, "pruned_site_ids_file": None,
        "effect_assignment": {"method": "random_sign_uniform",
                              "low": 0.5, "high": 1.5},
        "causal_var": 0.1, "kinship_var": 0.6,
        "residual_var": 0.3, "seed": 12345, "threshold_fdr": 0.05,
        "lmm": 1,
        "plot_bins": 50,
        "enrichment": {"top_percent": 0.05, "pool_num": 40,
                       "permutations": 1000, "show_progress": True,
                       "progress_interval": 2.0},
    }
    REQUIRED = {
        "name", "output_dir", "genome_fasta_file", "site_info_df_h5",
        "input_vcf_file", "kinship_ibs_file", "foreground_col",
    }

    def __init__(self, config, *, pruned_site_df=None):
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        if "beta_loc" in config or "beta_scale" in config:
            raise ValueError("Move beta_loc and beta_scale into effect_assignment")
        missing = self.REQUIRED - config.keys()
        unknown = config.keys() - (self.REQUIRED | self.DEFAULTS.keys())
        if missing or unknown:
            raise ValueError(f"Missing config keys: {sorted(missing)}; unknown keys: {sorted(unknown)}")
        self.config = {**self.DEFAULTS, **config}
        self.name = str(self.config["name"])
        if not self.name or Path(self.name).name != self.name or self.name in {".", ".."}:
            raise ValueError("name must be a nonempty file-name component")
        self.output_root = Path(self.config["output_dir"]).expanduser().resolve()
        self.genome_fasta_file = str(self.config["genome_fasta_file"])
        self.site_info_df_h5 = str(self.config["site_info_df_h5"])
        self.input_vcf_file = str(self.config["input_vcf_file"])
        self.kinship_ibs_file = str(self.config["kinship_ibs_file"])
        self.foreground_col = self.config["foreground_col"]
        self.site_info_key = self.config["site_info_key"]
        self.maf_col = self.config["maf_col"]
        self.pruned_site_df = pruned_site_df
        self.pruned_site_parameters = self.config["pruned_site_parameters"]
        self.input_pruned_site_ids_file = self.config["pruned_site_ids_file"]
        if pruned_site_df is not None and self.input_pruned_site_ids_file is not None:
            raise ValueError("Provide either pruned_site_df or pruned_site_ids_file, not both")
        if pruned_site_df is not None and (not isinstance(self.pruned_site_parameters, dict) or not self.pruned_site_parameters):
            raise ValueError("pruned_site_parameters must be a nonempty dictionary for supplied pruned_site_df")

        causal_site_num = self.config["causal_site_num"]
        causal_fraction = self.config["causal_fraction"]
        if (causal_site_num is None) == (causal_fraction is None):
            raise ValueError(
                "Specify exactly one of causal_site_num or causal_fraction"
            )

        self.causal_site_num = causal_site_num
        self.causal_fraction = causal_fraction
        ratio = self.config["enrichment_ratio"]
        foreground_fraction = self.config["causal_foreground_fraction"]
        if (ratio is None) == (foreground_fraction is None):
            raise ValueError(
                "Specify exactly one of enrichment_ratio or causal_foreground_fraction"
            )
        if ratio is not None:
            if isinstance(ratio, (bool, np.bool_)):
                raise ValueError("enrichment_ratio must be a finite positive number")
            try:
                ratio = float(ratio)
            except (TypeError, ValueError) as exc:
                raise ValueError("enrichment_ratio must be a finite positive number") from exc
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError("enrichment_ratio must be a finite positive number")
        else:
            if isinstance(foreground_fraction, (bool, np.bool_)):
                raise ValueError("causal_foreground_fraction must be in [0, 1]")
            try:
                foreground_fraction = float(foreground_fraction)
            except (TypeError, ValueError) as exc:
                raise ValueError("causal_foreground_fraction must be in [0, 1]") from exc
            if not np.isfinite(foreground_fraction) or not 0 <= foreground_fraction <= 1:
                raise ValueError("causal_foreground_fraction must be in [0, 1]")
        self.enrichment_ratio = ratio
        self.causal_foreground_fraction = foreground_fraction
        self.config["enrichment_ratio"] = ratio
        self.config["causal_foreground_fraction"] = foreground_fraction

        self.ld_window_size = int(self.config["ld_window_size"])
        self.r2_threshold = float(self.config["r2_threshold"])
        self.pool_num = int(self.config["pool_num"])

        self.effect_assignment = self._validate_effect_assignment(
            self.config["effect_assignment"]
        )
        self.config["effect_assignment"] = self.effect_assignment

        self.causal_var = float(self.config["causal_var"])
        self.kinship_var = float(self.config["kinship_var"])
        self.residual_var = float(self.config["residual_var"])

        if not np.isclose(
            self.causal_var + self.kinship_var + self.residual_var,
            1.0,
        ):
            raise ValueError(
                "causal_var + kinship_var + residual_var must equal 1"
            )
        if (self.effect_assignment["method"] == "constant"
                and self.effect_assignment["value"] == 0
                and self.causal_var > 0):
            raise ValueError(
                "constant effect value cannot be zero when causal_var > 0"
            )
        if self.effect_assignment["method"] == "vector":
            values = self.effect_assignment["values"]
            if self.causal_site_num is not None and len(values) != self.causal_site_num:
                raise ValueError(
                    f"effect_assignment values has {len(values)} entries, "
                    f"but causal_site_num is {self.causal_site_num}"
                )
            if self.causal_var > 0 and not any(values):
                raise ValueError(
                    "effect_assignment values cannot all be zero when causal_var > 0"
                )

        self.seed = int(self.config["seed"])
        self.prune_seed = self.seed + 1
        self.causal_sample_seed = self.seed + 2
        self.beta_seed = self.seed + 3
        self.phenotype_seed = self.seed + 4

        self.threshold_fdr = float(self.config["threshold_fdr"])
        self.lmm = self.config["lmm"]
        if isinstance(self.lmm, (bool, np.bool_)) or self.lmm not in (1, 4):
            raise ValueError("lmm must be 1 (Wald) or 4 (all tests)")
        self.lmm = int(self.lmm)
        self.config["lmm"] = self.lmm
        self.plot_bins = int(self.config["plot_bins"])
        self.enrichment = self._validate_enrichment(self.config["enrichment"])
        self.config["enrichment"] = self.enrichment
        self.output_dir = None

        self._start_time = None

    @staticmethod
    def _validate_effect_assignment(value):
        if not isinstance(value, dict):
            raise TypeError("effect_assignment must be a dictionary")
        method = value.get("method")
        if method not in {"normal", "constant", "random_sign",
                          "random_sign_uniform", "vector"}:
            raise ValueError(f"Unknown effect_assignment method: {method!r}")
        if method == "normal" and set(value) == {"method", "beta_loc", "beta_scale"}:
            value = {"method": "normal", "beta_loc": value["beta_loc"],
                     "scale": value["beta_scale"]}
        if method == "random_sign_uniform":
            value = {"low": 0.5, "high": 1.5, **value}
        required = {
            "normal": {"method", "beta_loc", "scale"},
            "constant": {"method", "value"},
            "random_sign": {"method", "magnitude"},
            "random_sign_uniform": {"method", "low", "high"},
            "vector": {"method", "values"},
        }[method]
        if set(value) != required:
            raise ValueError(
                f"effect_assignment method {method!r} requires exactly "
                f"{sorted(required)}"
            )
        if method == "vector":
            supplied = value["values"]
            if isinstance(supplied, (str, bytes, dict)):
                raise ValueError("effect_assignment values must be a one-dimensional numeric sequence")
            try:
                raw = np.asarray(supplied, dtype=object)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "effect_assignment values must be a one-dimensional numeric sequence"
                ) from exc
            if raw.ndim != 1 or len(raw) == 0 or any(
                isinstance(item, (bool, np.bool_)) for item in raw
            ):
                raise ValueError("effect_assignment values must be a nonempty one-dimensional numeric sequence")
            try:
                values = raw.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("effect_assignment values must be numeric") from exc
            if not np.isfinite(values).all():
                raise ValueError("effect_assignment values must be finite")
            return {"method": "vector", "values": values.tolist()}
        keys = required - {"method"}
        for key in keys:
            if isinstance(value[key], bool):
                raise ValueError(f"effect_assignment {key} must be a finite number")
            try:
                number = float(value[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"effect_assignment {key} must be a finite number"
                ) from exc
            if not np.isfinite(number):
                raise ValueError(f"effect_assignment {key} must be a finite number")
        result = {"method": method, **{key: float(value[key]) for key in keys}}
        if method == "normal" and result["scale"] <= 0:
            raise ValueError("effect_assignment scale must be > 0")
        if method == "random_sign" and result["magnitude"] <= 0:
            raise ValueError("effect_assignment magnitude must be > 0")
        if method == "random_sign_uniform":
            if result["low"] <= 0:
                raise ValueError("effect_assignment low must be > 0")
            if result["high"] <= result["low"]:
                raise ValueError("effect_assignment high must be > low")
        return result

    @classmethod
    def _validate_enrichment(cls, value):
        if not isinstance(value, dict):
            raise TypeError("enrichment must be a dictionary")
        defaults = cls.DEFAULTS["enrichment"]
        unknown = set(value) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown enrichment keys: {sorted(unknown)}")
        result = {**defaults, **value}
        for key in ("top_percent", "progress_interval"):
            if isinstance(result[key], (bool, np.bool_)):
                raise ValueError(f"enrichment {key} must be a finite number")
            try:
                result[key] = float(result[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"enrichment {key} must be a finite number") from exc
            if not np.isfinite(result[key]):
                raise ValueError(f"enrichment {key} must be a finite number")
        if not 0 < result["top_percent"] < 1:
            raise ValueError("enrichment top_percent must be in (0, 1)")
        if result["progress_interval"] <= 0:
            raise ValueError("enrichment progress_interval must be > 0")
        for key in ("pool_num", "permutations"):
            number = result[key]
            if isinstance(number, (bool, np.bool_)) or not isinstance(number, (int, np.integer)) or number <= 0:
                raise ValueError(f"enrichment {key} must be a positive integer")
            result[key] = int(number)
        if not isinstance(result["show_progress"], (bool, np.bool_)):
            raise ValueError("enrichment show_progress must be boolean")
        result["show_progress"] = bool(result["show_progress"])
        return result

    def _set_paths(self, directory):
        self.output_dir = Path(directory)
        self.output_prefix = self.output_dir / self.name
        self.log_file = self.output_dir / f"{self.name}.simulation.log"
        self.gemma_log_file = self.output_dir / f"{self.name}.gemma.log"
        self.config_file = self.output_dir / f"{self.name}.config.json"
        self.manifest_file = self.output_dir / f"{self.name}.manifest.json"
        self.causal_site_file = self.output_dir / f"{self.name}.causal_sites.tsv"
        self.causal_effect_file = self.output_dir / f"{self.name}.causal_effects.tsv"
        self.causal_sampling_file = self.output_dir / f"{self.name}.causal_sampling.tsv"
        self.qc_summary_file = self.output_dir / f"{self.name}.gwas_qc.json"
        self.qc_recovery_file = self.output_dir / f"{self.name}.gwas_qc_recovery.tsv"
        self.qc_causal_table_file = self.output_dir / f"{self.name}.gwas_qc_causal.tsv"
        self.phenotype_archive = self.output_dir / f"{self.name}.phenotype.csv.gz"
        self.gwas_archive = self.output_dir / f"{self.name}.gwas.csv.gz"
        self.results_json_archive = self.output_dir / f"{self.name}.results.json.gz"
        self.pruned_site_ids_file = self.output_dir / f"{self.name}.pruned_site_ids.txt.gz"
        self.pruning_metadata_file = self.output_dir / f"{self.name}.pruning.json"
        self.enrichment_file = self.output_dir / f"{self.name}.enrichment.json.gz"

    def _now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, message):
        line = f"[{self._now()}] {message}"
        print(line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def _step(self, i, n, message):
        self.log(f"[{i}/{n}] {message}")

    def _config(self):
        return {
            **self.config,
            "prune_seed": self.prune_seed,
            "causal_sample_seed": self.causal_sample_seed,
            "beta_seed": self.beta_seed,
            "phenotype_seed": self.phenotype_seed,
        }

    def _run_hash(self, pruned_site_df, plot):
        inputs = {}
        for key in ("genome_fasta_file", "site_info_df_h5", "input_vcf_file", "kinship_ibs_file"):
            path = Path(self.config[key]).expanduser().resolve()
            stat = path.stat()
            inputs[key] = {"path": str(path), "size": stat.st_size,
                           "mtime_ns": stat.st_mtime_ns}
        if pruned_site_df is not None:
            if not isinstance(pruned_site_df, pd.DataFrame) or pruned_site_df.empty:
                raise ValueError("pruned_site_df must be a nonempty DataFrame")
            h = hashlib.sha256()
            h.update(pd.util.hash_pandas_object(pruned_site_df, index=True).values.tobytes())
            h.update(json.dumps([(str(c), str(t)) for c, t in
                                 zip(pruned_site_df.columns, pruned_site_df.dtypes)]).encode())
            inputs["pruned_site_df_sha256"] = h.hexdigest()
        if self.input_pruned_site_ids_file is not None:
            path = Path(self.input_pruned_site_ids_file).expanduser().resolve()
            inputs["pruned_site_ids_file"] = {
                "path": str(path), "sha256": self._file_sha256(path),
            }
        pruning_mode = ("supplied_df" if pruned_site_df is not None else
                        "id_list" if self.input_pruned_site_ids_file is not None else
                        "automatic")
        payload = {"config": self._config(), "inputs": inputs,
                   "pruning_mode": pruning_mode,
                   "plot": bool(plot), "pipeline_version": 5}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest(), payload

    @staticmethod
    def _file_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _read_site_ids(path):
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as source:
            ids = [line.rstrip("\r\n") for line in source]
        if not ids or any(not item or item != item.strip() for item in ids):
            raise ValueError(f"ID list has no IDs or contains blank/whitespace IDs: {path}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"ID list contains duplicate SNP IDs: {path}")
        return ids

    @staticmethod
    def _from_site_ids(site_info_df, ids):
        if "ID" not in site_info_df.columns:
            raise ValueError("site_info_df must contain ID")
        site_ids = pd.Index(site_info_df["ID"].astype(str))
        if not site_ids.is_unique:
            raise ValueError("site_info_df contains duplicate SNP IDs")
        positions = site_ids.get_indexer(ids)
        missing = [ids[i] for i, position in enumerate(positions) if position < 0]
        if missing:
            raise ValueError(f"{len(missing)} pruned SNP IDs missing from site_info_df; examples: {missing[:10]}")
        return site_info_df.iloc[positions].copy()

    def _write_site_ids(self, pruned_site_df, pruning_record):
        if "ID" not in pruned_site_df.columns:
            raise ValueError("pruned_site_df must contain ID")
        raw_ids = pruned_site_df["ID"]
        if raw_ids.isna().any():
            raise ValueError("pruned_site_df contains missing SNP IDs")
        ids = raw_ids.astype(str).tolist()
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("pruned_site_df must contain unique SNP IDs")
        with gzip.open(self.pruned_site_ids_file, "wt") as out:
            for item in ids:
                out.write(item + "\n")
        with open(self.pruning_metadata_file, "w") as out:
            json.dump({"pruning_record": pruning_record,
                       "n_sites": len(ids),
                       "ids_sha256": self._file_sha256(self.pruned_site_ids_file)}, out, indent=2)
        self.log(f"Saved {len(ids):,} pruned SNP IDs: {self.pruned_site_ids_file}")

    def load_pruned_site_df(self):
        """Rebuild the completed run's pruned table from saved IDs and annotations."""
        if self.output_dir is None or not self.pruned_site_ids_file.is_file():
            raise FileNotFoundError("No completed pruned-site ID list is available")
        ids = self._read_site_ids(self.pruned_site_ids_file)
        site_info_df = pd.read_hdf(self.site_info_df_h5, key=self.site_info_key)
        return self._from_site_ids(site_info_df, ids)

    def _result(self, cached):
        with open(self.qc_summary_file) as f:
            summary = json.load(f)
        return {
            "name": self.name, "run_hash": self.run_hash, "cached": cached,
            "output_dir": self.output_dir, "config": self._config(),
            "phenotype_file": self.phenotype_archive,
            "gwas_file": self.gwas_archive,
            "results_json_file": self.results_json_archive,
            "pruned_site_ids_file": self.pruned_site_ids_file,
            "pruning_metadata_file": self.pruning_metadata_file,
            "enrichment_file": self.enrichment_file,
            "enrichment_result": self._read_enrichment_result(),
            "phenotype_df": pd.read_csv(self.phenotype_archive),
            "gwas_df": pd.read_csv(self.gwas_archive),
            "causal_site_df": pd.read_csv(self.causal_site_file, sep="\t"),
            "causal_effect_df": pd.read_csv(self.causal_effect_file, sep="\t"),
            "gwas_qc": {
                "summary": summary,
                "recovery": pd.read_csv(self.qc_recovery_file, sep="\t"),
                "causal_table": pd.read_csv(self.qc_causal_table_file, sep="\t"),
            },
            "log_file": self.log_file, "gemma_log_file": self.gemma_log_file,
            "config_file": self.config_file,
        }

    def _read_enrichment_result(self):
        with gzip.open(self.enrichment_file, "rt") as source:
            return json.load(source)

    def _collect_gemma_logs(self, source_dir=None):
        source_dir = self.output_dir if source_dir is None else Path(source_dir)
        blocks = source_dir / f"{self.name}.gemma" / "blocks"
        logs = sorted(blocks.glob("block_*.gemma/output/gemma_output.log.txt"))
        with open(self.gemma_log_file, "w") as out:
            for path in logs:
                out.write(f"===== {path.parent.parent.name} =====\n")
                with open(path, errors="replace") as source:
                    shutil.copyfileobj(source, out)
                out.write("\n")
                wrapper_log = path.parent.parent / "gemma.log"
                if wrapper_log.is_file():
                    out.write("--- command output ---\n")
                    with open(wrapper_log, errors="replace") as source:
                        shutil.copyfileobj(source, out)
                    out.write("\n")
        self.log(f"Collected GEMMA logs from {len(logs)} blocks")

    def _archive_and_cleanup(self, phenotype_outputs, gwas_df, gemma_job):
        phenotype = pd.read_csv(phenotype_outputs["phenotype_file"], sep="\t")
        phenotype[["sample_id", "phenotype"]].to_csv(
            self.phenotype_archive, index=False, compression="gzip")
        if not {"ID", "P_WALD"}.issubset(gwas_df.columns):
            raise ValueError("GWAS table must contain ID and P_WALD")
        gwas_df[["ID", "P_WALD"]].to_csv(
            self.gwas_archive, index=False, compression="gzip")
        with open(self.config_file) as source:
            run_spec = json.load(source)
        with open(self.qc_summary_file) as source:
            qc_summary = json.load(source)
        enrichment_result = self._read_enrichment_result()
        with gzip.open(self.results_json_archive, "wt") as out:
            json.dump({"run_hash": self.run_hash, "run_spec": run_spec,
                       "gwas_qc_summary": qc_summary,
                       "enrichment_stat": enrichment_result["stat"],
                       "n_samples": len(phenotype), "n_gwas": len(gwas_df)}, out)
        plot_file = Path(gemma_job.gwas_df_h5.replace(".h5", "_gwas_summary.png"))
        if plot_file.is_file():
            shutil.copy2(plot_file, self.output_dir / f"{self.name}.gwas_summary.png")
        if not self.gemma_log_file.is_file():
            self._collect_gemma_logs()
        shutil.rmtree(self.output_dir / f"{self.name}.gemma")
        for path in self.output_dir.glob(f"{self.name}.*"):
            if path.name.endswith((".gemma.pheno", ".phenotype.tsv", ".components.tsv",
                                   ".bimbam")):
                path.unlink()
        self.log("Archived phenotype and GWAS as gzip CSV; removed intermediate files")

    def _load_inputs(self):
        site_info_df = pd.read_hdf(
            self.site_info_df_h5,
            key=self.site_info_key,
        )

        sample_id_list = get_sample_list_from_vcf(
            self.input_vcf_file
        )

        kinship_df = pd.read_csv(
            self.kinship_ibs_file,
            sep=",",
            header=None,
            names=sample_id_list,
        )
        kinship_df.index = sample_id_list

        if self.foreground_col not in site_info_df.columns:
            raise ValueError(
                f"{self.foreground_col!r} not found in site_info_df"
            )

        if self.maf_col not in site_info_df.columns:
            raise ValueError(
                f"{self.maf_col!r} not found in site_info_df"
            )

        return site_info_df, sample_id_list, kinship_df

    def _log_fg_bg(self, df, label):
        fg_num = int(df[self.foreground_col].sum())
        bg_num = len(df) - fg_num
        fg_prop = fg_num / len(df)
        bg_prop = bg_num / len(df)

        self.log(
            f"{label}: total={len(df):,}, "
            f"FG={fg_num:,} ({fg_prop:.2%}), "
            f"BG={bg_num:,} ({bg_prop:.2%})"
        )

    def _prune_sites(self, site_info_df):
        self.log(
            f"LD pruning started: window={self.ld_window_size:,} bp, "
            f"r2>{self.r2_threshold}, seed={self.prune_seed}, "
            f"pool={self.pool_num}"
        )

        t0 = time.time()

        pruned_with_log_df = prune_sites_by_ld(
            site_info_df.sample(
                frac=1,
                random_state=self.prune_seed,
            ),
            self.input_vcf_file,
            ld_window_size=self.ld_window_size,
            r2_threshold=self.r2_threshold,
            pool_num=self.pool_num,
            full_output=True,
        )

        pruned_with_log_df.sort_index(inplace=True)

        pruned_site_df = pruned_with_log_df[
            pruned_with_log_df["LD_PRUNING_KEEP"]
        ].copy()

        self.log(
            f"LD pruning finished in {(time.time() - t0) / 60:.2f} min: "
            f"{len(site_info_df):,} -> {len(pruned_site_df):,} SNPs "
            f"({len(pruned_site_df) / len(site_info_df):.2%} retained)"
        )

        return pruned_site_df

    def _sample_causal_sites(self, pruned_site_df):
        if self.causal_site_num is not None:
            if self.causal_site_num <= 0:
                raise ValueError("causal_site_num must be > 0")
            if self.causal_site_num > len(pruned_site_df):
                raise ValueError(
                    "causal_site_num exceeds number of pruned SNPs"
                )

            causal_fraction = (
                self.causal_site_num / len(pruned_site_df)
            )
        else:
            causal_fraction = float(self.causal_fraction)

        control = (
            f"enrichment_ratio={self.enrichment_ratio:g}"
            if self.enrichment_ratio is not None else
            f"causal_foreground_fraction={self.causal_foreground_fraction:g}"
        )
        self.log(
            f"Sampling causal SNPs: target_fraction={causal_fraction:.6g}, "
            f"target_n≈{causal_fraction * len(pruned_site_df):.1f}, "
            f"{control}, "
            f"seed={self.causal_sample_seed}"
        )

        sampled_df, summary = sample_causal_snps(
            pruned_site_df,
            causal_fraction=causal_fraction,
            enrichment_ratio=self.enrichment_ratio,
            causal_foreground_fraction=self.causal_foreground_fraction,
            maf_col=self.maf_col,
            foreground_col=self.foreground_col,
            seed=self.causal_sample_seed,
            causal_col="causal",
        )

        causal_site_df = sampled_df[
            sampled_df["causal"]
        ].copy()

        fg = sampled_df[self.foreground_col].astype(bool)

        fg_causal_rate = sampled_df.loc[
            fg, "causal"
        ].mean()

        bg_causal_rate = sampled_df.loc[
            ~fg, "causal"
        ].mean()

        realized_enrichment = (
            fg_causal_rate / bg_causal_rate
            if bg_causal_rate > 0
            else np.nan
        )

        n_causal_fg = int(
            causal_site_df[self.foreground_col].sum()
        )
        n_causal_bg = len(causal_site_df) - n_causal_fg

        self.log(
            f"Causal sampling finished: n={len(causal_site_df):,}, "
            f"FG={n_causal_fg:,}, BG={n_causal_bg:,}, "
            f"FG rate={fg_causal_rate:.4%}, "
            f"BG rate={bg_causal_rate:.4%}, "
            f"realized enrichment={realized_enrichment:.4f}"
        )

        summary.to_csv(
            self.causal_sampling_file,
            sep="\t",
            index=False,
        )

        causal_site_df.to_csv(
            self.causal_site_file,
            sep="\t",
            index=False,
        )

        return sampled_df, causal_site_df, summary

    def _prepare_causal_effects(self, causal_site_df):
        self.log(
            f"Extracting genotype for {len(causal_site_df):,} causal SNPs"
        )

        t0 = time.time()

        causal_genotype_df = (
            get_genotype_df_from_vcf_by_site_df(
                self.input_vcf_file,
                causal_site_df,
            )
            .transpose()
        )

        self.log(
            f"Causal genotype extraction finished in "
            f"{time.time() - t0:.2f} s: "
            f"shape={causal_genotype_df.shape}"
        )

        causal_ids = (
            causal_site_df["ID"]
            .astype(str)
            .tolist()
        )

        causal_genotype_df.columns = (
            causal_genotype_df.columns.astype(str)
        )

        if set(causal_genotype_df.columns) != set(causal_ids):
            raise ValueError(
                "Extracted causal genotype SNP IDs do not match causal_site_df"
            )

        causal_genotype_df = causal_genotype_df.loc[
            :,
            causal_ids,
        ]

        causal_effect_df = pd.DataFrame({
            "SNP_ID": causal_ids,
            "beta_raw": self._assign_effects(len(causal_ids)),
        })

        causal_effect_df.to_csv(
            self.causal_effect_file,
            sep="\t",
            index=False,
        )

        if self.effect_assignment["method"] == "normal":
            description = (
                f"N({self.effect_assignment['beta_loc']:g}, "
                f"{self.effect_assignment['scale']:g}^2), seed={self.beta_seed}"
            )
        elif self.effect_assignment["method"] == "constant":
            description = f"constant={self.effect_assignment['value']:g}"
        elif self.effect_assignment["method"] == "random_sign":
            description = (
                f"random_sign magnitude={self.effect_assignment['magnitude']:g}, "
                f"P(+)=P(-)=0.5, seed={self.beta_seed}"
            )
        elif self.effect_assignment["method"] == "random_sign_uniform":
            description = (
                f"random_sign_uniform magnitude=U("
                f"{self.effect_assignment['low']:g}, "
                f"{self.effect_assignment['high']:g}), "
                f"P(+)=P(-)=0.5, seed={self.beta_seed}"
            )
        else:
            description = f"vector n={len(self.effect_assignment['values'])}"
        self.log(
            f"Beta assigned: {description}, "
            f"mean={causal_effect_df['beta_raw'].mean():.4f}, "
            f"sd={causal_effect_df['beta_raw'].std(ddof=0):.4f}"
        )

        return causal_genotype_df, causal_effect_df

    def _assign_effects(self, n_sites):
        """Assign raw betas according to the configured effect rule."""
        rule = self.effect_assignment
        if rule["method"] == "normal":
            return np.random.default_rng(self.beta_seed).normal(
                loc=rule["beta_loc"], scale=rule["scale"],
                size=n_sites,
            )
        if rule["method"] == "constant":
            return np.full(n_sites, rule["value"], dtype=float)
        if rule["method"] == "random_sign":
            signs = np.random.default_rng(self.beta_seed).choice(
                [-1.0, 1.0], size=n_sites
            )
            return signs * rule["magnitude"]
        if rule["method"] == "random_sign_uniform":
            beta_rng = np.random.default_rng(self.beta_seed)
            signs = beta_rng.choice([-1.0, 1.0], size=n_sites)
            magnitudes = beta_rng.uniform(
                rule["low"], rule["high"], size=n_sites
            )
            return signs * magnitudes
        if rule["method"] == "vector":
            values = rule["values"]
            if len(values) != n_sites:
                raise ValueError(
                    f"effect_assignment values has {len(values)} entries, "
                    f"but {n_sites} causal SNPs were sampled"
                )
            return np.asarray(values, dtype=float).copy()
        raise RuntimeError(f"Unsupported effect assignment: {rule['method']}")

    def _simulate_phenotype(
        self,
        causal_genotype_df,
        causal_effect_df,
        kinship_df,
    ):
        self.log(
            f"Phenotype simulation started: "
            f"causal_var={self.causal_var:g}, "
            f"kinship_var={self.kinship_var:g}, "
            f"residual_var={self.residual_var:g}, "
            f"seed={self.phenotype_seed}"
        )

        t0 = time.time()

        outputs = simulate_phenotype_with_kinship(
            causal_genotype_df=causal_genotype_df,
            causal_effect_df=causal_effect_df,
            kinship_df=kinship_df,
            output_prefix=self.output_prefix,
            causal_var=self.causal_var,
            kinship_var=self.kinship_var,
            residual_var=self.residual_var,
            seed=self.phenotype_seed,
        )

        meta = outputs["metadata"]

        self.log(
            f"Phenotype simulation finished in {time.time() - t0:.2f} s: "
            f"n={meta['n_samples']}, causal={meta['n_causal']}, "
            f"phenotype_var={meta['realized_phenotype_variance']:.4f}"
        )

        return outputs

    def _prepare_gemma_pheno(self, outputs):
        phenotype_file = str(
            outputs["phenotype_file"]
        )

        gemma_pheno_file = (
            str(self.output_prefix)
            + ".bimbam"
        )

        pheno_df = pd.read_csv(
            phenotype_file,
            sep="\t",
        )

        pheno_df["phenotype"].to_csv(
            gemma_pheno_file,
            index=False,
            header=False,
        )

        self.log(
            f"GEMMA phenotype prepared: {gemma_pheno_file}"
        )

        return gemma_pheno_file

    def _find_existing_gemma_result(self):
        """Find a previous GWAS for the same phenotype, inputs and LMM mode."""
        current_inputs = self._current_run_spec["inputs"]
        current_phenotype = pd.read_csv(
            f"{self.output_prefix}.phenotype.tsv", sep="\t"
        )[["sample_id", "phenotype"]]
        for directory in sorted(self.output_root.iterdir(),
                                key=lambda path: path.stat().st_mtime_ns,
                                reverse=True):
            if not directory.is_dir() or directory == self.output_dir:
                continue
            spec_file = directory / f"{self.name}.config.json"
            if not spec_file.is_file():
                continue
            try:
                with open(spec_file) as source:
                    spec = json.load(source)
                candidate_inputs = spec["inputs"]
                if any(candidate_inputs.get(key) != current_inputs.get(key)
                       for key in ("input_vcf_file", "kinship_ibs_file")):
                    continue
                if spec["config"].get("lmm", 4) != self.lmm:
                    continue
                phenotype_file = directory / f"{self.name}.phenotype.csv.gz"
                if phenotype_file.is_file():
                    previous_phenotype = pd.read_csv(phenotype_file)
                else:
                    phenotype_file = directory / f"{self.name}.phenotype.tsv"
                    if not phenotype_file.is_file():
                        continue
                    previous_phenotype = pd.read_csv(phenotype_file, sep="\t")
                previous_phenotype = previous_phenotype[["sample_id", "phenotype"]]
                if (previous_phenotype["sample_id"].astype(str).tolist()
                        != current_phenotype["sample_id"].astype(str).tolist()
                        or not np.array_equal(
                            previous_phenotype["phenotype"].to_numpy(),
                            current_phenotype["phenotype"].to_numpy(),
                        )):
                    continue
                h5_file = (directory / f"{self.name}.gemma" /
                           f"{self.name}.gemma.gwas_df.h5")
                if h5_file.is_file():
                    gwas_df = pd.read_hdf(h5_file, key="df")
                    source_file = h5_file
                else:
                    source_file = directory / f"{self.name}.gwas.csv.gz"
                    if not source_file.is_file():
                        continue
                    gwas_df = pd.read_csv(source_file)
                if gwas_df.empty or not {"ID", "P_WALD"}.issubset(gwas_df.columns):
                    continue
                return directory, source_file, gwas_df
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return None

    def _restore_gemma_result(self, existing, gemma_job):
        directory, source_file, gwas_df = existing
        if source_file.suffix == ".gz":
            site_info_df = self._gemma_site_info_df[["ID", "CHROM", "POS"]].copy()
            site_info_df["ID"] = site_info_df["ID"].astype(str)
            gwas_df["ID"] = gwas_df["ID"].astype(str)
            if site_info_df["ID"].duplicated().any() or gwas_df["ID"].duplicated().any():
                raise ValueError("Cannot restore GWAS with duplicate SNP IDs")
            gwas_df = gwas_df.merge(site_info_df, on="ID", how="left",
                                    validate="one_to_one")
            if gwas_df[["CHROM", "POS"]].isna().any().any():
                raise ValueError("Archived GWAS SNP IDs are missing from site_info_df")
            gwas_df["P_VALUE"] = gwas_df["P_WALD"]
            gwas_df["FDR"] = multipletests(gwas_df["P_VALUE"], method="fdr_bh")[1]
        Path(gemma_job.gwas_df_h5).parent.mkdir(parents=True, exist_ok=True)
        if source_file.suffix == ".h5":
            shutil.copy2(source_file, gemma_job.gwas_df_h5)
        else:
            gwas_df.to_hdf(gemma_job.gwas_df_h5, key="df", mode="w")
        prior_log = directory / f"{self.name}.gemma.log"
        if prior_log.is_file():
            shutil.copy2(prior_log, self.gemma_log_file)
        else:
            self._collect_gemma_logs(directory)
        with open(self.config_file) as source:
            run_spec = json.load(source)
        run_spec["gemma_reused_from"] = str(source_file)
        with open(self.config_file, "w") as out:
            json.dump(run_spec, out, indent=2, ensure_ascii=False)
        self.log(f"Reused GEMMA GWAS from {source_file}")
        return gwas_df

    def _run_gemma(self, gemma_pheno_file):
        self.log("GEMMA GWAS started")

        t0 = time.time()

        gemma_job = GEMMA_JOB(
            self.name,
            str(self.output_dir),
            self.input_vcf_file,
            gemma_pheno_file,
            self.kinship_ibs_file,
            self.lmm,
        )

        existing = self._find_existing_gemma_result()
        if existing is None:
            gemma_job.run()
            gwas_df = pd.read_hdf(gemma_job.gwas_df_h5)
        else:
            gwas_df = self._restore_gemma_result(existing, gemma_job)

        self.log(
            f"GEMMA GWAS finished in {(time.time() - t0) / 60:.2f} min"
        )

        self.log(
            f"GWAS result loaded: {len(gwas_df):,} SNPs"
        )

        return gemma_job, gwas_df

    def _plot_gwas(self, gemma_job):
        self.log("Generating GWAS summary plot")

        plot_gwas_summary_panel(
            gemma_job.gwas_df_h5,
            self.genome_fasta_file,
            threshold_fdr=self.threshold_fdr,
            bins=self.plot_bins,
            title=None,
            figsize=(14, 6),
            save=True,
        )

        self.log("GWAS summary plot finished")

    def _evaluate_gwas(
        self,
        causal_effect_df,
        gwas_df,
    ):
        self.log("Evaluating causal SNP GWAS recovery")

        qc = evaluate_causal_gwas_recovery(
            causal_effect_df=causal_effect_df,
            gwas_df=gwas_df,
        )

        qc["recovery"].to_csv(
            self.qc_recovery_file,
            sep="\t",
            index=False,
        )

        qc["causal_table"].to_csv(
            self.qc_causal_table_file,
            sep="\t",
            index=False,
        )

        with open(
            self.qc_summary_file,
            "w",
        ) as f:
            json.dump(
                qc["summary"],
                f,
                indent=2,
            )

        summary = qc["summary"]

        self.log(
            f"GWAS recovery QC: "
            f"AUC={summary['causal_vs_noncausal_auc']:.4f}, "
            f"causal median rank percentile="
            f"{summary['causal_rank_percentile_median']:.4f}, "
            f"beta2~GWAS Spearman="
            f"{summary['beta2_vs_gwas_score_spearman_r']:.4f}"
        )

        for _, row in qc["recovery"].iterrows():
            self.log(
                f"Top {row['top_fraction']:.1%}: "
                f"{int(row['n_causal_top'])} causal SNPs, "
                f"recovery={row['causal_recovery_rate']:.3%}, "
                f"lift={row['recovery_lift']:.3f}"
            )

        return qc

    def _run_enrichment(self, site_info_df, gwas_df):
        required = {"ID", "CHROM", "POS", self.maf_col, self.foreground_col}
        missing = required - set(site_info_df.columns)
        if missing:
            raise ValueError(f"site_info_df missing enrichment columns: {sorted(missing)}")
        if not {"ID", "P_WALD"}.issubset(gwas_df.columns):
            raise ValueError("GWAS table must contain ID and P_WALD")

        context = site_info_df[["ID", "CHROM", "POS", self.maf_col,
                                self.foreground_col]].copy()
        context = context.rename(columns={self.foreground_col: "foreground"})
        context["ID"] = context["ID"].astype(str)
        gwas = gwas_df[["ID", "P_WALD"]].copy()
        gwas["ID"] = gwas["ID"].astype(str)
        if context["ID"].duplicated().any() or gwas["ID"].duplicated().any():
            raise ValueError("Enrichment requires unique site and GWAS IDs")
        if not context["foreground"].isin([True, False]).all():
            raise ValueError("Enrichment foreground must contain only True/False")
        gwas["P_WALD"] = pd.to_numeric(gwas["P_WALD"], errors="coerce")
        if gwas["P_WALD"].isna().any() or not np.isfinite(gwas["P_WALD"]).all():
            raise ValueError("GWAS P_WALD must be finite numeric values")
        if ((gwas["P_WALD"] < 0) | (gwas["P_WALD"] > 1)).any():
            raise ValueError("GWAS P_WALD must be in [0, 1]")
        context = context.merge(gwas, on="ID", how="left", validate="one_to_one")
        context[self.name] = -np.log10(
            context["P_WALD"].clip(lower=np.finfo(float).tiny)
        )
        for foreground in (True, False):
            if not context.loc[context["foreground"] == foreground, self.name].notna().any():
                raise ValueError(f"No GWAS P-values for enrichment foreground={foreground}")

        options = self.enrichment
        self.log(
            f"Enrichment permutations started: top={options['top_percent']:.1%}, "
            f"pool={options['pool_num']}, permutations={options['permutations']}"
        )
        started = time.time()
        result = enrichment_permutation_analysis(
            context, options["top_percent"], [self.name],
            options["pool_num"], options["permutations"],
            stratify_id="all", show_progress=options["show_progress"],
            progress_interval=options["progress_interval"],
        )
        with gzip.open(self.enrichment_file, "wt") as out:
            json.dump(result, out, allow_nan=False, default=lambda item: item.item())
        stat = result["stat"][self.name]
        self.log(
            f"Enrichment finished in {(time.time() - started) / 60:.2f} min: "
            f"fold={stat['obs_value']:.4g}, empirical p={stat['pval']:.4g}"
        )
        return result

    def run(self, pruned_site_df=None, plot=False):
        """Return archived results, or reuse a completed identical run."""
        if pruned_site_df is None:
            pruned_site_df = self.pruned_site_df
        if pruned_site_df is not None and self.input_pruned_site_ids_file is not None:
            raise ValueError("Provide either pruned_site_df or pruned_site_ids_file, not both")
        if pruned_site_df is not None and (
            not isinstance(self.pruned_site_parameters, dict)
            or not self.pruned_site_parameters
        ):
            raise ValueError("pruned_site_parameters must document supplied pruned_site_df")
        self.run_hash, run_spec = self._run_hash(pruned_site_df, plot)
        self._current_run_spec = run_spec
        self.output_root.mkdir(parents=True, exist_ok=True)
        final_dir = self.output_root / f"{self.name}.{self.run_hash[:16]}"
        if final_dir.exists():
            self._set_paths(final_dir)
            if not self.manifest_file.is_file():
                raise FileExistsError(f"Incomplete run directory: {final_dir}")
            with open(self.manifest_file) as f:
                manifest = json.load(f)
            if manifest.get("run_hash") != self.run_hash or manifest.get("status") != "complete":
                raise ValueError(f"Run manifest does not match parameters: {self.manifest_file}")
            for file_name in manifest["files"]:
                if not (final_dir / file_name).is_file():
                    raise FileNotFoundError(f"Cached result is incomplete: {file_name}")
            self.log(f"Reusing completed run {self.run_hash}")
            return self._result(cached=True)

        stage_dir = Path(tempfile.mkdtemp(
            prefix=f".{self.name}.{self.run_hash[:16]}.", dir=self.output_root))
        self._set_paths(stage_dir)
        self._start_time = time.time()
        with open(self.config_file, "w") as f:
            json.dump(run_spec, f, indent=2, ensure_ascii=False)
        self.log("=" * 70)
        self.log(f"Simulation job started: {self.name}; hash={self.run_hash}")
        self.log(f"Seeds: prune={self.prune_seed}, causal={self.causal_sample_seed}, "
                 f"beta={self.beta_seed}, phenotype={self.phenotype_seed}")

        n_steps = 9
        self._step(1, n_steps, "Loading site annotation, samples and kinship")
        site_info_df, sample_id_list, kinship_df = self._load_inputs()
        self._gemma_site_info_df = site_info_df
        self._log_fg_bg(site_info_df, "All SNPs")
        self.log(f"Samples={len(sample_id_list)}, kinship shape={kinship_df.shape}")

        self._step(2, n_steps, "LD pruning")
        if pruned_site_df is None and self.input_pruned_site_ids_file is None:
            pruned_site_df = self._prune_sites(site_info_df)
            pruning_record = {
                "source": "automatic", "method": "prune_sites_by_ld",
                "ld_window_size": self.ld_window_size,
                "r2_threshold": self.r2_threshold,
                "pool_num": self.pool_num, "seed": self.prune_seed,
            }
        elif self.input_pruned_site_ids_file is not None:
            source_path = Path(self.input_pruned_site_ids_file).expanduser().resolve()
            ids = self._read_site_ids(source_path)
            pruned_site_df = self._from_site_ids(site_info_df, ids)
            pruning_record = {
                "source": "id_list", "id_list_file": str(source_path),
                "ids_sha256": self._file_sha256(source_path),
            }
            sidecar = source_path.with_name(
                source_path.name.replace(".pruned_site_ids.txt.gz", ".pruning.json"))
            if sidecar != source_path and sidecar.is_file():
                with open(sidecar) as source:
                    source_metadata = json.load(source)
                if source_metadata.get("ids_sha256") != pruning_record["ids_sha256"]:
                    raise ValueError(f"Pruning metadata does not match ID list: {sidecar}")
                pruning_record["source_pruning"] = source_metadata["pruning_record"]
            self.log(f"Reconstructed {len(ids):,} pruned SNPs from {source_path}")
        else:
            pruned_site_df = pruned_site_df.copy()
            pruning_record = {"source": "supplied", **self.pruned_site_parameters}
            self.log(f"Using supplied pruned_site_df: {len(pruned_site_df):,} SNPs")
        if self.input_pruned_site_ids_file is None and not set(
            pruned_site_df["ID"].astype(str)
        ).issubset(set(site_info_df["ID"].astype(str))):
            raise ValueError("pruned_site_df contains SNP IDs absent from site_info_df")
        self._write_site_ids(pruned_site_df, pruning_record)
        run_spec["pruning_record"] = pruning_record
        with open(self.config_file, "w") as f:
            json.dump(run_spec, f, indent=2, ensure_ascii=False)
        self._log_fg_bg(pruned_site_df, "Pruned SNPs")

        self._step(3, n_steps, "Sampling causal SNPs")
        _, causal_site_df, _ = self._sample_causal_sites(pruned_site_df)
        self._step(4, n_steps, "Preparing causal genotypes and beta")
        causal_genotype_df, causal_effect_df = self._prepare_causal_effects(causal_site_df)
        causal_genotype_df = causal_genotype_df.loc[sample_id_list]
        self._step(5, n_steps, "Simulating phenotype")
        phenotype_outputs = self._simulate_phenotype(
            causal_genotype_df, causal_effect_df, kinship_df)
        self._step(6, n_steps, "Running GEMMA GWAS")
        gemma_pheno_file = self._prepare_gemma_pheno(phenotype_outputs)
        gemma_job, gwas_df = self._run_gemma(gemma_pheno_file)
        self._step(7, n_steps, "GWAS visualization")
        if plot:
            self._plot_gwas(gemma_job)
        else:
            self.log("GWAS plotting skipped")
        self._step(8, n_steps, "Causal recovery QC")
        self._evaluate_gwas(causal_effect_df, gwas_df)
        self._step(9, n_steps, "Foreground enrichment permutations")
        self._run_enrichment(site_info_df, gwas_df)
        self._archive_and_cleanup(phenotype_outputs, gwas_df, gemma_job)
        elapsed = time.time() - self._start_time
        self.log(f"Simulation job completed: {self.name}; runtime={elapsed / 60:.2f} min")
        self.log("=" * 70)
        files = sorted(path.name for path in stage_dir.iterdir() if path.is_file())
        with open(self.manifest_file, "w") as f:
            json.dump({"status": "complete", "run_hash": self.run_hash,
                       "files": files}, f, indent=2)
        if final_dir.exists():
            raise FileExistsError(f"Another run completed this hash: {final_dir}")
        stage_dir.rename(final_dir)
        self._set_paths(final_dir)
        return self._result(cached=False)


if __name__ == "__main__":









    import os
    import pandas as pd
    import numpy as np
    from src.vcf import get_sample_list_from_vcf

    genome_fasta_file = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data/source/Sbicolor_730_v5.0.fa.gz"
    site_info_df_h5 = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data/results/site_contexts.genome.maf0.05.h5"
    input_vcf_file = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz"
    kinship_ibs_file = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data/source/landraces_snp_pruned.win15000.miss0.5.maf0.01.het0.01.rq0.5.ibs.matrix"
    site_info_df = pd.read_hdf(site_info_df_h5, key='df')
    sample_id_list = get_sample_list_from_vcf(input_vcf_file)
    kinship_df = pd.read_csv(kinship_ibs_file, sep=",",
                             header=None, names=sample_id_list)
    kinship_df.index = sample_id_list
    prune_seed = 12345 + 1
    ld_window_size = 15000
    pool_num = 40
    r2_threshold = 0.5

    foreground_col = 'genic_2kb'
    all_fg_num = site_info_df[foreground_col].sum()
    all_fg_prop = all_fg_num / len(site_info_df)
    all_bg_num = len(site_info_df) - all_fg_num
    all_bg_prop = all_bg_num / len(site_info_df)
    print(
        f"All foreground number: {all_fg_num} ({all_fg_prop:.2%}), All background number: {all_bg_num} ({all_bg_prop:.2%})")

    from src.ld import prune_sites_by_ld

    pruned_site_with_log_df = prune_sites_by_ld(
        site_info_df.sample(
            frac=1,
            random_state=prune_seed,
        ),
        input_vcf_file,
        ld_window_size=ld_window_size,
        r2_threshold=r2_threshold,
        pool_num=pool_num,
        full_output=True,
    )

    pruned_site_with_log_df.sort_index(inplace=True)
    pruned_site_df = pruned_site_with_log_df[pruned_site_with_log_df['LD_PRUNING_KEEP']].copy(
    )
    pruned_fg_num = pruned_site_df[foreground_col].sum()
    pruned_fg_prop = pruned_fg_num / len(pruned_site_df)
    pruned_bg_num = len(pruned_site_df) - pruned_fg_num
    pruned_bg_prop = pruned_bg_num / len(pruned_site_df)
    print(
        f"Pruned foreground number: {pruned_fg_num} ({pruned_fg_prop:.2%}), Pruned background number: {pruned_bg_num} ({pruned_bg_prop:.2%})")

    simulate_causal_site_num = 5
    simulate_causal_site_proportion = simulate_causal_site_num / \
        len(pruned_site_df)

    causal_sample_seed = 12345 + 2

    pruned_site_df, summary = sample_causal_snps(
        pruned_site_df,
        causal_fraction=simulate_causal_site_proportion,
        enrichment_ratio=1.0,
        maf_col="MAF",
        foreground_col=foreground_col,
        seed=causal_sample_seed,
        causal_col="causal",
    )

    causal_site_df = pruned_site_df[pruned_site_df['causal']]
    print(f"Sampled causal SNPs summary: {summary}")

    from src.vcf import get_genotype_df_from_vcf_by_site_df, get_sample_list_from_vcf

    causal_site_genotype_df = get_genotype_df_from_vcf_by_site_df(
        input_vcf_file, causal_site_df)
    causal_site_genotype_df = causal_site_genotype_df.transpose()

    beta_seed = 12345 + 3
    beta_rng = np.random.default_rng(beta_seed)

    causal_effect_df = pd.DataFrame({
        "SNP_ID": causal_site_df["ID"].astype(str).tolist(),
        "beta_raw": beta_rng.normal(
            loc=0,
            scale=1,
            size=len(causal_site_df),
        ),
    })

    output_dir = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data/results/simulate"
    output_prefix = os.path.join(output_dir, "e1")

    from src.sim.phenotype_simulator_wrap import simulate_phenotype_with_kinship

    outputs = simulate_phenotype_with_kinship(
        causal_genotype_df=causal_site_genotype_df,
        causal_effect_df=causal_effect_df,
        kinship_df=kinship_df,
        output_prefix=output_prefix,
        causal_var=0.3,
        kinship_var=0.5,
        residual_var=0.2,
        seed=12345 + 4,
    )

    phenotype_file = str(outputs["phenotype_file"])
    gemma_pheno_file = phenotype_file.replace(".tsv", ".bimbam")
    pheno_df = pd.read_csv(phenotype_file, sep="\t")
    pheno_df['phenotype'].to_csv(gemma_pheno_file, index=False, header=False)

    from src.gwas import GEMMA_JOB, plot_gwas_summary_panel

    gemma_job = GEMMA_JOB("e1", output_dir, input_vcf_file,
                          gemma_pheno_file, kinship_ibs_file)
    gemma_job.run()

    gwas_df = pd.read_hdf(gemma_job.gwas_df_h5)

    plot_gwas_summary_panel(gemma_job.gwas_df_h5, genome_fasta_file,
                            threshold_fdr=0.05,
                            bins=50,
                            title=None,
                            figsize=(14, 6),
                            save=True)

    gwas_qc_dict = evaluate_causal_gwas_recovery(
        causal_effect_df=causal_effect_df,
        gwas_df=gwas_df,
    )

    gwas_qc_dict


    from src.simulate import SIMULATION_JOB

    PROJECT = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum"

    config = {
        "name": "causal_5_snp",
        "output_dir": f"{PROJECT}/data/results/simulate",
        "genome_fasta_file": f"{PROJECT}/data/source/Sbicolor_730_v5.0.fa.gz",
        "site_info_df_h5": f"{PROJECT}/data/results/site_contexts.genome.maf0.05.h5",
        "input_vcf_file": f"{PROJECT}/data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz",
        "kinship_ibs_file": (
            f"{PROJECT}/data/source/"
            "landraces_snp_pruned.win15000.miss0.5.maf0.01.het0.01.rq0.5.ibs.matrix"
        ),
        "foreground_col": "genic_2kb",
        "causal_site_num": 5,
        "enrichment_ratio": 1.0,
        "ld_window_size": 15000,
        "r2_threshold": 0.5,
        "pool_num": 40,
        "effect_assignment": {"method": "normal", "beta_loc": 0.0,
                              "scale": 1.0},
        "causal_var": 0.3,
        "kinship_var": 0.5,
        "residual_var": 0.2,
        "seed": 42,
    }

    result = SIMULATION_JOB(config).run(plot=True)
    print(f"cached={result['cached']}")
    print(f"output_dir={result['output_dir']}")
    print(f"run_hash={result['run_hash']}")
