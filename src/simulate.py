import gc
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from numba import njit, prange


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