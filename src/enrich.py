import ast
import pandas as pd
import multiprocessing as mp
from time import perf_counter
import numpy as np
from .utils import have_file
import os
import gzip
import json
from matplotlib import pyplot as plt
from scipy.stats import zscore
from .plot import set_plot_style
import re


def filter_df_by_set_expr(df, expr):
    """
    根据布尔列的集合运算表达式筛选 DataFrame。

    支持：
        A | B   并集
        A & B   交集
        A - B   差集
        (...)   括号

    普通列名可直接使用：
        "(UMR | ACR) & (genic_proximal - TE)"

    如果列名包含 "."、"-"、空格等特殊字符，
    用反引号包起来，例如：
        "(`missense_cds_0.01` | ACR) & (genic_proximal - TE)"
    """

    # 将反引号中的列名替换为合法的 Python 临时变量名
    safe_name_to_col = {}

    def replace_backtick(match):
        col = match.group(1)

        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

        safe_name = f"__col_{len(safe_name_to_col)}__"
        safe_name_to_col[safe_name] = col

        return safe_name

    expr_safe = re.sub(
        r"`([^`]+)`",
        replace_backtick,
        expr
    )

    tree = ast.parse(expr_safe, mode="eval")

    def evaluate(node):

        # 列名
        if isinstance(node, ast.Name):
            col = node.id

            # 如果是反引号列名替换出来的临时变量，
            # 转回真实 DataFrame 列名
            if col in safe_name_to_col:
                col = safe_name_to_col[col]

            if col not in df.columns:
                raise ValueError(f"Column not found: {col}")

            s = df[col]

            # 确保这一列是布尔值
            valid_values = set(s.dropna().unique())
            if not valid_values <= {True, False}:
                raise ValueError(
                    f"Column '{col}' contains non-boolean values: "
                    f"{valid_values}"
                )

            return s.fillna(False).astype(bool)

        # A | B / A & B / A - B
        if isinstance(node, ast.BinOp):

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.BitOr):
                return left | right

            elif isinstance(node.op, ast.BitAnd):
                return left & right

            elif isinstance(node.op, ast.Sub):
                return left & ~right

            else:
                raise ValueError(
                    f"Unsupported operator: {type(node.op).__name__}"
                )

        raise ValueError(
            f"Unsupported expression: {ast.dump(node)}"
        )

    mask = evaluate(tree.body)

    return mask


def permute_array(feature_array, seed=None):
    np.random.seed(seed)
    random_number = np.random.randint(0, len(feature_array))
    permuted_feature_array = np.roll(feature_array, random_number)
    return permuted_feature_array

# fold enrichment


def get_fold_enrichment(data_df, foreground_mask, data_col, top_percent):
    pvalues = data_df[data_col]
    fore_pvalues = pvalues[foreground_mask]
    back_pvalues = pvalues[~foreground_mask]
    fore_pvalues = fore_pvalues[~np.isnan(fore_pvalues)]
    back_pvalues = back_pvalues[~np.isnan(back_pvalues)]

    fore_top_pvalue = np.quantile(fore_pvalues, 1 - top_percent)
    back_top_pvalue = np.quantile(back_pvalues, 1 - top_percent)

    return fore_top_pvalue/back_top_pvalue


def get_permutation_fold_enrichment(perm_id, run_context_arrays, env_id_list, top_percent):
    permuted_foreground = permute_array(
        run_context_arrays['foreground'], seed=perm_id)
    return {
        env_id: get_fold_enrichment(
            run_context_arrays, permuted_foreground, env_id, top_percent
        )
        for env_id in env_id_list
    }


_permutation_worker_state = None


def _init_permutation_worker(run_context_arrays, env_id_list, top_percent):
    # With fork, these arrays are inherited once and used read-only by each worker.
    global _permutation_worker_state
    _permutation_worker_state = (run_context_arrays, env_id_list, top_percent)


def _run_permutation_worker(perm_id):
    # A top-level callable is pickleable; each queued task contains only an integer.
    return get_permutation_fold_enrichment(perm_id, *_permutation_worker_state)


def _collect_permutation_results(result_iter, total, start_time, show_progress, progress_interval):
    """Collect ordered results, refreshing progress only in the parent process."""
    if not show_progress:
        return dict(enumerate(result_iter))

    last_width = 0

    def report(completed, now):
        nonlocal last_width
        elapsed = max(now - start_time, 0.0)
        rate = completed / elapsed if elapsed else 0.0
        percent = 100.0 * completed / total if total else 100.0
        if completed == total:
            eta = "0.0s"
        else:
            eta = f"{(total - completed) / rate:.1f}s" if rate else "--"
        status = (
            f"Permutations: {completed}/{total} ({percent:5.1f}%) | "
            f"elapsed {elapsed:.1f}s | {rate:.2f}/s | ETA {eta}"
        )
        print("\r" + status.ljust(last_width), end="", flush=True)
        last_width = len(status)

    collected = {}
    last_update = perf_counter()
    report(0, last_update)
    try:
        for perm_id, result in enumerate(result_iter):
            collected[perm_id] = result
            completed = perm_id + 1
            now = perf_counter()
            if completed == total or now - last_update >= progress_interval:
                report(completed, now)
                last_update = now
    finally:
        # End the progress line even if a worker fails or the user interrupts.
        print(flush=True)
    return collected


def enrichment_permutation_analysis(
    input_data_df, top_percent, env_id_list, pool_num, permutations,
    stratify_id='all', show_progress=True, progress_interval=2.0,
):
    """
    Perform enrichment analysis with permutations.
    input_data_df: DataFrame with columns 'foreground' and env_id_list
    top_percent: float, the top percentage to consider for enrichment
    env_id_list: list of environment IDs (column names in input_data_df)
    pool_num: int, number of parallel processes to use
    permutations: int, number of permutations to perform
    stratify_id: str, column name to stratify the data (default 'all')
    show_progress: bool, whether to show progress (default True)
    progress_interval: float, seconds between progress updates (default 2.0)
    """
    if show_progress and (not np.isfinite(progress_interval) or progress_interval <= 0):
        raise ValueError(
            "progress_interval must be a finite positive number of seconds")

    # Fork workers inherit the arrays; no large arrays are sent through the task queue.
    if pool_num > 1 and mp.get_start_method() != 'fork':
        raise RuntimeError(
            "This notebook parallel path requires the 'fork' start method. "
            "Use pool_num = 1 for serial execution in this kernel."
        )

    if stratify_id == 'all':
        run_context_df = input_data_df.copy()
    else:
        run_context_df = input_data_df[input_data_df[stratify_id]].copy()

    # Extract once and reuse the arrays in observed and permutation calculations.
    run_context_arrays = {
        col: run_context_df[col].to_numpy()
        for col in ['foreground'] + env_id_list
    }

    obs_values = {}
    for env_id in env_id_list:
        obs_values[env_id] = get_fold_enrichment(
            run_context_arrays, run_context_arrays['foreground'], env_id, top_percent)

    permutation_start = perf_counter()
    if pool_num == 1:
        result_iter = (
            get_permutation_fold_enrichment(
                perm_id, run_context_arrays, env_id_list, top_percent
            )
            for perm_id in range(permutations)
        )
        mlt_dict = _collect_permutation_results(
            result_iter, permutations, permutation_start, show_progress, progress_interval
        )
    else:
        with mp.Pool(
            processes=pool_num,
            initializer=_init_permutation_worker,
            initargs=(run_context_arrays, tuple(env_id_list), top_percent),
        ) as pool:
            result_iter = pool.imap(
                _run_permutation_worker, range(permutations), chunksize=1
            )
            mlt_dict = _collect_permutation_results(
                result_iter, permutations, permutation_start, show_progress, progress_interval
            )

    # Preserve the serial output structure and permutation-ID order.
    perm_values = {
        env_id: [mlt_dict[perm_id][env_id] for perm_id in range(permutations)]
        for env_id in env_id_list
    }
    permutation_seconds = perf_counter() - permutation_start
    print(
        f"{permutations} permutations with pool_num={pool_num}: "
        f"{permutation_seconds:.2f} s "
        f"({permutations / permutation_seconds:.2f} permutations/s)"
    )

    stat_dict = {}
    for env_id in env_id_list:
        perm_array = np.array(perm_values[env_id])
        obs_value = obs_values[env_id]
        if perm_array.size == 0 or not np.isfinite(obs_value) or not np.all(np.isfinite(perm_array)):
            raise ValueError(
                f"Cannot calculate empirical P for {env_id} ({stratify_id}): "
                "the observed statistic and a non-empty permutation distribution must be finite."
            )
        # Two-sided empirical P: double the smaller inclusive tail, with +1 correction.
        n_perm = perm_array.size
        p_lower = (np.sum(perm_array <= obs_value) + 1) / (n_perm + 1)
        p_upper = (np.sum(perm_array >= obs_value) + 1) / (n_perm + 1)
        pval = float(min(1.0, 2 * min(p_lower, p_upper)))
        stat_dict[env_id] = {
            "obs_value": obs_value,
            "pval": pval,
            "mean_perm": np.mean(perm_array),
            "std_perm": np.std(perm_array),
        }

    results_json = {
        "obs_values": obs_values,
        "perm_values": perm_values,
        "stat": stat_dict,
        "permutations": permutations,
        "stratify_id": stratify_id,
        "top_percent": top_percent,
        "env_id_list": env_id_list,
    }

    return results_json


def contrast_runner(contrast_id, contrast_json, context_df, env_id_list, output_dir, permutations=1000, top_percent=0.05, pool_num=40):

    results_json_file = os.path.join(
        output_dir, f"{contrast_id}.enrichment.json.gz")
    if not have_file(results_json_file):
        enrich_data_df = context_df.copy()
        foreground_mask = filter_df_by_set_expr(
            enrich_data_df, contrast_json["foreground"])
        background_mask = filter_df_by_set_expr(
            enrich_data_df, contrast_json["background"])
        enrich_data_df['foreground'] = foreground_mask
        enrich_data_df['background'] = background_mask
        enrich_data_df = enrich_data_df.loc[foreground_mask | background_mask].copy(
        )
        stratify_cols = [i for i in contrast_json.get(
            "maf_stratify", []) if i != 'all']
        enrich_data_df = enrich_data_df[[
            'ID', 'foreground'] + stratify_cols + env_id_list].copy()

        enrich_results_json = {}
        for stratify_id in ['all'] + stratify_cols:
            results_json = enrichment_permutation_analysis(
                enrich_data_df, top_percent, env_id_list, pool_num, permutations, stratify_id
            )
            enrich_results_json[stratify_id] = results_json

        json_file = os.path.join(
            output_dir, f"{contrast_id}.enrichment.json.gz")
        with gzip.open(json_file, 'wt', encoding='utf-8') as f:
            json.dump(enrich_results_json, f, indent=4)
    else:
        print(
            f"Enrichment results already exist for contrast {contrast_id}, skipping.")

    return results_json_file


def wrap_label(label, width=20):
    import textwrap
    return "\n".join(textwrap.wrap(label, width))


def plot_enrichment_results(enrich_results_json, env_id_list, env_metadata_df, stratify_label_map, significance_threshold=0.05):

    set_plot_style()
    
    stratify_list = [s[0] for s in stratify_label_map]

    # stratify_list.reverse()

    fig, axs = plt.subplots(figsize=(10, 3), nrows=1,
                            ncols=len(env_id_list), sharey=True)

    y_values = {s: i for i, s in enumerate(stratify_list)}

    for ax, env_id in zip(axs, env_id_list):
        for s in stratify_list:
            obs_value = enrich_results_json[s]['obs_values'][env_id]
            perm_values = enrich_results_json[s]['perm_values'][env_id]
            obs_z = zscore([obs_value] + perm_values)[0]
            perm_z = zscore([obs_value] + perm_values)[1:]

            if s == 'all':
                ax.scatter(perm_z, [y_values[s]] * len(perm_values), color='#B3AF57',
                           s=40, alpha=0.2, marker='o', facecolors='none')  # 灰色点
            else:
                ax.scatter(perm_z, [y_values[s]] * len(perm_values), color='gray',
                           s=40, alpha=0.2, marker='o', facecolors='none')  # 灰色点
            ax.scatter(obs_z, y_values[s],
                       color='#5580B0', s=120, alpha=1)  # 蓝色点

            # 绘制显著性标记
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            p_value = enrich_results_json[s]['stat'][env_id]['pval']
            mean_perm = enrich_results_json[s]['stat'][env_id]['mean_perm']

            if p_value < significance_threshold:
                if obs_value > mean_perm:
                    ax.plot(obs_z+(current_xlim[1] - current_xlim[0])*0.15, y_values[s], '*',
                            markersize=14, color='black')
                elif obs_value < mean_perm:
                    ax.plot(obs_z-(current_xlim[1] - current_xlim[0])*0.15, y_values[s], '*',
                            markersize=14, color='black')

        stratify_label_map_dict = {s[0]: s[1] for s in stratify_label_map}

        ax.set_yticks(list(y_values.values()))
        ax.set_yticklabels(
            [stratify_label_map_dict.get(s, s) for s in y_values], fontsize=14
        )
        # ax.set_xlabel('Fold Enrichment (Z-score)', fontsize=14)
        ax.axvline(x=0, color='gray', linestyle='dashed', linewidth=0.7)

        # 设置x轴粗细
        # ax.spines['bottom'].set_linewidth(0.7)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.grid(axis='x', linestyle='dashed', alpha=0.5)
        ax.tick_params(axis='y', which='both', left=False)

        ax.set_ylim(-0.5, len(stratify_list) - 0.2)

    # Attach each title to its axes so it follows tight_layout and stays centered.
    for ax, env_id in zip(axs, env_id_list):
        env_label = env_metadata_df.loc[
            env_metadata_df['env_id'] == env_id, 'description'
        ].values[0]
        ax.set_title(
            wrap_label(env_label, width=20),
            loc='center', x=0.5, pad=6, fontsize=14,
            ha='center', multialignment='center',
        )

    # ax.yaxis_inverted()

    plt.tight_layout()
    plt.show()
