import ast
import pandas as pd
import multiprocessing as mp
from time import perf_counter
import numpy as np
import os
import gzip
import json
from matplotlib import pyplot as plt
from scipy.stats import zscore
import re
from scipy.optimize import linear_sum_assignment
import matplotlib.colors as mcolors
from .utils import have_file
from .plot import set_plot_style


CHROMOSOME_PERMUTATION_SCHEME = 'within_chromosome_roll_v1'
GENE_PERMUTATION_SCHEME = 'gene_complete_case_within_chromosome_roll_v1'
WINDOW_PERMUTATION_SCHEME = 'window_boolean_complete_case_within_chromosome_roll_v1'
SITE_JACCARD_PERMUTATION_SCHEME = 'site_boolean_complete_case_within_chromosome_roll_v1'


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


def chromosome_slices(chromosomes):
    """Return contiguous chromosome slices for rows sorted by CHROM and POS."""
    chromosomes = np.asarray(chromosomes)
    boundaries = np.r_[0, np.flatnonzero(chromosomes[1:] != chromosomes[:-1]) + 1,
                       len(chromosomes)]
    return tuple(slice(int(start), int(end))
                 for start, end in zip(boundaries[:-1], boundaries[1:]))


def permute_array_within_chromosomes(feature_array, chrom_slices, seed=None):
    """Independently circular-shift each chromosome without moving sites across it."""
    rng = np.random.default_rng(seed)
    permuted = np.empty_like(feature_array)
    for chrom_slice in chrom_slices:
        segment = feature_array[chrom_slice]
        permuted[chrom_slice] = np.roll(segment, rng.integers(len(segment)))
    return permuted

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


def get_permutation_fold_enrichment(perm_id, run_context_arrays, env_id_list, top_percent,
                                    chrom_slices):
    permuted_foreground = permute_array_within_chromosomes(
        run_context_arrays['foreground'], chrom_slices, seed=perm_id)
    return {
        env_id: get_fold_enrichment(
            run_context_arrays, permuted_foreground, env_id, top_percent
        )
        for env_id in env_id_list
    }


_permutation_worker_state = None


def _init_permutation_worker(run_context_arrays, env_id_list, top_percent, chrom_slices):
    # With fork, these arrays are inherited once and used read-only by each worker.
    global _permutation_worker_state
    _permutation_worker_state = (run_context_arrays, env_id_list, top_percent, chrom_slices)


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

    input_data_df = input_data_df.sort_values(
        by=['CHROM', 'POS'], ascending=[True, True]).copy()

    if stratify_id == 'all':
        run_context_df = input_data_df.copy()
    else:
        run_context_df = input_data_df[input_data_df[stratify_id]].copy()

    # Extract once and reuse the arrays in observed and permutation calculations.
    run_context_arrays = {
        col: run_context_df[col].to_numpy()
        for col in ['foreground'] + env_id_list
    }
    chrom_slices = chromosome_slices(run_context_df['CHROM'].to_numpy())

    obs_values = {}
    for env_id in env_id_list:
        obs_values[env_id] = get_fold_enrichment(
            run_context_arrays, run_context_arrays['foreground'], env_id, top_percent)

    permutation_start = perf_counter()
    if pool_num == 1:
        result_iter = (
            get_permutation_fold_enrichment(
                perm_id, run_context_arrays, env_id_list, top_percent, chrom_slices
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
            initargs=(run_context_arrays, tuple(env_id_list), top_percent, chrom_slices),
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
        "permutation_scheme": CHROMOSOME_PERMUTATION_SCHEME,
    }

    return results_json


def two_sided_empirical_pvalue(obs_value, perm_array):
    # Two-sided empirical P: double the smaller inclusive tail, with +1 correction.
    perm_array = np.array(perm_array)
    n_perm = perm_array.size
    p_lower = (np.sum(perm_array <= obs_value) + 1) / (n_perm + 1)
    p_upper = (np.sum(perm_array >= obs_value) + 1) / (n_perm + 1)
    mean_perm = np.mean(perm_array)
    std_perm = np.std(perm_array)
    pval = float(min(1.0, 2 * min(p_lower, p_upper)))

    results_json = {
        "obs_value": obs_value,
        "pval": pval,
        "mean_perm": mean_perm,
        "std_perm": std_perm,
    }

    return results_json


def build_enrich_data_df(context_df, contrast_json, env_id_list, other_columns=[]):
    enrich_data_df = context_df.copy()
    foreground_mask = filter_df_by_set_expr(
        enrich_data_df, contrast_json["foreground"])
    background_mask = filter_df_by_set_expr(
        enrich_data_df, contrast_json["background"])
    enrich_data_df['foreground'] = foreground_mask
    enrich_data_df['background'] = background_mask
    enrich_data_df = enrich_data_df.loc[foreground_mask | background_mask].copy(
    )

    if 'POS' in enrich_data_df.columns:
        enrich_data_df = enrich_data_df[[
            'ID', 'CHROM', 'POS', 'foreground'] + env_id_list + other_columns].copy()
    elif 'START' in enrich_data_df.columns and 'END' in enrich_data_df.columns:
        enrich_data_df = enrich_data_df[[
            'ID', 'CHROM', 'START', 'END', 'foreground'] + env_id_list + other_columns].copy()

    return enrich_data_df


def site_contrast_runner(contrast_id, contrast_json, context_df, env_id_list, output_dir, permutations=1000, top_percent=0.05, pool_num=40):

    results_json_file = os.path.join(
        output_dir, f"{contrast_id}.enrichment.json.gz")
    reuse_result = have_file(results_json_file)
    if reuse_result:
        try:
            with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
                existing_results = json.load(f)
            reuse_result = (isinstance(existing_results, dict) and
                            all(isinstance(result, dict) and
                                result.get('permutation_scheme') == CHROMOSOME_PERMUTATION_SCHEME
                                for result in existing_results.values()))
        except (OSError, EOFError, UnicodeError, json.JSONDecodeError):
            reuse_result = False
    if not reuse_result:
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
            'ID', 'CHROM', 'POS', 'foreground'] + stratify_cols + env_id_list].copy()

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

# jaccard


def get_jaccard(set1_mask, set2_mask):
    intersection = np.sum(set1_mask & set2_mask)
    union = np.sum(set1_mask | set2_mask)
    if union == 0:
        return 0.0
    return intersection / union


def gene_contrast_runner(contrast_id, contrast_json, context_df, env_id_list, output_dir, permutations=1000, top_percent=0.05):
    """Test gene-set overlap using chromosome-local shifts on per-environment backgrounds.

    Each background excludes missing scores, target labels, and coordinates, and
    is sorted by CHROM/START. All filtering and sorting use private copies.
    """
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be in (0, 1]")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    env_id_list = list(env_id_list)
    target = contrast_json["target"]
    results_json_file = os.path.join(
        output_dir, f"{contrast_id}.enrichment.json.gz")

    # Old global-roll results must not be reused under the new null model.
    if have_file(results_json_file):
        try:
            with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
                existing = json.load(f)['all']
            if (existing.get('permutation_scheme') == GENE_PERMUTATION_SCHEME
                    and existing.get('permutations') == permutations
                    and existing.get('top_percent') == top_percent
                    and existing.get('env_id_list') == env_id_list
                    and existing.get('target') == target):
                print(f"Enrichment results already exist for contrast {contrast_id}, skipping.")
                return results_json_file
        except (OSError, EOFError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
            pass

    results_json = {
        'obs_values': {}, 'perm_values': {}, 'stat': {}, 'background_info': {},
        'permutation_scheme': GENE_PERMUTATION_SCHEME,
        'permutations': permutations, 'top_percent': top_percent,
        'env_id_list': env_id_list, 'target': target,
    }
    for env_id in env_id_list:
        columns = list(dict.fromkeys(['CHROM', 'START', target, env_id]))
        run_context_df = context_df[columns].dropna(subset=columns).copy()
        run_context_df = run_context_df.sort_values(
            ['CHROM', 'START'], kind='stable').reset_index(drop=True)
        if run_context_df.empty:
            raise ValueError(f"No complete genes for {env_id} ({contrast_id})")
        if not set(run_context_df[target].unique()) <= {True, False}:
            raise ValueError(f"Target column {target!r} must contain boolean labels")
        scores = run_context_df[env_id].to_numpy(dtype=float)
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"Non-finite gene scores for {env_id} ({contrast_id})")
        target_mask = run_context_df[target].to_numpy(dtype=bool)
        threshold = np.quantile(scores, 1 - top_percent)
        env_top_mask = scores >= threshold
        chrom_slices = chromosome_slices(run_context_df['CHROM'].to_numpy())
        obs_value = get_jaccard(target_mask, env_top_mask)
        perm_values = [
            get_jaccard(permute_array_within_chromosomes(
                target_mask, chrom_slices, seed=perm_id), env_top_mask)
            for perm_id in range(permutations)
        ]
        results_json['obs_values'][env_id] = obs_value
        results_json['perm_values'][env_id] = perm_values
        results_json['stat'][env_id] = two_sided_empirical_pvalue(obs_value, perm_values)
        results_json['background_info'][env_id] = {
            'n_input': len(context_df), 'n_genes': len(run_context_df),
            'n_excluded': len(context_df) - len(run_context_df),
            'n_target': int(target_mask.sum()), 'n_top': int(env_top_mask.sum()),
            'top_score_threshold': float(threshold),
            'chromosome_counts': {
                str(chrom): int(count) for chrom, count in
                run_context_df.groupby('CHROM', observed=True).size().items()
            },
        }
        print(f"{contrast_id} / {env_id}: {len(run_context_df)}/{len(context_df)} genes "
              f"retained; {int(target_mask.sum())} target, {int(env_top_mask.sum())} top genes")

    with gzip.open(results_json_file, 'wt', encoding='utf-8') as f:
        json.dump({'all': results_json}, f, indent=4)
    return results_json_file


def win_contrast_runner(contrast_id, contrast_json, context_df, trait_id_list, output_dir, permutations=1000):
    """Test overlap of a boolean window target with boolean window traits.

    For each trait, exclude missing CHROM/START/END, target, or trait values,
    then sort a private copy by CHROM/START/END. Keep the trait fixed and
    independently circular-shift target labels within each chromosome.
    No score threshold is applied; both sets use their True-valued windows.
    Return the gzip JSON path, with results under 'all' as in gene_contrast_runner.
    """
    if permutations < 1:
        raise ValueError("permutations must be positive")
    trait_id_list = list(trait_id_list)
    target = contrast_json["target"]
    results_json_file = os.path.join(
        output_dir, f"{contrast_id}.enrichment.json.gz")

    if have_file(results_json_file):
        try:
            with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
                existing = json.load(f)['all']
            if (existing.get('permutation_scheme') == WINDOW_PERMUTATION_SCHEME
                    and existing.get('permutations') == permutations
                    and existing.get('trait_id_list') == trait_id_list
                    and existing.get('target') == target):
                print(f"Enrichment results already exist for contrast {contrast_id}, skipping.")
                return results_json_file
        except (OSError, EOFError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
            pass

    results_json = {
        'obs_values': {}, 'perm_values': {}, 'stat': {}, 'background_info': {},
        'permutation_scheme': WINDOW_PERMUTATION_SCHEME,
        'permutations': permutations, 'trait_id_list': trait_id_list,
        'target': target,
    }
    for trait_id in trait_id_list:
        columns = list(dict.fromkeys(['CHROM', 'START', 'END', target, trait_id]))
        run_context_df = context_df[columns].dropna(subset=columns).copy()
        run_context_df = run_context_df.sort_values(
            ['CHROM', 'START', 'END'], kind='stable').reset_index(drop=True)
        if run_context_df.empty:
            raise ValueError(f"No complete windows for {trait_id} ({contrast_id})")
        for label in dict.fromkeys([target, trait_id]):
            if not set(run_context_df[label].unique()) <= {True, False}:
                raise ValueError(f"Window column {label!r} must contain boolean labels")
        target_mask = run_context_df[target].to_numpy(dtype=bool)
        trait_mask = run_context_df[trait_id].to_numpy(dtype=bool)
        chrom_slices = chromosome_slices(run_context_df['CHROM'].to_numpy())
        obs_value = get_jaccard(target_mask, trait_mask)
        perm_values = [
            get_jaccard(permute_array_within_chromosomes(
                target_mask, chrom_slices, seed=perm_id), trait_mask)
            for perm_id in range(permutations)
        ]
        results_json['obs_values'][trait_id] = obs_value
        results_json['perm_values'][trait_id] = perm_values
        results_json['stat'][trait_id] = two_sided_empirical_pvalue(obs_value, perm_values)
        results_json['background_info'][trait_id] = {
            'n_input': len(context_df), 'n_windows': len(run_context_df),
            'n_excluded': len(context_df) - len(run_context_df),
            'n_target': int(target_mask.sum()), 'n_trait': int(trait_mask.sum()),
            'chromosome_counts': {
                str(chrom): int(count) for chrom, count in
                run_context_df.groupby('CHROM', observed=True).size().items()
            },
        }
        print(f"{contrast_id} / {trait_id}: {len(run_context_df)}/{len(context_df)} windows "
              f"retained; {int(target_mask.sum())} target, {int(trait_mask.sum())} trait windows")

    with gzip.open(results_json_file, 'wt', encoding='utf-8') as f:
        json.dump({'all': results_json}, f, indent=4)
    return results_json_file


def site_jaccard_contrast_runner(contrast_id, contrast_json, context_df, trait_id_list, output_dir, permutations=1000):
    """Test overlap of a boolean site target with boolean site traits.

    For each trait, exclude missing CHROM/POS, target, or trait values,
    then sort a private copy by CHROM/POS. Keep the trait fixed and
    independently circular-shift target labels within each chromosome.
    No score threshold is applied; both sets use their True-valued sites.
    Return the gzip JSON path, with results under 'all' as in gene_contrast_runner.
    """
    if permutations < 1:
        raise ValueError("permutations must be positive")
    trait_id_list = list(trait_id_list)
    target = contrast_json["target"]
    results_json_file = os.path.join(
        output_dir, f"{contrast_id}.enrichment.json.gz")

    if have_file(results_json_file):
        try:
            with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
                existing = json.load(f)['all']
            if (existing.get('permutation_scheme') == SITE_JACCARD_PERMUTATION_SCHEME
                    and existing.get('permutations') == permutations
                    and existing.get('trait_id_list') == trait_id_list
                    and existing.get('target') == target):
                print(f"Enrichment results already exist for contrast {contrast_id}, skipping.")
                return results_json_file
        except (OSError, EOFError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
            pass

    results_json = {
        'obs_values': {}, 'perm_values': {}, 'stat': {}, 'background_info': {},
        'permutation_scheme': SITE_JACCARD_PERMUTATION_SCHEME,
        'permutations': permutations, 'trait_id_list': trait_id_list,
        'target': target,
    }
    for trait_id in trait_id_list:
        columns = list(dict.fromkeys(['CHROM', 'POS', target, trait_id]))
        run_context_df = context_df[columns].dropna(subset=columns).copy()
        run_context_df = run_context_df.sort_values(
            ['CHROM', 'POS'], kind='stable').reset_index(drop=True)
        if run_context_df.empty:
            raise ValueError(f"No complete sites for {trait_id} ({contrast_id})")
        for label in dict.fromkeys([target, trait_id]):
            if not set(run_context_df[label].unique()) <= {True, False}:
                raise ValueError(f"Site column {label!r} must contain boolean labels")
        target_mask = run_context_df[target].to_numpy(dtype=bool)
        trait_mask = run_context_df[trait_id].to_numpy(dtype=bool)
        chrom_slices = chromosome_slices(run_context_df['CHROM'].to_numpy())
        obs_value = get_jaccard(target_mask, trait_mask)
        perm_values = [
            get_jaccard(permute_array_within_chromosomes(
                target_mask, chrom_slices, seed=perm_id), trait_mask)
            for perm_id in range(permutations)
        ]
        results_json['obs_values'][trait_id] = obs_value
        results_json['perm_values'][trait_id] = perm_values
        results_json['stat'][trait_id] = two_sided_empirical_pvalue(obs_value, perm_values)
        results_json['background_info'][trait_id] = {
            'n_input': len(context_df), 'n_sites': len(run_context_df),
            'n_excluded': len(context_df) - len(run_context_df),
            'n_target': int(target_mask.sum()), 'n_trait': int(trait_mask.sum()),
            'chromosome_counts': {
                str(chrom): int(count) for chrom, count in
                run_context_df.groupby('CHROM', observed=True).size().items()
            },
        }
        print(f"{contrast_id} / {trait_id}: {len(run_context_df)}/{len(context_df)} sites "
              f"retained; {int(target_mask.sum())} target, {int(trait_mask.sum())} trait sites")

    with gzip.open(results_json_file, 'wt', encoding='utf-8') as f:
        json.dump({'all': results_json}, f, indent=4)
    return results_json_file


def wrap_label(label, width=20):
    import textwrap
    return "\n".join(textwrap.wrap(label, width))


def plot_enrichment_with_stratify(
    enrich_results_json,
    env_id_list,
    env_metadata_df,
    stratify_label_map,
    significance_threshold=0.05,
    figsize=(10, 3),
    perm_color='gray',
    perm_all_color='#B3AF57',
    perm_size=40,
    perm_alpha=0.2,
    perm_marker='o',
    perm_facecolors='none',
    obs_color='#5580B0',
    obs_size=120,
    obs_alpha=1.0,
    obs_marker='o',
    star_marker='*',
    star_color='black',
    star_size=14,
    star_offset_fraction=0.15,
    zero_line=True,
    zero_line_color='gray',
    zero_line_style='dashed',
    zero_line_width=0.7,
    grid=True,
    grid_color=None,
    grid_style='dashed',
    grid_alpha=0.5,
    grid_linewidth=None,
    xlim=None,
    xlabel=None,
    xlabel_fontsize=14,
    tick_fontsize=None,
    stratify_tick_fontsize=14,
    ylim_pad=(0.5, 0.2),
    title_fontsize=14,
    title_pad=6,
    title_wrap_width=20,
    remove_left_spine=True,
    remove_right_spine=True,
    remove_top_spine=True,
    tick_direction=None,
    tight_layout=True,
    show=True,
    return_fig_ax=True,
):
    """Plot enrichment results across stratification groups.

    The original data arguments and their order are preserved. Plot styling
    is configurable through optional arguments appended to the signature.
    When ``return_fig_ax`` is true, return ``(fig, axes)``.
    """

    set_plot_style()

    stratify_list = [s[0] for s in stratify_label_map]

    # stratify_list.reverse()

    fig, axs = plt.subplots(figsize=figsize, nrows=1,
                            ncols=len(env_id_list), sharey=True)
    axs = np.atleast_1d(axs)

    y_values = {s: i for i, s in enumerate(stratify_list)}

    for ax, env_id in zip(axs, env_id_list):
        for s in stratify_list:
            obs_value = enrich_results_json[s]['obs_values'][env_id]
            perm_values = enrich_results_json[s]['perm_values'][env_id]
            obs_z = zscore([obs_value] + perm_values)[0]
            perm_z = zscore([obs_value] + perm_values)[1:]

            if s == 'all':
                current_perm_color = perm_all_color
            else:
                current_perm_color = perm_color
            ax.scatter(perm_z, [y_values[s]] * len(perm_values),
                       color=current_perm_color, s=perm_size,
                       alpha=perm_alpha, marker=perm_marker,
                       facecolors=perm_facecolors)
            ax.scatter(obs_z, y_values[s],
                       color=obs_color, s=obs_size, alpha=obs_alpha,
                       marker=obs_marker)

            # 绘制显著性标记
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            p_value = enrich_results_json[s]['stat'][env_id]['pval']
            mean_perm = enrich_results_json[s]['stat'][env_id]['mean_perm']

            if p_value < significance_threshold:
                if obs_value > mean_perm:
                    ax.plot(obs_z+(current_xlim[1] - current_xlim[0])*star_offset_fraction,
                            y_values[s], marker=star_marker, markersize=star_size,
                            color=star_color, linestyle='none')
                elif obs_value < mean_perm:
                    ax.plot(obs_z-(current_xlim[1] - current_xlim[0])*star_offset_fraction,
                            y_values[s], marker=star_marker, markersize=star_size,
                            color=star_color, linestyle='none')

        stratify_label_map_dict = {s[0]: s[1] for s in stratify_label_map}

        ax.set_yticks(list(y_values.values()))
        ax.set_yticklabels(
            [stratify_label_map_dict.get(s, s) for s in y_values],
            fontsize=stratify_tick_fontsize,
        )
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        if xlim is not None:
            ax.set_xlim(xlim)
        if zero_line:
            ax.axvline(x=0, color=zero_line_color,
                       linestyle=zero_line_style, linewidth=zero_line_width)

        # 设置x轴粗细
        # ax.spines['bottom'].set_linewidth(0.7)

        if remove_right_spine:
            ax.spines['right'].set_visible(False)
        if remove_top_spine:
            ax.spines['top'].set_visible(False)
        if remove_left_spine:
            ax.spines['left'].set_visible(False)

        if grid:
            grid_kwargs = {
                'linestyle': grid_style,
                'alpha': grid_alpha,
            }
            if grid_color is not None:
                grid_kwargs['color'] = grid_color
            if grid_linewidth is not None:
                grid_kwargs['linewidth'] = grid_linewidth
            ax.grid(axis='x', **grid_kwargs)
        ax.tick_params(axis='y', which='both', left=False)
        x_tick_params = {}
        if tick_fontsize is not None:
            x_tick_params['labelsize'] = tick_fontsize
        if tick_direction is not None:
            x_tick_params['direction'] = tick_direction
        if x_tick_params:
            ax.tick_params(axis='x', **x_tick_params)

        ax.set_ylim(-ylim_pad[0], len(stratify_list) - ylim_pad[1])

    # Attach each title to its axes so it follows tight_layout and stays centered.
    for ax, env_id in zip(axs, env_id_list):
        env_label = env_metadata_df.loc[
            env_metadata_df['env_id'] == env_id, 'description'
        ].values[0]
        ax.set_title(
            wrap_label(env_label, width=title_wrap_width),
            loc='center', x=0.5, pad=title_pad, fontsize=title_fontsize,
            ha='center', multialignment='center',
        )

    # ax.yaxis_inverted()

    if tight_layout:
        fig.tight_layout()
    if show:
        plt.show()
    if return_fig_ax:
        return fig, axs


def plot_set_enrichment(
    set_results_json,
    env_id_list,
    env_metadata_df,
    set_label_map,
    significance_threshold=0.05,
    figsize=None,
    height_per_set=0.5,
    perm_color='gray',
    perm_size=40,
    perm_alpha=0.2,
    perm_marker='o',
    perm_facecolors='none',
    obs_color='#5580B0',
    obs_size=120,
    obs_alpha=1.0,
    obs_marker='o',
    star_marker='*',
    star_color='black',
    star_size=14,
    star_offset_fraction=0.15,
    zero_line=True,
    zero_line_color='gray',
    zero_line_style='dashed',
    zero_line_width=0.7,
    grid=True,
    grid_color=None,
    grid_style='dashed',
    grid_alpha=0.5,
    grid_linewidth=None,
    xlim=None,
    xlabel=None,
    xlabel_fontsize=14,
    tick_fontsize=None,
    set_tick_fontsize=14,
    ylim_pad=(0.5, 0.2),
    title_fontsize=14,
    title_pad=6,
    title_wrap_width=20,
    remove_left_spine=True,
    remove_right_spine=True,
    remove_top_spine=True,
    tick_direction=None,
    tight_layout=True,
    show=True,
    return_fig_ax=True,
):
    """Plot enrichment results for multiple sets.

    The original data arguments and their order are preserved. Plot styling
    is configurable through optional arguments appended to the signature.
    When ``return_fig_ax`` is true, return ``(fig, axes)``.
    """

    set_plot_style()

    # stratify_list.reverse()

    if figsize is None:
        figsize = (10, height_per_set * len(set_results_json))
    fig, axs = plt.subplots(figsize=figsize, nrows=1,
                            ncols=len(env_id_list), sharey=True)
    axs = np.atleast_1d(axs)

    y_values = {s: i for i, s in enumerate([s[0] for s in set_label_map])}

    for ax, env_id in zip(axs, env_id_list):
        for s, label in set_label_map:
            obs_value = set_results_json[s]['all']['obs_values'][env_id]
            perm_values = set_results_json[s]['all']['perm_values'][env_id]
            obs_z = zscore([obs_value] + perm_values)[0]
            perm_z = zscore([obs_value] + perm_values)[1:]

            ax.scatter(perm_z, [y_values[s]] * len(perm_values),
                       color=perm_color, s=perm_size, alpha=perm_alpha,
                       marker=perm_marker, facecolors=perm_facecolors)
            ax.scatter(obs_z, y_values[s],
                       color=obs_color, s=obs_size, alpha=obs_alpha,
                       marker=obs_marker)

            # 绘制显著性标记
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            p_value = set_results_json[s]['all']['stat'][env_id]['pval']
            mean_perm = set_results_json[s]['all']['stat'][env_id]['mean_perm']

            if p_value < significance_threshold:
                if obs_value > mean_perm:
                    ax.plot(obs_z+(current_xlim[1] - current_xlim[0])*star_offset_fraction,
                            y_values[s], marker=star_marker, markersize=star_size,
                            color=star_color, linestyle='none')
                elif obs_value < mean_perm:
                    ax.plot(obs_z-(current_xlim[1] - current_xlim[0])*star_offset_fraction,
                            y_values[s], marker=star_marker, markersize=star_size,
                            color=star_color, linestyle='none')

        stratify_label_map_dict = {s[0]: s[1] for s in set_label_map}

        ax.set_yticks(list(y_values.values()))
        ax.set_yticklabels(
            [stratify_label_map_dict.get(s, s) for s in y_values],
            fontsize=set_tick_fontsize,
        )
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        if xlim is not None:
            ax.set_xlim(xlim)
        if zero_line:
            ax.axvline(x=0, color=zero_line_color,
                       linestyle=zero_line_style, linewidth=zero_line_width)

        # 设置x轴粗细
        # ax.spines['bottom'].set_linewidth(0.7)

        if remove_right_spine:
            ax.spines['right'].set_visible(False)
        if remove_top_spine:
            ax.spines['top'].set_visible(False)
        if remove_left_spine:
            ax.spines['left'].set_visible(False)

        if grid:
            grid_kwargs = {
                'linestyle': grid_style,
                'alpha': grid_alpha,
            }
            if grid_color is not None:
                grid_kwargs['color'] = grid_color
            if grid_linewidth is not None:
                grid_kwargs['linewidth'] = grid_linewidth
            ax.grid(axis='x', **grid_kwargs)
        ax.tick_params(axis='y', which='both', left=False)
        x_tick_params = {}
        if tick_fontsize is not None:
            x_tick_params['labelsize'] = tick_fontsize
        if tick_direction is not None:
            x_tick_params['direction'] = tick_direction
        if x_tick_params:
            ax.tick_params(axis='x', **x_tick_params)

        ax.set_ylim(-ylim_pad[0], len(y_values) - ylim_pad[1])

    # Attach each title to its axes so it follows tight_layout and stays centered.
    for ax, env_id in zip(axs, env_id_list):
        env_label = env_metadata_df.loc[
            env_metadata_df['env_id'] == env_id, 'description'
        ].values[0]
        ax.set_title(
            wrap_label(env_label, width=title_wrap_width),
            loc='center', x=0.5, pad=title_pad, fontsize=title_fontsize,
            ha='center', multialignment='center',
        )

    # ax.yaxis_inverted()

    if tight_layout:
        fig.tight_layout()
    if show:
        plt.show()
    if return_fig_ax:
        return fig, axs

# simulate ranked sites for testing enrichment analysis


def simulate_ranked_sites(
    n_sites=10_000,
    fg_fraction=0.1,
    enrichment_strength=0,
    seed=None,
):
    rng = np.random.default_rng(seed)

    n_fg = round(n_sites * fg_fraction)

    # 0 = strongest GEA, 1 = weakest GEA
    x = np.linspace(0, 1, n_sites)

    # stronger GEA ranks have larger foreground weight
    weights = np.exp(-enrichment_strength * x)
    weights /= weights.sum()

    fg_idx = rng.choice(
        n_sites,
        size=n_fg,
        replace=False,
        p=weights,
    )

    foreground = np.zeros(n_sites, dtype=bool)
    foreground[fg_idx] = True

    return foreground


def assign_fixed_foreground_labels(
    pruned_site_dfs,
    baseline_foreground_df,
    foreground_cols,
    *,
    fg_fraction=0.1,
    seed=42,
):
    """Attach fixed per-ID simulated labels to every LD threshold's sites.

    baseline_foreground_df contains unique ID values and existing boolean
    label columns (e.g. one column per enrichment strength). Baseline labels
    are preserved exactly, and every threshold must contain all baseline IDs.
    For each label column, draw independent Bernoulli(fg_fraction) labels once
    for the union of non-baseline IDs, then reuse them across thresholds.
    Thus fg_fraction is an expected proportion for additions, not an exact
    count constraint. This function does not regenerate baseline labels.

    Threshold keys are visited in sorted order to build the ID union, so dict
    insertion order does not affect the draws. A fixed seed, input row orders,
    and foreground_cols order reproduce the labels. Inputs and the global
    NumPy RNG are untouched. Output rows/index are preserved and labels have
    boolean dtype with no missing values.
    """
    foreground_cols = list(foreground_cols)
    if not foreground_cols or not pd.Index(foreground_cols).is_unique or 'ID' in foreground_cols:
        raise ValueError(
            "foreground_cols must be non-empty, unique, and exclude 'ID'")
    if not np.isfinite(fg_fraction) or not 0 <= fg_fraction <= 1:
        raise ValueError("fg_fraction must be finite and in [0, 1]")
    if not pruned_site_dfs:
        raise ValueError("pruned_site_dfs must contain at least one threshold")

    required = ['ID'] + foreground_cols
    missing = [col for col in required if col not in baseline_foreground_df]
    if missing:
        raise ValueError(f"Missing baseline columns: {missing}")
    if baseline_foreground_df['ID'].isna().any() or not baseline_foreground_df['ID'].is_unique:
        raise ValueError("Baseline IDs must be unique and non-missing")
    for col in foreground_cols:
        labels = baseline_foreground_df[col]
        if labels.isna().any() or not pd.api.types.is_bool_dtype(labels.dtype):
            raise ValueError(
                f"Baseline foreground column {col!r} must be boolean and non-missing")

    baseline = baseline_foreground_df.set_index('ID')[foreground_cols]
    id_series = []
    for threshold in sorted(pruned_site_dfs):
        frame = pruned_site_dfs[threshold]
        if 'ID' not in frame or frame['ID'].isna().any() or not frame['ID'].is_unique:
            raise ValueError(
                f"IDs at R2={threshold} must be unique and non-missing")
        if not baseline.index.isin(frame['ID']).all():
            raise ValueError(
                f"Sites at R2={threshold} are missing baseline IDs")
        id_series.append(frame['ID'])
    all_ids = pd.Index(
        pd.concat(id_series, ignore_index=True).drop_duplicates())
    added_ids = all_ids[~all_ids.isin(baseline.index)]
    rng = np.random.default_rng(seed)
    results = {threshold: frame.copy()
               for threshold, frame in pruned_site_dfs.items()}
    for col in foreground_cols:
        added_labels = pd.Series(rng.random(
            len(added_ids)) < fg_fraction, index=added_ids)
        label_by_id = pd.concat([baseline[col].astype(bool), added_labels])
        for frame in results.values():
            mapped = frame['ID'].map(label_by_id)
            if mapped.isna().any():
                raise RuntimeError(
                    "A site is missing its fixed foreground label")
            frame[col] = mapped.astype(bool)
    return results


def plot_enrichment_simulation_results(
    stats_df,
    *,
    permutations=1000,
    significance_level=0.05,
    title="Enrichment simulation",
    figsize=(14, 6.5),
    output_file=None,
    dpi=200,
    show=True,
    return_fig_ax=True,
):
    """Plot observed FE curves and a categorical permutation-P heatmap.

    stats_df must contain r2_threshold, enrichment_strength, obs_value, pval,
    with one row per combination and a complete parameter grid. Repeated
    simulations must be selected separately; no averaging is done here.
    Rows may be unordered and both parameter grids may be nonuniform. Curves
    use numerical strength coordinates; heatmap columns are discrete levels.

    permutations must match the two-sided, +1-corrected empirical P values,
    whose minimum is 2 / (permutations + 1). The heatmap distinguishes
    non-significant P, significant P above that minimum, and P at the minimum.
    Input data and global plot settings are unchanged. Optionally save the
    figure (PNG/PDF/SVG according to output_file) and/or display it.

    Returns (fig, axes) for further customization when return_fig_ax is true.
    The caller may close fig with plt.close(fig) when show=False. Example:
        fig, axes = plot_enrichment_simulation_results(stats_df)
    """
    from operator import index
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    required = ['r2_threshold', 'enrichment_strength', 'obs_value', 'pval']
    missing = [col for col in required if col not in stats_df]
    if missing:
        raise ValueError(f"Missing simulation result columns: {missing}")
    data = stats_df[required].apply(pd.to_numeric, errors='raise').copy()
    if data.empty or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Simulation results must be non-empty and finite")
    if data.duplicated(['r2_threshold', 'enrichment_strength']).any():
        raise ValueError(
            "Expected one row per R2/strength combination; select one replicate first")
    if (
        not data['r2_threshold'].between(0, 1).all()
        or not data['pval'].between(0, 1).all()
        or (data['obs_value'] < 0).any()
    ):
        raise ValueError(
            "R2 and P must be in [0, 1], and observed FE must be non-negative")
    permutations = index(permutations)
    if permutations < 1:
        raise ValueError("permutations must be a positive integer")
    if not np.isfinite(significance_level) or not 0 < significance_level < 1:
        raise ValueError(
            "significance_level must be finite and strictly between 0 and 1")
    p_floor = 2 / (permutations + 1)
    if p_floor >= significance_level:
        raise ValueError(
            "Use enough permutations for the minimum P to fall below significance_level")

    fe = data.pivot(index='enrichment_strength',
                    columns='r2_threshold', values='obs_value')
    pvals = data.pivot(index='r2_threshold',
                       columns='enrichment_strength', values='pval')
    fe = fe.sort_index().sort_index(axis=1)
    pvals = pvals.sort_index().sort_index(axis=1)
    if fe.isna().any().any():
        raise ValueError(
            "Simulation results must contain every R2/strength combination")
    p_array = pvals.to_numpy()
    at_floor = np.isclose(p_array, p_floor, rtol=1e-6, atol=1e-12)
    if np.any((p_array < p_floor) & ~at_floor):
        raise ValueError(
            "P values are below the theoretical minimum; check permutations and P definition")
    categories = np.where(at_floor, 2, np.where(
        p_array < significance_level, 1, 0))
    thresholds = pvals.index.to_numpy()
    strengths = pvals.columns.to_numpy()
    base_colors = ['#1B365D', '#336FA1',
                   '#5596AA', '#A17B29', '#C27232', '#A23858']
    if len(thresholds) <= len(base_colors):
        line_colors = base_colors[:len(thresholds)]
    else:
        line_colors = plt.get_cmap('viridis')(
            np.linspace(0.05, 0.85, len(thresholds)))
    heat_colors = ['#ECEFF3', '#75B0CE', '#224D78']

    with plt.rc_context({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    }):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.subplots_adjust(left=0.075, right=0.985,
                            bottom=0.28, top=0.85, wspace=0.27)
        fig.suptitle(title, x=0.075, y=0.97, ha='left',
                     fontsize=16, fontweight='bold')
        ax = axes[0]
        for threshold, color in zip(thresholds, line_colors):
            ax.plot(strengths, fe[threshold].to_numpy(), color=color,
                    marker='o', markersize=3, linewidth=1.8, label=f'$r^2$ = {threshold:g}')
        ax.axhline(1, color='#6B7280', linestyle='--', linewidth=1)
        ax.set_title('A  Observed enrichment', loc='left',
                     pad=15, fontweight='bold')
        ax.set_xlabel('Simulated enrichment strength')
        ax.set_ylabel('Observed fold enrichment (FE)')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E5E7EB', linewidth=0.8)
        ax.margins(x=0.02, y=0.12)
        ax.legend(loc='upper left', frameon=False, fontsize=8.5,
                  ncol=2 if len(thresholds) > 4 else 1)

        ax = axes[1]
        ax.imshow(categories, cmap=ListedColormap(heat_colors), vmin=-0.5, vmax=2.5,
                  aspect='auto', interpolation='nearest', origin='upper')
        ax.set_xticks(np.arange(len(strengths)))
        ax.set_xticklabels([f'{value:g}' for value in strengths],
                           rotation=45 if len(strengths) > 10 else 0,
                           ha='right' if len(strengths) > 10 else 'center', fontsize=8.5)
        ax.set_yticks(np.arange(len(thresholds)))
        ax.set_yticklabels([f'{value:g}' for value in thresholds])
        ax.set_xticks(np.arange(-0.5, len(strengths), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(thresholds), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.8)
        ax.tick_params(which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title('B  Permutation significance',
                     loc='left', pad=15, fontweight='bold')
        ax.set_xlabel('Simulated enrichment strength (discrete settings)')
        ax.set_ylabel('LD $r^2$ threshold')
        legend_labels = [
            f'P >= {significance_level:g}',
            f'{p_floor:.6g} < P < {significance_level:g}',
            f'P = {p_floor:.6g} (minimum)',
        ]
        ax.legend(handles=[Patch(facecolor=color, label=label)
                           for color, label in zip(heat_colors, legend_labels)],
                  loc='upper left', bbox_to_anchor=(0, -0.28), frameon=False, fontsize=9)
        fig.text(0.075, 0.025,
                 f'{len(data)} parameter combinations; {permutations:,} two-sided permutations. '
                 f'{int(at_floor.sum())} P-values reach the permutation minimum.',
                 fontsize=9, color='#4B5563')
        if output_file is not None:
            fig.savefig(output_file, dpi=dpi,
                        bbox_inches='tight', facecolor='white')
        if show:
            plt.show()
    if return_fig_ax:
        return fig, axes


def weighted_ld_jaccard(
    r2_df,
    foreground_index,
    background_index,
):
    fg_ids = list(foreground_index)
    bg_ids = list(background_index)

    fg_map = {x: i for i, x in enumerate(fg_ids)}
    bg_map = {x: i for i, x in enumerate(bg_ids)}

    weights = np.zeros(
        (len(fg_ids), len(bg_ids)),
        dtype=float
    )

    for row in r2_df.itertuples():

        if row.INDEX_1 not in fg_map:
            continue

        if row.INDEX_2 not in bg_map:
            continue

        w = max(0, (row.R2 - 0.5) / 0.5)

        weights[
            fg_map[row.INDEX_1],
            bg_map[row.INDEX_2]
        ] = w

    # scipy solves minimum-cost assignment,
    # so use negative weights
    fg_idx, bg_idx = linear_sum_assignment(
        -weights
    )

    intersection = weights[
        fg_idx,
        bg_idx
    ].sum()

    union = (
        len(fg_ids)
        + len(bg_ids)
        - intersection
    )

    return intersection / union if union > 0 else 0


def assign_child_tags_by_parent_ld(
    prune_context_df,
    fg_fraction,
    seed=42,
    r2_threshold=0.5,
    stickiness=1.0,
):
    """
    Assign FG/BG tags to LD-pruned child SNPs according to
    pruning-parent tag and parent-child LD strength.

    Core idea
    ---------
    For each child SNP i:

        parent tag:
            P_i = 1  if parent is FG
            P_i = 0  if parent is BG

        scaled LD strength:
            r_i = (R2_i - r2_threshold) / (1 - r2_threshold)

        effective inheritance strength:
            s_i = 1 - (1 - r_i) ** stickiness

        FG propensity:
            q_i = (1 - s_i) * fg_fraction + s_i * P_i

    Therefore:

        stickiness = 1
            original parent-LD inheritance model

        stickiness > 1
            stronger child-parent tag inheritance
            -> stronger FG-FG / BG-BG LD cohesion
            -> lower initial FG-BG LD contamination

    The final number of FG child SNPs is fixed exactly so that
    the total genome-wide FG fraction remains constant.

    Parameters
    ----------
    prune_context_df : pd.DataFrame
        Must contain:
            LD_PRUNING_KEEP
            LD_PRUNED_BY
            LD_PRUNED_R2
            foreground

        Anchor SNPs (LD_PRUNING_KEEP=True) must already have
        foreground tags assigned.

    fg_fraction : float
        Required global foreground fraction.

    seed : int
        Random seed used for weighted sampling.

    r2_threshold : float
        LD pruning threshold used to define child-parent links.

    stickiness : float
        Strength of child inheritance from its pruning parent.

        = 1 : original LD-based inheritance
        > 1 : stronger within-class LD cohesion
        < 1 : weaker inheritance

    Returns
    -------
    prune_context_df : pd.DataFrame
        Copy of input DataFrame with child foreground tags assigned.

    child_df : pd.DataFrame
        Diagnostic table containing:
            LD_PRUNED_BY
            LD_PRUNED_R2
            parent_foreground
            r_scale
            inheritance_strength
            p_fg
            foreground
    """

    if not 0 < fg_fraction < 1:
        raise ValueError(
            "fg_fraction must be between 0 and 1."
        )

    if not 0 <= r2_threshold < 1:
        raise ValueError(
            "r2_threshold must satisfy 0 <= r2_threshold < 1."
        )

    if stickiness <= 0:
        raise ValueError(
            "stickiness must be > 0."
        )

    prune_context_df = prune_context_df.copy()

    keep_mask = (
        prune_context_df[
            'LD_PRUNING_KEEP'
        ]
        .astype(bool)
    )

    child_mask = ~keep_mask

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # 1. Build child table
    # --------------------------------------------------

    child_df = prune_context_df.loc[
        child_mask,
        [
            'LD_PRUNED_BY',
            'LD_PRUNED_R2',
        ]
    ].copy()

    # --------------------------------------------------
    # 2. Map each child to its pruning-parent tag
    # --------------------------------------------------

    parent_tag_dict = (
        prune_context_df.loc[
            keep_mask,
            'foreground'
        ]
        .astype(bool)
        .to_dict()
    )

    child_df[
        'parent_foreground'
    ] = (
        child_df[
            'LD_PRUNED_BY'
        ]
        .map(parent_tag_dict)
    )

    if child_df[
        'parent_foreground'
    ].isna().any():
        raise ValueError(
            "Some child SNPs could not be mapped "
            "to their pruning parent."
        )

    # --------------------------------------------------
    # 3. Scale parent-child R2 to 0 ~ 1
    #
    # R2 = r2_threshold -> 0
    # R2 = 1            -> 1
    # --------------------------------------------------

    r2 = (
        child_df[
            'LD_PRUNED_R2'
        ]
        .to_numpy(
            dtype=float
        )
    )

    r_scale = (
        (r2 - r2_threshold)
        /
        (1.0 - r2_threshold)
    )

    r_scale = np.clip(
        r_scale,
        0.0,
        1.0,
    )

    child_df[
        'r_scale'
    ] = r_scale

    # --------------------------------------------------
    # 4. Strengthen / weaken parent inheritance
    #
    # stickiness = 1:
    #     inheritance_strength = r_scale
    #
    # stickiness > 1:
    #     stronger inheritance
    #
    # Example, r_scale=0.5:
    #
    #     stickiness=1 -> 0.50
    #     stickiness=2 -> 0.75
    #     stickiness=4 -> 0.9375
    # --------------------------------------------------

    inheritance_strength = (
        1.0
        - (
            1.0
            - r_scale
        ) ** stickiness
    )

    child_df[
        'inheritance_strength'
    ] = inheritance_strength

    # --------------------------------------------------
    # 5. Calculate FG propensity
    #
    # q_i = (1-s_i) * fg_fraction + s_i * parent_tag
    #
    # parent = FG:
    #     q_i moves toward 1
    #
    # parent = BG:
    #     q_i moves toward 0
    #
    # s_i = 0:
    #     q_i = fg_fraction
    #
    # s_i = 1:
    #     q_i = parent tag
    # --------------------------------------------------

    parent_fg = (
        child_df[
            'parent_foreground'
        ]
        .astype(float)
        .to_numpy()
    )

    p_fg = (
        (
            1.0
            - inheritance_strength
        )
        * fg_fraction
        +
        inheritance_strength
        * parent_fg
    )

    child_df[
        'p_fg'
    ] = p_fg

    # --------------------------------------------------
    # 6. Exact global FG quota
    # --------------------------------------------------

    K_total = round(
        len(
            prune_context_df
        )
        * fg_fraction
    )

    K_anchor = int(
        prune_context_df.loc[
            keep_mask,
            'foreground'
        ].sum()
    )

    K_child = (
        K_total
        - K_anchor
    )

    if not 0 <= K_child <= len(child_df):
        raise ValueError(
            f"Invalid K_child={K_child}; "
            f"K_total={K_total}, "
            f"K_anchor={K_anchor}, "
            f"N_child={len(child_df)}"
        )

    # --------------------------------------------------
    # 7. Sample exactly K_child foreground children
    #
    # p_fg is used as sampling propensity.
    # The final number of FG children is exact.
    # --------------------------------------------------

    child_foreground = np.zeros(
        len(child_df),
        dtype=bool,
    )

    if K_child > 0:

        weights = p_fg.copy()

        # Avoid zero-probability problems when exact quota
        # requires more sites than currently have nonzero weight.
        weights = np.maximum(
            weights,
            1e-12,
        )

        fg_pos = rng.choice(
            len(child_df),
            size=K_child,
            replace=False,
            p=(
                weights
                / weights.sum()
            ),
        )

        child_foreground[
            fg_pos
        ] = True

    # --------------------------------------------------
    # 8. Put child labels back
    # --------------------------------------------------

    prune_context_df.loc[
        child_df.index,
        'foreground'
    ] = child_foreground

    prune_context_df[
        'foreground'
    ] = (
        prune_context_df[
            'foreground'
        ]
        .astype(bool)
    )

    child_df[
        'foreground'
    ] = child_foreground

    # --------------------------------------------------
    # 9. Sanity check
    # --------------------------------------------------

    assert (
        int(
            prune_context_df[
                'foreground'
            ].sum()
        )
        == K_total
    )

    return (
        prune_context_df,
        child_df,
    )


def plot_enrichment_and_overlap(
    enrich_results_json,
    overlap_results,
    env_id_list,
    env_metadata_df,
    stratify_label_map,

    # ============================================================
    # enrichment statistics
    # ============================================================
    significance_threshold=0.05,

    # ============================================================
    # overlap dataframe columns
    # ============================================================
    start_col='gea_rank_start',
    end_col='gea_rank_end',
    value_col='harmonic_mean_overlap',

    # ============================================================
    # overall figure
    # ============================================================
    figsize_per_panel=(3.0, 4.2),  # (width, height) in inches
    height_ratios=(2.25, 1.0),
    wspace=0.15,
    hspace=0.30,

    # ============================================================
    # TOP: enrichment permutation points
    # ============================================================
    perm_color='gray',
    perm_all_color='#B3AF57',
    perm_size=40,
    perm_alpha=0.20,
    perm_marker='o',
    perm_facecolors='none',

    # observed points
    obs_color='#5580B0',
    obs_size=120,
    obs_alpha=1.0,
    obs_marker='o',

    # significance stars
    star_marker='*',
    star_color='black',
    star_size=14,
    star_offset_fraction=0.12,

    # zero line
    zero_line=True,
    zero_line_color='gray',
    zero_line_style='dashed',
    zero_line_width=0.7,

    # vertical grid
    top_grid=True,
    top_grid_color='gray',
    top_grid_style='dashed',
    top_grid_alpha=0.5,
    top_grid_linewidth=0.7,

    # top axes
    top_xlim=None,
    top_xlabel='Enrichment z-score',
    top_xlabel_fontsize=10,
    top_xlabel_pad=4,
    top_tick_fontsize=10,

    stratify_tick_fontsize=12,
    top_ylim_pad=(0.5, 0.2),

    # ============================================================
    # BOTTOM: LD linkage bars
    # ============================================================
    bar_color=None,           # None = matplotlib default blue
    bar_width_scale=1.0,
    bar_edgecolor='white',
    bar_linewidth=0.7,

    # emphasize top 0-0.05 bin
    highlight_bin_index=0,
    highlight_bar_alpha=1.0,
    other_bar_alpha=0.42,

    # random baseline
    baseline_color='black',
    baseline_linewidth=1.2,
    baseline_width_scale=0.82,

    # bottom axes
    bottom_xlabel='GEA rank quantile',
    bottom_ylabel='FG–BG LD\nconnectivity',
    bottom_xlim=(0.0, 1.0),
    bottom_x_margin=0.015,
    bottom_ylim=(0.0, 0.8),
    bottom_yticks=(0.0, 0.2, 0.4, 0.6, 0.8),

    n_xticks=6,
    x_tick_format='.2f',

    # ============================================================
    # fonts
    # ============================================================
    title_fontsize=14,
    title_pad=6,
    title_wrap_width=20,

    bottom_axis_label_fontsize=10,
    bottom_tick_fontsize=9,

    # ============================================================
    # spines / ticks
    # ============================================================
    top_remove_left_spine=True,
    top_remove_right_spine=True,
    top_remove_top_spine=True,

    bottom_remove_top_spine=True,
    bottom_remove_right_spine=True,

    tick_direction='out',

    # ============================================================
    # global margins
    # ============================================================
    fig_left=0.075,
    fig_right=0.995,
    fig_bottom=0.12,
    fig_top=0.90,

    # ============================================================
    # output
    # ============================================================
    show=True,
    return_fig_ax=True,
):
    """
    Combined figure:

        Top row:
            Enrichment z-score based on observed value versus
            permutation distribution.

        Bottom row:
            FG-BG LD linkage across GEA-rank quantiles.

    The default appearance is optimized for a manuscript-style
    4-panel figure, but most style parameters remain configurable.

    Returns
    -------
    fig, axes

    axes is a dictionary:
        axes['top']
        axes['bottom']
    """

    # ============================================================
    # Global style
    # ============================================================
    set_plot_style()

    n_env = len(env_id_list)

    if n_env == 0:
        raise ValueError("env_id_list is empty.")

    # ============================================================
    # Environmental variable names
    # ============================================================
    env_name_map = {
        row['env_id']: row['description']
        for _, row in env_metadata_df.iterrows()
    }

    # ============================================================
    # Stratification
    # ============================================================
    stratify_list = [x[0] for x in stratify_label_map]

    stratify_label_dict = {
        x[0]: x[1]
        for x in stratify_label_map
    }

    y_values = {
        s: i
        for i, s in enumerate(stratify_list)
    }

    # ============================================================
    # Figure
    # ============================================================
    fig = plt.figure(
        figsize=(
            figsize_per_panel[0] * n_env,
            figsize_per_panel[1],
        )
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=n_env,
        height_ratios=height_ratios,
        wspace=wspace,
        hspace=hspace,
    )

    top_axes = []
    bottom_axes = []

    for i in range(n_env):

        if i == 0:

            ax_top = fig.add_subplot(
                gs[0, i]
            )

            ax_bottom = fig.add_subplot(
                gs[1, i]
            )

        else:

            ax_top = fig.add_subplot(
                gs[0, i],
                sharey=top_axes[0],
            )

            ax_bottom = fig.add_subplot(
                gs[1, i],
                sharey=bottom_axes[0],
            )

        top_axes.append(ax_top)
        bottom_axes.append(ax_bottom)

    top_axes = np.asarray(top_axes)
    bottom_axes = np.asarray(bottom_axes)

    # ============================================================
    # TOP ROW
    # ============================================================
    for i, (ax, env_id) in enumerate(
        zip(top_axes, env_id_list)
    ):

        star_records = []

        # --------------------------------------------------------
        # Each MAF stratum
        # --------------------------------------------------------
        for s in stratify_list:

            obs_value = (
                enrich_results_json[s]
                ['obs_values'][env_id]
            )

            perm_values = (
                enrich_results_json[s]
                ['perm_values'][env_id]
            )

            # ----------------------------------------------------
            # Convert observed + permutation values to z-scores
            # ----------------------------------------------------
            values = np.asarray(
                [obs_value] + list(perm_values),
                dtype=float,
            )

            z_values = zscore(values)

            obs_z = z_values[0]
            perm_z = z_values[1:]

            # ----------------------------------------------------
            # Permutation points
            # ----------------------------------------------------
            if s == 'all':
                current_perm_color = perm_all_color
            else:
                current_perm_color = perm_color

            ax.scatter(
                perm_z,
                np.full(
                    len(perm_z),
                    y_values[s],
                ),
                color=current_perm_color,
                s=perm_size,
                alpha=perm_alpha,
                marker=perm_marker,
                facecolors=perm_facecolors,
                zorder=1,
            )

            # ----------------------------------------------------
            # Observed value
            # ----------------------------------------------------
            ax.scatter(
                obs_z,
                y_values[s],
                color=obs_color,
                s=obs_size,
                alpha=obs_alpha,
                marker=obs_marker,
                zorder=4,
            )

            # ----------------------------------------------------
            # Significance
            # ----------------------------------------------------
            stat = (
                enrich_results_json[s]
                ['stat'][env_id]
            )

            p_value = stat['pval']
            mean_perm = stat['mean_perm']

            if p_value < significance_threshold:

                if obs_value > mean_perm:
                    direction = 1

                elif obs_value < mean_perm:
                    direction = -1

                else:
                    direction = 0

                if direction != 0:
                    star_records.append(
                        (
                            obs_z,
                            y_values[s],
                            direction,
                        )
                    )

        # --------------------------------------------------------
        # Optional common x range
        # --------------------------------------------------------
        if top_xlim is not None:
            ax.set_xlim(top_xlim)

        current_xlim = ax.get_xlim()
        x_span = current_xlim[1] - current_xlim[0]

        # --------------------------------------------------------
        # Significance stars
        # --------------------------------------------------------
        for obs_z, y, direction in star_records:

            star_x = (
                obs_z
                + direction
                * x_span
                * star_offset_fraction
            )

            ax.plot(
                star_x,
                y,
                marker=star_marker,
                markersize=star_size,
                color=star_color,
                linestyle='none',
                zorder=6,
            )

        # --------------------------------------------------------
        # x = 0
        # --------------------------------------------------------
        if zero_line:

            ax.axvline(
                x=0,
                color=zero_line_color,
                linestyle=zero_line_style,
                linewidth=zero_line_width,
                zorder=0,
            )

        # --------------------------------------------------------
        # Grid
        # --------------------------------------------------------
        if top_grid:

            ax.grid(
                axis='x',
                color=top_grid_color,
                linestyle=top_grid_style,
                alpha=top_grid_alpha,
                linewidth=top_grid_linewidth,
            )

        # --------------------------------------------------------
        # Y labels
        # --------------------------------------------------------
        ax.set_yticks(
            list(y_values.values())
        )

        if i == 0:

            ax.set_yticklabels(
                [
                    stratify_label_dict.get(s, s)
                    for s in stratify_list
                ],
                fontsize=stratify_tick_fontsize,
            )

        else:

            ax.tick_params(
                axis='y',
                labelleft=False,
            )

        ax.tick_params(
            axis='y',
            which='both',
            left=False,
        )

        ax.set_ylim(
            -top_ylim_pad[0],
            len(stratify_list)
            - top_ylim_pad[1],
        )

        # --------------------------------------------------------
        # Each panel gets its own x label
        # --------------------------------------------------------
        ax.set_xlabel(
            top_xlabel,
            fontsize=top_xlabel_fontsize,
            labelpad=top_xlabel_pad,
        )

        ax.tick_params(
            axis='x',
            labelsize=top_tick_fontsize,
            direction=tick_direction,
        )

        # --------------------------------------------------------
        # Spines
        # --------------------------------------------------------
        if top_remove_left_spine:
            ax.spines['left'].set_visible(False)

        if top_remove_right_spine:
            ax.spines['right'].set_visible(False)

        if top_remove_top_spine:
            ax.spines['top'].set_visible(False)

        # --------------------------------------------------------
        # Title
        # --------------------------------------------------------
        display_name = env_name_map.get(
            env_id,
            env_id,
        )

        ax.set_title(
            wrap_label(
                display_name,
                width=title_wrap_width,
            ),
            loc='center',
            x=0.5,
            pad=title_pad,
            fontsize=title_fontsize,
            ha='center',
            multialignment='center',
        )

    # ============================================================
    # BOTTOM ROW
    # ============================================================

    # Get matplotlib default blue if bar_color is not specified
    if bar_color is None:

        bar_color = (
            plt.rcParams[
                'axes.prop_cycle'
            ]
            .by_key()['color'][0]
        )

    base_rgb = mcolors.to_rgb(
        bar_color
    )

    for i, (ax, env_id) in enumerate(
        zip(bottom_axes, env_id_list)
    ):

        # --------------------------------------------------------
        # Data
        # --------------------------------------------------------
        overlap_df = (
            overlap_results[env_id]
            ['overlap_df']
            .copy()
            .sort_values(
                [start_col, end_col]
            )
            .reset_index(drop=True)
        )

        random_overlap_df = (
            overlap_results[env_id]
            ['random_overlap_df']
            .copy()
            .sort_values(
                [start_col, end_col]
            )
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Consistency check
        # --------------------------------------------------------
        if len(overlap_df) != len(random_overlap_df):

            raise ValueError(
                f"{env_id}: overlap_df and "
                f"random_overlap_df have "
                f"different numbers of bins."
            )

        bin_left = (
            overlap_df[start_col]
            .to_numpy(dtype=float)
        )

        bin_right = (
            overlap_df[end_col]
            .to_numpy(dtype=float)
        )

        random_bin_left = (
            random_overlap_df[start_col]
            .to_numpy(dtype=float)
        )

        random_bin_right = (
            random_overlap_df[end_col]
            .to_numpy(dtype=float)
        )

        if not (
            np.allclose(
                bin_left,
                random_bin_left,
            )
            and
            np.allclose(
                bin_right,
                random_bin_right,
            )
        ):

            raise ValueError(
                f"{env_id}: bins in overlap_df "
                f"and random_overlap_df do not match."
            )

        # --------------------------------------------------------
        # Bin geometry
        # --------------------------------------------------------
        bin_center = (
            bin_left + bin_right
        ) / 2

        bin_width = (
            bin_right - bin_left
        )

        observed = (
            overlap_df[value_col]
            .to_numpy(dtype=float)
        )

        baseline = (
            random_overlap_df[value_col]
            .to_numpy(dtype=float)
        )

        # --------------------------------------------------------
        # Bar colors
        #
        # First 0-0.05 bin:
        #     full opacity
        #
        # Remaining bins:
        #     same color, lower alpha
        # --------------------------------------------------------
        bar_colors = []

        for j in range(len(observed)):

            if j == highlight_bin_index:
                alpha = highlight_bar_alpha

            else:
                alpha = other_bar_alpha

            bar_colors.append(
                (
                    base_rgb[0],
                    base_rgb[1],
                    base_rgb[2],
                    alpha,
                )
            )

        # --------------------------------------------------------
        # Bars
        # --------------------------------------------------------
        ax.bar(
            bin_center,
            observed,
            width=(
                bin_width
                * bar_width_scale
            ),
            align='center',
            color=bar_colors,
            edgecolor=bar_edgecolor,
            linewidth=bar_linewidth,
            zorder=2,
        )

        # --------------------------------------------------------
        # Random baseline
        # --------------------------------------------------------
        baseline_half_width = (
            bin_width
            * baseline_width_scale
            / 2
        )

        for x, half_width, y in zip(
            bin_center,
            baseline_half_width,
            baseline,
        ):

            ax.hlines(
                y=y,
                xmin=x - half_width,
                xmax=x + half_width,
                color=baseline_color,
                linewidth=baseline_linewidth,
                zorder=5,
            )

        # --------------------------------------------------------
        # X ticks
        # --------------------------------------------------------
        bin_edges = np.unique(
            np.concatenate(
                [
                    bin_left,
                    bin_right,
                ]
            )
        )

        if len(bin_edges) <= n_xticks:

            xticks = bin_edges

        else:

            tick_indices = np.linspace(
                0,
                len(bin_edges) - 1,
                n_xticks,
                dtype=int,
            )

            xticks = bin_edges[
                tick_indices
            ]

        ax.set_xticks(
            xticks
        )

        xtick_labels = []

        for x in xticks:

            if np.isclose(x, 0):

                xtick_labels.append(
                    '0'
                )

            elif np.isclose(x, 1):

                xtick_labels.append(
                    '1'
                )

            else:

                xtick_labels.append(
                    format(
                        x,
                        x_tick_format,
                    )
                )

        ax.set_xticklabels(
            xtick_labels,
            fontsize=bottom_tick_fontsize,
        )

        # --------------------------------------------------------
        # X limits + small horizontal margin
        # --------------------------------------------------------
        if bottom_xlim is None:

            xmin = bin_left.min()
            xmax = bin_right.max()

        else:

            xmin, xmax = bottom_xlim

        ax.set_xlim(
            xmin - bottom_x_margin,
            xmax + bottom_x_margin,
        )

        # --------------------------------------------------------
        # Y range fixed to 0-0.8
        # --------------------------------------------------------
        if bottom_ylim is not None:

            ax.set_ylim(
                bottom_ylim
            )

        if bottom_yticks is not None:

            ax.set_yticks(
                bottom_yticks
            )

        # --------------------------------------------------------
        # Axis labels
        # --------------------------------------------------------
        ax.set_xlabel(
            bottom_xlabel,
            fontsize=bottom_axis_label_fontsize,
        )

        if i == 0:

            ax.set_ylabel(
                bottom_ylabel,
                fontsize=bottom_axis_label_fontsize,
            )

        else:

            ax.set_ylabel('')

            ax.tick_params(
                axis='y',
                labelleft=False,
            )

        # --------------------------------------------------------
        # Tick style
        # --------------------------------------------------------
        ax.tick_params(
            axis='both',
            labelsize=bottom_tick_fontsize,
            direction=tick_direction,
        )

        # --------------------------------------------------------
        # Spines
        # --------------------------------------------------------
        if bottom_remove_top_spine:
            ax.spines['top'].set_visible(False)

        if bottom_remove_right_spine:
            ax.spines['right'].set_visible(False)

    # ============================================================
    # Global layout
    # ============================================================
    fig.subplots_adjust(
        left=fig_left,
        right=fig_right,
        bottom=fig_bottom,
        top=fig_top,
        wspace=wspace,
        hspace=hspace,
    )

    axes = {
        'top': top_axes,
        'bottom': bottom_axes,
    }

    if show:
        plt.show()

    if return_fig_ax:
        return fig, axes

# if __name__ == "__main__":

#     import os
#     import pandas as pd
#     import numpy as np

#     site_info_df_h5 = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data/results/site_contexts.genome.maf0.05.h5"
#     input_vcf_file = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz"
#     site_info_df = pd.read_hdf(site_info_df_h5, key='df')

#     data_dir = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data"
#     # genome site level envGWAS results
#     env_id_list = ["bio18", "gsl", "al", "striga"]
#     env_metadata_df = pd.read_csv(os.path.join(
#         data_dir, "source", "envir_metadata.csv"))
#     for env_id in env_id_list:
#         env_df = pd.read_hdf(os.path.join(data_dir, "results", "gwas", env_id,
#                              f"{env_id}.gemma", f"{env_id}.gemma.gwas_df.h5"), key="df")
#         env_df = env_df[['ID', 'P_WALD']]
#         env_df = env_df.rename(columns={'P_WALD': env_id})
#         env_df[env_id] = -np.log10(env_df[env_id])
#         site_info_df = site_info_df.merge(env_df, on='ID', how='left')

#     from src.ld import prune_sites_by_ld_nested
#     from src.enrich import enrichment_permutation_analysis, simulate_ranked_sites, assign_fixed_foreground_labels
#     from tqdm.auto import tqdm

#     ld_window_size = 15000

#     pool_num = 40

#     env_id = "bio18"
#     context_df = site_info_df.copy()
#     context_df = context_df[['ID', 'CHROM', 'POS', 'MAF', env_id]]
#     context_df = context_df.sort_values(
#         by=env_id, ascending=False).reset_index(drop=True)

#     # 逐步放宽r2；保留此前所有位点，并将每个结果恢复为GEA排名顺序
#     r2_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0]
#     pruned_context_dfs = prune_sites_by_ld_nested(
#         context_df, input_vcf_file, r2_thresholds,
#         ld_window_size=ld_window_size, pool_num=pool_num,
#     )

#     # 以r2_threshold=0.5为基准，模拟不同enrichment_strength下的foreground分布，并计算top 5%中foreground的数量
#     # enrichment_strengths = [round(i, 2) for i in np.arange(0, 1.01, 0.05)]
#     enrichment_strengths = [0.0, 0.025, 0.05, 0.075,
#                             0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.50]

#     r2_threshold = 0.5
#     fg_fraction = 0.1
#     prune_context_df = pruned_context_dfs[r2_threshold]
#     baseline_fg_enrich_df = prune_context_df[['ID', 'CHROM', 'POS', ]].copy()
#     for enrichment_strength in tqdm(enrichment_strengths, desc="Simulating foreground enrichment"):
#         baseline_fg_enrich_df[enrichment_strength] = simulate_ranked_sites(len(
#             prune_context_df), fg_fraction=fg_fraction, enrichment_strength=enrichment_strength, seed=int(42 + 1000 * enrichment_strength))

#     for enrichment_strength in enrichment_strengths:
#         foreground_sum = baseline_fg_enrich_df[enrichment_strength].sum()
#         print(
#             f"Enrichment strength: {enrichment_strength}, Foreground sum: {foreground_sum}")
#         foreground_sum = baseline_fg_enrich_df.head(
#             int(prune_context_df.shape[0] * 0.05))[enrichment_strength].sum()
#         print(
#             f"Enrichment strength: {enrichment_strength}, Foreground sum in top 5%: {foreground_sum}")
#         foreground_sum = baseline_fg_enrich_df.tail(
#             int(prune_context_df.shape[0] * 0.05))[enrichment_strength].sum()
#         print(
#             f"Enrichment strength: {enrichment_strength}, Foreground sum in bottom 5%: {foreground_sum}")

#     # 基准标签不变；非基准位点按ID一次性抽样，所有阈值复用同一标签
#     prune_context_dfs_with_fg = assign_fixed_foreground_labels(
#         pruned_context_dfs, baseline_fg_enrich_df, enrichment_strengths,
#         fg_fraction=fg_fraction, seed=42,
#     )

#     stats = []

#     for r2_threshold in r2_thresholds:
#         print(f"R2 threshold: {r2_threshold}")
#         prune_context_df = prune_context_dfs_with_fg[r2_threshold]
#         for enrichment_strength in tqdm(enrichment_strengths, desc="Analyzing foreground enrichment"):
#             context_df = prune_context_df[[
#                 'ID', 'CHROM', 'POS', env_id, enrichment_strength]].copy()
#             context_df = context_df.rename(
#                 columns={enrichment_strength: 'foreground'})
#             top_GEA_percent = 0.05
#             context_df = context_df.sort_values(
#                 by=['CHROM', 'POS'], ascending=True).reset_index(drop=True)

#             results_json = enrichment_permutation_analysis(
#                 context_df, top_GEA_percent, [env_id], 40, 1000,
#                 stratify_id='all', show_progress=True, progress_interval=2.0,
#             )

#             obs_value = results_json['stat'][env_id]['obs_value']
#             pval = results_json['stat'][env_id]['pval']

#             print(
#                 f"Enrichment strength: {enrichment_strength}, Observed value: {obs_value}, P-value: {pval}")

#             stats.append((r2_threshold, enrichment_strength, obs_value, pval))

#     stats_df = pd.DataFrame(
#         stats, columns=['r2_threshold', 'enrichment_strength', 'obs_value', 'pval'])
#     stats_df.to_csv(os.path.join(data_dir, "results",
#                     "enrichment_simulation_results.csv"), index=False)
#     plot_enrichment_simulation_results(
#         stats_df, title=f"Enrichment simulation: {env_id}",
#         output_file=os.path.join(
#             data_dir, "results", "enrichment_simulation_results.png"),
#     )

#     ld_overlap_results = []

#     for r2_threshold in r2_thresholds:
#         print(f"R2 threshold: {r2_threshold}")
#         prune_context_df = prune_context_dfs_with_fg[r2_threshold]
#         for enrichment_strength in tqdm(enrichment_strengths, desc="Analyzing foreground enrichment"):
#             context_df = prune_context_df[[
#                 'ID', 'CHROM', 'POS', env_id, enrichment_strength]].copy()
#             context_df = context_df.rename(
#                 columns={enrichment_strength: 'foreground'})
#             top_GEA_percent = 0.05
#             context_df = context_df.sort_values(
#                 by=['CHROM', 'POS'], ascending=True).reset_index(drop=True)

#             ld_overlap = calculate_ld_overlap(
#                 context_df,
#                 input_vcf_file,
#                 env_id,
#                 top_percent=0.05,
#                 min_R2=0.5,
#                 ld_window_size=15000,
#                 pool_num=40
#             )

#             ld_overlap_results.append((r2_threshold, enrichment_strength, ld_overlap['O_fore'], ld_overlap['O_back'], ld_overlap[
#                                       'mean_overlap'], ld_overlap['harmonic_mean_overlap'], ld_overlap['weighted_jaccard_overlap']))

#     ld_overlap_results_df = pd.DataFrame(ld_overlap_results, columns=[
#                                          'r2_threshold', 'enrichment_strength', 'O_fore', 'O_back', 'mean_overlap', 'harmonic_mean_overlap', 'weighted_jaccard_overlap'])
#     ld_overlap_results_df.to_csv(os.path.join(data_dir, "results",
#                                               "ld_overlap_simulation_results.csv"), index=False)

#     # --------------------------------------------------

#     from src.ld import prune_sites_by_ld

#     env_id = "bio18"

#     context_df = site_info_df.copy()
#     context_df = context_df[
#         ['ID', 'CHROM', 'POS', 'MAF', env_id]
#     ].copy()

#     # GEA strongest -> weakest
#     context_df = (
#         context_df
#         .sort_values(by=env_id, ascending=False)
#         .reset_index(drop=True)
#     )

#     r2_threshold = 0.5
#     ld_window_size = 15000
#     pool_num = 40

#     seed = 42
#     fg_fraction = 0.1
#     enrichment_strength = 0.0

#     # --------------------------------------------------
#     # 1. Greedy LD pruning
#     # --------------------------------------------------

#     prune_context_df = prune_sites_by_ld(
#         context_df,
#         input_vcf_file,
#         ld_window_size=ld_window_size,
#         r2_threshold=r2_threshold,
#         pool_num=pool_num,
#         full_output=True,
#     )

#     keep_mask = prune_context_df['LD_PRUNING_KEEP'].astype(bool)
#     child_mask = ~keep_mask

#     # --------------------------------------------------
#     # 2. Simulate true enrichment on independent anchors
#     # --------------------------------------------------

#     anchor_foreground = simulate_ranked_sites(
#         n_sites=keep_mask.sum(),
#         fg_fraction=fg_fraction,
#         enrichment_strength=enrichment_strength,
#         seed=seed,
#     )

#     prune_context_df['foreground'] = False

#     prune_context_df.loc[
#         keep_mask,
#         'foreground'
#     ] = anchor_foreground

#     # --------------------------------------------------
#     # 3. Build child table
#     # --------------------------------------------------

#     child_df = prune_context_df.loc[
#         child_mask,
#         ['LD_PRUNED_BY', 'LD_PRUNED_R2']
#     ].copy()

#     # parent index -> parent FG/BG tag
#     parent_tag_dict = (
#         prune_context_df.loc[
#             keep_mask,
#             'foreground'
#         ]
#         .astype(bool)
#         .to_dict()
#     )

#     child_df['parent_foreground'] = (
#         child_df['LD_PRUNED_BY']
#         .map(parent_tag_dict)
#     )

#     if child_df['parent_foreground'].isna().any():
#         raise ValueError(
#             "Some child sites could not be mapped to their pruning parent."
#         )

#     # --------------------------------------------------
#     # 4. Scale LD strength
#     #
#     # Here use |r| = sqrt(r2)
#     # --------------------------------------------------

#     r = np.sqrt(
#         child_df['LD_PRUNED_R2'].to_numpy(dtype=float)
#     )

#     r_min = np.sqrt(r2_threshold)

#     # r_scale = (
#     #     (r - r_min)
#     #     / (1 - r_min)
#     # )

#     # r_scale = np.clip(r_scale, 0, 1)
#     r_scale = r

#     child_df['r_scale'] = r_scale

#     # --------------------------------------------------
#     # 5. Signed LD-label preference
#     #
#     # BG parent -> z > 0
#     # FG parent -> z < 0
#     # --------------------------------------------------

#     f = (
#         child_df['parent_foreground']
#         .astype(int)
#         .to_numpy()
#     )

#     z = (
#         1 - 2 * f
#     ) * r_scale

#     child_df['z'] = z

#     # --------------------------------------------------
#     # 6. Fixed Gumbel noise
#     # --------------------------------------------------

#     rng = np.random.default_rng(seed + 1)

#     gumbel = rng.gumbel(
#         size=len(child_df)
#     )

#     child_df['gumbel'] = gumbel

#     # --------------------------------------------------
#     # 7. LD-label coupling
#     # --------------------------------------------------

#     c = 0.1

#     if not -1 < c < 1:
#         raise ValueError("c must be between -1 and 1")

#     k = np.arctanh(c)

#     score = (
#         k * child_df['z'].to_numpy()
#         + child_df['gumbel'].to_numpy()
#     )

#     # --------------------------------------------------
#     # 8. Exactly preserve total FG fraction
#     # --------------------------------------------------

#     K_total = round(
#         len(prune_context_df) * fg_fraction
#     )

#     K_anchor = int(
#         prune_context_df.loc[
#             keep_mask,
#             'foreground'
#         ].sum()
#     )

#     K_child = K_total - K_anchor

#     if not 0 <= K_child <= len(child_df):
#         raise ValueError(
#             f"Invalid K_child={K_child}"
#         )

#     child_foreground = np.zeros(
#         len(child_df),
#         dtype=bool
#     )

#     if K_child > 0:
#         fg_pos = np.argpartition(
#             score,
#             -K_child
#         )[-K_child:]

#         child_foreground[fg_pos] = True

#     # --------------------------------------------------
#     # 9. Put child labels back
#     # --------------------------------------------------

#     prune_context_df.loc[
#         child_df.index,
#         'foreground'
#     ] = child_foreground

#     # check
#     assert prune_context_df['foreground'].sum() == K_total

#     # --------------------------------------------------
#     # 10. Check LD overlap
#     # --------------------------------------------------

#     ld_overlap = calculate_ld_overlap(
#         prune_context_df,
#         input_vcf_file,
#         env_id,
#         top_percent=0.05,
#         min_R2=0.5,
#         ld_window_size=15000,
#         pool_num=40
#     )

#     # --------------------------------------------------
#     # 11. Check LD between c
#     # -------------------------------------------------

#     c_list = [-0.9, -0.75, -0.5, -0.3, -0.2, -0.1, -0.05, -
#               0.01, 0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]
#     harmonic_ld_overlap_list = []

#     for c in tqdm(c_list, desc="Testing LD-label coupling"):
#         if not -1 < c < 1:
#             raise ValueError("c must be between -1 and 1")
#         k = np.arctanh(c)

#         score = (
#             k * child_df['z'].to_numpy()
#             + child_df['gumbel'].to_numpy()
#         )

#         K_total = round(
#             len(prune_context_df) * fg_fraction
#         )

#         K_anchor = int(
#             prune_context_df.loc[
#                 keep_mask,
#                 'foreground'
#             ].sum()
#         )

#         K_child = K_total - K_anchor

#         if not 0 <= K_child <= len(child_df):
#             raise ValueError(
#                 f"Invalid K_child={K_child}"
#             )

#         child_foreground = np.zeros(
#             len(child_df),
#             dtype=bool
#         )

#         if K_child > 0:
#             fg_pos = np.argpartition(
#                 score,
#                 -K_child
#             )[-K_child:]

#             child_foreground[fg_pos] = True

#         prune_context_df.loc[
#             child_df.index,
#             'foreground'
#         ] = child_foreground

#         ld_overlap = calculate_ld_overlap(
#             prune_context_df,
#             input_vcf_file,
#             env_id,
#             top_percent=0.05,
#             min_R2=0.5,
#             ld_window_size=15000,
#             pool_num=40
#         )

#         harmonic_ld_overlap_list.append(ld_overlap['harmonic_mean_overlap'])

#     import os
#     import pandas as pd
#     import numpy as np

#     site_info_df_h5 = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data/results/site_contexts.genome.maf0.05.h5"
#     input_vcf_file = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz"
#     site_info_df = pd.read_hdf(site_info_df_h5, key='df')
#     data_dir = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data"
#     # genome site level envGWAS results
#     env_id_list = ["bio18", "gsl", "al", "striga"]
#     env_metadata_df = pd.read_csv(os.path.join(
#         data_dir, "source", "envir_metadata.csv"))
#     for env_id in env_id_list:
#         env_df = pd.read_hdf(os.path.join(data_dir, "results", "gwas", env_id,
#                              f"{env_id}.gemma", f"{env_id}.gemma.gwas_df.h5"), key="df")
#         env_df = env_df[['ID', 'P_WALD']]
#         env_df = env_df.rename(columns={'P_WALD': env_id})
#         env_df[env_id] = -np.log10(env_df[env_id])
#         site_info_df = site_info_df.merge(env_df, on='ID', how='left')

#     from src.ld import prune_sites_by_ld_nested
#     from src.enrich import enrichment_permutation_analysis, simulate_ranked_sites, assign_fixed_foreground_labels
#     from tqdm.auto import tqdm

#     ld_window_size = 15000

#     pool_num = 40

#     env_id = "bio18"
#     context_df = site_info_df.copy()
#     context_df = context_df[['ID', 'CHROM', 'POS', 'MAF', env_id]]
#     context_df = context_df.sort_values(
#         by=env_id, ascending=False).reset_index(drop=True)

#     # --------------------------------------------------
#     # Parameters
#     # --------------------------------------------------

#     env_id = "bio18"

#     r2_threshold = 0.5
#     top_GEA_percent = 0.05
#     ld_window_size = 15000
#     pool_num = 40

#     fg_fraction = 0.1
#     enrichment_strength = 0.1

#     # Fixed forever: defines LD units
#     prune_seed = 12345 + 1

#     # Varies across simulation replicates
#     seed = 42 + 1

#     anchor_seed = seed
#     child_seed = seed + 1
#     contamination_seed = seed + 2

#     # --------------------------------------------------
#     # Prepare context
#     # --------------------------------------------------

#     context_df = site_info_df[
#         ['ID', 'CHROM', 'POS', 'MAF', env_id]
#     ].copy()

#     # --------------------------------------------------
#     # 1. Define LD units independently of GEA
#     # --------------------------------------------------

#     from src.ld import prune_sites_by_ld

#     prune_context_df = prune_sites_by_ld(
#         context_df.sample(
#             frac=1,
#             random_state=prune_seed,
#         ),
#         input_vcf_file,
#         ld_window_size=ld_window_size,
#         r2_threshold=r2_threshold,
#         pool_num=pool_num,
#         full_output=True,
#     )

#     keep_mask = (
#         prune_context_df['LD_PRUNING_KEEP']
#         .astype(bool)
#     )

#     child_mask = ~keep_mask

#     # --------------------------------------------------
#     # 2. Simulate latent enrichment on independent anchors
#     # --------------------------------------------------

#     gea_index = (
#         prune_context_df.loc[
#             keep_mask
#         ]
#         .sort_values(
#             by=env_id,
#             ascending=False,
#         )
#         .index
#     )

#     anchor_foreground = simulate_ranked_sites(
#         n_sites=len(gea_index),
#         fg_fraction=fg_fraction,
#         enrichment_strength=enrichment_strength,
#         seed=anchor_seed,
#     )

#     prune_context_df.loc[
#         gea_index,
#         'foreground'
#     ] = anchor_foreground

#     # --------------------------------------------------
#     # 3. Assign child tags from parent + LD
#     # --------------------------------------------------
#     from src.enrich import assign_child_tags_by_parent_ld, calculate_ld_overlap_from_vcf, enrichment_permutation_analysis, calculate_ld_overlap_from_csr, calculate_ld_overlap_from_ld_df

#     prune_context_df, child_df = (
#         assign_child_tags_by_parent_ld(
#             prune_context_df,
#             fg_fraction=fg_fraction,
#             seed=child_seed,
#             r2_threshold=r2_threshold,
#             stickiness=1,
#             # stickiness=5.0,
#         )
#     )

#     ld_overlap = calculate_ld_overlap_from_vcf(
#         context_df=prune_context_df,
#         input_vcf_file=input_vcf_file,
#         env_id=env_id,
#         top_percent=0.05,
#         min_R2=0.5,
#         ld_window_size=15000,
#         pool_num=40
#     )

#     harmonic_mean_overlap = ld_overlap['harmonic_mean_overlap']

#     results_json = enrichment_permutation_analysis(
#         prune_context_df, top_GEA_percent, [env_id], 40, 1000,
#         stratify_id='all', show_progress=True, progress_interval=2.0,
#     )

#     fe = results_json['stat'][env_id]['obs_value']
#     pval = results_json['stat'][env_id]['pval']

#     print(
#         f"Initial harmonic_mean_overlap: {harmonic_mean_overlap}, Observed value: {fe}, P-value: {pval}")

#     # --------------------------------------------------
#     # 4. 计算LD
#     # --------------------------------------------------

#     from src.ld import calculate_ld_matrix
#     ld_df = calculate_ld_matrix(
#         prune_context_df,
#         input_vcf_file,
#         ld_window_size=ld_window_size,
#         min_r2=0.5,
#         pool_num=pool_num)

#     labels = prune_context_df['foreground'].to_numpy(dtype=np.int8).copy()
#     movable_mask = (~prune_context_df["LD_PRUNING_KEEP"]).to_numpy(dtype=bool)

#     min_R2 = 0.5

#     ld_df = ld_df.copy()
#     ld_df["R2_scale"] = (
#         (ld_df["R2"] - min_R2)
#         / (1.0 - min_R2)
#     ).clip(0, 1)

#     # --------------------------------------------------
#     # 5. 计算C值： C值越大，说明FG/BG标签越混合，LD越高
#     # --------------------------------------------------

#     from src.simulate import LDCutSimulator

#     sim = LDCutSimulator(
#         pruned_context_df=prune_context_df,
#         ld_df=ld_df,
#         source_col="INDEX_1",
#         target_col="INDEX_2",
#         weight_col="R2_scale",       # 也可以传预先算好的 R2_scale
#         weight_dtype=np.float32,
#         verbose=True,
#     )

#     print("initial C:", sim.C)
#     print("initial C_fraction:", sim.C_fraction)
#     print("initial C_relative:", sim.C_relative)
#     print("Initial parent match:", sim.parent_match_fraction)
#     print(
#         "Initial parent mismatch relative:",
#         sim.parent_mismatch_relative,
#     )

#     # --------------------------------------------------
#     # 6. 随机swap策略，人为操控Q值
#     # --------------------------------------------------
#     sim.reset_labels()
#     targets = np.linspace(0, 1.5, 16)

#     labels_by_target, summary_df = (
#         sim.move_through_contamination_targets(
#             targets=targets,
#             direction="increase",
#             seed=contamination_seed,
#             max_swaps_per_target=1_000_000,
#             verbose_every=10_000,
#         )
#     )

#     for q in targets:

#         labels_target = labels_by_target[q]

#         # IMPORTANT:
#         # sim labels are stored in node-ID order: 0, 1, ..., N-1
#         simulated_context_df = (
#             prune_context_df
#             .sort_index()
#             .copy()
#         )

#         simulated_context_df['foreground'] = (
#             labels_target.astype(bool)
#         )

#         ld_overlap = calculate_ld_overlap_from_ld_df(
#             context_df=simulated_context_df,
#             ld_df=ld_df,
#             env_id=env_id,
#             top_percent=0.05,
#             min_R2=0.5,
#             source_col="INDEX_1",
#             target_col="INDEX_2",
#             r2_col="R2",
#             chunk_size=5_000_000,
#         )

#         harmonic_mean_overlap = (
#             ld_overlap['harmonic_mean_overlap']
#         )

#         results_json = enrichment_permutation_analysis(
#             simulated_context_df,
#             top_GEA_percent,
#             [env_id],
#             40,
#             1000,
#             stratify_id='all',
#             show_progress=True,
#             progress_interval=2.0,
#         )

#         fe = results_json['stat'][env_id]['obs_value']
#         pval = results_json['stat'][env_id]['pval']

#         print(
#             f"Q: {q}, "
#             f"harmonic_mean_overlap: {harmonic_mean_overlap}, "
#             f"Observed value: {fe}, "
#             f"P-value: {pval}"
#         )
