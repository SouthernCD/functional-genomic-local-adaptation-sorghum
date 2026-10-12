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


def site_contrast_runner(contrast_id, contrast_json, context_df, env_id_list, output_dir, permutations=1000, top_percent=0.05, pool_num=40, force_recompute=False):

    results_json_file = os.path.join(
        output_dir, f"{contrast_id}.enrichment.json.gz")
    reuse_result = not force_recompute and os.path.exists(results_json_file)
    if reuse_result:
        try:
            with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
                existing_results = json.load(f)
            reuse_result = (isinstance(existing_results, dict) and
                            set(existing_results) == set(['all'] + [
                                s for s in contrast_json.get('maf_stratify', []) if s != 'all']) and
                            all(isinstance(result, dict) and
                                result.get('permutation_scheme') == CHROMOSOME_PERMUTATION_SCHEME and
                                result.get('permutations') == permutations and
                                result.get('top_percent') == top_percent and
                                result.get('env_id_list') == list(env_id_list)
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
        temporary_file = f'{json_file}.tmp'
        with gzip.open(temporary_file, 'wt', encoding='utf-8') as f:
            json.dump(enrich_results_json, f, indent=4)
        os.replace(temporary_file, json_file)
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


def gene_contrast_runner(contrast_id, contrast_json, context_df, env_id_list, output_dir, permutations=1000, top_percent=0.05, seed=42, force_recompute=False):
    """Test gene-set overlap using chromosome-local shifts.

    Parameters
    ----------
    contrast_id : str
        Contrast ID used for the saved result filename.
    contrast_json : dict
        Gene target column and dataset definition.
    context_df : pandas.DataFrame
        Gene coordinates, boolean target labels and association scores. Each
        environment retains complete scores, labels and CHROM/START coordinates.
    env_id_list : sequence of str
        Environmental score columns in analysis order.
    output_dir : str
        Directory for the atomically saved enrichment JSON.gz.
    permutations : int
        Number of chromosome-local label shifts; default 1000.
    top_percent : float
        Fraction of strongest gene scores, including threshold ties; default 0.05.
    seed : int
        Random seed for reproducible shifts; default 42.
    force_recompute : bool
        Replace saved results after input changes when True. Otherwise reuse
        results with matching method, parameters, environments and target.

    Returns
    -------
    str
        JSON.gz path with observed/permutation Jaccard values, two-sided empirical
        P values and background counts. Sorting and filtering use private copies.
    """
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be in (0, 1]")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    env_id_list = list(env_id_list)
    target = contrast_json["target"]
    results_json_file = os.path.join(
        output_dir, f"{contrast_id}.enrichment.json.gz")

    # Old global-roll results must not be reused under the new null model.
    if not force_recompute and os.path.exists(results_json_file):
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
                target_mask, chrom_slices, seed=seed + perm_id * 10000), env_top_mask)
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

    os.makedirs(output_dir, exist_ok=True)
    temporary_file = f'{results_json_file}.tmp'
    with gzip.open(temporary_file, 'wt', encoding='utf-8') as f:
        json.dump({'all': results_json}, f, indent=4)
    os.replace(temporary_file, results_json_file)
    return results_json_file


def win_contrast_runner(contrast_id, contrast_json, context_df, trait_id_list, output_dir, permutations=1000, force_recompute=False):
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

    if not force_recompute and os.path.exists(results_json_file):
        try:
            with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
                existing = json.load(f)['all']
            if (existing.get('permutation_scheme') == WINDOW_PERMUTATION_SCHEME
                    and existing.get('permutations') == permutations
                    and set(trait_id_list).issubset(existing.get('trait_id_list', []))
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

    temporary_file = f'{results_json_file}.tmp'
    with gzip.open(temporary_file, 'wt', encoding='utf-8') as f:
        json.dump({'all': results_json}, f, indent=4)
    os.replace(temporary_file, results_json_file)
    return results_json_file


def site_jaccard_contrast_runner(contrast_id, contrast_json, context_df, trait_id_list, output_dir, permutations=1000, force_recompute=False):
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

    if not force_recompute and os.path.exists(results_json_file):
        try:
            with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
                existing = json.load(f)['all']
            if (existing.get('permutation_scheme') == SITE_JACCARD_PERMUTATION_SCHEME
                    and existing.get('permutations') == permutations
                    and set(trait_id_list).issubset(existing.get('trait_id_list', []))
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

    temporary_file = f'{results_json_file}.tmp'
    with gzip.open(temporary_file, 'wt', encoding='utf-8') as f:
        json.dump({'all': results_json}, f, indent=4)
    os.replace(temporary_file, results_json_file)
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

    if symmetric_xlim:
        for ax in axs:
            bound = max(abs(value) for value in ax.get_xlim())
            ax.set_xlim(-bound, bound)

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


def _add_enrichment_header(fig, axes, labels, title_fontsize=14,
                           enrichment_direction=True):
    """Place shared environmental titles and direction arrows above the panels."""
    from matplotlib.patches import FancyArrowPatch

    height = fig.get_size_inches()[1]
    top = axes[0].get_position().y1
    for ax, label in zip(axes, labels):
        position = ax.get_position()
        label = str(label).replace('Striga', r'$\mathit{Striga}$')
        fig.text((position.x0 + position.x1) / 2, top + 0.38 / height,
                 label, fontsize=title_fontsize, ha='center', va='bottom',
                 multialignment='center')
    if enrichment_direction:
        position = axes[-1].get_position()
        center = (position.x0 + position.x1) / 2
        width = position.width
        y = top + 0.92 / height
        for label, start, end in [
            ('depleted', center - 0.05 * width, center - 0.48 * width),
            ('enriched', center + 0.05 * width, center + 0.48 * width),
        ]:
            arrow = FancyArrowPatch((start, y), (end, y), transform=fig.transFigure,
                                    arrowstyle='-|>', mutation_scale=13,
                                    linewidth=1.8, color='black', clip_on=False)
            fig.add_artist(arrow)
            fig.text((start + end) / 2, y + 0.05 / height, label,
                     fontsize=title_fontsize, ha='center', va='bottom')


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
    apply_style=True,
    output_prefix=None,
    figure_title=None,
    figure_title_fontsize=15,
    figure_title_position='above_environments',
    subplot_spec=None,
    wspace=0.15,
    enrichment_direction=False,
    symmetric_xlim=False,
):
    """Plot multiple enrichment sets as rows across environmental panels.

    Parameters
    ----------
    set_results_json : dict
        Set IDs mapped to saved contrast-runner results with an 'all' stratum.
        Reads supplied results only; never runs enrichment or LD calculations.
    env_id_list : sequence of str
        Environmental variables in horizontal display order.
    env_metadata_df : pandas.DataFrame
        Unique env_id values and description labels for environmental titles.
    set_label_map : sequence of (str, str)
        Set IDs and labels in bottom-to-top row order.
    apply_style : bool
        Apply the shared style when True; False retains notebook initialization.
    output_prefix : str or None
        Optional PDF, PNG and actual plotted values CSV.gz destination prefix.
    figure_title : str or None
        Optional figure heading aligned to the left above environmental titles.
    figure_title_position : str
        'above_environments' keeps the figure heading above column titles;
        'below_environments' places column titles in a shared top header and
        the figure heading immediately above the scatter panels.
    figure_title_fontsize : float
        Figure heading size; default 15. Other styling options control axes,
        markers, stars and spacing without altering saved statistics.
    return_fig_ax : bool
        Return figure and axes when True; default True.
    show : bool
        Display the figure when True; default True.

    subplot_spec : matplotlib.gridspec.SubplotSpec or None
        Embed panels in a parent figure when supplied; retain parent layout.
    wspace : float
        Horizontal panel gap for embedded plots.
    enrichment_direction : bool
        Add depleted/enriched arrows to the shared header when the figure title
        is below environmental titles; default False.
    symmetric_xlim : bool
        Center each panel on zero using its plotted limits; default False.

    Returns
    -------
    tuple or None
        (figure, axes) when requested. Plot Z scores use the observation and
        all permutations jointly. Optional exports preserve raw values,
        transformed values, saved P values and final row/column order.
    """

    if apply_style:
        set_plot_style()

    # stratify_list.reverse()

    if figsize is None:
        figsize = (10, height_per_set * len(set_results_json))
    if subplot_spec is None:
        fig, axs = plt.subplots(figsize=figsize, nrows=1,
                                ncols=len(env_id_list), sharey=True)
        axs = np.atleast_1d(axs)
    else:
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        fig = subplot_spec.get_gridspec().figure
        grid = GridSpecFromSubplotSpec(1, len(env_id_list), subplot_spec=subplot_spec,
                                      wspace=wspace)
        axs = []
        for column in range(len(env_id_list)):
            ax = fig.add_subplot(grid[0, column], sharey=axs[0] if axs else None)
            if column:
                ax.tick_params(axis='y', labelleft=False)
            axs.append(ax)
        axs = np.asarray(axs)
        tight_layout = False

    y_values = {s: i for i, s in enumerate([s[0] for s in set_label_map])}

    plot_frames = []
    for env_column, (ax, env_id) in enumerate(zip(axs, env_id_list)):
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
            if output_prefix is not None:
                values = np.asarray([obs_value] + list(perm_values), dtype=float)
                plot_frames.append(pd.DataFrame({
                    'CONTRAST': s, 'LABEL': label, 'ROW': y_values[s],
                    'ENV_ID': env_id, 'ENV_COLUMN': env_column,
                    'POINT_TYPE': ['observed'] + ['permutation'] * len(perm_values),
                    'PERMUTATION_ID': [-1] + list(range(len(perm_values))),
                    'RAW_VALUE': values, 'Z_SCORE': np.concatenate(([obs_z], perm_z)),
                    'P_VALUE': p_value, 'SIGNIFICANT': p_value < significance_threshold,
                }))

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

    if figure_title_position not in ['above_environments', 'below_environments']:
        raise ValueError('Unknown figure_title_position.')
    if figure_title is not None and figure_title_position == 'below_environments':
        column_titles = [ax.get_title() for ax in axs]
        for ax in axs:
            ax.set_title('')
        figure_height = fig.get_size_inches()[1]
        if tight_layout:
            fig.tight_layout(rect=(0, 0, 1, 1 - 1.15 / figure_height))
        top = axs[0].get_position().y1
        fig.text(0.015, top + 0.055 / figure_height, figure_title,
                 fontsize=figure_title_fontsize, ha='left', va='bottom')
        _add_enrichment_header(fig, axs, column_titles, title_fontsize,
                               enrichment_direction=enrichment_direction)
    else:
        if figure_title is not None:
            fig.suptitle(figure_title, x=0.015, ha='left', y=0.99,
                         fontsize=figure_title_fontsize)
        if tight_layout:
            if figure_title is None:
                fig.tight_layout()
            else:
                fig.tight_layout(rect=(0, 0, 1, 0.88))
    if output_prefix is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
        fig.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
        fig.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
        temporary_csv = f'{output_prefix}.csv.gz.tmp'
        pd.concat(plot_frames, ignore_index=True).to_csv(
            temporary_csv, index=False, compression='gzip',
        )
        os.replace(temporary_csv, f'{output_prefix}.csv.gz')
        print(f'Figure and plot data: {output_prefix}.{{pdf,png,csv.gz}}')
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
    top_xlabel=r'Enrichment $z$-score',
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
    apply_style=True,
    subplot_spec=None,
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

    Set apply_style=False to retain the notebook initialization style.
    Supply subplot_spec to embed the two-row panel in a parent figure;
    embedded panels retain the parent layout and ignore standalone margins.

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
    if apply_style:
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
    if subplot_spec is None:
        fig = plt.figure(figsize=(figsize_per_panel[0] * n_env, figsize_per_panel[1]))
        gs = fig.add_gridspec(2, n_env, height_ratios=height_ratios,
                             wspace=wspace, hspace=hspace)
    else:
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        fig = subplot_spec.get_gridspec().figure
        gs = GridSpecFromSubplotSpec(
            2, n_env, subplot_spec=subplot_spec, height_ratios=height_ratios,
            wspace=wspace, hspace=hspace,
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
    if subplot_spec is None:
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


def plot_gene_enrichment_histogram(results_json_file, env_id_list, env_metadata_df,
                                    output_prefix=None, bins=30, figsize=None,
                                    title_width=20, title_fontsize=13,
                                    label_fontsize=11, annotation_fontsize=10,
                                    wspace=0.15):
    """Display a single gene-set enrichment as per-environment null histograms.

    Parameters
    ----------
    results_json_file : str
        Existing gene_contrast_runner JSON.gz. Read raw observed/permutation
        Jaccard overlaps and the saved two-sided empirical P values under 'all'.
        This function does not run permutations or recalculate significance.
    env_id_list : sequence of str
        Environmental IDs in display order.
    env_metadata_df : pandas.DataFrame
        Unique env_id values and description labels for the panel titles.
    output_prefix : str or None
        When provided, save PDF, 300 dpi PNG, actual observed/permutation values
        in .csv.gz and displayed histogram edges/densities in .histogram.csv.gz.
    bins : int, sequence or str
        NumPy/Matplotlib histogram bins; default 30 bins per environment.
        Each panel retains its own range; density normalization is unchanged.
    figsize : tuple or None
        Inches; default (2.5 * number of environments, 2.8).
    title_width : int
        Wrap descriptive panel titles at this width.
    title_fontsize, label_fontsize, annotation_fontsize : float
        Panel title, axis label and P-value font sizes.
    wspace : float
        Fractional horizontal gap between panels sharing the density axis.

    Returns
    -------
    dict
        figure, axes, plot_data, histogram_data and exported files. Raw Jaccard
        values define the x axis; null densities are light blue with black edges,
        and red dashed vertical lines indicate observed overlaps. Shared y limits
        leave 20% headroom above the tallest histogram. Titles italicize
        Striga. Shared plotting defaults are retained; does not call show.
    """
    import textwrap

    with gzip.open(results_json_file, 'rt', encoding='utf-8') as handle:
        result = json.load(handle)['all']
    env_id_list = list(env_id_list)
    metadata = env_metadata_df.set_index('env_id')
    if not metadata.index.is_unique:
        raise ValueError('Environmental IDs must be unique for histogram labels.')
    if figsize is None:
        figsize = (2.5 * len(env_id_list), 2.8)
    fig, axes = plt.subplots(1, len(env_id_list), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    plot_frames, histogram_frames = [], []
    for ax, env_id in zip(axes, env_id_list):
        observed = float(result['obs_values'][env_id])
        permutations = np.asarray(result['perm_values'][env_id], dtype=float)
        p_value = float(result['stat'][env_id]['pval'])
        densities, edges = np.histogram(permutations, bins=bins, density=True)
        ax.hist(permutations, bins=edges, density=True, color='lightblue',
                edgecolor='black', linewidth=0.7)
        ax.axvline(observed, color='red', linestyle='dashed', linewidth=1.8)
        significant = p_value < 0.05
        p_label = (fr'$\mathbfit{{P}}\mathbf{{ = {p_value:.3f}}}$ *' if significant
                   else fr'$P = {p_value:.3f}$')
        ax.text(0.05, 0.92, p_label, transform=ax.transAxes,
                fontsize=annotation_fontsize, color='black', ha='left', va='top',
                fontweight='bold' if significant else 'normal')
        title = '\n'.join(textwrap.wrap(str(metadata.loc[env_id, 'description']), width=title_width))
        if env_id in ['striga', 'striga_crop']:
            title = title.replace('Striga', r'$Striga$')
        ax.set_title(title, fontsize=title_fontsize, pad=12)
        ax.set_xlabel('Jaccard index', fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=label_fontsize - 1)
        plot_frames.append(pd.DataFrame({
            'env_id': env_id, 'description': metadata.loc[env_id, 'description'],
            'point_type': ['observed'] + ['permutation'] * len(permutations),
            'permutation_id': [-1] + list(range(len(permutations))),
            'JACCARD': np.concatenate(([observed], permutations)),
            'P_VALUE': p_value, 'P_VALUE_METHOD': 'two_sided_empirical',
        }))
        histogram_frames.append(pd.DataFrame({
            'env_id': env_id, 'BIN_LEFT': edges[:-1], 'BIN_RIGHT': edges[1:],
            'DENSITY': densities, 'COUNT': np.histogram(permutations, bins=edges)[0],
        }))
    # Set the shared limits only after all panels have contributed their peaks.
    peak_density = max(frame['DENSITY'].max() for frame in histogram_frames)
    axes[0].set_ylim(0, peak_density * 1.2 if peak_density > 0 else 1)
    axes[0].set_ylabel('Density', fontsize=label_fontsize)
    fig.tight_layout()
    fig.subplots_adjust(wspace=wspace)
    plot_data = pd.concat(plot_frames, ignore_index=True)
    histogram_data = pd.concat(histogram_frames, ignore_index=True)
    files = {}
    if output_prefix is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
        files = {kind: f'{output_prefix}.{extension}' for kind, extension in
                 [('pdf', 'pdf'), ('png', 'png'), ('csv', 'csv.gz'),
                  ('histogram_csv', 'histogram.csv.gz')]}
        fig.savefig(files['pdf'], facecolor='none', edgecolor='none', bbox_inches='tight')
        fig.savefig(files['png'], dpi=300, bbox_inches='tight')
        plot_data.to_csv(files['csv'], index=False, compression='gzip')
        histogram_data.to_csv(files['histogram_csv'], index=False, compression='gzip')
        print(f'Figure and data: {output_prefix}.{{pdf,png,csv.gz,histogram.csv.gz}}')
    return {'figure': fig, 'axes': axes, 'plot_data': plot_data,
            'histogram_data': histogram_data, 'files': files}


def plot_enrichment_section(result_files, contrast_label_map, env_id_list,
                             env_metadata_df, stratify_label_map, output_prefix,
                             figsize=None, block_height=3.8, block_gap=0.22,
                             section_title_fontsize=18, significance_threshold=0.05,
                             panel_params=None, letter_labels=True,
                             env_titles_first_only=True, enrichment_direction=True):
    """Stack multiple saved enrichment/LD contrasts into one section figure.

    Parameters
    ----------
    result_files : dict
        Mapping of contrast ID to 'enrichment' JSON.gz and 'overlap' pickle
        paths returned by the existing contrast runners. Reads saved results
        only; never computes enrichment, LD or permutations.
    contrast_label_map : sequence of (str, str)
        Contrast IDs and descriptive headings in vertical display order.
    env_id_list : sequence of str
        Environmental variables in horizontal display order.
    env_metadata_df : pandas.DataFrame
        Unique env_id and description columns for environmental titles.
    stratify_label_map : sequence of (str, str)
        Saved MAF strata and labels, ordered from bottom to top in each panel.
    output_prefix : str
        Prefix for PDF, PNG, enrichment CSV.gz and LD-overlap CSV.gz exports.
    figsize : tuple or None
        Inches; default width is 3 times the number of environments and height
        is block_height times the number of contrasts.
    block_height, block_gap : float
        Default height per contrast and vertical spacing between blocks.
    section_title_fontsize : float
        Contrast heading font size.
    significance_threshold : float
        Empirical P cutoff for stars, passed to the existing panel plotter.
    panel_params : dict or None
        Styling arguments for plot_enrichment_and_overlap, such as wspace,
        hspace, height_ratios and axis limits. Parent layout is managed here.

    letter_labels : bool
        Prefix multiple contrast headings with A., B., etc. A single contrast
        has no prefix. Supports up to 26 contrasts.
    env_titles_first_only : bool
        Place environmental titles in a shared header above the first block.
        False retains environmental titles on every block.
    enrichment_direction : bool
        Show leftward depleted and rightward enriched arrows above the final
        environmental column in the shared header; default True.

    Returns
    -------
    dict
        figure, per-contrast axes, raw/transformed enrichment plot_data,
        ld_plot_data and files. Z scores use the observed statistic and all
        permutations jointly, matching the existing plotter. Both figure-data
        tables preserve contrast and environment IDs, row order and raw values.
        Uses the notebook's existing style and does not call show.
    """
    import pickle
    from datetime import datetime
    from tqdm.auto import tqdm

    labels = list(contrast_label_map)
    env_id_list = list(env_id_list)
    if not labels or not env_id_list or len({cid for cid, _ in labels}) != len(labels):
        raise ValueError('Contrasts and environments must be nonempty; contrast IDs must be unique.')
    if not env_metadata_df['env_id'].is_unique:
        raise ValueError('Environmental metadata IDs must be unique.')
    if letter_labels and len(labels) > 26:
        raise ValueError('Lettered headings support up to 26 contrasts.')
    params = dict(panel_params or {})
    if any(key in params for key in ['subplot_spec', 'show', 'return_fig_ax', 'apply_style', 'significance_threshold']):
        raise ValueError('Panel parameters must not override section layout or significance settings.')
    if figsize is None:
        figsize = (3 * len(env_id_list), block_height * len(labels))
    print(f'{datetime.now():%Y-%m-%d %H:%M:%S} Plotting {len(labels)} enrichment contrasts')
    fig = plt.figure(figsize=figsize)
    figure_height = figsize[1]
    top_margin_inches = 1.15 if env_titles_first_only else 1.45
    outer = fig.add_gridspec(
        len(labels), 1, left=0.13, right=0.99, bottom=0.45 / figure_height,
        top=1 - top_margin_inches / figure_height,
        hspace=block_gap if env_titles_first_only else block_gap + 0.18,
    )
    axes, enrichment_frames, overlap_frames = {}, [], []
    for row, (contrast_id, label) in enumerate(tqdm(labels, desc='Enrichment panels')):
        with gzip.open(result_files[contrast_id]['enrichment'], 'rt') as handle:
            enrichment = json.load(handle)
        with open(result_files[contrast_id]['overlap'], 'rb') as handle:
            overlap = pickle.load(handle)
        heading_label = f'{chr(65 + row)}. {label}' if letter_labels and len(labels) > 1 else label
        _, panel_axes = plot_enrichment_and_overlap(
            enrichment, overlap, env_id_list, env_metadata_df, stratify_label_map,
            significance_threshold=significance_threshold, show=False,
            return_fig_ax=True, apply_style=False, subplot_spec=outer[row], **params,
        )
        top_position = panel_axes['top'][0].get_position()
        # Headings sit immediately above each scatter row, outside the MAF labels.
        heading_offset = 0.055 if env_titles_first_only else 0.65
        heading = fig.text(
            0.015, top_position.y1 + heading_offset / figure_height, heading_label,
            fontsize=section_title_fontsize, ha='left', va='bottom',
        )
        if env_titles_first_only:
            if row == 0:
                _add_enrichment_header(
                    fig, panel_axes['top'], [ax.get_title() for ax in panel_axes['top']],
                    params.get('title_fontsize', 14), enrichment_direction,
                )
            for ax in panel_axes['top']:
                ax.set_title('')
        axes[contrast_id] = {'heading': heading, **panel_axes}
        for stratum_row, (stratum, _) in enumerate(stratify_label_map):
            for env_column, env_id in enumerate(env_id_list):
                result = enrichment[stratum]
                values = np.asarray([result['obs_values'][env_id]]
                                    + list(result['perm_values'][env_id]), dtype=float)
                p_value = result['stat'][env_id]['pval']
                enrichment_frames.append(pd.DataFrame({
                    'CONTRAST': contrast_id, 'LABEL': label, 'PANEL_TITLE': heading_label, 'CONTRAST_ROW': row,
                    'STRATUM': stratum, 'STRATUM_ROW': stratum_row,
                    'ENV_ID': env_id, 'ENV_COLUMN': env_column,
                    'POINT_TYPE': ['observed'] + ['permutation'] * (len(values) - 1),
                    'PERMUTATION_ID': [-1] + list(range(len(values) - 1)),
                    'SCORE_RATIO': values, 'Z_SCORE': zscore(values), 'P_VALUE': p_value,
                    'SIGNIFICANT': p_value < significance_threshold,
                }))
        for env_column, env_id in enumerate(env_id_list):
            for kind in ['overlap_df', 'random_overlap_df']:
                frame = overlap[env_id][kind].copy()
                frame['CONTRAST'], frame['LABEL'], frame['CONTRAST_ROW'] = contrast_id, label, row
                frame['PANEL_TITLE'] = heading_label
                frame['ENV_ID'], frame['ENV_COLUMN'], frame['POINT_TYPE'] = env_id, env_column, kind
                start_col, end_col = params.get('start_col', 'gea_rank_start'), params.get('end_col', 'gea_rank_end')
                frame['BIN_CENTER'] = (frame[start_col] + frame[end_col]) / 2
                overlap_frames.append(frame)
    plot_data = pd.concat(enrichment_frames, ignore_index=True)
    ld_plot_data = pd.concat(overlap_frames, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    files = {'pdf': f'{output_prefix}.pdf', 'png': f'{output_prefix}.png',
             'enrichment_csv': f'{output_prefix}.enrichment.csv.gz',
             'ld_overlap_csv': f'{output_prefix}.ld_overlap.csv.gz'}
    fig.savefig(files['pdf'], bbox_inches='tight')
    fig.savefig(files['png'], dpi=300, bbox_inches='tight')
    for key, frame in [('enrichment_csv', plot_data), ('ld_overlap_csv', ld_plot_data)]:
        temporary_csv = f'{files[key]}.tmp'
        frame.to_csv(temporary_csv, index=False, compression='gzip')
        os.replace(temporary_csv, files[key])
    print(f'{datetime.now():%Y-%m-%d %H:%M:%S} Saved {output_prefix}')
    return {'figure': fig, 'axes': axes, 'plot_data': plot_data,
            'ld_plot_data': ld_plot_data, 'files': files}


def summarize_site_contrasts(context_df, contrasts_settings, contrast_ids, output_csv=None):
    """Count foreground and comparison sites for configured MAF strata.

    Parameters
    ----------
    context_df : pandas.DataFrame
        Site annotations with unique ID values and boolean context/MAF columns.
        Counts describe annotation membership, not per-environment nonmissing
        association scores. All supplied contrasts must use this dataset.
    contrasts_settings : dict
        Foreground/background expressions and optional maf_stratify definitions.
        Direct contrasts require disjoint foreground and background sets.
    contrast_ids : sequence of str
        Contrasts in summary order.
    output_csv : str or None
        Optional atomic CSV.gz export of the count table.

    Returns
    -------
    pandas.DataFrame
        CONTRAST, STRATUM, N_FOREGROUND, N_BACKGROUND, N_TESTED, FOREGROUND
        and BACKGROUND. For nested backgrounds, comparison counts exclude the
        foreground, matching site_contrast_runner. Empty groups retain zero
        counts so they can be inspected before running permutations.
    """
    from datetime import datetime
    from tqdm.auto import tqdm

    if context_df['ID'].isna().any() or not context_df['ID'].is_unique:
        raise ValueError('Site IDs must be nonmissing and unique for counts.')
    print(f'[{datetime.now():%H:%M:%S}] Summarizing contrast site counts')
    rows = []
    for contrast_id in tqdm(contrast_ids, desc='Contrast site counts'):
        contrast = contrasts_settings[contrast_id]
        foreground = filter_df_by_set_expr(context_df, contrast['foreground'])
        background = filter_df_by_set_expr(context_df, contrast['background'])
        if contrast.get('contrast_type') == 'direct' and (foreground & background).any():
            raise ValueError(f'{contrast_id}: direct comparison sets overlap.')
        comparison = background & ~foreground
        strata = ['all'] + [s for s in contrast.get('maf_stratify', []) if s != 'all']
        for stratum in strata:
            mask = np.ones(len(context_df), dtype=bool) if stratum == 'all' else context_df[stratum]
            n_fore = int((foreground & mask).sum())
            n_back = int((comparison & mask).sum())
            rows.append({'CONTRAST': contrast_id, 'STRATUM': stratum,
                         'N_FOREGROUND': n_fore, 'N_BACKGROUND': n_back,
                         'N_TESTED': n_fore + n_back,
                         'FOREGROUND': contrast['foreground'], 'BACKGROUND': contrast['background']})
    counts = pd.DataFrame(rows)
    if output_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        temporary_csv = f'{output_csv}.tmp'
        counts.to_csv(temporary_csv, index=False, compression='gzip')
        os.replace(temporary_csv, output_csv)
    return counts


def plot_set_enrichment_section(result_files, group_label_map, set_label_maps,
                                 env_id_list, env_metadata_df, output_prefix,
                                 figsize=(12, 8), block_gap=0.5,
                                 section_title_fontsize=15,
                                 significance_threshold=0.05,
                                 value_column='RAW_VALUE', panel_params=None,
                                 enrichment_direction=True, left=0.25,
                                 heading_x=0.015, hypothesis_groups=None,
                                 hypothesis_line_x=0.07, hypothesis_label_x=0.03,
                                 hypothesis_fontsize=12, hypothesis_linewidth=3,
                                 hypothesis_bottom_offset=-0.18, header_labels=None,
                                 hypothesis_fontweight='normal'):
    """Stack groups of enrichment sets without MAF or LD panels.

    Parameters
    ----------
    result_files : dict
        Set IDs mapped to saved contrast-runner JSON.gz paths. No statistical
        calculations are launched; only plotting transformations are applied.
    group_label_map : sequence of (str, str)
        Group IDs and headings in top-to-bottom order; headings receive letters
        when more than one group is supplied.
    set_label_maps : dict
        Each group mapped to (set ID, label) pairs in top-to-bottom display order.
    env_id_list : sequence of str
        Environmental variables in left-to-right display order.
    env_metadata_df : pandas.DataFrame
        Unique env_id and description columns for the shared top header.
    output_prefix : str
        Destination prefix for PDF, PNG and actual plotted-value CSV.gz.
    figsize : tuple
        Figure dimensions in inches.
    block_gap : float
        Vertical gap relative to group height.
    section_title_fontsize : float
        Font size for lettered group headings.
    significance_threshold : float
        Saved P-value cutoff for stars; default 0.05.
    value_column : str or dict
        Raw-statistic column name, or group-to-column mapping for mixed site
        ('SCORE_RATIO') and gene ('JACCARD') panels.
    panel_params : dict or None
        Styling arguments for plot_set_enrichment, excluding layout/export
        controls. Existing notebook plotting defaults are retained.
    enrichment_direction : bool
        Show depleted/enriched arrows above the final environmental column;
        default True. Shared titles italicize Striga.
    left, heading_x : float
        Figure-relative panel left margin and heading position.
    hypothesis_groups : sequence of dict or None
        Optional vertical brackets with label, color and group_ids in contiguous
        display order. Brackets are annotations, not statistical comparisons.
    hypothesis_line_x, hypothesis_label_x : float
        Figure-relative positions of vertical lines and rotated labels.
    hypothesis_fontsize, hypothesis_linewidth : float
        Font size for hypothesis labels and line width in points.
    hypothesis_bottom_offset : float
        Bottom endpoint offset from the final panel bottom, in inches.
    header_labels : sequence of str or None
        Optional environmental header labels in column order; preserves supplied
        line breaks without altering environmental metadata.
    hypothesis_fontweight : str
        Font weight for the rotated hypothesis labels; default normal.

    Returns
    -------
    dict
        figure, per-group axes, plot_data and exported files. Raw statistics,
        Z scores, saved P values and final row/column order are preserved.
        Per-environment background counts are exported when present in the
        input results. Environmental titles appear once, above all groups.
    """
    from datetime import datetime
    from matplotlib.lines import Line2D
    from tqdm.auto import tqdm

    groups, env_id_list = list(group_label_map), list(env_id_list)
    if not groups or not env_id_list or len(groups) > 26:
        raise ValueError('Require one to 26 groups and at least one environment.')
    params = dict(panel_params or {})
    controlled = {'subplot_spec', 'show', 'return_fig_ax', 'apply_style',
                  'output_prefix', 'figure_title', 'significance_threshold', 'xlabel'}
    if controlled & params.keys():
        raise ValueError('Panel parameters cannot override group layout or exports.')
    params.setdefault('xlabel_fontsize', 10)
    print(f'[{datetime.now():%H:%M:%S}] Plotting grouped enrichment sets')
    fig = plt.figure(figsize=figsize)
    height = figsize[1]
    outer = fig.add_gridspec(len(groups), 1, left=left, right=0.99,
                            top=1 - 1.15 / height, bottom=0.55 / height,
                            hspace=block_gap,
                            height_ratios=[len(set_label_maps[key]) for key, _ in groups])
    axes, frames = {}, []
    for row, (group_id, title) in enumerate(tqdm(groups, desc='Enrichment groups')):
        labels = list(set_label_maps[group_id])
        results = {}
        for set_id, _ in labels:
            with gzip.open(result_files[set_id], 'rt') as handle:
                results[set_id] = json.load(handle)
        _, panel_axes = plot_set_enrichment(
            results, env_id_list, env_metadata_df, list(reversed(labels)),
            significance_threshold=significance_threshold, show=False,
            apply_style=False, subplot_spec=outer[row],
            xlabel=r'Enrichment $z$-score' if row == len(groups) - 1 else None,
            **params,
        )
        top = panel_axes[0].get_position().y1
        heading_label = f'{chr(65 + row)}. {title}' if len(groups) > 1 else title
        heading = fig.text(heading_x, top + 0.055 / height, heading_label,
                           fontsize=section_title_fontsize, ha='left', va='bottom')
        if row == 0:
            _add_enrichment_header(
                fig, panel_axes, (header_labels if header_labels is not None
                                  else [ax.get_title() for ax in panel_axes]),
                params.get('title_fontsize', 14), enrichment_direction,
            )
        for column, (ax, env_id) in enumerate(zip(panel_axes, env_id_list)):
            ax.set_title('')
            for set_row, (set_id, label) in enumerate(labels):
                result = results[set_id]['all']
                values = np.asarray([result['obs_values'][env_id]]
                                    + list(result['perm_values'][env_id]), dtype=float)
                p_value = result['stat'][env_id]['pval']
                frame = pd.DataFrame({
                    'GROUP': group_id, 'GROUP_ROW': row, 'PANEL_TITLE': heading_label,
                    'CONTRAST': set_id, 'LABEL': label, 'SET_ROW': set_row,
                    'PLOT_Y': len(labels) - 1 - set_row, 'ENV_ID': env_id, 'ENV_COLUMN': column,
                    'POINT_TYPE': ['observed'] + ['permutation'] * (len(values) - 1),
                    'PERMUTATION_ID': [-1] + list(range(len(values) - 1)),
                    (value_column[group_id] if isinstance(value_column, dict) else value_column): values,
                    'STATISTIC': value_column[group_id] if isinstance(value_column, dict) else value_column,
                    'STRATUM': 'all', 'Z_SCORE': zscore(values),
                    'P_VALUE': p_value, 'SIGNIFICANT': p_value < significance_threshold,
                })
                info = result.get('background_info', {}).get(env_id, {})
                for source, target in [('n_genes', 'N_BACKGROUND'), ('n_target', 'N_TARGET'), ('n_top', 'N_TOP')]:
                    if source in info:
                        frame[target] = info[source]
                frames.append(frame)
        axes[group_id] = {'heading': heading, 'panels': panel_axes}
    # Bracket selected contiguous groups after all axes positions are known.
    group_ids = [key for key, _ in groups]
    for bracket in hypothesis_groups or []:
        selected = list(bracket['group_ids'])
        positions = [group_ids.index(key) for key in selected]
        if not positions or positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError('Hypothesis groups must follow contiguous display order.')
        first = axes[selected[0]]['panels'][0].get_position()
        last = axes[selected[-1]]['panels'][0].get_position()
        top = first.y1 + 0.28 / height
        bottom = last.y0 + hypothesis_bottom_offset / height
        fig.add_artist(Line2D([hypothesis_line_x] * 2, [bottom, top],
                                 transform=fig.transFigure, color=bracket['color'],
                                 linewidth=hypothesis_linewidth, solid_capstyle='butt', clip_on=False))
        fig.text(hypothesis_label_x, (top + bottom) / 2, bracket['label'],
                 rotation=90, fontsize=hypothesis_fontsize, fontweight=hypothesis_fontweight,
                 ha='center', va='center')
    plot_data = pd.concat(frames, ignore_index=True)
    if hypothesis_groups:
        hypothesis_labels = {key: bracket['label'] for bracket in hypothesis_groups
                             for key in bracket['group_ids']}
        plot_data['HYPOTHESIS'] = plot_data['GROUP'].map(hypothesis_labels)
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    files = {kind: f'{output_prefix}.{extension}' for kind, extension in
             [('pdf', 'pdf'), ('png', 'png'), ('csv', 'csv.gz')]}
    fig.savefig(files['pdf'], bbox_inches='tight')
    fig.savefig(files['png'], dpi=300, bbox_inches='tight')
    temporary_csv = f'{files["csv"]}.tmp'
    plot_data.to_csv(temporary_csv, index=False, compression='gzip')
    os.replace(temporary_csv, files['csv'])
    print(f'[{datetime.now():%H:%M:%S}] Saved {output_prefix}')
    return {'figure': fig, 'axes': axes, 'plot_data': plot_data, 'files': files}


def prepare_low_tail_enrichment(context_h5, contrasts_settings, contrast_ids,
                                trait_id, output_dir, summary_csv,
                                lower_fraction=0.05, context_threshold=None,
                                permutations=1000, force_recompute=False):
    """Summarize annotation overlap with the lower tail of a numeric trait.

    Parameters
    ----------
    context_h5 : str
        Existing HDF table (key 'df'). Window annotations are numeric coverage
        fractions; site annotations are boolean and retain the table's MAF.
    contrasts_settings : dict
        Existing foreground/background definitions. Never modified in place.
    contrast_ids : sequence of str
        Requested contrasts, sharing one background and dataset.
    trait_id : str
        Numeric trait column, such as PI or MAF. The lower-tail threshold is
        calculated once across the common tested background, including ties.
    output_dir, summary_csv : str
        Per-contrast JSON.gz directory and small summary CSV.gz destination.
    lower_fraction : float
        Lower-tail fraction; default 0.05.
    context_threshold : float or None
        For windows, convert annotation fractions > this value to membership.
        None retains existing boolean site labels. Numeric traits stay intact.
    permutations : int
        Number of chromosome-local shifts, default 1000; seeds follow the
        existing runners (0 through permutations - 1).
    force_recompute : bool
        Recompute statistics after input/definition changes when True. Legacy
        results assume the original lower_fraction=0.05 and window threshold
        0.75; compatible stored counts and methods are checked before reuse.

    Returns
    -------
    pandas.DataFrame
        Threshold, group counts, lower-tail proportions, foreground-to-comparison
        risk ratio, observed Jaccard and empirical P for each contrast. Existing
        result JSON.gz files are reused when compatible. PI is never recalculated.
    """
    from datetime import datetime
    from tqdm.auto import tqdm

    if not 0 < lower_fraction < 1:
        raise ValueError('lower_fraction must be between zero and one.')
    contrasts = [dict(contrasts_settings[key]) for key in contrast_ids]
    if not contrasts or len({c['background'] for c in contrasts}) != 1 or len({c['dataset'] for c in contrasts}) != 1:
        raise ValueError('Contrasts must share one background and dataset.')
    print(f'[{datetime.now():%H:%M:%S}] Preparing lower-tail {trait_id} enrichment')
    data = pd.read_hdf(context_h5, key='df').copy()
    if 'ID' not in data:
        data['ID'] = data['CHROM'] + ':' + data['START'].astype(str) + '-' + data['END'].astype(str)
    if data['ID'].isna().any() or not data['ID'].is_unique:
        raise ValueError('Context IDs must be nonmissing and unique.')
    window_mode = context_threshold is not None
    if window_mode:
        statistics = {'ID', 'CHROM', 'START', 'END', 'VARIANT_COUNT', 'SNP_COUNT', 'PI', 'TajimaD', 'meanCmPerMb'}
        for column in data.columns.difference(list(statistics)):
            data[column] = data[column] > context_threshold
        data['genome'] = True
    else:
        data[contrasts[0]['dataset']] = True
    numeric = pd.to_numeric(data[trait_id], errors='raise')
    background = filter_df_by_set_expr(data, contrasts[0]['background'])
    valid = background & numeric.notna()
    if not valid.any() or not np.isfinite(numeric[valid]).all():
        raise ValueError('The common background must have finite trait values.')
    cutoff = float(numeric[valid].quantile(lower_fraction))
    data[trait_id] = pd.Series(np.nan, index=data.index, dtype=object)
    data.loc[numeric.notna(), trait_id] = numeric[numeric.notna()] <= cutoff
    runner = win_contrast_runner if window_mode else site_jaccard_contrast_runner
    scheme = WINDOW_PERMUTATION_SCHEME if window_mode else SITE_JACCARD_PERMUTATION_SCHEME
    definition = dict(lower_fraction=lower_fraction, context_threshold=context_threshold,
                      background=contrasts[0]['background'], dataset=contrasts[0]['dataset'],
                      trait_id=trait_id, threshold=cutoff)
    rows = []
    for contrast_id, contrast in tqdm(list(zip(contrast_ids, contrasts)), desc=f'{trait_id} contrasts'):
        contrast['target'] = 'foreground'
        foreground = filter_df_by_set_expr(data, contrast['foreground'])
        if (foreground & ~background).any():
            raise ValueError(f'{contrast_id}: foreground extends outside the common background.')
        fore, back = valid & foreground, valid & ~foreground
        if not fore.any() or not back.any():
            raise ValueError(f'{contrast_id}: a comparison group is empty.')
        n_fore, n_back = int(fore.sum()), int(back.sum())
        low = numeric <= cutoff
        low_fore, low_back = int((fore & low).sum()), int((back & low).sum())
        result_dir = f'{output_dir}/{contrast_id}'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/{contrast_id}.enrichment.json.gz'
        reuse = False
        if not force_recompute and os.path.exists(result_file):
            with gzip.open(result_file, 'rt') as handle:
                existing = json.load(handle)['all']
            info = existing.get('background_info', {}).get(trait_id, {})
            legacy = 'low_tail_definition' not in existing
            definition_ok = (lower_fraction == 0.05 and context_threshold in [None, 0.75]) if legacy else existing['low_tail_definition'] == definition
            reuse = (definition_ok and existing.get('permutation_scheme') == scheme
                     and existing.get('permutations') == permutations
                     and existing.get('target') == 'foreground'
                     and trait_id in existing.get('trait_id_list', [])
                     and info.get('n_windows' if window_mode else 'n_sites') == int(valid.sum())
                     and info.get('n_target') == n_fore
                     and info.get('n_trait') == int((valid & low).sum()))
            if reuse:
                print(f'{contrast_id}: reuse {result_file}' + (' (legacy threshold definition assumed)' if legacy else ''))
        if not reuse:
            contexts = build_enrich_data_df(data, contrast, [trait_id])
            runner(contrast_id, contrast, contexts, [trait_id], result_dir,
                   permutations=permutations, force_recompute=True)
            with gzip.open(result_file, 'rt') as handle:
                payload = json.load(handle)
            payload['all']['low_tail_definition'] = definition
            temporary = f'{result_file}.tmp'
            with gzip.open(temporary, 'wt') as handle:
                json.dump(payload, handle, indent=4)
            os.replace(temporary, result_file)
            existing = payload['all']
        fore_fraction, back_fraction = low_fore / n_fore, low_back / n_back
        rows.append(dict(CONTRAST=contrast_id, TRAIT=trait_id, BACKGROUND=contrast['background'],
                         THRESHOLD=cutoff, LOWER_FRACTION=lower_fraction,
                         N_FOREGROUND=n_fore, N_COMPARISON=n_back, N_LOW_FOREGROUND=low_fore,
                         N_LOW_COMPARISON=low_back, FOREGROUND_LOW_FRACTION=fore_fraction,
                         COMPARISON_LOW_FRACTION=back_fraction,
                         RISK_RATIO=fore_fraction / back_fraction if back_fraction else np.nan,
                         JACCARD=existing['obs_values'][trait_id], P_VALUE=existing['stat'][trait_id]['pval'],
                         RESULT_FILE=result_file))
    summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv)), exist_ok=True)
    temporary = f'{summary_csv}.tmp'
    summary.to_csv(temporary, index=False, compression='gzip')
    os.replace(temporary, summary_csv)
    return summary


def plot_low_tail_enrichment(panels, output_prefix, figsize=(11, 5.5),
                            title_fontsize=14, label_fontsize=12, tick_fontsize=10,
                            panel_letter_fontsize=20, wspace=0.9):
    """Display saved lower-tail enrichment results in horizontal panels.

    Parameters
    ----------
    panels : sequence of dict
        Each panel supplies trait_id, title, definition, background, labels in
        top-to-bottom order, result_files and summary_csv. Reads saved results
        only. No statistics, genotypes or permutations are calculated.
    output_prefix : str
        PDF, PNG and actual plotted-value CSV.gz destination prefix.
    figsize : tuple
        Canvas dimensions in inches.
    title_fontsize, label_fontsize, tick_fontsize, panel_letter_fontsize : float
        Typography sizes in points; retain the notebook's initialized style.
    wspace : float
        Horizontal panel gap allowing independent annotation row labels.

    Returns
    -------
    dict
        figure, axes, plot_data and files. Exports raw Jaccard values, joint
        observed/permutation Z scores, empirical P values, final row order and
        group proportions/risk ratios from the saved summaries.
    """
    from datetime import datetime
    from tqdm.auto import tqdm

    print(f'[{datetime.now():%H:%M:%S}] Plotting lower-tail enrichment')
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(1, len(panels), left=0.19, right=0.98,
                           bottom=0.16, top=0.57, wspace=wspace)
    axes, frames = [], []
    for index, panel in enumerate(tqdm(panels, desc='Lower-tail panels')):
        trait = panel['trait_id']
        summary = pd.read_csv(panel['summary_csv']).set_index('CONTRAST')
        results = {}
        for contrast_id, _ in panel['labels']:
            with gzip.open(panel['result_files'][contrast_id], 'rt') as handle:
                results[contrast_id] = json.load(handle)
        metadata = pd.DataFrame([dict(env_id=trait, description=panel['title'])])
        _, panel_axes = plot_set_enrichment(
            results, [trait], metadata, list(reversed(panel['labels'])),
            subplot_spec=grid[index], apply_style=False, show=False,
            xlabel=r'Enrichment $z$-score', xlabel_fontsize=label_fontsize,
            set_tick_fontsize=label_fontsize, tick_fontsize=tick_fontsize,
            symmetric_xlim=True, ylim_pad=(0.5, 0.5),
        )
        ax = panel_axes[0]
        ax.set_title('')
        position = ax.get_position()
        center, top = (position.x0 + position.x1) / 2, position.y1
        fig.text(center, top + 0.035, panel['title'], fontsize=title_fontsize,
                 ha='center', va='bottom')
        fig.text(center, top + 0.105, f"{panel['definition']}\nBackground: {panel['background']}",
                 fontsize=tick_fontsize, ha='center', va='bottom')
        _add_enrichment_header(fig, [ax], [''], title_fontsize, True)
        fig.text(position.x0 - 0.15, top + 0.025, chr(65 + index),
                 fontsize=panel_letter_fontsize, ha='left', va='bottom')
        for row, (contrast_id, label) in enumerate(panel['labels']):
            result = results[contrast_id]['all']
            values = np.asarray([result['obs_values'][trait]] + result['perm_values'][trait], dtype=float)
            frame = pd.DataFrame(dict(PANEL=chr(65 + index), TRAIT=trait, LABEL=label,
                                     CONTRAST=contrast_id, ROW=row, PLOT_Y=len(panel['labels']) - 1 - row,
                                     POINT_TYPE=['observed'] + ['permutation'] * (len(values) - 1),
                                     PERMUTATION_ID=[-1] + list(range(len(values) - 1)),
                                     JACCARD=values, Z_SCORE=zscore(values),
                                     P_VALUE=result['stat'][trait]['pval']))
            for column, value in summary.loc[contrast_id].items():
                if column not in frame and column != 'RESULT_FILE':
                    frame[column] = value
            frame['PANEL_TITLE'], frame['DEFINITION'], frame['BACKGROUND_LABEL'] = panel['title'], panel['definition'], panel['background']
            frames.append(frame)
        axes.append(ax)
    data = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    files = {key: f'{output_prefix}.{ext}' for key, ext in [('pdf', 'pdf'), ('png', 'png'), ('csv', 'csv.gz')]}
    fig.savefig(files['pdf'], bbox_inches='tight')
    fig.savefig(files['png'], dpi=300, bbox_inches='tight')
    temporary = f'{files["csv"]}.tmp'
    data.to_csv(temporary, index=False, compression='gzip')
    os.replace(temporary, files['csv'])
    return dict(figure=fig, axes=axes, plot_data=data, files=files)
