import numpy as np
import pandas as pd
import warnings
import multiprocessing as mp
from multiprocessing.util import Finalize
from collections import deque, OrderedDict
from operator import index
from time import perf_counter
from queue import Queue, Empty
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import pickle
from .plot import set_plot_style
from .vcf import VCF, get_genotype_matrix_from_vcf_by_pos_list
from .enrich import build_enrich_data_df
warnings.filterwarnings('ignore')


def calculate_LD(query_genotype_matrix, subject_genotype_matrix, query_pos_list, subject_pos_list):
    """
    Calculate LD between each row of query_genotype_matrix and each row of subject_genotype_matrix.

    # 示例用法
    query_genotype_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 1]])
    subject_genotype_matrix = np.array([[0, 0, 1, 1],
                                        [0, 1, 0, 1],
                                        [0, 0, 0, 1],
                                        [0, 1, 1, 1],
                                        [0, 0, 1, 1]])    
    """
    # 计算相关系数矩阵
    combined_matrix = np.vstack(
        (query_genotype_matrix, subject_genotype_matrix))
    correlation_matrix = np.corrcoef(combined_matrix)

    # 提取所需的相关系数
    n_query = query_genotype_matrix.shape[0]
    n_subject = subject_genotype_matrix.shape[0]

    # 相关系数矩阵的前 n_query 行和后 n_subject 列对应的子矩阵
    result_matrix = correlation_matrix[:n_query, n_query:n_query + n_subject]

    ld_matrix = result_matrix ** 2

    ld_df = pd.DataFrame(ld_matrix, index=query_pos_list,
                         columns=subject_pos_list)
    return ld_df


def _empty_link_pairs():
    return pd.DataFrame({
        'CHROM': pd.Series(dtype=object),
        'FOREGROUND_POS': pd.Series(dtype=np.int64),
        'BACKGROUND_POS': pd.Series(dtype=np.int64),
        'R2': pd.Series(dtype=np.float64),
    })


def _prepare_ld_chromosomes(site_info_df):
    required = ['CHROM', 'POS', 'foreground']
    missing = [column for column in required if column not in site_info_df]
    if missing:
        raise ValueError(f"Missing LD input columns: {missing}")
    if site_info_df.empty:
        return []
    if site_info_df[required].isna().any().any():
        raise ValueError("LD input columns must not contain missing values")
    if not pd.api.types.is_integer_dtype(site_info_df['POS']) or (site_info_df['POS'] < 1).any():
        raise ValueError("POS must contain positive integer coordinates")
    if not pd.api.types.is_bool_dtype(site_info_df['foreground']):
        raise ValueError("foreground must be a boolean column")

    positions = site_info_df['POS'].to_numpy(dtype=np.int64)
    foreground = site_info_df['foreground'].to_numpy(dtype=bool)
    groups = site_info_df.groupby('CHROM', sort=False, observed=True).indices
    chromosomes = []
    for chr_id in site_info_df['CHROM'].unique():
        rows = groups[chr_id]
        order = rows[np.argsort(positions[rows], kind='stable')]
        chr_positions = positions[order]
        chr_foreground = foreground[order]
        chr_positions.flags.writeable = False
        chr_foreground.flags.writeable = False
        chromosomes.append((chr_id, chr_positions, chr_foreground))
    return chromosomes


def _iter_ld_windows(chromosomes, window_size):
    window_id = 0
    for chr_index, (_, positions, _) in enumerate(chromosomes):
        maximum = int(positions[-1])
        # Include the last coordinate, also when a pair has identical positions.
        for start in range(1, maximum + 1, window_size // 2):
            end = min(start + 2 * window_size - 1, maximum)
            yield (window_id, chr_index, start, end)
            window_id += 1


def _position_groups(positions):
    """Return first record indices and record groups in VCF encounter order."""
    _, first, inverse, counts = np.unique(
        positions, return_index=True, return_inverse=True, return_counts=True
    )
    order = np.argsort(first)
    rank = np.empty(len(order), dtype=np.intp)
    rank[order] = np.arange(len(order))
    return first[order], rank[inverse], counts[order]


def _check_duplicate_r2(values, reference, chr_id, query_positions, subject_positions, r2_threshold):
    consistent = np.isclose(values, reference, rtol=1e-12,
                            atol=1e-12, equal_nan=True)
    consistent &= (values > r2_threshold) == (reference > r2_threshold)
    if not consistent.all():
        q_idx, s_idx = np.argwhere(~consistent)[0]
        expected = np.broadcast_to(reference, values.shape)[q_idx, s_idx]
        raise ValueError(
            f"Conflicting R2 at {chr_id}: FOREGROUND_POS={query_positions[q_idx]}, "
            f"BACKGROUND_POS={subject_positions[s_idx]} "
            f"({expected!r} versus {values[q_idx, s_idx]!r})"
        )


def _select_owned_ld_pairs(chr_id, query_positions, subject_positions, r2, start, window_size, r2_threshold):
    q_first, q_inverse, q_counts = _position_groups(query_positions)
    s_first, s_inverse, s_counts = _position_groups(subject_positions)
    q_positions = query_positions[q_first]
    s_positions = subject_positions[s_first]
    has_duplicates = (q_counts > 1).any() or (s_counts > 1).any()
    unique_r2 = r2[np.ix_(q_first, s_first)] if has_duplicates else r2

    eligible = np.abs(np.subtract.outer(
        q_positions, s_positions)) < window_size
    # Both positions are >= start. Their minimum must belong to this core interval.
    # This is equivalent to owner_start = 1 + ((min(q, s) - 1) // S) * S.
    core_end = start + window_size // 2
    eligible &= (q_positions[:, None] < core_end) | (
        s_positions[None, :] < core_end)

    # Validate every record combination before applying the R2 threshold.
    for q_idx in np.flatnonzero(q_counts > 1):
        q_records = np.flatnonzero(q_inverse == q_idx)
        s_records = np.flatnonzero(eligible[q_idx, s_inverse])
        if s_records.size:
            _check_duplicate_r2(
                r2[np.ix_(q_records, s_records)],
                unique_r2[q_idx, s_inverse[s_records]][None, :],
                chr_id, query_positions[q_records], subject_positions[s_records],
                r2_threshold,
            )
    for s_idx in np.flatnonzero(s_counts > 1):
        q_indices = np.flatnonzero(eligible[:, s_idx])
        s_records = np.flatnonzero(s_inverse == s_idx)
        if q_indices.size:
            _check_duplicate_r2(
                r2[np.ix_(q_first[q_indices], s_records)],
                unique_r2[q_indices, s_idx][:, None],
                chr_id, q_positions[q_indices], subject_positions[s_records],
                r2_threshold,
            )

    q_indices, s_indices = np.where(eligible & (unique_r2 > r2_threshold))
    return q_positions[q_indices], s_positions[s_indices], unique_r2[q_indices, s_indices]


def _calculate_ld_window(task, chromosomes, vcf_reader, window_size, r2_threshold):
    window_id, chr_index, start, end = task
    chr_id, positions, foreground = chromosomes[chr_index]
    empty = (window_id, chr_index, np.empty(0, dtype=np.int64),
             np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
    try:
        left = np.searchsorted(positions, start, side='left')
        right = np.searchsorted(positions, end, side='right')
        window_positions = positions[left:right]
        window_foreground = foreground[left:right]
        query_set = set(window_positions[window_foreground])
        subject_set = set(window_positions[~window_foreground])
        if not query_set or not subject_set:
            return empty
        genotypes, _, used_positions = get_genotype_matrix_from_vcf_by_pos_list(
            None, chr_id, query_set | subject_set, vcf_reader=vcf_reader
        )
        used_positions = np.asarray(used_positions, dtype=np.int64)
        query_mask = np.fromiter(
            (pos in query_set for pos in used_positions), dtype=bool)
        subject_mask = np.fromiter(
            (pos in subject_set for pos in used_positions), dtype=bool)
        if not query_mask.any() or not subject_mask.any():
            return empty
        query_positions = used_positions[query_mask]
        subject_positions = used_positions[subject_mask]
        ld_df = calculate_LD(
            genotypes[query_mask], genotypes[subject_mask],
            query_positions, subject_positions,
        )
        pairs = _select_owned_ld_pairs(
            chr_id, query_positions, subject_positions,
            ld_df.to_numpy(copy=False), start, window_size, r2_threshold,
        )
        return (window_id, chr_index, *pairs)
    except Exception as error:
        raise RuntimeError(
            f"LD window {chr_id}:{start}-{end} failed: {error}") from error


_ld_worker_state = None
_ld_worker_init_error = None
_ld_worker_limiter = None


def _close_ld_worker():
    global _ld_worker_state, _ld_worker_limiter
    try:
        if _ld_worker_state is not None:
            _ld_worker_state[1].close()
    finally:
        _ld_worker_state = None
        if _ld_worker_limiter is not None:
            _ld_worker_limiter.restore_original_limits()
            _ld_worker_limiter = None


def _init_ld_worker(chromosomes, input_vcf_file, window_size, r2_threshold):
    global _ld_worker_state, _ld_worker_init_error, _ld_worker_limiter
    _ld_worker_state = None
    _ld_worker_init_error = None
    _ld_worker_limiter = None
    try:
        _ld_worker_limiter = threadpool_limits(limits=1, user_api='blas')
        # Open after fork: native VCF handles must not be inherited or queued.
        _ld_worker_state = (chromosomes, VCF(
            input_vcf_file), window_size, r2_threshold)
        Finalize(None, _close_ld_worker, exitpriority=10)
    except Exception as error:
        _ld_worker_init_error = str(error)
        _close_ld_worker()
        # Report through a task instead of repeatedly respawning failed initializers.


def _run_ld_window(task):
    if _ld_worker_init_error is not None:
        raise RuntimeError(
            f"Could not initialize LD worker: {_ld_worker_init_error}")
    return _calculate_ld_window(task, *_ld_worker_state)


def _iter_parallel_ld_results(
    pool,
    tasks,
    pool_num,
    progress_interval=None,
    *,
    worker=_run_ld_window,
):
    """
    Dynamically schedule tasks and yield results as soon as any task finishes.

    Keep at most 2 * pool_num tasks in flight. Unlike waiting on pending[0],
    a slow task does not block completed tasks behind it from being collected
    and replaced with new work.

    If progress_interval is not None, yield None periodically when no task
    finishes, so the parent process can still update progress.
    """
    tasks = iter(tasks)
    completed_queue = Queue()

    max_pending = 2 * pool_num
    pending = 0
    tasks_exhausted = False

    def on_success(result):
        completed_queue.put((True, result))

    def on_error(error):
        completed_queue.put((False, error))

    def submit_one():
        nonlocal pending, tasks_exhausted

        if tasks_exhausted:
            return False

        task = next(tasks, None)

        if task is None:
            tasks_exhausted = True
            return False

        pool.apply_async(
            worker,
            (task,),
            callback=on_success,
            error_callback=on_error,
        )

        pending += 1
        return True

    # Initial task buffer
    for _ in range(max_pending):
        if not submit_one():
            break

    # Collect whichever task finishes first
    while pending:

        try:
            if progress_interval is None:
                success, payload = completed_queue.get()
            else:
                success, payload = completed_queue.get(
                    timeout=progress_interval
                )

        except Empty:
            # Parent-side heartbeat for progress reporting
            yield None
            continue

        pending -= 1

        if not success:
            raise payload

        # Immediately refill the queue before yielding the result.
        submit_one()

        yield payload

# def _iter_parallel_ld_results(pool, tasks, pool_num, progress_interval=None, *, worker=_run_ld_window):
#     tasks = iter(tasks)
#     pending = deque()
#     for _ in range(2 * pool_num):
#         task = next(tasks, None)
#         if task is None:
#             break
#         pending.append(pool.apply_async(worker, (task,)))
#     while pending:
#         try:
#             result = pending[0].get(timeout=progress_interval)
#         except mp.TimeoutError:
#             # A parent-only heartbeat; workers send no extra progress messages.
#             yield None
#             continue
#         pending.popleft()
#         yield result
#         task = next(tasks, None)
#         if task is not None:
#             pending.append(pool.apply_async(worker, (task,)))


def _collect_ld_results(results, chromosomes, total, start_time, show_progress, progress_interval):
    chr_chunks, query_chunks, subject_chunks, r2_chunks = [], [], [], []
    last_width = 0

    def report(completed, now):
        nonlocal last_width
        elapsed = now - start_time
        rate = completed / elapsed if elapsed else 0.0
        eta = f"{(total - completed) / rate:.1f}s" if rate else "--"
        status = (
            f"LD windows: {completed}/{total} ({100 * completed / total:5.1f}%) | "
            f"elapsed {elapsed:.1f}s | {rate:.2f}/s | ETA {eta}"
        )
        print("\r" + status.ljust(last_width), end="", flush=True)
        last_width = len(status)

    last_update = perf_counter()
    if show_progress:
        report(0, last_update)
    completed = 0
    try:
        for result in results:
            if result is not None:
                completed += 1
                _, chr_index, query_positions, subject_positions, r2 = result
                if query_positions.size:
                    chr_chunks.append(
                        np.full(query_positions.size, chromosomes[chr_index][0], dtype=object))
                    query_chunks.append(query_positions)
                    subject_chunks.append(subject_positions)
                    r2_chunks.append(r2)
            if show_progress:
                now = perf_counter()
                if completed == total or now - last_update >= progress_interval:
                    report(completed, now)
                    last_update = now
    finally:
        if show_progress:
            print(flush=True)
    if not query_chunks:
        return _empty_link_pairs()
    return pd.DataFrame({
        'CHROM': np.concatenate(chr_chunks),
        'FOREGROUND_POS': np.concatenate(query_chunks),
        'BACKGROUND_POS': np.concatenate(subject_chunks),
        'R2': np.concatenate(r2_chunks),
    }, copy=False)


def calculate_ld_for_foreground_and_background_sites(
    site_info_df, input_vcf_file,
    *, ld_window_size=15000, r2_threshold=0.5, pool_num=4, show_progress=True, progress_interval=2.0,
):
    """Return unique coordinate pairs with distance < W and R2 > r2_threshold.

    r2_threshold must be finite and in [0, 1] (default 0.5); equality is excluded.
    Use pool_num=1 for serial execution. Parallel execution requires fork and
    threadpoolctl; each worker reuses its own VCF reader and one BLAS thread.
    Overlapping windows emit a pair only in the window owning its smaller
    coordinate, without swapping foreground FOREGROUND_POS and background BACKGROUND_POS.
    Duplicate VCF coordinates retain the first record only if R2 values agree
    (rtol=atol=1e-12, including the > r2_threshold decision); conflicts raise before filtering.
    The parent reports collected windows every progress_interval seconds.
    The genotype encoding and calculate_LD correlation calculation are unchanged.
    """
    ld_window_size = index(ld_window_size)
    pool_num = index(pool_num)
    if ld_window_size < 2 or pool_num < 1:
        raise ValueError(
            "ld_window_size must be >= 2 and pool_num must be >= 1")
    r2_threshold = float(r2_threshold)
    if not np.isfinite(r2_threshold) or not 0 <= r2_threshold <= 1:
        raise ValueError("r2_threshold must be a finite number in [0, 1]")
    if show_progress and (not np.isfinite(progress_interval) or progress_interval <= 0):
        raise ValueError(
            "progress_interval must be a finite positive number of seconds")

    chromosomes = _prepare_ld_chromosomes(site_info_df)
    if not any(flags.any() and not flags.all() for _, _, flags in chromosomes):
        return _empty_link_pairs()
    if pool_num > 1:
        if mp.get_start_method() != 'fork':
            raise RuntimeError(
                "Parallel LD requires 'fork'; use pool_num=1 in this kernel.")
        # Check before starting workers so a missing dependency fails promptly.

    total = sum((int(positions[-1]) - 1) // (ld_window_size // 2) + 1
                for _, positions, _ in chromosomes)
    tasks = _iter_ld_windows(chromosomes, ld_window_size)
    start_time = perf_counter()
    if pool_num == 1:
        reader = VCF(input_vcf_file)
        try:
            results = (_calculate_ld_window(task, chromosomes, reader, ld_window_size, r2_threshold)
                       for task in tasks)
            return _collect_ld_results(
                results, chromosomes, total, start_time, show_progress, progress_interval
            )
        finally:
            reader.close()

    pool = mp.Pool(
        processes=pool_num, initializer=_init_ld_worker,
        initargs=(chromosomes, input_vcf_file, ld_window_size, r2_threshold),
    )
    try:
        results = _iter_parallel_ld_results(
            pool, tasks, pool_num, progress_interval if show_progress else None
        )
        link_pairs_df = _collect_ld_results(
            results, chromosomes, total, start_time, show_progress, progress_interval
        )
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    return link_pairs_df


def _prepare_ld_pruning_regions(site_info_df, window_size):
    required = ['CHROM', 'POS']
    missing = [column for column in required if column not in site_info_df]
    if missing:
        raise ValueError(f"Missing LD input columns: {missing}")
    if site_info_df.empty:
        return [], [], []
    if site_info_df[required].isna().any().any():
        raise ValueError("CHROM and POS must not contain missing values")
    if not pd.api.types.is_integer_dtype(site_info_df['POS']) or (site_info_df['POS'] < 1).any():
        raise ValueError("POS must contain positive integer coordinates")

    positions = site_info_df['POS'].to_numpy(dtype=np.int64)
    first_coordinate = ~site_info_df.duplicated(required).to_numpy()
    groups = site_info_df.groupby('CHROM', sort=False, observed=True).indices
    chromosomes, regions, isolated_rows = [], [], []
    for chr_index, chr_id in enumerate(site_info_df['CHROM'].unique()):
        rows = groups[chr_id]
        rows = rows[first_coordinate[rows]]
        rows = rows[np.argsort(positions[rows], kind='stable')]
        chr_positions = positions[rows]
        chr_positions.flags.writeable = False
        rows.flags.writeable = False
        chromosomes.append((chr_id, chr_positions, rows))
        # A gap >= W cannot contain a qualifying pair, so these regions are independent.
        cuts = np.flatnonzero(np.diff(chr_positions) >= window_size) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, len(rows)]
        isolated = ends - starts == 1
        isolated_rows.append(rows[starts[isolated]])
        regions.append((chr_index, starts[~isolated], ends[~isolated]))
    return chromosomes, regions, isolated_rows


def _iter_ld_pruning_regions(regions):
    task_id = 0
    for chr_index, starts, ends in regions:
        for left, right in zip(starts, ends):
            yield (task_id, chr_index, int(left), int(right))
            task_id += 1


class _LDPruningGenotypeCache:
    """LRU cache for genotype tiles within each worker."""

    def __init__(self, chromosomes, reader, window_size, max_tiles=64):
        self.chromosomes = chromosomes
        self.reader = reader
        self.window_size = window_size
        self.max_tiles = index(max_tiles)
        if self.max_tiles < 1:
            raise ValueError("max_tiles must be >= 1")
        self.entries = OrderedDict()

    def get(self, chr_index, tile):
        key = (chr_index, int(tile))

        if key not in self.entries:
            chr_id, positions, _ = self.chromosomes[chr_index]
            start = int(tile) * self.window_size + 1
            left = np.searchsorted(positions, start)
            right = np.searchsorted(positions, start + self.window_size)

            genotypes, _, used_positions = get_genotype_matrix_from_vcf_by_pos_list(
                None, chr_id, positions[left:right], vcf_reader=self.reader
            )

            groups = {}
            for record_index, position in enumerate(used_positions):
                groups.setdefault(position, []).append(record_index)

            normalized = genotypes.astype(np.float64)
            if len(normalized):
                normalized -= normalized.mean(axis=1, keepdims=True)
                norms = np.sqrt(np.einsum('ij,ij->i', normalized, normalized))
                with np.errstate(divide='ignore', invalid='ignore'):
                    normalized /= norms[:, None]

            self.entries[key] = (genotypes, normalized, groups)

            # Keep only the most recently used genotype tiles.
            if len(self.entries) > self.max_tiles:
                self.entries.popitem(last=False)

        self.entries.move_to_end(key)
        return self.entries[key]


def _ld_pruning_hits(chr_id, lead_position, query_raw, query_normalized,
                     candidate_positions, entry, r2_threshold, *, return_r2=False):
    raw, normalized, groups = entry
    found = [int(pos) for pos in candidate_positions if pos in groups]
    if not found:
        if return_r2:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        return []

    record_indices = [i for pos in found for i in groups[pos]]

    # Pearson correlations only for lead x candidates; avoid candidate x candidate work.
    r2 = query_normalized @ normalized[record_indices].T
    np.clip(r2, -1, 1, out=r2)
    np.square(r2, out=r2)

    counts = np.fromiter((len(groups[pos]) for pos in found), dtype=np.intp)
    first = np.r_[0, np.cumsum(counts[:-1])]
    references = r2[0, first].copy()

    duplicates = (counts > 1) | (len(query_raw) > 1)
    recheck = duplicates | np.isclose(
        references,
        r2_threshold,
        rtol=1e-12,
        atol=1e-12,
    )

    # Recheck duplicate records and threshold boundaries with the existing calculation.
    for candidate_index in np.flatnonzero(recheck):
        pos = found[candidate_index]
        indices = groups[pos]

        values = calculate_LD(
            query_raw,
            raw[indices],
            [lead_position] * len(query_raw),
            [pos] * len(indices),
        ).to_numpy(copy=False)

        reference = values[0, 0]
        references[candidate_index] = reference

        if duplicates[candidate_index]:
            consistent = np.isclose(
                values,
                reference,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
            consistent &= (
                (values > r2_threshold)
                == (reference > r2_threshold)
            )

            if not consistent.all():
                raise ValueError(
                    f"Conflicting R2 at {chr_id}: "
                    f"LEAD_POS={lead_position}, CANDIDATE_POS={pos}"
                )

    found = np.asarray(found, dtype=np.int64)
    keep = references > r2_threshold

    if return_r2:
        return found[keep], references[keep].astype(np.float64, copy=False)

    return found[keep]


def _advance_ld_pruning_progress(progress, processed, kept):
    if hasattr(progress, 'get_lock'):
        with progress.get_lock():
            progress[0] += processed
            progress[1] += kept
    else:
        progress[0] += processed
        progress[1] += kept


def _prune_ld_region(task, chromosomes, cache, window_size, r2_threshold,
                     progress=None, progress_interval=2.0, report=None):
    task_id, chr_index, left, right = task
    chr_id, all_positions, all_rows = chromosomes[chr_index]
    positions, rows = all_positions[left:right], all_rows[left:right]
    # rows are iloc offsets in the original, already ranked input, not index labels.
    order = np.argsort(rows)
    pending = np.ones(len(rows), dtype=bool)
    kept = []
    reported, reported_kept = 0, 0
    last_update = perf_counter()
    try:
        for completed, lead in enumerate(order, 1):
            if pending[lead]:
                pending[lead] = False
                kept.append(rows[lead])
                lead_position = int(positions[lead])
                near_left = np.searchsorted(
                    positions, lead_position - window_size, side='right')
                near_right = np.searchsorted(
                    positions, lead_position + window_size, side='left')
                candidates = positions[near_left:near_right][pending[near_left:near_right]]
                if candidates.size:
                    entry = cache.get(
                        chr_index, (lead_position - 1) // window_size)
                    query_indices = entry[2].get(lead_position, [])
                    if query_indices:
                        query_raw, query_normalized = entry[0][query_indices], entry[1][query_indices]
                        tiles = (candidates - 1) // window_size
                        for tile in np.unique(tiles):
                            entry = cache.get(chr_index, tile)
                            hits = _ld_pruning_hits(
                                chr_id, lead_position, query_raw, query_normalized,
                                candidates[tiles == tile], entry, r2_threshold,
                            )
                            if len(hits):
                                pending[np.searchsorted(
                                    positions, hits)] = False
            if progress is not None:
                now = perf_counter()
                if completed == len(rows) or now - last_update >= progress_interval:
                    _advance_ld_pruning_progress(
                        progress, completed -
                        reported, len(kept) - reported_kept
                    )
                    reported, reported_kept, last_update = completed, len(
                        kept), now
                    if report is not None:
                        report()
        return task_id, np.asarray(kept, dtype=np.int64)
    except Exception as error:
        raise RuntimeError(
            f"LD pruning region {chr_id}:{positions[0]}-{positions[-1]} failed: {error}"
        ) from error


_ld_pruning_cache = None
_ld_pruning_progress = None
_ld_pruning_interval = 2.0


def _init_ld_pruning_worker(chromosomes, input_vcf_file, window_size, r2_threshold,
                            cache_max_tiles, progress, progress_interval):
    global _ld_pruning_cache, _ld_pruning_progress, _ld_pruning_interval
    _init_ld_worker(chromosomes, input_vcf_file, window_size, r2_threshold)
    _ld_pruning_cache = None
    if _ld_worker_state is not None:
        _ld_pruning_cache = _LDPruningGenotypeCache(
            chromosomes,
            _ld_worker_state[1],
            window_size,
            max_tiles=cache_max_tiles,
        )
    _ld_pruning_progress, _ld_pruning_interval = progress, progress_interval


def _run_ld_pruning_region(task):
    if _ld_worker_init_error is not None:
        raise RuntimeError(
            f"Could not initialize LD worker: {_ld_worker_init_error}")
    chromosomes, _, window_size, r2_threshold = _ld_worker_state
    return _prune_ld_region(
        task, chromosomes, _ld_pruning_cache, window_size, r2_threshold,
        _ld_pruning_progress, _ld_pruning_interval,
    )


# ---------------------------------------------------------------------------
# Parallel LD-graph construction for pruning
# ---------------------------------------------------------------------------
_ld_graph_cache = None


def _iter_ld_graph_tasks(regions, graph_chunk_size):
    """
    Split each independent genomic region into blocks of lead sites.

    Each lead site is compared only with sites to its right in genomic order,
    so every unordered site pair with distance < window_size is evaluated
    exactly once.
    """
    task_id = 0
    for chr_index, starts, ends in regions:
        for region_left, region_right in zip(starts, ends):
            region_left = int(region_left)
            region_right = int(region_right)

            for block_left in range(region_left, region_right, graph_chunk_size):
                block_right = min(block_left + graph_chunk_size, region_right)

                yield (
                    task_id,
                    chr_index,
                    region_left,
                    region_right,
                    block_left,
                    block_right,
                )
                task_id += 1


def _calculate_ld_graph_chunk(
    task,
    chromosomes,
    cache,
    window_size,
    r2_threshold,
):
    """
    Calculate high-LD edges for one block of lead sites.

    The chromosome positions are sorted. For each lead site in this block,
    only later genomic positions within distance < window_size are tested.
    This makes LD-edge calculation independent across tasks.

    Returns
    -------
    task_id : int
    processed_leads : int
    row_a : np.ndarray[int64]
        Original DataFrame iloc offsets for one endpoint.
    row_b : np.ndarray[int64]
        Original DataFrame iloc offsets for the other endpoint.
    edge_r2 : np.ndarray[float64]
        R2 value for each returned edge.
    """
    (
        task_id,
        chr_index,
        region_left,
        region_right,
        block_left,
        block_right,
    ) = task

    chr_id, all_positions, all_rows = chromosomes[chr_index]

    edge_a_chunks = []
    edge_b_chunks = []
    edge_r2_chunks = []

    try:
        for lead_index in range(block_left, block_right):
            lead_position = int(all_positions[lead_index])

            # Only compare with positions to the right of the lead.
            # side='left' enforces distance strictly < window_size.
            candidate_right = np.searchsorted(
                all_positions,
                lead_position + window_size,
                side='left',
            )
            candidate_right = min(candidate_right, region_right)

            if candidate_right <= lead_index + 1:
                continue

            candidate_positions = all_positions[
                lead_index + 1:candidate_right
            ]

            # Fetch the lead genotype.
            lead_entry = cache.get(
                chr_index,
                (lead_position - 1) // window_size,
            )
            query_indices = lead_entry[2].get(lead_position, [])

            # A site absent from the VCF cannot create an LD edge.
            if not query_indices:
                continue

            query_raw = lead_entry[0][query_indices]
            query_normalized = lead_entry[1][query_indices]

            candidate_tiles = (candidate_positions - 1) // window_size

            for tile in np.unique(candidate_tiles):
                tile_candidates = candidate_positions[
                    candidate_tiles == tile
                ]

                entry = cache.get(chr_index, tile)

                hits, hit_r2 = _ld_pruning_hits(
                    chr_id,
                    lead_position,
                    query_raw,
                    query_normalized,
                    tile_candidates,
                    entry,
                    r2_threshold,
                    return_r2=True,
                )

                if len(hits):
                    hit_indices = np.searchsorted(all_positions, hits)

                    edge_a_chunks.append(
                        np.full(
                            len(hits),
                            all_rows[lead_index],
                            dtype=np.int64,
                        )
                    )
                    edge_b_chunks.append(
                        all_rows[hit_indices].astype(np.int64, copy=False)
                    )
                    edge_r2_chunks.append(
                        np.asarray(hit_r2, dtype=np.float64)
                    )

        if edge_a_chunks:
            row_a = np.concatenate(edge_a_chunks)
            row_b = np.concatenate(edge_b_chunks)
            edge_r2 = np.concatenate(edge_r2_chunks)
        else:
            row_a = np.empty(0, dtype=np.int64)
            row_b = np.empty(0, dtype=np.int64)
            edge_r2 = np.empty(0, dtype=np.float64)

        return (
            task_id,
            block_right - block_left,
            row_a,
            row_b,
            edge_r2,
        )

    except Exception as error:
        start_pos = int(all_positions[block_left])
        end_pos = int(all_positions[block_right - 1])
        raise RuntimeError(
            f"LD graph chunk {chr_id}:{start_pos}-{end_pos} failed: {error}"
        ) from error


def _init_ld_graph_worker(
    chromosomes,
    input_vcf_file,
    window_size,
    r2_threshold,
    cache_max_tiles,
):
    global _ld_graph_cache

    _init_ld_worker(
        chromosomes,
        input_vcf_file,
        window_size,
        r2_threshold,
    )

    _ld_graph_cache = None

    if _ld_worker_state is not None:
        _ld_graph_cache = _LDPruningGenotypeCache(
            chromosomes,
            _ld_worker_state[1],
            window_size,
            max_tiles=cache_max_tiles,
        )


def _run_ld_graph_chunk(task):
    if _ld_worker_init_error is not None:
        raise RuntimeError(
            f"Could not initialize LD worker: {_ld_worker_init_error}"
        )

    chromosomes, _, window_size, r2_threshold = _ld_worker_state

    return _calculate_ld_graph_chunk(
        task,
        chromosomes,
        _ld_graph_cache,
        window_size,
        r2_threshold,
    )


def _greedy_prune_from_ld_edges(
    site_count,
    unique_rows,
    edge_a,
    edge_b,
    edge_r2,
    *,
    return_pruning_details=False,
):
    """
    Apply rank-aware greedy pruning to a precomputed LD graph.

    DataFrame row order is the priority order. For every undirected LD edge,
    only the earlier-ranked endpoint can remove the later-ranked endpoint.

    When return_pruning_details=True, also return:
        removed_by
            int64 array indexed by iloc offset. A removed row stores the
            iloc offset of the retained higher-priority row that removed it;
            all other rows contain -1.
        removed_r2
            float64 array indexed by iloc offset. A removed row stores the
            R2 to the retained row that removed it; all other rows are NaN.
    """
    unique_rows = np.asarray(unique_rows, dtype=np.int64)

    removed_by = None
    removed_r2 = None

    if return_pruning_details:
        removed_by = np.full(site_count, -1, dtype=np.int64)
        removed_r2 = np.full(site_count, np.nan, dtype=np.float64)

    if unique_rows.size == 0:
        if return_pruning_details:
            return unique_rows, removed_by, removed_r2
        return unique_rows

    if len(edge_a) == 0:
        kept = np.sort(unique_rows)
        if return_pruning_details:
            return kept, removed_by, removed_r2
        return kept

    edge_a = np.asarray(edge_a, dtype=np.int64)
    edge_b = np.asarray(edge_b, dtype=np.int64)
    edge_r2 = np.asarray(edge_r2, dtype=np.float64)

    if not (len(edge_a) == len(edge_b) == len(edge_r2)):
        raise ValueError(
            "Internal LD pruning error: edge arrays have different lengths"
        )

    # Orient every undirected edge from higher priority to lower priority.
    earlier = np.minimum(edge_a, edge_b)
    later = np.maximum(edge_a, edge_b)

    # Stable sort keeps R2 aligned with its edge.
    order = np.argsort(earlier, kind='stable')
    earlier = earlier[order]
    later = later[order]
    edge_r2 = edge_r2[order]

    counts = np.bincount(earlier, minlength=site_count)
    offsets = np.empty(site_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    active = np.zeros(site_count, dtype=bool)
    active[unique_rows] = True

    kept = []

    # Serial dependency is now cheap: only graph traversal, no LD calculation.
    for row in np.sort(unique_rows):
        if not active[row]:
            continue

        kept.append(row)

        start = offsets[row]
        end = offsets[row + 1]

        if end <= start:
            continue

        neighbors = later[start:end]
        neighbor_r2 = edge_r2[start:end]

        # Only neighbors that are still active are actually removed by this
        # retained row. Thus each removed row gets the first/highest-priority
        # retained remover, exactly matching greedy pruning semantics.
        active_mask = active[neighbors]

        if not active_mask.any():
            continue

        to_remove = neighbors[active_mask]
        active[to_remove] = False

        if return_pruning_details:
            removed_by[to_remove] = row
            removed_r2[to_remove] = neighbor_r2[active_mask]

    kept = np.asarray(kept, dtype=np.int64)

    if return_pruning_details:
        return kept, removed_by, removed_r2

    return kept


def _format_ld_pruning_output(
    site_info_df,
    kept_rows,
    removed_by,
    removed_r2,
    *,
    full_output=False,
):
    """
    Format prune_sites_by_ld() output.

    Default:
        Return only retained rows, exactly as before.

    full_output=True:
        Return every original row and append:
            LD_PRUNING_KEEP : bool
            LD_PRUNED_BY    : original DataFrame index of the retained remover
            LD_PRUNED_R2    : R2 to that retained remover

        Retained rows have missing LD_PRUNED_BY / LD_PRUNED_R2.

    Duplicate input coordinates:
        The first (CHROM, POS) occurrence is the representative used by the
        pruning algorithm. Later duplicates are marked not kept.
        - If the representative is retained, the duplicate is attributed to
          that representative with R2 = 1.0.
        - If the representative is LD-pruned, the duplicate is attributed to
          the same retained remover and inherits the representative's R2.
    """
    kept_rows = np.asarray(kept_rows, dtype=np.int64)

    if not full_output:
        return site_info_df.iloc[kept_rows].copy()

    n_rows = len(site_info_df)
    result = site_info_df.copy()

    keep_mask = np.zeros(n_rows, dtype=bool)
    keep_mask[kept_rows] = True

    remover_rows = np.asarray(removed_by, dtype=np.int64).copy()
    remover_r2 = np.asarray(removed_r2, dtype=np.float64).copy()

    if len(remover_rows) != n_rows or len(remover_r2) != n_rows:
        raise ValueError(
            "Internal LD pruning error: pruning-detail arrays have "
            "unexpected length"
        )

    if n_rows:
        # Resolve duplicate input coordinates. The pruning core keeps only the
        # first occurrence of each (CHROM, POS).
        coordinates = pd.MultiIndex.from_frame(
            site_info_df[['CHROM', 'POS']]
        )
        codes = pd.factorize(coordinates, sort=False)[0]
        row_numbers = np.arange(n_rows, dtype=np.int64)

        first_by_code = np.full(
            int(codes.max()) + 1,
            n_rows,
            dtype=np.int64,
        )
        np.minimum.at(first_by_code, codes, row_numbers)

        representative = first_by_code[codes]
        duplicate_mask = representative != row_numbers

        if duplicate_mask.any():
            duplicate_rows = row_numbers[duplicate_mask]
            representative_rows = representative[duplicate_mask]

            representative_pruner = remover_rows[representative_rows]
            representative_r2 = remover_r2[representative_rows]

            rep_was_ld_pruned = representative_pruner >= 0

            # If the representative itself was LD-pruned, point to the same
            # retained remover. Otherwise the retained representative itself
            # is why a later duplicate is not retained.
            remover_rows[duplicate_rows] = np.where(
                rep_was_ld_pruned,
                representative_pruner,
                representative_rows,
            )
            remover_r2[duplicate_rows] = np.where(
                rep_was_ld_pruned,
                representative_r2,
                1.0,
            )

    reason = np.empty(n_rows, dtype=object)
    reason[:] = pd.NA

    removed_mask = ~keep_mask
    has_reason = removed_mask & (remover_rows >= 0)

    if has_reason.any():
        input_index = np.asarray(site_info_df.index, dtype=object)
        reason[has_reason] = input_index[remover_rows[has_reason]]

    r2_out = pd.array(
        np.where(removed_mask, remover_r2, np.nan),
        dtype="Float64",
    )

    result['LD_PRUNING_KEEP'] = keep_mask
    result['LD_PRUNED_BY'] = reason
    result['LD_PRUNED_R2'] = r2_out

    return result


def prune_sites_by_ld(
    site_info_df,
    input_vcf_file,
    *,
    ld_window_size=15000,
    r2_threshold=0.5,
    pool_num=4,
    cache_max_tiles=64,
    graph_chunk_size=256,
    full_output=False,
    show_progress=True,
    progress_interval=2.0,
):
    """
    Greedily retain the highest-ranked sites, using the input's current row order.

    This implementation separates LD calculation from greedy pruning:

    1. Build the high-LD graph in parallel.
       Every unique site pair on the same chromosome with
       distance strictly < ld_window_size is evaluated once.
       An edge is retained only when R2 is strictly > r2_threshold.

    2. Apply the original greedy pruning serially using DataFrame row order
       as priority. An earlier retained row removes its later-ranked LD
       neighbors.

    Repeated input (CHROM, POS) coordinates keep their first row.
    Sites absent from the VCF or with NaN R2 have no LD edge and therefore
    remain unless removed through another valid edge.

    Parameters
    ----------
    cache_max_tiles : int, default 64
        Maximum number of W-sized genotype tiles cached by each worker.

    graph_chunk_size : int, default 256
        Number of genomic-order lead sites per parallel LD-graph task.

    full_output : bool, default False
        False:
            Preserve the historical API and return only retained rows.

        True:
            Return all original input rows and append:
                LD_PRUNING_KEEP
                    True for retained rows, False otherwise.
                LD_PRUNED_BY
                    For a removed row, the original DataFrame index label of
                    the retained higher-priority row that removed it.
                    Retained rows contain pd.NA.
                LD_PRUNED_R2
                    R2 between the removed row and LD_PRUNED_BY.
                    Retained rows contain pd.NA.

            For repeated input (CHROM, POS) rows, only the first occurrence
            participates in pruning. A later duplicate points to the retained
            representative with R2=1.0, or, if that representative was itself
            LD-pruned, to the same retained remover and R2.

    Notes
    -----
    Parallel execution requires multiprocessing start method 'fork' and
    threadpoolctl. Each worker opens its own VCF reader and uses one BLAS
    thread. Only LD edges above the threshold are retained in memory.
    """
    ld_window_size = index(ld_window_size)
    pool_num = index(pool_num)
    cache_max_tiles = index(cache_max_tiles)
    graph_chunk_size = index(graph_chunk_size)

    if not isinstance(full_output, (bool, np.bool_)):
        raise TypeError("full_output must be True or False")
    full_output = bool(full_output)

    if (
        ld_window_size < 1
        or pool_num < 1
        or cache_max_tiles < 1
        or graph_chunk_size < 1
    ):
        raise ValueError(
            "ld_window_size, pool_num, cache_max_tiles and "
            "graph_chunk_size must all be >= 1"
        )

    r2_threshold = float(r2_threshold)

    if not np.isfinite(r2_threshold) or not 0 <= r2_threshold <= 1:
        raise ValueError(
            "r2_threshold must be a finite number in [0, 1]"
        )

    if show_progress and (
        not np.isfinite(progress_interval)
        or progress_interval <= 0
    ):
        raise ValueError(
            "progress_interval must be a finite positive number of seconds"
        )

    chromosomes, regions, isolated_rows = _prepare_ld_pruning_regions(
        site_info_df,
        ld_window_size,
    )

    unique_row_chunks = [
        rows
        for _, _, rows in chromosomes
        if len(rows)
    ]

    if not unique_row_chunks:
        kept_rows = np.empty(0, dtype=np.int64)

        if not full_output:
            return site_info_df.iloc[kept_rows].copy()

        removed_by = np.full(len(site_info_df), -1, dtype=np.int64)
        removed_r2 = np.full(len(site_info_df), np.nan, dtype=np.float64)

        return _format_ld_pruning_output(
            site_info_df,
            kept_rows,
            removed_by,
            removed_r2,
            full_output=True,
        )

    unique_rows = np.sort(np.concatenate(unique_row_chunks))

    total_nonisolated_sites = sum(
        int(np.sum(ends - starts))
        for _, starts, ends in regions
    )

    # If every unique site is isolated by >= W, no LD calculation is needed.
    if total_nonisolated_sites == 0:
        if not full_output:
            return site_info_df.iloc[unique_rows].copy()

        removed_by = np.full(len(site_info_df), -1, dtype=np.int64)
        removed_r2 = np.full(len(site_info_df), np.nan, dtype=np.float64)

        return _format_ld_pruning_output(
            site_info_df,
            unique_rows,
            removed_by,
            removed_r2,
            full_output=True,
        )

    if pool_num > 1:
        if mp.get_start_method() != 'fork':
            raise RuntimeError(
                "Parallel LD requires 'fork'; use pool_num=1 in this kernel."
            )

    tasks = _iter_ld_graph_tasks(
        regions,
        graph_chunk_size,
    )

    total_tasks = sum(
        (int(end - start) + graph_chunk_size - 1) // graph_chunk_size
        for _, starts, ends in regions
        for start, end in zip(starts, ends)
    )

    start_time = perf_counter()
    last_report = start_time
    last_width = 0
    processed_leads = 0
    edge_count = 0

    edge_a_chunks = []
    edge_b_chunks = []
    edge_r2_chunks = []

    def report(force=False):
        nonlocal last_report, last_width

        if not show_progress:
            return

        now = perf_counter()

        if not force and now - last_report < progress_interval:
            return

        elapsed = now - start_time
        pct = (
            100.0 * processed_leads / total_nonisolated_sites
            if total_nonisolated_sites
            else 100.0
        )

        status = (
            f"LD graph: {processed_leads}/{total_nonisolated_sites} "
            f"({pct:5.1f}%) | edges {edge_count} | "
            f"elapsed {elapsed:.1f}s"
        )

        print(
            '\r' + status.ljust(last_width),
            end='',
            flush=True,
        )

        last_report = now
        last_width = len(status)

    pool = None
    reader = None

    report(force=True)

    try:
        if pool_num == 1:
            reader = VCF(input_vcf_file)

            cache = _LDPruningGenotypeCache(
                chromosomes,
                reader,
                ld_window_size,
                max_tiles=cache_max_tiles,
            )

            results = (
                _calculate_ld_graph_chunk(
                    task,
                    chromosomes,
                    cache,
                    ld_window_size,
                    r2_threshold,
                )
                for task in tasks
            )

        else:
            workers = min(pool_num, total_tasks)

            pool = mp.Pool(
                processes=workers,
                initializer=_init_ld_graph_worker,
                initargs=(
                    chromosomes,
                    input_vcf_file,
                    ld_window_size,
                    r2_threshold,
                    cache_max_tiles,
                ),
            )

            results = _iter_parallel_ld_results(
                pool,
                tasks,
                workers,
                progress_interval if show_progress else None,
                worker=_run_ld_graph_chunk,
            )

        for result in results:
            if result is None:
                report()
                continue

            _, processed, edge_a, edge_b, edge_r2 = result

            processed_leads += processed

            if len(edge_a):
                edge_a_chunks.append(edge_a)
                edge_b_chunks.append(edge_b)
                edge_r2_chunks.append(edge_r2)
                edge_count += len(edge_a)

            report()

        if pool is not None:
            pool.close()
            pool.join()

        report(force=True)

    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise

    finally:
        if reader is not None:
            reader.close()

        if show_progress:
            print(flush=True)

    if edge_a_chunks:
        edge_a = np.concatenate(edge_a_chunks)
        edge_b = np.concatenate(edge_b_chunks)
        edge_r2 = np.concatenate(edge_r2_chunks)
    else:
        edge_a = np.empty(0, dtype=np.int64)
        edge_b = np.empty(0, dtype=np.int64)
        edge_r2 = np.empty(0, dtype=np.float64)

    greedy_start = perf_counter()

    if full_output:
        kept_rows, removed_by, removed_r2 = _greedy_prune_from_ld_edges(
            len(site_info_df),
            unique_rows,
            edge_a,
            edge_b,
            edge_r2,
            return_pruning_details=True,
        )
    else:
        kept_rows = _greedy_prune_from_ld_edges(
            len(site_info_df),
            unique_rows,
            edge_a,
            edge_b,
            edge_r2,
            return_pruning_details=False,
        )

    if show_progress:
        greedy_elapsed = perf_counter() - greedy_start
        total_elapsed = perf_counter() - start_time

        print(
            f"LD pruning complete: kept {len(kept_rows)}/{len(unique_rows)} "
            f"unique sites | edges {len(edge_a)} | "
            f"greedy {greedy_elapsed:.2f}s | total {total_elapsed:.1f}s",
            flush=True,
        )

    if not full_output:
        return site_info_df.iloc[kept_rows].copy()

    return _format_ld_pruning_output(
        site_info_df,
        kept_rows,
        removed_by,
        removed_r2,
        full_output=True,
    )


def prune_sites_by_ld_nested(
    site_info_df,
    input_vcf_file,
    r2_thresholds,
    *,
    ld_window_size=15000,
    pool_num=4,
    cache_max_tiles=64,
    graph_chunk_size=256,
    show_progress=True,
    progress_interval=2.0,
):
    """Build nested retained sets as strictly increasing R2 thresholds relax.

    The input row order defines GEA priority; sort by descending association
    score before calling. The first threshold uses ordinary greedy pruning.
    At each later threshold, all previously retained sites take priority over
    the remaining candidates, which are considered in their original order.
    Previous sites are mutually compatible at the relaxed threshold, so they
    remain retained. Returned DataFrames restore the original input order and
    index, including when the input index is non-unique.

    Returns a dict mapping each threshold to its retained DataFrame. The
    existing LD window, strict R2 comparison, and duplicate-coordinate rules
    are unchanged. LD is calculated separately for each threshold.
    """
    thresholds = np.asarray(list(r2_thresholds), dtype=float)
    if (
        thresholds.ndim != 1
        or thresholds.size == 0
        or not np.isfinite(thresholds).all()
        or np.any((thresholds < 0) | (thresholds > 1))
        or np.any(np.diff(thresholds) <= 0)
    ):
        raise ValueError(
            "r2_thresholds must be non-empty, strictly increasing, and in [0, 1]")

    # Internal row numbers survive reprioritization without relying on user IDs
    # or assuming that the caller's DataFrame index is unique.
    ranked_df = site_info_df.reset_index(drop=True)
    kept_rows = np.empty(0, dtype=np.int64)
    results = {}
    for threshold in thresholds:
        remaining = np.ones(len(ranked_df), dtype=bool)
        remaining[kept_rows] = False
        priority_rows = np.r_[kept_rows, np.flatnonzero(remaining)]
        if show_progress:
            print(
                f"Nested LD pruning: R2={threshold:g}, fixed sites={len(kept_rows)}")
        retained = prune_sites_by_ld(
            ranked_df.iloc[priority_rows],
            input_vcf_file,
            ld_window_size=ld_window_size,
            r2_threshold=float(threshold),
            pool_num=pool_num,
            cache_max_tiles=cache_max_tiles,
            graph_chunk_size=graph_chunk_size,
            show_progress=show_progress,
            progress_interval=progress_interval,
        )
        next_rows = np.sort(retained.index.to_numpy(dtype=np.int64))
        if not np.isin(kept_rows, next_rows).all():
            raise RuntimeError(
                "Nested LD pruning unexpectedly removed a previously retained site")
        kept_rows = next_rows
        results[float(threshold)] = site_info_df.iloc[kept_rows].copy()
    return results


def _calculate_ld_cross_chunk(task, chromosomes, cache, window_size, r2_threshold):
    """Calculate only first-table x second-table pairs for a block of leads."""
    task_id, chr_index, _, _, block_left, block_right = task
    chr_id, _, cross_sites = chromosomes[chr_index]
    query_positions, query_rows, subject_positions, subject_rows = cross_sites
    row_a_chunks, row_b_chunks, r2_chunks = [], [], []
    try:
        for lead_index in range(block_left, block_right):
            lead_position = int(query_positions[lead_index])
            left = np.searchsorted(
                subject_positions, lead_position - window_size, side='right')
            right = np.searchsorted(
                subject_positions, lead_position + window_size, side='left')
            candidates = subject_positions[left:right]
            if not candidates.size:
                continue
            lead_entry = cache.get(
                chr_index, (lead_position - 1) // window_size)
            query_indices = lead_entry[2].get(lead_position, [])
            if not query_indices:
                continue
            query_raw = lead_entry[0][query_indices]
            query_normalized = lead_entry[1][query_indices]
            tiles = (candidates - 1) // window_size
            for tile in np.unique(tiles):
                hits, r2 = _ld_pruning_hits(
                    chr_id, lead_position, query_raw, query_normalized,
                    candidates[tiles == tile], cache.get(chr_index, tile),
                    r2_threshold, return_r2=True,
                )
                if len(hits):
                    row_a_chunks.append(
                        np.full(len(hits), query_rows[lead_index], dtype=np.int64))
                    row_b_chunks.append(
                        subject_rows[np.searchsorted(subject_positions, hits)])
                    r2_chunks.append(r2)
        return (
            task_id, block_right - block_left,
            np.concatenate(row_a_chunks) if row_a_chunks else np.empty(
                0, dtype=np.int64),
            np.concatenate(row_b_chunks) if row_b_chunks else np.empty(
                0, dtype=np.int64),
            np.concatenate(r2_chunks) if r2_chunks else np.empty(
                0, dtype=np.float64),
        )
    except Exception as error:
        raise RuntimeError(
            f"Cross-table LD chunk {chr_id}:"
            f"{query_positions[block_left]}-{query_positions[block_right - 1]} failed: {error}"
        ) from error


def _run_ld_cross_chunk(task):
    if _ld_worker_init_error is not None:
        raise RuntimeError(
            f"Could not initialize LD worker: {_ld_worker_init_error}")
    chromosomes, _, window_size, r2_threshold = _ld_worker_state
    return _calculate_ld_cross_chunk(
        task, chromosomes, _ld_graph_cache, window_size, r2_threshold,
    )


def calculate_ld_matrix(
    site_info_df,
    input_vcf_file,
    ld_window_size=15000,
    *,
    site_info_df2=None,
    output_format="pairs",
    min_r2=0.5,
    sort=False,
    pool_num=4,
    cache_max_tiles=64,
    graph_chunk_size=256,
    show_progress=True,
    progress_interval=2.0,
):
    """Return within-table or cross-table R2 as pairs (default) or a matrix.

    site_info_df must contain CHROM and positive integer POS coordinates.
    Pairs on the same chromosome are calculated when their distance is
    strictly < ld_window_size, matching the existing LD functions.

    When site_info_df2 is provided, calculate only pairs between the two
    tables. It must also contain CHROM and positive integer POS coordinates.
    INDEX_1 / matrix rows use site_info_df.index; INDEX_2 / matrix columns
    use site_info_df2.index. Matrix axes preserve the original input order.
    Index labels from the two tables are treated as separate namespaces:
    cross-table pairs are ordered, never swapped or deduplicated in reverse.
    Matching coordinates are calculated from the VCF, so absent/constant
    sites remain NaN; no positional diagonal is forced to 1 in this mode.

    output_format="pairs" returns INDEX_1, INDEX_2 and R2 columns, using the
    original input index labels (tuples for a MultiIndex). Only finite R2
    values >= min_r2 are retained (default 0.5; equality is included).
    min_r2 must be finite and in [0, 1]; set it to 0 to retain zero R2.
    This parameter is ignored in matrix mode. In single-table mode pairs with
    identical index labels are excluded and each unordered pair appears once.
    In cross-table mode each ordered pair appears once, including matching
    index labels across tables. Repeated index-label pairs keep the first
    encountered record after optional sorting. Repeated coordinates expand
    to their original index labels.

    sort=False (default) skips pair-output sorting; output order may vary
    with parallel task completion. sort=True orders pairs by their original
    input row offsets. Pair deduplication is applied in both modes.
    This parameter is ignored in matrix mode.

    output_format="matrix" preserves the input indices and order on both axes.
    Cross-chromosome pairs, pairs outside the window, absent VCF sites and
    undefined correlations are NaN. In single-table mode the diagonal is 1
    by convention, even for absent/constant sites, including between copies
    of a coordinate. Repeated coordinates reuse their representative's
    results on each respective axis in both modes.

    Reuses the LD-graph chunks, genotype cache, duplicate-VCF validation and
    dynamic parallel scheduler used by prune_sites_by_ld, without pruning.
    Genotype encoding and missing-call handling are inherited unchanged.
    pool_num=1 runs serially; parallel execution requires 'fork' and
    threadpoolctl, with one VCF reader and one BLAS thread per worker.
    cache_max_tiles bounds cached genotype tiles per worker; graph_chunk_size
    controls the number of lead sites per task. Progress is reported by the
    parent every progress_interval seconds when show_progress is True.
    Pair output collects edges directly without allocating a dense matrix.
    Matrix output requires approximately 8 * n_rows * n_columns bytes,
    in addition to genotype caches and temporary arrays.
    """
    if output_format not in ("pairs", "matrix"):
        raise ValueError('output_format must be "pairs" or "matrix"')
    if output_format == "pairs":
        if not isinstance(sort, (bool, np.bool_)):
            raise TypeError("sort must be True or False")
        min_r2 = float(min_r2)
        if not np.isfinite(min_r2) or not 0 <= min_r2 <= 1:
            raise ValueError("min_r2 must be a finite number in [0, 1]")
    ld_window_size = index(ld_window_size)
    pool_num = index(pool_num)
    cache_max_tiles = index(cache_max_tiles)
    graph_chunk_size = index(graph_chunk_size)
    if min(ld_window_size, pool_num, cache_max_tiles, graph_chunk_size) < 1:
        raise ValueError(
            "ld_window_size, pool_num, cache_max_tiles and "
            "graph_chunk_size must all be >= 1"
        )
    if show_progress and (
        not np.isfinite(progress_interval) or progress_interval <= 0
    ):
        raise ValueError(
            "progress_interval must be a finite positive number of seconds")

    chromosomes, regions, _ = _prepare_ld_pruning_regions(
        site_info_df, ld_window_size,
    )
    cross_table = site_info_df2 is not None
    column_info_df = site_info_df2 if cross_table else site_info_df
    if cross_table:
        subject_chromosomes, _, _ = _prepare_ld_pruning_regions(
            site_info_df2, ld_window_size,
        )
        subjects = {chr_id: (positions, rows)
                    for chr_id, positions, rows in subject_chromosomes}
        cross_chromosomes, regions = [], []
        for chr_id, query_positions, query_rows in chromosomes:
            if chr_id not in subjects:
                continue
            subject_positions, subject_rows = subjects[chr_id]
            # Cache the union of coordinates, but lead tasks and candidate
            # searches retain their separate table memberships.
            chr_index = len(cross_chromosomes)
            cross_chromosomes.append((
                chr_id, np.union1d(query_positions, subject_positions),
                (query_positions, query_rows, subject_positions, subject_rows),
            ))
            regions.append((chr_index, np.array(
                [0]), np.array([len(query_positions)])))
        chromosomes = cross_chromosomes

    calculate_chunk = _calculate_ld_cross_chunk if cross_table else _calculate_ld_graph_chunk
    run_chunk = _run_ld_cross_chunk if cross_table else _run_ld_graph_chunk
    matrix = None
    row_a_chunks, row_b_chunks, r2_chunks = [], [], []
    if output_format == "matrix":
        matrix = np.full((len(site_info_df), len(
            column_info_df)), np.nan, dtype=np.float64)
        if not cross_table:
            np.fill_diagonal(matrix, 1.0)
    total_leads = sum(int(np.sum(ends - starts))
                      for _, starts, ends in regions)
    total_tasks = sum(
        (int(end - start) + graph_chunk_size - 1) // graph_chunk_size
        for _, starts, ends in regions
        for start, end in zip(starts, ends)
    )

    if total_tasks:
        if pool_num > 1:
            if mp.get_start_method() != 'fork':
                raise RuntimeError(
                    "Parallel LD requires 'fork'; use pool_num=1 in this kernel.")

        # The graph helper filters with R2 > threshold. -inf keeps finite
        # values including exactly zero, while undefined R2 stays NaN.
        r2_threshold = -np.inf
        tasks = _iter_ld_graph_tasks(regions, graph_chunk_size)
        start_time = perf_counter()
        last_report = start_time
        last_width = 0
        processed_leads = 0

        def report(force=False):
            nonlocal last_report, last_width
            if not show_progress:
                return
            now = perf_counter()
            if not force and now - last_report < progress_interval:
                return
            status = (
                f"LD {output_format}: {processed_leads}/{total_leads} "
                f"({100.0 * processed_leads / total_leads:5.1f}%) | "
                f"elapsed {now - start_time:.1f}s"
            )
            print('\r' + status.ljust(last_width), end='', flush=True)
            last_report, last_width = now, len(status)

        pool = None
        reader = None
        report(force=True)
        try:
            if pool_num == 1:
                reader = VCF(input_vcf_file)
                cache = _LDPruningGenotypeCache(
                    chromosomes, reader, ld_window_size, max_tiles=cache_max_tiles,
                )
                results = (
                    calculate_chunk(
                        task, chromosomes, cache, ld_window_size, r2_threshold,
                    )
                    for task in tasks
                )
            else:
                workers = min(pool_num, total_tasks)
                pool = mp.Pool(
                    processes=workers,
                    initializer=_init_ld_graph_worker,
                    initargs=(
                        chromosomes, input_vcf_file, ld_window_size,
                        r2_threshold, cache_max_tiles,
                    ),
                )
                results = _iter_parallel_ld_results(
                    pool, tasks, workers,
                    progress_interval if show_progress else None,
                    worker=run_chunk,
                )

            for result in results:
                if result is not None:
                    _, processed, row_a, row_b, r2 = result
                    if output_format == "matrix":
                        matrix[row_a, row_b] = r2
                        if not cross_table:
                            matrix[row_b, row_a] = r2
                    else:
                        keep = np.isfinite(r2) & (r2 >= min_r2)
                        if keep.any():
                            row_a_chunks.append(row_a[keep])
                            row_b_chunks.append(row_b[keep])
                            r2_chunks.append(r2[keep])
                    processed_leads += processed
                report()

            if pool is not None:
                pool.close()
                pool.join()
            report(force=True)
        except BaseException:
            if pool is not None:
                pool.terminate()
                pool.join()
            raise
        finally:
            if reader is not None:
                reader.close()
            if show_progress:
                print(flush=True)

    coordinate_maps = []
    for frame in (site_info_df, column_info_df) if cross_table else (site_info_df,):
        coordinates = pd.MultiIndex.from_frame(frame[['CHROM', 'POS']])
        first_rows = np.flatnonzero(~coordinates.duplicated())
        representatives = first_rows[pd.factorize(
            coordinates.to_numpy(), sort=False)[0]]
        duplicate_rows = np.flatnonzero(
            representatives != np.arange(len(frame)))
        coordinate_maps.append((representatives, duplicate_rows))
    if not cross_table:
        coordinate_maps.append(coordinate_maps[0])

    if output_format == "matrix":
        (row_representatives, duplicate_rows), (col_representatives,
                                                duplicate_cols) = coordinate_maps
        matrix[duplicate_rows, :] = matrix[row_representatives[duplicate_rows], :]
        matrix[:, duplicate_cols] = matrix[:,
                                           col_representatives[duplicate_cols]]
        return pd.DataFrame(
            matrix, index=site_info_df.index.copy(), columns=column_info_df.index.copy(),
        )

    if output_format == "pairs":
        pairs = pd.DataFrame({
            'row_a': np.concatenate(row_a_chunks) if row_a_chunks else np.empty(0, dtype=np.int64),
            'row_b': np.concatenate(row_b_chunks) if row_b_chunks else np.empty(0, dtype=np.int64),
            'R2': np.concatenate(r2_chunks) if r2_chunks else np.empty(0, dtype=np.float64),
        })
        if not cross_table:
            representatives, duplicate_rows = coordinate_maps[0]
            if duplicate_rows.size:
                # Match the matrix convention for copies of a coordinate.
                # Expansion below restores pairs between their input labels;
                # self-label pairs and reverse duplicates are removed later.
                repeated = np.unique(representatives[duplicate_rows])
                pairs = pd.concat([pairs, pd.DataFrame({
                    'row_a': repeated, 'row_b': repeated, 'R2': 1.0,
                })], ignore_index=True)
        for endpoint, (representatives, duplicate_rows) in zip(('row_a', 'row_b'), coordinate_maps):
            if duplicate_rows.size and not pairs.empty:
                # Expand each table's representatives independently, without
                # constructing a square or rectangular matrix.
                row_map = pd.DataFrame({
                    endpoint: representatives,
                    'original_row': np.arange(len(representatives), dtype=np.int64),
                })
                pairs = pairs.merge(row_map, on=endpoint, how='left')
                pairs[endpoint] = pairs.pop('original_row')

        row_a = pairs['row_a'].to_numpy()
        row_b = pairs['row_b'].to_numpy()
        if not cross_table:
            pairs['row_a'], pairs['row_b'] = np.minimum(
                row_a, row_b), np.maximum(row_a, row_b)
        if sort:
            pairs = pairs.sort_values(['row_a', 'row_b'], kind='stable')
        row_a = pairs['row_a'].to_numpy()
        row_b = pairs['row_b'].to_numpy()

        # Integer label codes let us canonicalize unordered pairs even with
        # mixed-type or repeated index labels, without comparing labels.
        codes = pd.factorize(site_info_df.index.to_numpy(), sort=False)[0]
        if cross_table:
            column_codes = pd.factorize(
                column_info_df.index.to_numpy(), sort=False)[0]
            first, second = row_a, row_b
        else:
            column_codes = codes
            swap = codes[row_a] > codes[row_b]
            first = np.where(swap, row_b, row_a)
            second = np.where(swap, row_a, row_b)
        pair_keys = pd.MultiIndex.from_arrays(
            [codes[first], column_codes[second]])
        keep = ~pair_keys.duplicated()
        if not cross_table:
            keep &= codes[first] != column_codes[second]
        return pd.DataFrame({
            'INDEX_1': site_info_df.index.take(first[keep]).to_numpy(),
            'INDEX_2': column_info_df.index.take(second[keep]).to_numpy(),
            'R2': pairs['R2'].to_numpy()[keep],
        })

def summarize_ld_decay_pairs(ld_pairs, site_info_df, *, max_decay_size=35000,
                             curve_bin_size=100):
    """Summarize unthresholded INDEX_1/INDEX_2/R2 pairs by chromosome.

    Pass pairs computed with min_r2=0; filtering low R2 before this step
    biases the mean upwards. Each unordered coordinate pair must occur once.
    Bins are (0, B], (B, 2B], ...; zero-distance pairs are excluded.
    Returns CHROM, bin_right, ld_sum, pair_count and ld_mean.
    """
    max_decay_size, curve_bin_size = index(max_decay_size), index(curve_bin_size)
    if min(max_decay_size, curve_bin_size) < 1:
        raise ValueError('Distance limit and bin size must be positive integers')
    if not site_info_df.index.is_unique:
        raise ValueError('site_info_df.index must uniquely identify sites')
    for frame, columns in ((ld_pairs, ['INDEX_1', 'INDEX_2', 'R2']),
                           (site_info_df, ['CHROM', 'POS'])):
        if not set(columns).issubset(frame.columns):
            raise ValueError(f'Required columns: {columns}')
    a = site_info_df.index.get_indexer(ld_pairs['INDEX_1'])
    b = site_info_df.index.get_indexer(ld_pairs['INDEX_2'])
    if (a < 0).any() or (b < 0).any():
        raise ValueError('LD pair labels are absent from site_info_df.index')
    positions = site_info_df['POS'].to_numpy(dtype=np.int64)
    chromosomes = site_info_df['CHROM'].to_numpy()
    distances = np.abs(positions[a] - positions[b])
    values = ld_pairs['R2'].to_numpy(dtype=float)
    keep = ((chromosomes[a] == chromosomes[b]) & (distances > 0)
            & (distances <= max_decay_size) & np.isfinite(values))
    if ((values[keep] < 0) | (values[keep] > 1)).any():
        raise ValueError('Finite R2 values must be in [0, 1]')
    data = pd.DataFrame({
        'CHROM': chromosomes[a[keep]],
        'bin_right': np.minimum(
            ((distances[keep] - 1) // curve_bin_size + 1) * curve_bin_size,
            max_decay_size),
        'R2': values[keep],
    })
    summary = data.groupby(['CHROM', 'bin_right'], sort=False, observed=True)['R2'].agg(
        ld_sum='sum', pair_count='size').reset_index()
    if not summary.empty:
        bin_right = np.minimum(
            np.arange(1, (max_decay_size + curve_bin_size - 1) // curve_bin_size + 1)
            * curve_bin_size, max_decay_size)
        grid = pd.MultiIndex.from_product(
            [data['CHROM'].unique(), bin_right], names=['CHROM', 'bin_right'])
        summary = summary.set_index(['CHROM', 'bin_right']).reindex(grid).reset_index()
        summary['ld_sum'] = summary['ld_sum'].fillna(0.0)
        summary['pair_count'] = summary['pair_count'].fillna(0).astype(np.int64)
    summary['ld_mean'] = summary['ld_sum'] / summary['pair_count'].replace(0, np.nan)
    return summary


def calculate_ld_decay(site_info_df, input_vcf_file, *, max_decay_size=35000,
                       curve_bin_size=100, pool_num=1, show_progress=True):
    """Calculate chromosome-level LD decay statistics from genotype dosages.

    Workflow
    --------
    1. Collapse duplicate coordinates and process each chromosome separately.
    2. Calculate each distinct coordinate pair within the distance limit once.
    3. Summarize finite squared correlations in distance bins.

    Parameters
    ----------
    site_info_df : pandas.DataFrame
        Retained sites with CHROM and positive integer POS (1-based bp),
        matching the input VCF. Chromosomes retain their encounter order.
    input_vcf_file : str
        VCF containing the selected sites and analysis samples.
    max_decay_size : int
        Largest included pair distance in bp; zero-distance pairs are excluded.
    curve_bin_size : int
        Bin width in bp, with intervals (0, B], (B, 2B], and so on.
    pool_num : int
        LD worker count. All finite pairs for one chromosome are held in
        memory before aggregation; choose resources accordingly.
    show_progress : bool
        Display chromosome and backend computation progress.

    Returns
    -------
    pandas.DataFrame
        CHROM, bin_right (bp), ld_sum, pair_count and ld_mean. Undefined
        correlations are excluded. Empty bins have zero count and NaN mean.
        Missing-call handling follows calculate_ld_matrix. No files are saved.
    """
    max_decay_size, curve_bin_size = index(max_decay_size), index(curve_bin_size)
    pool_num = index(pool_num)
    if min(max_decay_size, curve_bin_size, pool_num) < 1:
        raise ValueError('Distance, bin and worker settings must be positive integers')
    sites = site_info_df[['CHROM', 'POS']].copy()
    if sites.isna().any().any():
        raise ValueError('CHROM and POS must not contain missing values')
    if not pd.api.types.is_integer_dtype(sites['POS']) or (sites['POS'] < 1).any():
        raise ValueError('POS must contain positive integer coordinates')
    sites = sites.drop_duplicates(['CHROM', 'POS']).reset_index(drop=True)
    n_bins = (max_decay_size + curve_bin_size - 1) // curve_bin_size
    bin_right = np.minimum(np.arange(1, n_bins + 1) * curve_bin_size, max_decay_size)
    chromosome_summaries = []
    for chr_id, chromosome in sites.groupby('CHROM', sort=False, observed=True):
        chromosome = chromosome.sort_values('POS', kind='stable')
        if show_progress:
            print(f'LD decay {chr_id}: calculating windowed LD for {len(chromosome)} sites',
                  flush=True)
        pairs = calculate_ld_matrix(
            chromosome, input_vcf_file,
            ld_window_size=max_decay_size + 1, output_format='pairs',
            min_r2=0, pool_num=pool_num, show_progress=show_progress,
        )
        summary = summarize_ld_decay_pairs(
            pairs, chromosome, max_decay_size=max_decay_size,
            curve_bin_size=curve_bin_size,
        )
        del pairs
        # Retain empty bins/chromosomes as zero counts and undefined means.
        summary = summary.set_index('bin_right').reindex(bin_right)
        summary.index.name = 'bin_right'
        summary['CHROM'] = chr_id
        summary['ld_sum'] = summary['ld_sum'].fillna(0.0)
        summary['pair_count'] = summary['pair_count'].fillna(0).astype(np.int64)
        summary['ld_mean'] = summary['ld_sum'] / summary['pair_count'].replace(0, np.nan)
        chromosome_summaries.append(summary.reset_index()[
            ['CHROM', 'bin_right', 'ld_sum', 'pair_count', 'ld_mean']])
    if not chromosome_summaries:
        return pd.DataFrame(columns=['CHROM', 'bin_right', 'ld_sum', 'pair_count', 'ld_mean'])
    return pd.concat(chromosome_summaries, ignore_index=True)


def plot_ld_decay(ld_decay_df, *, half_ld_value=0.5, figsize=(12, 9),
                  xlim=(-1000, 35000), save_fig_path=None, ax=None):
    """Plot saved LD decay statistics and interpolate threshold distances.

    Parameters
    ----------
    ld_decay_df : pandas.DataFrame
        Binned statistics with CHROM, bin_right (bp), ld_mean and pair_count,
        as returned by calculate_ld_decay. Each chromosome/bin occurs once.
    half_ld_value : float
        Absolute mean r² threshold in (0, 1), rather than half the first bin.
    figsize : tuple of float
        Figure width and height in inches when creating a new axes.
    xlim : tuple of float
        Display bounds for distance in bp.
    save_fig_path : str or None
        Optional figure path; an existing file is overwritten.
    ax : matplotlib.axes.Axes or None
        Existing axes, or None to create a figure.

    Returns
    -------
    tuple
        Figure, axes and threshold table with CHROM, half_ld_value,
        half_ld_dist (bp) and status. The first downward crossing is linearly
        interpolated between adjacent observed bins; gaps are not bridged.
        Missing crossings have NaN distance and a diagnostic status.
        The plotted (0, 1) origin is a display convention, not an observed
        bin, and is excluded from threshold estimation. No LD is recomputed.
    """
    half_ld_value = float(half_ld_value)
    if not np.isfinite(half_ld_value) or not 0 < half_ld_value < 1:
        raise ValueError('half_ld_value must be in (0, 1)')
    required = ['CHROM', 'bin_right', 'ld_mean', 'pair_count']
    if not set(required).issubset(ld_decay_df.columns) or ld_decay_df.empty:
        raise ValueError(f'Expected nonempty decay data with columns {required}')
    if ld_decay_df.duplicated(['CHROM', 'bin_right']).any():
        raise ValueError('Each chromosome/bin must appear once')
    rows = []
    with plt.rc_context():
        set_plot_style()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        for chr_id, curve in ld_decay_df.groupby('CHROM', sort=False, observed=True):
            curve = curve.sort_values('bin_right')
            x = curve['bin_right'].to_numpy(dtype=float)
            y = curve['ld_mean'].to_numpy(dtype=float).copy()
            y[curve['pair_count'].to_numpy() <= 0] = np.nan
            finite = np.flatnonzero(np.isfinite(y))
            distance, status = np.nan, 'no_finite_pairs'
            if finite.size:
                first = finite[0]
                if y[first] == half_ld_value:
                    distance, status = x[first], 'reached'
                elif y[first] < half_ld_value:
                    status = 'below_in_first_bin'
                else:
                    status = 'not_reached'
                    for i in range(first + 1, len(x)):
                        if (np.isfinite(y[i - 1]) and np.isfinite(y[i])
                                and y[i - 1] > half_ld_value >= y[i]):
                            distance = x[i - 1] + (half_ld_value - y[i - 1]) * (
                                x[i] - x[i - 1]) / (y[i] - y[i - 1])
                            status = 'reached'
                            break
            if np.isfinite(distance):
                suffix = f'{int(distance)} bp'
            elif status == 'below_in_first_bin':
                suffix = f'<{x[finite[0]]:g} bp'
            else:
                suffix = {
                    'not_reached': 'not reached',
                    'no_finite_pairs': 'no finite pairs',
                }[status]
            ax.plot(np.r_[0, x], np.r_[1, y], marker='o', markersize=3,
                    label=f'{chr_id}: {suffix}')
            rows.append({'CHROM': chr_id, 'half_ld_value': half_ld_value,
                         'half_ld_dist': distance, 'status': status})
        ax.axhline(half_ld_value, color='gray', linestyle='dashed')
        ax.set_xlabel('Distance (bp)')
        ax.set_ylabel('Linkage Disequilibrium (r²)')
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.legend(loc='upper right',
                  title=f'LD decay distance\n(r² = {half_ld_value:g})')
        ax.grid(True)
        if save_fig_path is not None:
            fig.savefig(save_fig_path, facecolor='none', edgecolor='none', bbox_inches='tight')
    return fig, ax, pd.DataFrame(rows)


# Multiscale bins use (left, right]; the final bin has no upper bound.
_MULTISCALE_LD_EDGES = np.array([
    0, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 75000,
    100000, 125000, 150000, 175000, 200000, 225000, 250000, 300000,
    400000, 500000, 600000, 700000, 800000, 900000, 1000000,
    1500000, 2000000, 2500000, 3000000, 5000000, 10000000,
    15000000, 20000000, 30000000, 50000000, np.inf,
], dtype=float)
_MULTISCALE_LD_LABELS = [
    '0–0.5 kb', '0.5–1 kb', '1–2 kb', '2–5 kb', '5–10 kb',
    '10–20 kb', '20–30 kb', '30–50 kb', '50–75 kb', '75–100 kb',
    '100–125 kb', '125–150 kb', '150–175 kb', '175–200 kb',
    '200–225 kb', '225–250 kb', '250–300 kb', '300–400 kb',
    '400–500 kb', '500–600 kb', '600–700 kb', '700–800 kb',
    '800–900 kb', '900 kb–1 Mb', '1–1.5 Mb', '1.5–2 Mb',
    '2–2.5 Mb', '2.5–3 Mb', '3–5 Mb', '5–10 Mb', '10–15 Mb',
    '15–20 Mb', '20–30 Mb', '30–50 Mb', '>50 Mb',
]


def _multiscale_pair_bounds(positions, left, right):
    """Implicit unordered-pair population; allocate O(number of sites)."""
    lower = np.searchsorted(positions, positions + int(left), side='right')
    upper = (np.full(len(positions), len(positions), dtype=np.int64)
             if np.isinf(right) else
             np.searchsorted(positions, positions + int(right), side='right'))
    counts = upper - lower
    return lower, np.cumsum(counts, dtype=np.int64)


def _multiscale_sample_ranks(total, count, rng):
    """Floyd sampling: uniform without replacement, O(count) memory."""
    selected = set()
    for stop in range(total - count, total):
        draw = int(rng.integers(stop + 1))
        selected.add(stop if draw in selected else draw)
    return np.array(sorted(selected), dtype=np.int64)


def _plan_multiscale_ld_pairs(positions, chr_seed, random_seed, full_distance, max_pairs):
    """Count implicit populations and select only bounded long-range pairs."""
    populations = np.zeros(len(_MULTISCALE_LD_LABELS), dtype=np.int64)
    evaluated = np.zeros_like(populations)
    long_a, long_b, long_bins = [], [], []
    for bin_id, (left, right) in enumerate(zip(
            _MULTISCALE_LD_EDGES[:-1], _MULTISCALE_LD_EDGES[1:])):
        lower, cumulative = _multiscale_pair_bounds(positions, left, right)
        total = int(cumulative[-1]) if len(cumulative) else 0
        populations[bin_id] = total
        number = min(total, max_pairs) if right > full_distance else total
        evaluated[bin_id] = number
        if right <= full_distance or not number:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([random_seed, chr_seed, bin_id]))
        ranks = (_multiscale_sample_ranks(total, number, rng) if number < total
                 else np.arange(number, dtype=np.int64))
        a = np.searchsorted(cumulative, ranks, side='right')
        before = np.where(a > 0, cumulative[np.maximum(a - 1, 0)], 0)
        long_a.append(a)
        long_b.append(lower[a] + ranks - before)
        long_bins.append(np.full(number, bin_id, dtype=np.int64))
    empty = np.empty(0, dtype=np.int64)
    return populations, evaluated, tuple(
        np.concatenate(parts) if parts else empty for parts in (long_a, long_b, long_bins))


def _iter_multiscale_ld_tasks(positions, long_pairs, full_distance,
                              matrix_block_size, pair_batch_size):
    # Exhaustive local pairs: only the upper triangle is emitted, across all
    # short-distance bins at once, rather than rescanning pairs for every bin.
    near_right = np.searchsorted(positions, positions + full_distance, side='right')
    for start in range(0, len(positions), matrix_block_size):
        end = min(start + matrix_block_size, len(positions))
        if (near_right[start:end] > np.arange(start, end) + 1).any():
            yield ('short', start, end)
    a, b, bin_ids = long_pairs
    # Cluster all long-distance bins by lead-coordinate block for page reuse.
    order = np.argsort(a // matrix_block_size, kind='stable')
    for start in range(0, len(order), pair_batch_size):
        rows = order[start:start + pair_batch_size]
        yield ('long', a[rows], b[rows], bin_ids[rows])


def _calculate_multiscale_ld_batch(task, positions, normalized, full_distance,
                                  matrix_block_size):
    n_bins = len(_MULTISCALE_LD_LABELS)
    attempted = np.zeros(n_bins, dtype=np.int64)
    counts = np.zeros(n_bins, dtype=np.int64)
    sums = np.zeros(n_bins)
    distance_min = np.full(n_bins, np.inf)
    distance_max = np.full(n_bins, -np.inf)

    def aggregate(bin_ids, r2, distances):
        # Distance QC is requested only for the last two bins; avoid adding
        # indexed min/max work to billions of exhaustive short-range pairs.
        tail = bin_ids >= n_bins - 2
        if tail.any():
            np.minimum.at(distance_min, bin_ids[tail], distances[tail])
            np.maximum.at(distance_max, bin_ids[tail], distances[tail])
        attempted[:] += np.bincount(bin_ids, minlength=n_bins)
        finite = np.isfinite(r2)
        counts[:] += np.bincount(bin_ids[finite], minlength=n_bins)
        sums[:] += np.bincount(bin_ids[finite], weights=r2[finite], minlength=n_bins)

    if task[0] == 'long':
        _, a, b, bin_ids = task
        distances = np.abs(positions[b] - positions[a])
        if ((bin_ids < 0) | (bin_ids >= n_bins)).any():
            raise ValueError('Invalid multiscale distance bin index')
        eligible = ((a < b) & (distances > _MULTISCALE_LD_EDGES[bin_ids])
                    & (distances <= _MULTISCALE_LD_EDGES[bin_ids + 1]))
        if not eligible.all():
            raise ValueError('Long-range LD pair is outside its explicit (left, right] bin')
        r2 = np.einsum('ij,ij->i', normalized[a], normalized[b])
        np.clip(r2, -1, 1, out=r2)
        np.square(r2, out=r2)
        aggregate(bin_ids, r2, distances)
    else:
        _, start, end = task
        query_indices = np.arange(start, end)
        query = normalized[start:end]
        stop = np.searchsorted(positions, positions[end - 1] + full_distance, side='right')
        # Bound temporary distance/R2 matrices, including very dense SNP regions.
        for left in range(start + 1, stop, matrix_block_size * 8):
            right = min(left + matrix_block_size * 8, stop)
            subject_indices = np.arange(left, right)
            distances = positions[subject_indices][None, :] - positions[query_indices][:, None]
            eligible = ((subject_indices[None, :] > query_indices[:, None])
                        & (distances > 0) & (distances <= full_distance))
            if not eligible.any():
                continue
            r2 = query @ normalized[left:right].T
            np.clip(r2, -1, 1, out=r2)
            np.square(r2, out=r2)
            bin_ids = np.searchsorted(_MULTISCALE_LD_EDGES, distances[eligible], side='left') - 1
            aggregate(bin_ids, r2[eligible], distances[eligible])
    return attempted, counts, sums, distance_min, distance_max


_multiscale_worker_state = None
_multiscale_worker_error = None


def _close_multiscale_ld_worker():
    global _multiscale_worker_state
    if _multiscale_worker_state is not None:
        matrix, limiter = _multiscale_worker_state[1], _multiscale_worker_state[-1]
        matrix._mmap.close()
        limiter.restore_original_limits()
        _multiscale_worker_state = None


def _init_multiscale_ld_worker(filename, shape, positions, full_distance, matrix_block_size):
    global _multiscale_worker_state, _multiscale_worker_error
    _multiscale_worker_state = None
    _multiscale_worker_error = None
    limiter = None
    try:
        limiter = threadpool_limits(limits=1, user_api='blas')
        matrix = np.memmap(filename, dtype=np.float64, mode='r', shape=shape)
        _multiscale_worker_state = (positions, matrix, full_distance, matrix_block_size, limiter)
        Finalize(None, _close_multiscale_ld_worker, exitpriority=10)
    except Exception as error:
        if limiter is not None:
            limiter.restore_original_limits()
        _multiscale_worker_error = str(error)


def _run_multiscale_ld_batch(task):
    if _multiscale_worker_error is not None:
        raise RuntimeError(f'Could not initialize multiscale LD worker: {_multiscale_worker_error}')
    return _calculate_multiscale_ld_batch(task, *_multiscale_worker_state[:-1])


def _load_multiscale_ld_result(data):
    """A standalone CSV is sufficient for plotting, without input VCF/JSON."""
    if isinstance(data, (str, os.PathLike)):
        data = pd.read_csv(data)
    required = {'CHROM', 'BIN_LEFT', 'BIN_RIGHT', 'BIN_LABEL', 'PAIR_COUNT', 'LD_MEAN'}
    if not required.issubset(data.columns):
        raise ValueError(f'Multiscale LD result requires columns {sorted(required)}')
    return data


def calculate_multiscale_ld_decay(
    site_info_df, input_vcf_file, *, min_maf=None, max_het=None,
    full_distance=200000, max_pairs_per_bin=50000, random_seed=42,
    pool_num=1, pair_batch_size=4096, cache_max_tiles=4,
    show_progress=True, silence=False, progress_interval=2.0,
    matrix_block_size=128, scratch_dir=None, cache_file=None, force_recompute=False,
):
    """Exhaustively evaluate short-range LD and sample long-range LD.

    Workflow
    --------
    1. Reuse saved statistics, or select sites and plan chromosome-wise pairs.
    2. Normalize genotypes into temporary disk arrays shared by workers.
    3. Evaluate pairs, aggregate finite r² by distance bin, and save statistics.

    Parameters
    ----------
    site_info_df : pandas.DataFrame or str
        Sites with CHROM and positive integer POS in bp, or a chromosome-keyed
        variant-statistics HDF file. MAF/HETF are required when filtering.
    input_vcf_file : str
        VCF matching the selected coordinates; multiple records per selected
        coordinate are rejected.
    min_maf, max_het : float or None
        Inclusive MAF lower bound and heterozygosity upper bound.
    full_distance : int
        Exhaustive distance limit in bp; must be a finite predefined bin edge.
    max_pairs_per_bin : int
        Maximum uniformly sampled pairs without replacement per distant bin.
    random_seed : int
        Reproducible long-range pair sampling seed.
    pool_num : int
        Worker count; parallel execution requires fork and one BLAS thread
        per worker. No dense chromosome LD matrix is retained.
    pair_batch_size, matrix_block_size, cache_max_tiles : int
        Pair batch size, exhaustive matrix-block size and VCF tile-cache limit.
    show_progress, silence : bool
        Enable progress; silence overrides show_progress.
    progress_interval : float
        Seconds between pair-count, throughput and ETA messages.
    scratch_dir : str or None
        Temporary-array directory, or the system default. Storage is about
        8*n_sites*n_samples bytes per chromosome and removed after processing.
    cache_file : str or None
        Optional statistics CSV path. Existing CSV is loaded without checking
        current inputs or parameters; completed output is written atomically.
    force_recompute : bool
        Recompute and replace the CSV after input or parameter changes.

    Returns
    -------
    pandas.DataFrame
        Chromosome/bin statistics with bounds in bp, BIN_LABEL, LD_MEAN,
        PAIR_COUNT (finite r²), EVALUATED_PAIR_COUNT (attempted pairs) and
        POPULATION_PAIR_COUNT (eligible pairs). Undefined r² is excluded;
        empty means are NaN. PAIR_DISTANCE_MIN/MAX record attempted distance
        extrema for the final two bins, including undefined r² pairs.
        The >50 Mb bin requires distance >50,000,000 bp.
    """
    import hashlib
    import math
    import tempfile

    verbose = bool(show_progress) and not silence
    if cache_file is not None and not force_recompute and os.path.isfile(cache_file):
        if verbose:
            print(f'Reading existing multiscale LD CSV: {cache_file}', flush=True)
        return _load_multiscale_ld_result(cache_file)
    full_distance = index(full_distance)
    max_pairs_per_bin, random_seed = index(max_pairs_per_bin), index(random_seed)
    pool_num, pair_batch_size = index(pool_num), index(pair_batch_size)
    cache_max_tiles, matrix_block_size = index(cache_max_tiles), index(matrix_block_size)
    if (full_distance not in _MULTISCALE_LD_EDGES[1:-1]
            or min(max_pairs_per_bin, pool_num, pair_batch_size, cache_max_tiles, matrix_block_size) < 1
            or random_seed < 0):
        raise ValueError('Use a finite bin edge, positive sizes, and seed >= 0')
    if verbose and (not np.isfinite(progress_interval) or progress_interval <= 0):
        raise ValueError('progress_interval must be finite and positive')
    if pool_num > 1 and mp.get_start_method() != 'fork':
        raise RuntimeError("Parallel LD requires 'fork'; use pool_num=1 in this kernel.")
    start_time = perf_counter()
    if verbose:
        print('Multiscale LD: loading and selecting sites', flush=True)
    if isinstance(site_info_df, (str, os.PathLike)):
        with pd.HDFStore(site_info_df, mode='r') as store:
            site_info_df = pd.concat([store[key] for key in store.keys()], ignore_index=True)
    sites = site_info_df.copy()
    for column, cutoff, comparison in [('MAF', min_maf, 'min'), ('HETF', max_het, 'max')]:
        if cutoff is not None:
            cutoff = float(cutoff)
            if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
                raise ValueError(f'{column} cutoff must be finite and in [0, 1]')
            sites = sites.loc[sites[column] >= cutoff] if comparison == 'min' else sites.loc[sites[column] <= cutoff]
    sites = sites[['CHROM', 'POS']].copy()
    if sites.isna().any().any() or not pd.api.types.is_integer_dtype(sites['POS']) or (sites['POS'] < 1).any():
        raise ValueError('Require nonmissing CHROM and positive integer POS')
    sites['CHROM'] = sites['CHROM'].astype(str)
    sites = sites.drop_duplicates().sort_values(['CHROM', 'POS']).reset_index(drop=True)
    if sites.empty:
        raise ValueError('No sites remain for multiscale LD')
    rows = []
    n_chromosomes = sites['CHROM'].nunique()
    for chr_number, (chr_id, chromosome) in enumerate(
            sites.groupby('CHROM', sort=False, observed=True), 1):
        positions = chromosome['POS'].to_numpy(dtype=np.int64)
        chromosomes = [(chr_id, positions, np.arange(len(positions)))]
        chr_seed = int.from_bytes(hashlib.sha256(chr_id.encode()).digest()[:4], 'little')
        populations, expected, long_pairs = _plan_multiscale_ld_pairs(
            positions, chr_seed, random_seed, full_distance, max_pairs_per_bin)
        total_pairs = int(expected.sum())
        n_bins = len(_MULTISCALE_LD_LABELS)
        evaluated, counts = np.zeros(n_bins, dtype=np.int64), np.zeros(n_bins, dtype=np.int64)
        batch_sums = [[] for _ in range(n_bins)]
        distance_min = np.full(n_bins, np.inf)
        distance_max = np.full(n_bins, -np.inf)
        chr_start = perf_counter()
        if verbose:
            print(f'[{chr_number}/{n_chromosomes}] {chr_id}: {len(positions):,} sites; '
                  f'{total_pairs:,} pairs planned; elapsed {chr_start - start_time:.1f}s', flush=True)
        if total_pairs:
            with tempfile.TemporaryDirectory(prefix='multiscale_ld_', dir=scratch_dir) as temporary:
                reader = VCF(input_vcf_file)
                matrix = None
                try:
                    n_samples = len(reader.samples)
                    if n_samples < 2:
                        raise ValueError('LD requires at least two samples')
                    filename = os.path.join(temporary, 'normalized.f64')
                    matrix = np.memmap(filename, dtype=np.float64, mode='w+',
                                       shape=(len(positions), n_samples))
                    cache = _LDPruningGenotypeCache(
                        chromosomes, reader, full_distance, max_tiles=cache_max_tiles)
                    tiles = (positions - 1) // full_distance
                    unique_tiles = np.unique(tiles)
                    last_report = perf_counter()
                    for tile in unique_tiles:
                        # Sorted coordinates make rows for one tile contiguous.
                        left = np.searchsorted(tiles, tile)
                        right = np.searchsorted(tiles, tile, side='right')
                        matrix[left:right] = np.nan
                        _, normalized, groups = cache.get(0, tile)
                        for row in range(left, right):
                            records = groups.get(int(positions[row]), [])
                            if len(records) > 1:
                                raise ValueError(f'Multiple VCF records at {chr_id}:{positions[row]}')
                            if records:
                                matrix[row] = normalized[records[0]]
                        now = perf_counter()
                        if verbose and (now - last_report >= progress_interval or right == len(positions)):
                            print(f'{chr_id} normalize: {right:,}/{len(positions):,} sites '
                                  f'({100 * right / len(positions):.1f}%); '
                                  f'elapsed {now - chr_start:.1f}s', flush=True)
                            last_report = now
                    matrix.flush()
                    del cache
                finally:
                    reader.close()
                    if matrix is not None:
                        matrix._mmap.close()
                shape = (len(positions), n_samples)
                matrix = np.memmap(filename, dtype=np.float64, mode='r', shape=shape)
                tasks = _iter_multiscale_ld_tasks(
                    positions, long_pairs, full_distance, matrix_block_size, pair_batch_size)
                pair_start = last_report = perf_counter()
                pool = None
                try:
                    if pool_num == 1:
                        results = (_calculate_multiscale_ld_batch(
                            task, positions, matrix, full_distance, matrix_block_size) for task in tasks)
                    else:
                        pool = mp.Pool(
                            processes=pool_num, initializer=_init_multiscale_ld_worker,
                            initargs=(filename, shape, positions, full_distance, matrix_block_size))
                        results = _iter_parallel_ld_results(
                            pool, tasks, pool_num, progress_interval if verbose else None,
                            worker=_run_multiscale_ld_batch)
                    for result in results:
                        if result is not None:
                            attempted, finite, sums, batch_min, batch_max = result
                            distance_min = np.minimum(distance_min, batch_min)
                            distance_max = np.maximum(distance_max, batch_max)
                            evaluated += attempted
                            counts += finite
                            for bin_id in np.flatnonzero(finite):
                                batch_sums[bin_id].append(float(sums[bin_id]))
                        now = perf_counter()
                        if verbose and (now - last_report >= progress_interval
                                        or int(evaluated.sum()) == total_pairs):
                            done = int(evaluated.sum())
                            elapsed = now - pair_start
                            rate = done / elapsed if elapsed else 0
                            eta = f'{(total_pairs - done) / rate:.1f}s' if rate else '--'
                            print(f'{chr_id} LD: {done:,}/{total_pairs:,} pairs '
                                  f'({100 * done / total_pairs:.1f}%); elapsed {now - chr_start:.1f}s; '
                                  f'{rate:,.0f} pairs/s; ETA {eta}', flush=True)
                            last_report = now
                    if pool is not None:
                        pool.close()
                        pool.join()
                except BaseException:
                    if pool is not None:
                        pool.terminate()
                        pool.join()
                    raise
                finally:
                    matrix._mmap.close()
        if not np.array_equal(evaluated, expected):
            raise RuntimeError(f'Pair-count mismatch while computing {chr_id}')
        for bin_id, label in enumerate(_MULTISCALE_LD_LABELS):
            rows.append({
                'CHROM': chr_id, 'BIN_LEFT': _MULTISCALE_LD_EDGES[bin_id],
                'BIN_RIGHT': _MULTISCALE_LD_EDGES[bin_id + 1], 'BIN_LABEL': label,
                'PAIR_COUNT': counts[bin_id],
                'LD_MEAN': math.fsum(batch_sums[bin_id]) / counts[bin_id] if counts[bin_id] else np.nan,
                'EVALUATED_PAIR_COUNT': evaluated[bin_id],
                'PAIR_DISTANCE_MIN': distance_min[bin_id] if np.isfinite(distance_min[bin_id]) else np.nan,
                'PAIR_DISTANCE_MAX': distance_max[bin_id] if np.isfinite(distance_max[bin_id]) else np.nan,
                'POPULATION_PAIR_COUNT': populations[bin_id],
                'SAMPLED': bool(_MULTISCALE_LD_EDGES[bin_id + 1] > full_distance
                                and populations[bin_id] > max_pairs_per_bin),
            })
        if verbose:
            print(f'{chr_id} complete: {perf_counter() - chr_start:.1f}s; '
                  f'total elapsed {perf_counter() - start_time:.1f}s', flush=True)
    result = pd.DataFrame(rows)
    if cache_file is not None:
        output = os.fspath(cache_file)
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        # Do not expose a partial CSV to a separate plotting process.
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(output) or '.', prefix=os.path.basename(output) + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary_output = handle.name
        try:
            result.to_csv(temporary_output, index=False,
                          compression='gzip' if str(output).endswith('.gz') else None)
            os.replace(temporary_output, output)
        finally:
            try:
                os.remove(temporary_output)
            except FileNotFoundError:
                pass
        if verbose:
            print(f'Saved {output}; total elapsed {perf_counter() - start_time:.1f}s', flush=True)
    return result


def summarize_multiscale_ld_overflow(ld_decay_df):
    """Inspect attempted distances in the two longest-range bins.

    Parameters
    ----------
    ld_decay_df : pandas.DataFrame or str
        Multiscale LD statistics or their saved CSV path.

    Returns
    -------
    pandas.DataFrame
        Rows for 30–50 Mb and >50 Mb with candidate, attempted and finite
        pair counts, mean r², attempted distance extrema in bp and DISTANCE_QC.
        Missing distance records yield not_recorded; no pairs yield no_pairs;
        recorded distances yield pass or out_of_bin. No LD is recomputed.
    """
    data = _load_multiscale_ld_result(ld_decay_df)
    result = data.loc[data['BIN_LABEL'].isin(['30–50 Mb', '>50 Mb'])].copy()
    columns = ['CHROM', 'BIN_LABEL', 'POPULATION_PAIR_COUNT',
               'EVALUATED_PAIR_COUNT', 'PAIR_COUNT', 'LD_MEAN',
               'PAIR_DISTANCE_MIN', 'PAIR_DISTANCE_MAX']
    result = result.reindex(columns=columns)
    recorded = (np.isfinite(result['PAIR_DISTANCE_MIN'])
                & np.isfinite(result['PAIR_DISTANCE_MAX']))
    lower = np.where(result['BIN_LABEL'] == '>50 Mb', 50000000, 30000000)
    upper = np.where(result['BIN_LABEL'] == '>50 Mb', np.inf, 50000000)
    valid = ((result['PAIR_DISTANCE_MIN'] > lower)
             & (result['PAIR_DISTANCE_MAX'] <= upper)
             & (result['PAIR_DISTANCE_MIN'] <= result['PAIR_DISTANCE_MAX']))
    result['DISTANCE_QC'] = np.where(
        result['EVALUATED_PAIR_COUNT'] == 0, 'no_pairs',
        np.where(~recorded, 'not_recorded', np.where(valid, 'pass', 'out_of_bin')))
    return result.reset_index(drop=True)


def summarize_ld_decay_thresholds(ld_decay_df, thresholds=(0.2, 0.1)):
    """Find the first observed distance bin below each mean r² threshold.

    Parameters
    ----------
    ld_decay_df : pandas.DataFrame or str
        Multiscale LD statistics or their saved CSV path.
    thresholds : tuple of float
        Absolute mean r² thresholds in (0, 1).

    Returns
    -------
    pandas.DataFrame
        CHROM, THRESHOLD, BIN_LEFT/RIGHT in bp, BIN_LABEL and STATUS.
        Uses only finite means with positive pair counts; no interpolation.
        Missing crossings retain NaN bounds and a diagnostic status.
    """
    ld_decay_df = _load_multiscale_ld_result(ld_decay_df)
    rows = []
    for chr_id, curve in ld_decay_df.groupby('CHROM', sort=False, observed=True):
        curve = curve.sort_values('BIN_LEFT')
        valid = (curve['PAIR_COUNT'] > 0) & np.isfinite(curve['LD_MEAN'])
        for threshold in thresholds:
            threshold = float(threshold)
            if not np.isfinite(threshold) or not 0 < threshold < 1:
                raise ValueError('Thresholds must be finite and in (0, 1)')
            below = curve.loc[valid & (curve['LD_MEAN'] < threshold)]
            first = below.iloc[0] if not below.empty else None
            rows.append({
                'CHROM': chr_id, 'THRESHOLD': threshold,
                'BIN_LEFT': first['BIN_LEFT'] if first is not None else np.nan,
                'BIN_RIGHT': first['BIN_RIGHT'] if first is not None else np.nan,
                'BIN_LABEL': first['BIN_LABEL'] if first is not None else None,
                'STATUS': ('below_threshold' if first is not None else
                           'not_reached' if valid.any() else 'no_finite_pairs'),
            })
    return pd.DataFrame(rows)


def plot_multiscale_ld_decay(ld_decay_df, *, figsize=(12, 9), save_fig_path=None,
                             ax=None):
    """Plot chromosome-wise multiscale LD means at equal bin spacing.

    Parameters
    ----------
    ld_decay_df : pandas.DataFrame or str
        Multiscale LD statistics or their saved CSV path.
    figsize : tuple of float
        Figure width and height in inches when creating a new axes.
    save_fig_path : str or None
        Optional figure path; an existing figure is overwritten.
    ax : matplotlib.axes.Axes or None
        Existing axes, or None to create a figure.

    Returns
    -------
    tuple
        Figure and axes. Horizontal spacing is not proportional to physical
        distance; labels show bin right edges, with >50 MB for the final bin.
        The (0, 1) origin is display-only and does not change input statistics
        or threshold summaries. Applies shared style; no LD is recomputed.
    """
    ld_decay_df = _load_multiscale_ld_result(ld_decay_df)
    if ld_decay_df.empty or ld_decay_df.duplicated(['CHROM', 'BIN_LABEL']).any():
        raise ValueError('Require nonempty, unique chromosome/bin rows')
    if not ld_decay_df['BIN_LABEL'].isin(_MULTISCALE_LD_LABELS).all():
        raise ValueError('Unknown multiscale distance bin')
    with plt.rc_context():
        set_plot_style()
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        x = np.arange(len(_MULTISCALE_LD_LABELS) + 1)
        tick_labels = ['0'] + [
            '>50 MB' if np.isinf(right) else
            f'{right / 1_000_000:g}MB' if right >= 1_000_000 else
            f'{right / 1_000:g}K'
            for right in _MULTISCALE_LD_EDGES[1:]
        ]
        for chr_id, curve in ld_decay_df.groupby('CHROM', sort=False, observed=True):
            curve = curve.set_index('BIN_LABEL').reindex(_MULTISCALE_LD_LABELS)
            values = curve['LD_MEAN'].where(curve['PAIR_COUNT'] > 0)
            ax.plot(x, np.r_[1, values.to_numpy(dtype=float)], marker='o',
                    markersize=3, label=chr_id)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=90, ha='center', fontsize=10)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel('Distance')
        ax.set_ylabel('Linkage Disequilibrium (r²)')
        ax.grid(True)
        ax.legend(loc='upper right')
        fig.tight_layout()
        if save_fig_path is not None:
            fig.savefig(save_fig_path, facecolor='none', edgecolor='none', bbox_inches='tight')
    return fig, ax


def weighted_ld_jaccard_sparse(
    r2_df,
    foreground_index,
    background_index,
):
    """
    Sparse weighted LD-Jaccard.

    r2_df requires:
        INDEX_1   foreground site index
        INDEX_2   background site index
        R2_scale  scaled LD strength in (0, 1]

    R2_scale:
        R2 <= 0.5 -> 0
        R2 = 0.75 -> 0.5
        R2 = 1.0 -> 1.0

    Returns
    -------
    weighted_jaccard : float
    intersection : float
        Sum of R2_scale over the optimal one-to-one FG-BG matching.
    matched_pairs : int
        Number of real FG-BG pairs selected in the optimal matching.
    """

    fg_ids = pd.Index(foreground_index)
    bg_ids = pd.Index(background_index)

    if not fg_ids.is_unique:
        raise ValueError("foreground_index must be unique")

    if not bg_ids.is_unique:
        raise ValueError("background_index must be unique")

    n_fg = len(fg_ids)
    n_bg = len(bg_ids)

    if n_fg == 0 or n_bg == 0:
        return 0.0, 0.0, 0

    # Only positive-weight LD edges matter
    edges = r2_df.loc[
        r2_df["R2_scale"] > 0,
        ["INDEX_1", "INDEX_2", "R2_scale"]
    ].copy()

    if edges.empty:
        return 0.0, 0.0, 0

    # Safety: each FG-BG pair should occur only once.
    # If duplicates exist, retain its strongest LD value.
    if edges.duplicated(["INDEX_1", "INDEX_2"]).any():
        edges = (
            edges
            .groupby(
                ["INDEX_1", "INDEX_2"],
                sort=False,
                as_index=False
            )["R2_scale"]
            .max()
        )

    # Convert DataFrame index labels into matrix row/column numbers
    row = fg_ids.get_indexer(edges["INDEX_1"])
    col = bg_ids.get_indexer(edges["INDEX_2"])

    valid = (row >= 0) & (col >= 0)

    row = row[valid]
    col = col[valid]

    weight = edges.loc[
        valid, "R2_scale"
    ].to_numpy(dtype=np.float64)

    # ---------------------------------------------------------
    # Maximum weight -> minimum cost
    #
    # real edge:
    #     cost = 2 - weight
    #
    # Therefore:
    #     weight=1 -> cost=1
    #     weight=0.5 -> cost=1.5
    #
    # unmatched:
    #     represented by dummy edge with cost=2
    #
    # Minimizing total cost is therefore exactly equivalent to
    # maximizing total matched R2_scale.
    # ---------------------------------------------------------

    real_cost = 2.0 - weight

    # Give every foreground site its own dummy background node.
    #
    # This allows a foreground site to remain unmatched without
    # constructing all zero FG x BG pairs.
    dummy_row = np.arange(n_fg, dtype=np.int64)
    dummy_col = n_bg + dummy_row
    dummy_cost = np.full(n_fg, 2.0, dtype=np.float64)

    rows = np.concatenate([row, dummy_row])
    cols = np.concatenate([col, dummy_col])
    costs = np.concatenate([real_cost, dummy_cost])

    cost_matrix = coo_matrix(
        (costs, (rows, cols)),
        shape=(n_fg, n_bg + n_fg)
    ).tocsr()

    # Sparse minimum-cost full matching
    matched_row, matched_col = min_weight_full_bipartite_matching(
        cost_matrix
    )

    # Extract costs of selected edges
    selected_cost = np.asarray(
        cost_matrix[matched_row, matched_col]
    ).ravel()

    # Dummy edge has cost 2 -> contributes weight 0
    # Real edge has cost 2-w -> contributes w
    selected_weight = 2.0 - selected_cost

    intersection = selected_weight.sum()

    matched_pairs = np.sum(
        matched_col < n_bg
    )

    union = n_fg + n_bg - intersection

    weighted_jaccard = (
        intersection / union
        if union > 0
        else 0.0
    )

    return (
        weighted_jaccard,
        intersection,
        int(matched_pairs)
    )


def calculate_ld_overlap_from_vcf(
    context_df,
    input_vcf_file,
    env_id,
    top_percent=0.05,
    min_R2=0.5,
    ld_window_size=15000,
    pool_num=40
):

    sorted_context_df = context_df.sort_values(
        by=env_id,
        ascending=False
    )

    foreground_df = sorted_context_df[
        sorted_context_df['foreground']
    ]

    background_df = sorted_context_df[
        ~sorted_context_df['foreground']
    ]

    n_fore = max(1, int(len(foreground_df) * top_percent))
    n_back = max(1, int(len(background_df) * top_percent))

    top_GEA_foreground_df = foreground_df.head(n_fore).copy()
    top_GEA_background_df = background_df.head(n_back).copy()

    top_GEA_R2_df = calculate_ld_matrix(
        top_GEA_foreground_df,
        input_vcf_file,
        ld_window_size=ld_window_size,
        pool_num=pool_num,
        min_r2=min_R2,
        site_info_df2=top_GEA_background_df
    )

    # R2 <= min_R2 -> 0
    # R2 = 1 -> 1
    top_GEA_R2_df['R2_scale'] = (
        (top_GEA_R2_df['R2'] - min_R2)
        / (1 - min_R2)
    ).clip(lower=0, upper=1)

    # 每个 FG 找最强 BG partner
    fore_max = (
        top_GEA_R2_df
        .groupby('INDEX_1')['R2_scale']
        .max()
    )

    # 每个 BG 找最强 FG partner
    back_max = (
        top_GEA_R2_df
        .groupby('INDEX_2')['R2_scale']
        .max()
    )

    # 没有 partner 的 site 自动相当于贡献 0，
    # 因为分母是全部 top GEA sites
    o_fore = (
        fore_max.sum()
        / len(top_GEA_foreground_df)
    )

    o_back = (
        back_max.sum()
        / len(top_GEA_background_df)
    )

    mean_overlap = (o_fore + o_back) / 2

    harmonic_mean_overlap = (
        2 * o_fore * o_back / (o_fore + o_back)
        if o_fore > 0 and o_back > 0
        else 0
    )

    # Jaccard matching 中完全没有贡献的 edge 不需要保留
    strong_ld_df = top_GEA_R2_df[
        top_GEA_R2_df['R2_scale'] > 0
    ].copy()

    weighted_jaccard_overlap, weighted_intersection, matched_pairs = (
        weighted_ld_jaccard_sparse(
            strong_ld_df,
            top_GEA_foreground_df.index,
            top_GEA_background_df.index,
        )
    )
    return {
        'O_fore': o_fore,
        'O_back': o_back,
        'foreground_count': len(top_GEA_foreground_df),
        'background_count': len(top_GEA_background_df),
        'mean_overlap': mean_overlap,
        'harmonic_mean_overlap': harmonic_mean_overlap,
        'weighted_intersection': weighted_intersection,
        'matched_pairs': matched_pairs,
        'weighted_jaccard_overlap': weighted_jaccard_overlap,
    }


def calculate_ld_overlap_by_gea_bins_from_ld_df(
    context_df,
    ld_df,
    env_id,
    bin_width=0.05,
    min_R2=0.5,
    source_col="INDEX_1",
    target_col="INDEX_2",
    r2_col="R2",
    chunk_size=5_000_000,
):
    """
    Calculate cross-class LD overlap across non-overlapping GEA-rank bins.

    FG and BG sites are ranked SEPARATELY by env_id from strongest to weakest.
    The rank distribution of each class is then divided into bins of width
    `bin_width`.

    Example:
        bin_width = 0.05

        bin 0: strongest 0-5%
        bin 1: 5-10%
        bin 2: 10-15%
        ...
        bin 19: 95-100%

    For each bin, only LD edges connecting:
        FG sites in that bin
            <->
        BG sites in the same bin
    are used.

    For every site, the maximum scaled R2 to the opposite class is calculated:

        R2_scale = (R2 - min_R2) / (1 - min_R2)

    with R2 <= min_R2 contributing 0.

    Directional overlaps:

        O_fore = mean(max opposite-class R2_scale for FG sites)
        O_back = mean(max opposite-class R2_scale for BG sites)

    Harmonic mean:

        H = 2 * O_fore * O_back / (O_fore + O_back)

    Parameters
    ----------
    context_df : pandas.DataFrame
        Must contain:
            env_id
            foreground

        context_df.index must use the same SNP/node IDs as ld_df.

    ld_df : pandas.DataFrame
        Precomputed LD edge table.

    env_id : str
        GEA score column. Larger value = stronger GEA signal.

    bin_width : float
        Width of each GEA-rank percentile bin.
        Example:
            0.05 -> 5%-wide bins
            0.10 -> 10%-wide bins

    min_R2 : float
        Minimum R2 defining positive LD overlap.

    Returns
    -------
    pandas.DataFrame
        One row per GEA-rank bin.
    """

    # =========================================================
    # 1. Validation
    # =========================================================

    if env_id not in context_df.columns:
        raise ValueError(
            f"Column not found in context_df: {env_id}"
        )

    if "foreground" not in context_df.columns:
        raise ValueError(
            "Column not found in context_df: foreground"
        )

    for col in [source_col, target_col, r2_col]:
        if col not in ld_df.columns:
            raise ValueError(
                f"Column not found in ld_df: {col}"
            )

    if not context_df.index.is_unique:
        raise ValueError(
            "context_df.index must be unique"
        )

    if context_df.index.hasnans:
        raise ValueError(
            "context_df.index cannot contain missing values"
        )

    if not 0 < bin_width <= 1:
        raise ValueError(
            "bin_width must be in (0, 1]"
        )

    if not 0 <= min_R2 < 1:
        raise ValueError(
            "min_R2 must be in [0, 1)"
        )

    foreground_raw = context_df["foreground"]

    if foreground_raw.isna().any():
        raise ValueError(
            "foreground contains missing values"
        )

    valid_values = set(
        pd.unique(foreground_raw)
    )

    if not valid_values <= {0, 1, False, True}:
        raise ValueError(
            "foreground must contain only True/False or 0/1"
        )

    foreground = foreground_raw.astype(bool)

    # =========================================================
    # 2. Rank FG and BG separately by GEA
    # =========================================================

    sorted_index = (
        context_df[env_id]
        .sort_values(
            ascending=False,
            kind="mergesort",
        )
        .index
    )

    sorted_foreground = (
        foreground.loc[sorted_index]
    )

    foreground_ids = pd.Index(
        sorted_index[
            sorted_foreground.to_numpy()
        ]
    )

    background_ids = pd.Index(
        sorted_index[
            ~sorted_foreground.to_numpy()
        ]
    )

    n_fore = len(foreground_ids)
    n_back = len(background_ids)

    if n_fore == 0:
        raise ValueError("No foreground sites")

    if n_back == 0:
        raise ValueError("No background sites")

    # =========================================================
    # 3. Assign percentile bins separately within FG / BG
    # =========================================================

    n_bins = int(
        np.ceil(1.0 / bin_width)
    )

    # rank_fraction:
    # strongest site -> 0
    # weakest site  -> approximately 1
    fore_rank_fraction = (
        np.arange(n_fore, dtype=np.float64)
        / n_fore
    )

    back_rank_fraction = (
        np.arange(n_back, dtype=np.float64)
        / n_back
    )

    fore_bin = np.floor(
        fore_rank_fraction / bin_width
    ).astype(np.int32)

    back_bin = np.floor(
        back_rank_fraction / bin_width
    ).astype(np.int32)

    # Protect against floating-point edge cases.
    np.minimum(
        fore_bin,
        n_bins - 1,
        out=fore_bin,
    )

    np.minimum(
        back_bin,
        n_bins - 1,
        out=back_bin,
    )

    # =========================================================
    # 4. Map node ID -> context position
    # =========================================================

    context_index = context_df.index
    n_nodes = len(context_df)

    index_dtype = (
        np.int32
        if n_nodes < np.iinfo(np.int32).max
        else np.int64
    )

    # role:
    #   1 = foreground
    #   2 = background
    role = np.zeros(
        n_nodes,
        dtype=np.int8,
    )

    # GEA percentile bin for every context node
    site_bin = np.full(
        n_nodes,
        -1,
        dtype=np.int32,
    )

    # Local position within the ranked FG/BG arrays
    fore_local_pos = np.full(
        n_nodes,
        -1,
        dtype=index_dtype,
    )

    back_local_pos = np.full(
        n_nodes,
        -1,
        dtype=index_dtype,
    )

    fore_context_pos = (
        context_index.get_indexer(
            foreground_ids
        )
    )

    back_context_pos = (
        context_index.get_indexer(
            background_ids
        )
    )

    if (
        np.any(fore_context_pos < 0)
        or np.any(back_context_pos < 0)
    ):
        raise RuntimeError(
            "FG/BG IDs could not be mapped back to context_df.index"
        )

    role[fore_context_pos] = 1
    role[back_context_pos] = 2

    site_bin[fore_context_pos] = fore_bin
    site_bin[back_context_pos] = back_bin

    fore_local_pos[fore_context_pos] = np.arange(
        n_fore,
        dtype=index_dtype,
    )

    back_local_pos[back_context_pos] = np.arange(
        n_back,
        dtype=index_dtype,
    )

    # =========================================================
    # 5. Maximum opposite-class LD for every site
    # =========================================================

    fore_max = np.zeros(
        n_fore,
        dtype=np.float64,
    )

    back_max = np.zeros(
        n_back,
        dtype=np.float64,
    )

    # Number of qualifying FG-BG LD edges in each bin
    cross_edge_count = np.zeros(
        n_bins,
        dtype=np.int64,
    )

    # =========================================================
    # 6. Scan LD graph once
    # =========================================================

    n_edges = len(ld_df)

    for start in range(
        0,
        n_edges,
        chunk_size,
    ):

        stop = min(
            start + chunk_size,
            n_edges,
        )

        chunk = ld_df.iloc[start:stop]

        src_id = chunk[
            source_col
        ].to_numpy(copy=False)

        dst_id = chunk[
            target_col
        ].to_numpy(copy=False)

        r2 = chunk[
            r2_col
        ].to_numpy(
            dtype=np.float64,
            copy=False,
        )

        # ---------------------------------------------
        # node ID -> context position
        # ---------------------------------------------

        src_pos = context_index.get_indexer(
            src_id
        )

        dst_pos = context_index.get_indexer(
            dst_id
        )

        valid = (
            (src_pos >= 0)
            & (dst_pos >= 0)
            & np.isfinite(r2)
            & (r2 > min_R2)
        )

        if not valid.any():
            continue

        valid_idx = np.flatnonzero(valid)

        src_pos_v = src_pos[valid_idx]
        dst_pos_v = dst_pos[valid_idx]

        src_role = role[src_pos_v]
        dst_role = role[dst_pos_v]

        # ---------------------------------------------
        # FG -> BG
        # ---------------------------------------------

        mask_fg_bg = (
            (src_role == 1)
            & (dst_role == 2)
        )

        # ---------------------------------------------
        # BG -> FG
        # ---------------------------------------------

        mask_bg_fg = (
            (src_role == 2)
            & (dst_role == 1)
        )

        if (
            not mask_fg_bg.any()
            and not mask_bg_fg.any()
        ):
            continue

        fg_context_parts = []
        bg_context_parts = []
        edge_r2_parts = []

        if mask_fg_bg.any():

            selected = valid_idx[
                mask_fg_bg
            ]

            fg_context_parts.append(
                src_pos[selected]
            )

            bg_context_parts.append(
                dst_pos[selected]
            )

            edge_r2_parts.append(
                r2[selected]
            )

        if mask_bg_fg.any():

            selected = valid_idx[
                mask_bg_fg
            ]

            fg_context_parts.append(
                dst_pos[selected]
            )

            bg_context_parts.append(
                src_pos[selected]
            )

            edge_r2_parts.append(
                r2[selected]
            )

        if len(fg_context_parts) == 1:

            fg_context = fg_context_parts[0]
            bg_context = bg_context_parts[0]
            edge_r2 = edge_r2_parts[0]

        else:

            fg_context = np.concatenate(
                fg_context_parts
            )

            bg_context = np.concatenate(
                bg_context_parts
            )

            edge_r2 = np.concatenate(
                edge_r2_parts
            )

        # ---------------------------------------------
        # IMPORTANT:
        # only compare FG/BG sites from the SAME
        # GEA percentile bin
        # ---------------------------------------------

        edge_fore_bin = site_bin[
            fg_context
        ]

        edge_back_bin = site_bin[
            bg_context
        ]

        same_bin = (
            edge_fore_bin
            == edge_back_bin
        )

        if not same_bin.any():
            continue

        fg_context = fg_context[same_bin]
        bg_context = bg_context[same_bin]
        edge_r2 = edge_r2[same_bin]

        edge_bin = edge_fore_bin[
            same_bin
        ]

        # ---------------------------------------------
        # Scale R2 exactly as in original function
        # ---------------------------------------------

        r2_scale = (
            (edge_r2 - min_R2)
            / (1.0 - min_R2)
        )

        r2_scale = np.clip(
            r2_scale,
            0.0,
            1.0,
        )

        # ---------------------------------------------
        # Local FG/BG positions
        # ---------------------------------------------

        fg_local = fore_local_pos[
            fg_context
        ]

        bg_local = back_local_pos[
            bg_context
        ]

        np.maximum.at(
            fore_max,
            fg_local,
            r2_scale,
        )

        np.maximum.at(
            back_max,
            bg_local,
            r2_scale,
        )

        # Count qualifying cross-class LD edges per bin
        cross_edge_count += np.bincount(
            edge_bin,
            minlength=n_bins,
        )

    # =========================================================
    # 7. Aggregate site-level maxima by GEA bin
    # =========================================================

    fore_count = np.bincount(
        fore_bin,
        minlength=n_bins,
    )

    back_count = np.bincount(
        back_bin,
        minlength=n_bins,
    )

    fore_sum = np.bincount(
        fore_bin,
        weights=fore_max,
        minlength=n_bins,
    )

    back_sum = np.bincount(
        back_bin,
        weights=back_max,
        minlength=n_bins,
    )

    o_fore = np.full(
        n_bins,
        np.nan,
        dtype=np.float64,
    )

    o_back = np.full(
        n_bins,
        np.nan,
        dtype=np.float64,
    )

    np.divide(
        fore_sum,
        fore_count,
        out=o_fore,
        where=fore_count > 0,
    )

    np.divide(
        back_sum,
        back_count,
        out=o_back,
        where=back_count > 0,
    )

    mean_overlap = (
        o_fore + o_back
    ) / 2.0

    harmonic_mean_overlap = np.full(
        n_bins,
        np.nan,
        dtype=np.float64,
    )

    valid_h = (
        np.isfinite(o_fore)
        & np.isfinite(o_back)
    )

    positive_h = (
        valid_h
        & (o_fore > 0)
        & (o_back > 0)
    )

    harmonic_mean_overlap[
        valid_h
    ] = 0.0

    harmonic_mean_overlap[
        positive_h
    ] = (
        2.0
        * o_fore[positive_h]
        * o_back[positive_h]
        / (
            o_fore[positive_h]
            + o_back[positive_h]
        )
    )

    # =========================================================
    # 8. Build output table
    # =========================================================

    bin_id = np.arange(
        n_bins,
        dtype=int,
    )

    bin_start = (
        bin_id * bin_width
    )

    bin_end = np.minimum(
        (bin_id + 1) * bin_width,
        1.0,
    )

    result_df = pd.DataFrame({
        "bin_id": bin_id,

        # 0 = strongest end of GEA ranking
        "gea_rank_start": bin_start,
        "gea_rank_end": bin_end,

        "gea_rank_start_percent": (
            bin_start * 100
        ),

        "gea_rank_end_percent": (
            bin_end * 100
        ),

        "gea_rank_mid_percent": (
            (bin_start + bin_end)
            / 2.0
            * 100
        ),

        "foreground_count": fore_count,
        "background_count": back_count,

        "cross_ld_edge_count": cross_edge_count,

        "O_fore": o_fore,
        "O_back": o_back,

        "mean_overlap": mean_overlap,

        "harmonic_mean_overlap":
            harmonic_mean_overlap,
    })

    return result_df


def calculate_ld_overlap_from_ld_df(
    context_df,
    ld_df,
    env_id,
    top_percent=0.05,
    min_R2=0.5,
    source_col="INDEX_1",
    target_col="INDEX_2",
    r2_col="R2",
    chunk_size=5_000_000,
):
    """
    Directly calculate cross-class LD overlap from a precomputed LD edge table.

    Parameters
    ----------
    context_df : pandas.DataFrame
        Must contain:
            env_id
            foreground

        context_df.index must use exactly the same SNP/node IDs as
        ld_df[source_col] and ld_df[target_col].

        foreground may be:
            bool
            0 / 1

    ld_df : pandas.DataFrame
        Precomputed LD edge table.

        Required columns:
            source_col
            target_col
            r2_col

        Each row represents one LD edge.

    env_id : str
        GEA score column. Larger value = stronger GEA signal.

    top_percent : float
        Fraction of FG and BG sites separately used as top-GEA sites.

    min_R2 : float
        Minimum R2 used to define positive LD overlap.

        R2 <= min_R2 -> R2_scale = 0
        R2 = 1       -> R2_scale = 1

    chunk_size : int
        Number of LD edges processed per chunk.

    Returns
    -------
    dict
    """

    # =========================================================
    # 1. Validation
    # =========================================================

    if env_id not in context_df.columns:
        raise ValueError(
            f"Column not found in context_df: {env_id}"
        )

    if "foreground" not in context_df.columns:
        raise ValueError(
            "Column not found in context_df: foreground"
        )

    for col in [source_col, target_col, r2_col]:
        if col not in ld_df.columns:
            raise ValueError(
                f"Column not found in ld_df: {col}"
            )

    if not context_df.index.is_unique:
        raise ValueError(
            "context_df.index must be unique"
        )

    if context_df.index.hasnans:
        raise ValueError(
            "context_df.index cannot contain missing values"
        )

    if not 0 < top_percent <= 1:
        raise ValueError(
            "top_percent must be in (0, 1]"
        )

    if not 0 <= min_R2 < 1:
        raise ValueError(
            "min_R2 must be in [0, 1)"
        )

    # ---------------------------------------------------------
    # foreground -> bool
    # ---------------------------------------------------------

    foreground_raw = context_df["foreground"]

    if foreground_raw.isna().any():
        raise ValueError(
            "foreground contains missing values"
        )

    valid_values = set(
        pd.unique(foreground_raw)
    )

    if not valid_values <= {0, 1, False, True}:
        raise ValueError(
            "foreground must contain only "
            "True/False or 0/1"
        )

    foreground = foreground_raw.astype(
        bool
    )

    # =========================================================
    # 2. Select top-GEA FG / BG separately
    # =========================================================

    # Important:
    #
    # We sort using env_id.
    # context_df's current row order does NOT matter.
    #
    # SNP identity is always context_df.index.

    sorted_index = (
        context_df[env_id]
        .sort_values(
            ascending=False
        )
        .index
    )

    sorted_foreground = (
        foreground
        .loc[sorted_index]
    )

    foreground_ids = (
        sorted_index[
            sorted_foreground.to_numpy()
        ]
    )

    background_ids = (
        sorted_index[
            ~sorted_foreground.to_numpy()
        ]
    )

    if len(foreground_ids) == 0:
        raise ValueError(
            "No foreground sites"
        )

    if len(background_ids) == 0:
        raise ValueError(
            "No background sites"
        )

    n_fore = max(
        1,
        int(
            len(foreground_ids)
            * top_percent
        )
    )

    n_back = max(
        1,
        int(
            len(background_ids)
            * top_percent
        )
    )

    top_fore_ids = pd.Index(
        foreground_ids[:n_fore]
    )

    top_back_ids = pd.Index(
        background_ids[:n_back]
    )

    # =========================================================
    # 3. Build node-ID -> context position mapping
    #
    # This does NOT assume context_df.index is 0..N-1.
    # =========================================================

    context_index = context_df.index

    n_nodes = len(context_df)

    index_dtype = (
        np.int32
        if n_nodes < np.iinfo(
            np.int32
        ).max
        else np.int64
    )

    # ---------------------------------------------------------
    # For each context position:
    #
    # role:
    #   0 = irrelevant
    #   1 = top FG
    #   2 = top BG
    #
    # fore_pos / back_pos:
    #   local position within top FG / top BG arrays
    # ---------------------------------------------------------

    role = np.zeros(
        n_nodes,
        dtype=np.int8,
    )

    fore_local_pos = np.full(
        n_nodes,
        -1,
        dtype=index_dtype,
    )

    back_local_pos = np.full(
        n_nodes,
        -1,
        dtype=index_dtype,
    )

    top_fore_context_pos = (
        context_index.get_indexer(
            top_fore_ids
        )
    )

    top_back_context_pos = (
        context_index.get_indexer(
            top_back_ids
        )
    )

    if (
        np.any(top_fore_context_pos < 0)
        or np.any(top_back_context_pos < 0)
    ):
        raise RuntimeError(
            "Top FG/BG IDs could not be mapped "
            "back to context_df.index"
        )

    role[
        top_fore_context_pos
    ] = 1

    role[
        top_back_context_pos
    ] = 2

    fore_local_pos[
        top_fore_context_pos
    ] = np.arange(
        n_fore,
        dtype=index_dtype,
    )

    back_local_pos[
        top_back_context_pos
    ] = np.arange(
        n_back,
        dtype=index_dtype,
    )

    # =========================================================
    # 4. Maximum opposite-class LD for every top site
    # =========================================================

    fore_max = np.zeros(
        n_fore,
        dtype=np.float64,
    )

    back_max = np.zeros(
        n_back,
        dtype=np.float64,
    )

    # Save only top-FG <-> top-BG edges.
    # Usually vastly smaller than full ld_df.
    edge_fore_parts = []
    edge_back_parts = []
    edge_weight_parts = []

    # =========================================================
    # 5. Scan LD graph
    # =========================================================

    n_edges = len(ld_df)

    for start in range(
        0,
        n_edges,
        chunk_size,
    ):

        stop = min(
            start + chunk_size,
            n_edges,
        )

        chunk = ld_df.iloc[
            start:stop
        ]

        src_id = chunk[
            source_col
        ].to_numpy(
            copy=False
        )

        dst_id = chunk[
            target_col
        ].to_numpy(
            copy=False
        )

        r2 = chunk[
            r2_col
        ].to_numpy(
            dtype=np.float64,
            copy=False,
        )

        # -----------------------------------------------------
        # Convert ld_df node IDs -> context positions
        # -----------------------------------------------------

        src_pos = (
            context_index.get_indexer(
                src_id
            )
        )

        dst_pos = (
            context_index.get_indexer(
                dst_id
            )
        )

        valid = (
            (src_pos >= 0)
            & (dst_pos >= 0)
            & np.isfinite(r2)
            & (r2 > min_R2)
        )

        if not valid.any():
            continue

        valid_idx = np.flatnonzero(
            valid
        )

        src_pos_v = src_pos[
            valid_idx
        ]

        dst_pos_v = dst_pos[
            valid_idx
        ]

        src_role = role[
            src_pos_v
        ]

        dst_role = role[
            dst_pos_v
        ]

        # -----------------------------------------------------
        # Orientation 1:
        #
        # source = top FG
        # target = top BG
        # -----------------------------------------------------

        mask_fg_bg = (
            (src_role == 1)
            & (dst_role == 2)
        )

        # -----------------------------------------------------
        # Orientation 2:
        #
        # source = top BG
        # target = top FG
        # -----------------------------------------------------

        mask_bg_fg = (
            (src_role == 2)
            & (dst_role == 1)
        )

        if (
            not mask_fg_bg.any()
            and not mask_bg_fg.any()
        ):
            continue

        fg_id_parts = []
        bg_id_parts = []

        fg_context_pos_parts = []
        bg_context_pos_parts = []

        edge_r2_parts = []

        # -----------------------------------------------------
        # FG -> BG orientation
        # -----------------------------------------------------

        if mask_fg_bg.any():

            selected = valid_idx[
                mask_fg_bg
            ]

            fg_id_parts.append(
                src_id[selected]
            )

            bg_id_parts.append(
                dst_id[selected]
            )

            fg_context_pos_parts.append(
                src_pos[selected]
            )

            bg_context_pos_parts.append(
                dst_pos[selected]
            )

            edge_r2_parts.append(
                r2[selected]
            )

        # -----------------------------------------------------
        # BG -> FG:
        #
        # reverse orientation so every retained edge is:
        #
        # FG -> BG
        # -----------------------------------------------------

        if mask_bg_fg.any():

            selected = valid_idx[
                mask_bg_fg
            ]

            fg_id_parts.append(
                dst_id[selected]
            )

            bg_id_parts.append(
                src_id[selected]
            )

            fg_context_pos_parts.append(
                dst_pos[selected]
            )

            bg_context_pos_parts.append(
                src_pos[selected]
            )

            edge_r2_parts.append(
                r2[selected]
            )

        # -----------------------------------------------------
        # Combine two possible orientations
        # -----------------------------------------------------

        if len(fg_id_parts) == 1:

            fg_ids = fg_id_parts[0]
            bg_ids = bg_id_parts[0]

            fg_context_pos = (
                fg_context_pos_parts[0]
            )

            bg_context_pos = (
                bg_context_pos_parts[0]
            )

            edge_r2 = edge_r2_parts[0]

        else:

            fg_ids = np.concatenate(
                fg_id_parts
            )

            bg_ids = np.concatenate(
                bg_id_parts
            )

            fg_context_pos = (
                np.concatenate(
                    fg_context_pos_parts
                )
            )

            bg_context_pos = (
                np.concatenate(
                    bg_context_pos_parts
                )
            )

            edge_r2 = np.concatenate(
                edge_r2_parts
            )

        # =====================================================
        # 6. Scale R2
        # =====================================================

        r2_scale = (
            (edge_r2 - min_R2)
            / (1.0 - min_R2)
        )

        r2_scale = np.clip(
            r2_scale,
            0.0,
            1.0,
        )

        # =====================================================
        # 7. Directional maxima
        # =====================================================

        fg_local = (
            fore_local_pos[
                fg_context_pos
            ]
        )

        bg_local = (
            back_local_pos[
                bg_context_pos
            ]
        )

        np.maximum.at(
            fore_max,
            fg_local,
            r2_scale,
        )

        np.maximum.at(
            back_max,
            bg_local,
            r2_scale,
        )

        # =====================================================
        # 8. Save edges for weighted Jaccard
        # =====================================================

        edge_fore_parts.append(
            np.asarray(
                fg_ids
            )
        )

        edge_back_parts.append(
            np.asarray(
                bg_ids
            )
        )

        edge_weight_parts.append(
            np.asarray(
                r2_scale,
                dtype=np.float64,
            )
        )

    # =========================================================
    # 9. Directional overlap
    # =========================================================

    o_fore = (
        fore_max.sum()
        / n_fore
    )

    o_back = (
        back_max.sum()
        / n_back
    )

    mean_overlap = (
        o_fore + o_back
    ) / 2.0

    harmonic_mean_overlap = (
        2.0
        * o_fore
        * o_back
        / (o_fore + o_back)
        if (
            o_fore > 0
            and o_back > 0
        )
        else 0.0
    )

    # =========================================================
    # 10. Weighted Jaccard
    # =========================================================

    if edge_fore_parts:

        strong_ld_df = pd.DataFrame(
            {
                "INDEX_1": np.concatenate(
                    edge_fore_parts
                ),
                "INDEX_2": np.concatenate(
                    edge_back_parts
                ),
                "R2_scale": np.concatenate(
                    edge_weight_parts
                ),
            }
        )

    else:

        strong_ld_df = pd.DataFrame(
            {
                "INDEX_1": pd.Series(
                    dtype=context_df.index.dtype
                ),
                "INDEX_2": pd.Series(
                    dtype=context_df.index.dtype
                ),
                "R2_scale": pd.Series(
                    dtype=float
                ),
            }
        )

    (
        weighted_jaccard_overlap,
        weighted_intersection,
        matched_pairs,
    ) = weighted_ld_jaccard_sparse(
        strong_ld_df,
        top_fore_ids,
        top_back_ids,
    )

    # =========================================================
    # 11. Return
    # =========================================================

    return {
        "O_fore": float(o_fore),
        "O_back": float(o_back),

        "foreground_count": int(
            n_fore
        ),

        "background_count": int(
            n_back
        ),

        "mean_overlap": float(
            mean_overlap
        ),

        "harmonic_mean_overlap": float(
            harmonic_mean_overlap
        ),

        "weighted_intersection": float(
            weighted_intersection
        ),

        "matched_pairs": int(
            matched_pairs
        ),

        "weighted_jaccard_overlap": float(
            weighted_jaccard_overlap
        ),
    }


def calculate_ld_overlap_from_csr(
    context_df,
    ld_csr,
    env_id,
    top_percent=0.05,
    min_R2=0.5,
):
    """
    Calculate LD overlap using a pre-built undirected CSR LD graph.

    Parameters
    ----------
    context_df : pd.DataFrame
        Must contain:
            env_id
            foreground

        context_df.index must correspond exactly to CSR node IDs.

    ld_csr : scipy.sparse.csr_matrix
        Symmetric LD adjacency matrix.
        Matrix data contain raw R2 values.

    env_id : str
        GEA score column; larger = stronger.

    top_percent : float
        Top fraction selected separately from FG and BG.

    min_R2 : float
        R2 <= min_R2 contributes zero.
        R2 = 1 contributes one.

    Returns
    -------
    dict
    """

    # --------------------------------------------------
    # 1. Validate foreground
    # --------------------------------------------------

    foreground_raw = context_df['foreground']

    if foreground_raw.isna().any():
        raise ValueError(
            "foreground contains missing values"
        )

    valid_values = set(
        pd.unique(foreground_raw)
    )

    if not valid_values <= {0, 1, False, True}:
        raise ValueError(
            "foreground must contain only "
            "True/False or 0/1"
        )

    foreground = foreground_raw.astype(
        bool
    )

    # --------------------------------------------------
    # 2. Select top GEA FG/BG
    # --------------------------------------------------

    sorted_context_df = context_df.sort_values(
        env_id,
        ascending=False,
    )

    foreground_df = sorted_context_df[
        foreground.loc[
            sorted_context_df.index
        ].to_numpy()
    ]

    background_df = sorted_context_df[
        ~foreground.loc[
            sorted_context_df.index
        ].to_numpy()
    ]

    if foreground_df.empty:
        raise ValueError("No foreground sites")

    if background_df.empty:
        raise ValueError("No background sites")

    n_fore = max(
        1,
        int(
            len(foreground_df)
            * top_percent
        ),
    )

    n_back = max(
        1,
        int(
            len(background_df)
            * top_percent
        ),
    )

    top_fore_ids = (
        foreground_df
        .index[:n_fore]
        .to_numpy(dtype=np.int64)
    )

    top_back_ids = (
        background_df
        .index[:n_back]
        .to_numpy(dtype=np.int64)
    )

    # --------------------------------------------------
    # 3. Fast BG membership lookup
    # --------------------------------------------------

    n_nodes = ld_csr.shape[0]

    is_top_back = np.zeros(
        n_nodes,
        dtype=bool,
    )

    is_top_back[
        top_back_ids
    ] = True

    # BG node -> local BG position
    back_pos = np.full(
        n_nodes,
        -1,
        dtype=np.int32,
    )

    back_pos[
        top_back_ids
    ] = np.arange(
        n_back,
        dtype=np.int32,
    )

    # --------------------------------------------------
    # 4. Directional maxima
    # --------------------------------------------------

    fore_max = np.zeros(
        n_fore,
        dtype=np.float64,
    )

    back_max = np.zeros(
        n_back,
        dtype=np.float64,
    )

    # For weighted Jaccard
    edge_fore = []
    edge_back = []
    edge_weight = []

    indptr = ld_csr.indptr
    indices = ld_csr.indices
    data = ld_csr.data

    # --------------------------------------------------
    # 5. Only traverse neighbors of top-FG sites
    # --------------------------------------------------

    for fore_local, fg in enumerate(
        top_fore_ids
    ):

        start = indptr[fg]
        end = indptr[fg + 1]

        neighbors = indices[
            start:end
        ]

        r2 = data[
            start:end
        ]

        # Only top-BG neighbors
        mask = (
            is_top_back[neighbors]
            & (r2 > min_R2)
        )

        if not mask.any():
            continue

        bg_nodes = neighbors[
            mask
        ]

        edge_r2 = r2[
            mask
        ].astype(
            np.float64,
            copy=False,
        )

        r2_scale = (
            (edge_r2 - min_R2)
            / (1.0 - min_R2)
        )

        r2_scale = np.clip(
            r2_scale,
            0.0,
            1.0,
        )

        # ----------------------------------------------
        # O_fore
        # ----------------------------------------------

        fore_max[
            fore_local
        ] = r2_scale.max()

        # ----------------------------------------------
        # O_back
        # ----------------------------------------------

        bg_local = back_pos[
            bg_nodes
        ]

        np.maximum.at(
            back_max,
            bg_local,
            r2_scale,
        )

        # ----------------------------------------------
        # Save edges for weighted Jaccard
        # ----------------------------------------------

        edge_fore.append(
            np.full(
                len(bg_nodes),
                fg,
                dtype=np.int64,
            )
        )

        edge_back.append(
            bg_nodes.astype(
                np.int64,
                copy=False,
            )
        )

        edge_weight.append(
            r2_scale
        )

    # --------------------------------------------------
    # 6. Directional overlap
    # --------------------------------------------------

    o_fore = (
        fore_max.sum()
        / n_fore
    )

    o_back = (
        back_max.sum()
        / n_back
    )

    mean_overlap = (
        o_fore + o_back
    ) / 2.0

    harmonic_mean_overlap = (
        2.0
        * o_fore
        * o_back
        / (o_fore + o_back)
        if o_fore > 0
        and o_back > 0
        else 0.0
    )

    # --------------------------------------------------
    # 7. Weighted Jaccard
    # --------------------------------------------------

    if edge_fore:

        strong_ld_df = pd.DataFrame({
            'INDEX_1': np.concatenate(
                edge_fore
            ),
            'INDEX_2': np.concatenate(
                edge_back
            ),
            'R2_scale': np.concatenate(
                edge_weight
            ),
        })

    else:

        strong_ld_df = pd.DataFrame({
            'INDEX_1': pd.Series(
                dtype=np.int64
            ),
            'INDEX_2': pd.Series(
                dtype=np.int64
            ),
            'R2_scale': pd.Series(
                dtype=float
            ),
        })

    (
        weighted_jaccard_overlap,
        weighted_intersection,
        matched_pairs,
    ) = weighted_ld_jaccard_sparse(
        strong_ld_df,
        top_fore_ids,
        top_back_ids,
    )

    return {
        'O_fore': float(o_fore),
        'O_back': float(o_back),

        'foreground_count': int(n_fore),
        'background_count': int(n_back),

        'mean_overlap': float(
            mean_overlap
        ),

        'harmonic_mean_overlap': float(
            harmonic_mean_overlap
        ),

        'weighted_intersection': float(
            weighted_intersection
        ),

        'matched_pairs': int(
            matched_pairs
        ),

        'weighted_jaccard_overlap': float(
            weighted_jaccard_overlap
        ),
    }


def calculate_contrast_ld_overlap(contrast_id, contrast_json, context_df, env_id_list, ld_df, output_dir, bin_width=0.05, min_R2=0.5, random_seed=42):

    results_pkl_file = os.path.join(
        output_dir, f"{contrast_id}_overlap_results.pkl")

    if not os.path.exists(results_pkl_file):
        enrich_data_df = build_enrich_data_df(
            context_df, contrast_json, env_id_list)

        overlap_results = {}

        for env_id in env_id_list:
            env_enrich_data_df = enrich_data_df[[
                'ID', 'foreground', env_id]].copy()
            overlap_df = calculate_ld_overlap_by_gea_bins_from_ld_df(
                env_enrich_data_df, ld_df, env_id, bin_width=bin_width, min_R2=min_R2)
            random_enrich_data_df = env_enrich_data_df.copy()
            rng = np.random.default_rng(random_seed)
            random_enrich_data_df[env_id] = rng.permutation(
                random_enrich_data_df[env_id].to_numpy())
            random_overlap_df = calculate_ld_overlap_by_gea_bins_from_ld_df(
                random_enrich_data_df, ld_df, env_id, bin_width=bin_width, min_R2=min_R2)

            overlap_results[env_id] = {
                "overlap_df": overlap_df,
                "random_overlap_df": random_overlap_df,
            }

        with open(results_pkl_file, "wb") as f:
            pickle.dump(overlap_results, f)

    else:
        print(f"LD overlap results already exist for contrast {contrast_id}. Skipping calculation.")

    return results_pkl_file


def plot_overlap_by_quantile(
    overlap_results,
    env_id_list,
    env_metadata_df,

    # -------------------------
    # dataframe columns
    # -------------------------
    start_col='gea_rank_start',
    end_col='gea_rank_end',
    value_col='harmonic_mean_overlap',

    # -------------------------
    # figure
    # -------------------------
    figsize_per_panel=(3.5, 4),
    sharey=True,
    wspace=0.04,

    # -------------------------
    # bars
    # -------------------------
    bar_width_scale=1.0,
    bar_edgecolor='white',
    bar_linewidth=0.8,
    bar_alpha=1.0,

    # -------------------------
    # baseline
    # -------------------------
    baseline_linewidth=1.8,
    baseline_width_scale=0.85,

    # -------------------------
    # axes
    # -------------------------
    xlabel='GEA rank quantile',
    ylabel='Cross-group LD linkage proportion',
    ylim=(0.0, 1.0),
    xlim=(0.0, 1.0),

    # x ticks
    n_xticks=6,
    x_tick_format='.2f',

    # -------------------------
    # fonts
    # -------------------------
    title_fontsize=14,
    axis_label_fontsize=12,
    tick_fontsize=10,

    # -------------------------
    # title
    # -------------------------
    title_pad=8,

    # -------------------------
    # legend
    # -------------------------
    legend=True,
    legend_panel='last',
    legend_loc='upper right',
    legend_frameon=False,
    legend_fontsize=10,
    observed_label='Observed',
    baseline_label='Baseline',

    # -------------------------
    # spines / ticks
    # -------------------------
    remove_top_spine=True,
    remove_right_spine=True,
    tick_direction='out',

    # -------------------------
    # global figure margins
    # -------------------------
    fig_left=0.07,
    fig_right=0.995,
    fig_bottom=0.15,
    fig_top=0.90,

    # -------------------------
    # return
    # -------------------------
    return_fig_ax=True,
):
    """
    Plot observed overlap across GEA-rank quantile bins,
    with one horizontal baseline segment for each bin.

    Expected structure
    ------------------
    overlap_results[env_id]['overlap_df']
    overlap_results[env_id]['random_overlap_df']

    Parameters
    ----------
    overlap_results : dict

    env_id_list : list
        Environmental variable IDs.

    env_name_map : dict or None
        Mapping from env_id to display title.

        Example:
        {
            'bio18': 'Precipitation of warmest quarter',
            'gsl': 'Growing season length',
            'al': 'Aluminum',
            'striga': 'Striga pressure'
        }

    bar_width_scale : float
        1.0 means adjacent bars geometrically touch.

    baseline_width_scale : float
        Width of each baseline segment relative to the bin width.

    legend_panel : {'last', 'first'} or int
        Which panel contains the legend.

    Returns
    -------
    fig, axs
    """
    set_plot_style()

    env_name_map = {}
    for _, row in env_metadata_df.iterrows():
        env_id = row['env_id']
        env_name = row['description']
        env_name_map[env_id] = env_name

    n_env = len(env_id_list)

    if n_env == 0:
        raise ValueError("env_id_list is empty.")

    fig, axs = plt.subplots(
        1,
        n_env,
        figsize=(
            figsize_per_panel[0] * n_env,
            figsize_per_panel[1]
        ),
        sharey=sharey,
        gridspec_kw={'wspace': wspace},
    )

    # Always make axs iterable
    if n_env == 1:
        axs = np.array([axs])

    # ============================================================
    # Legend panel
    # ============================================================

    if legend_panel == 'last':
        legend_idx = n_env - 1

    elif legend_panel == 'first':
        legend_idx = 0

    elif isinstance(legend_panel, int):
        legend_idx = legend_panel

        if not (0 <= legend_idx < n_env):
            raise ValueError(
                f"legend_panel index {legend_idx} is out of range "
                f"for {n_env} panels."
            )

    else:
        raise ValueError(
            "legend_panel must be 'last', 'first', or an integer."
        )

    # ============================================================
    # Plot each environment
    # ============================================================

    for i, env_id in enumerate(env_id_list):

        ax = axs[i]

        # -------------------------
        # Extract data
        # -------------------------

        overlap_df = (
            overlap_results[env_id]['overlap_df']
            .copy()
            .sort_values([start_col, end_col])
            .reset_index(drop=True)
        )

        random_overlap_df = (
            overlap_results[env_id]['random_overlap_df']
            .copy()
            .sort_values([start_col, end_col])
            .reset_index(drop=True)
        )

        # -------------------------
        # Basic consistency check
        # -------------------------

        if len(overlap_df) != len(random_overlap_df):
            raise ValueError(
                f"{env_id}: overlap_df and random_overlap_df "
                f"have different numbers of bins."
            )

        # -------------------------
        # Bin geometry
        # -------------------------

        bin_left = overlap_df[start_col].to_numpy(dtype=float)
        bin_right = overlap_df[end_col].to_numpy(dtype=float)

        bin_center = (bin_left + bin_right) / 2
        bin_width = bin_right - bin_left

        observed = overlap_df[value_col].to_numpy(dtype=float)
        baseline = random_overlap_df[value_col].to_numpy(dtype=float)

        # -------------------------
        # Check bins between observed
        # and random data
        # -------------------------

        random_bin_left = random_overlap_df[start_col].to_numpy(dtype=float)
        random_bin_right = random_overlap_df[end_col].to_numpy(dtype=float)

        if not (
            np.allclose(bin_left, random_bin_left)
            and np.allclose(bin_right, random_bin_right)
        ):
            raise ValueError(
                f"{env_id}: bins in overlap_df and random_overlap_df "
                f"do not match."
            )

        # -------------------------
        # Bars
        # -------------------------

        ax.bar(
            bin_center,
            observed,
            width=bin_width * bar_width_scale,
            align='center',
            edgecolor=bar_edgecolor,
            linewidth=bar_linewidth,
            alpha=bar_alpha,
            zorder=2,
        )

        # -------------------------
        # Baseline
        # -------------------------

        baseline_half_width = (
            bin_width * baseline_width_scale / 2
        )

        for x, half_width, y in zip(
            bin_center,
            baseline_half_width,
            baseline
        ):
            ax.hlines(
                y=y,
                xmin=x - half_width,
                xmax=x + half_width,
                linewidth=baseline_linewidth,
                zorder=5,
            )

        # -------------------------
        # Title
        # -------------------------

        display_name = env_name_map.get(env_id, env_id)

        ax.set_title(
            display_name,
            fontsize=title_fontsize,
            pad=title_pad,
        )

        # -------------------------
        # X ticks based on bin edges
        # -------------------------

        bin_edges = np.unique(
            np.concatenate([bin_left, bin_right])
        )

        if len(bin_edges) <= n_xticks:
            xticks = bin_edges

        else:
            tick_indices = np.linspace(
                0,
                len(bin_edges) - 1,
                n_xticks,
                dtype=int
            )

            xticks = bin_edges[tick_indices]

        ax.set_xticks(xticks)

        xtick_labels = []
        for x in xticks:
            if np.isclose(x, 0):
                xtick_labels.append('0')
            elif np.isclose(x, 1):
                xtick_labels.append('1')
            else:
                xtick_labels.append(format(x, x_tick_format))

        ax.set_xticklabels(
            xtick_labels,
            fontsize=tick_fontsize,
        )

        # -------------------------
        # Limits
        # -------------------------

        if xlim is None:
            ax.set_xlim(
                bin_left.min(),
                bin_right.max()
            )
        else:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        # -------------------------
        # Axis labels
        # -------------------------

        ax.set_xlabel(
            xlabel,
            fontsize=axis_label_fontsize,
        )

        if i == 0:
            ax.set_ylabel(
                ylabel,
                fontsize=axis_label_fontsize,
            )
        else:
            ax.set_ylabel('')

        # -------------------------
        # Tick style
        # -------------------------

        ax.tick_params(
            axis='both',
            labelsize=tick_fontsize,
            direction=tick_direction,
        )

        # -------------------------
        # Spines
        # -------------------------

        if remove_top_spine:
            ax.spines['top'].set_visible(False)

        if remove_right_spine:
            ax.spines['right'].set_visible(False)

        # -------------------------
        # Legend
        # only one panel
        # -------------------------

        if legend and i == legend_idx:

            legend_handles = [
                Patch(
                    label=observed_label
                ),
                Line2D(
                    [0], [0],
                    linewidth=baseline_linewidth,
                    label=baseline_label,
                ),
            ]

            ax.legend(
                handles=legend_handles,
                loc=legend_loc,
                frameon=legend_frameon,
                fontsize=legend_fontsize,
            )

    # ============================================================
    # Global layout
    # ============================================================

    fig.subplots_adjust(
        left=fig_left,
        right=fig_right,
        bottom=fig_bottom,
        top=fig_top,
        wspace=wspace,
    )

    if return_fig_ax:
        return fig, axs


def prune_vcf_by_ld(
    input_vcf_file,
    output_vcf_file,
    *,
    ld_window_size=15000,
    r2_threshold=0.5,
    random_seed=42,
    min_maf=0.05,
    max_het=0.1,
    max_miss=0.5,
    site_info_h5_file=None,
    pool_num=1,
    cache_max_tiles=64,
    graph_chunk_size=256,
    show_progress=True,
    progress_interval=2.0,
):
    """Write a quality-filtered, randomly prioritized LD-pruned VCF.

    Reuse ``<input_vcf_file>.var_stat.h5`` if it exists; otherwise build it
    with vcf.build_var_stat_table. site_info_h5_file overrides that cache path.
    The HDF must contain the per-chromosome tables produced by that function
    and belong to this exact input VCF and sample set. Remove a stale cache
    yourself after replacing the input. QC and random ordering are never
    saved back into the cache, so it can be reused with different parameters.

    Retain MAF >= min_maf, HETF <= max_het and missing fraction <= max_miss.
    None disables an individual QC threshold. QC uses the cached MAF, HETF
    and MISSF columns directly; sites without called genotypes are excluded.
    Rebuild old caches generated before the get_var_stat MISSF correction.
    Statistics inherit build_var_stat_table's diploid/biallelic assumptions.

    Shuffle qualifying rows with numpy.default_rng(random_seed), then call
    prune_sites_by_ld: same-chromosome distance < ld_window_size and
    R2 > r2_threshold remove later-priority neighbors of retained sites.
    LD genotype encoding and missing-call handling are inherited unchanged.
    Output preserves the original header, samples, records and genomic order.
    A repeated coordinate emits only the first VCF record matching the
    retained row's REF and first ALT, consistent with coordinate-based pruning.

    Input must be a coordinate-sorted BGZF VCF; bcftools must be on PATH.
    A missing input index is created. Output is BGZF VCF with a TBI index;
    '.gz' is appended when needed. Return the final output path (str).
    pool_num controls both statistics and LD workers. pool_num=1 is serial;
    larger values require a fork-based kernel.

    从 VCF 的统计 HDF 缓存加载位点，先按 MAF、杂合率和缺失率过滤，
    再按 random_seed 随机打乱优先级并进行贪婪 LD pruning。已有缓存
    直接复用；缺失时自动构建，过滤和打乱不修改缓存。None 关闭对应
    质量阈值。输出保留原始 VCF 的记录顺序和样本，压缩并建立索引，
    返回输出路径。输入更新后需自行移除旧缓存；统计沿用二倍体、
    二等位假设，LD 沿用原函数的基因型及缺失值处理。
    """
    import subprocess
    import tempfile
    from cyvcf2 import Writer
    from .vcf import build_var_stat_table

    input_vcf_file = os.path.abspath(os.fspath(input_vcf_file))
    output_vcf_file = os.fspath(output_vcf_file)
    if not output_vcf_file.endswith('.gz'):
        output_vcf_file += '.gz'
    output_vcf_file = os.path.abspath(output_vcf_file)
    site_info_h5_file = os.path.abspath(os.fspath(
        site_info_h5_file if site_info_h5_file is not None
        else input_vcf_file + '.var_stat.h5'
    ))
    paths = [os.path.realpath(path) for path in
             (input_vcf_file, output_vcf_file, site_info_h5_file)]
    if len(set(paths)) != 3:
        raise ValueError('Input VCF, output VCF and statistics HDF paths must differ.')
    if not os.path.isfile(input_vcf_file):
        raise FileNotFoundError(input_vcf_file)

    random_seed = index(random_seed)
    if random_seed < 0:
        raise ValueError('random_seed must be a nonnegative integer.')
    ld_window_size, pool_num, cache_max_tiles, graph_chunk_size = map(
        index, (ld_window_size, pool_num, cache_max_tiles, graph_chunk_size)
    )
    if min(ld_window_size, pool_num, cache_max_tiles, graph_chunk_size) < 1:
        raise ValueError('LD window, pool and cache/chunk sizes must be positive.')
    r2_threshold = float(r2_threshold)
    if not np.isfinite(r2_threshold) or not 0 <= r2_threshold <= 1:
        raise ValueError('r2_threshold must be finite and in [0, 1].')
    thresholds = {}
    for name, value, maximum in (
        ('min_maf', min_maf, 0.5), ('max_het', max_het, 1),
        ('max_miss', max_miss, 1),
    ):
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f'{name} must be None or in [0, {maximum}].')
        thresholds[name] = value
    if show_progress and (not np.isfinite(progress_interval) or progress_interval <= 0):
        raise ValueError('progress_interval must be finite and positive.')
    if pool_num > 1 and mp.get_start_method() != 'fork':
        raise RuntimeError("Parallel LD requires 'fork'; use pool_num=1 in this kernel.")

    if not any(os.path.isfile(input_vcf_file + suffix) for suffix in ('.tbi', '.csi')):
        subprocess.run(['bcftools', 'index', '-t', input_vcf_file], check=True)
    if not os.path.isfile(site_info_h5_file):
        # Publish only a complete cache: a failed build must not be reused later.
        with tempfile.TemporaryDirectory(
            prefix='ld_stats_', dir=os.path.dirname(site_info_h5_file)
        ) as temporary_dir:
            temporary_h5 = os.path.join(temporary_dir, 'var_stat.h5')
            build_var_stat_table(input_vcf_file, temporary_h5, n_jobs=pool_num)
            os.replace(temporary_h5, site_info_h5_file)
    elif show_progress:
        print(f'Reusing variant statistics: {site_info_h5_file}', flush=True)

    with pd.HDFStore(site_info_h5_file, mode='r') as store:
        frames = [store[key] for key in sorted(store.keys())]
    required = ['CHROM', 'POS', 'REF', 'ALT', 'MAF', 'HETF', 'MISSF',
                'HOM_REF', 'HET', 'HOM_ALT']
    site_df = (pd.concat(frames, ignore_index=True) if frames
               else pd.DataFrame(columns=required))
    missing = [column for column in required if column not in site_df]
    if missing:
        raise ValueError(f'Missing statistics HDF columns: {missing}')

    called = site_df[['HOM_REF', 'HET', 'HOM_ALT']].sum(axis=1, skipna=False)
    eligible = called.gt(0)
    if thresholds['min_maf'] is not None:
        eligible &= site_df['MAF'].ge(thresholds['min_maf'])
    if thresholds['max_het'] is not None:
        eligible &= site_df['HETF'].le(thresholds['max_het'])
    if thresholds['max_miss'] is not None:
        eligible &= site_df['MISSF'].le(thresholds['max_miss'])
    qualified = site_df.loc[eligible].copy()
    rng = np.random.default_rng(random_seed)
    shuffled = qualified.iloc[rng.permutation(len(qualified))].reset_index(drop=True)
    retained = prune_sites_by_ld(
        shuffled, input_vcf_file, ld_window_size=ld_window_size,
        r2_threshold=r2_threshold, pool_num=pool_num,
        cache_max_tiles=cache_max_tiles, graph_chunk_size=graph_chunk_size,
        show_progress=show_progress, progress_interval=progress_interval,
    )
    selected = {
        (str(chrom), int(pos)): (ref, alt)
        for chrom, pos, ref, alt in retained[['CHROM', 'POS', 'REF', 'ALT']].itertuples(
            index=False, name=None
        )
    }
    with tempfile.TemporaryDirectory(
        prefix='ld_pruned_', dir=os.path.dirname(output_vcf_file)
    ) as temporary_dir:
        temporary_vcf = os.path.join(temporary_dir, 'pruned.vcf.gz')
        reader = VCF(input_vcf_file)
        try:
            writer = Writer(temporary_vcf, reader, mode='wz')
            try:
                for record in reader:
                    coordinate = (record.CHROM, record.POS)
                    if (coordinate in selected and record.ALT
                            and selected[coordinate] == (record.REF, record.ALT[0])):
                        writer.write_record(record)
                        del selected[coordinate]
            finally:
                writer.close()
        finally:
            reader.close()
        if selected:
            raise ValueError('Retained sites are absent from the VCF; check the statistics cache.')
        subprocess.run(['bcftools', 'index', '-t', temporary_vcf], check=True)
        os.replace(temporary_vcf, output_vcf_file)
        os.replace(temporary_vcf + '.tbi', output_vcf_file + '.tbi')
        # A previous CSI would otherwise take precedence over the new TBI.
        if os.path.exists(output_vcf_file + '.csi'):
            os.remove(output_vcf_file + '.csi')
    if show_progress:
        print(f'LD pruning: {len(site_df)} input, {len(qualified)} QC-passing, '
              f'{len(retained)} retained sites -> {output_vcf_file}', flush=True)
    return output_vcf_file


if __name__ == "__main__":
    pass
