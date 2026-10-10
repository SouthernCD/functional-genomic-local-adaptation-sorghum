import os
"""LD weighted coverage of a binary feature by top GEA sites.

The genome-wide LD pairs are calculated in one call. Their high-LD
edges define blocks for both the observed feature and circular-shift nulls;
per-site maxima against each environment's top sites are also cached once.
"""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
try:
    from numba import njit
except ImportError:
    njit = None

from .enrich import (CHROMOSOME_PERMUTATION_SCHEME, chromosome_slices,
                     permute_array_within_chromosomes,
                     two_sided_empirical_pvalue)
from .ld import calculate_ld_matrix


def _blocks_python(foreground, indptr, indices):
    """Return representative row for each foreground site, in genomic order."""
    representatives = np.full(len(foreground), -1, dtype=np.int64)
    for row in np.flatnonzero(foreground):
        if representatives[row] != -1:
            continue
        representatives[row] = row
        neighbors = indices[indptr[row]:indptr[row + 1]]
        candidates = neighbors[foreground[neighbors] & (representatives[neighbors] == -1)]
        representatives[candidates] = row
    return representatives


if njit is not None:
    @njit
    def _blocks_compiled(foreground, indptr, indices):
        representatives = np.full(len(foreground), -1, dtype=np.int64)
        for row in range(len(foreground)):
            if not foreground[row] or representatives[row] != -1:
                continue
            representatives[row] = row
            for ptr in range(indptr[row], indptr[row + 1]):
                neighbor = indices[ptr]
                if foreground[neighbor] and representatives[neighbor] == -1:
                    representatives[neighbor] = row
        return representatives
else:
    _blocks_compiled = _blocks_python


def _blocks(foreground, graph):
    """Greedy LD blocks; compiled on systems with Numba."""
    return _blocks_compiled(foreground, graph.indptr, graph.indices)


def _scores(representatives, site_max):
    selected = representatives >= 0
    block_ids, inverse = np.unique(representatives[selected], return_inverse=True)
    counts = np.bincount(inverse, minlength=len(block_ids))
    scores = np.zeros((site_max.shape[0], len(block_ids)), dtype=np.float64)
    for env_idx in range(site_max.shape[0]):
        np.maximum.at(scores[env_idx], inverse, site_max[env_idx, selected])
    return block_ids, inverse, counts, scores


def _score_permutation(perm_id, foreground, graph, site_max, chrom_slices):
    permuted = permute_array_within_chromosomes(
        foreground, chrom_slices, seed=perm_id)
    representatives = _blocks(permuted, graph)
    _, _, _, scores = _scores(representatives, site_max)
    return perm_id, tuple(float(scores[i].mean()) for i in range(scores.shape[0]))


_permutation_worker_state = None


def _init_permutation_worker(foreground, graph, site_max, chrom_slices):
    # Under fork these read-only arrays are inherited without copying their
    # large buffers. Only integer permutation IDs enter the task queue.
    global _permutation_worker_state
    _permutation_worker_state = foreground, graph, site_max, chrom_slices


def _run_permutation_worker(perm_id):
    return _score_permutation(perm_id, *_permutation_worker_state)


class FEATURE_COVERAGE_JOB:
    """Run observed and circular-shift LD coverage analysis.

    ``input_data_df`` contains ID, CHROM, POS, a boolean foreground column,
    and numeric GEA score columns (larger means stronger association).
    ``maf_stratify`` names boolean MAF strata, as in site_contrast_runner.
    Each stratum has its own top GEA set and foreground shift universe; LD
    pairs are calculated just once over the full input table.
    ``permutation_pool_num`` defaults to at most four forked workers, separate
    from the LD calculation's ``pool_num``, to bound worker-private memory.
    """

    def __init__(self, input_data_df, input_vcf_file, output_dir, *,
                 env_id_list, foreground_col='foreground', maf_stratify=None,
                 top_percent=0.05,
                 ld_window_size=15000, block_r2_threshold=0.5,
                 score_min_r2=0.25, coverage_r2_thresholds=(0.2, 0.5, 0.8),
                 permutations=1000, pool_num=20, permutation_pool_num=None,
                 silence=True):
        self.input_vcf_file = str(input_vcf_file)
        self.output_dir = os.fspath(output_dir)
        self.foreground_col = foreground_col
        self.env_id_list = list(env_id_list)
        self.stratify_ids = ['all'] + [x for x in (maf_stratify or []) if x != 'all']
        if len(set(self.stratify_ids)) != len(self.stratify_ids):
            raise ValueError('maf_stratify must contain distinct column names')
        self.top_percent = float(top_percent)
        self.ld_window_size = int(ld_window_size)
        self.block_r2_threshold = float(block_r2_threshold)
        self.score_min_r2 = float(score_min_r2)
        self.coverage_r2_thresholds = tuple(float(x) for x in coverage_r2_thresholds)
        self.permutations = int(permutations)
        self.pool_num = int(pool_num)
        self.permutation_pool_num = (
            min(self.pool_num, 4) if permutation_pool_num is None
            else int(permutation_pool_num))
        if not isinstance(silence, (bool, np.bool_)):
            raise TypeError("silence must be True or False")
        self.silence = bool(silence)
        if not self.env_id_list or len(set(self.env_id_list)) != len(self.env_id_list):
            raise ValueError("env_id_list must contain distinct environment columns")
        if not 0 < self.top_percent < 1 or not 0 <= self.score_min_r2 <= 1:
            raise ValueError("top_percent or score_min_r2 is out of range")
        if not 0 <= self.block_r2_threshold <= 1 or any(
                x < 0 or x > 1 for x in self.coverage_r2_thresholds):
            raise ValueError("LD thresholds must be in [0, 1]")
        if min(self.permutations, self.pool_num, self.permutation_pool_num,
               self.ld_window_size) < 1:
            raise ValueError("permutations, pool_num, permutation_pool_num "
                             "and ld_window_size must be positive")
        if self.permutation_pool_num > 1 and 'fork' not in mp.get_all_start_methods():
            raise RuntimeError('Parallel permutations require fork; '
                               'set permutation_pool_num=1 on this platform')
        needed = list(dict.fromkeys(['ID', 'CHROM', 'POS', foreground_col]
                                    + self.stratify_ids[1:] + self.env_id_list))
        missing = [col for col in needed if col not in input_data_df]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if input_data_df.ID.isna().any() or input_data_df.ID.duplicated().any():
            raise ValueError("ID must be present and unique")
        for col in [foreground_col] + self.stratify_ids[1:]:
            if input_data_df[col].isna().any() or not set(
                    input_data_df[col].unique()) <= {True, False, 0, 1}:
                raise ValueError(f'{col} must contain only nonmissing booleans')
        self.site_info_df = input_data_df[needed].sort_values(
            ['CHROM', 'POS'], kind='stable').reset_index(drop=True).copy()
        self.site_info_df['POS'] = pd.to_numeric(self.site_info_df.POS, errors='raise')
        if self.site_info_df.POS.isna().any() or (self.site_info_df.POS < 1).any():
            raise ValueError("POS must contain positive coordinates")
        for env in self.env_id_list:
            self.site_info_df[env] = pd.to_numeric(self.site_info_df[env], errors='raise')
            if not np.isfinite(self.site_info_df[env].dropna().to_numpy()).all():
                raise ValueError(f"{env} has nonfinite GEA scores")
        self.foreground = self.site_info_df[foreground_col].to_numpy(dtype=bool)
        if not self.foreground.any():
            raise ValueError("No foreground sites")
        for stratify_id in self.stratify_ids[1:]:
            mask = self.site_info_df[stratify_id].to_numpy(dtype=bool)
            if not mask.any() or not self.foreground[mask].any():
                raise ValueError(f'{stratify_id} has no foreground sites')

    def _log(self, message):
        if not self.silence:
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            elapsed = (f' +{perf_counter() - self._run_start:.1f}s'
                       if hasattr(self, '_run_start') else '')
            print(f'[{timestamp}{elapsed}] [feature coverage] {message}', flush=True)

    def _result_settings(self):
        return {
            'foreground_col': self.foreground_col,
            'env_id_list': self.env_id_list,
            'maf_stratify': self.stratify_ids,
            'top_percent': self.top_percent,
            'permutations': self.permutations,
            'permutation_scheme': CHROMOSOME_PERMUTATION_SCHEME,
            'ld_window_size': self.ld_window_size,
            'block_r2_threshold': self.block_r2_threshold,
            'score_min_r2': self.score_min_r2,
            'coverage_r2_thresholds': list(self.coverage_r2_thresholds),
            'site_count': len(self.site_info_df),
            'foreground_count': int(self.foreground.sum()),
        }

    def _existing_result(self):
        result_file = os.path.join(self.output_dir, 'feature_coverage.json.gz')
        if not os.path.isfile(result_file):
            return None
        try:
            with gzip.open(result_file, 'rt') as stream:
                result = json.load(stream)
        except (OSError, EOFError, UnicodeError, json.JSONDecodeError):
            self._log('Existing result JSON is incomplete; rerunning')
            return None
        if not isinstance(result, dict):
            self._log('Existing result JSON has an invalid structure; rerunning')
            return None
        settings_file = os.path.join(self.output_dir, 'feature_coverage.settings.json')
        if not os.path.isfile(settings_file):
            self._log('Existing result has no settings file; rerunning')
            return None
        try:
            with open(settings_file) as stream:
                settings_record = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError):
            self._log('Existing settings file is incomplete; rerunning')
            return None
        settings = self._result_settings()
        if settings_record.get('permutation_scheme') != CHROMOSOME_PERMUTATION_SCHEME:
            self._log('Existing results use an earlier permutation scheme; rerunning')
            return None
        mismatches = [key for key, value in settings.items()
                      if settings_record.get(key) != value]
        if settings_record.get('input_vcf_file') != self.input_vcf_file:
            mismatches.append('input_vcf_file')
        if mismatches:
            raise ValueError(
                f'Existing feature coverage results in {self.output_dir} have '
                f'different settings: {", ".join(mismatches)}. '
                'Choose another output_dir to run this configuration.')
        expected_files = [os.path.join(self.output_dir, 'report.md')]
        for stratify_id in self.stratify_ids:
            context_dir = os.path.join(self.output_dir, stratify_id)
            expected_files.extend([
                os.path.join(context_dir, 'feature.LD_blocks.tsv.gz'),
                os.path.join(context_dir, 'feature_block_score.summary.tsv'),
                *(os.path.join(context_dir, f'{env}.feature_block_scores.tsv.gz')
                  for env in self.env_id_list),
            ])
        complete = (all(os.path.isfile(path) and os.stat(path).st_size > 0
                        for path in expected_files)
                    and set(result) == set(self.stratify_ids))
        for stratify_id in self.stratify_ids:
            context = result.get(stratify_id, {})
            complete = (complete and isinstance(context, dict)
                        and context.get('stratify_id') == stratify_id
                        and context.get('env_id_list') == self.env_id_list
                        and context.get('permutations') == self.permutations
                        and all(env in context.get('stat', {})
                                and isinstance(context.get('perm_values', {}).get(env), list)
                                and len(context['perm_values'][env]) == self.permutations
                                for env in self.env_id_list))
        if not complete:
            self._log('Existing result files are incomplete; rerunning')
            return None
        return result

    def _make_contexts(self):
        """Select MAF rows and top GEA sites within each selected universe."""
        contexts = {}
        for stratify_id in self.stratify_ids:
            mask = (np.ones(len(self.site_info_df), dtype=bool) if stratify_id == 'all'
                    else self.site_info_df[stratify_id].to_numpy(dtype=bool))
            rows = np.flatnonzero(mask)
            frame = self.site_info_df.iloc[rows].reset_index(drop=True)
            top_masks, thresholds, counts = [], {}, {}
            for env in self.env_id_list:
                values = frame[env]
                if not values.notna().any():
                    raise ValueError(f'No valid GEA scores for {env} in {stratify_id}')
                threshold = float(values.quantile(1 - self.top_percent))
                top = values.ge(threshold).to_numpy()
                top_masks.append(top)
                thresholds[env] = threshold
                counts[env] = int(top.sum())
            contexts[stratify_id] = dict(
                rows=rows, frame=frame, foreground=self.foreground[rows],
                top_masks=np.stack(top_masks), thresholds=thresholds,
                top_counts=counts)
            self._log(f'{stratify_id}: {len(rows):,} sites, '
                      f'{int(self.foreground[rows].sum()):,} foreground sites')
            for env in self.env_id_list:
                self._log(f'{stratify_id}/{env}: top threshold '
                          f'{thresholds[env]:.6f}, {counts[env]:,} sites')
        return contexts

    def _ld_contexts(self, contexts):
        """Calculate LD once, then restrict the resulting pairs to MAF rows."""
        ld_start = perf_counter()
        pairs = calculate_ld_matrix(
            self.site_info_df[['CHROM', 'POS']], self.input_vcf_file,
            ld_window_size=self.ld_window_size, min_r2=self.score_min_r2,
            pool_num=self.pool_num, show_progress=not self.silence)
        self._log(f'Genome-wide LD: {len(pairs):,} pairs in '
                  f'{perf_counter() - ld_start:.1f}s')
        if pairs.empty:
            a = b = np.empty(0, dtype=np.int64)
            r2 = np.empty(0, dtype=np.float32)
        else:
            a = pairs.INDEX_1.to_numpy(dtype=np.int64)
            b = pairs.INDEX_2.to_numpy(dtype=np.int64)
            r2 = pairs.R2.to_numpy(dtype=np.float32)
        n = len(self.site_info_df)
        for stratify_id, context in contexts.items():
            rows = context['rows']
            # An LD pair contributes only when both sites are in this MAF bin.
            if stratify_id == 'all':
                left, right, values = a, b, r2
                inside_count = len(a)
            else:
                member = np.zeros(n, dtype=bool)
                member[rows] = True
                inside = member[a] & member[b]
                left = np.searchsorted(rows, a[inside])
                right = np.searchsorted(rows, b[inside])
                values = r2[inside]
                inside_count = int(inside.sum())
            top_masks = context['top_masks']
            site_max = np.zeros((len(self.env_id_list), len(rows)), dtype=np.float32)
            site_max[top_masks] = 1.0
            for env_idx, top in enumerate(top_masks):
                reverse_hit = top[right]
                forward_hit = top[left]
                np.maximum.at(site_max[env_idx], left[reverse_hit], values[reverse_hit])
                np.maximum.at(site_max[env_idx], right[forward_hit], values[forward_hit])
            high = values > self.block_r2_threshold
            high_left, high_right = left[high], right[high]
            graph = coo_matrix(
                (np.ones(2 * len(high_left), dtype=np.uint8),
                 (np.r_[high_left, high_right], np.r_[high_right, high_left])),
                shape=(len(rows), len(rows))).tocsr()
            context['graph'] = graph
            context['site_max'] = site_max
            self._log(f'{stratify_id}: {inside_count:,} in-stratum LD pairs, '
                      f'{graph.nnz // 2:,} high-LD edges')
        del pairs, a, b, r2
        return contexts

    def _observed_context(self, stratify_id, context):
        start = perf_counter()
        frame = context['frame']
        foreground = context['foreground']
        graph = context['graph']
        site_max = context['site_max']
        top_masks = context['top_masks']
        context_dir = os.path.join(self.output_dir, stratify_id)
        os.makedirs(context_dir, exist_ok=True)
        representatives = _blocks(foreground, graph)
        block_ids, inverse, counts, scores_by_env = _scores(representatives, site_max)
        if not len(block_ids):
            raise ValueError(f'No foreground LD blocks in {stratify_id}')
        selected_rows = np.flatnonzero(foreground)
        pd.DataFrame({
            'ID': frame.ID.iloc[selected_rows].to_numpy(),
            'CHROM': frame.CHROM.iloc[selected_rows].to_numpy(),
            'POS': frame.POS.iloc[selected_rows].to_numpy(),
            'feature_block_id': inverse,
            'representative_ID': frame.ID.iloc[block_ids[inverse]].to_numpy(),
        }).to_csv(os.path.join(context_dir, 'feature.LD_blocks.tsv.gz'), sep='\t',
                  index=False, compression='gzip')
        observed, summaries = {}, []
        for env_idx, env in enumerate(self.env_id_list):
            scores = scores_by_env[env_idx]
            direct_hit = np.bincount(
                inverse, weights=top_masks[env_idx, selected_rows].astype(np.int8),
                minlength=len(block_ids)) > 0
            block_df = pd.DataFrame({
                'feature_block_id': np.arange(len(block_ids)),
                'representative_ID': frame.ID.iloc[block_ids].to_numpy(),
                'n_feature_snps': counts, 'max_r2_topGEA': scores,
            })
            for threshold in self.coverage_r2_thresholds:
                block_df[f'hit_r2_{threshold:g}'] = scores >= threshold
            block_df.to_csv(os.path.join(context_dir, f'{env}.feature_block_scores.tsv.gz'),
                            sep='\t', index=False, compression='gzip')
            record = {
                'STRATIFY_ID': stratify_id, 'ENV': env,
                'SITE_NUM': len(frame), 'FOREGROUND_NUM': int(foreground.sum()),
                'TOP_GEA_THRESHOLD': context['thresholds'][env],
                'TOP_GEA_NUM': context['top_counts'][env],
                'FEATURE_BLOCK_NUM': len(block_ids),
                'TOTAL_SCORE': float(scores.sum()),
                'NORMALIZED_SCORE': float(scores.mean()),
                'DIRECT_FEATURE_HIT_PROP': float(np.mean(direct_hit)),
            }
            for threshold in self.coverage_r2_thresholds:
                record[f'COVERAGE_R2_{threshold:g}'] = float(np.mean(scores >= threshold))
            observed[env] = record['NORMALIZED_SCORE']
            summaries.append(record)
            self._log(f'{stratify_id}/{env}: {len(block_ids):,} blocks, '
                      f'normalized score {observed[env]:.6f}')
        self._log(f'{stratify_id}: observed scoring finished in '
                  f'{perf_counter() - start:.1f}s')
        return observed, summaries

    def _permutations_context(self, stratify_id, context):
        workers = min(self.permutation_pool_num, self.permutations)
        self._log(f'{stratify_id}: running {self.permutations:,} circular-shift '
                  f'permutations with {workers} worker(s)')
        start = perf_counter()
        progress_step = max(1, self.permutations // 40)
        perm_results = np.empty((self.permutations, len(self.env_id_list)), dtype=float)
        foreground = context['foreground']
        graph = context['graph']
        site_max = context['site_max']
        chrom_slices = chromosome_slices(context['frame']['CHROM'].to_numpy())

        def collect(results):
            for completed, (perm_id, values) in enumerate(results, start=1):
                perm_results[perm_id] = values
                if completed % progress_step == 0 or completed == self.permutations:
                    elapsed = perf_counter() - start
                    remaining = elapsed / completed * (self.permutations - completed)
                    self._log(f'{stratify_id} permutations: '
                              f'{completed:,}/{self.permutations:,} '
                              f'({completed / self.permutations:.1%}); '
                              f'elapsed {elapsed:.1f}s, ETA {remaining:.1f}s')

        if workers == 1:
            collect(_score_permutation(i, foreground, graph, site_max, chrom_slices)
                    for i in range(self.permutations))
        else:
            for array in (foreground, site_max, graph.indptr, graph.indices, graph.data):
                array.flags.writeable = False
            with mp.get_context('fork').Pool(
                processes=workers, initializer=_init_permutation_worker,
                initargs=(foreground, graph, site_max, chrom_slices),
            ) as pool:
                collect(pool.imap_unordered(_run_permutation_worker,
                                            range(self.permutations), chunksize=1))
        self._log(f'{stratify_id}: permutations finished in '
                  f'{perf_counter() - start:.1f}s')
        return {env: perm_results[:, i].tolist()
                for i, env in enumerate(self.env_id_list)}

    def run(self):
        run_start = perf_counter()
        self._run_start = run_start
        existing = self._existing_result()
        if existing is not None:
            print(f'Feature coverage results already exist in {self.output_dir}; '
                  'skipping run.', flush=True)
            result_file = os.path.join(self.output_dir, 'feature_coverage.json.gz')
            return result_file
        os.makedirs(self.output_dir, exist_ok=True)
        self._log(f'Starting {self.foreground_col}: '
                  f'{len(self.site_info_df):,} sites, '
                  f'{int(self.foreground.sum()):,} foreground sites')
        self._log('Selecting top GEA sites within each MAF stratum')
        start = perf_counter()
        contexts = self._make_contexts()
        self._log(f'Top GEA selection finished in {perf_counter() - start:.1f}s')
        self._log('Calculating genome-wide LD once and assigning pairs to MAF strata')
        self._ld_contexts(contexts)
        results, all_summaries = {}, []
        for stratify_id, context in contexts.items():
            self._log(f'{stratify_id}: defining observed LD blocks and scores')
            observed, summaries = self._observed_context(stratify_id, context)
            permutation_values = self._permutations_context(stratify_id, context)
            stat = {}
            for record in summaries:
                env = record['ENV']
                stat[env] = two_sided_empirical_pvalue(
                    observed[env], permutation_values[env])
                record.update({'P_VALUE': stat[env]['pval'],
                               'MEAN_PERM': float(stat[env]['mean_perm']),
                               'STD_PERM': float(stat[env]['std_perm'])})
                self._log(f'{stratify_id}/{env}: empirical P = '
                          f'{stat[env]["pval"]:.6g}')
            pd.DataFrame(summaries).to_csv(
                os.path.join(os.path.join(self.output_dir, stratify_id), 'feature_block_score.summary.tsv'),
                sep='\t', index=False)
            all_summaries.extend(summaries)
            # Match the result schema returned by enrichment_permutation_analysis.
            results[stratify_id] = {
                'obs_values': observed, 'perm_values': permutation_values,
                'stat': stat, 'permutations': self.permutations,
                'stratify_id': stratify_id, 'top_percent': self.top_percent,
                'env_id_list': self.env_id_list,
            }
            del context['graph'], context['site_max']
        summary = pd.DataFrame(all_summaries)
        summary.to_csv(os.path.join(self.output_dir, 'feature_block_score.summary.tsv'),
                       sep='\t', index=False)
        report = [
            f'# Feature coverage: {self.foreground_col}', '',
            f'- Sites: {len(self.site_info_df):,}',
            f'- Foreground sites: {int(self.foreground.sum()):,}',
            f'- MAF strata: {", ".join(self.stratify_ids)}',
            f'- Top GEA fraction within each stratum: {self.top_percent:g}',
            f'- LD window: {self.ld_window_size:,} bp',
            f'- Block threshold: r² > {self.block_r2_threshold:g}',
            f'- Minimum scored LD: r² ≥ {self.score_min_r2:g}',
            f'- Within-chromosome circular-shift permutations per stratum: '
            f'{self.permutations:,}', '',
            'LD is calculated once across all sites. For each MAF stratum, only '
            'pairs whose endpoints both belong to the stratum contribute to '
            'blocks and scores. Top GEA sites use the sites in that stratum; '
            'foreground labels are circularly shifted within each chromosome.', '',
            'Coverage thresholds below the minimum scored LD use that minimum '
            'as their effective cutoff, apart from direct top-site hits.', '',
            '## Results', '', summary.to_string(index=False), '',
            'Empirical two-sided P values use inclusive tails and a +1 correction.',
        ]
        with open(f'{self.output_dir}/report.md', 'w') as stream:
            stream.write('\n'.join(report) + '\n')
        settings = {**self._result_settings(), 'input_vcf_file': self.input_vcf_file}
        with open(f'{self.output_dir}/feature_coverage.settings.json', 'w') as stream:
            stream.write(json.dumps(settings, indent=2) + '\n')
        result_file = os.path.join(self.output_dir, 'feature_coverage.json.gz')
        temporary_result_file = os.path.join(self.output_dir, 'feature_coverage.json.gz.tmp')
        with gzip.open(temporary_result_file, 'wt') as stream:
            json.dump(results, stream, indent=2)
        os.replace(temporary_result_file, result_file)
        self._log(f'Finished in {perf_counter() - run_start:.1f}s; '
                  f'outputs: {self.output_dir}')
        return result_file
