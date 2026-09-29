import gzip
import io
import json
import multiprocessing as mp
import tempfile
import unittest
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import cyvcf2
except ImportError:
    sys.modules['src.ld'] = types.SimpleNamespace(calculate_ld_matrix=lambda *args, **kwargs: None)
    from src.feature_coverage import FEATURE_COVERAGE_JOB
    del sys.modules['src.ld']
else:
    from src.feature_coverage import FEATURE_COVERAGE_JOB



class FeatureCoverageTest(unittest.TestCase):
    def test_permutations_keep_foreground_within_each_chromosome(self):
        sites = pd.DataFrame({
            'ID': list('abcdef'), 'CHROM': ['1'] * 3 + ['2'] * 3,
            'POS': [1, 2, 3, 1, 2, 3],
            'foreground': [True, False, False, True, False, False],
            'env': [1, 2, 3, 4, 5, 6],
        })
        empty_pairs = pd.DataFrame(columns=['INDEX_1', 'INDEX_2', 'R2'])
        with tempfile.TemporaryDirectory() as tmp, patch(
                'src.feature_coverage.calculate_ld_matrix', return_value=empty_pairs):
            result_path = FEATURE_COVERAGE_JOB(
                sites, 'unused.vcf', tmp, env_id_list=['env'],
                top_percent=.5, permutations=20, pool_num=1,
                permutation_pool_num=1).run()
            with gzip.open(result_path, 'rt') as stream:
                result = json.load(stream)
        self.assertEqual(result['all']['perm_values']['env'], [.5] * 20)

    def test_maf_strata_select_top_sites_and_drop_cross_stratum_ld(self):
        sites = pd.DataFrame({
            'ID': list('abcd'), 'CHROM': ['1'] * 4, 'POS': [1, 2, 3, 4],
            'foreground': [True, False, True, False],
            'maf_low': [True, True, False, False],
            'maf_high': [False, False, True, True],
            'env': [1, 2, 3, 4],
        })
        pairs = pd.DataFrame({
            'INDEX_1': [0, 0], 'INDEX_2': [1, 2], 'R2': [.4, .9],
        })
        with tempfile.TemporaryDirectory() as tmp, patch(
                'src.feature_coverage.calculate_ld_matrix', return_value=pairs) as ld:
            result = FEATURE_COVERAGE_JOB(
                input_data_df=sites, input_vcf_file='unused.vcf',
                output_dir=tmp, env_id_list=['env'],
                maf_stratify=['all', 'maf_low', 'maf_high'],
                top_percent=.5, permutations=4, pool_num=1,
                permutation_pool_num=1).run()
            with gzip.open(result, 'rt') as stream:
                result = json.load(stream)
            ld.assert_called_once()
            self.assertEqual(list(result), ['all', 'maf_low', 'maf_high'])
            self.assertAlmostEqual(result['maf_low']['obs_values']['env'], .4)
            self.assertAlmostEqual(result['maf_high']['obs_values']['env'], 0.0)
            self.assertAlmostEqual(result['all']['obs_values']['env'], 1.0)
            low_summary = pd.read_csv(
                Path(tmp, 'maf_low', 'feature_block_score.summary.tsv'), sep='\t')
            self.assertEqual(low_summary.TOP_GEA_NUM.iloc[0], 1)
            self.assertEqual(low_summary.FEATURE_BLOCK_NUM.iloc[0], 1)
            with gzip.open(Path(tmp, 'feature_coverage.json.gz'), 'rt') as stream:
                self.assertEqual(json.load(stream), result)

    @unittest.skipUnless('fork' in mp.get_all_start_methods(), 'fork unavailable')
    def test_parallel_permutations_match_serial_by_id(self):
        sites = pd.DataFrame({
            'ID': list('abcdef'), 'CHROM': ['1'] * 3 + ['2'] * 3,
            'POS': [1, 2, 3, 4, 5, 6],
            'feature': [True, False, True, False, True, False],
            'maf_low': [True, True, True, False, False, False],
            'env_a': [0, 1, 2, 3, 4, 5],
            'env_b': [5, 4, 3, 2, 1, 0],
        })
        pairs = pd.DataFrame({
            'INDEX_1': [0, 1, 2, 3, 4],
            'INDEX_2': [1, 2, 3, 4, 5],
            'R2': [.7, .3, .8, .4, .9],
        })
        with tempfile.TemporaryDirectory() as tmp, patch(
                'src.feature_coverage.calculate_ld_matrix', return_value=pairs):
            common = dict(foreground_col='feature',
                          env_id_list=['env_a', 'env_b'], maf_stratify=['all', 'maf_low'],
                          top_percent=.3, permutations=8, pool_num=1)
            serial_path = FEATURE_COVERAGE_JOB(
                sites, 'unused.vcf', Path(tmp) / 'serial',
                permutation_pool_num=1, **common).run()
            parallel_path = FEATURE_COVERAGE_JOB(
                sites, 'unused.vcf', Path(tmp) / 'parallel',
                permutation_pool_num=2, **common).run()
            with gzip.open(serial_path, 'rt') as stream:
                serial = json.load(stream)
            with gzip.open(parallel_path, 'rt') as stream:
                parallel = json.load(stream)
            self.assertEqual(parallel, serial)

    def test_genome_wide_ld_called_once_and_reused_for_permutations(self):
        sites = pd.DataFrame({
            'ID': list('abcde'), 'CHROM': ['1'] * 5,
            'POS': [1, 3, 9, 11, 20],
            'feature': [True, True, False, True, False],
            'maf_low': [True, True, True, False, False],
            'env_a': [0, 0, 5, 0, 0],
            'env_b': [0, 5, 0, 0, 0],
        })
        pairs = {(0, 1): .8, (1, 2): .4, (2, 3): .9, (3, 4): .3}
        calls = []

        def fake_ld(tile, _vcf, **kwargs):
            calls.append(tuple(tile.index))
            rows = [dict(INDEX_1=a, INDEX_2=b, R2=r)
                    for (a, b), r in pairs.items()
                    if a in tile.index and b in tile.index and r >= kwargs['min_r2']]
            return pd.DataFrame(rows, columns=['INDEX_1', 'INDEX_2', 'R2'])

        with tempfile.TemporaryDirectory() as tmp, patch(
                'src.feature_coverage.calculate_ld_matrix', side_effect=fake_ld):
            job = FEATURE_COVERAGE_JOB(
                sites, 'unused.vcf', tmp, foreground_col='feature',
                env_id_list=['env_a', 'env_b'], maf_stratify=['maf_low'], top_percent=.2,
                ld_window_size=5, score_min_r2=.2,
                permutations=5, pool_num=1)
            silent_output = io.StringIO()
            with redirect_stdout(silent_output):
                result_path = job.run()
            with gzip.open(result_path, 'rt') as stream:
                result = json.load(stream)
            self.assertEqual(silent_output.getvalue(), '')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], tuple(range(len(sites))))
            self.assertEqual(set(result), {'all', 'maf_low'})
            self.assertEqual(set(result['all']), {'obs_values', 'perm_values', 'stat', 'permutations', 'stratify_id', 'top_percent', 'env_id_list'})
            self.assertEqual(result['all']['stratify_id'], 'all')
            self.assertEqual(result['maf_low']['stratify_id'], 'maf_low')
            summary = pd.read_csv(Path(tmp, 'feature_block_score.summary.tsv'), sep='\t')
            self.assertEqual(summary.loc[summary.STRATIFY_ID.eq('all'), 'FEATURE_BLOCK_NUM'].iloc[0], 2)
            self.assertAlmostEqual(result['all']['obs_values']['env_a'], .65)
            self.assertAlmostEqual(result['all']['obs_values']['env_b'], .5)
            self.assertEqual(len(result['all']['perm_values']['env_a']), 5)
            self.assertEqual(summary.loc[summary.STRATIFY_ID.eq('all'), 'DIRECT_FEATURE_HIT_PROP'].iloc[0], 0.0)
            self.assertEqual(summary.loc[summary.STRATIFY_ID.eq('all'), 'DIRECT_FEATURE_HIT_PROP'].iloc[1], .5)
            self.assertTrue(Path(tmp, 'all', 'feature.LD_blocks.tsv.gz').is_file())
            with gzip.open(Path(tmp, 'feature_coverage.json.gz'), 'rt') as stream:
                self.assertEqual(json.load(stream)['all']['obs_values'], result['all']['obs_values'])
            skip_output = io.StringIO()
            with redirect_stdout(skip_output):
                cached = job.run()
            self.assertIn('already exist', skip_output.getvalue())
            self.assertEqual(cached, result_path)
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(ValueError, 'different settings'):
                FEATURE_COVERAGE_JOB(
                    sites, 'unused.vcf', tmp, foreground_col='feature',
                    env_id_list=['env_a', 'env_b'], maf_stratify=['maf_low'], top_percent=.3,
                    ld_window_size=5, score_min_r2=.2,
                    permutations=5, pool_num=1).run()
            Path(tmp, 'report.md').unlink()
            with redirect_stdout(io.StringIO()):
                job.run()
            self.assertEqual(len(calls), 2)
            verbose_job = FEATURE_COVERAGE_JOB(
                sites, 'unused.vcf', Path(tmp) / 'verbose',
                foreground_col='feature', env_id_list=['env_a', 'env_b'],
                top_percent=.2, ld_window_size=5, score_min_r2=.2,
                permutations=5, pool_num=1, silence=False)
            verbose_output = io.StringIO()
            with redirect_stdout(verbose_output):
                verbose_job.run()
            self.assertIn('Genome-wide LD:', verbose_output.getvalue())
            self.assertIn('permutations: 5/5', verbose_output.getvalue())
            self.assertRegex(verbose_output.getvalue(),
                             r'\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z \+\d+\.\ds\]')
            self.assertIn('observed scoring finished in', verbose_output.getvalue())
            self.assertIn('Finished in', verbose_output.getvalue())


if __name__ == '__main__':
    unittest.main()
