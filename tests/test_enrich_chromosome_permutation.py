import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.enrich import (chromosome_slices, enrichment_permutation_analysis,
                        get_fold_enrichment, get_jaccard, gene_contrast_runner,
                        GENE_PERMUTATION_SCHEME, permute_array_within_chromosomes)
from src.enrich import win_contrast_runner, WINDOW_PERMUTATION_SCHEME
from src.enrich import site_jaccard_contrast_runner, SITE_JACCARD_PERMUTATION_SCHEME


class ChromosomePermutationTest(unittest.TestCase):
    def test_each_chromosome_retains_its_foreground_count(self):
        foreground = np.array([True, False, False, False, True, True])
        slices = chromosome_slices(np.array(['1', '1', '1', '2', '2', '2']))
        self.assertEqual(slices, (slice(0, 3), slice(3, 6)))
        for seed in range(20):
            shifted = permute_array_within_chromosomes(foreground, slices, seed)
            self.assertEqual(int(shifted[:3].sum()), 1)
            self.assertEqual(int(shifted[3:].sum()), 2)
            np.testing.assert_array_equal(
                shifted, permute_array_within_chromosomes(foreground, slices, seed))

    def test_site_enrichment_uses_chromosome_local_shifts(self):
        frame = pd.DataFrame({
            'CHROM': ['2', '1', '2', '1', '2', '1'],
            'POS': [2, 1, 1, 3, 3, 2],
            'foreground': [False, True, True, False, False, False],
            'env': [5., 1., 4., 3., 6., 2.],
        })
        result = enrichment_permutation_analysis(
            frame, .5, ['env'], pool_num=1, permutations=12,
            show_progress=False)
        sorted_frame = frame.sort_values(['CHROM', 'POS'])
        foreground = sorted_frame.foreground.to_numpy()
        arrays = {'env': sorted_frame.env.to_numpy()}
        slices = chromosome_slices(sorted_frame.CHROM.to_numpy())
        expected = [
            get_fold_enrichment(
                arrays, permute_array_within_chromosomes(foreground, slices, seed),
                'env', .5)
            for seed in range(12)
        ]
        np.testing.assert_allclose(result['perm_values']['env'], expected)


class GenePermutationTest(unittest.TestCase):
    def test_environment_backgrounds_sort_and_preserve_input(self):
        frame = pd.DataFrame({
            'CHROM': ['2', '1', '2', '1', '1', None],
            'START': [30, 20, 10, 10, 30, 1],
            'target': [True, False, False, True, True, True],
            'env1': [6., np.nan, 4., 1., 3., 10.],
            'env2': [np.nan, 5., 4., 1., 3., 10.],
        }, index=[8, 2, 9, 1, 4, 7])
        original = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            result_path = gene_contrast_runner(
                'genes', {'target': 'target'}, frame, ['env1', 'env2'],
                directory, permutations=12, top_percent=.5)
            with gzip.open(result_path, 'rt') as f:
                result = json.load(f)['all']
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(result['permutation_scheme'], GENE_PERMUTATION_SCHEME)
        self.assertEqual(result['background_info']['env1']['n_target'], 3)
        self.assertEqual(result['background_info']['env2']['n_target'], 2)
        for env in ['env1', 'env2']:
            clean = frame.dropna(subset=['CHROM', 'START', 'target', env]).sort_values(['CHROM', 'START'])
            mask = clean.target.to_numpy(dtype=bool)
            scores = clean[env].to_numpy()
            top = scores >= np.quantile(scores, .5)
            slices = chromosome_slices(clean.CHROM.to_numpy())
            expected = [get_jaccard(permute_array_within_chromosomes(mask, slices, seed), top)
                        for seed in range(12)]
            np.testing.assert_allclose(result['perm_values'][env], expected)
            self.assertEqual(result['obs_values'][env], get_jaccard(mask, top))
            self.assertEqual(result['background_info'][env]['n_excluded'], 2)

    def test_old_cache_is_recomputed_and_new_cache_reused(self):
        frame = pd.DataFrame({'CHROM': ['1', '1'], 'START': [2, 1],
                              'target': [True, False], 'env': [2., 1.]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'genes.enrichment.json.gz'
            with gzip.open(path, 'wt') as f:
                json.dump({'all': {'obs_values': {'env': -99}}}, f)
            gene_contrast_runner('genes', {'target': 'target'}, frame, ['env'],
                                 directory, permutations=3)
            with gzip.open(path, 'rt') as f:
                self.assertEqual(json.load(f)['all']['obs_values']['env'], 1.)
            with patch('src.enrich.permute_array_within_chromosomes', side_effect=AssertionError('cache not reused')):
                gene_contrast_runner('genes', {'target': 'target'}, frame, ['env'],
                                     directory, permutations=3)
            gene_contrast_runner('genes', {'target': 'target'}, frame, ['env'],
                                 directory, permutations=4)
            with gzip.open(path, 'rt') as f:
                self.assertEqual(len(json.load(f)['all']['perm_values']['env']), 4)


class WindowPermutationTest(unittest.TestCase):
    def test_boolean_traits_use_separate_backgrounds_without_mutating_input(self):
        frame = pd.DataFrame({
            'ID': ['Chr02:501-1000', 'Chr01:501-1000', 'Chr02:1-500',
                   'Chr01:1-500', 'Chr01:1001-1500', 'missing'],
            'CHROM': ['Chr02', 'Chr01', 'Chr02', 'Chr01', 'Chr01', 'Chr01'],
            'START': [501, 501, 1, 1, 1001, 1501],
            'END': [1000, 1000, 500, 500, 1500, np.nan],
            'ACR': pd.array([True, False, True, True, False, True], dtype='boolean'),
            'Low_TajimaD': pd.array([True, None, False, True, False, True], dtype='boolean'),
            'Low_Pi': pd.array([None, True, True, False, False, True], dtype='boolean'),
        }, index=[8, 2, 9, 1, 4, 7])
        original = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            path = win_contrast_runner('windows', {'target': 'ACR'}, frame,
                                       ['Low_TajimaD', 'Low_Pi'], directory, permutations=20)
            with gzip.open(path, 'rt') as f:
                result = json.load(f)['all']
        pd.testing.assert_frame_equal(frame, original)
        self.assertNotIn('top_percent', result)
        self.assertEqual(result['permutation_scheme'], WINDOW_PERMUTATION_SCHEME)
        # Chr02 target labels are all True and cannot leave that chromosome.
        # Low_TajimaD: one fixed Chr02 overlap, plus zero or one on Chr01.
        self.assertEqual(set(result['perm_values']['Low_TajimaD']), {1 / 4, 2 / 3})
        self.assertEqual(result['obs_values']['Low_TajimaD'], 2 / 3)
        upper_count = result['perm_values']['Low_TajimaD'].count(2 / 3)
        self.assertEqual(result['stat']['Low_TajimaD']['pval'], 2 * (upper_count + 1) / 21)
        # Low_Pi: one fixed Chr02 overlap, plus either zero or one on Chr01.
        self.assertEqual(result['obs_values']['Low_Pi'], 1 / 3)
        self.assertTrue(set(result['perm_values']['Low_Pi']) <= {1 / 3, 1.})
        self.assertEqual(set(result['perm_values']['Low_Pi']), {1 / 3, 1.})
        for trait in ['Low_TajimaD', 'Low_Pi']:
            self.assertEqual(result['background_info'][trait]['n_windows'], 4)
            self.assertEqual(result['background_info'][trait]['n_excluded'], 2)

    def test_non_boolean_trait_is_rejected(self):
        frame = pd.DataFrame({'CHROM': ['Chr01'], 'START': [1], 'END': [500],
                              'ACR': [True], 'Pi': [.2]})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Pi.*boolean"):
                win_contrast_runner('windows', {'target': 'ACR'}, frame,
                                    ['Pi'], directory, permutations=3)

    def test_cache_reuse_and_trait_change(self):
        frame = pd.DataFrame({'CHROM': ['Chr01', 'Chr01'], 'START': [501, 1],
                              'END': [1000, 500], 'ACR': [True, False],
                              'Low_Pi': [True, False], 'High_Pi': [False, True]})
        with tempfile.TemporaryDirectory() as directory:
            path = win_contrast_runner('windows', {'target': 'ACR'}, frame,
                                       ['Low_Pi'], directory, permutations=3)
            with patch('src.enrich.permute_array_within_chromosomes', side_effect=AssertionError('cache not reused')):
                self.assertEqual(win_contrast_runner('windows', {'target': 'ACR'}, frame,
                                                     ['Low_Pi'], directory, permutations=3), path)
            win_contrast_runner('windows', {'target': 'ACR'}, frame,
                                ['High_Pi'], directory, permutations=3)
            with gzip.open(path, 'rt') as f:
                self.assertEqual(json.load(f)['all']['trait_id_list'], ['High_Pi'])


class SiteJaccardPermutationTest(unittest.TestCase):
    def test_pos_backgrounds_and_chromosome_shifts_preserve_input(self):
        frame = pd.DataFrame({
            'ID': ['Chr02:501', 'Chr01:501', 'Chr02:1', 'Chr01:1', 'Chr01:1001', 'missing'],
            'CHROM': ['Chr02', 'Chr01', 'Chr02', 'Chr01', 'Chr01', 'Chr01'],
            'POS': [501, 501, 1, 1, 1001, np.nan],
            'ACR': pd.array([True, False, True, True, False, True], dtype='boolean'),
            'Low_TajimaD': pd.array([True, None, False, True, False, True], dtype='boolean'),
            'Low_Pi': pd.array([None, True, True, False, False, True], dtype='boolean'),
        }, index=[8, 2, 9, 1, 4, 7])
        original = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            path = site_jaccard_contrast_runner(
                'sites', {'target': 'ACR'}, frame, ['Low_TajimaD', 'Low_Pi'],
                directory, permutations=20)
            with gzip.open(path, 'rt') as f:
                result = json.load(f)['all']
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(result['permutation_scheme'], SITE_JACCARD_PERMUTATION_SCHEME)
        self.assertNotIn('top_percent', result)
        self.assertEqual(result['obs_values']['Low_TajimaD'], 2 / 3)
        self.assertEqual(set(result['perm_values']['Low_TajimaD']), {1 / 4, 2 / 3})
        self.assertEqual(result['obs_values']['Low_Pi'], 1 / 3)
        self.assertEqual(set(result['perm_values']['Low_Pi']), {1 / 3, 1.})
        for trait in ['Low_TajimaD', 'Low_Pi']:
            self.assertEqual(result['background_info'][trait]['n_sites'], 4)
            self.assertEqual(result['background_info'][trait]['n_excluded'], 2)

    def test_non_boolean_trait_and_empty_background_are_rejected(self):
        frame = pd.DataFrame({'CHROM': ['Chr01'], 'POS': [1],
                              'ACR': [True], 'Pi': [.2]})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Pi.*boolean'):
                site_jaccard_contrast_runner('sites', {'target': 'ACR'}, frame,
                                             ['Pi'], directory, permutations=3)
            frame['Pi'] = np.nan
            with self.assertRaisesRegex(ValueError, 'No complete sites'):
                site_jaccard_contrast_runner('sites', {'target': 'ACR'}, frame,
                                             ['Pi'], directory, permutations=3)

    def test_cache_reuse_and_permutation_change(self):
        frame = pd.DataFrame({'CHROM': ['Chr01', 'Chr01'], 'POS': [501, 1],
                              'ACR': [True, False], 'Low_Pi': [True, False]})
        with tempfile.TemporaryDirectory() as directory:
            path = site_jaccard_contrast_runner('sites', {'target': 'ACR'}, frame,
                                                ['Low_Pi'], directory, permutations=3)
            with patch('src.enrich.permute_array_within_chromosomes', side_effect=AssertionError('cache not reused')):
                self.assertEqual(site_jaccard_contrast_runner(
                    'sites', {'target': 'ACR'}, frame, ['Low_Pi'], directory, permutations=3), path)
            site_jaccard_contrast_runner('sites', {'target': 'ACR'}, frame,
                                        ['Low_Pi'], directory, permutations=4)
            with gzip.open(path, 'rt') as f:
                self.assertEqual(len(json.load(f)['all']['perm_values']['Low_Pi']), 4)


if __name__ == '__main__':
    unittest.main()
