import unittest

import numpy as np
import pandas as pd

from src.enrich import (chromosome_slices, enrichment_permutation_analysis,
                        get_fold_enrichment, permute_array_within_chromosomes)


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


if __name__ == '__main__':
    unittest.main()
