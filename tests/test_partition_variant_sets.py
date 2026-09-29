import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.partition import build_variant_sets, run_partition_contrast


def _site_df(foreground):
    n_sites = len(foreground)
    first_bin = np.arange(n_sites) < (n_sites // 2)
    return pd.DataFrame({
        "ID": [f"site_{i}" for i in range(n_sites)],
        "foreground": foreground,
        "maf_0.05_0.1": first_bin,
        "maf_0.1_0.2": ~first_bin,
        "maf_0.2_0.3": False,
        "maf_0.3_0.4": False,
        "maf_0.4_0.5": False,
    })


def _load_variant_sets(prefix):
    foreground = np.load(f"{prefix}.foreground.npy")
    permutations = np.load(f"{prefix}.permutations.npy")
    with open(f"{prefix}.metadata.json") as f:
        metadata = json.load(f)
    return foreground, permutations, metadata


class PartitionVariantSetTests(unittest.TestCase):
    def test_default_sampling_is_matched_unique_and_disjoint(self):
        site_df = _site_df([
            True, True, False, False, False,
            True, True, False, False, False,
        ])
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "default")
            build_variant_sets(
                site_df,
                match_controls=("MAF",),
                n_perm=8,
                seed=17,
                output=prefix,
                show_progress=False,
            )
            foreground, permutations, metadata = _load_variant_sets(prefix)

        foreground_set = set(foreground)
        self.assertEqual(permutations.shape, (8, 4))
        self.assertEqual(metadata["control_pool"], "effective_background")
        self.assertEqual(metadata["foreground_overlap_counts"], [0] * 8)
        for permutation in permutations:
            self.assertEqual(len(np.unique(permutation)), 4)
            self.assertTrue(foreground_set.isdisjoint(permutation))
            self.assertEqual(sum(permutation < 5), 2)

    def test_majority_foreground_fails_before_sampling(self):
        site_df = _site_df([True] * 6 + [False] * 4)

        with self.assertRaisesRegex(ValueError, "effective foreground has 6 sites"):
            build_variant_sets(
                site_df,
                match_controls=("MAF",),
                n_perm=2,
                output=None,
                show_progress=False,
            )

    def test_complement_target_uses_original_foreground_as_control_pool(self):
        site_df = _site_df([True] * 6 + [False] * 4)
        effective_foreground = ~site_df["foreground"].to_numpy(dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "complement")
            build_variant_sets(
                site_df,
                foreground=effective_foreground,
                match_controls=("MAF",),
                n_perm=8,
                seed=23,
                target_definition="complement_of_original_foreground",
                output=prefix,
                show_progress=False,
            )
            foreground, permutations, metadata = _load_variant_sets(prefix)

        self.assertEqual(foreground.tolist(), [6, 7, 8, 9])
        self.assertEqual(
            metadata["target_definition"],
            "complement_of_original_foreground",
        )
        for permutation in permutations:
            self.assertTrue(set(permutation).issubset(set(range(6))))

    def test_all_sites_supports_majority_and_records_overlap(self):
        site_df = _site_df([True] * 6 + [False] * 4)
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "all_sites")
            build_variant_sets(
                site_df,
                match_controls=("MAF",),
                n_perm=8,
                seed=31,
                sample_from_all_sites=True,
                output=prefix,
                show_progress=False,
            )
            foreground, permutations, metadata = _load_variant_sets(prefix)

        self.assertEqual(foreground.tolist(), list(range(6)))
        self.assertEqual(permutations.shape, (8, 6))
        self.assertEqual(metadata["control_pool"], "all_sites")
        expected_overlap = [
            int(np.isin(row, foreground).sum()) for row in permutations
        ]
        self.assertEqual(metadata["foreground_overlap_counts"], expected_overlap)
        for permutation in permutations:
            self.assertEqual(len(np.unique(permutation)), 6)
            self.assertEqual(sum(permutation < 5), 5)

    def test_reproducibility_and_stale_configuration_protection(self):
        site_df = _site_df([
            True, True, False, False, False,
            True, True, False, False, False,
        ])
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first")
            second = os.path.join(tmp, "second")
            for prefix in (first, second):
                build_variant_sets(
                    site_df,
                    match_controls=("MAF",),
                    n_perm=5,
                    seed=41,
                    output=prefix,
                    show_progress=False,
                )

            np.testing.assert_array_equal(
                np.load(f"{first}.permutations.npy"),
                np.load(f"{second}.permutations.npy"),
            )

            with self.assertRaisesRegex(ValueError, "different configuration"):
                build_variant_sets(
                    site_df,
                    match_controls=("MAF",),
                    n_perm=6,
                    seed=41,
                    output=first,
                    show_progress=False,
                )

    def test_run_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            run_partition_contrast(
                "contrast",
                {},
                None,
                None,
                [],
                [],
                "unused.vcf",
                "unused-results",
                use_complement=True,
                sample_from_all_sites=True,
            )

    def test_run_logs_and_passes_each_manual_mode(self):
        class StopAfterVariantSets(Exception):
            pass

        site_df = pd.DataFrame({
            "ID": ["a", "b", "c", "d"],
            "foreground": [True, True, True, False],
            "genic_2kb": [True, False, True, False],
        })

        cases = [
            ({}, "default", [True, True, True, False], False),
            (
                {"use_complement": True},
                "complement",
                [False, False, False, True],
                False,
            ),
            (
                {"sample_from_all_sites": True},
                "all_sites",
                [True, True, True, False],
                True,
            ),
        ]

        for kwargs, mode, expected_foreground, expected_all_sites in cases:
            captured = {}

            def fake_build_variant_sets(_site_df, **build_kwargs):
                captured.update(build_kwargs)
                raise StopAfterVariantSets

            with tempfile.TemporaryDirectory() as tmp, patch(
                "src.partition.build_enrich_data_df",
                return_value=site_df.copy(),
            ), patch(
                "src.partition.build_variant_sets",
                side_effect=fake_build_variant_sets,
            ):
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(StopAfterVariantSets):
                    run_partition_contrast(
                        "contrast",
                        {},
                        site_df,
                        None,
                        [],
                        [],
                        "unused.vcf",
                        tmp,
                        match_controls=(),
                        n_perm=2,
                        **kwargs,
                    )

            self.assertIn(f"partition_mode: {mode}", output.getvalue())
            np.testing.assert_array_equal(
                captured["foreground"], expected_foreground
            )
            self.assertEqual(
                captured["sample_from_all_sites"], expected_all_sites
            )

    def test_existing_final_results_reject_a_mode_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_dir = os.path.join(tmp, "final")
            os.makedirs(final_dir)
            result_df = pd.DataFrame({"set": ["ALL_SITE"]})
            result_df.to_csv(os.path.join(final_dir, "reml_summary.csv"), index=False)
            result_df.to_csv(
                os.path.join(final_dir, "reml_summary_unadjusted.csv"),
                index=False,
            )
            result_df.to_csv(os.path.join(final_dir, "score_summary.csv"), index=False)
            with open(os.path.join(final_dir, "matching_metadata.json"), "w") as f:
                json.dump({}, f)
            with open(os.path.join(final_dir, "run_metadata.json"), "w") as f:
                json.dump({
                    "partition_sampling": {
                        "partition_mode": "default",
                        "use_complement": False,
                        "sample_from_all_sites": False,
                        "target_definition": "original_foreground",
                        "control_pool": "effective_background",
                        "match_controls": [],
                        "n_permutations": 2,
                        "seed": 42,
                    },
                    "population_structure": {
                        "pruned_vcf_file": None,
                        "requested_n_pcs": None,
                        "pruned_vcf_signature": None,
                    },
                }, f)

            with self.assertRaisesRegex(
                ValueError,
                "different or unrecorded partition-sampling configuration",
            ):
                run_partition_contrast(
                    "contrast",
                    {},
                    None,
                    None,
                    [],
                    [],
                    "unused.vcf",
                    tmp,
                    match_controls=(),
                    n_perm=2,
                    use_complement=True,
                )


if __name__ == "__main__":
    unittest.main()
