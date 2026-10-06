import json
import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.simulate import SIMULATION_JOB, sample_causal_snps


class SimulationJobTests(unittest.TestCase):
    def _config(self, root):
        for name in ("genome.fa", "sites.h5", "samples.vcf.gz", "kinship.csv"):
            (root / name).write_text(name)
        return {
            "name": "test", "output_dir": str(root / "results"),
            "genome_fasta_file": str(root / "genome.fa"),
            "site_info_df_h5": str(root / "sites.h5"),
            "input_vcf_file": str(root / "samples.vcf.gz"),
            "kinship_ibs_file": str(root / "kinship.csv"),
            "foreground_col": "foreground", "causal_site_num": 1,
            "pruned_site_parameters": {"method": "test", "seed": 7},
        }

    def test_contrast_build_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            settings = root / "contrasts.json"
            settings.write_text(json.dumps({"te": {
                "foreground": "proximal & TE", "background": "proximal",
                "dataset": "genome"}}))
            config.update(contrast_id="te", contrasts_settings_json_file=str(settings))
            config.pop("foreground_col")
            job = SIMULATION_JOB(config)
            table = pd.DataFrame({"ID": ["a", "b", "c"], "CHROM": [1, 1, 1],
                                  "POS": [1, 2, 3], "MAF": [0.1, 0.2, 0.3],
                                  "proximal": [True, True, False], "TE": [True, False, True]})
            built = job._build_site_info_df(table)
            self.assertEqual(built["ID"].tolist(), ["a", "b"])
            self.assertEqual(built["foreground"].tolist(), [True, False])
            self.assertEqual(built["MAF"].tolist(), [0.1, 0.2])
            self.assertNotIn("genome", table)
            universe = job._site_universe(built)
            self.assertEqual(universe, job._site_universe(built.iloc[::-1]))
            self.assertNotEqual(universe, job._site_universe(table))
            changed = built.copy()
            changed["foreground"] = ~changed["foreground"]
            self.assertEqual(universe, job._site_universe(changed))
            original_hash = job._run_hash(None, False)[0]
            job.contrast = {**job.contrast, "foreground": "TE"}
            self.assertNotEqual(original_hash, job._run_hash(None, False)[0])
            with self.assertRaisesRegex(ValueError, "Unknown contrast_id"):
                SIMULATION_JOB({**config, "contrast_id": "absent"})
            with self.assertRaisesRegex(ValueError, "Provide both"):
                SIMULATION_JOB({**config, "contrasts_settings_json_file": None})
            with self.assertRaisesRegex(ValueError, "requires foreground_col"):
                SIMULATION_JOB({**config, "foreground_col": "TE"})

    def test_pruning_universe_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = SIMULATION_JOB(self._config(root))
            table = pd.DataFrame({"ID": ["a", "b", "c"], "foreground": [True, False, False],
                                  "MAF": [0.1, 0.2, 0.3]})
            job.site_universe = job._site_universe(table)
            job._set_paths(root)
            job._write_site_ids(table.iloc[:2], {"source": "automatic", "seed": 1})
            path = job.pruned_site_ids_file
            self.assertIn(job.site_universe["sha256"][:16], path.name)
            self.assertEqual(job._pruning_sidecar(path), job.pruning_metadata_file)
            ids = job._read_site_ids(path)
            job._validate_pruning_source(path, table.iloc[::-1], ids)
            # All retained IDs still exist, but a discarded candidate was removed.
            with self.assertRaisesRegex(ValueError, "candidate SNP universe"):
                job._validate_pruning_source(path, table.iloc[:2], ids)
            with gzip.open(path, "wt") as out:
                out.write("b\na\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                job._validate_pruning_source(path, table, job._read_site_ids(path))
            job.pruning_metadata_file.unlink()
            with self.assertRaisesRegex(ValueError, "Missing pruning provenance"):
                job._validate_pruning_source(path, table, ids)

    def test_supplied_table_changes_hash_and_requires_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            table = pd.DataFrame({"ID": ["s1"], "foreground": [True], "MAF": [0.2]})
            job = SIMULATION_JOB(config, pruned_site_df=table)
            first, _ = job._run_hash(table, False)
            changed = table.copy()
            changed.loc[0, "MAF"] = 0.3
            second, _ = job._run_hash(changed, False)
            self.assertNotEqual(first, second)
            config.pop("pruned_site_parameters")
            with self.assertRaisesRegex(ValueError, "pruned_site_parameters"):
                SIMULATION_JOB(config, pruned_site_df=table)

    def test_causal_foreground_fraction_is_exact_and_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            config["causal_site_num"] = 20
            fixed = SIMULATION_JOB({**config, "enrichment_ratio": None,
                                    "causal_foreground_fraction": 0.35})
            self.assertIsNone(fixed.enrichment_ratio)
            self.assertEqual(fixed.causal_foreground_fraction, 0.35)
            self.assertNotEqual(fixed._run_hash(None, False)[0],
                                SIMULATION_JOB(config)._run_hash(None, False)[0])
            table = pd.DataFrame({
                "ID": [f"s{i}" for i in range(100)],
                "MAF": [0.1 if i < 50 else 0.3 for i in range(100)],
                "foreground": [i % 2 == 0 for i in range(100)],
            })
            ratio_sampled, _ = sample_causal_snps(
                table, causal_fraction=0.2, seed=17,
            )
            self.assertEqual(int(ratio_sampled["causal"].sum()), 20)
            sampled, summary = sample_causal_snps(
                table, causal_fraction=0.2, enrichment_ratio=None,
                causal_foreground_fraction=0.35, seed=17,
            )
            self.assertEqual(int(sampled["causal"].sum()), 20)
            self.assertEqual(int(sampled.loc[sampled["causal"], "foreground"].sum()), 7)
            self.assertEqual(int(summary["n_causal_fg"].sum()), 7)
            for fraction, expected_fg in ((0.0, 0), (1.0, 20)):
                selected, _ = sample_causal_snps(
                    table, causal_fraction=0.2, enrichment_ratio=None,
                    causal_foreground_fraction=fraction, seed=17,
                )
                self.assertEqual(int(selected.loc[selected["causal"], "foreground"].sum()), expected_fg)

            for updates in (
                {"causal_foreground_fraction": 0.3},
                {"enrichment_ratio": None},
            ):
                with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, "exactly one"):
                    SIMULATION_JOB({**config, **updates})
            for value in (-0.1, 1.1, float("nan"), True):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "causal_foreground_fraction"):
                    SIMULATION_JOB({**config, "enrichment_ratio": None,
                                    "causal_foreground_fraction": value})

            scarce = table.copy()
            scarce["foreground"] = False
            scarce.loc[:4, "foreground"] = True
            with self.assertRaisesRegex(ValueError, "infeasible"):
                sample_causal_snps(
                    scarce, causal_fraction=0.2, enrichment_ratio=None,
                    causal_foreground_fraction=0.5, seed=17,
                )

    def test_gemma_reuses_matching_archived_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            job = SIMULATION_JOB(config)
            job._current_run_spec = job._run_hash(None, False)[1]
            job.output_root.mkdir()
            old = job.output_root / "test.previous"
            old.mkdir()
            with open(old / "test.config.json", "w") as out:
                json.dump(job._current_run_spec, out)
            phenotype = pd.DataFrame({"sample_id": ["a", "b", "c"],
                                      "phenotype": [0.2, 1.3, -0.4]})
            phenotype.to_csv(old / "test.phenotype.csv.gz", index=False,
                             compression="gzip")
            pd.DataFrame({"ID": ["s1", "s2"], "P_WALD": [0.01, 0.2]}).to_csv(
                old / "test.gwas.csv.gz", index=False, compression="gzip")
            (old / "test.gemma.log").write_text("previous GEMMA log\n")
            stage = job.output_root / ".test.current"
            stage.mkdir()
            job._set_paths(stage)
            phenotype.to_csv(f"{job.output_prefix}.phenotype.tsv", sep="\t",
                             index=False)
            with open(job.config_file, "w") as out:
                json.dump(job._current_run_spec, out)
            job._gemma_site_info_df = pd.DataFrame({
                "ID": ["s1", "s2"], "CHROM": ["1", "1"], "POS": [10, 20]})
            with patch("src.simulate.GEMMA_JOB.run",
                       side_effect=AssertionError("GEMMA was rerun")):
                gemma_job, gwas = job._run_gemma("unused.bimbam")
            self.assertEqual(gwas["P_WALD"].tolist(), [0.01, 0.2])
            self.assertTrue(Path(gemma_job.gwas_df_h5).is_file())
            self.assertIn("previous GEMMA log", job.gemma_log_file.read_text())
            self.assertEqual(json.loads(job.config_file.read_text())[
                "gemma_reused_from"], str(old / "test.gwas.csv.gz"))
            phenotype.loc[0, "phenotype"] = 9.0
            phenotype.to_csv(f"{job.output_prefix}.phenotype.tsv", sep="\t",
                             index=False)
            self.assertIsNone(job._find_existing_gemma_result())

    def test_gemma_reuses_unfinished_stage_hdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            job = SIMULATION_JOB(config)
            job._current_run_spec = job._run_hash(None, False)[1]
            job.output_root.mkdir()
            previous = job.output_root / ".test.previous"
            previous.mkdir()
            with open(previous / "test.config.json", "w") as out:
                json.dump(job._current_run_spec, out)
            phenotype = pd.DataFrame({"sample_id": ["a", "b"],
                                      "phenotype": [0.2, 1.3]})
            phenotype.to_csv(previous / "test.phenotype.tsv", sep="\t", index=False)
            h5 = previous / "test.gemma/test.gemma.gwas_df.h5"
            h5.parent.mkdir()
            expected = pd.DataFrame({"ID": ["s1"], "P_WALD": [0.01],
                                     "CHROM": ["1"], "POS": [10],
                                     "P_VALUE": [0.01], "FDR": [0.01]})
            expected.to_hdf(h5, key="df", mode="w")
            stage = job.output_root / ".test.current"
            stage.mkdir()
            job._set_paths(stage)
            phenotype.to_csv(f"{job.output_prefix}.phenotype.tsv", sep="\t",
                             index=False)
            with open(job.config_file, "w") as out:
                json.dump(job._current_run_spec, out)
            with patch("src.simulate.GEMMA_JOB.run",
                       side_effect=AssertionError("GEMMA was rerun")):
                gemma_job, actual = job._run_gemma("unused.bimbam")
            pd.testing.assert_frame_equal(actual, expected)
            pd.testing.assert_frame_equal(pd.read_hdf(gemma_job.gwas_df_h5), expected)

    def test_effect_assignment_normal_constant_and_random_sign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            normal_config = {**config, "effect_assignment": {
                "method": "normal", "beta_loc": 0.5, "scale": 1.2}}
            normal_job = SIMULATION_JOB(normal_config)
            expected = np.random.default_rng(12348).normal(0.5, 1.2, 3)
            np.testing.assert_allclose(normal_job._assign_effects(3), expected)
            alias_job = SIMULATION_JOB({**config, "effect_assignment": {
                "method": "normal", "beta_loc": 0.5, "beta_scale": 1.2}})
            self.assertEqual(alias_job.effect_assignment, normal_job.effect_assignment)
            constant_config = {**config, "effect_assignment": {
                "method": "constant", "value": -0.75}}
            constant_job = SIMULATION_JOB(constant_config)
            np.testing.assert_array_equal(
                constant_job._assign_effects(3), [-0.75] * 3)
            signed_config = {**config, "effect_assignment": {
                "method": "random_sign", "magnitude": 0.75}}
            signed_job = SIMULATION_JOB(signed_config)
            signed = signed_job._assign_effects(10000)
            np.testing.assert_array_equal(np.abs(signed), np.full(10000, 0.75))
            self.assertEqual(set(np.unique(signed)), {-0.75, 0.75})
            self.assertTrue(0.47 < np.mean(signed > 0) < 0.53)
            np.testing.assert_array_equal(signed_job._assign_effects(10000), signed)
            normal_hash, _ = normal_job._run_hash(None, False)
            alias_hash, _ = alias_job._run_hash(None, False)
            constant_hash, _ = constant_job._run_hash(None, False)
            signed_hash, _ = signed_job._run_hash(None, False)
            self.assertEqual(normal_hash, alias_hash)
            self.assertNotEqual(normal_hash, constant_hash)
            self.assertNotEqual(signed_hash, constant_hash)

            constant_job._set_paths(root)
            sites = pd.DataFrame({"ID": ["s1", "s2"]})
            genotypes = pd.DataFrame([[0, 1], [2, 1]],
                                     index=["s1", "s2"], columns=["a", "b"])
            with patch("src.simulate.get_genotype_df_from_vcf_by_site_df",
                       return_value=genotypes):
                _, effects = constant_job._prepare_causal_effects(sites)
            self.assertEqual(effects["beta_raw"].tolist(), [-0.75, -0.75])
            saved = pd.read_csv(constant_job.causal_effect_file, sep="\t")
            self.assertEqual(saved["beta_raw"].tolist(), [-0.75, -0.75])
            signed_job._set_paths(root)
            with patch("src.simulate.get_genotype_df_from_vcf_by_site_df",
                       return_value=genotypes):
                _, signed_effects = signed_job._prepare_causal_effects(sites)
            np.testing.assert_array_equal(
                signed_effects["beta_raw"], signed_job._assign_effects(2))

    def test_effect_assignment_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            invalid = [
                ({"method": "normal", "beta_loc": 0.0}, "requires exactly"),
                ({"method": "normal", "beta_loc": 0.0, "scale": 0}, "scale"),
                ({"method": "constant", "value": float("nan")}, "finite number"),
                ({"method": "constant", "value": 0}, "cannot be zero"),
                ({"method": "random_sign"}, "requires exactly"),
                ({"method": "random_sign", "magnitude": 0}, "magnitude"),
                ({"method": "random_sign", "magnitude": float("inf")}, "finite number"),
                ({"method": "unknown", "value": 1}, "Unknown"),
            ]
            for assignment, message in invalid:
                with self.subTest(assignment=assignment):
                    with self.assertRaisesRegex(ValueError, message):
                        SIMULATION_JOB({**config, "effect_assignment": assignment})
            with self.assertRaisesRegex(ValueError, "Move beta_loc"):
                SIMULATION_JOB({**config, "beta_loc": 0.0})

    def test_random_sign_uniform_effect_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            default_job = SIMULATION_JOB({**config, "effect_assignment": {
                "method": "random_sign_uniform"}})
            self.assertEqual(default_job.effect_assignment, {
                "method": "random_sign_uniform", "low": 0.5, "high": 1.5})
            beta_rng = np.random.default_rng(default_job.beta_seed)
            expected_signs = beta_rng.choice([-1.0, 1.0], size=4)
            expected_magnitudes = beta_rng.uniform(0.5, 1.5, size=4)
            expected = expected_signs * expected_magnitudes
            np.testing.assert_array_equal(default_job._assign_effects(4), expected)
            np.testing.assert_array_equal(default_job._assign_effects(4), expected)

            explicit_job = SIMULATION_JOB({**config, "effect_assignment": {
                "method": "random_sign_uniform", "low": 0.5, "high": 1.5}})
            custom_job = SIMULATION_JOB({**config, "effect_assignment": {
                "method": "random_sign_uniform", "low": 0.2, "high": 0.8}})
            self.assertEqual(default_job._run_hash(None, False)[0],
                             explicit_job._run_hash(None, False)[0])
            self.assertNotEqual(default_job._run_hash(None, False)[0],
                                custom_job._run_hash(None, False)[0])
            self.assertTrue(np.all(np.abs(custom_job._assign_effects(100)) >= 0.2))
            self.assertTrue(np.all(np.abs(custom_job._assign_effects(100)) < 0.8))

            default_job._set_paths(root)
            sites = pd.DataFrame({"ID": ["s2", "s1"]})
            genotypes = pd.DataFrame([[0, 1], [2, 1]],
                                     index=["s1", "s2"], columns=["a", "b"])
            with patch("src.simulate.get_genotype_df_from_vcf_by_site_df",
                       return_value=genotypes):
                _, effects = default_job._prepare_causal_effects(sites)
            self.assertEqual(effects["SNP_ID"].tolist(), ["s2", "s1"])
            np.testing.assert_array_equal(effects["beta_raw"],
                                          default_job._assign_effects(2))
            saved = pd.read_csv(default_job.causal_effect_file, sep="\t")
            np.testing.assert_allclose(saved["beta_raw"],
                                       default_job._assign_effects(2))

    def test_random_sign_uniform_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            invalid = [
                ({"low": 0}, "low"),
                ({"low": -0.1}, "low"),
                ({"low": 1, "high": 1}, "high"),
                ({"low": 2, "high": 1}, "high"),
                ({"high": float("inf")}, "finite number"),
                ({"low": True}, "finite number"),
                ({"extra": 1}, "requires exactly"),
            ]
            for options, message in invalid:
                with self.subTest(options=options):
                    with self.assertRaisesRegex(ValueError, message):
                        SIMULATION_JOB({**config, "effect_assignment": {
                            "method": "random_sign_uniform", **options}})

    def test_random_sign_normal_effect_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            std_job = SIMULATION_JOB({**config, "effect_assignment": {
                "method": "random_sign_normal", "mean": 1.0, "std": 0.2}})

            rng = np.random.default_rng(std_job.beta_seed)
            magnitudes = np.abs(rng.normal(1.0, 0.2, size=10000))
            signs = rng.choice([-1.0, 1.0], size=10000)
            effects = std_job._assign_effects(10000)
            np.testing.assert_array_equal(effects, signs * magnitudes)
            np.testing.assert_array_equal(std_job._assign_effects(10000), effects)
            self.assertAlmostEqual(np.median(np.abs(effects)), 1.0, delta=0.01)
            self.assertAlmostEqual(np.std(np.abs(effects)), 0.2, delta=0.01)
            self.assertTrue(0.47 < np.mean(effects > 0) < 0.53)

    def test_random_sign_normal_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            invalid = [
                ({"mean": 1.0}, "requires exactly"),
                ({"mean": 1.0, "std": 0.2, "variance": 0.04}, "requires exactly"),
                ({"mean": -1.0, "std": 0.2}, "mean"),
                ({"mean": 1.0, "std": 0}, "std"),
                ({"mean": 1.0, "std": float("inf")}, "finite number"),
                ({"mean": 1.0, "std": True}, "finite number"),
                ({"mean": 1.0, "std": 0.2, "extra": 1}, "requires exactly"),
            ]
            for options, message in invalid:
                with self.subTest(options=options):
                    with self.assertRaisesRegex(ValueError, message):
                        SIMULATION_JOB({**config, "effect_assignment": {
                            "method": "random_sign_normal", **options}})

    def test_random_sign_lognormal_effect_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            job = SIMULATION_JOB({**config, "effect_assignment": {
                "method": "random_sign_lognormal", "log_mean": 0.0,
                "log_std": 1.0}})
            implicit_job = SIMULATION_JOB(config)
            self.assertEqual(implicit_job.effect_assignment,
                             job.effect_assignment)
            self.assertEqual(implicit_job._run_hash(None, False)[0],
                             job._run_hash(None, False)[0])
            rng = np.random.default_rng(job.beta_seed)
            expected = rng.lognormal(0.0, 1.0, size=10000)
            expected *= rng.choice([-1.0, 1.0], size=10000)
            effects = job._assign_effects(10000)
            np.testing.assert_array_equal(effects, expected)
            np.testing.assert_array_equal(implicit_job._assign_effects(10000),
                                          expected)
            np.testing.assert_array_equal(job._assign_effects(10000), effects)
            self.assertTrue(0.47 < np.mean(effects > 0) < 0.53)
            self.assertGreater(np.quantile(np.abs(effects), 0.9) /
                               np.quantile(np.abs(effects), 0.1), 10)

            other = SIMULATION_JOB({**config, "effect_assignment": {
                "method": "random_sign_lognormal", "log_mean": 0.0,
                "log_std": 0.5}})
            self.assertNotEqual(job._run_hash(None, False)[0],
                                other._run_hash(None, False)[0])
            job._set_paths(root)
            sites = pd.DataFrame({"ID": ["s2", "s1"]})
            genotypes = pd.DataFrame([[0, 1], [2, 1]],
                                     index=["s1", "s2"], columns=["a", "b"])
            with patch("src.simulate.get_genotype_df_from_vcf_by_site_df",
                       return_value=genotypes):
                _, saved_effects = job._prepare_causal_effects(sites)
            self.assertEqual(saved_effects["SNP_ID"].tolist(), ["s2", "s1"])
            np.testing.assert_array_equal(saved_effects["beta_raw"],
                                          job._assign_effects(2))
            np.testing.assert_allclose(
                pd.read_csv(job.causal_effect_file, sep="\t")["beta_raw"],
                job._assign_effects(2))

    def test_normal_mixture_effect_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            components = [
                {"weight": 0.5, "std": 0.1},
                {"weight": 0.3, "std": 0.5},
                {"weight": 0.2, "std": 1.0},
            ]
            job = SIMULATION_JOB({**config, "effect_assignment": {
                "method": "normal_mixture", "components": components}})
            components[0]["std"] = 99.0
            self.assertEqual(job.effect_assignment["components"][0]["std"], 0.1)
            rng = np.random.default_rng(job.beta_seed)
            selected = rng.choice(3, size=10000, p=[0.5, 0.3, 0.2])
            expected = rng.normal(0.0, np.array([0.1, 0.5, 1.0])[selected])
            np.testing.assert_array_equal(job._assign_effects(10000), expected)
            np.testing.assert_array_equal(job._assign_effects(10000), expected)
            self.assertTrue(0.47 < np.mean(expected > 0) < 0.53)
            self.assertGreater(np.quantile(np.abs(expected), 0.99),
                               10 * np.quantile(np.abs(expected), 0.5))

    def test_new_effect_assignment_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            invalid = [
                ({"method": "random_sign_lognormal", "log_mean": 0}, "requires exactly"),
                ({"method": "random_sign_lognormal", "log_mean": 0,
                  "log_std": 0}, "log_std"),
                ({"method": "random_sign_lognormal", "log_mean": float("nan"),
                  "log_std": 1}, "finite number"),
                ({"method": "normal_mixture", "components": []}, "at least two"),
                ({"method": "normal_mixture", "components": [
                    {"weight": 0.5, "std": 0.1}, {"weight": 0.5, "std": 0}]}, "std"),
                ({"method": "normal_mixture", "components": [
                    {"weight": 0.5, "std": 0.1}, {"weight": 0.4, "std": 1.0}]},
                 "sum to one"),
                ({"method": "normal_mixture", "components": [
                    {"weight": True, "std": 0.1}, {"weight": 0.5, "std": 1.0}]},
                 "weight"),
            ]
            for assignment, message in invalid:
                with self.subTest(assignment=assignment):
                    with self.assertRaisesRegex(ValueError, message):
                        SIMULATION_JOB({**config, "effect_assignment": assignment})

    def test_vector_effect_assignment_order_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            config["causal_site_num"] = 3
            supplied = np.array([0.25, -1.5, 0.0])
            job = SIMULATION_JOB({**config, "effect_assignment": {
                "method": "vector", "values": supplied}})
            supplied[0] = 99
            self.assertEqual(job.effect_assignment["values"], [0.25, -1.5, 0.0])
            np.testing.assert_array_equal(job._assign_effects(3), [0.25, -1.5, 0.0])

            other = SIMULATION_JOB({**config, "effect_assignment": {
                "method": "vector", "values": [0.25, -1.5, 0.1]}})
            self.assertNotEqual(job._run_hash(None, False)[0],
                                other._run_hash(None, False)[0])

            job._set_paths(root)
            sites = pd.DataFrame({"ID": ["s3", "s1", "s2"]})
            genotypes = pd.DataFrame([[0, 1], [2, 1], [1, 0]],
                                     index=["s1", "s2", "s3"], columns=["a", "b"])
            with patch("src.simulate.get_genotype_df_from_vcf_by_site_df",
                       return_value=genotypes):
                extracted, effects = job._prepare_causal_effects(sites)
            self.assertEqual(extracted.columns.tolist(), ["s3", "s1", "s2"])
            self.assertEqual(effects["SNP_ID"].tolist(), ["s3", "s1", "s2"])
            self.assertEqual(effects["beta_raw"].tolist(), [0.25, -1.5, 0.0])
            saved = pd.read_csv(job.causal_effect_file, sep="\t")
            self.assertEqual(saved["beta_raw"].tolist(), [0.25, -1.5, 0.0])

    def test_vector_effect_assignment_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            config["causal_site_num"] = 2
            invalid = [
                ([1.0], "causal_site_num"),
                ([], "nonempty"),
                ([[1.0, 2.0]], "one-dimensional"),
                ([1.0, float("nan")], "finite"),
                ([1.0, float("inf")], "finite"),
                ([1.0, "bad"], "numeric"),
                ([True, 1.0], "one-dimensional"),
                ([0.0, 0.0], "all be zero"),
            ]
            for values, message in invalid:
                with self.subTest(values=values):
                    with self.assertRaisesRegex(ValueError, message):
                        SIMULATION_JOB({**config, "effect_assignment": {
                            "method": "vector", "values": values}})

            fraction_config = {**config, "causal_site_num": None,
                               "causal_fraction": 0.5,
                               "effect_assignment": {"method": "vector",
                                                     "values": [1.0, -1.0]}}
            job = SIMULATION_JOB(fraction_config)
            with self.assertRaisesRegex(ValueError, "3 causal SNPs were sampled"):
                job._assign_effects(3)

    def test_enrichment_builds_scores_and_archives_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            config["enrichment"] = {"top_percent": 0.1, "pool_num": 1,
                                    "permutations": 3, "show_progress": False}
            job = SIMULATION_JOB(config)
            job._set_paths(root)
            site_info = pd.DataFrame({
                "ID": ["s2", "s1", "s3", "s4"],
                "CHROM": ["2", "1", "3", "4"], "POS": [20, 10, 30, 40],
                "MAF": [0.2, 0.3, 0.2, 0.3],
                "foreground": [False, True, False, True],
            })
            gwas = pd.DataFrame({"ID": ["s1", "s2", "s3", "s4"],
                                 "P_WALD": [0.01, 0.0, 0.5, 0.1]})
            returned = {"stat": {"test": {"obs_value": 1.2, "pval": 0.5}},
                        "perm_values": {"test": [1.0, 1.1, 1.3]}}
            with patch("src.simulate.enrichment_permutation_analysis",
                       return_value=returned) as analysis:
                self.assertEqual(job._run_enrichment(site_info, gwas), returned)
            args, kwargs = analysis.call_args
            context = args[0]
            self.assertEqual(context["ID"].tolist(), site_info["ID"].tolist())
            self.assertEqual(context["test"].iloc[1], 2.0)
            self.assertTrue(np.isfinite(context["test"]).all())
            self.assertEqual(args[1:], (0.1, ["test"], 1, 3))
            self.assertEqual(kwargs["stratify_id"], "all")
            self.assertFalse(kwargs["show_progress"])
            self.assertEqual(job._read_enrichment_result(), returned)

    def test_enrichment_config_validation_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            default = SIMULATION_JOB(config)
            explicit = SIMULATION_JOB({**config, "enrichment": {
                "top_percent": 0.05, "pool_num": 40, "permutations": 1000,
                "show_progress": True, "progress_interval": 2.0}})
            self.assertEqual(default._run_hash(None, False)[0],
                             explicit._run_hash(None, False)[0])
            changed = SIMULATION_JOB({**config, "enrichment": {
                "permutations": 10}})
            self.assertNotEqual(default._run_hash(None, False)[0],
                                changed._run_hash(None, False)[0])
            for option in ({"top_percent": 0}, {"top_percent": 1},
                           {"pool_num": 0}, {"permutations": 1.5},
                           {"show_progress": 1}, {"progress_interval": 0},
                           {"unknown": 1}):
                with self.subTest(option=option), self.assertRaises((ValueError, TypeError)):
                    SIMULATION_JOB({**config, "enrichment": option})

    def test_automatic_pruning_writes_ordered_id_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            config.pop("pruned_site_parameters")
            table = pd.DataFrame({"ID": ["s2", "s1"],
                                  "foreground": [False, True], "MAF": [0.3, 0.2]})
            job = SIMULATION_JOB(config)
            load = (table, ["a", "b"], pd.DataFrame(np.eye(2)))
            with patch.object(job, "_load_inputs", return_value=load), \
                 patch.object(job, "_prune_sites", return_value=table), \
                 patch.object(job, "_sample_causal_sites", side_effect=RuntimeError("stop")):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    job.run(plot=False)
            self.assertEqual(job._read_site_ids(job.pruned_site_ids_file), ["s2", "s1"])
            metadata = json.loads(job.pruning_metadata_file.read_text())
            self.assertEqual(metadata["pruning_record"]["source"], "automatic")
            self.assertEqual(metadata["pruning_record"]["seed"], 12346)

    def test_archive_cache_and_pruning_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = pd.DataFrame({"ID": ["s1", "s2"], "foreground": [True, False],
                                  "MAF": [0.2, 0.3]})
            config = self._config(root)
            table.to_hdf(config["site_info_df_h5"], key="df", mode="w")
            job = SIMULATION_JOB(config, pruned_site_df=table)
            load = (table, ["a", "b"], pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"]))

            def sample(_):
                causal = table.iloc[[0]].copy()
                causal["causal"] = True
                causal.to_csv(job.causal_site_file, sep="\t", index=False)
                pd.DataFrame({"bin": [1]}).to_csv(job.causal_sampling_file, sep="\t", index=False)
                return table, causal, None

            effects = pd.DataFrame({"SNP_ID": ["s1"], "beta_raw": [1.0]})
            genotype = pd.DataFrame({"s1": [0, 2]}, index=["a", "b"])

            def prep(_):
                effects.to_csv(job.causal_effect_file, sep="\t", index=False)
                return genotype, effects

            def phenotype(*_):
                path = Path(f"{job.output_prefix}.phenotype.tsv")
                pd.DataFrame({"sample_id": ["a", "b"], "phenotype": [0.2, 1.2]}).to_csv(
                    path, sep="\t", index=False)
                return {"phenotype_file": path}

            def gemma(_):
                folder = job.output_dir / "test.gemma"
                block = folder / "blocks/block_00016.gemma/output"
                block.mkdir(parents=True)
                (block / "gemma_output.log.txt").write_text("GEMMA succeeded\n")
                h5 = folder / "test.gemma.gwas_df.h5"
                h5.write_text("temporary")
                return SimpleNamespace(gwas_df_h5=str(h5)), pd.DataFrame({
                    "ID": ["s1", "s2"], "P_WALD": [0.01, 0.5], "extra": [4, 5]})

            def qc(*_):
                job.qc_summary_file.write_text(json.dumps({"auc": 1.0}))
                pd.DataFrame({"recovery": [1]}).to_csv(job.qc_recovery_file, sep="\t", index=False)
                pd.DataFrame({"SNP_ID": ["s1"]}).to_csv(job.qc_causal_table_file, sep="\t", index=False)

            def enrichment(*_):
                result = {"stat": {"test": {"obs_value": 1.2, "pval": 0.5}},
                          "perm_values": {"test": [1.0, 1.1]}}
                with gzip.open(job.enrichment_file, "wt") as out:
                    json.dump(result, out)
                return result

            with patch.object(job, "_load_inputs", return_value=load), \
                 patch.object(job, "_sample_causal_sites", side_effect=sample), \
                 patch.object(job, "_prepare_causal_effects", side_effect=prep), \
                 patch.object(job, "_simulate_phenotype", side_effect=phenotype), \
                 patch.object(job, "_prepare_gemma_pheno", return_value="pheno"), \
                 patch.object(job, "_run_gemma", side_effect=gemma), \
                 patch.object(job, "_evaluate_gwas", side_effect=qc), \
                 patch.object(job, "_run_enrichment", side_effect=enrichment):
                result = job.run(plot=False)
            self.assertFalse(result["cached"])
            self.assertEqual(result["gwas_df"].columns.tolist(), ["ID", "P_WALD"])
            self.assertIn("GEMMA succeeded", result["gemma_log_file"].read_text())
            self.assertEqual(result["enrichment_result"]["stat"]["test"]["pval"], 0.5)
            with gzip.open(result["results_json_file"], "rt") as source:
                self.assertEqual(json.load(source)["n_gwas"], 2)
            with gzip.open(result["results_json_file"], "rt") as source:
                self.assertEqual(json.load(source)["enrichment_stat"]["test"]["pval"], 0.5)
            self.assertFalse((result["output_dir"] / "test.gemma").exists())
            saved = json.loads(result["config_file"].read_text())
            self.assertEqual(saved["pruning_record"]["seed"], 7)
            self.assertEqual(
                SIMULATION_JOB._read_site_ids(result["pruned_site_ids_file"]),
                ["s1", "s2"],
            )
            cached_job = SIMULATION_JOB(config, pruned_site_df=table)
            again = cached_job.run(plot=False)
            self.assertTrue(again["cached"])
            self.assertEqual(again["run_hash"], result["run_hash"])
            self.assertEqual(again["enrichment_result"], result["enrichment_result"])
            pd.testing.assert_frame_equal(cached_job.load_pruned_site_df(), table)

            reused_config = {**config, "name": "reused",
                             "pruned_site_parameters": None,
                             "pruned_site_ids_file": str(result["pruned_site_ids_file"])}
            reused = SIMULATION_JOB(reused_config)
            loaded = (table.iloc[::-1], load[1], load[2])
            captured = []

            def stop_after_reconstruction(reconstructed):
                captured.extend(reconstructed["ID"].tolist())
                raise RuntimeError("stopped after reconstruction")

            with patch.object(reused, "_load_inputs", return_value=loaded), \
                 patch.object(reused, "_prune_sites", side_effect=AssertionError("LD was rerun")), \
                 patch.object(reused, "_sample_causal_sites", side_effect=stop_after_reconstruction):
                with self.assertRaisesRegex(RuntimeError, "stopped after reconstruction"):
                    reused.run(plot=False)
            self.assertEqual(captured, ["s1", "s2"])

            missing = root / "missing_ids.txt"
            missing.write_text("s1\nabsent\n")
            missing_config = {**reused_config, "name": "missing",
                              "pruned_site_ids_file": str(missing)}
            with patch.object(SIMULATION_JOB, "_load_inputs", return_value=load), \
                 patch.object(SIMULATION_JOB, "_prune_sites", side_effect=AssertionError("LD was rerun")):
                with self.assertRaisesRegex(ValueError, "Missing pruning provenance metadata"):
                    SIMULATION_JOB(missing_config).run(plot=False)
            duplicate = root / "duplicate_ids.txt"
            duplicate.write_text("s1\ns1\n")
            with self.assertRaisesRegex(ValueError, "duplicate SNP IDs"):
                SIMULATION_JOB._read_site_ids(duplicate)
