"""Use fixed causal genotypes and effects with PhenotypeSimulator random effects.

The caller selects causal SNPs and beta. This module never samples either one.
Component targets are *realized sample variances* (ddof=1): the R backend
rescales each draw. Consequently a rescaled random effect is not an exact
unconditional N(0, sigma² K) draw. See README.md before interpreting h².
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


_DOBBY_RSCRIPT = Path(
    "/lustre/home/dobby/Program/agent_program/micromamba-root/"
    "envs/phenotype-simulator/bin/Rscript"
)
_SUFFIXES = {
    "phenotype_file": ".phenotype.tsv",
    "gemma_phenotype_file": ".gemma.pheno",
    "component_file": ".components.tsv",
    "causal_file": ".causal.tsv",
    "backend_info_file": ".backend.tsv",
    "metadata_file": ".meta.json",
}


def _ids(values, label):
    raw = pd.Index(values)
    if raw.hasnans:
        raise ValueError(f"{label} must contain unique, nonempty IDs")
    ids = raw.astype(str)
    if not ids.is_unique or any(not x for x in ids):
        raise ValueError(f"{label} must contain unique, nonempty IDs")
    if any("\t" in x or "\n" in x or "\r" in x for x in ids):
        raise ValueError(f"{label} cannot contain tabs or newlines")
    return ids


def _numeric(frame, label):
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains missing or nonfinite values")
    return values


def simulate_phenotype_with_kinship(
    causal_genotype_df: pd.DataFrame,
    causal_effect_df: pd.DataFrame,
    kinship_df: pd.DataFrame,
    output_prefix: str | Path,
    *,
    causal_var: float,
    kinship_var: float,
    residual_var: float,
    seed: int,
    rscript: str | Path | None = None,
    backend_script: str | Path | None = None,
) -> dict:
    """Simulate one trait from fixed causal design and a sample-aligned K.

    Genotypes are samples x SNPs, with sample IDs in the index. Effects have
    unique SNP_ID and beta_raw columns. K has sample IDs on both axes in the
    *same order* as genotype rows. No samples or missing calls are imputed.
    The three nonnegative target sample variances must sum to one.
    """
    if not isinstance(causal_genotype_df, pd.DataFrame) or causal_genotype_df.empty:
        raise ValueError("causal_genotype_df must be a nonempty DataFrame")
    if not isinstance(causal_effect_df, pd.DataFrame):
        raise ValueError("causal_effect_df must be a DataFrame")
    if not isinstance(kinship_df, pd.DataFrame):
        raise ValueError("kinship_df must be a DataFrame")
    if not {"SNP_ID", "beta_raw"}.issubset(causal_effect_df.columns):
        raise ValueError("causal_effect_df requires SNP_ID and beta_raw columns")

    genotype = causal_genotype_df.copy()
    genotype.index = _ids(genotype.index, "genotype sample IDs")
    genotype.columns = _ids(genotype.columns, "genotype SNP IDs")
    if "sample_id" in genotype.columns:
        raise ValueError("sample_id is reserved for the genotype sample column")
    effects = causal_effect_df[["SNP_ID", "beta_raw"]].copy()
    effects["SNP_ID"] = _ids(effects["SNP_ID"], "effect SNP IDs")
    if set(effects["SNP_ID"]) != set(genotype.columns):
        raise ValueError("effect SNP IDs must exactly match genotype SNP IDs")
    genotype = genotype.loc[:, effects["SNP_ID"].tolist()]
    G = _numeric(genotype, "causal genotype")
    beta = _numeric(effects[["beta_raw"]], "beta_raw").ravel()
    if ((G < 0) | (G > 2)).any():
        raise ValueError("causal genotype dosages must be in [0, 2]")
    if len(genotype) < 2:
        raise ValueError("at least two samples are required")
    if (np.var(G, axis=0, ddof=1) <= 0).any():
        raise ValueError("causal genotype contains a monomorphic SNP")

    K_df = kinship_df.copy()
    K_df.index = _ids(K_df.index, "kinship row IDs")
    K_df.columns = _ids(K_df.columns, "kinship column IDs")
    if not K_df.index.equals(genotype.index) or not K_df.columns.equals(genotype.index):
        raise ValueError("kinship rows and columns must match genotype sample order exactly")
    K = _numeric(K_df, "kinship")
    if not np.allclose(K, K.T, rtol=1e-8, atol=1e-10):
        raise ValueError("kinship must be symmetric")
    eigenvalues = np.linalg.eigvalsh((K + K.T) / 2)
    if eigenvalues[0] <= max(1e-10, 1e-8 * abs(eigenvalues[-1])):
        raise ValueError(
            "kinship must be positive definite for PhenotypeSimulator chol(); "
            f"minimum eigenvalue is {eigenvalues[0]:.6g}"
        )

    targets = np.asarray([causal_var, kinship_var, residual_var], dtype=float)
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("variance targets must be finite and nonnegative")
    if not np.isclose(targets.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("causal_var + kinship_var + residual_var must equal 1")
    if targets[0] > 0 and np.var(G @ beta, ddof=1) <= 0:
        raise ValueError("G @ beta has zero sample variance")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**31 - 2002:
        raise ValueError("seed must be an integer in [0, 2147481646]")

    backend = Path(backend_script) if backend_script else Path(__file__).with_name(
        "phenotype_simulator_backend.R"
    )
    if not backend.is_file():
        raise FileNotFoundError(f"R backend not found: {backend}")
    rscript = str(rscript or (_DOBBY_RSCRIPT if _DOBBY_RSCRIPT.is_file() else "Rscript"))
    prefix = Path(output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {key: Path(f"{prefix}{suffix}") for key, suffix in _SUFFIXES.items()}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"output already exists: {existing}")

    with tempfile.TemporaryDirectory(prefix="phenosim_", dir=prefix.parent) as dirname:
        tmp = Path(dirname)
        genotype.insert(0, "sample_id", genotype.index)
        K_df.insert(0, "sample_id", K_df.index)
        geno_file, effect_file, K_file = [tmp / x for x in ("genotype.tsv", "effects.tsv", "kinship.tsv")]
        genotype.to_csv(geno_file, sep="\t", index=False)
        effects.to_csv(effect_file, sep="\t", index=False)
        K_df.to_csv(K_file, sep="\t", index=False, float_format="%.17g")
        tmp_prefix = tmp / "output"
        command = [rscript, str(backend), str(geno_file), str(effect_file),
                   str(K_file), str(tmp_prefix), *map(str, targets), str(int(seed))]
        try:
            run = subprocess.run(command, text=True, capture_output=True, check=False)
        except OSError as exc:
            raise RuntimeError(f"could not start Rscript: {rscript}") from exc
        if run.returncode:
            raise RuntimeError(f"PhenotypeSimulator backend failed:\n{run.stderr}\n{run.stdout}")

        tables = {key: tmp / f"output{suffix}" for key, suffix in _SUFFIXES.items()
                  if key != "metadata_file"}
        if any(not p.is_file() for p in tables.values()):
            raise RuntimeError("R backend did not produce every expected output")
        components = pd.read_csv(tables["component_file"], sep="\t")
        causal = pd.read_csv(tables["causal_file"], sep="\t")
        phenotype = pd.read_csv(tables["phenotype_file"], sep="\t")
        gemma = pd.read_csv(tables["gemma_phenotype_file"], header=None)
        cols = ["causal_effect", "kinship_effect", "residual"]
        if (components["sample_id"].tolist() != genotype["sample_id"].tolist()
                or phenotype["sample_id"].tolist() != genotype["sample_id"].tolist()
                or causal["SNP_ID"].tolist() != effects["SNP_ID"].tolist()
                or len(gemma) != len(genotype)):
            raise RuntimeError("R output order or dimensions do not match input")
        values = components[cols + ["phenotype"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values[:, :3].sum(axis=1), values[:, 3], atol=1e-10):
            raise RuntimeError("R output components are invalid")
        if not np.allclose(phenotype["phenotype"], values[:, 3], atol=1e-10):
            raise RuntimeError("phenotype output differs from components")
        if not np.allclose(gemma.iloc[:, 0], values[:, 3], atol=1e-10):
            raise RuntimeError("GEMMA phenotype output differs from components")
        if not np.allclose(G @ causal["beta_scaled"].to_numpy(dtype=float), values[:, 0], atol=1e-9):
            raise RuntimeError("scaled causal beta does not reproduce causal component")
        realized = components[cols].var(ddof=1)
        if not np.allclose(realized.to_numpy(), targets, rtol=1e-6, atol=1e-8):
            raise RuntimeError("component sample variances missed their targets")
        backend_info = dict(pd.read_csv(tables["backend_info_file"], sep="\t").values)
        metadata = {
            "seed": int(seed), "n_samples": len(genotype), "n_causal": len(effects),
            "variance_semantics": "realized sample variance, ddof=1",
            "target_variance": dict(zip(cols, map(float, targets))),
            "realized_component_variance": realized.to_dict(),
            "realized_phenotype_variance": float(components["phenotype"].var(ddof=1)),
            "component_covariance": components[cols].cov().to_dict(),
            "kinship_min_eigenvalue": float(eigenvalues[0]),
            "backend": backend_info,
        }
        tables["metadata_file"] = tmp / "output.meta.json"
        tables["metadata_file"].write_text(json.dumps(metadata, indent=2) + "\n")
        for key, destination in outputs.items():
            shutil.move(str(tables[key]), str(destination))
    return {**outputs, "metadata": metadata}
