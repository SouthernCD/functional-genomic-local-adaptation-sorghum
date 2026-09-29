import os
import re
import json
import gzip
import shutil
import hashlib
import subprocess

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .enrich import build_enrich_data_df

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .plot import set_plot_style


def _run_cmd(cmd, cwd=None, quiet=False):
    cmd = [str(x) for x in cmd]

    if not quiet:
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=cwd, check=True)
        return

    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        print("\nCommand failed:")
        print(" ".join(cmd))
        print("\nProgram output:")
        print(result.stdout)
        result.check_returncode()


def _have_score(prefix):
    f = os.path.expanduser(f"{prefix}.profile")
    return os.path.isfile(f) and os.path.getsize(f) > 0


def _remove_bfile(prefix):
    """Remove temporary PLINK files."""
    for ext in ("bed", "bim", "fam", "log", "nosex"):
        f = f"{prefix}.{ext}"
        if os.path.isfile(f):
            os.remove(f)


def _read_ldak_profile(profile_file):
    """Read LDAK 5.2 calc-scores output; Profile_1 is the Genetics score."""
    df = pd.read_csv(profile_file, sep=r"\s+")

    if "Profile_1" not in df.columns:
        raise ValueError(
            f"'Profile_1' not found in {profile_file}. "
            f"Columns: {df.columns.tolist()}"
        )

    if "ID2" in df.columns:
        sample_col = "ID2"
    elif "ID1" in df.columns:
        sample_col = "ID1"
    else:
        raise ValueError(f"Neither ID1 nor ID2 found in {profile_file}.")

    out = df[[sample_col, "Profile_1"]].copy()
    out.columns = ["LIB", "Genetics"]
    out["LIB"] = out["LIB"].astype(str)

    return out


def predict_scores(
    grm_dir,
    prediction_sample_ids,
    pheno_cols=("gsl", "bio18", "al", "striga"),
    blup_dir=None,
    output=None,
    n_jobs=8,
    power=0,
    allow_multi=True,
    keep_pred_match=False,
    overwrite=False,
    show_progress=True,
    plink="plink",
    ldak="ldak",
):
    """
    Apply phenotype-specific BLUP effects to independent prediction samples.

    For each variant set:
      1. Extract its training-QC variants from the common prediction genotype.
      2. Reuse this pred_match for all phenotypes.
      3. Run LDAK --calc-scores.
      4. Read Profile_1 as Genetics.
    """

    if shutil.which(plink) is None:
        raise FileNotFoundError(f"PLINK executable not found: {plink}")
    if shutil.which(ldak) is None:
        raise FileNotFoundError(f"LDAK executable not found: {ldak}")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    grm_dir = os.path.abspath(os.path.expanduser(grm_dir))
    blup_dir = (
        os.path.abspath(os.path.expanduser(blup_dir))
        if blup_dir is not None
        else os.path.join(grm_dir, "blups")
    )
    output = (
        os.path.abspath(os.path.expanduser(output))
        if output is not None
        else os.path.join(grm_dir, "scores")
    )
    os.makedirs(output, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Get persistent master PLINK from build_grms metadata
    # ---------------------------------------------------------------
    grm_metadata_file = os.path.join(grm_dir, "metadata.json")

    if not os.path.isfile(grm_metadata_file):
        raise FileNotFoundError(grm_metadata_file)

    with open(grm_metadata_file) as f:
        grm_metadata = json.load(f)

    master_bfile = grm_metadata.get("master_bfile")

    if master_bfile is None:
        raise ValueError("'master_bfile' not found in build_grms metadata.")

    master_bfile = os.path.abspath(os.path.expanduser(master_bfile))

    if not _have_bfile(master_bfile):
        raise FileNotFoundError(
            f"Master PLINK dataset is incomplete: {master_bfile}"
        )

    # ---------------------------------------------------------------
    # 2. Resolve prediction samples against master FAM
    # ---------------------------------------------------------------
    prediction_sample_ids = list(
        dict.fromkeys(str(x) for x in prediction_sample_ids)
    )

    if not prediction_sample_ids:
        raise ValueError("prediction_sample_ids is empty.")

    fam = pd.read_csv(
        f"{master_bfile}.fam",
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        dtype=str,
    )
    fam.columns = ["FID", "IID"]

    iid_counts = fam["IID"].value_counts()
    duplicated_iids = set(iid_counts[iid_counts > 1].index)

    ambiguous = [
        iid for iid in prediction_sample_ids
        if iid in duplicated_iids
    ]
    if ambiguous:
        raise ValueError(
            f"{len(ambiguous)} prediction IID(s) occur under multiple FIDs. "
            f"Examples: {ambiguous[:10]}"
        )

    fam_by_iid = fam.set_index("IID")

    missing_samples = [
        iid for iid in prediction_sample_ids
        if iid not in fam_by_iid.index
    ]
    if missing_samples:
        raise ValueError(
            f"{len(missing_samples):,} prediction samples are absent from "
            f"master PLINK. Examples: {missing_samples[:10]}"
        )

    sample_pairs = [
        (fam_by_iid.loc[iid, "FID"], iid)
        for iid in prediction_sample_ids
    ]

    # ---------------------------------------------------------------
    # 3. Build prediction genotype once
    # ---------------------------------------------------------------
    prediction_dir = os.path.join(output, "_prediction")
    os.makedirs(prediction_dir, exist_ok=True)

    keep_file = os.path.join(prediction_dir, "samples.keep")
    prediction_bfile = os.path.join(prediction_dir, "genotype")

    with open(keep_file, "w") as f:
        for fid, iid in sample_pairs:
            f.write(f"{fid}\t{iid}\n")

    if _have_bfile(prediction_bfile):
        print(f"Using existing prediction genotype: {prediction_bfile}")
    else:
        print("Building prediction genotype dataset...")

        _run_cmd([
            plink,
            "--bfile", master_bfile,
            "--keep", keep_file,
            "--make-bed",
            "--out", prediction_bfile,
        ])

        if not _have_bfile(prediction_bfile):
            raise RuntimeError("Failed to create prediction genotype dataset.")

    # ---------------------------------------------------------------
    # 4. Find variant sets
    # ---------------------------------------------------------------
    set_names = []

    for name in os.listdir(grm_dir):
        set_dir = os.path.join(grm_dir, name)

        if not os.path.isdir(set_dir):
            continue

        if os.path.isfile(os.path.join(set_dir, "variants.txt")):
            set_names.append(name)

    def set_sort_key(name):
        if name == "ALL_SITE":
            return (-2, 0)

        if name == "FOREGROUND":
            return (-1, 0)

        m = re.fullmatch(r"PERM_(\d+)", name)
        if m:
            return (0, int(m.group(1)))

        return (1, name)

    set_names = sorted(set_names, key=set_sort_key)

    if not set_names:
        raise ValueError(f"No variant sets found under {grm_dir}")

    print(
        f"Prediction: {len(set_names):,} sets × "
        f"{len(pheno_cols):,} phenotypes = "
        f"{len(set_names) * len(pheno_cols):,} score models"
    )

    # ---------------------------------------------------------------
    # 5. Process one variant set
    # ---------------------------------------------------------------
    def process_set(set_name):
        variant_file = os.path.join(
            grm_dir, set_name, "variants.txt"
        )

        set_output = os.path.join(
            output, set_name
        )
        os.makedirs(set_output, exist_ok=True)

        jobs_needed = []

        for pheno_col in pheno_cols:
            blup_file = os.path.join(
                blup_dir,
                set_name,
                pheno_col,
                "model.blup",
            )

            score_prefix = os.path.join(
                set_output,
                pheno_col,
                "model",
            )

            if not os.path.isfile(blup_file):
                raise FileNotFoundError(blup_file)

            if overwrite or not _have_score(score_prefix):
                jobs_needed.append(
                    (pheno_col, blup_file, score_prefix)
                )

        # Only construct pred_match when at least one score is missing.
        if jobs_needed:
            tmp_dir = os.path.join(
                output,
                "_tmp",
                set_name,
            )
            os.makedirs(tmp_dir, exist_ok=True)

            pred_match = os.path.join(
                tmp_dir,
                "pred_match",
            )

            if not _have_bfile(pred_match):
                _run_cmd([
                    plink,
                    "--bfile", prediction_bfile,
                    "--extract", variant_file,
                    "--make-bed",
                    "--out", pred_match,
                ], quiet=(n_jobs > 1))

            if not _have_bfile(pred_match):
                raise RuntimeError(
                    f"{set_name}: failed to create pred_match."
                )

            with open(f"{pred_match}.bim") as f:
                n_prediction_variants = sum(1 for _ in f)

            if n_prediction_variants == 0:
                raise RuntimeError(
                    f"{set_name}: no variants found in prediction genotype."
                )

            for pheno_col, blup_file, score_prefix in jobs_needed:
                os.makedirs(
                    os.path.dirname(score_prefix),
                    exist_ok=True,
                )

                cmd = [
                    ldak,
                    "--calc-scores", score_prefix,
                    "--scorefile", blup_file,
                    "--bfile", pred_match,
                    "--power", str(power),
                ]

                if allow_multi:
                    cmd += ["--allow-multi", "YES"]

                _run_cmd(
                    cmd,
                    quiet=(n_jobs > 1),
                )

                if not _have_score(score_prefix):
                    raise RuntimeError(
                        f"{set_name} / {pheno_col}: "
                        "LDAK did not produce .profile output."
                    )

            if not keep_pred_match:
                _remove_bfile(pred_match)

                try:
                    os.rmdir(tmp_dir)
                except OSError:
                    pass

        # Read all profiles, including skipped ones.
        set_results = []

        for pheno_col in pheno_cols:
            profile_file = os.path.join(
                set_output,
                pheno_col,
                "model.profile",
            )

            if not os.path.isfile(profile_file):
                raise FileNotFoundError(profile_file)

            score_df = _read_ldak_profile(profile_file)

            score_df.insert(0, "phenotype", pheno_col)
            score_df.insert(0, "set", set_name)

            set_results.append(score_df)

        return pd.concat(
            set_results,
            ignore_index=True,
        )

    # ---------------------------------------------------------------
    # 6. Run at variant-set level
    # ---------------------------------------------------------------
    results = []

    if n_jobs == 1:
        iterator = tqdm(
            set_names,
            desc="Predicting scores",
            disable=not show_progress,
        )

        for set_name in iterator:
            results.append(
                process_set(set_name)
            )

    else:
        pbar = tqdm(
            total=len(set_names),
            desc="Predicting scores",
            disable=not show_progress,
        )

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(process_set, set_name): set_name
                for set_name in set_names
            }

            for future in as_completed(futures):
                set_name = futures[future]

                try:
                    result = future.result()
                except Exception as e:
                    pbar.close()
                    raise RuntimeError(
                        f"Prediction failed for {set_name}"
                    ) from e

                results.append(result)
                pbar.update(1)

        pbar.close()

    # ---------------------------------------------------------------
    # 7. Combine scores
    # ---------------------------------------------------------------
    summary_df = pd.concat(
        results,
        ignore_index=True,
    )

    summary_df["perm_index"] = (
        summary_df["set"]
        .str.extract(r"PERM_(\d+)")[0]
        .astype("Int64")
    )

    summary_df["is_foreground"] = (
        summary_df["set"] == "FOREGROUND"
    )

    summary_df["is_all_site"] = (
        summary_df["set"] == "ALL_SITE"
    )

    summary_df = summary_df.sort_values(
        ["phenotype", "is_all_site", "is_foreground", "perm_index", "LIB"],
        ascending=[True, False, False, True, True],
        na_position="first",
    ).reset_index(drop=True)

    summary_file = os.path.join(
        output,
        "score_summary.csv",
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    metadata = {
        "grm_dir": grm_dir,
        "blup_dir": blup_dir,
        "master_bfile": master_bfile,
        "prediction_bfile": prediction_bfile,
        "n_prediction_samples": len(prediction_sample_ids),
        "n_sets": len(set_names),
        "phenotypes": list(pheno_cols),
        "power": power,
        "allow_multi": allow_multi,
        "keep_pred_match": keep_pred_match,
        "n_jobs": n_jobs,
        "summary_file": summary_file,
    }

    with open(
        os.path.join(output, "metadata.json"),
        "w",
    ) as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Finished prediction: "
        f"{len(set_names):,} sets × "
        f"{len(pheno_cols):,} phenotypes."
    )
    print(f"Summary: {summary_file}")

    return summary_df


def _have_blup(prefix):
    """Check whether the main LDAK BLUP outputs exist and are non-empty."""
    files = [
        os.path.expanduser(f"{prefix}.blup"),
        os.path.expanduser(f"{prefix}.pred"),
    ]
    return all(os.path.isfile(f) and os.path.getsize(f) > 0 for f in files)


def run_blups(
    grm_dir,
    reml_dir=None,
    pheno_cols=("gsl", "bio18", "al", "striga"),
    output=None,
    n_jobs=8,
    allow_multi=True,
    overwrite=False,
    show_progress=True,
    ldak="ldak",
):
    """
    Calculate LDAK BLUP SNP effects for every GRM × phenotype combination.

    Requires outputs from:
        build_grms()
        run_reml()

    Parameters
    ----------
    grm_dir : str
        Directory produced by build_grms().

    reml_dir : str or None
        Directory produced by run_reml().
        Default: <grm_dir>/reml

    pheno_cols : iterable
        Phenotypes to process.

    output : str or None
        BLUP output directory.
        Default: <grm_dir>/blups

    n_jobs : int
        Number of simultaneous LDAK jobs.

    allow_multi : bool
        Add '--allow-multi YES', matching the original LDAK 5.2 pipeline.

    overwrite : bool
        Recalculate existing BLUP results.

    show_progress : bool
        Show progress bar.

    ldak : str
        LDAK executable.

    Returns
    -------
    pandas.DataFrame
        Summary of all BLUP jobs.
    """

    if shutil.which(ldak) is None:
        raise FileNotFoundError(f"LDAK executable not found: {ldak}")

    grm_dir = os.path.abspath(os.path.expanduser(grm_dir))
    reml_dir = (
        os.path.abspath(os.path.expanduser(reml_dir))
        if reml_dir is not None
        else os.path.join(grm_dir, "reml")
    )
    output = (
        os.path.abspath(os.path.expanduser(output))
        if output is not None
        else os.path.join(grm_dir, "blups")
    )

    os.makedirs(output, exist_ok=True)

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    # ----------------------------------------------------------------
    # 1. Shared training genotype generated by build_grms()
    # ----------------------------------------------------------------
    training_bfile = os.path.join(grm_dir, "_training_qc", "genotype")

    for ext in ("bed", "bim", "fam"):
        f = f"{training_bfile}.{ext}"
        if not os.path.isfile(f):
            raise FileNotFoundError(f)

    # ----------------------------------------------------------------
    # 2. Find all GRM sets
    # ----------------------------------------------------------------
    set_names = []

    for name in os.listdir(grm_dir):
        set_dir = os.path.join(grm_dir, name)

        if not os.path.isdir(set_dir):
            continue
        if name.startswith("_") or name in {"reml", "blups"}:
            continue

        grm_prefix = os.path.join(set_dir, "grm")
        variant_file = os.path.join(set_dir, "variants.txt")

        if os.path.isfile(f"{grm_prefix}.grm.bin") and os.path.isfile(variant_file):
            set_names.append(name)

    def set_sort_key(name):
        if name == "ALL_SITE":
            return (-2, 0)

        if name == "FOREGROUND":
            return (-1, 0)

        m = re.fullmatch(r"PERM_(\d+)", name)
        if m:
            return (0, int(m.group(1)))

        return (1, name)

    set_names = sorted(set_names, key=set_sort_key)

    if not set_names:
        raise ValueError(f"No valid GRM sets found under {grm_dir}")

    # ----------------------------------------------------------------
    # 3. Validate phenotype + REML files and construct jobs
    # ----------------------------------------------------------------
    jobs = []

    for set_name in set_names:
        grm_prefix = os.path.join(grm_dir, set_name, "grm")
        variant_file = os.path.join(grm_dir, set_name, "variants.txt")

        for pheno_col in pheno_cols:
            pheno_file = os.path.join(
                reml_dir, "_phenotypes", f"{pheno_col}.pheno")
            reml_file = os.path.join(
                reml_dir, set_name, pheno_col, "model.reml")

            if not os.path.isfile(pheno_file):
                raise FileNotFoundError(pheno_file)
            if not os.path.isfile(reml_file):
                raise FileNotFoundError(reml_file)

            jobs.append({
                "set": set_name,
                "phenotype": pheno_col,
                "grm_prefix": grm_prefix,
                "variant_file": variant_file,
                "pheno_file": pheno_file,
                "reml_file": reml_file,
            })

    total_jobs = len(jobs)

    print(
        f"BLUP jobs: {len(set_names):,} GRMs "
        f"× {len(pheno_cols):,} phenotypes "
        f"= {total_jobs:,}"
    )

    # ----------------------------------------------------------------
    # 4. One BLUP job
    # ----------------------------------------------------------------
    def run_one(job):
        set_name = job["set"]
        pheno_col = job["phenotype"]

        result_dir = os.path.join(output, set_name, pheno_col)
        os.makedirs(result_dir, exist_ok=True)

        blup_prefix = os.path.join(result_dir, "model")

        if not overwrite and _have_blup(blup_prefix):
            status = "skipped"

        else:
            cmd = [
                ldak,
                "--calc-blups", blup_prefix,
                "--remlfile", job["reml_file"],
                "--grm", job["grm_prefix"],
                "--bfile", training_bfile,
                "--pheno", job["pheno_file"],
            ]

            if allow_multi:
                cmd += ["--allow-multi", "YES"]

            _run_cmd(cmd, quiet=(n_jobs > 1))

            if not _have_blup(blup_prefix):
                raise RuntimeError(
                    f"{set_name} / {pheno_col}: "
                    "LDAK did not produce complete BLUP output."
                )

            status = "completed"

        return {
            "set": set_name,
            "phenotype": pheno_col,
            "blup_prefix": blup_prefix,
            "blup_file": f"{blup_prefix}.blup",
            "pred_file": f"{blup_prefix}.pred",
            "reml_file": job["reml_file"],
            "grm_prefix": job["grm_prefix"],
            "variant_file": job["variant_file"],
            "status": status,
        }

    # ----------------------------------------------------------------
    # 5. Run jobs
    # ----------------------------------------------------------------
    results = []
    completed = 0
    skipped = 0

    if n_jobs == 1:
        iterator = tqdm(jobs, desc="Running BLUPs", disable=not show_progress)

        for job in iterator:
            result = run_one(job)
            results.append(result)

            if result["status"] == "completed":
                completed += 1
            else:
                skipped += 1

            iterator.set_postfix(completed=completed, skipped=skipped)

    else:
        pbar = tqdm(
            total=total_jobs,
            desc="Running BLUPs",
            disable=not show_progress,
        )

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(run_one, job): job for job in jobs}

            for future in as_completed(futures):
                job = futures[future]

                try:
                    result = future.result()
                except Exception as e:
                    pbar.close()
                    raise RuntimeError(
                        f"BLUP failed for {job['set']} / {job['phenotype']}"
                    ) from e

                results.append(result)

                if result["status"] == "completed":
                    completed += 1
                else:
                    skipped += 1

                pbar.update(1)
                pbar.set_postfix(completed=completed, skipped=skipped)

        pbar.close()

    # ----------------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------------
    summary_df = pd.DataFrame(results)

    summary_df["perm_index"] = summary_df["set"].str.extract(
        r"PERM_(\d+)"
    )[0].astype("Int64")

    summary_df["is_foreground"] = summary_df["set"].eq("FOREGROUND")
    summary_df["is_all_site"] = summary_df["set"].eq("ALL_SITE")

    summary_df = summary_df.sort_values(
        ["phenotype", "is_all_site", "is_foreground", "perm_index"],
        ascending=[True, False, False, True],
        na_position="first",
    ).reset_index(drop=True)

    summary_file = os.path.join(output, "blup_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    metadata = {
        "grm_dir": grm_dir,
        "reml_dir": reml_dir,
        "training_bfile": training_bfile,
        "output": output,
        "n_sets": len(set_names),
        "phenotypes": list(pheno_cols),
        "n_jobs": n_jobs,
        "allow_multi": allow_multi,
        "summary_file": summary_file,
    }

    with open(os.path.join(output, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Finished BLUPs: {completed:,} calculated, "
        f"{skipped:,} skipped, {total_jobs:,} total."
    )
    print(f"Summary: {summary_file}")

    return summary_df


def _have_reml(prefix):
    f = os.path.expanduser(f"{prefix}.reml")
    return os.path.isfile(f) and os.path.getsize(f) > 0


def _parse_ldak_reml(reml_file):
    """
    Parse LDAK 5.2 .reml output.

    Returns
    -------
    dict with keys such as:
        total_samples
        with_phenotypes
        converged
        null_likelihood
        alt_likelihood
        lrt_stat
        lrt_p
        heritability
        heritability_sd
        covar_heritability
    """

    reml_file = os.path.expanduser(reml_file)

    if not os.path.isfile(reml_file):
        raise FileNotFoundError(reml_file)

    result = {
        "total_samples": None,
        "with_phenotypes": None,
        "converged": None,
        "null_likelihood": None,
        "alt_likelihood": None,
        "lrt_stat": None,
        "lrt_p": None,
        "heritability": None,
        "heritability_sd": None,
        "covar_heritability": None,
    }

    with open(reml_file) as f:
        for line in f:
            fields = line.strip().split()

            if not fields:
                continue

            key = fields[0]

            if key == "Total_Samples":
                result["total_samples"] = int(fields[1])

            elif key == "With_Phenotypes":
                result["with_phenotypes"] = int(fields[1])

            elif key == "Converged":
                result["converged"] = fields[1].upper() == "YES"

            elif key == "Null_Likelihood":
                result["null_likelihood"] = float(fields[1])

            elif key == "Alt_Likelihood":
                result["alt_likelihood"] = float(fields[1])

            elif key == "LRT_Stat":
                result["lrt_stat"] = float(fields[1])

            elif key == "LRT_P":
                result["lrt_p"] = float(fields[1])

            elif key == "Covar_Heritability":
                result["covar_heritability"] = float(fields[1])

            elif key.lower() == "her_all":
                result["heritability"] = float(fields[1])
                result["heritability_sd"] = float(fields[2])

    if result["heritability"] is None:
        raise ValueError(
            f"Could not find Her_All in {reml_file}"
        )

    return result


def run_reml(
    grm_dir,
    pheno_df,
    pheno_cols=("gsl", "bio18", "al", "striga"),
    sample_col="LIB",
    output=None,
    n_jobs=8,
    covar_file=None,
    kinship_details=False,
    overwrite=False,
    show_progress=True,
    ldak="ldak",
):
    """
    Run LDAK REML for every GRM × phenotype combination.

    Parameters
    ----------
    grm_dir : str
        Output directory produced by build_grms().

        Expected structure:
            grm_dir/
                FOREGROUND/grm.*
                PERM_0/grm.*
                PERM_1/grm.*
                ...

    pheno_df : pandas.DataFrame
        Phenotype/environment dataframe.

        Must contain:
            sample_col
            phenotype columns listed in pheno_cols

    pheno_cols : iterable
        Phenotypes to analyse.

    sample_col : str
        Sample ID column in pheno_df.

    output : str or None
        Output directory for REML results.
        Default:
            <grm_dir>/reml

    n_jobs : int
        Number of REML jobs run simultaneously.

    covar_file : str or None
        Optional quantitative covariate file in PLINK/LDAK format. When
        provided, the same covariates are included as fixed effects in every
        FOREGROUND, PERM and ALL_SITE REML model.

    kinship_details : bool
        Passed to:
            --kinship-details YES/NO

        Default False, matching the original pipeline.

    overwrite : bool
        Re-run existing REML jobs.

    show_progress : bool
        Show tqdm progress.

    ldak : str
        LDAK executable.

    Returns
    -------
    pandas.DataFrame
        Summary table of all REML analyses.
    """

    if shutil.which(ldak) is None:
        raise FileNotFoundError(f"LDAK executable not found: {ldak}")

    grm_dir = os.path.abspath(os.path.expanduser(grm_dir))

    if covar_file is not None:
        covar_file = os.path.abspath(os.path.expanduser(covar_file))
        if not os.path.isfile(covar_file):
            raise FileNotFoundError(f"Covariate file not found: {covar_file}")

    if output is None:
        output = os.path.join(grm_dir, "reml")

    output = os.path.abspath(os.path.expanduser(output))
    os.makedirs(output, exist_ok=True)

    if sample_col not in pheno_df.columns:
        raise ValueError(
            f"sample_col '{sample_col}' not found in pheno_df."
        )

    missing_cols = [
        col for col in pheno_cols
        if col not in pheno_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Phenotype columns not found: {missing_cols}"
        )

    if pheno_df[sample_col].duplicated().any():
        dup = pheno_df.loc[
            pheno_df[sample_col].duplicated(),
            sample_col
        ].astype(str).tolist()

        raise ValueError(
            f"Duplicate sample IDs in pheno_df. "
            f"Examples: {dup[:10]}"
        )

    pheno_df = pheno_df.copy()
    pheno_df[sample_col] = pheno_df[sample_col].astype(str)

    # ================================================================
    # 1. Find GRM sets
    # ================================================================
    set_names = []

    for name in os.listdir(grm_dir):
        set_dir = os.path.join(grm_dir, name)

        if not os.path.isdir(set_dir):
            continue

        if name == "_training_qc" or name == "reml":
            continue

        grm_prefix = os.path.join(set_dir, "grm")

        if os.path.isfile(f"{grm_prefix}.grm.bin"):
            set_names.append(name)

    def set_sort_key(name):
        if name == "ALL_SITE":
            return (-2, 0)

        if name == "FOREGROUND":
            return (-1, 0)

        m = re.fullmatch(r"PERM_(\d+)", name)
        if m:
            return (0, int(m.group(1)))

        return (1, name)

    set_names = sorted(set_names, key=set_sort_key)

    if not set_names:
        raise ValueError(
            f"No GRMs found under {grm_dir}"
        )

    print(f"Found {len(set_names):,} GRM sets.")

    # ================================================================
    # 2. Read GRM training samples
    #
    # All GRMs were built from the same training universe, so use the
    # FOREGROUND GRM IDs as the reference.
    # ================================================================
    reference_set = (
        "FOREGROUND"
        if "FOREGROUND" in set_names
        else set_names[0]
    )

    grm_id_file = os.path.join(
        grm_dir,
        reference_set,
        "grm.grm.id",
    )

    if not os.path.isfile(grm_id_file):
        raise FileNotFoundError(grm_id_file)

    grm_ids = pd.read_csv(
        grm_id_file,
        sep=r"\s+",
        header=None,
        dtype=str,
    )

    if grm_ids.shape[1] < 2:
        raise ValueError(
            f"Unexpected GRM ID file format: {grm_id_file}"
        )

    grm_ids = grm_ids.iloc[:, :2]
    grm_ids.columns = ["FID", "IID"]

    training_iids = set(grm_ids["IID"])

    # ================================================================
    # 3. Prepare one phenotype file per phenotype
    #
    # Important:
    #   only phenotype-nonmissing samples that are present in GRM.
    # ================================================================
    pheno_info = {}

    pheno_dir = os.path.join(output, "_phenotypes")
    os.makedirs(pheno_dir, exist_ok=True)

    for pheno_col in pheno_cols:
        sub = pheno_df[
            [sample_col, pheno_col]
        ].dropna().copy()

        sub = sub[
            sub[sample_col].isin(training_iids)
        ].copy()

        if sub.empty:
            raise ValueError(
                f"No usable training samples for phenotype '{pheno_col}'."
            )

        fid_map = dict(
            zip(grm_ids["IID"], grm_ids["FID"])
        )

        sub["FID"] = sub[sample_col].map(fid_map)
        sub["IID"] = sub[sample_col]

        pheno_file = os.path.join(
            pheno_dir,
            f"{pheno_col}.pheno",
        )

        # PLINK/LDAK phenotype format:
        # FID IID phenotype
        sub[
            ["FID", "IID", pheno_col]
        ].to_csv(
            pheno_file,
            sep="\t",
            header=False,
            index=False,
        )

        pheno_info[pheno_col] = {
            "file": pheno_file,
            "n_samples": len(sub),
        }

        print(
            f"{pheno_col}: "
            f"{len(sub):,} non-missing training samples"
        )

    # ================================================================
    # 4. Construct jobs
    # ================================================================
    jobs = []

    for set_name in set_names:
        for pheno_col in pheno_cols:
            jobs.append(
                (set_name, pheno_col)
            )

    total_jobs = len(jobs)

    print(
        f"REML jobs: {len(set_names):,} GRMs "
        f"× {len(pheno_cols)} phenotypes "
        f"= {total_jobs:,}"
    )

    # ================================================================
    # 5. One REML job
    # ================================================================
    def run_one(set_name, pheno_col):
        set_dir = os.path.join(grm_dir, set_name)
        grm_prefix = os.path.join(set_dir, "grm")

        result_dir = os.path.join(
            output,
            set_name,
            pheno_col,
        )

        os.makedirs(result_dir, exist_ok=True)

        reml_prefix = os.path.join(
            result_dir,
            "model",
        )

        reml_file = f"{reml_prefix}.reml"

        if not overwrite and _have_reml(reml_prefix):
            status = "skipped"

        else:
            cmd = [
                ldak,
                "--reml", reml_prefix,
                "--grm", grm_prefix,
                "--pheno", pheno_info[pheno_col]["file"],
                "--kinship-details",
                "YES" if kinship_details else "NO",
            ]

            if covar_file is not None:
                cmd += ["--covar", covar_file]

            _run_cmd(
                cmd,
                quiet=(n_jobs > 1),
            )

            if not _have_reml(reml_prefix):
                raise RuntimeError(
                    f"{set_name} / {pheno_col}: "
                    "LDAK did not produce .reml output."
                )

            status = "completed"

        parsed = _parse_ldak_reml(reml_file)

        result = {
            "set": set_name,
            "phenotype": pheno_col,
            "n_samples": pheno_info[pheno_col]["n_samples"],
            "grm_prefix": grm_prefix,
            "reml_prefix": reml_prefix,
            "reml_file": reml_file,
            "status": status,
        }

        result.update(parsed)

        return result

    # ================================================================
    # 6. Run serially or in parallel
    # ================================================================
    results = []

    if n_jobs == 1:
        iterator = jobs

        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_jobs,
                desc="Running REML",
            )

        for set_name, pheno_col in iterator:
            result = run_one(
                set_name,
                pheno_col,
            )

            results.append(result)

    else:
        pbar = tqdm(
            total=total_jobs,
            desc="Running REML",
            disable=not show_progress,
        )

        with ThreadPoolExecutor(
            max_workers=n_jobs
        ) as executor:

            futures = {
                executor.submit(
                    run_one,
                    set_name,
                    pheno_col,
                ): (set_name, pheno_col)
                for set_name, pheno_col in jobs
            }

            for future in as_completed(futures):
                set_name, pheno_col = futures[future]

                try:
                    result = future.result()

                except Exception as e:
                    pbar.close()

                    raise RuntimeError(
                        f"REML failed for "
                        f"{set_name} / {pheno_col}"
                    ) from e

                results.append(result)
                pbar.update(1)

        pbar.close()

    # ================================================================
    # 7. Summary table
    # ================================================================
    summary_df = pd.DataFrame(results)

    summary_df["perm_index"] = summary_df["set"].str.extract(
        r"PERM_(\d+)"
    )[0].astype("Int64")

    summary_df["is_foreground"] = (
        summary_df["set"] == "FOREGROUND"
    )

    summary_df["is_all_site"] = (
        summary_df["set"] == "ALL_SITE"
    )

    summary_df = summary_df.sort_values(
        ["phenotype", "is_all_site", "is_foreground", "perm_index"],
        ascending=[True, False, False, True],
        na_position="first",
    ).reset_index(drop=True)

    summary_file = os.path.join(
        output,
        "reml_summary.csv",
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    metadata = {
        "grm_dir": grm_dir,
        "output": output,
        "n_grms": len(set_names),
        "phenotypes": list(pheno_cols),
        "n_jobs": n_jobs,
        "covar_file": covar_file,
        "kinship_details": kinship_details,
        "summary_file": summary_file,
    }

    with open(
        os.path.join(output, "metadata.json"),
        "w",
    ) as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Finished REML: "
        f"{len(summary_df):,} analyses."
    )
    print(
        f"Summary: {summary_file}"
    )

    return summary_df


def _have_bfile(prefix):
    return all(os.path.isfile(f"{prefix}.{ext}") for ext in ("bed", "bim", "fam"))


def _have_grm(prefix):
    files = [
        f"{prefix}.grm.bin",
        f"{prefix}.grm.id",
    ]
    return all(os.path.isfile(f) and os.path.getsize(f) > 0 for f in files)


def _ensure_master_bfile(vcf_file, plink="plink"):
    """Create persistent PLINK BED/BIM/FAM beside VCF if necessary."""
    vcf_file = os.path.abspath(vcf_file)

    if vcf_file.endswith(".vcf.gz"):
        base = vcf_file[:-7]
    elif vcf_file.endswith(".vcf"):
        base = vcf_file[:-4]
    else:
        raise ValueError("vcf_file must end with '.vcf' or '.vcf.gz'.")

    bfile = base + ".plink"

    if _have_bfile(bfile):
        print(f"Using existing master PLINK dataset: {bfile}")
        return bfile

    print(f"Creating master PLINK dataset: {bfile}")
    _run_cmd([plink, "--vcf", vcf_file, "--make-bed", "--out", bfile])

    if not _have_bfile(bfile):
        raise RuntimeError(f"Failed to create master PLINK dataset: {bfile}")

    return bfile


def _sample_signature(sample_pairs, master_bfile, maf, geno):
    """Signature used to determine whether training QC can be reused."""
    text = "\n".join(
        [os.path.abspath(master_bfile), f"maf={maf}", f"geno={geno}"]
        + [f"{fid}\t{iid}" for fid, iid in sorted(sample_pairs)]
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _read_ldak_pca_eigenvalues(values_file):
    """Read eigenvalues from an LDAK PCA .values file."""
    rows = []

    with open(values_file) as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue

            numeric = []
            for token in tokens:
                try:
                    numeric.append(float(token))
                except ValueError:
                    pass

            if numeric:
                rows.append(numeric)

    if not rows:
        raise ValueError(f"No eigenvalues found in {values_file}")

    if len(rows) == 1:
        return np.asarray(rows[0], dtype=float)

    if all(len(row) == 1 for row in rows):
        return np.asarray([row[0] for row in rows], dtype=float)

    # Some LDAK versions may include an axis/index column before the
    # eigenvalue; in that case use the final numeric field on each row.
    return np.asarray([row[-1] for row in rows], dtype=float)


def _select_pca_elbow(eigenvalues):
    """Select the scree elbow by maximum distance from the endpoint line."""
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError("No finite PCA eigenvalues are available for auto selection.")
    if len(values) <= 2:
        return 1, np.zeros(len(values), dtype=float)

    # Suppress tiny numerical upward fluctuations without changing the
    # overall scree shape.
    values = np.minimum.accumulate(values)

    x = np.linspace(0.0, 1.0, len(values))
    value_range = values[0] - values[-1]

    if not np.isfinite(value_range) or abs(value_range) < 1e-15:
        return 1, np.zeros(len(values), dtype=float)

    y = (values - values[-1]) / value_range

    # Perpendicular distance to the straight line connecting the first and
    # last scree points. The point with maximum distance is the elbow.
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    distances = np.abs(
        (y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1
    ) / denom

    distances[0] = -np.inf
    distances[-1] = -np.inf
    selected = int(np.argmax(distances)) + 1
    distances[~np.isfinite(distances)] = 0.0

    return max(selected, 1), distances


def _write_pca_covariates(source_file, output_file, n_pcs):
    """Write a covariate file containing only the first n_pcs PCA axes."""
    n_cols = 2 + int(n_pcs)
    n_rows = 0

    with open(source_file) as src, open(output_file, "w") as dst:
        for line in src:
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) < n_cols:
                raise ValueError(
                    f"PCA covariate file {source_file} has only {len(tokens)} "
                    f"columns; expected at least {n_cols}."
                )
            dst.write("\t".join(tokens[:n_cols]) + "\n")
            n_rows += 1

    if n_rows == 0:
        raise ValueError(f"PCA covariate file is empty: {source_file}")


def build_pca_covariates(
    pruned_vcf_file,
    training_sample_ids,
    output,
    n_pcs=5,
    auto_max_pcs=20,
    power=-1,
    overwrite=False,
    plink="plink",
    ldak="ldak",
):
    """Build fixed population-structure covariates from an LD-pruned VCF.

    ``n_pcs`` can be a positive integer or ``"auto"``. In auto mode, LDAK
    calculates up to ``auto_max_pcs`` leading axes, then the number of PCs is
    selected from the elbow of the eigenvalue scree curve. No phenotype or
    environmental variable is used in this selection.
    """
    if shutil.which(plink) is None:
        raise FileNotFoundError(f"PLINK executable not found: {plink}")
    if shutil.which(ldak) is None:
        raise FileNotFoundError(f"LDAK executable not found: {ldak}")

    if isinstance(n_pcs, str):
        if n_pcs.lower() != "auto":
            raise ValueError("n_pcs must be a positive integer or 'auto'.")
        requested_n_pcs = "auto"
    else:
        n_pcs = int(n_pcs)
        if n_pcs < 1:
            raise ValueError("n_pcs must be >= 1.")
        requested_n_pcs = n_pcs

    auto_max_pcs = int(auto_max_pcs)
    if auto_max_pcs < 3:
        raise ValueError("auto_max_pcs must be >= 3.")

    pruned_vcf_file = os.path.abspath(os.path.expanduser(pruned_vcf_file))
    if not os.path.isfile(pruned_vcf_file):
        raise FileNotFoundError(pruned_vcf_file)

    output = os.path.abspath(os.path.expanduser(output))
    os.makedirs(output, exist_ok=True)

    master_bfile = _ensure_master_bfile(pruned_vcf_file, plink=plink)
    fam = pd.read_csv(
        f"{master_bfile}.fam",
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        dtype=str,
    )
    fam.columns = ["FID", "IID"]

    training_sample_ids = list(dict.fromkeys(str(x) for x in training_sample_ids))
    if not training_sample_ids:
        raise ValueError("training_sample_ids is empty.")

    max_sample_pcs = len(training_sample_ids) - 1
    if max_sample_pcs < 1:
        raise ValueError("At least two training samples are required for PCA.")

    iid_counts = fam["IID"].value_counts()
    duplicated_iids = set(iid_counts[iid_counts > 1].index)
    ambiguous = [x for x in training_sample_ids if x in duplicated_iids]
    if ambiguous:
        raise ValueError(
            f"{len(ambiguous)} training IID(s) occur under multiple FIDs in "
            f"the pruned VCF. Examples: {ambiguous[:10]}"
        )

    fam_by_iid = fam.set_index("IID")
    missing_samples = [x for x in training_sample_ids if x not in fam_by_iid.index]
    if missing_samples:
        raise ValueError(
            f"{len(missing_samples):,} training samples are absent from the "
            f"pruned VCF. Examples: {missing_samples[:10]}"
        )

    sample_pairs = [(fam_by_iid.loc[iid, "FID"], iid) for iid in training_sample_ids]
    keep_file = os.path.join(output, "samples.keep")
    training_bfile = os.path.join(output, "genotype")
    kinship_prefix = os.path.join(output, "kinship")
    pca_prefix = os.path.join(output, "pca")
    source_covar_file = f"{pca_prefix}.vect"
    alternate_source_covar_file = f"{pca_prefix}.vector"
    values_file = f"{pca_prefix}.values"
    covar_file = os.path.join(output, "population_pcs.covar")
    selection_file = os.path.join(output, "pc_selection.csv")
    metadata_file = os.path.join(output, "metadata.json")

    stat = os.stat(pruned_vcf_file)
    signature_text = "\n".join(
        [
            pruned_vcf_file,
            f"size={stat.st_size}",
            f"mtime_ns={stat.st_mtime_ns}",
            f"requested_n_pcs={requested_n_pcs}",
            f"auto_max_pcs={auto_max_pcs}",
            f"power={power}",
        ]
        + [f"{fid}\t{iid}" for fid, iid in sorted(sample_pairs)]
    )
    signature = hashlib.sha256(signature_text.encode()).hexdigest()

    if not overwrite and os.path.isfile(metadata_file):
        try:
            with open(metadata_file) as f:
                old_meta = json.load(f)
            old_covar = old_meta.get("covar_file")
            if (
                old_meta.get("signature") == signature
                and old_covar
                and os.path.isfile(old_covar)
            ):
                print(
                    f"Using existing population covariates: {old_covar} "
                    f"({old_meta.get('n_pcs')} PCs)"
                )
                old_meta["rebuilt"] = False
                return old_meta
        except Exception:
            pass

    with open(keep_file, "w") as f:
        for fid, iid in sample_pairs:
            f.write(f"{fid}\t{iid}\n")

    # Reuse genotype/kinship when possible. Their content does not depend on
    # the requested number of PCs.
    if _have_bfile(training_bfile) and not overwrite:
        print(f"Using existing LD-pruned training genotype: {training_bfile}")
    else:
        print("Building LD-pruned training genotype for population PCA...")
        _run_cmd([
            plink,
            "--bfile", master_bfile,
            "--keep", keep_file,
            "--make-bed",
            "--out", training_bfile,
        ])

        if not _have_bfile(training_bfile):
            raise RuntimeError("Failed to create LD-pruned training genotype.")

    with open(f"{training_bfile}.bim") as f:
        n_variants = sum(1 for _ in f)
    with open(f"{training_bfile}.fam") as f:
        n_samples = sum(1 for _ in f)

    max_available_pcs = min(max_sample_pcs, n_variants)
    if max_available_pcs < 1:
        raise ValueError("No PCA axes can be calculated from the pruned genotype.")

    if requested_n_pcs == "auto":
        axes_to_calculate = min(auto_max_pcs, max_available_pcs)
    else:
        if requested_n_pcs > max_available_pcs:
            raise ValueError(
                f"n_pcs={requested_n_pcs} exceeds the available PCA rank "
                f"({max_available_pcs})."
            )
        axes_to_calculate = requested_n_pcs

    if _have_grm(kinship_prefix) and not overwrite:
        print(f"Using existing LD-pruned population kinship: {kinship_prefix}")
    else:
        print("Building LD-pruned kinship for population PCA...")
        _run_cmd([
            ldak,
            "--calc-kins-direct", kinship_prefix,
            "--bfile", training_bfile,
            "--power", str(power),
            "--ignore-weights", "YES",
            "--allow-multi", "YES",
            "--kinship-raw", "YES",
        ])

        if not _have_grm(kinship_prefix):
            raise RuntimeError("Failed to create LD-pruned population kinship.")

    if requested_n_pcs == "auto":
        print(
            f"Calculating top {axes_to_calculate} population PCs for "
            "automatic scree-elbow selection..."
        )
    else:
        print(f"Calculating top {axes_to_calculate} population PCs...")

    _run_cmd([
        ldak,
        "--pca", pca_prefix,
        "--grm", kinship_prefix,
        "--axes", str(axes_to_calculate),
    ])

    if not os.path.isfile(source_covar_file):
        if os.path.isfile(alternate_source_covar_file):
            source_covar_file = alternate_source_covar_file
        else:
            raise RuntimeError("LDAK PCA did not produce a .vect/.vector file.")

    if not os.path.isfile(values_file):
        raise RuntimeError("LDAK PCA did not produce a .values file.")

    eigenvalues = _read_ldak_pca_eigenvalues(values_file)
    if len(eigenvalues) < axes_to_calculate:
        raise RuntimeError(
            f"Expected {axes_to_calculate} PCA eigenvalues, but found "
            f"{len(eigenvalues)} in {values_file}."
        )
    eigenvalues = eigenvalues[:axes_to_calculate]

    if requested_n_pcs == "auto":
        selected_n_pcs, elbow_distances = _select_pca_elbow(eigenvalues)
        selection_method = "scree_elbow_max_distance"
        print(
            f"Automatically selected {selected_n_pcs} population PCs "
            f"from the first {axes_to_calculate} axes."
        )
    else:
        selected_n_pcs = requested_n_pcs
        elbow_distances = np.full(len(eigenvalues), np.nan)
        selection_method = "manual"

    _write_pca_covariates(
        source_file=source_covar_file,
        output_file=covar_file,
        n_pcs=selected_n_pcs,
    )

    positive_eigenvalues = np.clip(eigenvalues, 0, None)
    positive_sum = positive_eigenvalues.sum()
    if positive_sum > 0:
        proportions = positive_eigenvalues / positive_sum
        cumulative = np.cumsum(proportions)
    else:
        proportions = np.full(len(eigenvalues), np.nan)
        cumulative = np.full(len(eigenvalues), np.nan)

    selection_df = pd.DataFrame({
        "pc": np.arange(1, len(eigenvalues) + 1),
        "eigenvalue": eigenvalues,
        "eigenvalue_proportion_within_calculated_axes": proportions,
        "cumulative_proportion_within_calculated_axes": cumulative,
        "elbow_distance": elbow_distances,
    })
    selection_df["selected"] = selection_df["pc"] <= selected_n_pcs
    selection_df.to_csv(selection_file, index=False)

    metadata = {
        "signature": signature,
        "pruned_vcf_file": pruned_vcf_file,
        "master_bfile": master_bfile,
        "training_bfile": training_bfile,
        "kinship_prefix": kinship_prefix,
        "pca_prefix": pca_prefix,
        "pca_values_file": values_file,
        "source_covar_file": source_covar_file,
        "covar_file": covar_file,
        "selection_file": selection_file,
        "requested_n_pcs": requested_n_pcs,
        "n_pcs": int(selected_n_pcs),
        "pca_axes_calculated": int(axes_to_calculate),
        "auto_max_pcs": int(auto_max_pcs),
        "selection_method": selection_method,
        "n_training_samples": int(n_samples),
        "n_pruned_variants": int(n_variants),
        "power": power,
        "rebuilt": True,
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

def build_grms(
    context_df,
    vcf_file,
    training_sample_ids,
    variant_sets,
    output,
    maf=None,
    geno=None,
    power=-1,
    ignore_weights=True,
    allow_multi=True,
    n_jobs=1,
    overwrite=False,
    show_progress=True,
    plink="plink",
    ldak="ldak",
):
    """
    Build GRMs for FOREGROUND and all matched permutation variant sets.

    Workflow
    --------
    1. Convert VCF -> persistent master PLINK dataset once.
    2. Apply training-sample filtering + MAF/GENO QC once.
    3. Load FOREGROUND/PERM positional indices from build_variant_sets().
    4. Intersect each variant set with variants surviving training QC.
    5. Build one LDAK GRM per variant set using --extract.

    Parameters
    ----------
    context_df : pandas.DataFrame
        Exactly the same dataframe/order used by build_variant_sets().
        Must contain unique column "ID".

    vcf_file : str
        VCF/VCF.GZ containing all required samples and variants.
        Persistent <vcf_prefix>.plink.* files are automatically created
        beside the VCF and reused on future runs.

    training_sample_ids : iterable
        Unified landrace training universe. Do NOT include external
        GxE prediction samples.

    variant_sets : str
        Prefix generated by build_variant_sets(), e.g.:
            /work/ACRs/variant_sets
        Requires:
            <prefix>.foreground.npy
            <prefix>.permutations.npy

    output : str
        GRM output directory.

    maf, geno : float
        PLINK QC thresholds applied ONCE to the unified training samples.

    power : float
        LDAK --power value.

    ignore_weights : bool
        Add '--ignore-weights YES' when True.

    allow_multi : bool
        Add '--allow-multi YES' when True.

    n_jobs : int
        Number of variant sets processed simultaneously.
        Start conservatively (e.g. 4-8) because all jobs read the same BED.

    overwrite : bool
        Rebuild existing GRMs. Does not rebuild master PLINK or an
        identical training-QC dataset.

    Returns
    -------
    str
        Absolute path to output directory.
    """

    if shutil.which(plink) is None:
        raise FileNotFoundError(f"PLINK executable not found: {plink}")
    if shutil.which(ldak) is None:
        raise FileNotFoundError(f"LDAK executable not found: {ldak}")
    if "ID" not in context_df.columns:
        raise ValueError("context_df must contain column 'ID'.")
    if context_df["ID"].duplicated().any():
        raise ValueError("context_df['ID'] must contain unique variant IDs.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    output = os.path.abspath(output)
    variant_sets = os.path.abspath(variant_sets)
    os.makedirs(output, exist_ok=True)

    # ================================================================
    # 1. VCF -> persistent master PLINK
    # ================================================================
    master_bfile = _ensure_master_bfile(vcf_file, plink=plink)

    # ================================================================
    # 2. Resolve training samples against master FAM
    # ================================================================
    training_sample_ids = list(dict.fromkeys(str(x)
                               for x in training_sample_ids))

    if not training_sample_ids:
        raise ValueError("training_sample_ids is empty.")

    fam = pd.read_csv(
        f"{master_bfile}.fam",
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        dtype=str,
    )
    fam.columns = ["FID", "IID"]

    iid_counts = fam["IID"].value_counts()
    duplicated_iids = set(iid_counts[iid_counts > 1].index)

    ambiguous = [x for x in training_sample_ids if x in duplicated_iids]
    if ambiguous:
        raise ValueError(
            f"{len(ambiguous)} requested IID(s) occur under multiple FIDs. "
            f"Examples: {ambiguous[:10]}"
        )

    fam_by_iid = fam.set_index("IID")
    missing_samples = [
        x for x in training_sample_ids if x not in fam_by_iid.index]

    if missing_samples:
        raise ValueError(
            f"{len(missing_samples):,} training samples are absent from master PLINK. "
            f"Examples: {missing_samples[:10]}"
        )

    sample_pairs = [
        (fam_by_iid.loc[iid, "FID"], iid)
        for iid in training_sample_ids
    ]

    # ================================================================
    # 3. Build/reuse ONE unified training-QC genotype dataset
    # ================================================================
    qc_dir = os.path.join(output, "_training_qc")
    os.makedirs(qc_dir, exist_ok=True)

    keep_file = os.path.join(qc_dir, "samples.keep")
    training_bfile = os.path.join(qc_dir, "genotype")
    qc_metadata_file = os.path.join(qc_dir, "metadata.json")

    with open(keep_file, "w") as f:
        for fid, iid in sample_pairs:
            f.write(f"{fid}\t{iid}\n")

    signature = _sample_signature(sample_pairs, master_bfile, maf, geno)
    reuse_training_qc = False

    if _have_bfile(training_bfile) and os.path.isfile(qc_metadata_file):
        try:
            with open(qc_metadata_file) as f:
                old_meta = json.load(f)
            reuse_training_qc = old_meta.get("signature") == signature
        except Exception:
            reuse_training_qc = False

    if reuse_training_qc:
        print(f"Using existing training QC dataset: {training_bfile}")
    else:
        print("Building unified training QC dataset...")

        cmd = [
            plink,
            "--bfile", master_bfile,
            "--keep", keep_file,
        ]

        if maf is not None:
            cmd += ["--maf", str(maf)]

        if geno is not None:
            cmd += ["--geno", str(geno)]

        cmd += [
            "--make-bed",
            "--out", training_bfile,
        ]

        _run_cmd(cmd)

        if not _have_bfile(training_bfile):
            raise RuntimeError("Failed to create unified training QC dataset.")

        with open(f"{training_bfile}.bim") as f:
            n_qc_variants = sum(1 for _ in f)

        with open(f"{training_bfile}.fam") as f:
            n_qc_samples = sum(1 for _ in f)

        qc_meta = {
            "signature": signature,
            "master_bfile": master_bfile,
            "n_training_samples": n_qc_samples,
            "n_qc_variants": n_qc_variants,
            "maf": maf,
            "geno": geno,
        }

        with open(qc_metadata_file, "w") as f:
            json.dump(qc_meta, f, indent=2)

    # ================================================================
    # 4. Load variant-set matrices
    # ================================================================
    fg_file = f"{variant_sets}.foreground.npy"
    perm_file = f"{variant_sets}.permutations.npy"

    if not os.path.isfile(fg_file):
        raise FileNotFoundError(fg_file)
    if not os.path.isfile(perm_file):
        raise FileNotFoundError(perm_file)

    foreground_idx = np.load(fg_file, mmap_mode="r")
    perm_mat = np.load(perm_file, mmap_mode="r")

    if foreground_idx.ndim != 1:
        raise ValueError("foreground.npy must be one-dimensional.")
    if perm_mat.ndim != 2:
        raise ValueError("permutations.npy must be two-dimensional.")
    if perm_mat.shape[1] != len(foreground_idx):
        raise ValueError(
            f"Foreground contains {len(foreground_idx):,} variants but "
            f"permutations contain {perm_mat.shape[1]:,} each."
        )

    n_perm = perm_mat.shape[0]
    n_context = len(context_df)

    min_idx = min(int(foreground_idx.min()), int(perm_mat.min()))
    max_idx = max(int(foreground_idx.max()), int(perm_mat.max()))

    if min_idx < 0 or max_idx >= n_context:
        raise ValueError(
            "Variant-set indices are incompatible with context_df. "
            "Make sure this is exactly the same context_df/order used by "
            "build_variant_sets()."
        )

    site_ids = context_df["ID"].astype(str).to_numpy()

    # ================================================================
    # 5. Determine which context variants survived unified QC
    #    Done once, then every PERM can be filtered by integer indexing.
    # ================================================================
    print("Mapping training-QC variants back to context_df...")

    qc_ids = pd.read_csv(
        f"{training_bfile}.bim",
        sep=r"\s+",
        header=None,
        usecols=[1],
        dtype=str,
    )[1]

    qc_pass_mask = context_df["ID"].astype(str).isin(qc_ids).to_numpy()
    del qc_ids

    # ================================================================
    # 6. Global metadata
    # ================================================================
    global_metadata = {
        "vcf_file": os.path.abspath(vcf_file),
        "master_bfile": master_bfile,
        "training_bfile": training_bfile,
        "variant_sets": variant_sets,
        "n_context_sites": n_context,
        "n_training_samples": len(training_sample_ids),
        "n_foreground_input": int(len(foreground_idx)),
        "n_permutations": int(n_perm),
        "maf": maf,
        "geno": geno,
        "power": power,
        "ignore_weights": ignore_weights,
        "allow_multi": allow_multi,
    }

    with open(os.path.join(output, "metadata.json"), "w") as f:
        json.dump(global_metadata, f, indent=2)

    # ================================================================
    # 7. One GRM
    # ================================================================
    def build_one(tag, indices):
        job_dir = os.path.join(output, tag)
        os.makedirs(job_dir, exist_ok=True)

        grm_prefix = os.path.join(job_dir, "grm")
        variant_file = os.path.join(job_dir, "variants.txt")
        metadata_file = os.path.join(job_dir, "metadata.json")

        if not overwrite and _have_grm(grm_prefix):
            return tag, "skipped"

        indices = np.asarray(indices, dtype=np.int64)

        # Keep only variants that survived unified training QC.
        qc_indices = indices[qc_pass_mask[indices]]

        if len(qc_indices) == 0:
            raise RuntimeError(f"{tag}: no variants remain after unified QC.")

        variant_ids = site_ids[qc_indices]

        # Keep this file permanently: REML/BLUP stages can reuse it.
        with open(variant_file, "w") as f:
            f.write("\n".join(variant_ids))
            f.write("\n")

        cmd = [
            ldak,
            "--calc-kins-direct", grm_prefix,
            "--bfile", training_bfile,
            "--extract", variant_file,
            "--power", str(power),
        ]

        if ignore_weights:
            cmd += ["--ignore-weights", "YES"]

        if allow_multi:
            cmd += ["--allow-multi", "YES"]

        _run_cmd(cmd, quiet=(n_jobs > 1))

        if not _have_grm(grm_prefix):
            raise RuntimeError(f"{tag}: incomplete GRM output.")

        metadata = {
            "tag": tag,
            "n_input_variants": int(len(indices)),
            "n_qc_variants": int(len(qc_indices)),
            "n_removed_by_qc": int(len(indices) - len(qc_indices)),
            "variant_file": variant_file,
            "grm_prefix": grm_prefix,
            "completed": True,
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return tag, "completed"

    # ================================================================
    # 8. ALL_SITE + FOREGROUND + PERM_0 ... PERM_n
    # ================================================================
    def iter_jobs():
        # ALL_SITE starts from every site in context_df, then build_one()
        # applies the same unified training-sample MAF/GENO QC as all other sets.
        yield "ALL_SITE", np.arange(n_context, dtype=np.int64)
        yield "FOREGROUND", foreground_idx
        for i in range(n_perm):
            yield f"PERM_{i}", perm_mat[i]

    total_jobs = n_perm + 2
    completed = 0
    skipped = 0

    if n_jobs == 1:
        jobs = iter_jobs()

        if show_progress:
            jobs = tqdm(jobs, total=total_jobs, desc="Building GRMs")

        for tag, indices in jobs:
            _, status = build_one(tag, indices)

            if status == "completed":
                completed += 1
            else:
                skipped += 1

            if show_progress:
                jobs.set_postfix(completed=completed, skipped=skipped)

    else:
        # Submit only lightweight views of the mmap arrays.
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(build_one, tag, indices): tag
                for tag, indices in iter_jobs()
            }

            iterator = as_completed(futures)

            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=total_jobs,
                    desc="Building GRMs",
                )

            for future in iterator:
                tag = futures[future]

                try:
                    _, status = future.result()
                except Exception as e:
                    raise RuntimeError(
                        f"GRM construction failed for {tag}") from e

                if status == "completed":
                    completed += 1
                else:
                    skipped += 1

                if show_progress:
                    iterator.set_postfix(
                        completed=completed,
                        skipped=skipped,
                    )

    print(
        f"Finished GRM construction: "
        f"{completed} built, {skipped} skipped, {total_jobs} total."
    )

    return output


def build_variant_sets(
    site_df,
    foreground="foreground",
    match_controls=("MAF", "Recomb", "Pi", "Gene"),
    n_perm=1000,
    seed=42,
    unique_within_set=True,
    allow_fallback=True,
    store="index",
    output="variant_sets",
    output_format="npy",
    show_progress=True,
    sample_from_all_sites=False,
    target_definition="original_foreground",
):
    """
    Build matched control variant sets by stratified sampling.

    All sites in `site_df` are treated as the eligible pool.
    Sites with all-False bins within one matching dimension are assigned
    to an additional "unclassified" category. Overlapping bins (>1 True)
    are treated as errors.

    Parameters
    ----------
    store : {"index", "id"}
        Content stored in the result.
        "index" uses 0-based positional indices of site_df.
        "id" uses site_df["ID"].
    output : str
        Base name for the output files.
    output_format : {"json", "npy"}
        "json":
            Save everything into `output`, optionally .json.gz.
        "npy":
            Save:
                {output}.foreground.npy
                {output}.permutations.npy
                {output}.metadata.json
            NPY output requires store="index".
    """
    match_definitions = {
        "MAF": ["maf_0.05_0.1", "maf_0.1_0.2", "maf_0.2_0.3", "maf_0.3_0.4", "maf_0.4_0.5"],
        "Recomb": ["low_recombination", "med_recombination", "high_recombination"],
        "Pi": ["low_pi_genome", "med_pi_genome", "high_pi_genome"],
        "Gene": ["genic_2kb", "non_genic"],
    }

    match_controls = tuple(match_controls)

    if not isinstance(sample_from_all_sites, (bool, np.bool_)):
        raise TypeError("sample_from_all_sites must be boolean.")
    n_perm = int(n_perm)
    seed = int(seed)
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1.")

    if store not in {"id", "index"}:
        raise ValueError("store must be 'id' or 'index'.")
    if output_format not in {"json", "npy"}:
        raise ValueError("output_format must be 'json' or 'npy'.")
    if output_format == "npy" and store != "index":
        raise ValueError(
            "output_format='npy' currently requires store='index'.")
    if "ID" not in site_df.columns:
        raise ValueError("site_df must contain an 'ID' column.")
    if site_df["ID"].duplicated().any():
        raise ValueError(
            f"site_df contains {site_df['ID'].duplicated().sum():,} duplicated IDs.")

    unknown = set(match_controls) - set(match_definitions)
    if unknown:
        raise ValueError(f"Unsupported match_controls: {sorted(unknown)}")

    required_cols = [c for m in match_controls for c in match_definitions[m]]
    missing = [c for c in required_cols if c not in site_df.columns]
    if missing:
        raise ValueError(f"Missing matching columns: {missing}")

    if isinstance(foreground, str):
        if foreground not in site_df.columns:
            raise ValueError(f"Foreground column '{foreground}' not found.")
        fg = site_df[foreground].fillna(False).to_numpy(bool)
        foreground_name = foreground
    else:
        fg = np.asarray(foreground, dtype=bool)
        if len(fg) != len(site_df):
            raise ValueError("Foreground mask length does not match site_df.")
        foreground_name = "<boolean mask>"

    n_sites = len(site_df)
    n_fg = int(fg.sum())
    if n_fg == 0:
        raise ValueError("No foreground sites found.")
    if n_fg == n_sites:
        raise ValueError("No background sites available.")
    if n_sites > np.iinfo(np.int32).max:
        raise ValueError("site_df is too large for int32 indices.")

    n_bg = n_sites - n_fg
    if not sample_from_all_sites and unique_within_set and n_fg > n_bg:
        raise ValueError(
            "Cannot build a unique background-only permutation set: "
            f"effective foreground has {n_fg:,} sites but its complement "
            f"has only {n_bg:,}. Use run_partition_contrast(..., "
            "use_complement=True) to study the smaller complement, or "
            "sample_from_all_sites=True to sample controls from all sites."
        )

    requested_config = {
        "n_total_sites": int(n_sites),
        "n_foreground": int(n_fg),
        "match_controls": list(match_controls),
        "n_permutations": int(n_perm),
        "seed": int(seed),
        "unique_within_set": bool(unique_within_set),
        "allow_fallback": bool(allow_fallback),
        "sample_from_all_sites": bool(sample_from_all_sites),
        "target_definition": str(target_definition),
    }

    # Never silently reuse arrays generated under a different sampling
    # configuration. run_partition_contrast(overwrite=True) removes these
    # files before reaching this function.
    if output is not None:
        if output_format == "npy":
            output_files = [
                f"{output}.foreground.npy",
                f"{output}.permutations.npy",
                f"{output}.metadata.json",
            ]
            existing = [os.path.isfile(path) for path in output_files]
            if any(existing):
                if not all(existing):
                    raise ValueError(
                        f"Incomplete existing variant-set output for {output}. "
                        "Use overwrite=True in run_partition_contrast or a "
                        "different results_dir."
                    )
                with open(output_files[2]) as f:
                    old_metadata = json.load(f)
                mismatched = [
                    key for key, value in requested_config.items()
                    if old_metadata.get(key) != value
                ]
                if mismatched:
                    raise ValueError(
                        f"Existing variant sets at {output} use a different "
                        f"configuration ({', '.join(mismatched)}). Use "
                        "overwrite=True in run_partition_contrast or a "
                        "different results_dir."
                    )
                print(f"Compatible existing output found, skipping: {output}")
                return
        elif output_format == "json" and os.path.isfile(output):
            opener = gzip.open if str(output).endswith(".gz") else open
            with opener(output, "rt", encoding="utf-8") as f:
                old_result = json.load(f)
            old_metadata = old_result.get("metadata", {})
            mismatched = [
                key for key, value in requested_config.items()
                if old_metadata.get(key) != value
            ]
            if mismatched:
                raise ValueError(
                    f"Existing variant sets at {output} use a different "
                    f"configuration ({', '.join(mismatched)}). Choose a "
                    "different output path."
                )
            print(f"Compatible existing output found, skipping: {output}")
            return

    site_ids = site_df["ID"].astype(str).to_numpy()

    # Encode matching dimensions: normal bins = 0..n-1; all-False = n.
    group_codes = []
    category_names = {}
    unclassified_counts = {}
    overlap_counts = {}

    for name in match_controls:
        cols = match_definitions[name]
        mat = site_df[cols].fillna(False).to_numpy(bool)
        n_true = mat.sum(axis=1)

        normal = n_true == 1
        unclassified = n_true == 0
        overlap = n_true > 1

        unclassified_counts[name] = int(unclassified.sum())
        overlap_counts[name] = int(overlap.sum())

        if overlap.any():
            raise ValueError(
                f"{name}: {overlap.sum():,} sites belong to >1 bin "
                f"({(overlap & fg).sum():,} foreground)."
            )

        code = np.empty(n_sites, dtype=np.int16)
        code[normal] = np.argmax(mat[normal], axis=1)
        code[unclassified] = len(cols)
        group_codes.append(code)
        category_names[name] = cols + ["unclassified"]

    if match_controls:
        shape = tuple(
            len(match_definitions[name]) + 1 for name in match_controls)
        stratum_code = np.ravel_multi_index(
            tuple(group_codes), dims=shape).astype(np.int32)
    else:
        shape = (1,)
        stratum_code = np.zeros(n_sites, dtype=np.int32)

    # Precompute candidates in each foreground-containing stratum. The default
    # pool is the effective background; all-sites mode deliberately permits
    # foreground/control overlap under a random-set null.
    candidate_pool_mask = (
        np.ones(n_sites, dtype=bool)
        if sample_from_all_sites else ~fg
    )
    strata = []
    for s in np.unique(stratum_code[fg]):
        mask = stratum_code == s
        fg_idx = np.flatnonzero(mask & fg)
        candidate_idx = np.flatnonzero(mask & candidate_pool_mask)
        strata.append({
            "stratum": int(s),
            "need": len(fg_idx),
            "available": len(candidate_idx),
            "candidate_idx": candidate_idx,
        })

    insufficient = [
        {
            "stratum": x["stratum"],
            "need": x["need"],
            "available": x["available"],
            "shortfall": x["need"] - x["available"],
        }
        for x in strata if x["available"] < x["need"]
    ]

    if insufficient and not allow_fallback:
        total_shortfall = sum(x["shortfall"] for x in insufficient)
        preview = "\n".join(
            f"  stratum={x['stratum']}: need={x['need']}, "
            f"available={x['available']}, shortfall={x['shortfall']}"
            for x in insufficient[:20]
        )
        raise ValueError(
            f"Strict matched sampling is impossible.\n"
            f"Foreground sites: {n_fg:,}\n"
            f"Insufficient strata: {len(insufficient):,}\n"
            f"Total shortfall: {total_shortfall:,}\n{preview}"
        )

    all_candidate_idx = np.flatnonzero(candidate_pool_mask)
    foreground_idx = np.flatnonzero(fg).astype(np.int32)
    perm_mat = np.empty((n_perm, n_fg), dtype=np.int32)
    fallback_counts = np.zeros(n_perm, dtype=np.int32)
    foreground_overlap_counts = np.zeros(n_perm, dtype=np.int32)

    iterator = range(n_perm)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating matched variant sets")

    for p in iterator:
        rng = np.random.default_rng(seed + p)
        selected_parts = []
        selected_mask = np.zeros(
            n_sites, dtype=bool) if unique_within_set else None
        fallback_needed = 0

        for x in strata:
            candidates = x["candidate_idx"]
            if unique_within_set:
                candidates = candidates[~selected_mask[candidates]]

            n_take = min(x["need"], len(candidates))
            if n_take:
                picked = rng.choice(candidates, size=n_take, replace=False)
                selected_parts.append(picked)
                if unique_within_set:
                    selected_mask[picked] = True

            fallback_needed += x["need"] - n_take

        if fallback_needed:
            if not allow_fallback:
                raise RuntimeError(
                    f"PERM_{p}: {fallback_needed:,} sites could not be matched.")

            fallback_pool = all_candidate_idx
            if unique_within_set:
                fallback_pool = fallback_pool[~selected_mask[fallback_pool]]

            if len(fallback_pool) < fallback_needed:
                raise ValueError(
                    f"PERM_{p}: fallback pool too small "
                    f"(need={fallback_needed:,}, available={len(fallback_pool):,})."
                )

            picked = rng.choice(
                fallback_pool, size=fallback_needed, replace=False)
            selected_parts.append(picked)
            fallback_counts[p] = fallback_needed

        selected_idx = np.concatenate(selected_parts)

        if unique_within_set and len(np.unique(selected_idx)) != len(selected_idx):
            raise RuntimeError(f"Duplicate sites detected in PERM_{p}.")
        if len(selected_idx) != n_fg:
            raise RuntimeError(
                f"PERM_{p} has {len(selected_idx):,} sites; foreground has {n_fg:,}.")

        rng.shuffle(selected_idx)
        perm_mat[p] = selected_idx.astype(np.int32)
        foreground_overlap_counts[p] = int(fg[selected_idx].sum())

    foreground_unclassified = {
        name: int((fg & (code == len(match_definitions[name]))).sum())
        for name, code in zip(match_controls, group_codes)
    }

    metadata = {
        "n_total_sites": n_sites,
        "n_foreground": n_fg,
        "n_background": n_bg,
        "foreground": foreground_name,
        "target_definition": str(target_definition),
        "control_pool": (
            "all_sites" if sample_from_all_sites else "effective_background"
        ),
        "sample_from_all_sites": bool(sample_from_all_sites),
        "store": store,
        "index_type": "0-based positional index" if store == "index" else None,
        "match_controls": list(match_controls),
        "match_definitions": {name: category_names[name] for name in match_controls},
        "unclassified_site_counts": unclassified_counts,
        "foreground_unclassified_counts": foreground_unclassified,
        "overlapping_bin_counts": overlap_counts,
        "n_possible_strata": int(np.prod(shape)),
        "n_foreground_strata": len(strata),
        "n_insufficient_strata": len(insufficient),
        "insufficient_strata": insufficient,
        "n_permutations": n_perm,
        "seed": seed,
        "unique_within_set": unique_within_set,
        "allow_fallback": allow_fallback,
        "fallback_counts": fallback_counts.astype(int).tolist(),
        "foreground_overlap_counts": (
            foreground_overlap_counts.astype(int).tolist()
        ),
        "foreground_overlap_fractions": (
            foreground_overlap_counts.astype(float) / n_fg
        ).tolist(),
        "foreground_overlap_summary": {
            "min": int(foreground_overlap_counts.min()),
            "max": int(foreground_overlap_counts.max()),
            "mean": float(foreground_overlap_counts.mean()),
        },
    }

    if store == "index":
        foreground_out = foreground_idx
        permutations_out = perm_mat
    else:
        foreground_out = site_ids[foreground_idx].tolist()
        permutations_out = {
            f"PERM_{p}": site_ids[perm_mat[p]].tolist()
            for p in range(n_perm)
        }

    result = {
        "metadata": metadata,
        "foreground": foreground_out,
        "permutations": permutations_out,
    }

    if output is not None:
        if output_format == "json":
            json_result = {
                "metadata": metadata,
                "foreground": foreground_out.tolist() if store == "index" else foreground_out,
                "permutations": (
                    {f"PERM_{p}": perm_mat[p].astype(
                        int).tolist() for p in range(n_perm)}
                    if store == "index" else permutations_out
                ),
            }
            opener = gzip.open if str(output).endswith(".gz") else open
            with opener(output, "wt", encoding="utf-8") as f:
                json.dump(json_result, f, ensure_ascii=False,
                          separators=(",", ":"))

        elif output_format == "npy":
            np.save(f"{output}.foreground.npy", foreground_idx)
            np.save(f"{output}.permutations.npy", perm_mat)
            with open(f"{output}.metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

    return


def run_partition_contrast(
    contrast_id,
    contrast_json,
    context_df,
    landraces_df,
    env_id_list,
    predict_sample_ids,
    vcf_file,
    results_dir,
    match_controls=("MAF", "Recomb", "Pi", "Gene"),
    n_perm=1000,
    seed=42,
    grm_n_jobs=80,
    reml_n_jobs=80,
    blup_n_jobs=80,
    score_n_jobs=40,
    debug=False,
    overwrite=False,
    cleanup_grms=None,
    cleanup_variant_sets=None,
    pruned_vcf_file=None,
    n_pcs=5,
    auto_max_pcs=20,
    use_complement=False,
    sample_from_all_sites=False,
):
    """
    Run the complete partition / prediction workflow for one contrast.

    Final publication-relevant outputs are saved under:
        <results_dir>/final/

    Intermediate files are saved under:
        <results_dir>/_work/

    Parameters
    ----------
    debug : bool
        If True, retain intermediate files for inspection/debugging.

    cleanup_grms : bool or None
        Delete the complete GRM/REML/BLUP/score working directory after
        final outputs are saved. If None, defaults to not debug.

    cleanup_variant_sets : bool or None
        Delete foreground/permutation NPY files after successful completion.
        If None, defaults to not debug.

    pruned_vcf_file : str or None
        Optional LD-pruned VCF used to calculate genome-wide population PCs.
        When provided, the same PCs are included as fixed covariates only in
        the REML analysis used for variance partitioning (FOREGROUND, all
        PERMs and ALL_SITE). BLUP fitting and genetic-score prediction remain
        on the original unadjusted pipeline.

    n_pcs : int or "auto"
        Number of PCs from pruned_vcf_file to include as fixed covariates in
        the variance-partitioning REML analysis. If "auto", select the number
        from the PCA eigenvalue scree elbow without using phenotype data.
        Ignored when pruned_vcf_file is None.

    auto_max_pcs : int
        Maximum number of leading PCs considered when n_pcs="auto". Default 20.

    use_complement : bool
        If True, use the complement of the contrast-defined foreground as the
        effective FOREGROUND set. Permutations are then sampled from the
        original foreground. Mutually exclusive with sample_from_all_sites.

    sample_from_all_sites : bool
        If True, retain the contrast-defined foreground but sample each
        matched permutation from all eligible context sites, including the
        foreground. Mutually exclusive with use_complement.

    Returns
    -------
    dict
        {
            "contrast_id": ...,
            "contrast_dir": ...,
            "final_dir": ...,
            "reml_df": ...,
            "score_df": ...
        }
    """

    if not isinstance(use_complement, (bool, np.bool_)):
        raise TypeError("use_complement must be boolean.")
    if not isinstance(sample_from_all_sites, (bool, np.bool_)):
        raise TypeError("sample_from_all_sites must be boolean.")
    if use_complement and sample_from_all_sites:
        raise ValueError(
            "use_complement and sample_from_all_sites are mutually exclusive."
        )

    match_controls = tuple(match_controls)
    n_perm = int(n_perm)
    seed = int(seed)
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1.")

    if use_complement:
        partition_mode = "complement"
        target_definition = "complement_of_original_foreground"
        control_pool = "effective_background"
    elif sample_from_all_sites:
        partition_mode = "all_sites"
        target_definition = "original_foreground"
        control_pool = "all_sites"
    else:
        partition_mode = "default"
        target_definition = "original_foreground"
        control_pool = "effective_background"

    requested_partition_config = {
        "partition_mode": partition_mode,
        "use_complement": bool(use_complement),
        "sample_from_all_sites": bool(sample_from_all_sites),
        "target_definition": target_definition,
        "control_pool": control_pool,
        "match_controls": list(match_controls),
        "n_permutations": n_perm,
        "seed": seed,
    }

    pruned_vcf_signature = None
    requested_n_pcs = None

    if pruned_vcf_file is not None:
        pruned_vcf_file = os.path.abspath(os.path.expanduser(pruned_vcf_file))
        if not os.path.isfile(pruned_vcf_file):
            raise FileNotFoundError(pruned_vcf_file)

        if isinstance(n_pcs, str):
            if n_pcs.lower() != "auto":
                raise ValueError("n_pcs must be a positive integer or 'auto'.")
            requested_n_pcs = "auto"
            n_pcs = "auto"
        else:
            n_pcs = int(n_pcs)
            if n_pcs < 1:
                raise ValueError("n_pcs must be >= 1.")
            requested_n_pcs = n_pcs

        auto_max_pcs = int(auto_max_pcs)
        if auto_max_pcs < 3:
            raise ValueError("auto_max_pcs must be >= 3.")

        stat = os.stat(pruned_vcf_file)
        pruned_vcf_signature = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    if cleanup_grms is None:
        cleanup_grms = not debug

    if cleanup_variant_sets is None:
        cleanup_variant_sets = not debug

    # ================================================================
    # 1. Directory structure
    # ================================================================

    work_dir = os.path.join(results_dir, "_work")
    final_dir = os.path.join(results_dir, "final")
    run_metadata_file = os.path.join(final_dir, "run_metadata.json")

    old_run_meta = None
    if not overwrite and os.path.isfile(run_metadata_file):
        try:
            with open(run_metadata_file) as f:
                old_run_meta = json.load(f)
        except Exception as exc:
            raise ValueError(
                f"[{contrast_id}] Cannot validate existing partition "
                f"configuration in {run_metadata_file}. Use overwrite=True "
                "or a different results_dir."
            ) from exc

        old_partition = old_run_meta.get("partition_sampling", {})
        partition_config_matches = all(
            old_partition.get(key) == value
            for key, value in requested_partition_config.items()
        )
        if not partition_config_matches:
            raise ValueError(
                f"[{contrast_id}] Existing results in {results_dir} use a "
                "different or unrecorded partition-sampling configuration. "
                "Use overwrite=True or a different results_dir."
            )

    final_files = [
        "reml_summary.csv",
        "reml_summary_unadjusted.csv",
        "score_summary.csv",
        "matching_metadata.json",
        "run_metadata.json",
    ]

    if pruned_vcf_file is not None:
        final_files += [
            "reml_summary_pc_adjusted.csv",
            "population_pcs.covar",
            "population_pca_selection.csv",
            "population_structure_metadata.json",
        ]

    final_complete = all(
        os.path.isfile(os.path.join(final_dir, f))
        for f in final_files
    )

    if not overwrite and final_complete:
        reml_df = pd.read_csv(os.path.join(final_dir, "reml_summary.csv"))
        score_df = pd.read_csv(os.path.join(final_dir, "score_summary.csv"))

        has_all_site = (
            "set" in reml_df.columns
            and "set" in score_df.columns
            and "ALL_SITE" in set(reml_df["set"].astype(str))
            and "ALL_SITE" in set(score_df["set"].astype(str))
        )

        population_config_matches = False
        try:
            old_pop = old_run_meta.get("population_structure", {})
            old_requested_n_pcs = old_pop.get(
                "requested_n_pcs", old_pop.get("n_pcs")
            )
            population_config_matches = (
                old_pop.get("pruned_vcf_file") == pruned_vcf_file
                and old_requested_n_pcs == requested_n_pcs
                and old_pop.get("pruned_vcf_signature") == pruned_vcf_signature
                and (
                    requested_n_pcs != "auto"
                    or old_pop.get("auto_max_pcs", 20) == auto_max_pcs
                )
            )
        except Exception:
            population_config_matches = False

        if has_all_site and population_config_matches:
            print(
                f"\n[{contrast_id}] Final outputs already exist, skipping contrast.")
            return {
                "contrast_id": contrast_id,
                "contrast_dir": results_dir,
                "final_dir": final_dir,
                "reml_df": reml_df,
                "score_df": score_df,
            }

        if not has_all_site:
            print(
                f"\n[{contrast_id}] Existing final outputs do not contain "
                "ALL_SITE; continuing analysis."
            )
        else:
            print(
                f"\n[{contrast_id}] Existing final outputs use different "
                "population-structure settings; continuing analysis."
            )

    pheno_cols = env_id_list

    if overwrite and os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    if overwrite and os.path.isdir(final_dir):
        shutil.rmtree(final_dir)

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    variant_sets_prefix = os.path.join(
        work_dir,
        "variant_sets",
    )

    grms_dir = os.path.join(
        work_dir,
        "grms",
    )

    os.makedirs(grms_dir, exist_ok=True)

    # ================================================================
    # 2. Build context dataframe
    # ================================================================

    other_columns = [
        "genic_2kb",
        "maf_0.05_0.1",
        "maf_0.1_0.2",
        "maf_0.2_0.3",
        "maf_0.3_0.4",
        "maf_0.4_0.5",
        "low_pi_genome",
        "med_pi_genome",
        "high_pi_genome",
        "low_recombination",
        "med_recombination",
        "high_recombination",
    ]

    context_df = build_enrich_data_df(
        context_df,
        contrast_json,
        env_id_list,
        other_columns=other_columns,
    )

    context_df["non_genic"] = ~context_df["genic_2kb"]

    original_foreground = (
        context_df["foreground"].fillna(False).to_numpy(dtype=bool)
    )
    effective_foreground = (
        ~original_foreground if use_complement else original_foreground.copy()
    )

    n_context_sites = int(len(context_df))
    n_original_foreground = int(original_foreground.sum())
    n_effective_foreground = int(effective_foreground.sum())
    n_effective_background = n_context_sites - n_effective_foreground

    partition_sampling = {
        **requested_partition_config,
        "n_context_sites": n_context_sites,
        "n_original_foreground_sites": n_original_foreground,
        "n_original_background_sites": n_context_sites - n_original_foreground,
        "n_effective_foreground_sites": n_effective_foreground,
        "n_effective_background_sites": n_effective_background,
        "original_foreground_fraction": (
            n_original_foreground / n_context_sites if n_context_sites else None
        ),
        "effective_foreground_fraction": (
            n_effective_foreground / n_context_sites if n_context_sites else None
        ),
    }

    print(f"\n[{contrast_id}] Partition sampling configuration:")
    print(f"  partition_mode: {partition_mode}")
    print(f"  target_definition: {target_definition}")
    print(f"  control_pool: {control_pool}")
    print(
        "  original foreground/background: "
        f"{n_original_foreground:,}/"
        f"{n_context_sites - n_original_foreground:,} "
        f"({partition_sampling['original_foreground_fraction']:.2%} foreground)"
    )
    print(
        "  effective foreground/background: "
        f"{n_effective_foreground:,}/{n_effective_background:,} "
        f"({partition_sampling['effective_foreground_fraction']:.2%} foreground)"
    )

    # ================================================================
    # 3. Build matched variant sets
    # ================================================================

    print(f"\n[{contrast_id}] Building matched variant sets...")

    build_variant_sets(
        context_df,
        foreground=effective_foreground,
        match_controls=match_controls,
        n_perm=n_perm,
        seed=seed,
        sample_from_all_sites=sample_from_all_sites,
        target_definition=target_definition,
        output=variant_sets_prefix,
    )

    matching_metadata_file = f"{variant_sets_prefix}.metadata.json"
    with open(matching_metadata_file) as f:
        matching_metadata = json.load(f)
    matching_metadata["partition_mode"] = partition_mode
    matching_metadata["use_complement"] = bool(use_complement)
    matching_metadata["partition_sampling"] = partition_sampling
    with open(matching_metadata_file, "w") as f:
        json.dump(matching_metadata, f, indent=2)

    # ================================================================
    # 4. Load training / prediction sample information
    # ================================================================

    landrace_sample_ids = (
        landraces_df["LIB"]
        .astype(str)
        .tolist()
    )

    # ================================================================
    # 5. GRMs
    # ================================================================

    print(f"\n[{contrast_id}] Building GRMs...")

    grm_dir = build_grms(
        context_df=context_df,
        vcf_file=vcf_file,
        training_sample_ids=landrace_sample_ids,
        variant_sets=variant_sets_prefix,
        output=grms_dir,
        n_jobs=grm_n_jobs,
    )

    # ================================================================
    # 6. Optional population-structure covariates
    # ================================================================

    population_info = None
    covar_file = None

    if pruned_vcf_file is not None:
        if requested_n_pcs == "auto":
            print(
                f"\n[{contrast_id}] Building/reusing genome-wide population "
                f"PCs from the LD-pruned VCF; n_pcs will be selected "
                f"automatically from the first {auto_max_pcs} axes..."
            )
        else:
            print(
                f"\n[{contrast_id}] Building/reusing {requested_n_pcs} "
                "genome-wide population PCs from the LD-pruned VCF..."
            )

        population_info = build_pca_covariates(
            pruned_vcf_file=pruned_vcf_file,
            training_sample_ids=landrace_sample_ids,
            output=os.path.join(grm_dir, "_population_structure"),
            n_pcs=requested_n_pcs,
            auto_max_pcs=auto_max_pcs,
            overwrite=overwrite,
        )
        covar_file = population_info["covar_file"]

    # The variance-partitioning REML can optionally control population PCs,
    # while BLUP / prediction deliberately stay on the original unadjusted
    # model. This keeps the two questions separate:
    #   1) variance partitioning conditional on broad population structure;
    #   2) genetic prediction using the full signal carried by each GRM.
    prediction_reml_dir = os.path.join(grm_dir, "reml")
    analysis_reml_dir = (
        os.path.join(grm_dir, "reml_pc")
        if covar_file is not None
        else prediction_reml_dir
    )
    blup_dir = os.path.join(grm_dir, "blups")
    score_dir = os.path.join(grm_dir, "scores")

    analysis_reml_overwrite = (
        overwrite
        or (
            covar_file is not None
            and population_info.get("rebuilt", False)
        )
    )

    # ================================================================
    # 7. REML
    #
    # Always keep the unadjusted REML because BLUP / prediction uses it.
    # When population PCs are available, also run a second PC-adjusted REML
    # for variance partitioning. Both result sets are retained in the final
    # output and explicitly labelled.
    # ================================================================

    print(
        f"\n[{contrast_id}] Running unadjusted REML "
        "for BLUP / prediction..."
    )

    reml_unadjusted_df = run_reml(
        grm_dir=grm_dir,
        pheno_df=landraces_df,
        pheno_cols=pheno_cols,
        sample_col="LIB",
        output=prediction_reml_dir,
        n_jobs=reml_n_jobs,
        covar_file=None,
        overwrite=overwrite,
    )
    reml_unadjusted_df = reml_unadjusted_df.copy()
    reml_unadjusted_df["pc_controlled"] = False
    reml_unadjusted_df["n_pcs"] = 0

    reml_pc_adjusted_df = None

    if covar_file is not None:
        print(
            f"\n[{contrast_id}] Running PC-adjusted REML "
            "for variance partitioning..."
        )

        reml_pc_adjusted_df = run_reml(
            grm_dir=grm_dir,
            pheno_df=landraces_df,
            pheno_cols=pheno_cols,
            sample_col="LIB",
            output=analysis_reml_dir,
            n_jobs=reml_n_jobs,
            covar_file=covar_file,
            overwrite=analysis_reml_overwrite,
        )
        reml_pc_adjusted_df = reml_pc_adjusted_df.copy()
        reml_pc_adjusted_df["pc_controlled"] = True
        reml_pc_adjusted_df["n_pcs"] = int(population_info["n_pcs"])

    reml_columns = [
        "set",
        "phenotype",
        "perm_index",
        "is_foreground",
        "is_all_site",
        "pc_controlled",
        "n_pcs",
        "n_samples",
        "total_samples",
        "with_phenotypes",
        "converged",
        "heritability",
        "heritability_sd",
        "covar_heritability",
        "lrt_stat",
        "lrt_p",
    ]

    reml_unadjusted_df = reml_unadjusted_df[reml_columns].copy()

    if reml_pc_adjusted_df is not None:
        reml_pc_adjusted_df = reml_pc_adjusted_df[reml_columns].copy()
        reml_df = pd.concat(
            [reml_unadjusted_df, reml_pc_adjusted_df],
            ignore_index=True,
        )
    else:
        reml_df = reml_unadjusted_df.copy()

    reml_df = reml_df.sort_values(
        ["phenotype", "is_all_site", "is_foreground", "perm_index", "pc_controlled"],
        ascending=[True, False, False, True, True],
        na_position="first",
    ).reset_index(drop=True)

    # ================================================================
    # 8. BLUP
    # ================================================================

    print(f"\n[{contrast_id}] Running BLUPs...")

    run_blups(
        grm_dir=grm_dir,
        reml_dir=prediction_reml_dir,
        pheno_cols=pheno_cols,
        output=blup_dir,
        n_jobs=blup_n_jobs,
        overwrite=overwrite,
    )

    # ================================================================
    # 9. Prediction
    # ================================================================

    print(f"\n[{contrast_id}] Predicting GxE scores...")

    score_df = predict_scores(
        grm_dir=grm_dir,
        prediction_sample_ids=predict_sample_ids,
        pheno_cols=pheno_cols,
        blup_dir=blup_dir,
        output=score_dir,
        n_jobs=score_n_jobs,
        overwrite=overwrite,
    )

    # ================================================================
    # 10. Save publication-relevant results
    # ================================================================

    reml_final_file = os.path.join(
        final_dir,
        "reml_summary.csv",
    )
    reml_unadjusted_final_file = os.path.join(
        final_dir,
        "reml_summary_unadjusted.csv",
    )
    reml_pc_adjusted_final_file = (
        os.path.join(final_dir, "reml_summary_pc_adjusted.csv")
        if reml_pc_adjusted_df is not None
        else None
    )

    score_final_file = os.path.join(
        final_dir,
        "score_summary.csv",
    )

    # Combined table: one or two rows per GRM × phenotype, distinguished by
    # pc_controlled / n_pcs. Also keep branch-specific tables for convenience.
    reml_df.to_csv(
        reml_final_file,
        index=False,
    )
    reml_unadjusted_df.to_csv(
        reml_unadjusted_final_file,
        index=False,
    )

    if reml_pc_adjusted_df is not None:
        reml_pc_adjusted_df.to_csv(
            reml_pc_adjusted_final_file,
            index=False,
        )
    else:
        stale_pc_file = os.path.join(final_dir, "reml_summary_pc_adjusted.csv")
        if os.path.isfile(stale_pc_file):
            os.remove(stale_pc_file)

    score_df.to_csv(
        score_final_file,
        index=False,
    )

    # Preserve matching metadata, but not necessarily the huge NPY arrays.
    matching_metadata_source = (
        f"{variant_sets_prefix}.metadata.json"
    )

    matching_metadata_final = os.path.join(
        final_dir,
        "matching_metadata.json",
    )

    if os.path.isfile(matching_metadata_source):
        shutil.copy2(
            matching_metadata_source,
            matching_metadata_final,
        )

    population_pcs_final = None
    population_selection_final = None
    population_metadata_final = None

    if population_info is not None:
        population_pcs_final = os.path.join(
            final_dir,
            "population_pcs.covar",
        )
        population_selection_final = os.path.join(
            final_dir,
            "population_pca_selection.csv",
        )
        population_metadata_final = os.path.join(
            final_dir,
            "population_structure_metadata.json",
        )

        shutil.copy2(
            population_info["covar_file"],
            population_pcs_final,
        )
        shutil.copy2(
            population_info["selection_file"],
            population_selection_final,
        )
        shutil.copy2(
            os.path.join(grm_dir, "_population_structure", "metadata.json"),
            population_metadata_final,
        )
    else:
        for name in (
            "population_pcs.covar",
            "population_pca_selection.csv",
            "population_structure_metadata.json",
        ):
            f = os.path.join(final_dir, name)
            if os.path.isfile(f):
                os.remove(f)

    # ================================================================
    # 11. Run metadata / provenance
    # ================================================================

    run_metadata = {
        "contrast_id": contrast_id,
        "contrast_settings": contrast_json,
        "results_dir": results_dir,
        "n_context_sites": n_context_sites,
        "n_foreground_sites": n_effective_foreground,
        "n_original_foreground_sites": n_original_foreground,
        "partition_sampling": partition_sampling,
        "match_controls": list(match_controls),
        "n_permutations": int(n_perm),
        "seed": int(seed),
        "phenotypes": list(pheno_cols),
        "vcf_file": os.path.abspath(vcf_file),
        "population_structure": {
            "pruned_vcf_file": pruned_vcf_file,
            "pruned_vcf_signature": pruned_vcf_signature,
            "requested_n_pcs": requested_n_pcs,
            "n_pcs": (
                population_info["n_pcs"]
                if population_info is not None
                else None
            ),
            "auto_max_pcs": (
                auto_max_pcs
                if requested_n_pcs == "auto"
                else None
            ),
            "selection_method": (
                population_info["selection_method"]
                if population_info is not None
                else None
            ),
            "covariate_method": (
                "LDAK PCA fixed covariates from LD-pruned VCF"
                if pruned_vcf_file is not None
                else None
            ),
            "applied_to": (
                "variance-partitioning REML only"
                if pruned_vcf_file is not None
                else None
            ),
            "prediction_uses_population_covariates": False,
            "population_pcs": population_pcs_final,
            "pca_selection": population_selection_final,
            "metadata": population_metadata_final,
        },
        "n_training_samples": len(
            landrace_sample_ids
        ),
        "n_prediction_samples": len(
            predict_sample_ids
        ),
        "grm_n_jobs": grm_n_jobs,
        "reml_n_jobs": reml_n_jobs,
        "blup_n_jobs": blup_n_jobs,
        "score_n_jobs": score_n_jobs,
        "variance_reml_dir": analysis_reml_dir,
        "prediction_reml_dir": prediction_reml_dir,
        "blup_dir": blup_dir,
        "score_dir": score_dir,
        "debug": debug,
        "final_outputs": {
            "reml_summary": reml_final_file,
            "reml_summary_unadjusted": reml_unadjusted_final_file,
            "reml_summary_pc_adjusted": reml_pc_adjusted_final_file,
            "score_summary": score_final_file,
            "matching_metadata": matching_metadata_final,
            "population_pcs": population_pcs_final,
            "population_pca_selection": population_selection_final,
            "population_structure_metadata": population_metadata_final,
        },
    }

    with open(
        run_metadata_file,
        "w",
    ) as f:
        json.dump(
            run_metadata,
            f,
            indent=2,
        )

    # ================================================================
    # 12. Cleanup
    #
    # Only clean AFTER all final results were successfully saved.
    # ================================================================

    if cleanup_grms:
        print(
            f"\n[{contrast_id}] "
            f"Removing intermediate GRM directory..."
        )

        if os.path.isdir(grms_dir):
            shutil.rmtree(grms_dir)

    if cleanup_variant_sets:
        print(
            f"[{contrast_id}] "
            f"Removing matched-set intermediate arrays..."
        )

        for suffix in (
            ".foreground.npy",
            ".permutations.npy",
            ".metadata.json",
        ):
            f = f"{variant_sets_prefix}{suffix}"

            if os.path.isfile(f):
                os.remove(f)

    # Remove empty work directory if possible.
    if not debug:
        try:
            os.rmdir(work_dir)
        except OSError:
            pass

    print(
        f"\n[{contrast_id}] Finished."
    )
    print(
        f"Final results: {final_dir}"
    )

    return {
        "contrast_id": contrast_id,
        "results_dir": results_dir,
    }


def _as_numeric_array(x):
    """Convert array-like values to a clean 1D numpy array."""
    arr = np.asarray(x, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _validate_variance_dict(data):
    """
    Expected structure:

    data[env][feature] = {
        "foreground_heritability": float,
        "genome_heritability": float,
        "permutation_heritability_list": list,

        # optional
        "direction": "left" or "right",
        "p_value": float,
    }
    """

    if not isinstance(data, dict):
        raise TypeError("data must be a dict.")

    required = {
        "foreground_heritability",
        "genome_heritability",
        "permutation_heritability_list",
    }

    for env, feature_dict in data.items():

        if not isinstance(feature_dict, dict):
            raise TypeError(
                f"data[{env!r}] must be a dict of features."
            )

        for feature, result in feature_dict.items():

            if not isinstance(result, dict):
                raise TypeError(
                    f"data[{env!r}][{feature!r}] must be a dict."
                )

            missing = required - set(result)

            if missing:
                raise ValueError(
                    f"Missing keys in data[{env!r}][{feature!r}]: "
                    f"{sorted(missing)}"
                )

            if "direction" in result:

                if result["direction"] not in ("left", "right"):
                    raise ValueError(
                        f"'direction' must be 'left' or 'right', "
                        f"got {result['direction']!r} "
                        f"for env={env!r}, feature={feature!r}."
                    )


def _get_result(data, env, feature):
    """Return one env × feature result or None if absent."""
    return data.get(env, {}).get(feature)


def _kde_density(
    values,
    xs,
    x_min,
    x_max,
    bw_method="scott",
):
    """
    Calculate KDE density.

    scipy.stats.gaussian_kde is used when possible.
    A simple Gaussian fallback is used otherwise.
    """

    values = _as_numeric_array(values)

    if values.size >= 2 and np.std(values) > 0:

        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(
                values,
                bw_method=bw_method,
            )

            return kde(xs)

        except Exception:
            pass

    # fallback
    if values.size == 0:
        return np.zeros_like(xs)

    if values.size == 1:

        h = max(
            (x_max - x_min) / 50,
            1e-9,
        )

    else:

        std = np.std(
            values,
            ddof=1,
        )

        q75, q25 = np.percentile(
            values,
            [75, 25],
        )

        iqr_sigma = (
            (q75 - q25) / 1.349
            if q75 > q25
            else np.nan
        )

        candidates = [
            v
            for v in (std, iqr_sigma)
            if np.isfinite(v) and v > 0
        ]

        if candidates:
            sigma = min(candidates)

        else:
            sigma = max(
                (x_max - x_min) / 20,
                1e-9,
            )

        h = max(
            0.9
            * sigma
            * values.size ** (-1 / 5),
            1e-9,
        )

    z = (
        xs[:, None]
        - values[None, :]
    ) / h

    density = (
        np.exp(-0.5 * z**2).sum(axis=1)
        / (
            values.size
            * h
            * np.sqrt(2 * np.pi)
        )
    )

    return density


def _plot_variance_cell(
    ax,
    control,
    foreground,
    genome,
    xlim,

    half_width=0.22,

    violin_color="#BDBA82",
    violin_edgecolor="black",
    violin_edge_lw=0.3,
    violin_alpha=0.85,

    median_lw=0.5,

    foreground_color="#4D77A3",
    genome_color="#AEAEAF",

    line_lw=3.0,

    bw_method="scott",
    gridsize=256,
):
    """
    Plot one half-violin panel.
    """

    control = _as_numeric_array(control)

    x_min, x_max = xlim

    xs = np.linspace(
        x_min,
        x_max,
        gridsize,
    )

    density = _kde_density(
        control,
        xs,
        x_min,
        x_max,
        bw_method=bw_method,
    )

    if density.size and np.nanmax(density) > 0:

        width = (
            density
            / np.nanmax(density)
            * half_width
        )

    else:

        width = np.zeros_like(density)

    # --------------------------------------------------------
    # half violin
    # --------------------------------------------------------

    ax.fill_between(
        xs,
        0,
        width,
        facecolor=violin_color,
        edgecolor=violin_edgecolor,
        linewidth=violin_edge_lw,
        alpha=violin_alpha,
        zorder=2,
    )

    # --------------------------------------------------------
    # permutation median
    # --------------------------------------------------------

    if control.size:

        median = float(
            np.median(control)
        )

        idx = int(
            np.argmin(
                np.abs(xs - median)
            )
        )

        median_height = (
            float(width[idx])
            if width.size
            else half_width * 0.2
        )

        ax.vlines(
            median,
            0,
            median_height,
            color="black",
            linewidth=median_lw,
            zorder=3,
        )

    if width.size and np.nanmax(width) > 0:

        y_high = float(
            np.nanmax(width)
        )

    else:

        y_high = half_width

    # --------------------------------------------------------
    # foreground
    # --------------------------------------------------------

    ax.vlines(
        foreground,
        0,
        y_high,
        color=foreground_color,
        linewidth=line_lw,
        zorder=4,
    )

    # --------------------------------------------------------
    # genome-wide
    # --------------------------------------------------------

    ax.vlines(
        genome,
        0,
        y_high,
        color=genome_color,
        linewidth=line_lw,
        zorder=4,
    )

    # baseline
    ax.hlines(
        0,
        x_min,
        x_max,
        color="black",
        linewidth=0.6,
        zorder=1,
    )

    ax.set_xlim(xlim)

    ax.set_ylim(
        0,
        half_width + 0.10,
    )

    ax.set_yticks([])

    ax.tick_params(
        axis="y",
        left=False,
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=3)
    )

    for spine in (
        "top",
        "right",
        "left",
    ):
        ax.spines[spine].set_visible(False)

    return y_high


def plot_variance_explained(
    data,

    env_order=None,
    feature_order=None,

    env_labels=None,
    feature_labels=None,

    *,
    figsize_per_col=3.0,
    figsize_per_row=1.15,

    half_width=0.22,

    violin_color="#BDBA82",
    violin_edgecolor="black",
    violin_edge_lw=0.3,
    violin_alpha=0.85,

    median_lw=0.5,

    foreground_color="#4D77A3",
    genome_color="#AEAEAF",

    line_lw=3.0,

    p_value_threshold=0.05,

    star_size=10,
    star_offset_frac=0.08,
    star_y_factor=0.95,

    x_axis_label="Variance explained (%)",
    display_percent=True,

    column_title_fontsize=14,
    row_label_fontsize=14,
    tick_labelsize=10,
    axis_labelsize=11,
    title_pad=0,

    label_wrap=20,

    legend=True,

    foreground_legend_label="Variants in context",
    genome_legend_label="All variants",
    control_legend_label="Matched control variants",

    legend_fontsize=11,
    legend_x=0.7,
    legend_y=-0.1,

    wspace=0.15,
    hspace=0.05,

    left=0.14,
    right=0.98,
    top=0.7,

    save=None,
    dpi=300,

    show=True,
):
    """
    Plot an environment × genomic-feature matrix.

    Expected input
    --------------
    data[env][feature] = {
        "foreground_heritability": float,
        "genome_heritability": float,
        "permutation_heritability_list": list,

        # optional
        "direction": "left" or "right",
        "p_value": float,
    }

    direction controls which side of the foreground line
    the significance star is placed.

    Returns
    -------
    fig, axes
    """

    set_plot_style()

    _validate_variance_dict(data)

    if env_labels is None:
        env_labels = {}

    if feature_labels is None:
        feature_labels = {}

    # ========================================================
    # Plot order
    # ========================================================

    if env_order is None:
        env_order = list(data.keys())

    if feature_order is None:

        feature_order = []

        for env in env_order:
            for feature in data.get(env, {}):

                if feature not in feature_order:
                    feature_order.append(feature)

    nrows = len(feature_order)
    ncols = len(env_order)

    if nrows == 0 or ncols == 0:
        raise ValueError(
            "No environments or features available for plotting."
        )

    # ========================================================
    # Figure size
    #
    # Reserve explicit vertical space for:
    #   top titles
    #   x labels
    #   legend
    #
    # This prevents the legend from overlapping the x-axis
    # when only one row is plotted.
    # ========================================================

    top_space = 0.65
    bottom_space = 0.85 if legend else 0.40

    fig_height = (
        figsize_per_row * nrows
        + top_space
        + bottom_space
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,

        figsize=(
            figsize_per_col * ncols,
            fig_height,
        ),

        squeeze=False,
    )

    # Convert reserved bottom space to figure fraction
    bottom = bottom_space / fig_height

    # ========================================================
    # Determine x range for each environment
    # ========================================================

    env_xlim = {}

    for env in env_order:

        values = []

        for feature in feature_order:

            result = _get_result(
                data,
                env,
                feature,
            )

            if result is None:
                continue

            control = _as_numeric_array(
                result[
                    "permutation_heritability_list"
                ]
            )

            values.extend(
                control.tolist()
            )

            values.append(
                float(
                    result[
                        "foreground_heritability"
                    ]
                )
            )

            values.append(
                float(
                    result[
                        "genome_heritability"
                    ]
                )
            )

        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:

            x_min = 0.0
            x_max = 1.0

        else:

            x_min = float(
                values.min()
            )

            x_max = float(
                values.max()
            )

            if x_min == x_max:

                delta = max(
                    abs(x_min) * 0.05,
                    1e-6,
                )

                x_min -= delta
                x_max += delta

        span = x_max - x_min

        # leave horizontal room for significance stars
        padding = span * (
            0.10
            + star_offset_frac
        )

        env_xlim[env] = (
            x_min - padding,
            x_max + padding,
        )

    # ========================================================
    # Plot each panel
    # ========================================================

    for i, feature in enumerate(
        feature_order
    ):

        for j, env in enumerate(
            env_order
        ):

            ax = axes[i, j]

            result = _get_result(
                data,
                env,
                feature,
            )

            # ------------------------------------------------
            # Missing env × feature
            # ------------------------------------------------

            if result is None:

                ax.set_visible(False)
                continue

            # ------------------------------------------------
            # Values
            # ------------------------------------------------

            control = result[
                "permutation_heritability_list"
            ]

            foreground = float(
                result[
                    "foreground_heritability"
                ]
            )

            genome = float(
                result[
                    "genome_heritability"
                ]
            )

            p_value = result.get(
                "p_value",
                None,
            )

            direction = result.get(
                "direction",
                None,
            )

            # ------------------------------------------------
            # Infer direction when omitted
            # ------------------------------------------------

            if direction is None:

                control_array = _as_numeric_array(
                    control
                )

                if control_array.size:

                    direction = (
                        "right"
                        if foreground >= np.median(control_array)
                        else "left"
                    )

                else:

                    direction = "right"

            direction_sign = (
                1
                if direction == "right"
                else -1
            )

            # ------------------------------------------------
            # Main panel
            # ------------------------------------------------

            xlim = env_xlim[env]

            _plot_variance_cell(
                ax=ax,

                control=control,
                foreground=foreground,
                genome=genome,

                xlim=xlim,

                half_width=half_width,

                violin_color=violin_color,
                violin_edgecolor=violin_edgecolor,
                violin_edge_lw=violin_edge_lw,
                violin_alpha=violin_alpha,

                median_lw=median_lw,

                foreground_color=foreground_color,
                genome_color=genome_color,

                line_lw=line_lw,
            )

            # ------------------------------------------------
            # Significance star
            # ------------------------------------------------

            if (
                p_value is not None
                and np.isfinite(p_value)
                and p_value < p_value_threshold
            ):

                span = (
                    xlim[1]
                    - xlim[0]
                )

                x_star = (
                    foreground
                    + direction_sign
                    * star_offset_frac
                    * span
                )

                x_star = np.clip(
                    x_star,
                    xlim[0] + 0.03 * span,
                    xlim[1] - 0.03 * span,
                )

                # --------------------------------------------
                # Fixed height relative to violin height.
                #
                # This is intentionally NOT based on y_high,
                # so single-row and multi-row figures look
                # visually consistent.
                # --------------------------------------------

                y_star = (
                    half_width
                    * star_y_factor
                )

                ax.plot(
                    x_star,
                    y_star,

                    marker="*",
                    color="black",

                    markersize=star_size,

                    zorder=20,
                )

            # ------------------------------------------------
            # Row labels
            # ------------------------------------------------

            if j == 0:

                import textwrap

                label = feature_labels.get(
                    feature,
                    feature,
                )

                label = "\n".join(
                    textwrap.wrap(
                        str(label),
                        label_wrap,
                    )
                )

                ax.text(
                    -0.04,
                    0.35,

                    label,

                    transform=ax.transAxes,

                    ha="right",
                    va="center",

                    fontsize=row_label_fontsize,
                )

            # ------------------------------------------------
            # Column titles
            # ------------------------------------------------

            if i == 0:

                import textwrap

                title = env_labels.get(
                    env,
                    env,
                )

                title = "\n".join(
                    textwrap.wrap(
                        str(title),
                        label_wrap,
                    )
                )

                ax.set_title(
                    title,
                    fontsize=column_title_fontsize,
                    pad=title_pad,
                )

            # ------------------------------------------------
            # X axis
            # ------------------------------------------------

            if i != nrows - 1:

                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                )

            else:

                ax.tick_params(
                    axis="x",
                    labelsize=tick_labelsize,
                )

                if display_percent:

                    ax.xaxis.set_major_formatter(
                        FuncFormatter(
                            lambda x, pos:
                            f"{x * 100:g}"
                        )
                    )

                ax.set_xlabel(
                    x_axis_label,
                    fontsize=axis_labelsize,
                )

    # ========================================================
    # Layout
    # ========================================================

    fig.subplots_adjust(
        left=left,
        right=right,

        bottom=bottom,
        top=top,

        wspace=wspace,
        hspace=hspace,
    )

    # ========================================================
    # Legend
    # ========================================================

    if legend:

        handles = [

            Line2D(
                [0],
                [0],

                color=foreground_color,
                lw=line_lw,

                label=foreground_legend_label,
            ),

            Line2D(
                [0],
                [0],

                color=genome_color,
                lw=line_lw,

                label=genome_legend_label,
            ),

            Patch(
                facecolor=violin_color,
                edgecolor=violin_edgecolor,

                alpha=violin_alpha,

                label=control_legend_label,
            ),
        ]

        fig.legend(
            handles=handles,

            loc="lower center",

            bbox_to_anchor=(
                legend_x,
                legend_y,
            ),

            ncol=3,

            frameon=False,

            fontsize=legend_fontsize,
        )

    # ========================================================
    # Save / show
    # ========================================================

    if save is not None:

        fig.savefig(
            save,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, axes


def _validate_prediction_dict(data):
    """
    Expected structure:

    data[phenotype][feature] = {
        "environment": str,
        "foreground_r": float,
        "genome_r": float,
        "permutation_r_list": list,
        "genetic_score": array-like,
        "phenotype_score": array-like,

        # optional
        "prediction_r": float,
        "direction": "left" or "right",
        "p_value": float,
    }
    """
    required = {
        "environment",
        "foreground_r",
        "genome_r",
        "permutation_r_list",
        "genetic_score",
        "phenotype_score",
    }

    if not isinstance(data, dict):
        raise TypeError("data must be a dict.")

    for phenotype, feature_dict in data.items():
        if not isinstance(feature_dict, dict):
            raise TypeError(f"data[{phenotype!r}] must be a dict of features.")

        for feature, result in feature_dict.items():
            if not isinstance(result, dict):
                raise TypeError(f"data[{phenotype!r}][{feature!r}] must be a dict.")

            missing = required - set(result)
            if missing:
                raise ValueError(
                    f"Missing keys in data[{phenotype!r}][{feature!r}]: "
                    f"{sorted(missing)}"
                )

            x = np.asarray(result["genetic_score"]).ravel()
            y = np.asarray(result["phenotype_score"]).ravel()

            if x.size != y.size:
                raise ValueError(
                    "genetic_score and phenotype_score must have the same "
                    f"length for phenotype={phenotype!r}, feature={feature!r}."
                )

            for key in ("foreground_r", "genome_r"):
                if not np.isfinite(float(result[key])):
                    raise ValueError(
                        f"{key} must be finite for "
                        f"phenotype={phenotype!r}, feature={feature!r}."
                    )

            if (
                "prediction_r" in result
                and not np.isfinite(float(result["prediction_r"]))
            ):
                raise ValueError(
                    f"prediction_r must be finite for "
                    f"phenotype={phenotype!r}, feature={feature!r}."
                )

            if "direction" in result and result["direction"] not in ("left", "right"):
                raise ValueError(
                    f"'direction' must be 'left' or 'right', got "
                    f"{result['direction']!r} for phenotype={phenotype!r}, "
                    f"feature={feature!r}."
                )


def _clean_prediction_xy(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _plot_prediction_violin_cell(
    ax,
    control,
    foreground,
    xlim,
    half_width=0.22,
    y_pad=0.06,
    violin_color="#BDBA82",
    violin_edgecolor="black",
    violin_edge_lw=0.3,
    violin_alpha=0.85,
    median_lw=0.5,
    foreground_color="#4D77A3",
    foreground_lw=2.0,
    bw_method="scott",
    gridsize=256,
):
    control = _as_numeric_array(control)
    x_min, x_max = xlim
    xs = np.linspace(x_min, x_max, gridsize)
    density = _kde_density(
        control,
        xs,
        x_min,
        x_max,
        bw_method=bw_method,
    )

    if density.size and np.nanmax(density) > 0:
        width = density / np.nanmax(density) * half_width
    else:
        width = np.zeros_like(density)

    ax.fill_between(
        xs,
        0,
        width,
        facecolor=violin_color,
        edgecolor=violin_edgecolor,
        linewidth=violin_edge_lw,
        alpha=violin_alpha,
        zorder=2,
    )

    median = float(np.median(control)) if control.size else float(foreground)
    if control.size:
        idx = int(np.argmin(np.abs(xs - median)))
        median_height = float(width[idx]) if width.size else half_width * 0.2
        ax.vlines(
            median,
            0,
            median_height,
            color="black",
            linewidth=median_lw,
            zorder=3,
        )

    y_high = half_width + y_pad
    ax.vlines(
        foreground,
        0,
        y_high,
        color=foreground_color,
        linewidth=foreground_lw,
        zorder=4,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(0, y_high)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    return {
        "xs": xs,
        "width": width,
        "median": median,
    }


def _pick_prediction_r_label(
    ax,
    geom,
    foreground,
    xlim,
    border_frac=0.10,
    y_try=(0.80, 0.90, 0.72, 0.96, 0.62),
):
    x_min, x_max = xlim
    span = x_max - x_min if x_max > x_min else 1.0
    median = geom["median"]
    rel = (foreground - x_min) / span
    margin = border_frac * span

    if rel > 0.5:
        if foreground < median:
            side = "left"
        else:
            side = (
                "right"
                if abs(foreground - median) < abs(x_max - foreground)
                else "left"
            )
        if side == "right" and x_max - foreground < margin:
            side = "left"
    else:
        if foreground > median:
            side = "right"
        else:
            side = (
                "left"
                if abs(foreground - median) < abs(foreground - x_min)
                else "right"
            )
        if side == "left" and foreground - x_min < margin:
            side = "right"

    dx = 0.02 * span
    if side == "left":
        candidates = [
            (foreground - dx, "right"),
            (foreground + dx, "left"),
        ]
    else:
        candidates = [
            (foreground + dx, "left"),
            (foreground - dx, "right"),
        ]

    y_min, y_max = ax.get_ylim()
    for y_axes in y_try:
        y_data = y_min + y_axes * (y_max - y_min)

        for x_text, ha in candidates:
            idx = int(np.argmin(np.abs(geom["xs"] - x_text)))

            if y_data > float(geom["width"][idx]) + 0.005:
                return x_text, ha, y_axes, None

    bbox = {
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.85,
        "pad": 0.6,
    }

    return candidates[0][0], candidates[0][1], y_try[0], bbox


def _pick_prediction_corner(
    ax,
    x,
    y,
    dedup_r=0.10,
    corner_pad=0.03,
):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    def normalize(values, vmin, vmax):
        span = vmax - vmin
        if not np.isfinite(span) or span <= 0:
            return np.full_like(values, 0.5, dtype=float)
        return np.clip((values - vmin) / span, 0, 1)

    u = normalize(np.asarray(x, dtype=float), x_min, x_max)
    v = normalize(np.asarray(y, dtype=float), y_min, y_max)

    mask = np.isfinite(u) & np.isfinite(v)
    u = u[mask]
    v = v[mask]

    rep_u = []
    rep_v = []

    for ui, vi in zip(u, v):
        if not rep_u:
            rep_u.append(ui)
            rep_v.append(vi)
            continue

        dist2 = (
            (ui - np.asarray(rep_u)) ** 2
            + (vi - np.asarray(rep_v)) ** 2
        )

        if np.min(dist2) >= dedup_r**2:
            rep_u.append(ui)
            rep_v.append(vi)

    rep_u = np.asarray(rep_u)
    rep_v = np.asarray(rep_v)

    center_u = float(np.mean(rep_u)) if rep_u.size else 0.5
    center_v = float(np.mean(rep_v)) if rep_v.size else 0.5

    corners = [
        (
            (rep_u < 1 / 3) & (rep_v < 1 / 3),
            1 / 6,
            1 / 6,
            corner_pad,
            corner_pad,
            "left",
            "bottom",
        ),
        (
            (rep_u >= 2 / 3) & (rep_v < 1 / 3),
            5 / 6,
            1 / 6,
            1 - corner_pad,
            corner_pad,
            "right",
            "bottom",
        ),
        (
            (rep_u < 1 / 3) & (rep_v >= 2 / 3),
            1 / 6,
            5 / 6,
            corner_pad,
            1 - corner_pad,
            "left",
            "top",
        ),
        (
            (rep_u >= 2 / 3) & (rep_v >= 2 / 3),
            5 / 6,
            5 / 6,
            1 - corner_pad,
            1 - corner_pad,
            "right",
            "top",
        ),
    ]

    best = min(
        corners,
        key=lambda z: (
            int(np.sum(z[0])),
            -(
                (z[1] - center_u) ** 2
                + (z[2] - center_v) ** 2
            ),
        ),
    )

    return best[3], best[4], best[5], best[6]


def _draw_prediction_genome_lines(
    fig,
    axes,
    data,
    phenotype_order,
    feature_order,
    genome_color="#A9A9A9",
    genome_lw=2.0,
    label_fontsize=7,
    label_y=0.80,
    label_dx_frac=0.015,
):
    fig.canvas.draw()

    for j, phenotype in enumerate(phenotype_order):
        items = [
            (i, data.get(phenotype, {}).get(feature))
            for i, feature in enumerate(feature_order)
        ]
        items = [(i, result) for i, result in items if result is not None]

        if not items:
            continue

        genome_values = np.asarray(
            [
                float(result["genome_r"])
                for _, result in items
            ]
        )

        if not np.allclose(genome_values, genome_values[0]):
            raise ValueError(
                f"genome_r must be identical across features "
                f"for phenotype={phenotype!r}."
            )

        genome = float(genome_values[0])
        hist_axes = [axes[i, j, 0] for i, _ in items]
        ax_ref = hist_axes[0]
        x_min, x_max = ax_ref.get_xlim()

        if not (x_min <= genome <= x_max):
            continue

        x_display = ax_ref.transData.transform((genome, 0))[0]
        x_fig = fig.transFigure.inverted().transform((x_display, 0))[0]

        y_fig0 = min(ax.get_position().y0 for ax in hist_axes)
        y_fig1 = max(ax.get_position().y1 for ax in hist_axes)

        fig.add_artist(
            Line2D(
                [x_fig, x_fig],
                [y_fig0, y_fig1],
                transform=fig.transFigure,
                color=genome_color,
                linewidth=genome_lw,
                zorder=3.5,
                solid_capstyle="butt",
            )
        )

        # Redraw the lower spine so the gray line does not cover it.
        for ax in hist_axes:
            bbox = ax.get_position()

            fig.add_artist(
                Line2D(
                    [bbox.x0, bbox.x1],
                    [bbox.y0, bbox.y0],
                    transform=fig.transFigure,
                    color="black",
                    linewidth=ax.spines["bottom"].get_linewidth(),
                    zorder=4,
                    solid_capstyle="butt",
                )
            )

        ax_ref.text(
            genome - label_dx_frac * (x_max - x_min),
            label_y,
            "Genome-wide\n" + fr"$r = {genome:.3f}$",
            transform=ax_ref.get_xaxis_transform(),
            ha="right",
            va="center",
            fontsize=label_fontsize,
            clip_on=False,
        )


def plot_prediction_performance(
    data,
    phenotype_order=None,
    feature_order=None,
    phenotype_labels=None,
    feature_labels=None,
    environment_labels=None,
    *,
    figsize_per_col=5.0,
    figsize_per_row=1.6,
    width_ratios_pair=(2.2, 1.0),
    half_width=0.22,
    violin_color="#BDBA82",
    violin_edgecolor="black",
    violin_edge_lw=0.3,
    violin_alpha=0.85,
    median_lw=0.5,
    foreground_color="#4D77A3",
    foreground_lw=2.0,
    genome_color="#A9A9A9",
    genome_lw=2.0,
    scatter_color="gray",
    scatter_s=11,
    scatter_alpha=0.6,
    regression_lw=1.5,
    p_value_threshold=0.05,
    star_size=10,
    star_offset_frac=0.08,
    star_y_factor=0.95,
    hist_xtick_step=0.02,
    hist_axis_label="Pearson's r",
    phenotype_axis_label="Phenotype score",
    control_label="Matched control variants\n(MAF, recombination, π)",
    r_border_frac=0.10,
    r_y_try=(0.80, 0.90, 0.72, 0.96, 0.62),
    n_corner_pad=0.03,
    n_dedup_r=0.10,
    n_bbox_alpha=0.6,
    n_bbox_pad=0.3,
    genome_label_y=0.80,
    genome_label_dx_frac=0.015,
    title_fontsize=11,
    row_label_fontsize=10,
    stat_fontsize=7,
    axis_label_fontsize=7,
    hist_axis_label_fontsize=9,
    group_label_fontsize=8,
    tick_labelsize=7,
    title_pad=18,
    row_label_x=-0.40,
    row_label_y=0.80,
    scatter_xlabel_y=-0.30,
    scatter_ylabel_x=-0.18,
    wspace=0.32,
    hspace=0.68,
    save=None,
    dpi=300,
    show=True,
):
    """
    Plot prediction performance for phenotype × genomic-feature combinations.

    Expected input
    --------------
    data[phenotype][feature] = {
        "environment": str,
        "foreground_r": float,
        "genome_r": float,
        "permutation_r_list": list,
        "genetic_score": array-like,
        "phenotype_score": array-like,

        # optional
        "direction": "left" or "right",
        "p_value": float,
    }

    direction controls which side of the foreground line
    the significance star is placed. If omitted, direction
    is inferred from foreground_r relative to the permutation median.

    Returns
    -------
    fig, axes

    axes[i, j, 0] : half-violin panel
    axes[i, j, 1] : prediction scatter panel
    """
    set_plot_style()
    _validate_prediction_dict(data)

    phenotype_labels = {} if phenotype_labels is None else phenotype_labels
    feature_labels = {} if feature_labels is None else feature_labels
    environment_labels = {} if environment_labels is None else environment_labels

    if phenotype_order is None:
        phenotype_order = list(data.keys())
    else:
        phenotype_order = list(phenotype_order)

    if feature_order is None:
        feature_order = []

        for phenotype in phenotype_order:
            for feature in data.get(phenotype, {}):
                if feature not in feature_order:
                    feature_order.append(feature)
    else:
        feature_order = list(feature_order)

    nrows = len(feature_order)
    ncols = len(phenotype_order)

    if nrows == 0 or ncols == 0:
        raise ValueError(
            "No phenotypes or features available for plotting."
        )

    fig = plt.figure(
        figsize=(
            figsize_per_col * ncols,
            figsize_per_row * nrows,
        )
    )

    gs = fig.add_gridspec(
        nrows,
        ncols * 2,
        width_ratios=list(width_ratios_pair) * ncols,
        wspace=wspace,
        hspace=hspace,
    )

    axes = np.empty(
        (nrows, ncols, 2),
        dtype=object,
    )

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j, 0] = fig.add_subplot(
                gs[i, j * 2]
            )
            axes[i, j, 1] = fig.add_subplot(
                gs[i, j * 2 + 1]
            )

    # Shared r range within each phenotype column.
    phenotype_xlim = {}
    phenotype_xticks = {}

    for phenotype in phenotype_order:
        values = []

        for feature in feature_order:
            result = data.get(
                phenotype,
                {},
            ).get(feature)

            if result is None:
                continue

            values.extend(
                _as_numeric_array(
                    result["permutation_r_list"]
                ).tolist()
            )

            values.extend(
                [
                    float(result["foreground_r"]),
                    float(result["genome_r"]),
                ]
            )

        values = _as_numeric_array(values)

        if values.size:
            x_min = float(values.min())
            x_max = float(values.max())

            if x_min == x_max:
                delta = max(
                    abs(x_min) * 0.05,
                    1e-6,
                )
                x_min -= delta
                x_max += delta
        else:
            x_min = 0.0
            x_max = 1.0

        span = x_max - x_min
        pad = 0.06 * span if span > 0 else 0.06

        phenotype_xlim[phenotype] = (
            x_min - pad,
            x_max + pad,
        )

        if hist_xtick_step is not None:
            tick_start = (
                np.ceil(
                    (x_min - pad)
                    / hist_xtick_step
                )
                * hist_xtick_step
            )

            tick_end = (
                np.floor(
                    (x_max + pad)
                    / hist_xtick_step
                )
                * hist_xtick_step
            )

            phenotype_xticks[phenotype] = np.round(
                np.arange(
                    tick_start,
                    tick_end + 0.5 * hist_xtick_step,
                    hist_xtick_step,
                ),
                10,
            )

    for i, feature in enumerate(feature_order):
        for j, phenotype in enumerate(phenotype_order):
            ax_hist = axes[i, j, 0]
            ax_scatter = axes[i, j, 1]

            result = data.get(
                phenotype,
                {},
            ).get(feature)

            if result is None:
                ax_hist.set_visible(False)
                ax_scatter.set_visible(False)
                continue

            feature_label = feature_labels.get(
                feature,
                feature,
            )

            environment = result["environment"]

            environment_label = environment_labels.get(
                environment,
                environment,
            )

            foreground = float(
                result["foreground_r"]
            )

            display_r = foreground
            control = result["permutation_r_list"]
            p_value = result.get("p_value", None)
            direction = result.get("direction", None)

            if direction is None:
                control_array = _as_numeric_array(control)
                if control_array.size:
                    direction = (
                        "right"
                        if foreground >= np.median(control_array)
                        else "left"
                    )
                else:
                    direction = "right"

            direction_sign = 1 if direction == "right" else -1
            xlim = phenotype_xlim[phenotype]

            # Half violin.
            geom = _plot_prediction_violin_cell(
                ax_hist,
                control,
                foreground,
                xlim,
                half_width=half_width,
                violin_color=violin_color,
                violin_edgecolor=violin_edgecolor,
                violin_edge_lw=violin_edge_lw,
                violin_alpha=violin_alpha,
                median_lw=median_lw,
                foreground_color=foreground_color,
                foreground_lw=foreground_lw,
            )

            if hist_xtick_step is None:
                ax_hist.xaxis.set_major_locator(
                    MaxNLocator(nbins=4)
                )
            else:
                ax_hist.set_xticks(
                    phenotype_xticks[phenotype]
                )

            ax_hist.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, pos: f"{x:.2f}"
                )
            )

            ax_hist.tick_params(
                axis="x",
                labelsize=tick_labelsize,
            )

            if (
                p_value is not None
                and np.isfinite(p_value)
                and p_value < p_value_threshold
            ):
                span = xlim[1] - xlim[0]
                x_star = foreground + direction_sign * star_offset_frac * span
                x_star = np.clip(
                    x_star,
                    xlim[0] + 0.03 * span,
                    xlim[1] - 0.03 * span,
                )
                y_star = half_width * star_y_factor
                ax_hist.plot(
                    x_star,
                    y_star,
                    marker="*",
                    color="black",
                    markersize=star_size,
                    zorder=20,
                )

            if j == 0:
                ax_hist.set_yticks(
                    [half_width * 0.5]
                )

                ax_hist.set_yticklabels(
                    [control_label],
                    fontsize=group_label_fontsize,
                )

                ax_hist.tick_params(
                    axis="y",
                    length=0,
                    pad=2,
                )

                ax_hist.text(
                    row_label_x,
                    row_label_y,
                    feature_label,
                    transform=ax_hist.transAxes,
                    ha="left",
                    va="center",
                    fontsize=row_label_fontsize,
                    clip_on=False,
                )
            else:
                ax_hist.set_yticks([])

            if i == 0:
                ax_hist.set_title(
                    phenotype_labels.get(
                        phenotype,
                        phenotype,
                    ),
                    fontsize=title_fontsize,
                    pad=title_pad,
                )

            if i == nrows - 1:
                ax_hist.set_xlabel(
                    hist_axis_label,
                    fontsize=hist_axis_label_fontsize,
                    labelpad=15,
                )

            x_text, ha, y_text, bbox = (
                _pick_prediction_r_label(
                    ax_hist,
                    geom,
                    foreground,
                    xlim,
                    border_frac=r_border_frac,
                    y_try=r_y_try,
                )
            )

            ax_hist.text(
                x_text,
                y_text,
                f"{feature_label}\n"
                + fr"$r = {display_r:.3f}$",
                transform=ax_hist.get_xaxis_transform(),
                ha=ha,
                va="center",
                fontsize=stat_fontsize,
                bbox=bbox,
                clip_on=False,
            )

            # Prediction scatter.
            x_raw, y = _clean_prediction_xy(
                result["genetic_score"],
                result["phenotype_score"],
            )

            if x_raw.size:
                sd = np.std(x_raw)

                if np.isfinite(sd) and sd > 0:
                    x = (
                        x_raw
                        - np.mean(x_raw)
                    ) / sd
                else:
                    x = np.zeros_like(
                        x_raw
                    )
            else:
                x = x_raw

            ax_scatter.scatter(
                x,
                y,
                alpha=scatter_alpha,
                s=scatter_s,
                linewidths=0,
                color=scatter_color,
                zorder=2,
            )

            if x.size >= 2 and np.std(x) > 0:
                slope, intercept = np.polyfit(
                    x,
                    y,
                    1,
                )

                x_line = np.linspace(
                    x.min(),
                    x.max(),
                    200,
                )

                ax_scatter.plot(
                    x_line,
                    slope * x_line + intercept,
                    color=foreground_color,
                    linestyle="--",
                    linewidth=regression_lw,
                    zorder=3,
                )

            xa, ya, ha_n, va_n = (
                _pick_prediction_corner(
                    ax_scatter,
                    x,
                    y,
                    dedup_r=n_dedup_r,
                    corner_pad=n_corner_pad,
                )
            )

            ax_scatter.text(
                xa,
                ya,
                fr"$n = {x.size}$",
                transform=ax_scatter.transAxes,
                ha=ha_n,
                va=va_n,
                fontsize=stat_fontsize,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": n_bbox_alpha,
                    "pad": n_bbox_pad,
                },
                clip_on=False,
            )

            ax_scatter.text(
                0.5,
                scatter_xlabel_y,
                f"{environment_label} score\n"
                f"({feature_label} kin.)",
                transform=ax_scatter.transAxes,
                ha="center",
                va="top",
                fontsize=axis_label_fontsize,
                clip_on=False,
            )

            ax_scatter.text(
                scatter_ylabel_x,
                0.5,
                phenotype_axis_label,
                transform=ax_scatter.transAxes,
                ha="right",
                va="center",
                rotation=90,
                fontsize=axis_label_fontsize,
                clip_on=False,
            )

            ax_scatter.set_xlabel("")
            ax_scatter.set_ylabel("")

            ax_scatter.tick_params(
                axis="both",
                labelsize=tick_labelsize,
            )

    _draw_prediction_genome_lines(
        fig,
        axes,
        data,
        phenotype_order,
        feature_order,
        genome_color=genome_color,
        genome_lw=genome_lw,
        label_fontsize=stat_fontsize,
        label_y=genome_label_y,
        label_dx_frac=genome_label_dx_frac,
    )

    if save is not None:
        fig.savefig(
            save,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="none",
            edgecolor="none",
        )

    if show:
        plt.show()

    return fig, axes

if __name__ == "__main__":
    # # -------------------------------
    # partition_dir = os.path.join(results_dir, "partition")
    # os.makedirs(partition_dir, exist_ok=True)

    # from src.partition import run_partition_contrast

    # contrast_id = 'proximal_ACR_vs_proximal'
    # contrast_json = contrasts_settings[contrast_id]
    # context_df = datasets[contrast_json['dataset']]
    # contrast_dir = os.path.join(partition_dir, f"{contrast_id}")
    # landraces_df = pd.read_csv(os.path.join(
    #     data_dir, "source", "landraces_metadata.csv"))
    # vcf_file = os.path.join(
    #     data_dir, "source", "landraces_GxE_miss0.5_maf0.05_het0.05.vcf.gz")
    # predict_sample_ids = pd.read_csv(os.path.join(
    #     data_dir, "source", "Bernardino_2024_GxE_phenotypes.csv"))['LIB'].to_list()

    # run_partition_contrast(
    #     contrast_id,
    #     contrast_json,
    #     context_df,
    #     landraces_df,
    #     env_id_list,
    #     predict_sample_ids,
    #     vcf_file,
    #     contrast_dir,
    #     match_controls=("MAF", "Recomb", "Pi", "Gene"),
    #     n_perm=1000,
    #     seed=42,
    #     use_complement=False,       # True: analyze the smaller complement
    #     sample_from_all_sites=False,  # True: random sets may overlap foreground
    #     grm_n_jobs=80,
    #     reml_n_jobs=80,
    #     blup_n_jobs=80,
    #     score_n_jobs=40,
    #     debug=False,
    #     overwrite=False,
    #     cleanup_grms=None,
    #     cleanup_variant_sets=None,
    #     pruned_vcf_file=None,  # or path to an LD-pruned training VCF
    #     n_pcs="auto",
    #     auto_max_pcs=20,
    # )

    pass
