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
    if output is not None:
        if output_format == "json":
            if os.path.isfile(output):
                print(f"Existing output file found, skipping: {output}")
                return

        elif output_format == "npy":
            if os.path.isfile(f"{output}.foreground.npy"):
                print(f"Existing output file found, skipping: {output}")
                return

    match_definitions = {
        "MAF": ["maf_0.05_0.1", "maf_0.1_0.2", "maf_0.2_0.3", "maf_0.3_0.4", "maf_0.4_0.5"],
        "Recomb": ["low_recombination", "med_recombination", "high_recombination"],
        "Pi": ["low_pi_genome", "med_pi_genome", "high_pi_genome"],
        "Gene": ["genic_2kb", "non_genic"],
    }

    match_controls = tuple(match_controls)

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

    # Precompute foreground/background members of each foreground-containing stratum.
    strata = []
    for s in np.unique(stratum_code[fg]):
        mask = stratum_code == s
        fg_idx = np.flatnonzero(mask & fg)
        bg_idx = np.flatnonzero(mask & ~fg)
        strata.append({
            "stratum": int(s),
            "need": len(fg_idx),
            "available": len(bg_idx),
            "bg_idx": bg_idx,
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

    all_bg_idx = np.flatnonzero(~fg)
    foreground_idx = np.flatnonzero(fg).astype(np.int32)
    perm_mat = np.empty((n_perm, n_fg), dtype=np.int32)
    fallback_counts = np.zeros(n_perm, dtype=np.int32)

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
            candidates = x["bg_idx"]
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

            fallback_pool = all_bg_idx
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

    foreground_unclassified = {
        name: int((fg & (code == len(match_definitions[name]))).sum())
        for name, code in zip(match_controls, group_codes)
    }

    metadata = {
        "n_total_sites": n_sites,
        "n_foreground": n_fg,
        "n_background": n_sites - n_fg,
        "foreground": foreground_name,
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

    if cleanup_grms is None:
        cleanup_grms = not debug

    if cleanup_variant_sets is None:
        cleanup_variant_sets = not debug

    # ================================================================
    # 1. Directory structure
    # ================================================================

    work_dir = os.path.join(results_dir, "_work")
    final_dir = os.path.join(results_dir, "final")

    final_files = [
        "reml_summary.csv",
        "score_summary.csv",
        "matching_metadata.json",
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

        if has_all_site:
            print(
                f"\n[{contrast_id}] Final outputs already exist, skipping contrast.")
            return {
                "contrast_id": contrast_id,
                "contrast_dir": results_dir,
                "final_dir": final_dir,
                "reml_df": reml_df,
                "score_df": score_df,
            }

        print(
            f"\n[{contrast_id}] Existing final outputs do not contain "
            "ALL_SITE; continuing analysis."
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

    # ================================================================
    # 3. Build matched variant sets
    # ================================================================

    print(f"\n[{contrast_id}] Building matched variant sets...")

    build_variant_sets(
        context_df,
        match_controls=match_controls,
        n_perm=n_perm,
        seed=seed,
        output=variant_sets_prefix,
    )

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
    # 6. REML
    # ================================================================

    print(f"\n[{contrast_id}] Running REML...")

    reml_df = run_reml(
        grm_dir=grm_dir,
        pheno_df=landraces_df,
        pheno_cols=pheno_cols,
        sample_col="LIB",
        n_jobs=reml_n_jobs,
    )

    reml_df = reml_df[
        [
            "set",
            "phenotype",
            "perm_index",
            "is_foreground",
            "is_all_site",
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
    ].copy()

    # ================================================================
    # 7. BLUP
    # ================================================================

    print(f"\n[{contrast_id}] Running BLUPs...")

    run_blups(
        grm_dir=grm_dir,
        pheno_cols=pheno_cols,
        n_jobs=blup_n_jobs,
    )

    # ================================================================
    # 8. Prediction
    # ================================================================

    print(f"\n[{contrast_id}] Predicting GxE scores...")

    score_df = predict_scores(
        grm_dir=grm_dir,
        prediction_sample_ids=predict_sample_ids,
        pheno_cols=pheno_cols,
        n_jobs=score_n_jobs,
    )

    # ================================================================
    # 9. Save publication-relevant results
    # ================================================================

    reml_final_file = os.path.join(
        final_dir,
        "reml_summary.csv",
    )

    score_final_file = os.path.join(
        final_dir,
        "score_summary.csv",
    )

    reml_df.to_csv(
        reml_final_file,
        index=False,
    )

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

    # ================================================================
    # 10. Run metadata / provenance
    # ================================================================

    run_metadata = {
        "contrast_id": contrast_id,
        "contrast_settings": contrast_json,
        "results_dir": results_dir,
        "n_context_sites": int(len(context_df)),
        "n_foreground_sites": int(
            context_df["foreground"].fillna(False).sum()
        ),
        "match_controls": list(match_controls),
        "n_permutations": int(n_perm),
        "seed": int(seed),
        "phenotypes": list(pheno_cols),
        "vcf_file": os.path.abspath(vcf_file),
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
        "debug": debug,
        "final_outputs": {
            "reml_summary": reml_final_file,
            "score_summary": score_final_file,
            "matching_metadata": matching_metadata_final,
        },
    }

    with open(
        os.path.join(
            final_dir,
            "run_metadata.json",
        ),
        "w",
    ) as f:
        json.dump(
            run_metadata,
            f,
            indent=2,
        )

    # ================================================================
    # 11. Cleanup
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
    #     grm_n_jobs=80,
    #     reml_n_jobs=80,
    #     blup_n_jobs=80,
    #     score_n_jobs=40,
    #     debug=False,
    #     overwrite=False,
    #     cleanup_grms=None,
    #     cleanup_variant_sets=None,
    # )

    pass
