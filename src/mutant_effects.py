"""Infer single-mutation MPRA effects and summarize TSS-centered profiles.

Raw barcode tables retain the original additive sparse least-squares model.
Importing this module performs no fitting, plotting or file operations.
"""

import os
import re
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.sparse.linalg import lsqr
from scipy.stats import binom
from tqdm.auto import tqdm


def _save_csv(frame, path):
    """Write a CSV or CSV.gz through an atomic replacement."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f'{path}.tmp'
    frame.to_csv(temporary, index=False, compression='gzip' if path.endswith('.gz') else None)
    os.replace(temporary, path)


def _is_true(s):
    return s if pd.api.types.is_bool_dtype(s) else s.astype(str).str.lower().isin(["true", "1", "t", "yes"])

def choose_barcode_col(df, gene=None):
    if gene == "SBPase" and "bc_x" in df.columns:
        return "bc_x"
    if "bc" in df.columns:
        return "bc"
    if "bc_x" in df.columns:
        return "bc_x"
    raise ValueError("No barcode column found. Expected 'bc' or 'bc_x'.")

def choose_y_log2(df):
    # mean_lrr_all is the all-replicate mean in natural-log units in 3B/3C.
    if "mean_lrr_all" in df.columns:
        return pd.to_numeric(df["mean_lrr_all"], errors="coerce") / np.log(2), "mean_lrr_all / ln(2)"
    if "mean_lrr_loge" in df.columns:
        return pd.to_numeric(df["mean_lrr_loge"], errors="coerce") / np.log(2), "mean_lrr_loge / ln(2)"
    if "mean_lrr_log2" in df.columns:
        return pd.to_numeric(df["mean_lrr_log2"], errors="coerce"), "mean_lrr_log2"
    raise ValueError("No usable log-read-ratio column found.")

def split_mutations(value):
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s == "" or s.lower() == "wt":
        return []
    return [x.strip() for x in s.split(":") if x.strip() and x.strip().lower() != "wt"]

def parse_tss_pos(mut_tss0):
    m = re.search(r"-?\d+", str(mut_tss0))
    return float(m.group()) if m else np.nan

def classify_mutation(mutation_id):
    m = str(mutation_id)
    if m.startswith("I"):
        return "insertion"
    if "_" in m:
        return "deletion"
    return "substitution"

def build_sparse_design(df, mutation_col="muts_oprc"):
    mut_to_col = {}
    rows = []
    cols = []

    for row_i, muts in enumerate(df[mutation_col].map(split_mutations)):
        for mut in muts:
            col_i = mut_to_col.setdefault(mut, len(mut_to_col))
            rows.append(row_i)
            cols.append(col_i)

    if not mut_to_col:
        raise ValueError("No non-WT mutations found after filtering.")

    data = np.ones(len(rows), dtype=np.float64)
    X = coo_matrix((data, (rows, cols)), shape=(len(df), len(mut_to_col))).tocsr()
    X.sum_duplicates()
    X.data = np.ones_like(X.data)
    mutations = np.array(sorted(mut_to_col, key=mut_to_col.get), dtype=object)
    return X, mutations

def build_annotation(df, mutation_col="muts_oprc", tss_col="mut_tss0"):
    records = []
    seen = set()
    mismatch_rows = 0

    for mut_value, tss_value in zip(df[mutation_col], df[tss_col]):
        muts = split_mutations(mut_value)
        tsses = split_mutations(tss_value)
        if len(muts) != len(tsses):
            mismatch_rows += 1
        for mut, tss in zip(muts, tsses):
            if mut in seen:
                continue
            seen.add(mut)
            records.append({"mutation_id": mut, "mut_tss0": tss, "tss_pos": parse_tss_pos(tss)})

    annot = pd.DataFrame.from_records(records)
    if mismatch_rows:
        print(f"Warning: {mismatch_rows} rows had mismatched muts_oprc/mut_tss0 counts; matched pairs were kept.")
    return annot

def infer_mutation_effects(csv_path, gene, mutation_col="muts_oprc", min_barcodes=1,
                           atol=1e-8, btol=1e-8, iter_lim=3000):
    """Estimate mutation coefficients from barcode-level expression.

    Parameters
    ----------
    csv_path : str
        Original CSV/CSV.gz with mutation combinations and log read ratios.
    gene : str
        Gene label; SBPase uses bc_x when available.
    mutation_col : str
        Colon-delimited construct mutation column; default muts_oprc.
    min_barcodes : int
        Minimum supporting barcodes for exported effects; default 1.
    atol, btol : float
        LSQR absolute/relative convergence tolerances; defaults 1e-8.
    iter_lim : int
        Maximum LSQR iterations; default 3000.

    Returns
    -------
    pandas.DataFrame
        Mutation effect in log2 expression units, support, type and relative TSS
        position. Attributes include fitting/filtering diagnostics. Matching
        barcodes, permitted mutations and nonmissing expression are retained;
        repeated barcode IDs retain the first row after barcode sorting.
    """
    raw = pd.read_csv(csv_path)
    barcode_col = choose_barcode_col(raw, gene=gene)
    y_log2, y_source = choose_y_log2(raw)

    keep = _is_true(raw["matching_bcs"]) & (~_is_true(raw["mut_forbidden"])) & y_log2.notna()
    df = raw.loc[keep].copy()
    df["_y_log2"] = y_log2.loc[keep].to_numpy()

    if not df[barcode_col].is_unique:
        before = len(df)
        df = df.sort_values(barcode_col).drop_duplicates(barcode_col, keep="first")
        print(f"{gene}: dropped {before - len(df)} duplicated barcode rows from {barcode_col}.")

    X, mutations = build_sparse_design(df, mutation_col=mutation_col)
    y = df["_y_log2"].to_numpy(dtype=np.float64)

    intercept = csr_matrix(np.ones((X.shape[0], 1), dtype=np.float64))
    design = hstack([intercept, X], format="csr")
    fit = lsqr(design, y, atol=atol, btol=btol, iter_lim=iter_lim)
    if fit[1] not in [1, 2, 4, 5]:
        raise RuntimeError(f"{gene}: LSQR did not converge (istop={fit[1]}, iterations={fit[2]}).")
    coef = fit[0][1:]

    counts = np.asarray(X.sum(axis=0)).ravel()
    mean_lrr_containing = np.asarray(X.T @ y).ravel() / np.maximum(counts, 1)

    effects = pd.DataFrame({
        "gene": gene,
        "mutation_id": mutations,
        "effect": coef,
        "n_barcodes": counts.astype(int),
        "mean_lrr_containing_log2": mean_lrr_containing,
    })
    effects["mutation_type"] = effects["mutation_id"].map(classify_mutation)

    annot = build_annotation(df, mutation_col=mutation_col, tss_col="mut_tss0")
    effects = effects.merge(annot, on="mutation_id", how="left")
    effects = effects.loc[effects["n_barcodes"] >= min_barcodes].copy()

    print(
        f"{gene}: rows={len(raw):,}, kept_barcodes={len(df):,}, "
        f"mutations={len(effects):,}, y={y_source}, lsqr_istop={fit[1]}, lsqr_iter={fit[2]}"
    )
    effects = effects.sort_values(["tss_pos", "mutation_id"]).reset_index(drop=True)
    effects.attrs['fit_summary'] = dict(gene=gene, raw_rows=len(raw), kept_barcodes=len(df),
                                       mutations=len(effects), mapped_to_tss=int(effects['tss_pos'].notna().sum()),
                                       min_pos=effects['tss_pos'].min(), max_pos=effects['tss_pos'].max(),
                                       mean_effect=effects['effect'].mean(), y_source=y_source,
                                       lsqr_istop=fit[1], lsqr_iterations=fit[2],
                                       residual_norm=fit[3], atol=atol, btol=btol, iter_lim=iter_lim,
                                       min_barcodes=min_barcodes)
    return effects

def rolling_mean_effect(effect_df, x_min=-2000, x_max=250, window_bp=100, step_bp=10, min_obs=5):
    rows = []
    half = window_bp / 2
    data = effect_df.dropna(subset=["tss_pos", "effect"])
    centers = np.arange(x_min, x_max + step_bp, step_bp)

    for center in centers:
        left = center - half
        right = center + half
        vals = data.loc[(data["tss_pos"] >= left) & (data["tss_pos"] < right), "effect"]
        if len(vals) >= min_obs:
            rows.append({
                "gene": data["gene"].iloc[0],
                "center": center,
                "win_start": left,
                "win_end": right,
                "mean_effect": vals.mean(),
                "variance": vals.var(ddof=0),
                "sd": vals.std(ddof=0),
                "n": len(vals),
            })
    return pd.DataFrame(rows)

def concat_effects(all_effects):
    """Turn all_effects dict into a single tidy DataFrame."""
    dfs = []
    for gene, df in all_effects.items():
        tmp = df.copy()
        if "gene" not in tmp.columns:
            tmp["gene"] = gene
        dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True)

def add_high_effect_labels(effect_df, high_effect_quantile=0.95, threshold_mode="per_gene"):
    """
    Label high-effect mutations by absolute effect size.

    threshold_mode:
    - "per_gene": use the within-gene absolute-effect quantile.
    - "pooled": use a common absolute-effect threshold across genes.
    """
    df = effect_df.dropna(subset=["effect", "tss_pos"]).copy()
    df["abs_effect"] = df["effect"].abs()

    if threshold_mode == "per_gene":
        df["high_effect_threshold"] = df.groupby("gene")["abs_effect"].transform(
            lambda s: s.quantile(high_effect_quantile)
        )
    elif threshold_mode == "pooled":
        df["high_effect_threshold"] = df["abs_effect"].quantile(high_effect_quantile)
    else:
        raise ValueError("threshold_mode must be 'per_gene' or 'pooled'.")

    df["is_high_effect"] = df["abs_effect"] >= df["high_effect_threshold"]
    return df

def compute_high_effect_probability_windows(
    all_effects,
    x_min=-5000,
    x_max=1000,
    window_bp=500,
    step_bp=10,
    high_effect_quantile=0.95,
    threshold_mode="per_gene",
    min_obs=5,
    alpha=0.5,
):
    """
    Compute a probability-based high-effect mutation enrichment index for each TSS-centered window.

    The smoothed probability uses a beta-binomial/Laplace style estimate:
        p_window = (k + alpha) / (n + 2 * alpha)
    where k is high-effect mutation count and n is all mutation count in the window.
    """
    effect_df = concat_effects(all_effects)
    labeled_df = add_high_effect_labels(
        effect_df,
        high_effect_quantile=high_effect_quantile,
        threshold_mode=threshold_mode,
    )

    rows = []
    half_window = window_bp / 2
    centers = np.arange(x_min, x_max + step_bp, step_bp)

    for gene, gene_df in labeled_df.groupby("gene", sort=False):
        gene_df = gene_df.dropna(subset=["tss_pos", "effect"]).copy()
        background_n = len(gene_df)
        background_k = int(gene_df["is_high_effect"].sum())
        background_probability = (background_k + alpha) / (background_n + 2 * alpha)

        for center in centers:
            left = center - half_window
            right = center + half_window
            win_df = gene_df.loc[(gene_df["tss_pos"] >= left) & (gene_df["tss_pos"] < right)]
            n = len(win_df)
            if n < min_obs:
                continue

            k = int(win_df["is_high_effect"].sum())
            p_window = (k + alpha) / (n + 2 * alpha)
            relative_probability_index = p_window / background_probability
            binom_tail_p = binom.sf(k - 1, n, background_probability) if k > 0 else 1.0
            signed_log10_binom_p = -np.log10(max(binom_tail_p, np.finfo(float).tiny))
            if p_window < background_probability:
                signed_log10_binom_p *= -1

            rows.append({
                "gene": gene,
                "center": center,
                "win_start": left,
                "win_end": right,
                "n_mutations": n,
                "n_high_effect": k,
                "high_effect_probability": p_window,
                "background_probability": background_probability,
                "relative_probability_index": relative_probability_index,
                "binom_tail_p": binom_tail_p,
                "signed_log10_binom_p": signed_log10_binom_p,
                "high_effect_quantile": high_effect_quantile,
                "threshold_mode": threshold_mode,
            })

    window_probability_df = pd.DataFrame(rows)
    return labeled_df, window_probability_df

def combine_gene_high_effect_index(window_probability_df, alpha=0.5):
    """
    Combine the three genes into one TSS-window index.

    Each gene first defines high-effect mutations by its own abs(effect) threshold.
    Then, for each window, counts are pooled across genes:
        combined_p = (sum(high_effect_count) + alpha) / (sum(mutation_count) + 2 * alpha)
        combined_background_p = weighted mean of gene-level background probabilities
        combined_index = combined_p / combined_background_p

    This keeps gene-specific effect-scale differences out of the definition, while
    producing one unified probability index across the three promoters.
    """
    rows = []
    for center, g in window_probability_df.groupby("center", sort=True):
        total_n = int(g["n_mutations"].sum())
        total_k = int(g["n_high_effect"].sum())
        if total_n == 0:
            continue

        combined_background_p = np.average(
            g["background_probability"].to_numpy(dtype=float),
            weights=g["n_mutations"].to_numpy(dtype=float),
        )
        combined_p = (total_k + alpha) / (total_n + 2 * alpha)
        combined_index = combined_p / combined_background_p
        binom_tail_p = binom.sf(total_k - 1, total_n, combined_background_p) if total_k > 0 else 1.0
        signed_log10_binom_p = -np.log10(max(binom_tail_p, np.finfo(float).tiny))
        if combined_p < combined_background_p:
            signed_log10_binom_p *= -1

        rows.append({
            "center": center,
            "n_mutations": total_n,
            "n_high_effect": total_k,
            "high_effect_probability": combined_p,
            "background_probability": combined_background_p,
            "relative_probability_index": combined_index,
            "binom_tail_p": binom_tail_p,
            "signed_log10_binom_p": signed_log10_binom_p,
            "n_genes_present": g["gene"].nunique(),
        })
    return pd.DataFrame(rows)


def prepare_mutation_effects(gene_csvs, output_dir, min_barcodes=1,
                             atol=1e-8, btol=1e-8, iter_lim=3000,
                             force_recompute=False):
    """Fit or reuse single-mutation effects for a collection of MPRA genes.

    Parameters
    ----------
    gene_csvs : dict
        Gene names mapped to original barcode CSV/CSV.gz paths.
    output_dir : str
        Destination for per-gene effects and fitting-summary CSV.gz.
    min_barcodes, atol, btol, iter_lim : int or float
        Supporting-barcode cutoff and sparse least-squares settings.
    force_recompute : bool
        Refit after input/parameter changes when True; default False. Otherwise
        reuse a complete effects/fit-summary pair with matching saved settings.

    Returns
    -------
    dict
        effect_files, fit_summary and summary_file. Genes are fitted sequentially
        with progress and timing; original input files remain unchanged.
    """
    effect_files, summaries = {}, []
    for gene, csv in tqdm(gene_csvs.items(), desc='Single-mutation effects'):
        start = perf_counter()
        path = f'{output_dir}/{gene}.mutation_effects.csv.gz'
        fit_file = f'{output_dir}/{gene}.fit_summary.csv.gz'
        settings = dict(input_csv=csv, min_barcodes=min_barcodes, atol=atol, btol=btol, iter_lim=iter_lim)
        reuse = not force_recompute and os.path.exists(path) and os.path.exists(fit_file)
        if reuse:
            saved = pd.read_csv(fit_file).iloc[0]
            reuse = all(saved[key] == value for key, value in settings.items())
        if reuse:
            summary = pd.read_csv(fit_file)
            print(f'[{datetime.now():%H:%M:%S}] Reusing {path}; input-content changes require force_recompute=True.')
        else:
            print(f'[{datetime.now():%H:%M:%S}] Fitting {gene}', flush=True)
            effects = infer_mutation_effects(csv, gene, min_barcodes=min_barcodes,
                                            atol=atol, btol=btol, iter_lim=iter_lim)
            summary = pd.DataFrame([dict(effects.attrs['fit_summary'], **{'input_csv': csv})])
            _save_csv(effects, path)
            _save_csv(summary, fit_file)
            print(f'[{datetime.now():%H:%M:%S}] {gene} completed in {perf_counter()-start:.1f}s', flush=True)
        effect_files[gene] = path
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    summary_file = f'{output_dir}/mutation_effects.fit_summary.csv.gz'
    _save_csv(summary, summary_file)
    return dict(effect_files=effect_files, fit_summary=summary, summary_file=summary_file)


def prepare_mutation_profiles(effect_files, output_dir, mean_xlim=(-2000, 250),
                               mean_window_bp=100, mean_step_bp=10, min_obs=5,
                               index_xlim=(-5000, 1000), index_window_bp=500,
                               index_step_bp=10, high_effect_quantile=0.95,
                               alpha=0.5, force_recompute=False):
    """Prepare rolling mean effects and the combined high-effect index.

    Parameters
    ----------
    effect_files : dict
        Gene names mapped to saved single-mutation effect CSV.gz files.
    output_dir : str
        Output directory, including combined_high_effect_index.csv.
    mean_xlim, index_xlim : tuple
        Inclusive window-center ranges in bp relative to TSS.
    mean_window_bp, mean_step_bp : int
        Mean-effect window length/step; defaults 100/10 bp.
    min_obs : int
        Minimum mutations per gene window; default 5.
    index_window_bp, index_step_bp : int
        High-effect index window length/step; defaults 500/10 bp.
    high_effect_quantile : float
        Within-gene absolute-effect threshold; default 0.95, including ties.
    alpha : float
        Symmetric probability pseudocount; default 0.5. Gene backgrounds include
        all valid positioned mutations, not only mutations in the plot range.
        Window counts determine gene weights in the combined background.
    force_recompute : bool
        Rebuild after changes when True; otherwise reuse complete outputs with
        matching saved parameters. Changed effect-table contents require force.

    Returns
    -------
    dict
        Saved file paths for rolling means, labeled effects, per-gene windows,
        combined index and summaries. Binomial diagnostics retain the legacy
        calculation but are not used as enrichment/depletion significance tests.
    """
    if min(mean_window_bp, mean_step_bp, index_window_bp, index_step_bp, min_obs) < 1:
        raise ValueError('Window sizes, steps and minimum observations must be positive.')
    if not 0 < high_effect_quantile < 1 or alpha <= 0:
        raise ValueError('Require a quantile in (0, 1) and positive pseudocount.')
    files = dict(mean=f'{output_dir}/mean_mutation_effects.windows.csv.gz',
                 labeled=f'{output_dir}/mutation_effects.high_effect_labels.csv.gz',
                 windows=f'{output_dir}/high_effect_probability.windows.csv.gz',
                 combined=f'{output_dir}/combined_high_effect_index.csv',
                 summary=f'{output_dir}/high_effect.summary.csv.gz',
                 parameters=f'{output_dir}/mutation_profiles.parameters.csv.gz')
    settings = dict(mean_x_min=mean_xlim[0], mean_x_max=mean_xlim[1],
                    mean_window_bp=mean_window_bp, mean_step_bp=mean_step_bp, min_obs=min_obs,
                    index_x_min=index_xlim[0], index_x_max=index_xlim[1],
                    index_window_bp=index_window_bp, index_step_bp=index_step_bp,
                    high_effect_quantile=high_effect_quantile, alpha=alpha)
    reuse = not force_recompute and all(os.path.exists(path) for path in files.values())
    if reuse:
        saved = pd.read_csv(files['parameters']).iloc[0]
        reuse = all(saved[key] == value for key, value in settings.items())
    if reuse:
        print(f'[{datetime.now():%H:%M:%S}] Reusing mutation profiles; changed effect-table contents require force_recompute=True.')
        return files
    print(f'[{datetime.now():%H:%M:%S}] Summarizing mutation profiles')
    effects = {gene: pd.read_csv(path) for gene, path in effect_files.items()}
    means = pd.concat([rolling_mean_effect(frame, x_min=mean_xlim[0], x_max=mean_xlim[1],
                                           window_bp=mean_window_bp, step_bp=mean_step_bp, min_obs=min_obs)
                       for frame in effects.values()], ignore_index=True)
    labeled, windows = compute_high_effect_probability_windows(
        effects, x_min=index_xlim[0], x_max=index_xlim[1], window_bp=index_window_bp,
        step_bp=index_step_bp, high_effect_quantile=high_effect_quantile,
        threshold_mode='per_gene', min_obs=min_obs, alpha=alpha)
    combined = combine_gene_high_effect_index(windows, alpha=alpha)
    summary = labeled.groupby('gene', sort=True).agg(
        mutations=('mutation_id', 'count'), high_effect_mutations=('is_high_effect', 'sum'),
        abs_effect_threshold=('high_effect_threshold', 'first')).reset_index()
    for key, frame in [('mean', means), ('labeled', labeled), ('windows', windows),
                       ('combined', combined), ('summary', summary),
                       ('parameters', pd.DataFrame([settings]))]:
        _save_csv(frame, files[key])
    return files


def plot_mean_mutation_effects(windows_csv, output_prefix, xlim=(-2000, 250),
                               spread='variance', figsize=(8.5, 6.5)):
    """Plot original-style rolling effect curves for three MPRA genes.

    Parameters
    ----------
    windows_csv : str
        Saved rolling means with gene, center, mean_effect, variance and sd.
    output_prefix : str
        PDF/PNG/actual plotted-data CSV.gz prefix.
    xlim : tuple
        Relative TSS position range in bp.
    spread : str
        Band width column; default variance exactly retains the original plot.
        The band is mean ± variance, not a confidence interval.
    figsize : tuple
        Figure dimensions in inches; default (8.5, 6.5).

    Returns
    -------
    dict
        figure, axes, plot_data and files. Shared style is not reset.
    """
    windows = pd.read_csv(windows_csv)
    genes = ['PsbS', 'Raf1', 'SBPase']
    colors = {'PsbS': '#2b6cb0', 'Raf1': '#6b8e23', 'SBPase': '#b83280'}
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
    frames = []
    for ax, gene in zip(axes, genes):
        w = windows.loc[(windows['gene'] == gene) & windows['center'].between(*xlim)].sort_values('center').copy()
        if w.empty:
            raise ValueError(f'No rolling-effect windows for {gene}.')
        band = w[spread].fillna(0)
        w['BAND_LOWER'], w['BAND_UPPER'] = w['mean_effect'] - band, w['mean_effect'] + band
        w['BAND_DEFINITION'] = f'mean_effect ± {spread}'
        ax.plot(w['center'], w['mean_effect'], color=colors[gene], lw=1.8)
        ax.fill_between(w['center'], w['BAND_LOWER'], w['BAND_UPPER'], color=colors[gene], alpha=0.20, lw=0)
        ax.axhline(0, color='0.25', lw=0.8)
        ax.axvline(0, color='0.25', lw=0.8)
        ax.axvspan(-400, xlim[1], color='0.9', alpha=0.45, zorder=-10)
        ax.text(0.01, 0.82, gene, transform=ax.transAxes, fontweight='bold')
        ax.set_ylabel('Avg. Emut')
        frames.append(w)
    axes[-1].set_xlim(*xlim)
    axes[-1].set_xlabel('Position relative to TSS (bp)')
    fig.suptitle('Rolling average effect of all mutations')
    fig.tight_layout()
    data = pd.concat(frames, ignore_index=True)
    return _export_mutation_plot(fig, axes, data, output_prefix)


def plot_combined_high_effect_index(index_csv, output_prefix, xlim=(-5000, 1000),
                                    figsize=(8.5, 3.2)):
    """Plot the saved combined high-effect probability index.

    Parameters
    ----------
    index_csv : str
        Saved combined index containing center and relative_probability_index.
    output_prefix : str
        PDF/PNG/actual plotted-data CSV.gz prefix.
    xlim : tuple
        Relative TSS position range in bp; default (-5000, 1000).
    figsize : tuple
        Canvas size in inches; default (8.5, 3.2).

    Returns
    -------
    dict
        figure, axes, plot_data and files. Index=1 marks background probability;
        no binomial significance is inferred or displayed.
    """
    data = pd.read_csv(index_csv).sort_values('center')
    data = data.loc[data['center'].between(*xlim)].copy()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data['center'], data['relative_probability_index'], color='#111111', lw=2,
            label='Combined high-|effect| index')
    ax.axhline(1, color='0.35', lw=0.9, ls='--')
    ax.axvline(0, color='0.25', lw=0.8)
    ax.axvspan(-400, xlim[1], color='0.9', alpha=0.45, zorder=-10)
    ax.set_xlim(*xlim)
    ax.set_xlabel('Position relative to TSS (bp)')
    ax.set_ylabel('Relative probability index')
    ax.set_title('Window enrichment of high-|effect| mutations')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    return _export_mutation_plot(fig, ax, data, output_prefix)


def _export_mutation_plot(fig, axes, data, output_prefix):
    """Save a figure and its actual plotted values without modifying style."""
    files = {key: f'{output_prefix}.{ext}' for key, ext in [('pdf', 'pdf'), ('png', 'png'), ('csv', 'csv.gz')]}
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    fig.savefig(files['pdf'], bbox_inches='tight')
    fig.savefig(files['png'], dpi=300, bbox_inches='tight')
    _save_csv(data, files['csv'])
    return dict(figure=fig, axes=axes, plot_data=data, files=files)
