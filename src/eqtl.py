"""Prepare empirical cis-eQTL summaries and individual-gene landscape figures."""

import os
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm


def _write_csv(df, path):
    """Atomically save a compressed table without its dataframe index."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f'{path}.tmp'
    df.to_csv(temporary, index=False, compression='gzip')
    os.replace(temporary, path)


def _message(message):
    print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)


def prepare_cis_eqtl_fdr(pval_csv, output_dir, fdr_threshold=0.05, force_recompute=False):
    """
    Reconstruct the historical gene-level empirical FDR curve and P threshold.

    Parameters
    ----------
    pval_csv : str
        Compressed CSV with pval and type (obs or perm). Each value is the
        minimum nominal P across one gene's cis region, observed or permuted.
    output_dir : str
        Directory for cis_eQTL.fdr.csv.gz and cis_eQTL.threshold.csv.gz.
    fdr_threshold : float, default 0.05
        Empirical FDR at which to interpolate the nominal P threshold.
    force_recompute : bool, default False
        Reuse both saved tables unless explicitly recomputing. Input or
        parameter changes require explicit recomputation, including downstream.

    Returns
    -------
    dict
        curve and summary dataframes, pvalue_threshold, and files. Tables are
        written atomically. No association analyses or permutations are run.

    Notes
    -----
    At each observed row in ascending P order, FDR is the cumulative fraction
    of permutations divided by the cumulative fraction of observations.
    Linear interpolation of FDR to P follows the historical implementation;
    no smoothing, monotonic adjustment, or clipping is applied.
    """
    start = perf_counter()
    files = {'curve': f'{output_dir}/cis_eQTL.fdr.csv.gz',
             'summary': f'{output_dir}/cis_eQTL.threshold.csv.gz'}
    if not force_recompute and all(os.path.exists(p) for p in files.values()):
        curve, summary = (pd.read_csv(files[k]) for k in ['curve', 'summary'])
        _message(f'Reusing empirical FDR: {files["curve"]}')
    else:
        _message('Calculating gene-level empirical cis-eQTL FDR')
        values = pd.read_csv(pval_csv, usecols=['pval', 'type'])
        if not values['type'].isin(['obs', 'perm']).all():
            raise ValueError('P-value type must be obs or perm.')
        if not np.isfinite(values['pval']).all() or not values['pval'].between(0, 1).all():
            raise ValueError('P values must be finite and within [0, 1].')
        # Already sorted inputs retain their historical ordering of tied values.
        if not values['pval'].is_monotonic_increasing:
            values = values.sort_values('pval', ignore_index=True)
        observed = values['type'].eq('obs').to_numpy()
        n_obs, n_perm = int(observed.sum()), int((~observed).sum())
        if not n_obs or not n_perm:
            raise ValueError('Both observed and permuted minima are required.')
        obs_cumulative = np.cumsum(observed)
        perm_cumulative = np.cumsum(~observed)
        curve = pd.DataFrame({'pval': values.loc[observed, 'pval'].to_numpy(),
                              'fdr': (perm_cumulative[observed] / n_perm) /
                                     (obs_cumulative[observed] / n_obs)})
        curve['NEG_LOG10_P'] = -np.log10(curve['pval'])
        threshold = float(interp1d(curve['fdr'], curve['pval'], kind='linear',
                                  bounds_error=False, fill_value='extrapolate')(fdr_threshold))
        if not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError('Interpolated P threshold is not finite or within (0, 1].')
        summary = pd.DataFrame([{'FDR_THRESHOLD': fdr_threshold, 'P_VALUE_THRESHOLD': threshold,
                                 'N_OBS': n_obs, 'N_PERM': n_perm,
                                 'PERMUTATIONS_PER_GENE': n_perm / n_obs}])
        _write_csv(curve, files['curve'])
        _write_csv(summary, files['summary'])
        _message(f'Empirical FDR ready ({perf_counter() - start:.1f}s)')
    return {'curve': curve, 'summary': summary,
            'pvalue_threshold': float(summary.iloc[0]['P_VALUE_THRESHOLD']), 'files': files}


def plot_cis_eqtl_fdr(curve_csv, threshold_csv, output_prefix, figsize=(6, 6)):
    """
    Plot a saved empirical FDR curve using the historical scatter style.

    Parameters
    ----------
    curve_csv, threshold_csv : str
        Saved tables from prepare_cis_eqtl_fdr; no statistics are recomputed.
    output_prefix : str
        Prefix for PDF, PNG, and actual plotting data CSV.gz.
    figsize : tuple, default (6, 6)
        Figure width and height in inches. Existing global style is preserved.

    Returns
    -------
    dict
        figure, axes, plot_data, and files. The plotting table includes both
        threshold values; figures are exported but not displayed automatically.
    """
    import matplotlib.pyplot as plt

    _message('Plotting empirical cis-eQTL FDR')
    data = pd.read_csv(curve_csv)
    summary = pd.read_csv(threshold_csv).iloc[0]
    threshold, fdr = float(summary['P_VALUE_THRESHOLD']), float(summary['FDR_THRESHOLD'])
    data['FDR_THRESHOLD'], data['P_VALUE_THRESHOLD'] = fdr, threshold
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(data['NEG_LOG10_P'], data['fdr'], s=1)
    ax.axhline(fdr, color='r', linestyle='--', label=f'FDR Threshold: {fdr}')
    ax.axvline(-np.log10(threshold), color='g', linestyle='--',
               label=f'P-value Threshold: {threshold:.4e}')
    ax.set(title='FDR vs P-value', xlabel='-log10(P-value)', ylabel='FDR', xlim=(-0.00001, 8))
    ax.legend()
    files = _export_figure(fig, data, output_prefix)
    return {'figure': fig, 'axes': ax, 'plot_data': data, 'files': files}


def prepare_cis_eqtl_egenes(site_csv, threshold_csv, output_dir, force_recompute=False):
    """
    Identify all genes with at least one nominal P strictly below the threshold.

    Parameters
    ----------
    site_csv : str
        Complete cis-eQTL CSV containing GENE, ID, CHROM, POS and P_WALD.
        PASS_FDR is ignored; both significant and nonsignificant sites are used.
    threshold_csv : str
        Saved empirical threshold summary from prepare_cis_eqtl_fdr.
    output_dir : str
        Directory for cis_eQTL.egenes.csv.gz and cis_eQTL_gene_id_list.txt.
    force_recompute : bool, default False
        Reuse both complete outputs; changing inputs requires explicit rerun.

    Returns
    -------
    dict
        summary dataframe of eGenes, sorted gene_ids, and files. Each summary
        row records tested/significant counts, minimum P, and the lead variant.
        Both outputs are replaced atomically; the complete input is read in chunks.
    """
    files = {'summary': f'{output_dir}/cis_eQTL.egenes.csv.gz',
             'gene_list': f'{output_dir}/cis_eQTL_gene_id_list.txt'}
    if not force_recompute and all(os.path.exists(p) for p in files.values()):
        summary = pd.read_csv(files['summary'])
        with open(files['gene_list']) as handle:
            gene_ids = handle.read().splitlines()
        _message(f'Reusing {len(gene_ids):,} eGenes')
        return {'summary': summary, 'gene_ids': gene_ids, 'files': files}
    start = perf_counter()
    threshold = float(pd.read_csv(threshold_csv).iloc[0]['P_VALUE_THRESHOLD'])
    _message('Summarizing cis-eQTL genes')
    counts, leads = [], []
    reader = pd.read_csv(site_csv, usecols=['GENE', 'ID', 'CHROM', 'POS', 'P_WALD'], chunksize=300000)
    for chunk in tqdm(reader, desc='Summarizing cis-eQTL', unit='chunk'):
        if not np.isfinite(chunk['P_WALD']).all() or not chunk['P_WALD'].between(0, 1).all():
            raise ValueError('Invalid nominal P values in cis-eQTL sites.')
        chunk['SIGNIFICANT'] = chunk['P_WALD'] < threshold
        counts.append(chunk.groupby('GENE', as_index=False).agg(
            N_TESTED=('ID', 'size'), N_SIGNIFICANT=('SIGNIFICANT', 'sum')))
        leads.append(chunk.sort_values(['P_WALD', 'POS', 'ID']).drop_duplicates('GENE')
                     [['GENE', 'ID', 'CHROM', 'POS', 'P_WALD']])
    counts = pd.concat(counts).groupby('GENE', as_index=False)[['N_TESTED', 'N_SIGNIFICANT']].sum()
    leads = pd.concat(leads).sort_values(['P_WALD', 'POS', 'ID']).drop_duplicates('GENE')
    summary = counts[counts['N_SIGNIFICANT'] > 0].merge(leads, on='GENE', validate='one_to_one')
    summary = summary.rename(columns={'ID': 'LEAD_ID', 'POS': 'LEAD_POS', 'P_WALD': 'MIN_P'})
    summary = summary.sort_values('GENE', ignore_index=True)
    summary['P_VALUE_THRESHOLD'] = threshold
    gene_ids = summary['GENE'].tolist()
    _write_csv(summary, files['summary'])
    temporary = f'{files["gene_list"]}.tmp'
    with open(temporary, 'w') as handle:
        handle.write(''.join(f'{g}\n' for g in gene_ids))
    os.replace(temporary, files['gene_list'])
    _message(f'{len(gene_ids):,} eGenes saved ({perf_counter() - start:.1f}s)')
    return {'summary': summary, 'gene_ids': gene_ids, 'files': files}


def _locus_prefix(output_dir, config):
    return f'{output_dir}/{config["gene_id"]}.{config["env_id"]}'


def _valid_nominal_sites(table, label):
    """Exclude undefined Wald P values before ranking, BH or marker selection."""
    finite = np.isfinite(table['P_WALD'])
    excluded = int((~finite).sum())
    table = table.loc[finite].copy()
    if excluded:
        _message(f'{label}: excluded {excluded:,} undefined/nonfinite Wald P values; '
                 f'{len(table):,} sites retained')
    if table.empty or not table['P_WALD'].between(0, 1).all():
        raise ValueError(f'{label} must contain valid nominal P values in [0, 1].')
    return table


def prepare_flowering_gene_loci(configs, site_csv, tpm_csv, threshold_csv, vcf_file,
                                gff_file, landrace_csv, environment_csv, gwas_dir,
                                output_dir, force_recompute=False):
    """
    Prepare regional GEA/eQTL, gene models and genotype-stratified observations.

    Workflow
    --------
    Read target cis tables and TPM once, calculate BH across valid cis sites,
    process one environment GWAS at a time, match marker alleles in the indexed
    all-sample VCF, and save each gene/environment's plotting inputs.

    Parameters
    ----------
    configs : list of dict
        gene_id, gene_name and env_id are required. Optional marker_id overrides
        automatic selection; marker_index defaults to 0 and flank_size to 18000.
        Automatic candidates satisfy nominal P <= the empirical threshold and
        are ranked by environmental P, with ID breaking ties.
    site_csv, tpm_csv, threshold_csv : str
        Complete cis sites, raw gene-by-LIB TPM (first column is gene ID), and
        saved global empirical threshold. Site columns include REF and ALT.
    vcf_file : str
        Indexed all-sample biallelic VCF. Match CHROM/POS/ID/REF/ALT exactly.
    gff_file : str
        Gene/exon GFF3; parsed by src.feature in 1-based inclusive coordinates.
    landrace_csv, environment_csv : str
        Metadata with LIB/Latitude/Longitude/environment values, and env_id,
        short description and units respectively.
    gwas_dir : str
        Root containing {env}/{env}.gemma/{env}.gemma.gwas_df.h5, key df.
        Missing/nonfinite Wald P values are excluded before genome-wide
        quantiles, regional selection and matching shared markers.
    output_dir : str
        Directory for per-gene .locus.csv.gz and .summary.csv.gz tables.
    force_recompute : bool, default False
        Reuse only when every requested output exists. Changed inputs/configs
        require explicit recomputation. No GWAS is launched.

    Returns
    -------
    pandas.DataFrame
        One row per configuration with selected marker, thresholds and saved
        file paths. Saved locus rows distinguish gea, eqtl, gene, exon,
        env_box and expr_box. BH uses all cis sites with finite Wald P values
        before regional selection; N_CIS_TESTED counts these retained sites;
        boxplots retain only nonmissing 0/0 and 1/1 observations.
    """
    from cyvcf2 import VCF
    from src.feature import read_gff_file

    pairs = [(c['gene_id'], c['env_id']) for c in configs]
    if len(pairs) != len(set(pairs)):
        raise ValueError('Gene/environment configurations must be unique.')
    prefixes = [_locus_prefix(output_dir, c) for c in configs]
    if not force_recompute and all(os.path.exists(f'{p}.{suffix}.csv.gz')
                                   for p in prefixes for suffix in ['locus', 'summary']):
        _message('Reusing prepared flowering-gene loci')
        return pd.concat([pd.read_csv(f'{p}.summary.csv.gz') for p in prefixes], ignore_index=True)
    start_time = perf_counter()
    _message('Preparing flowering-gene loci')
    threshold = float(pd.read_csv(threshold_csv).iloc[0]['P_VALUE_THRESHOLD'])
    target_ids = set(c['gene_id'] for c in configs)
    parts = []
    for chunk in tqdm(pd.read_csv(site_csv, chunksize=300000), desc='Reading target cis sites', unit='chunk'):
        parts.append(chunk[chunk['GENE'].isin(target_ids)].copy())
    sites = pd.concat(parts, ignore_index=True)
    if sites.duplicated(['GENE', 'ID']).any():
        raise ValueError('Duplicate gene/variant records in cis-eQTL input.')
    tpm = pd.read_csv(tpm_csv, index_col=0)
    if not tpm.index.is_unique or not tpm.columns.is_unique:
        raise ValueError('TPM gene and sample IDs must be unique.')
    missing = target_ids - set(tpm.index)
    if missing or target_ids - set(sites['GENE']):
        raise ValueError(f'Target genes missing from TPM or cis-eQTL: {missing | (target_ids - set(sites.GENE))}')
    tpm = tpm.loc[sorted(target_ids)]
    gene_dict = read_gff_file(gff_file)['gene']
    missing = target_ids - set(gene_dict)
    if missing:
        raise ValueError(f'Target genes missing from GFF: {missing}')
    landraces = pd.read_csv(landrace_csv)
    env_info = pd.read_csv(environment_csv).set_index('env_id')
    if not landraces['LIB'].is_unique or not env_info.index.is_unique:
        raise ValueError('Landrace LIB and environment IDs must be unique.')
    per_gene = {}
    for gene_id in target_ids:
        table = sites[sites['GENE'] == gene_id].copy()
        table = _valid_nominal_sites(table, gene_id)
        table['BH_FDR'] = multipletests(table['P_WALD'], method='fdr_bh')[1]
        per_gene[gene_id] = table
    del sites, parts
    vcf = VCF(vcf_file)
    sample_index = {sample: i for i, sample in enumerate(vcf.samples)}
    required_samples = set(tpm.columns) | set(landraces['LIB'])
    if required_samples - set(sample_index):
        vcf.close()
        raise ValueError(f'Samples missing from VCF: {sorted(required_samples - set(sample_index))}')
    summaries = []
    try:
        for env_id in tqdm(list(dict.fromkeys(c['env_id'] for c in configs)), desc='Preparing environments', unit='environment'):
            if env_id not in env_info.index:
                raise ValueError(f'Environment missing from metadata: {env_id}')
            env = pd.read_hdf(f'{gwas_dir}/{env_id}/{env_id}.gemma/{env_id}.gemma.gwas_df.h5', key='df')
            env = env[['ID', 'CHROM', 'POS', 'MAF', 'P_WALD']].copy()
            if not env['ID'].is_unique:
                raise ValueError(f'Duplicate GWAS IDs for {env_id}.')
            env = _valid_nominal_sites(env, env_id)
            top1, top5 = env['P_WALD'].quantile([0.01, 0.05])
            for config in [c for c in configs if c['env_id'] == env_id]:
                gene_id = config['gene_id']
                gene = gene_dict[gene_id]
                eqtl = per_gene[gene_id]
                candidates = eqtl if config.get('marker_id') is not None else eqtl[eqtl['P_WALD'] <= threshold]
                shared = candidates.merge(env, on='ID', suffixes=('_EQTL', '_GEA'), validate='one_to_one')
                if config.get('marker_id') is not None:
                    shared = shared[shared['ID'] == config['marker_id']]
                    marker_index = 0
                else:
                    marker_index = config.get('marker_index', 0)
                shared = shared.sort_values(['P_WALD_GEA', 'ID'], ignore_index=True)
                if marker_index < 0 or marker_index >= len(shared):
                    raise ValueError(f'No requested shared marker for {gene_id}/{env_id}.')
                marker = shared.iloc[marker_index]
                chrom, pos = marker['CHROM_EQTL'], int(marker['POS_EQTL'])
                if chrom != marker['CHROM_GEA'] or pos != int(marker['POS_GEA']):
                    raise ValueError(f'GEA/eQTL coordinate mismatch: {marker.ID}')
                matches = [record for record in vcf(f'{chrom}:{pos}-{pos}')
                           if record.ID == marker['ID'] and record.REF == marker['REF']
                           and record.ALT == [marker['ALT']]]
                if len(matches) != 1:
                    raise ValueError(f'Expected one VCF record matching ID and alleles: {marker.ID}')
                gt = np.asarray([g[:2] for g in matches[0].genotypes], dtype=float)
                dosage = gt.sum(axis=1)
                dosage[(gt < 0).any(axis=1)] = np.nan
                env_box = landraces[['LIB', 'Latitude', 'Longitude', env_id]].copy()
                env_box['GENOTYPE'] = [dosage[sample_index[s]] for s in env_box['LIB']]
                env_box = env_box.dropna().loc[lambda d: d['GENOTYPE'].isin([0, 2])]
                env_box = env_box.rename(columns={env_id: 'VALUE'})
                expr_box = pd.DataFrame({'LIB': tpm.columns, 'VALUE': tpm.loc[gene_id].to_numpy(),
                                         'GENOTYPE': [dosage[sample_index[s]] for s in tpm.columns]})
                expr_box = expr_box.dropna().loc[lambda d: d['GENOTYPE'].isin([0, 2])]
                for name, box in [('environment', env_box), ('expression', expr_box)]:
                    if not {0, 2}.issubset(set(box['GENOTYPE'])):
                        raise ValueError(f'Both homozygous groups are required in {name} for {gene_id}.')
                flank = config.get('flank_size', 18000)
                start, end = max(1, gene.start - flank), gene.end + flank
                if chrom != gene.chr_id or not start <= pos <= end:
                    raise ValueError(f'Marker lies outside the plotted locus for {gene_id}.')
                regional = []
                for panel, table in [('gea', env), ('eqtl', eqtl)]:
                    local = table[(table['CHROM'] == gene.chr_id) & table['POS'].between(start, end)].copy()
                    local['PANEL'] = panel
                    local['VALUE'] = -np.log10(local['P_WALD'].clip(lower=np.nextafter(0.0, 1.0)))
                    local['IS_MARKER'] = local['ID'] == marker['ID']
                    regional.append(local)
                models = []
                neighbors = [g for g in gene_dict.values() if g.chr_id == gene.chr_id and g.start >= start and g.end <= end]
                for track, neighbor in enumerate(neighbors):
                    base = {'GENE': neighbor.id, 'CHROM': neighbor.chr_id, 'STRAND': neighbor.strand,
                            'TRACK': track % 2, 'IS_TARGET': neighbor.id == gene_id}
                    models.append(dict(base, PANEL='gene', START=neighbor.start, END=neighbor.end))
                    for exon in (neighbor.sub_features[0].sub_features or []) if neighbor.sub_features else []:
                        if exon.type == 'exon':
                            models.append(dict(base, PANEL='exon', START=exon.start, END=exon.end))
                env_box['PANEL'], expr_box['PANEL'] = 'env_box', 'expr_box'
                data = pd.concat(regional + [pd.DataFrame(models), env_box, expr_box], ignore_index=True)
                data['TARGET_GENE'], data['ENV_ID'] = gene_id, env_id
                prefix = _locus_prefix(output_dir, config)
                info = env_info.loc[env_id]
                summary = {'GENE': gene_id, 'GENE_NAME': config['gene_name'], 'ENV_ID': env_id,
                           'ENV_NAME': info['short description'], 'ENV_UNITS': info['units'],
                           'CHROM': gene.chr_id, 'START': start, 'END': end, 'FLANK_SIZE': flank,
                           'MARKER_ID': marker['ID'], 'MARKER_POS': pos, 'MARKER_REF': marker['REF'],
                           'MARKER_ALT': marker['ALT'], 'MARKER_MAF': marker['MAF'],
                           'GEA_P': marker['P_WALD_GEA'], 'EQTL_P': marker['P_WALD_EQTL'],
                           'EQTL_BH_FDR': marker['BH_FDR'], 'TOP1_P': top1, 'TOP5_P': top5,
                           'EQTL_THRESHOLD': threshold, 'N_CIS_TESTED': len(eqtl),
                           'N_ENV_REF': int((env_box['GENOTYPE'] == 0).sum()),
                           'N_ENV_ALT': int((env_box['GENOTYPE'] == 2).sum()),
                           'N_EXPR_REF': int((expr_box['GENOTYPE'] == 0).sum()),
                           'N_EXPR_ALT': int((expr_box['GENOTYPE'] == 2).sum()),
                           'LOCUS_FILE': f'{prefix}.locus.csv.gz', 'SUMMARY_FILE': f'{prefix}.summary.csv.gz'}
                _write_csv(data, summary['LOCUS_FILE'])
                _write_csv(pd.DataFrame([summary]), summary['SUMMARY_FILE'])
                summaries.append(summary)
            del env
    finally:
        vcf.close()
    _message(f'Flowering-gene loci ready ({perf_counter() - start_time:.1f}s)')
    return pd.DataFrame(summaries)


def _export_figure(fig, data, prefix):
    files = {'pdf': f'{prefix}.pdf', 'png': f'{prefix}.png', 'data': f'{prefix}.csv.gz'}
    _write_csv(data, files['data'])
    fig.savefig(files['pdf'], bbox_inches='tight', facecolor='none', edgecolor='none')
    fig.savefig(files['png'], dpi=300, bbox_inches='tight')
    return files


def plot_flowering_gene_locus(locus_csv, summary_csv, output_prefix, figsize=(10, 4.2),
                             title_fontsize=15, panel_label_fontsize=11, y_label_fontsize=12,
                             tick_fontsize=10, gene_track_base=1.06, gene_track_gap=0.10,
                             gene_track_height=0.05, gene_arrow_offset=0.025,
                             gene_arrow_fontsize=11, box_xlabel_fontsize=10,
                             box_title_fontsize=10, step_mb=0.01, decimals=2,
                             wspace=0.28, hspace=0.14):
    """
    Draw the historical compact regional GEA/eQTL and paired-boxplot figure.

    Parameters
    ----------
    locus_csv, summary_csv : str
        Prepared per-gene tables from prepare_flowering_gene_loci. No GWAS,
        threshold estimation, genotype extraction, or BH correction is run.
    output_prefix : str
        Prefix for PDF, PNG and actual plotting data CSV.gz.
    figsize : tuple, default (10, 4.2)
        Width and height in inches.
    title_fontsize, panel_label_fontsize, y_label_fontsize, tick_fontsize : float
        Font sizes for the gene title, panel labels, Y labels and ticks.
    gene_track_base, gene_track_gap, gene_track_height, gene_arrow_offset : float
        Gene-model geometry in axes-relative Y coordinates.
    gene_arrow_fontsize, box_xlabel_fontsize, box_title_fontsize : float
        Font sizes for strand arrows, shared marker label and boxplot titles.
    step_mb : float, default 0.01
        Genomic tick spacing in Mb.
    decimals : int, default 2
        Decimal places in genomic tick labels.
    wspace, hspace : float
        Horizontal and vertical GridSpec spacing; global style is not reset.

    Returns
    -------
    dict
        figure, named axes, plot_data and files. Exported data includes source
        values, transformed values, display ordering, model geometry and saved
        marker/threshold metadata. Figures are not displayed automatically.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    _message(f'Plotting flowering-gene locus: {locus_csv}')
    data, meta = pd.read_csv(locus_csv), pd.read_csv(summary_csv).iloc[0]
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, width_ratios=[4.2, 4.2, 1.15, 1.15],
                  height_ratios=[1, 1], wspace=wspace, hspace=hspace)
    ax_env = fig.add_subplot(gs[0, :2])
    ax_eqtl = fig.add_subplot(gs[1, :2], sharex=ax_env)
    ax_env_box, ax_expr_box = fig.add_subplot(gs[:, 2]), fig.add_subplot(gs[:, 3])
    axes = {'gea': ax_env, 'eqtl': ax_eqtl, 'environment_box': ax_env_box, 'expression_box': ax_expr_box}
    start, end = float(meta['START']), float(meta['END'])
    for panel, ax, color in [('gea', ax_env, '#5E7DBA'), ('eqtl', ax_eqtl, '#E7969E')]:
        points = data[data['PANEL'] == panel]
        ax.scatter(points['POS'], points['VALUE'], s=18, c=color, alpha=0.95, edgecolors='none', zorder=1)
        selected = points[points['ID'] == meta['MARKER_ID']]
        ax.scatter(selected['POS'], selected['VALUE'], s=46, facecolors='#69AF97',
                   edgecolors='black', linewidths=1, zorder=5)
        lines = [(meta['TOP1_P'], 'Top 1%', 'black'), (meta['TOP5_P'], 'Top 5%', '0.6')] if panel == 'gea' else [(meta['EQTL_THRESHOLD'], 'FDR = 0.05', 'black')]
        line_values = [-np.log10(p) for p, _, _ in lines]
        maximum = max(float(points['VALUE'].max()), max(line_values))
        ax.set_ylim(ax.get_ylim()[0], maximum + max(maximum * 0.22, 0.8))
        for (p, label, line_color), y in zip(lines, line_values):
            ax.axhline(y, color=line_color, linestyle='--', lw=1)
            ax.text(start + (end - start) * 0.01, y + np.diff(ax.get_ylim())[0] * 0.03,
                    label, fontsize=panel_label_fontsize, color=line_color, va='bottom')
        label = meta['ENV_NAME'] if panel == 'gea' else rf'Expression of $\it{{{meta["GENE_NAME"]}}}$'
        ax.text(end, ax.get_ylim()[1] * 0.93, label, fontsize=panel_label_fontsize + 1, ha='right', va='center')
        ax.set_ylabel(r'$-log_{10}(P)$', fontsize=y_label_fontsize)
        ax.set_xlim(start, end)
        ax.tick_params(axis='both', labelsize=tick_fontsize, width=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    transform = ax_env.get_xaxis_transform()
    models = data[data['PANEL'].isin(['gene', 'exon'])]
    for row in models.itertuples():
        y = gene_track_base + row.TRACK * gene_track_gap
        color = 'black' if row.GENE == meta['GENE'] else '#8a8a8a'
        if row.PANEL == 'gene':
            ax_env.plot([row.START, row.END], [y, y], color=color, linewidth=2, transform=transform, clip_on=False)
            arrow = '→' if str(row.STRAND) in ['+', '1', '1.0'] else '←'
            ax_env.text((row.START + row.END) / 2, y + gene_arrow_offset, arrow,
                        transform=transform, ha='center', va='bottom', fontsize=gene_arrow_fontsize,
                        color=color, clip_on=False)
        else:
            ax_env.add_patch(plt.Rectangle((row.START, y - gene_track_height / 2),
                                          row.END - row.START, gene_track_height,
                                          transform=transform, color=color, clip_on=False))
    ax_env.text(0.5, 1.20, meta['GENE_NAME'], transform=ax_env.transAxes,
                ha='center', va='bottom', fontsize=title_fontsize, fontstyle='italic')
    ax_env.tick_params(axis='x', labelbottom=False)
    step = step_mb * 1e6
    ticks = np.arange(np.ceil(start / step) * step, np.floor(end / step) * step + step * 0.5, step)
    ax_eqtl.set_xticks(ticks)
    ax_eqtl.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1e6:.{decimals}f}'))
    ax_eqtl.set_xlabel(f'{meta["CHROM"]} (Mb)', fontsize=tick_fontsize + 1)
    genotype_order = ['0', '2']
    palette = dict(zip(genotype_order, sns.color_palette('Set2')[:2]))
    units = '' if pd.isna(meta['ENV_UNITS']) else f' ({meta["ENV_UNITS"]})'
    for panel, ax, ylabel, statistic, p in [
        ('env_box', ax_env_box, f'{meta["ENV_NAME"]}{units}', 'P', meta['GEA_P']),
        ('expr_box', ax_expr_box, rf'Expression of $\it{{{meta["GENE_NAME"]}}}$ (TPM)', 'FDR', meta['EQTL_BH_FDR']),
    ]:
        box = data[data['PANEL'] == panel].copy()
        box['GENOTYPE'] = box['GENOTYPE'].astype(int).astype(str)
        sns.boxplot(x='GENOTYPE', y='VALUE', hue='GENOTYPE', data=box,
                    order=genotype_order, hue_order=genotype_order, palette=palette,
                    dodge=False, legend=False, width=0.6, fliersize=0, linewidth=1, ax=ax)
        formatted = f'{p:.2e}' if p < 1e-3 else f'{p:.3f}'
        ax.set_title(rf'${statistic} = {formatted}$', fontsize=box_title_fontsize, pad=8)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontsize=panel_label_fontsize)
        ax.set_xticks([0, 1], ['0', '2'])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis='y', labelrotation=90, labelsize=max(tick_fontsize - 2, 7), pad=0, length=0)
        ax.tick_params(axis='x', labelsize=tick_fontsize, length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor('0.25')
    left, right = ax_env_box.get_position(), ax_expr_box.get_position()
    marker_label = f'{meta["CHROM"]}: {int(meta["MARKER_POS"]):,} bp (MAF = {meta["MARKER_MAF"]:.3f})'
    fig.text((left.x0 + right.x1) / 2, min(left.y0, right.y0) - 0.06, marker_label,
             ha='center', va='top', fontsize=box_xlabel_fontsize)
    data['PLOT_ORDER'] = data.groupby('PANEL', sort=False).cumcount()
    data['POSITION_MB'] = data['POS'] / 1e6
    data['MODEL_Y'] = gene_track_base + data['TRACK'] * gene_track_gap
    data['MODEL_HEIGHT'] = np.where(data['PANEL'].eq('exon'), gene_track_height, np.nan)
    for key in ['MARKER_ID', 'MARKER_POS', 'MARKER_MAF', 'TOP1_P', 'TOP5_P', 'EQTL_THRESHOLD', 'GEA_P', 'EQTL_BH_FDR']:
        data[key] = meta[key]
    files = _export_figure(fig, data, output_prefix)
    return {'figure': fig, 'axes': axes, 'plot_data': data, 'files': files}


def prepare_regulatory_gea_genes(egene_list_file, site_csv, threshold_csv, gwas_h5,
                                 tpm_csv, landrace_csv, vcf_file, gff_file,
                                 output_prefix, env_id='gsl', flank_size=15000,
                                 top_fraction=0.01, r2_threshold=0.25,
                                 force_recompute=False):
    """
    Compare independent lead cis-eQTL and environmental variants within eGenes.

    Parameters
    ----------
    egene_list_file : str
        Saved full eGene background, one unique gene ID per line.
    site_csv, threshold_csv : str
        Complete cis-site CSV and saved empirical nominal P threshold.
    gwas_h5 : str
        Full environmental GEMMA table, key df, including ID, CHROM, POS,
        REF, ALT and P_WALD. The top fraction is defined genome-wide.
    tpm_csv, landrace_csv : str
        TPM CSV with gene IDs in column one and RNA sample LIB IDs in its
        header, and landrace metadata containing LIB. Only IDs are read.
    vcf_file, gff_file : str
        Indexed all-sample VCF and gene GFF in 1-based inclusive coordinates.
    output_prefix : str
        Prefix for pairs/audit, candidate genes, background and summary CSV.gz,
        and candidate gene ID list TXT. All outputs are atomically replaced.
    env_id : str, default 'gsl'
        Environmental identifier recorded in saved results.
    flank_size : int, default 15000
        bp on each side of the gene body for both independent lead searches.
    top_fraction : float, default 0.01
        Genome-wide nominal P quantile for environmental candidates; ties pass.
    r2_threshold : float, default 0.25
        Inclusive squared Pearson dosage correlation threshold.
    force_recompute : bool, default False
        Reuse only a complete output set. Changed inputs or parameters require
        explicit recomputation. No association analyses are launched.

    Returns
    -------
    dict
        pairs, candidates, summary and files. Every background gene appears in
        the pairs/audit table, including absent local GEA or undefined r2.
        Correlation uses the deduplicated union of landrace and RNA samples;
        missing dosages are excluded pairwise, and constant dosages give NaN.
        These candidate genes are an LD-based operational definition, not a
        formal colocalization result.
    """
    from cyvcf2 import VCF
    from src.feature import read_gff_file

    files = {key: f'{output_prefix}.{key}.csv.gz'
             for key in ['pairs', 'candidates', 'background', 'summary']}
    files['gene_list'] = f'{output_prefix}.gene_ids.txt'
    if not force_recompute and all(os.path.exists(p) for p in files.values()):
        _message(f'Reusing regulatory GEA candidates: {output_prefix}')
        return {key: pd.read_csv(files[key]) for key in ['pairs', 'candidates', 'summary']} | {'files': files}
    start_time = perf_counter()
    _message(f'Comparing lead cis-eQTL and {env_id} GEA variants')
    with open(egene_list_file) as handle:
        background = handle.read().splitlines()
    if not background or len(background) != len(set(background)):
        raise ValueError('The eGene background must be nonempty with unique IDs.')
    genes = read_gff_file(gff_file)['gene']
    missing = set(background) - set(genes)
    if missing:
        raise ValueError(f'eGene IDs missing from GFF: {sorted(missing)}')
    regions = pd.DataFrame([{'GENE': g, 'GENE_CHROM': genes[g].chr_id,
                             'REGION_START': max(1, genes[g].start - flank_size),
                             'REGION_END': genes[g].end + flank_size} for g in background])
    threshold = float(pd.read_csv(threshold_csv).iloc[0]['P_VALUE_THRESHOLD'])
    leads = []
    columns = ['GENE', 'ID', 'CHROM', 'POS', 'REF', 'ALT', 'P_WALD']
    for chunk in tqdm(pd.read_csv(site_csv, usecols=columns, chunksize=300000),
                      desc='Selecting lead cis-eQTLs', unit='chunk'):
        local = chunk[chunk['GENE'].isin(background) & (chunk['P_WALD'] <= threshold)].merge(
            regions, on='GENE', validate='many_to_one')
        local = local[(local['CHROM'] == local['GENE_CHROM']) &
                      local['POS'].between(local['REGION_START'], local['REGION_END'])]
        leads.append(local.sort_values(['P_WALD', 'POS', 'ID']).drop_duplicates('GENE')[columns])
    leads = pd.concat(leads).sort_values(['P_WALD', 'POS', 'ID']).drop_duplicates('GENE').set_index('GENE')
    env = pd.read_hdf(gwas_h5, key='df')[['ID', 'CHROM', 'POS', 'REF', 'ALT', 'P_WALD']]
    if not env['ID'].is_unique:
        raise ValueError('Environmental GWAS must have unique IDs.')
    env = _valid_nominal_sites(env, 'Environmental GWAS')
    env_threshold = float(env['P_WALD'].quantile(top_fraction))
    eligible = env[env['P_WALD'] <= env_threshold].sort_values(['CHROM', 'POS', 'ID'])
    by_chrom = {chrom: table.reset_index(drop=True) for chrom, table in eligible.groupby('CHROM')}
    del env, eligible
    rna_samples = pd.read_csv(tpm_csv, nrows=0).columns[1:].tolist()
    landrace_samples = pd.read_csv(landrace_csv, usecols=['LIB'])['LIB'].tolist()
    samples = sorted(set(rna_samples) | set(landrace_samples))
    vcf = VCF(vcf_file)
    sample_index = {sample: i for i, sample in enumerate(vcf.samples)}
    missing = set(samples) - set(sample_index)
    if missing:
        vcf.close()
        raise ValueError(f'Samples missing from VCF: {sorted(missing)}')
    indices = [sample_index[s] for s in samples]
    dosages = {}

    def marker_dosage(site):
        key = (site['CHROM'], int(site['POS']), site['ID'], site['REF'], site['ALT'])
        if key not in dosages:
            chrom, pos, site_id, ref, alt = key
            matches = [v for v in vcf(f'{chrom}:{pos}-{pos}')
                       if v.ID == site_id and v.REF == ref and v.ALT == [alt]]
            if len(matches) != 1:
                raise ValueError(f'Expected one VCF record matching ID and alleles: {site_id}')
            gt = np.asarray([g[:2] for g in matches[0].genotypes], dtype=float)[indices]
            dosage = gt.sum(axis=1)
            dosage[(gt < 0).any(axis=1)] = np.nan
            dosages[key] = dosage
        return dosages[key]

    rows = []
    try:
        for region in tqdm(regions.to_dict('records'), desc='Comparing lead genotypes', unit='gene'):
            gene_id = region['GENE']
            row = dict(region, ENV_ID=env_id, STATUS='no_significant_cis_variant',
                       GENOTYPE_COR=np.nan, GENOTYPE_R2=np.nan, N_PAIRED_SAMPLES=0)
            if gene_id not in leads.index:
                rows.append(row)
                continue
            cis = leads.loc[gene_id]
            row.update({f'CIS_{c}': cis[c] for c in ['ID', 'CHROM', 'POS', 'REF', 'ALT', 'P_WALD']})
            table = by_chrom.get(region['GENE_CHROM'])
            if table is None:
                row['STATUS'] = 'no_top_gea_variant'
                rows.append(row)
                continue
            positions = table['POS'].to_numpy()
            lo = np.searchsorted(positions, region['REGION_START'], side='left')
            hi = np.searchsorted(positions, region['REGION_END'], side='right')
            local = table.iloc[lo:hi]
            if local.empty:
                row['STATUS'] = 'no_top_gea_variant'
                rows.append(row)
                continue
            gea = local.sort_values(['P_WALD', 'POS', 'ID']).iloc[0]
            row.update({f'GEA_{c}': gea[c] for c in ['ID', 'CHROM', 'POS', 'REF', 'ALT', 'P_WALD']})
            x, y = marker_dosage(cis), marker_dosage(gea)
            valid = np.isfinite(x) & np.isfinite(y)
            row['N_PAIRED_SAMPLES'] = int(valid.sum())
            if valid.sum() < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
                row['STATUS'] = 'undefined_correlation'
            else:
                cor = float(np.corrcoef(x[valid], y[valid])[0, 1])
                row.update(STATUS='compared', GENOTYPE_COR=cor, GENOTYPE_R2=cor ** 2)
            rows.append(row)
    finally:
        vcf.close()
    pairs = pd.DataFrame(rows).sort_values('GENE', ignore_index=True)
    pairs['REG_GEA'] = pairs['GENOTYPE_R2'] >= r2_threshold
    candidates = pairs[pairs['REG_GEA']].copy()
    summary = pd.DataFrame([{'ENV_ID': env_id, 'N_EGENES': len(background),
                             'N_LEAD_PAIRS': int(pairs['GEA_ID'].notna().sum()) if 'GEA_ID' in pairs else 0,
                             'N_VALID_CORRELATIONS': int(pairs['STATUS'].eq('compared').sum()),
                             'N_REG_GEA': len(candidates), 'N_RNA_SAMPLES': len(set(rna_samples)),
                             'N_LANDRACE_SAMPLES': len(set(landrace_samples)), 'N_UNION_SAMPLES': len(samples),
                             'FLANK_SIZE': flank_size, 'TOP_FRACTION': top_fraction,
                             'GEA_P_THRESHOLD': env_threshold, 'EQTL_P_THRESHOLD': threshold,
                             'R2_THRESHOLD': r2_threshold}])
    for key, table in [('pairs', pairs), ('candidates', candidates),
                       ('background', pd.DataFrame({'GENE': sorted(background)})), ('summary', summary)]:
        _write_csv(table, files[key])
    temporary = f'{files["gene_list"]}.tmp'
    with open(temporary, 'w') as handle:
        handle.write(''.join(f'{g}\n' for g in candidates['GENE']))
    os.replace(temporary, files['gene_list'])
    _message(f'{len(candidates):,} regulatory GEA candidates ready ({perf_counter() - start_time:.1f}s)')
    return {'pairs': pairs, 'candidates': candidates, 'summary': summary, 'files': files}


def plot_regulatory_gea_r2(pairs_csv, summary_csv, output_prefix, bins=20, figsize=(6.4, 4.8)):
    """
    Plot saved lead-pair r2 values without recomputing genotypes or correlations.

    Parameters
    ----------
    pairs_csv, summary_csv : str
        Saved pairs and parameter summary from prepare_regulatory_gea_genes.
    output_prefix : str
        Prefix for PDF, PNG, per-gene plotting CSV.gz and histogram CSV.gz.
    bins : int, default 20
        Number of equally spaced bins spanning the observed r2 range, matching
        the historical pandas histogram convention.
    figsize : tuple, default (6.4, 4.8)
        Figure size in inches; global plot style is preserved.

    Returns
    -------
    dict
        figure, axes, plot_data, histogram_data and files. Only finite r2 values
        appear in the histogram; the inclusive candidate threshold is marked.
    """
    import matplotlib.pyplot as plt

    _message('Plotting lead-variant genotype correlations')
    data = pd.read_csv(pairs_csv)
    data = data[np.isfinite(data['GENOTYPE_R2'])].copy()
    if data.empty:
        raise ValueError('No finite lead-pair r2 values to plot.')
    threshold = float(pd.read_csv(summary_csv).iloc[0]['R2_THRESHOLD'])
    fig, ax = plt.subplots(figsize=figsize)
    counts, edges, _ = ax.hist(data['GENOTYPE_R2'], bins=bins)
    ax.axvline(threshold, color='r', linestyle='--', label=f'r² ≥ {threshold:g}')
    ax.set(xlabel='Lead-variant genotype r²', ylabel='Frequency')
    ax.legend()
    data['R2_THRESHOLD'] = threshold
    histogram = pd.DataFrame({'BIN_LEFT': edges[:-1], 'BIN_RIGHT': edges[1:], 'COUNT': counts.astype(int)})
    files = _export_figure(fig, data, output_prefix)
    files['histogram'] = f'{output_prefix}.histogram.csv.gz'
    _write_csv(histogram, files['histogram'])
    return {'figure': fig, 'axes': ax, 'plot_data': data, 'histogram_data': histogram, 'files': files}


def prepare_regulatory_gea_enrichment(background_csv, candidates_csv, flowering_csv,
                                      output_prefix, annotation_source=None,
                                      force_recompute=False):
    """
    Test flowering-gene enrichment using the full eGene background.

    Parameters
    ----------
    background_csv, candidates_csv : str
        Saved background and candidates from prepare_regulatory_gea_genes.
        The background includes eGenes without a local top GEA variant.
    flowering_csv : str
        Gene_ID column and, when filtering, Annotation_source. Selected flowering
        genes are deduplicated and intersected with the eGene background.
    output_prefix : str
        Prefix for enrichment, overlap and contingency CSV.gz tables.
    annotation_source : str or None, default None
        None selects all flowering genes. A string selects rows whose
        Annotation_source exactly matches that value.
    force_recompute : bool, default False
        Reuse the full output set for the same annotation selection. A changed
        selection is recomputed; changed inputs require explicit recomputation.

    Returns
    -------
    dict
        summary, overlap, contingency and files. Hypergeometric survival P and
        one-sided Fisher greater P use the same 2-by-2 contingency table;
        odds ratio and observed/expected overlap are also saved.
    """
    from scipy.stats import hypergeom, fisher_exact

    selection_label = 'All flowering genes' if annotation_source is None else annotation_source
    files = {key: f'{output_prefix}.{key}.csv.gz' for key in ['enrichment', 'overlap', 'contingency']}
    if not force_recompute and all(os.path.exists(p) for p in files.values()):
        saved_summary = pd.read_csv(files['enrichment'])
        if saved_summary.iloc[0]['ANNOTATION_SOURCE'] == selection_label:
            _message('Reusing flowering-gene regulatory GEA enrichment')
            return {'summary': saved_summary, 'overlap': pd.read_csv(files['overlap']),
                    'contingency': pd.read_csv(files['contingency']), 'files': files}
    _message(f'Testing flowering-gene enrichment within eGenes: {selection_label}')
    background = set(pd.read_csv(background_csv)['GENE'])
    candidates = pd.read_csv(candidates_csv)
    target = set(candidates['GENE'])
    if not target.issubset(background) or not background:
        raise ValueError('Candidates must be a subset of a nonempty eGene background.')
    flowering = pd.read_csv(flowering_csv)
    selected = flowering if annotation_source is None else flowering[flowering['Annotation_source'] == annotation_source]
    selected = selected.drop_duplicates('Gene_ID')
    if selected.empty:
        raise ValueError(f'No flowering genes selected: {selection_label}.')
    selected_ids = set(selected['Gene_ID']) & background
    M, n, N = len(background), len(selected_ids), len(target)
    x = len(selected_ids & target)
    table = [[x, n - x], [N - x, M - n - N + x]]
    odds, fisher_p = fisher_exact(table, alternative='greater')
    expected = n * N / M
    summary = pd.DataFrame([{'BACKGROUND': 'eGenes', 'ANNOTATION_SOURCE': selection_label,
                             'N_BACKGROUND': M, 'N_FLOWERING': n, 'N_REG_GEA': N,
                             'N_OVERLAP': x, 'EXPECTED_OVERLAP': expected,
                             'FOLD_ENRICHMENT': x / expected if expected else np.nan,
                             'ODDS_RATIO': odds, 'HYPERGEOM_P': hypergeom.sf(x - 1, M, n, N),
                             'FISHER_GREATER_P': fisher_p}])
    overlap = candidates[candidates['GENE'].isin(selected_ids)].merge(
        selected, left_on='GENE', right_on='Gene_ID', validate='one_to_one')
    contingency = pd.DataFrame({'FLOWERING_CLASS': [selection_label, 'Other eGenes'],
                                'REG_GEA': [x, N - x], 'NOT_REG_GEA': [n - x, M - n - N + x]})
    for key, data in [('enrichment', summary), ('overlap', overlap), ('contingency', contingency)]:
        _write_csv(data, files[key])
    return {'summary': summary, 'overlap': overlap, 'contingency': contingency, 'files': files}
