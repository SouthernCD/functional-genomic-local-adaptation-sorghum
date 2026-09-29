"""Phenotype-blind, intercept-only GCTB benchmark for sorghum.

This intentionally does not call SIMULATION_JOB: that class also performs
functional enrichment and deletes raw GEMMA association output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from .gwas import _run_gemma_parallel
from .sim import simulate_phenotype_with_kinship
from .vcf import convert_vcf_to_bimbam


PROJECT = Path('/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum')
BENCH = PROJECT / 'data/results/gwfm_rc_benchmark'
SOFTWARE = Path('/lustre/home/dobby/Program/agent_program/bin')
KINSHIP = PROJECT / 'data/source/landraces_snp_pruned.win15000.miss0.5.maf0.01.het0.01.rq0.5.ibs.matrix'
REQUIRED_GCTB = ('.snpRes', '.parRes', '.gcs', '.gcsRes', '.lcs', '.lcsRes')


def _run(command: list[str], log: Path | None = None, **kwargs) -> None:
    if log is None:
        subprocess.run(command, check=True, **kwargs)
    else:
        with log.open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, **kwargs)


def _load_panel(panel: Path) -> pd.DataFrame:
    if not (panel / 'COMPLETE').is_file():
        raise RuntimeError(f'panel is incomplete: {panel}')
    df = pd.read_csv(panel / 'variants.tsv', sep='\t', dtype={'SNP': str})
    if df.SNP.duplicated().any() or df.empty:
        raise ValueError('panel SNP IDs must be unique and nonempty')
    return df.set_index('SNP')


def _kinship(panel: Path) -> pd.DataFrame:
    samples = (panel / 'samples.txt').read_text().splitlines()
    matrix = pd.read_csv(KINSHIP, header=None).to_numpy(dtype=float)
    if matrix.shape != (len(samples), len(samples)):
        raise ValueError('kinship dimensions differ from panel samples')
    return pd.DataFrame(matrix, index=samples, columns=samples)


def _effect_weights(maf: np.ndarray, architecture: str,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Normalize by expected genotype SD so MAF does not determine effect class.
    effects = rng.normal(size=len(maf)) / np.sqrt(2 * maf * (1 - maf))
    large_mask = np.zeros(len(effects), dtype=bool)
    if architecture == 'large':
        if len(effects) != 500:
            raise ValueError('large architecture requires 500 causal SNPs')
        large = rng.choice(len(effects), size=10, replace=False)
        large_mask[large] = True
        weights = np.full(len(effects), np.sqrt(0.8 / 490))
        weights[large] = np.sqrt(0.2 / 10)
        effects *= weights
    return effects, large_mask


def _choose_truth(panel: Path, architecture: str, count: int, seed: int,
                  ld_scores: Path | None) -> pd.DataFrame:
    if architecture not in ('gaussian', 'large', 'ldms', 'null'):
        raise ValueError(f'unknown architecture: {architecture}')
    if architecture in ('large', 'ldms') and count != 500:
        raise ValueError(f'{architecture} requires 500 causal SNPs')
    panel_df = _load_panel(panel)
    complete = pd.Index((panel / 'complete_genotype_variant_ids.txt').read_text().splitlines())
    candidates = panel_df.loc[complete]
    if architecture == 'ldms':
        if ld_scores is None:
            raise ValueError('LDMS requires a precomputed genotype-only LD score file')
        scores = pd.read_csv(ld_scores, sep='\t').set_index('SNP')['LD_SCORE']
        candidates = candidates.join(scores, how='inner')
        for quantile in (0.75, 0.70, 0.65, 0.60, 0.55, 0.50):
            maf_min = candidates.MAF.quantile(quantile)
            score_min = candidates.LD_SCORE.quantile(quantile)
            stratum = candidates[(candidates.MAF >= maf_min) &
                                 (candidates.LD_SCORE >= score_min)]
            if len(stratum) >= count:
                candidates = stratum.copy()
                candidates['ldms_quantile'] = quantile
                candidates['ldms_maf_min'] = maf_min
                candidates['ldms_score_min'] = score_min
                break
        else:
            raise ValueError('fewer than 500 SNPs in joint upper-half MAF/LD stratum')
    if len(candidates) < max(count, 1):
        raise ValueError('too few complete-genotype variants in causal candidate pool')
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(candidates), size=max(count, 1), replace=False)
    truth = candidates.iloc[selected].copy()
    truth.insert(0, 'SNP', truth.index)
    if architecture == 'null':
        truth['beta_raw'] = 0.0
        truth['effect_class'] = 'null'
    else:
        effects, large_mask = _effect_weights(
            truth.MAF.to_numpy(dtype=float), architecture,
            np.random.default_rng(seed + 1))
        truth['beta_raw'] = effects
        truth['effect_class'] = np.where(large_mask, 'large', 'background')
    truth['is_causal'] = architecture != 'null'
    return truth.reset_index(drop=True)


def _extract_causal_dosages(panel: Path, truth: pd.DataFrame,
                            directory: Path) -> pd.DataFrame:
    (directory / 'causal_ids.txt').write_text('\n'.join(truth.SNP) + '\n')
    prefix = directory / 'causal_genotype'
    _run([str(SOFTWARE / 'plink'), '--bfile', str(panel / 'panel'),
          '--extract', str(directory / 'causal_ids.txt'), '--recode', 'A',
          '--out', str(prefix)], directory / 'plink_extract.log')
    raw = pd.read_csv(f'{prefix}.raw', sep=r'\s+', dtype={'IID': str})
    sample_ids = (panel / 'samples.txt').read_text().splitlines()
    if raw.IID.astype(str).tolist() != sample_ids:
        raise ValueError('PLINK raw sample order differs from VCF')
    dosage_columns = list(raw.columns[6:])
    if len(dosage_columns) != len(truth):
        raise ValueError('PLINK causal extraction lost SNPs')
    info = truth.set_index('SNP')
    matrix = {}
    for column in dosage_columns:
        snp, allele = column.rsplit('_', 1)
        if snp not in info.index:
            raise ValueError(f'unexpected dosage SNP: {snp}')
        alt, ref = info.loc[snp, ['ALT', 'REF']]
        value = raw[column].to_numpy(dtype=float)
        if allele == ref:
            value = 2 - value
        elif allele != alt:
            raise ValueError(f'PLINK dosage allele mismatch: {column}')
        if not np.isfinite(value).all():
            raise ValueError(f'missing causal genotype: {snp}')
        matrix[snp] = value
    return pd.DataFrame(matrix, index=sample_ids).loc[:, truth.SNP.tolist()]


def make_phenotype(panel: Path, directory: Path, architecture: str,
                   count: int, causal_var: float, seed: int,
                   ld_scores: Path | None = None) -> Path:
    if directory.exists() and any(
            child.name not in {'command.txt', 'driver.log', 'FAILED'}
            for child in directory.iterdir()):
        raise FileExistsError(f'run directory contains a partial phenotype: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    truth = _choose_truth(panel, architecture, count, seed, ld_scores)
    genotype = _extract_causal_dosages(panel, truth, directory)
    large_share = None
    if architecture == 'large':
        large = truth.effect_class.eq('large').to_numpy()
        dosage = genotype.to_numpy(dtype=float)
        beta = truth.beta_raw.to_numpy(dtype=float).copy()
        big_var = np.var(dosage[:, large] @ beta[large], ddof=1)
        small_var = np.var(dosage[:, ~large] @ beta[~large], ddof=1)
        if big_var <= 0 or small_var <= 0:
            raise ValueError('large-effect groups have zero genotype variance')
        beta[large] *= np.sqrt(0.2 * small_var / (0.8 * big_var))
        truth['beta_raw'] = beta
        adjusted_big_var = np.var(dosage[:, large] @ beta[large], ddof=1)
        large_share = adjusted_big_var / (adjusted_big_var + small_var)
    effects = truth[['SNP', 'beta_raw']].rename(columns={'SNP': 'SNP_ID'})
    output = simulate_phenotype_with_kinship(
        genotype, effects, _kinship(panel), directory / 'trait',
        causal_var=causal_var, kinship_var=0.5,
        residual_var=0.5 - causal_var, seed=seed + 2)
    scaled = pd.read_csv(output['causal_file'], sep='\t').set_index('SNP_ID')
    truth['beta_scaled'] = truth.SNP.map(scaled.beta_scaled)
    truth = truth[truth.is_causal]
    truth.to_csv(directory / 'truth.tsv', sep='\t', index=False)
    (directory / 'design.json').write_text(json.dumps({
        'architecture': architecture, 'n_causal': len(truth),
        'causal_var': causal_var, 'kinship_var': 0.5,
        'residual_var': 0.5 - causal_var, 'seed': seed,
        'phenotype_seed': seed + 2, 'panel': str(panel),
        'no_functional_annotations': True,
        'large_group_variance_share_excluding_cross_covariance': large_share
    }, indent=2) + '\n')
    return Path(output['gemma_phenotype_file'])


def prepare_gemma(panel: Path) -> Path:
    target = panel / 'panel.bimbam'
    if not target.is_file():
        partial = panel / 'panel.bimbam.partial'
        if partial.exists():
            raise RuntimeError(f'incomplete BIMBAM conversion needs inspection: {partial}')
        convert_vcf_to_bimbam(str(panel / 'panel.vcf.gz'), str(partial))
        partial.replace(target)
    return target


def make_block_map(panel: Path, destination: Path,
                   max_snps: int = 1500) -> None:
    """Place boundaries near sparse marker gaps without using any phenotype."""
    bim = pd.read_csv(panel / 'panel.bim', sep=r'\s+', header=None,
                      names=['chrom', 'snp', 'cm', 'pos', 'a1', 'a2'],
                      usecols=[0, 1, 3])
    if max_snps < 100:
        raise ValueError('max_snps must be at least 100')
    blocks = []
    block_id = 1
    for chrom, group in bim.groupby('chrom', sort=True):
        positions = group.pos.to_numpy(dtype=int)
        if np.any(np.diff(positions) < 0):
            raise ValueError(f'PLINK BIM is not sorted on chromosome {chrom}')
        start_index = 0
        block_start = int(positions[0])
        while start_index < len(positions):
            desired = min(start_index + max_snps, len(positions))
            if desired == len(positions):
                end_index = desired
            else:
                lo = max(start_index + max_snps // 2, desired - 150)
                hi = min(len(positions) - 1, desired + 150)
                gaps = np.diff(positions[lo - 1:hi + 1])
                end_index = lo + int(np.argmax(gaps))
                while end_index < len(positions) and positions[end_index] == positions[end_index - 1]:
                    end_index += 1
            block_end = int(positions[end_index - 1])
            blocks.append((block_id, chrom, block_start, block_end))
            block_id += 1
            start_index = end_index
            if start_index < len(positions):
                block_start = block_end + 1
    with destination.open('w') as out:
        out.write('Block Chrom StartBP EndBP\n')
        for row in blocks:
            out.write('%d %s %d %d\n' % row)


def make_bp_block_map(panel: Path, destination: Path,
                      width_bp: int = 15_000) -> None:
    """Write non-overlapping physical-position LD blocks for every panel SNP."""
    if width_bp < 1:
        raise ValueError('width_bp must be positive')
    block_id = 0
    seen_chrom: set[str] = set()
    last_chrom = None
    last_pos = -1
    last_start = -1
    with (panel / 'panel.bim').open() as source, destination.open('w') as out:
        out.write('Block Chrom StartBP EndBP\n')
        for line in source:
            chrom, _, _, pos_text, *_ = line.split()
            pos = int(pos_text)
            if chrom == last_chrom and pos < last_pos:
                raise ValueError(f'BIM is not sorted by position on chromosome {chrom}')
            if chrom != last_chrom and chrom in seen_chrom:
                raise ValueError(f'BIM chromosome {chrom} is not contiguous')
            if chrom != last_chrom:
                seen_chrom.add(chrom)
            start = ((pos - 1) // width_bp) * width_bp + 1
            if chrom != last_chrom or start != last_start:
                block_id += 1
                out.write(f'{block_id} {chrom} {start} {start + width_bp - 1}\n')
                last_start = start
            last_chrom, last_pos = chrom, pos
    if block_id == 0:
        raise ValueError('panel BIM is empty')


def audit_block_boundaries(panel: Path, block_map: Path,
                           destination: Path, per_chrom: int = 100) -> None:
    from cyvcf2 import VCF
    from .ld import calculate_LD

    bim = pd.read_csv(panel / 'panel.bim', sep=r'\s+', header=None,
                      names=['chrom', 'snp', 'cm', 'pos', 'a1', 'a2'],
                      usecols=[0, 1, 3])
    blocks = pd.read_csv(block_map, sep=r'\s+')
    reader = VCF(str(panel / 'panel.vcf.gz'))
    rows = []
    for chrom, group in bim.groupby('chrom', sort=True):
        records = group.reset_index(drop=True)
        boundaries = blocks.loc[blocks.Chrom == chrom, 'EndBP'].to_numpy()[:-1]
        if not len(boundaries):
            continue
        selected = np.unique(np.linspace(0, len(boundaries) - 1,
                                         min(per_chrom, len(boundaries)), dtype=int))
        for index in selected:
            end_bp = int(boundaries[index])
            split = int(np.searchsorted(records.pos.to_numpy(), end_bp, side='right'))
            left = records.iloc[max(0, split - 25):split]
            right = records.iloc[split:split + 25]
            if left.empty or right.empty:
                continue
            wanted = set(left.snp) | set(right.snp)
            genotype = {}
            for variant in reader(f'{chrom}:{int(left.pos.min())}-{int(right.pos.max())}'):
                if variant.ID in wanted:
                    calls = np.asarray(variant.genotypes)[:, :2]
                    dose = np.where((calls < 0).any(axis=1), np.nan,
                                    calls.sum(axis=1).astype(float))
                    if np.isnan(dose).any():
                        dose = np.where(np.isnan(dose), np.nanmean(dose), dose)
                    genotype[variant.ID] = dose
            if len(genotype) != len(wanted):
                raise ValueError(f'boundary genotype extraction mismatch: {chrom}:{end_bp}')
            x = np.stack([genotype[snp] for snp in left.snp])
            y = np.stack([genotype[snp] for snp in right.snp])
            r2 = calculate_LD(x, y, left.snp, right.snp).to_numpy()
            if not np.isfinite(r2).all():
                raise ValueError(f'undefined boundary LD: {chrom}:{end_bp}')
            rows.append((chrom, end_bp, len(left), len(right),
                         float(r2.max()), float(np.mean(r2 > 0.5))))
    reader.close()
    pd.DataFrame(rows, columns=['chrom', 'end_bp', 'n_left', 'n_right',
                                'max_cross_r2', 'fraction_cross_r2_gt_0_5']) \
      .to_csv(destination, sep='\t', index=False)


def make_ld_scores(pairwise_file: Path, destination: Path) -> None:
    """Sum genotype-derived r² over GCTB's r²>0.5 pairs for LDMS sampling."""
    scores: dict[str, float] = {}
    with pairwise_file.open() as source:
        header = source.readline().split()
        if header != ['SNP1', 'SNP2', 'LDcorrelation']:
            raise ValueError(f'unexpected pairwise LD header: {header}')
        for line in source:
            first, second, correlation = line.split()
            value = float(correlation) ** 2
            scores[first] = scores.get(first, 0.0) + value
            scores[second] = scores.get(second, 0.0) + value
    with destination.open('w') as out:
        out.write('SNP\tLD_SCORE\n')
        for snp in sorted(scores):
            out.write(f'{snp}\t{scores[snp]:.9g}\n')


def run_gemma(panel: Path, directory: Path, phenotype: Path,
              workers: int = 40) -> Path:
    bimbam = prepare_gemma(panel)
    os.environ['PATH'] = f'{SOFTWARE}:{os.environ.get("PATH", "")}'
    assoc = directory / 'gemma/output/gemma_output.assoc.txt'
    if not assoc.is_file():
        (directory / 'gemma').mkdir(parents=True, exist_ok=True)
        _run_gemma_parallel(str(directory / 'gemma'), str(bimbam),
                            str(phenotype), str(KINSHIP),
                            str(panel / 'panel.vcf.gz'), retry=3,
                            dry_run=False, block_size=5_000_000,
                            max_workers=workers,
                            block_cache_dir=str(panel / 'gemma_block_cache'),
                            lmm=1)
    if not assoc.is_file() or assoc.stat().st_size == 0:
        raise RuntimeError('GEMMA association file is missing or empty')
    return assoc


def write_gctb_summary(panel: Path, assoc: Path, destination: Path) -> int:
    """Harmonize raw GEMMA allele1 effects to VCF ALT, retaining per-SNP N."""
    variants = _load_panel(panel)[['REF', 'ALT']]
    seen: set[str] = set()
    total = 0
    with destination.open('w') as out:
        out.write('SNP A1 A2 freq b se p N\n')
        for chunk in pd.read_csv(assoc, sep='\t', chunksize=100_000):
            needed = {'rs', 'allele1', 'allele0', 'af', 'beta', 'se',
                      'p_wald'}
            if not needed.issubset(chunk.columns):
                raise ValueError(f'GEMMA columns missing: {needed - set(chunk)}')
            if 'n_obs' not in chunk and 'n_miss' not in chunk:
                raise ValueError('GEMMA output has neither n_obs nor n_miss')
            if chunk.rs.duplicated().any() or any(x in seen for x in chunk.rs):
                raise ValueError('duplicate SNP in GEMMA output')
            seen.update(chunk.rs)
            mapped = variants.reindex(chunk.rs)
            if mapped.isna().any().any():
                raise ValueError('GEMMA contains SNPs absent from panel')
            alt = mapped.ALT.to_numpy()
            ref = mapped.REF.to_numpy()
            a1 = chunk.allele1.to_numpy()
            a0 = chunk.allele0.to_numpy()
            forward = (a1 == alt) & (a0 == ref)
            reverse = (a1 == ref) & (a0 == alt)
            if not (forward | reverse).all():
                raise ValueError('GEMMA allele1/allele0 mismatch with VCF')
            af = np.where(forward, chunk.af.to_numpy(dtype=float),
                          1 - chunk.af.to_numpy(dtype=float))
            beta = np.where(forward, chunk.beta.to_numpy(dtype=float),
                            -chunk.beta.to_numpy(dtype=float))
            se = chunk.se.to_numpy(dtype=float)
            p = chunk.p_wald.to_numpy(dtype=float)
            if 'n_obs' in chunk:
                n = chunk.n_obs.to_numpy(dtype=int)
            else:
                n_samples = sum(1 for _ in (panel / 'samples.txt').open())
                n = n_samples - chunk.n_miss.to_numpy(dtype=int)
            if not (np.isfinite(af).all() and np.isfinite(beta).all() and
                    np.isfinite(se).all() and np.isfinite(p).all() and
                    ((af > 0) & (af < 1)).all() and (se > 0).all() and
                    ((p >= 0) & (p <= 1)).all() and (n > 0).all()):
                raise ValueError('nonfinite or invalid GEMMA summary field')
            p = np.maximum(p, np.nextafter(0, 1))
            for snp, aa, rr, freq, effect, error, prob, size in zip(
                    chunk.rs, alt, ref, af, beta, se, p, n):
                out.write(f'{snp} {aa} {rr} {freq:.9g} {effect:.9g} '
                          f'{error:.9g} {prob:.9g} {size}\n')
            total += len(chunk)
    if total != len(variants):
        raise ValueError(f'GEMMA/LD SNP-set mismatch: {total} of {len(variants)}')
    return total


def run_gctb(directory: Path, ldm: Path, threads: int = 32) -> Path:
    summary = directory / 'sumstats.ma'
    annotation = ldm / 'intercept_only.txt'
    prefix = directory / 'rc'
    log = directory / 'gctb.log'
    guard = PROJECT / 'benchmark_scripts/gwfm_resource_guard.py'
    _run([str(SOFTWARE / 'python3'), str(guard), '--log', str(log),
          '--resources', str(directory / 'gctb.resources.json'), '--',
          str(SOFTWARE / 'gctb'), '--gwfm', 'RC', '--ldm-eigen', str(ldm),
          '--gwas-summary', str(summary), '--annot', str(annotation),
          '--thread', str(threads), '--out', str(prefix)])
    contents = log.read_text(errors='replace')
    if 'Analysis finished:' not in contents or re.search(r'\bError[: ]', contents):
        raise RuntimeError(f'GCTB log reports failure: {log}')
    missing = [suffix for suffix in REQUIRED_GCTB
               if not Path(f'{prefix}{suffix}').is_file()]
    if missing:
        raise RuntimeError(f'GCTB output missing: {missing}')
    return prefix


def _count_true_ids(path: Path, truth_ids: set[str], column: int) -> tuple[int, int]:
    found = set()
    count = 0
    with path.open() as source:
        next(source)
        for line in source:
            fields = line.split()
            if len(fields) <= column:
                raise ValueError(f'malformed SNP row in {path}')
            count += 1
            if fields[column] in truth_ids:
                found.add(fields[column])
    return len(found), count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(directory: Path, ldm: Path) -> dict:
    truth = pd.read_csv(directory / 'truth.tsv', sep='\t')
    result = pd.read_csv(directory / 'rc.snpRes', sep=r'\s+')
    if not {'Name', 'PIP', 'Chrom', 'Position'}.issubset(result):
        raise ValueError('unexpected GCTB snpRes columns')
    if result.Name.duplicated().any() or not result.PIP.between(0, 1).all():
        raise ValueError('invalid GCTB SNP IDs or PIPs')
    ids = set(truth.SNP)
    result['true_causal'] = result.Name.isin(ids)
    gemma_true, gemma_total = _count_true_ids(
        directory / 'gemma/output/gemma_output.assoc.txt', ids, 1)
    ld_true, ld_total = _count_true_ids(ldm / 'snp.info', ids, 1)
    metrics = {'n_true': len(ids), 'n_true_in_panel': len(ids),
               'n_gemma': gemma_total, 'n_true_in_gemma': gemma_true,
               'n_ld': ld_total, 'n_true_in_ld': ld_true,
               'n_gctb': len(result),
               'n_true_in_gctb': int(result.true_causal.sum())}
    resources = directory / 'gctb.resources.json'
    if resources.is_file():
        usage = json.loads(resources.read_text())
        metrics['gctb_elapsed_seconds'] = usage['elapsed_seconds']
        metrics['gctb_peak_rss_gib'] = usage['max_rss_gib']
    if 'GelmanRubin_R' in result:
        rhat = pd.to_numeric(result.GelmanRubin_R, errors='coerce')
        metrics['rhat_gt_1_2'] = int((rhat > 1.2).sum())
        metrics['rhat_finite'] = int(np.isfinite(rhat).sum())
    lcs = pd.read_csv(directory / 'rc.lcs', sep=r'\s+')
    if not {'SNP', 'Size'}.issubset(lcs.columns):
        raise ValueError('unexpected GCTB local credible set columns')
    covered = set()
    for entry in lcs.SNP.astype(str):
        covered.update(entry.split(','))
    metrics['lcs_count'] = len(lcs)
    metrics['lcs_mean_size'] = float(lcs.Size.mean()) if len(lcs) else None
    metrics['lcs_causal_covered'] = len(ids & covered)
    metrics['lcs_causal_coverage'] = (len(ids & covered) / len(ids)
                                      if ids else None)
    metrics['lcs_sets_with_causal'] = int(sum(
        bool(ids & set(entry.split(','))) for entry in lcs.SNP.astype(str)))
    positions = {chrom: np.sort(group.POS.to_numpy(dtype=int))
                 for chrom, group in truth.groupby('CHROM')}
    prioritized = result[result.PIP >= 0.1].copy()
    prioritized['distance_to_causal_bp'] = np.nan
    if positions:
        for chrom, group in prioritized.groupby('Chrom'):
            targets = positions.get(chrom)
            if targets is None or not len(targets):
                continue
            query = group.Position.to_numpy(dtype=int)
            right = np.searchsorted(targets, query)
            lower = targets[np.maximum(right - 1, 0)]
            upper = targets[np.minimum(right, len(targets) - 1)]
            distance = np.minimum(abs(query - lower), abs(query - upper))
            prioritized.loc[group.index, 'distance_to_causal_bp'] = distance
    prioritized[['Name', 'Chrom', 'Position', 'PIP', 'true_causal',
                 'distance_to_causal_bp']].to_csv(
                     directory / 'prioritized.tsv', sep='\t', index=False)
    for threshold in (0.1, 0.5, 0.9):
        chosen = result[result.PIP >= threshold]
        hit = int(chosen.true_causal.sum())
        distances = prioritized.loc[prioritized.PIP >= threshold,
                                    'distance_to_causal_bp'].dropna()
        metrics[f'pip_{threshold:g}'] = {
            'selected': len(chosen), 'true_positive': hit,
            'recall': hit / len(ids) if ids else None,
            'fdr': (len(chosen) - hit) / len(chosen) if len(chosen) else None,
            'distance_unavailable_n': len(chosen) - len(distances),
            'median_distance_bp': float(distances.median()) if len(distances) else None,
            'p90_distance_bp': float(distances.quantile(0.9)) if len(distances) else None,
        }
    bins = np.linspace(0, 1, 11)
    result['pip_bin'] = pd.cut(result.PIP, bins, include_lowest=True)
    calibration = result.groupby('pip_bin', observed=False).agg(
        n=('PIP', 'size'), mean_pip=('PIP', 'mean'),
        true_discovery_rate=('true_causal', 'mean')).reset_index()
    calibration['pip_bin'] = calibration.pip_bin.astype(str)
    calibration.to_csv(directory / 'calibration.tsv', sep='\t', index=False)
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def aggregate(output_root: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    calibration_rows = []
    for marker in sorted(output_root.glob('*/COMPLETE')):
        run = marker.parent
        if not (run / 'design.json').is_file():
            continue
        design = json.loads((run / 'design.json').read_text())
        metrics = json.loads((run / 'metrics.json').read_text())
        rows.append({'run': run.name, **design,
                     'n_true_in_gemma': metrics['n_true_in_gemma'],
                     'n_true_in_ld': metrics['n_true_in_ld'],
                     'n_true_in_gctb': metrics['n_true_in_gctb'],
                     'n_gctb': metrics['n_gctb'],
                     'rhat_gt_1_2': metrics.get('rhat_gt_1_2'),
                     'gctb_elapsed_seconds': metrics.get('gctb_elapsed_seconds'),
                     'gctb_peak_rss_gib': metrics.get('gctb_peak_rss_gib'),
                     'selected_0.1': metrics['pip_0.1']['selected'],
                     'selected_0.5': metrics['pip_0.5']['selected'],
                     'selected_0.9': metrics['pip_0.9']['selected'],
                     'recall_0.1': metrics['pip_0.1']['recall'],
                     'recall_0.5': metrics['pip_0.5']['recall'],
                     'recall_0.9': metrics['pip_0.9']['recall'],
                     'fdr_0.1': metrics['pip_0.1']['fdr'],
                     'fdr_0.5': metrics['pip_0.5']['fdr'],
                     'fdr_0.9': metrics['pip_0.9']['fdr'],
                     'median_distance_0.1_bp': metrics['pip_0.1']['median_distance_bp'],
                     'median_distance_0.5_bp': metrics['pip_0.5']['median_distance_bp'],
                     'median_distance_0.9_bp': metrics['pip_0.9']['median_distance_bp'],
                     'lcs_coverage': metrics['lcs_causal_coverage'],
                     'lcs_mean_size': metrics['lcs_mean_size'],
                     'lcs_count': metrics['lcs_count']})
        frame = pd.read_csv(run / 'calibration.tsv', sep='\t')
        frame['run'] = run.name
        frame['architecture'] = design['architecture']
        frame['n_causal'] = design['n_causal']
        frame['causal_var'] = design['causal_var']
        calibration_rows.append(frame)
    if not rows:
        raise ValueError(f'no completed benchmark runs in {output_root}')
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / 'run_metrics.tsv', sep='\t', index=False)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    calibration.to_csv(output_root / 'calibration_by_run.tsv',
                       sep='\t', index=False)
    selected = calibration[calibration.n > 0].copy()
    selected['pip_sum'] = selected.n * selected.mean_pip
    selected['true_sum'] = selected.n * selected.true_discovery_rate
    grouped = selected.groupby(
        ['architecture', 'n_causal', 'causal_var', 'pip_bin'],
        sort=False)[['n', 'pip_sum', 'true_sum']].sum().reset_index()
    grouped['mean_pip'] = grouped.pip_sum / grouped.n
    grouped['true_discovery_rate'] = grouped.true_sum / grouped.n
    grouped['evidence_sufficient'] = grouped.n >= 30
    grouped.drop(columns=['pip_sum', 'true_sum']).to_csv(
        output_root / 'pip_calibration_summary.tsv', sep='\t', index=False)
    scenarios = [('gaussian', 50), ('gaussian', 500),
                 ('gaussian', 5000), ('large', 500), ('ldms', 500)]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    for ax, (architecture, n_causal) in zip(axes, scenarios):
        points = grouped[(grouped.architecture == architecture) &
                         (grouped.n_causal == n_causal) &
                         (grouped.causal_var == 0.15)]
        if not points.empty:
            ax.scatter(points.mean_pip, points.true_discovery_rate,
                       s=np.clip(points.n / 100, 15, 100))
            for row in points.itertuples():
                ax.annotate(f'n={row.n}' if row.n >= 30 else f'n={row.n}*',
                            (row.mean_pip, row.true_discovery_rate),
                            xytext=(2, 3), textcoords='offset points',
                            fontsize=7)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
        ax.set_title(f'{architecture}, {n_causal} causal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Mean PIP in bin')
    axes[0].set_ylabel('True discovery rate')
    fig.suptitle('Causal variance 0.15; * fewer than 30 SNPs in bin')
    fig.tight_layout()
    fig.savefig(output_root / 'pip_calibration.png', dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('run-one')
    one.add_argument('--panel', type=Path, default=BENCH / 'panel_all_variants')
    one.add_argument('--ldm', type=Path, required=True)
    one.add_argument('--output', type=Path, required=True)
    one.add_argument('--architecture', choices=['gaussian', 'large', 'ldms', 'null'],
                     required=True)
    one.add_argument('--count', type=int, required=True)
    one.add_argument('--causal-var', type=float, required=True)
    one.add_argument('--seed', type=int, required=True)
    one.add_argument('--ld-scores', type=Path)
    one.add_argument('--gemma-workers', type=int, default=40)
    one.add_argument('--gctb-threads', type=int, default=32)
    report = sub.add_parser('aggregate')
    report.add_argument('--output-root', type=Path,
                        default=BENCH / 'runs')
    block = sub.add_parser('block-map')
    block.add_argument('--panel', type=Path, default=BENCH / 'panel_all_variants')
    block.add_argument('--output', type=Path, required=True)
    block.add_argument('--max-snps', type=int, default=1500)
    bp_block = sub.add_parser('bp-block-map')
    bp_block.add_argument('--panel', type=Path, default=BENCH / 'panel_all_variants')
    bp_block.add_argument('--output', type=Path, required=True)
    bp_block.add_argument('--width-bp', type=int, default=15_000)
    audit = sub.add_parser('audit-boundaries')
    audit.add_argument('--panel', type=Path, default=BENCH / 'panel_all_variants')
    audit.add_argument('--blocks', type=Path, required=True)
    audit.add_argument('--output', type=Path, required=True)
    ldscore = sub.add_parser('ld-scores')
    ldscore.add_argument('--pairs', type=Path, required=True)
    ldscore.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run-one':
        if args.architecture == 'null' and args.count != 0:
            parser.error('null architecture requires count 0')
        if args.architecture != 'null' and args.count <= 0:
            parser.error('nonnull architecture requires a positive causal count')
        if args.architecture == 'null' and args.causal_var != 0:
            parser.error('null architecture requires causal-var 0')
        if args.architecture != 'null' and not 0 < args.causal_var <= 0.3:
            parser.error('causal-var must be in (0, 0.3]')
        if (args.output / 'COMPLETE').is_file():
            print(f'Already complete: {args.output}')
            return
        phenotype = args.output / 'trait.gemma.pheno'
        if phenotype.is_file():
            saved = json.loads((args.output / 'design.json').read_text())
            if (saved['architecture'], saved['n_causal'],
                    saved['causal_var'], saved['seed']) != (
                    args.architecture, args.count, args.causal_var, args.seed):
                raise ValueError('existing phenotype belongs to another design')
        else:
            phenotype = make_phenotype(args.panel, args.output,
                                       args.architecture, args.count,
                                       args.causal_var, args.seed, args.ld_scores)
        assoc = run_gemma(args.panel, args.output, phenotype,
                          args.gemma_workers)
        write_gctb_summary(args.panel, assoc, args.output / 'sumstats.ma')
        run_gctb(args.output, args.ldm, args.gctb_threads)
        (args.output / 'inputs.json').write_text(json.dumps({
            'panel': str(args.panel),
            'panel_checksums_sha256': _sha256(args.panel / 'SHA256SUMS')
                if (args.panel / 'SHA256SUMS').is_file() else None,
            'kinship': str(KINSHIP), 'kinship_sha256': _sha256(KINSHIP),
            'ldm': str(args.ldm),
            'ldm_info_sha256': _sha256(args.ldm / 'ldm.info'),
            'annotation_sha256': _sha256(args.ldm / 'intercept_only.txt'),
            'gemma_assoc': str(assoc), 'gctb': str(SOFTWARE / 'gctb')
        }, indent=2) + '\n')
        metrics = evaluate(args.output, args.ldm)
        (args.output / 'COMPLETE').write_text('complete\n')
        print(json.dumps(metrics, indent=2))
    elif args.command == 'aggregate':
        aggregate(args.output_root)
    elif args.command == 'block-map':
        make_block_map(args.panel, args.output, args.max_snps)
    elif args.command == 'bp-block-map':
        make_bp_block_map(args.panel, args.output, args.width_bp)
    elif args.command == 'audit-boundaries':
        audit_block_boundaries(args.panel, args.blocks, args.output)
    elif args.command == 'ld-scores':
        make_ld_scores(args.pairs, args.output)


if __name__ == '__main__':
    main()
