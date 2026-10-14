"""Prepare PLINK-pruned SNPs, genetic distances and covariance kinship."""

import os
import subprocess
import tempfile
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
from Bio import Phylo

from .vcf import get_sample_list_from_vcf


def prune_snps_with_plink(input_vcf_file, output_prefix, *, min_maf=0.01,
                          ld_window_kb=500, ld_step=50, r2_threshold=0.2,
                          threads=20, memory_mb=16000, plink='plink'):
    """Import and LD-prune the project's diploid Chr01-Chr10 SNPs with PLINK 1.9.

    Workflow
    --------
    1. Import biallelic A/C/G/T SNPs and retain MAF >= min_maf.
    2. Map chromosome names to 1-10 and give every SNP a unique allele-based ID.
    3. Run --indep-pairwise and extract retained SNPs into BED/BIM/FAM.
    4. Export an indexed VCF with original chromosome names and sample order.

    Parameters
    ----------
    input_vcf_file : str
        Imputed landrace VCF, with unique sample names and Chr01-Chr10 contigs.
        No heterozygosity, missingness or HWE threshold is applied.
    output_prefix : str
        Full parameterized basename supplied by the notebook. Writes
        .bed/.bim/.fam, .prune.in/.prune.out and .vcf.gz with a TBI index.
    min_maf : float
        Inclusive minor-allele frequency threshold.
    ld_window_kb, ld_step : int
        PLINK physical window width in kb and step in variant counts.
    r2_threshold : float
        Pairwise LD threshold for --indep-pairwise.
    threads, memory_mb : int
        PLINK threads and workspace memory in MB, within compute resources.
    plink : str
        PLINK 1.9 command or executable path; bcftools must also be on PATH.

    Returns
    -------
    str
        Pruned VCF path. Numeric chromosome codes remain in BIM. SNP IDs are
        CHROM:POS:A1:A2 in the PLINK panel and exported VCF; other original
        VCF annotations are not retained. Logs are saved beside the outputs.
        The caller owns reuse; complete outputs are replaced on a fresh call.
    """
    input_vcf_file = os.path.abspath(input_vcf_file)
    output_prefix = os.path.abspath(output_prefix)
    output_dir = os.path.dirname(output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    sample_ids = get_sample_list_from_vcf(input_vcf_file)
    if not pd.Index(sample_ids).is_unique:
        raise ValueError('VCF sample IDs must be unique.')
    common = ['--chr-set', '10', 'no-x', 'no-y', 'no-xy', 'no-mt',
              '--keep-allele-order', '--threads', str(threads), '--memory', str(memory_mb)]
    start_time = perf_counter()

    with tempfile.TemporaryDirectory(prefix='plink_prune_', dir=output_dir) as temp_dir:
        sample_order_file = f'{temp_dir}/sample_order.tsv'
        pd.DataFrame({'FID': sample_ids, 'IID': sample_ids}).to_csv(
            sample_order_file, sep='\t', index=False, header=False)
        maf_prefix = f'{temp_dir}/maf'
        print(f'[{datetime.now():%H:%M:%S}] Importing SNPs with MAF >= {min_maf}', flush=True)
        subprocess.run([
            plink, '--vcf', input_vcf_file, '--double-id', '--allow-extra-chr',
            '--biallelic-only', 'strict', '--snps-only', 'just-acgt',
            '--maf', str(min_maf), '--indiv-sort', 'file', sample_order_file,
            '--make-bed', '--out', maf_prefix, *common,
        ], check=True)
        os.replace(f'{maf_prefix}.log', f'{output_prefix}.maf.log')
        bim = pd.read_csv(f'{maf_prefix}.bim', sep=r'\s+', header=None, dtype=str)
        chromosome_map = {f'Chr{i:02d}': str(i) for i in range(1, 11)}
        plink_chromosome_map = {**chromosome_map, **{str(i): str(i) for i in range(1, 11)}}
        if not bim[0].isin(plink_chromosome_map).all():
            raise ValueError('Expected Chr01-Chr10 SNP chromosome names.')
        bim[0] = bim[0].map(plink_chromosome_map)
        project_chromosomes = bim[0].map({str(i): f'Chr{i:02d}' for i in range(1, 11)})
        bim[1] = project_chromosomes + ':' + bim[3] + ':' + bim[4] + ':' + bim[5]
        if bim[1].duplicated().any():
            raise ValueError('Duplicate SNP coordinates/alleles make PLINK extraction ambiguous.')
        bim.to_csv(f'{maf_prefix}.bim', sep='\t', index=False, header=False)
        print(f'[{datetime.now():%H:%M:%S}] LD pruning {len(bim):,} SNPs', flush=True)
        pruned_prefix = f'{temp_dir}/pruned'
        subprocess.run([
            plink, '--bfile', maf_prefix, '--indep-pairwise', f'{ld_window_kb}kb',
            str(ld_step), str(r2_threshold), '--out', pruned_prefix, *common,
        ], check=True)
        os.replace(f'{pruned_prefix}.log', f'{output_prefix}.pruning.log')
        subprocess.run([
            plink, '--bfile', maf_prefix, '--extract', f'{pruned_prefix}.prune.in',
            '--indiv-sort', 'file', sample_order_file, '--make-bed',
            '--out', pruned_prefix, *common,
        ], check=True)
        os.replace(f'{pruned_prefix}.log', f'{output_prefix}.extract.log')
        fam = pd.read_csv(f'{pruned_prefix}.fam', sep=r'\s+', header=None, dtype=str)
        if fam[1].tolist() != sample_ids:
            raise ValueError('PLINK panel and input VCF sample orders differ.')
        # PLINK disallows --indiv-sort with --recode; export the ordered panel separately.
        subprocess.run([
            plink, '--bfile', pruned_prefix, '--recode', 'vcf-iid', 'bgz',
            '--out', pruned_prefix, *common,
        ], check=True)
        os.replace(f'{pruned_prefix}.log', f'{output_prefix}.export.log')
        chromosome_map_file = f'{temp_dir}/chromosome_map.tsv'
        pd.DataFrame([(value, key) for key, value in chromosome_map.items()]).to_csv(
            chromosome_map_file, sep='\t', index=False, header=False)
        pruned_vcf = f'{temp_dir}/project_chr.vcf.gz'
        subprocess.run([
            'bcftools', 'annotate', '--rename-chrs', chromosome_map_file,
            f'{pruned_prefix}.vcf.gz', '-Oz', '-o', pruned_vcf,
        ], check=True)
        subprocess.run(['bcftools', 'index', '-t', pruned_vcf], check=True)
        if get_sample_list_from_vcf(pruned_vcf) != sample_ids:
            raise ValueError('Exported VCF and PLINK panel sample orders differ.')
        for suffix in ('bed', 'bim', 'fam', 'prune.in', 'prune.out'):
            os.replace(f'{pruned_prefix}.{suffix}', f'{output_prefix}.{suffix}')
        os.replace(pruned_vcf, f'{output_prefix}.vcf.gz')
        os.replace(f'{pruned_vcf}.tbi', f'{output_prefix}.vcf.gz.tbi')
    print(f'[{datetime.now():%H:%M:%S}] Pruned SNP panel ready ({perf_counter() - start_time:.1f} s)', flush=True)
    return f'{output_prefix}.vcf.gz'


def calculate_plink_kinship(bfile_prefix, *, threads=20, memory_mb=16000, plink='plink'):
    """Calculate PLINK IBS distance and unstandardized covariance kinship.

    Parameters
    ----------
    bfile_prefix : str
        LD-pruned BED/BIM/FAM basename; chromosomes use numeric codes 1-10.
    threads, memory_mb : int
        PLINK threads and workspace memory in MB.
    plink : str
        PLINK 1.9 command or executable path.

    Returns
    -------
    tuple of str
        Paths to .ibs.mdist and .grm_cov.rel. Runs --distance 1-ibs square
        and --make-rel square cov, respectively. The cov modifier disables
        per-SNP variance standardization. Saves corresponding .id files and
        logs; checks matrix dimensions, symmetry, finite values and IID order
        against FAM. The caller owns reuse; fresh calls replace matrices.
    """
    common = [plink, '--bfile', bfile_prefix, '--chr-set', '10',
              'no-x', 'no-y', 'no-xy', 'no-mt', '--threads', str(threads),
              '--memory', str(memory_mb)]
    sample_ids = pd.read_csv(f'{bfile_prefix}.fam', sep=r'\s+', header=None, dtype=str)[1].tolist()
    distance_file = f'{bfile_prefix}.ibs.mdist'
    kinship_file = f'{bfile_prefix}.grm_cov.rel'
    start_time = perf_counter()
    # Publish both complete matrices together so interrupted jobs cannot be reused.
    with tempfile.TemporaryDirectory(prefix='plink_kinship_', dir=os.path.dirname(os.path.abspath(bfile_prefix))) as temp_dir:
        for label, command, suffix in (
            ('ibs', ['--distance', '1-ibs', 'square'], 'mdist'),
            ('grm_cov', ['--make-rel', 'square', 'cov'], 'rel'),
        ):
            prefix = f'{temp_dir}/{label}'
            print(f'[{datetime.now():%H:%M:%S}] Calculating {label}', flush=True)
            subprocess.run([*common, *command, '--out', prefix], check=True)
            ids = pd.read_csv(f'{prefix}.{suffix}.id', sep=r'\s+', header=None, dtype=str)[1].tolist()
            matrix = np.loadtxt(f'{prefix}.{suffix}', ndmin=2)
            if ids != sample_ids or matrix.shape != (len(ids), len(ids)):
                raise ValueError(f'{label} matrix sample order/dimensions differ from FAM.')
            if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
                raise ValueError(f'{label} matrix must be finite and symmetric.')
        for label, suffix in (('ibs', 'mdist'), ('grm_cov', 'rel')):
            for extension in (suffix, f'{suffix}.id', 'log'):
                os.replace(f'{temp_dir}/{label}.{extension}', f'{bfile_prefix}.{label}.{extension}')
    print(f'[{datetime.now():%H:%M:%S}] PLINK matrices ready ({perf_counter() - start_time:.1f} s)', flush=True)
    return distance_file, kinship_file


def build_nj_tree(distance_file, sample_id_file, output_prefix, *, threads=20, fastme='fastme'):
    """Infer a pure neighbor-joining tree from a PLINK square IBS distance matrix.

    Parameters
    ----------
    distance_file, sample_id_file : str
        PLINK .mdist and .mdist.id; second ID column contains original VCF IDs.
    output_prefix : str
        Write .phylip and .nj.tree beside the distance matrix.
    threads : int
        FastME thread count within the compute resources.
    fastme : str
        FastME 2.x command or executable path.

    Returns
    -------
    str
        Newick tree path. Writes PHYLIP distances, runs FastME -m N -w n
        without NNI and verifies all tip names. The caller owns result reuse.
    """
    sample_ids = pd.read_csv(sample_id_file, sep=r'\s+', header=None, dtype=str)[1].tolist()
    distance = np.loadtxt(distance_file, ndmin=2)
    if (len(sample_ids) < 2 or not pd.Index(sample_ids).is_unique
            or any(any(char.isspace() for char in name) for name in sample_ids)):
        raise ValueError('NJ requires at least two unique sample IDs without whitespace.')
    if (distance.shape != (len(sample_ids), len(sample_ids))
            or not np.isfinite(distance).all() or not np.allclose(distance, distance.T)
            or not np.allclose(np.diag(distance), 0)):
        raise ValueError('NJ requires a finite symmetric distance matrix with zero diagonal.')
    start_time = perf_counter()
    with tempfile.TemporaryDirectory(prefix='fastme_', dir=os.path.dirname(os.path.abspath(output_prefix))) as temp_dir:
        phylip_file = f'{temp_dir}/distance.phylip'
        tree_file = f'{temp_dir}/distance.nj.tree'
        with open(phylip_file, 'w') as handle:
            handle.write(f'{len(sample_ids)}\n')
            for name, row in zip(sample_ids, distance):
                handle.write(name + ' ' + ' '.join(f'{value:.10g}' for value in row) + '\n')
        print(f'[{datetime.now():%H:%M:%S}] Building NJ tree with FastME', flush=True)
        subprocess.run([
            fastme, '-i', phylip_file, '-m', 'N', '-w', 'n',
            '-T', str(threads), '-o', tree_file,
        ], check=True)
        tree = Phylo.read(tree_file, 'newick')
        if sorted(tip.name for tip in tree.get_terminals()) != sorted(sample_ids):
            raise ValueError('NJ tree tips differ from PLINK sample IDs.')
        os.replace(phylip_file, f'{output_prefix}.phylip')
        os.replace(tree_file, f'{output_prefix}.nj.tree')
    print(f'[{datetime.now():%H:%M:%S}] NJ tree ready ({perf_counter() - start_time:.1f} s)', flush=True)
    return f'{output_prefix}.nj.tree'
