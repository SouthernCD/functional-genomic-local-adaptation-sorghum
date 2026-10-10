from .utils import cmd_run
from .vcf import build_var_stat_table, convert_vcf_to_bimbam, get_chr_id_list_from_vcf, read_hdf_with_retry, get_sample_list_from_vcf
from .feature import read_gff_file, get_gene_range_df, read_fasta_by_faidx
from .plot import set_plot_style
from statsmodels.sandbox.stats.multicomp import multipletests
import pandas as pd
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from itertools import zip_longest
import fcntl
import hashlib
import json
import os
from glob import glob
import shlex
import shutil
import subprocess
import tempfile
from cyvcf2 import VCF


def prepare_gemma_phenotype_file(env_name, phenotype_df, phenotype_file):
    """Write one environmental phenotype in the supplied genotype sample order.

    Parameters
    ----------
    env_name : str
        Environmental column to export.
    phenotype_df : pandas.DataFrame
        Rows already aligned to VCF samples; sample IDs are the index.
    phenotype_file : str
        BIMBAM phenotype path, one value per row; missing values use NA.
        Reuse an existing file only if values match the supplied sample order.
        A mismatch raises rather than overwriting phenotypes of existing GWAS.

    Returns
    -------
    str
        Phenotype path; new files are written through a temporary sibling.
    """
    values = phenotype_df[env_name].to_numpy(dtype=float)
    if os.path.exists(phenotype_file):
        previous = pd.read_csv(phenotype_file, header=None, float_precision='round_trip')[0].to_numpy(dtype=float)
        if previous.shape != values.shape or not np.allclose(previous, values, rtol=0, atol=0, equal_nan=True):
            raise ValueError(f'Existing phenotype differs from aligned metadata: {phenotype_file}')
        return phenotype_file
    temporary_file = f'{phenotype_file}.tmp'
    phenotype_df[[env_name]].to_csv(temporary_file, index=False, header=False, na_rep='NA')
    os.replace(temporary_file, phenotype_file)
    return phenotype_file


def gemma_result_parser(gemma_output_assoc_file, var_stat_h5, chr_list):
    """Annotate GEMMA associations with variant positions and global FDR.

    Parameters
    ----------
    gemma_output_assoc_file : str
        Merged GEMMA association file; P_VALUE is the Wald-test P value.
    var_stat_h5 : str or None
        Chromosome-keyed VCF statistics for ID/CHROM/POS/MAF/REF/ALT.
    chr_list : sequence of str
        Keys to read from the variant-statistics table.

    Returns
    -------
    pandas.DataFrame
        Associations with Benjamini–Hochberg FDR over all merged variants,
        sorted by chromosome and position when coordinates are available.
    """

    print(f'Parsing {gemma_output_assoc_file}...')
    print(f'Loading gemma_output_assoc_file...')
    gemma_output_assoc_df = pd.read_csv(gemma_output_assoc_file, sep='\t')
    required_columns = ['rs', 'af', 'beta', 'se', 'p_wald']
    missing_columns = [column for column in required_columns
                       if column not in gemma_output_assoc_df.columns]
    if missing_columns:
        raise ValueError(f'Missing GEMMA association columns: {missing_columns}')
    gemma_output_assoc_df = gemma_output_assoc_df.reindex(columns=[
        'rs', 'af', 'beta', 'se', 'logl_H1', 'l_remle', 'l_mle',
        'p_wald', 'p_lrt', 'p_score'])
    gemma_output_assoc_df = gemma_output_assoc_df.rename(
        columns={'rs': 'ID', 'af': 'MAF', 'beta': 'BETA', 'se': 'SE', 'logl_H1': 'LOGL_H1', 'l_remle': 'L_REMLE', 'l_mle': 'L_MLE', 'p_wald': 'P_WALD', 'p_lrt': 'P_LRT', 'p_score': 'P_SCORE'})
    gemma_output_assoc_df['P_VALUE'] = gemma_output_assoc_df['P_WALD']
    gemma_output_assoc_df['MAF'] = gemma_output_assoc_df['MAF'].apply(
        lambda x: 1-x if x > 0.5 else x)

    print(f'Calculating FDR...')
    gemma_output_assoc_df['FDR'] = multipletests(
        gemma_output_assoc_df['P_VALUE'], method='fdr_bh')[1]
    # gemma_output_assoc_df.sort_values('FDR')

    if var_stat_h5 is None:
        return gemma_output_assoc_df

    print(f'Loading var_stat_h5...')
    chr_pos_df_list = []
    for chr_name in chr_list:
        # print(f'Loading {chr_name}...')
        chr_pos_df = read_hdf_with_retry(var_stat_h5, chr_name)
        if len(chr_pos_df) == 0:
            continue
        chr_pos_df = chr_pos_df[['ID', 'CHROM', 'POS', 'MAF', 'REF', 'ALT']]
        chr_pos_df_list.append(chr_pos_df)

    print(f'Merging...')
    chr_pos_df = pd.concat(chr_pos_df_list)
    gemma_output_assoc_df = gemma_output_assoc_df.drop(columns=['MAF'])
    merged_df = pd.merge(gemma_output_assoc_df,
                         chr_pos_df, on='ID', how='left')
    merged_df = merged_df.sort_values(['CHROM', 'POS'], kind='stable').reset_index(drop=True)

    return merged_df


def _split_gemma_bimbam(input_vcf_file, bimbam_file, block_dir, block_size):
    """Split BIMBAM rows by the matching VCF record's genomic 1-based position."""
    os.makedirs(block_dir, exist_ok=True)
    blocks = OrderedDict()
    handles = OrderedDict()
    initialized = set()
    vcf = VCF(input_vcf_file)
    try:
        with open(bimbam_file) as bimbam:
            for index, pair in enumerate(zip_longest(vcf, bimbam), 1):
                record, line = pair
                if record is None or line is None:
                    raise ValueError('VCF and BIMBAM have different variant counts')
                if line.split(',', 1)[0] != record.ID:
                    raise ValueError(f'VCF/BIMBAM variant mismatch at row {index}')
                key = (record.CHROM, (record.POS - 1) // block_size)
                if key not in blocks:
                    blocks[key] = os.path.join(block_dir, f'block_{len(blocks):05d}')
                path = blocks[key] + '.bimbam'
                if path not in handles:
                    if len(handles) == 32:
                        _, old_handle = handles.popitem(last=False)
                        old_handle.close()
                    handles[path] = open(path, 'a' if path in initialized else 'w')
                    initialized.add(path)
                handles.move_to_end(path)
                handles[path].write(line)
    finally:
        vcf.close()
        for handle in handles.values():
            handle.close()
    return list(blocks.values())


def _gemma_input_signature(path):
    stat = os.stat(path)
    return {'path': os.path.realpath(path), 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def _get_gemma_blocks(input_vcf_file, bimbam_file, block_size, block_cache_dir=None):
    """Reuse a complete, immutable split when its input signatures match."""
    signature = {'version': 1, 'block_size': block_size,
                 'vcf': _gemma_input_signature(input_vcf_file),
                 'bimbam': _gemma_input_signature(bimbam_file)}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_root = os.path.abspath(block_cache_dir or bimbam_file + '.gemma_blocks')
    cache_dir = os.path.join(cache_root, digest)
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, '.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pointer_file = os.path.join(cache_dir, 'current.json')
        try:
            with open(pointer_file) as source:
                revision = json.load(source)['revision']
            revision_dir = os.path.join(cache_dir, revision)
            if not revision.startswith('revision_') or os.path.basename(revision) != revision:
                raise ValueError('Invalid cache revision')
            with open(os.path.join(revision_dir, 'manifest.json')) as source:
                manifest = json.load(source)
            if manifest['signature'] != signature or not manifest['blocks']:
                raise ValueError('Cache parameters changed')
            for block in manifest['blocks']:
                if os.path.basename(block['name']) != block['name']:
                    raise ValueError('Invalid cache block name')
                path = os.path.join(revision_dir, block['name'])
                stat = os.stat(path)
                if stat.st_size != block['size'] or stat.st_mtime_ns != block['mtime_ns']:
                    raise ValueError(f'Incomplete GEMMA block: {path}')
        except (OSError, ValueError, KeyError, TypeError):
            revision_dir = tempfile.mkdtemp(prefix='revision_', dir=cache_dir)
            prefixes = _split_gemma_bimbam(input_vcf_file, bimbam_file,
                                            revision_dir, block_size)
            if not prefixes:
                raise ValueError('VCF has no variants')
            if (_gemma_input_signature(input_vcf_file) != signature['vcf'] or
                    _gemma_input_signature(bimbam_file) != signature['bimbam']):
                raise RuntimeError('VCF or BIMBAM changed during GEMMA block creation')
            manifest = {'signature': signature, 'blocks': []}
            for prefix in prefixes:
                path = prefix + '.bimbam'
                stat = os.stat(path)
                manifest['blocks'].append({'name': os.path.basename(path),
                                           'size': stat.st_size,
                                           'mtime_ns': stat.st_mtime_ns})
            with open(os.path.join(revision_dir, 'manifest.json'), 'w') as target:
                json.dump(manifest, target)
            temporary_pointer = pointer_file + '.tmp'
            with open(temporary_pointer, 'w') as target:
                json.dump({'revision': os.path.basename(revision_dir)}, target)
            os.replace(temporary_pointer, pointer_file)
            print(f'Created {len(prefixes)} GEMMA blocks in {revision_dir}')
        else:
            print(f'Reusing {len(manifest["blocks"])} GEMMA blocks from {revision_dir}')
        return [os.path.join(revision_dir, block['name'])
                for block in manifest['blocks']]


def _run_gemma_block(bimbam_file, block_dir, phenotype_file, kinship_file,
                     retry, lmm):
    os.makedirs(block_dir, exist_ok=True)
    command = ['gemma', '-g', bimbam_file, '-k', kinship_file,
               '-lmm', str(lmm), '-miss', '0.1', '-p', phenotype_file,
               '-o', 'gemma_output']
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    assoc_file = os.path.join(block_dir, 'output', 'gemma_output.assoc.txt')
    for attempt in range(max(1, retry)):
        with open(os.path.join(block_dir, 'gemma.log'), 'a') as log:
            result = subprocess.run(command, cwd=block_dir, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode == 0 and os.path.exists(assoc_file):
            return assoc_file
        if attempt + 1 < max(1, retry):
            sleep(30)
    raise RuntimeError(f'GEMMA failed for {bimbam_file}; see {block_dir}/gemma.log')


def _run_gemma_parallel(work_dir, bimbam_file, phenotype_file, kinship_file,
                        input_vcf_file, retry, dry_run, block_size, max_workers,
                        block_cache_dir, lmm):
    if block_size <= 0 or max_workers <= 0:
        raise ValueError('block_size and max_workers must be positive')
    bimbam_file = os.path.abspath(bimbam_file)
    phenotype_file = os.path.abspath(phenotype_file)
    kinship_file = os.path.abspath(kinship_file)
    input_vcf_file = os.path.abspath(input_vcf_file)
    work_dir = os.path.abspath(work_dir)
    block_files = _get_gemma_blocks(input_vcf_file, bimbam_file, block_size,
                                    block_cache_dir)
    blocks = [(block_file, os.path.join(work_dir, 'blocks',
               os.path.basename(block_file).removesuffix('.bimbam') + '.gemma'))
              for block_file in block_files]
    with open(os.path.join(work_dir, 'gemma_cmd.sh'), 'w') as script:
        script.write('#!/bin/sh\nset -eu\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n')
        for block_file, block_dir in blocks:
            command = ['gemma', '-g', block_file, '-k', kinship_file,
                       '-lmm', str(lmm), '-miss', '0.1', '-p', phenotype_file,
                       '-o', 'gemma_output']
            script.write(f'mkdir -p {shlex.quote(block_dir)}\n'
                         f'(cd {shlex.quote(block_dir)} && '
                         f'{shlex.join(command)})\n')
    if dry_run:
        return None
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_gemma_block, block_file, block_dir,
                                   phenotype_file, kinship_file, retry, lmm): block_file
                   for block_file, block_dir in blocks}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    output_dir = os.path.join(work_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    assoc_file = os.path.join(output_dir, 'gemma_output.assoc.txt')
    temporary_file = assoc_file + '.tmp'
    with open(temporary_file, 'w') as merged:
        for index, (block_file, _) in enumerate(blocks):
            with open(results[block_file]) as source:
                header = next(source)
                if index == 0:
                    merged.write(header)
                for line in source:
                    merged.write(line)
    os.replace(temporary_file, assoc_file)
    return assoc_file


def _check_gemma_lmm_output(work_dir, lmm):
    metadata_file = os.path.join(work_dir, 'gemma_run.json')
    if os.path.exists(metadata_file):
        with open(metadata_file) as source:
            previous_lmm = json.load(source)['lmm']
    else:
        previous_lmm = 4  # Results made before the metadata file used -lmm 4.
    if previous_lmm != lmm:
        raise ValueError(f'Existing GEMMA output in {work_dir} uses -lmm {previous_lmm}; '
                         f'use a different name or work_dir for -lmm {lmm}')


def gemma_run(name, work_dir, bimbam_file, phenotype_file, kinship_file, input_vcf_file=None, genome_chr_list=None, retry=3, dry_run=False, vcf_var_stat_h5=None, block_size=10_000_000, max_workers=80, block_cache_dir=None, lmm=4):
    """Run GEMMA LMM blocks, merge associations and save one result HDF.

    Parameters
    ----------
    name, work_dir : str
        Analysis name and GEMMA directory; output is <name>.gemma.gwas_df.h5.
    bimbam_file, phenotype_file, kinship_file : str
        Genotypes, phenotype and relatedness matrix in identical sample order.
    input_vcf_file : str or None
        VCF for block partitioning and variant-coordinate annotation.
    genome_chr_list : sequence or None
        Chromosome keys of the variant-statistics HDF.
    retry : int
        Maximum attempts per GEMMA block.
    dry_run : bool
        Prepare commands without running GEMMA.
    vcf_var_stat_h5 : str or None
        Optional precomputed variant-statistics HDF.
    block_size : int
        Physical block width in bp; blocks reuse the same full kinship matrix.
    max_workers : int
        Concurrent GEMMA processes; each uses one OpenMP/OpenBLAS thread.
    block_cache_dir : str or None
        Optional shared BIMBAM block cache.
    lmm : int
        GEMMA test mode: 1 for Wald or 4 for all tests. P_VALUE uses p_wald;
        FDR is Benjamini–Hochberg across merged tested variants.

    Returns
    -------
    str
        Result HDF path (key df), or association path for raw/dry-run modes.
        Existing complete outputs are reused; failed block execution raises.
        After saving the result HDF, remove per-variable block outputs while
        retaining the merged association table and one GEMMA log in output/.
        Shared BIMBAM input blocks are retained for other variables.
    """
    if lmm not in (1, 4):
        raise ValueError('lmm must be 1 (Wald) or 4 (all tests)')

    os.makedirs(work_dir, exist_ok=True)


    gemma_output_assoc_file = f'{work_dir}/output/gemma_output.assoc.txt'
    gwas_df_h5 = f'{work_dir}/{name}.gemma.gwas_df.h5'
    if os.path.exists(gwas_df_h5):
        _check_gemma_lmm_output(work_dir, lmm)
    assoc_exists = os.path.exists(gemma_output_assoc_file)
    if assoc_exists:
        _check_gemma_lmm_output(work_dir, lmm)
        print(
            f'GEMMA GWAS result already exists.')
    elif not dry_run:
        metadata_file = os.path.join(work_dir, 'gemma_run.json')
        with open(metadata_file, 'w') as target:
            json.dump({'lmm': lmm}, target)
    if not assoc_exists and input_vcf_file is not None:
        _run_gemma_parallel(work_dir, bimbam_file, phenotype_file, kinship_file,
                            input_vcf_file, retry, dry_run, block_size, max_workers,
                            block_cache_dir, lmm)
    elif not assoc_exists and dry_run:
        with open(f'{work_dir}/gemma_cmd.sh', 'w') as f:
            f.write(
                f"gemma -g {bimbam_file} -k {kinship_file} -lmm {lmm} -miss 0.1 -p {phenotype_file} -o gemma_output -outdir {work_dir}")
    elif not assoc_exists:
        with open(f'{work_dir}/gemma_cmd.sh', 'w') as f:
            f.write(
                f"gemma -g {bimbam_file} -k {kinship_file} -lmm {lmm} -miss 0.1 -p {phenotype_file} -o gemma_output -outdir {work_dir}")

        f, o, e = cmd_run(
            f"gemma -g {bimbam_file} -k {kinship_file} -lmm {lmm} -miss 0.1 -p {phenotype_file} -o gemma_output", cwd=work_dir)

        retry_count = 0
        if not os.path.exists(gemma_output_assoc_file):
            print(f, o, e)
            retry_count += 1
            if retry_count < retry:
                print(
                    f'GEMMA GWAS failed, retrying {retry_count}...')
                sleep(30)
                cmd_run(
                    f"gemma -g {bimbam_file} -k {kinship_file} -lmm {lmm} -miss 0.1 -p {phenotype_file} -o gemma_output", cwd=work_dir)
            else:
                raise FileNotFoundError(
                    f'GEMMA GWAS failed: {gemma_output_assoc_file}')

    if input_vcf_file is None and vcf_var_stat_h5 is None:
        return gemma_output_assoc_file
    if dry_run:
        return gemma_output_assoc_file


    if not os.path.exists(gwas_df_h5):
        gwas_df = gemma_result_parser(
            gemma_output_assoc_file, f'{input_vcf_file}.var_stat.h5' if vcf_var_stat_h5 is None else vcf_var_stat_h5, genome_chr_list)
        gwas_df.to_hdf(
            gwas_df_h5, key='df', mode='w')

    block_output_dir = f'{work_dir}/blocks'
    if os.path.exists(block_output_dir):
        # Keep the sample-count log for MAGMA; shared input blocks are separate.
        saved_log_file = f'{work_dir}/output/gemma_output.log.txt'
        if not os.path.exists(saved_log_file):
            gemma_log_file = sorted(glob(f'{block_output_dir}/*.gemma/output/gemma_output.log.txt'))[0]
            shutil.copyfile(gemma_log_file, f'{saved_log_file}.tmp')
            os.replace(f'{saved_log_file}.tmp', saved_log_file)
        shutil.rmtree(block_output_dir)
    return gwas_df_h5


class GEMMA_JOB():
    """GEMMA inputs and results for one environmental variable.

    Parameters
    ----------
    name, work_dir : str
        Variable name and parent directory; results use <name>.gemma.
    input_vcf_file : str
        Genotypes for the genome-wide or CDS panel.
    phenotype_file, kinship_file : str
        BIMBAM phenotype and IBS matrix in VCF sample order.
    lmm : int
        GEMMA mode, default 4 (all tests); parsed P_VALUE is p_wald.

    Notes
    -----
    Construction records paths without launching tools. prepare() builds
    variant statistics and BIMBAM input if absent; run() executes GEMMA only
    when the result HDF is absent, then annotates coordinates and global FDR.
    """
    def __init__(self, name, work_dir, input_vcf_file, phenotype_file,
                 kinship_file, lmm=4):
        if lmm not in (1, 4):
            raise ValueError('lmm must be 1 (Wald) or 4 (all tests)')
        self.name = name
        self.work_dir = work_dir
        self.input_vcf_file = input_vcf_file
        self.kinship_file = kinship_file
        self.phenotype_file = phenotype_file
        self.lmm = lmm

        self.gwas_df_h5 = f'{self.work_dir}/{self.name}.gemma/{self.name}.gemma.gwas_df.h5'

    def prepare(self):
        """Prepare chromosome statistics and BIMBAM genotypes, reusing files."""
        self.var_stat_h5 = f'{self.input_vcf_file}.var_stat.h5'
        if not os.path.exists(self.var_stat_h5):
            print(f'Building variant stat table...')
            build_var_stat_table(self.input_vcf_file, self.var_stat_h5)
        self.bimbam_file = f'{self.input_vcf_file}'.replace(
            '.vcf.gz', '.bimbam')
        if not os.path.exists(self.bimbam_file):
            print(f'Converting VCF to BIMBAM...')
            convert_vcf_to_bimbam(self.input_vcf_file, self.bimbam_file)
        self.genome_chr_list = get_chr_id_list_from_vcf(self.input_vcf_file)

    def run(self, retry=3, dry_run=False, block_size=5_000_000,
            max_workers=80, block_cache_dir=None):
        """Run or reuse association results with the requested process limit.

        Parameters
        ----------
        retry : int
            Maximum block attempts, default 3.
        dry_run : bool
            Prepare commands without execution.
        block_size : int
            Physical GEMMA task width in bp, default 5 Mb.
        max_workers : int
            Concurrent single-threaded processes. Set within node resources.
        block_cache_dir : str or None
            Optional shared BIMBAM block cache directory.

        Returns
        -------
        None
            Set gwas_df_h5 to the reusable result table path.
        """
        if os.path.exists(self.gwas_df_h5):
            _check_gemma_lmm_output(f'{self.work_dir}/{self.name}.gemma', self.lmm)
            print(
                f'GEMMA GWAS result already exists.')
        else:
            self.prepare()
            self.gwas_df_h5 = gemma_run(self.name, f'{self.work_dir}/{self.name}.gemma', self.bimbam_file,
                                        self.phenotype_file, self.kinship_file, self.input_vcf_file, self.genome_chr_list,
                                        retry=retry, dry_run=dry_run, block_size=block_size,
                                        max_workers=max_workers, block_cache_dir=block_cache_dir,
                                        lmm=self.lmm)


def get_genefile(input_gff3_file, genefile, chr_id_map_file=None):
    """Export MAGMA gene locations with the established chromosome mapping.

    Parameters
    ----------
    input_gff3_file : str
        Gene GFF with inclusive coordinates.
    genefile : str
        MAGMA gene-location output path.
    chr_id_map_file : str or None
        Optional two-column project-to-MAGMA chromosome map. Gene endpoints
        are already 1-based inclusive and match the PLINK variant positions.

    Returns
    -------
    str
        Gene-location path; overwrite the supplied destination.
    """
    chr_id_map_dict = {}
    if chr_id_map_file:
        chr_id_map_dict = {}
        with open(chr_id_map_file) as f:
            for line in f:
                old_name, new_name = line.strip().split()
                chr_id_map_dict[old_name] = new_name

    gene_dict = read_gff_file(input_gff3_file)['gene']
    if not chr_id_map_file:
        chr_id_map_dict = dict.fromkeys(gene.chr_id for gene in gene_dict.values())
    gene_id_list = []
    for chr_id in chr_id_map_dict:
        chr_gen_list = [
            gene_id for gene_id in gene_dict if gene_dict[gene_id].chr_id == chr_id]
        chr_gen_list = sorted(chr_gen_list, key=lambda x: gene_dict[x].start)
        gene_id_list.extend(chr_gen_list)

    with open(genefile, 'w') as f:
        for gene_id in gene_id_list:
            gene = gene_dict[gene_id]
            if chr_id_map_file:
                if gene.chr_id not in chr_id_map_dict:
                    continue
                f.write(
                    f"{gene_id} {chr_id_map_dict[gene.chr_id]} {gene.start} {gene.end} {gene.strand}\n")
            else:
                f.write(
                    f"{gene_id} {gene.chr_id} {gene.start} {gene.end} {gene.strand}\n")
    return genefile


class MAGMA_JOB():
    """Aggregate site associations into genes with the existing MAGMA model.

    Parameters
    ----------
    name, work_dir : str
        Variable name and parent directory; results use <name>.magma.
    input_vcf_file, gff_file : str
        LD-reference genotypes and gene annotations.
    site_gwas_df_h5 : str
        GEMMA table with ID and P_VALUE, HDF key df.
    phenotype_file : str or None
        Single-column BIMBAM phenotype in VCF sample order. Required by run();
        nonmissing values determine MAGMA's GWAS sample size N.
    threads, memory_mb : int
        PLINK preparation threads and workspace limit in MB.
    max_workers : int
        Maximum concurrent MAGMA batch processes, default 80. MAGMA may
        reduce the batch count to preserve correlations between nearby genes.

    Notes
    -----
    Construction records inputs. prepare() renames chromosomes and converts
    the VCF to PLINK binary format; annotate() uses 2 kb gene flanks; run()
    executes SNP-wise gene analysis and saves gene P_VALUE and global FDR.
    GWAS sample size excludes missing environmental phenotypes; the complete
    genotype panel remains the LD reference. Available GEMMA logs verify N.
    """
    def __init__(self, name, work_dir, input_vcf_file, gff_file, site_gwas_df_h5=None, phenotype_file=None,
                 threads=8, memory_mb=16000, max_workers=80):
        if max_workers < 1:
            raise ValueError('MAGMA max_workers must be at least 1.')
        self.max_workers = max_workers
        self.threads = threads
        self.memory_mb = memory_mb
        self.name = name
        self.work_dir = work_dir
        self.input_vcf_file = input_vcf_file
        self.gff_file = gff_file
        self.phenotype_file = phenotype_file
        self.site_gwas_df_h5 = site_gwas_df_h5

        self.gwas_df_h5 = f'{self.work_dir}/{self.name}.magma/{self.name}.magma.gwas_df.h5'

    def prepare(self):
        """Prepare chromosome mapping, renamed VCF and PLINK BED/BIM/FAM."""
        chr_id_map_file = self.input_vcf_file.replace(".vcf.gz", ".chr_id_map")
        renamed_vcf_file = self.input_vcf_file.replace(
            ".vcf.gz", ".chr_rename.vcf.gz")
        bfile_prefix = renamed_vcf_file.replace(".vcf.gz", "")

        if not (all(os.path.exists(f'{bfile_prefix}.{suffix}') for suffix in ['bed', 'bim', 'fam']) and os.path.exists(chr_id_map_file) and os.path.exists(renamed_vcf_file)):
            # chrom rename
            chr_id_list = get_chr_id_list_from_vcf(self.input_vcf_file)

            with open(chr_id_map_file, 'w') as f:
                for i, chr_id in enumerate(chr_id_list):
                    f.write(f"{chr_id}\t{i+1}\n")

            cmd_run(
                f"bcftools annotate --rename-chrs {chr_id_map_file} {self.input_vcf_file} -O z -o {renamed_vcf_file}")
            cmd_run(f"tabix -f -p vcf {renamed_vcf_file}")

            # plink
            cmd_run(
                f"plink --vcf {renamed_vcf_file} --make-bed --threads {self.threads} --memory {self.memory_mb} --out {bfile_prefix}")

        self.chr_id_map_file = chr_id_map_file
        self.renamed_vcf_file = renamed_vcf_file
        self.bfile_prefix = bfile_prefix

    def make_p_value_file(self):
        """Export variant IDs and Wald P values from the GEMMA result HDF."""
        site_gwas_df = pd.read_hdf(self.site_gwas_df_h5, key='df')
        p_value_file = self.site_gwas_df_h5.replace(".h5", ".p_value.txt")
        site_gwas_df[['ID', 'P_VALUE']].to_csv(
            p_value_file, sep='\t', index=False, header=False)
        self.p_value_file = p_value_file

    def annotate(self):
        """Assign variants to genes with 2 kb upstream and downstream flanks."""
        bfile_bim_file = self.bfile_prefix + ".bim"
        # Keep older annotations untouched; they used shifted gene endpoints.
        annotation_prefix = self.bfile_prefix + '.1based'
        self.gene_annot_file = annotation_prefix + ".genes.annot"

        if not os.path.exists(self.gene_annot_file):
            gene_loc_file = annotation_prefix + ".ref_gene_loc"
            get_genefile(self.gff_file, gene_loc_file, self.chr_id_map_file)
            cmd_run(
                f"magma --annotate nonhuman window=2,2 --snp-loc {bfile_bim_file} --gene-loc {gene_loc_file} --out {annotation_prefix}")

    def gene_analysis(self, model='snp_pvalue'):
        """Run SNP-wise MAGMA batches and merge complete gene associations.

        Each subprocess uses the same inputs and model settings. MAGMA's
        ordinary batch merge verifies all effective batches, including when
        MAGMA reduces the requested count to retain gene correlations.
        Start batch 1, read the effective batch count from its startup logs,
        and launch remaining batches without waiting for batch 1 to finish.
        Batch outputs and per-process logs remain in a separate run directory.
        An existing merged genes.out is reused.
        """
        if model == 'snp_pvalue':
            self.make_p_value_file()
            self.magma_output_prefix = f"{self.output_dir}/magma.snp_pvalue.results"
            if not os.path.exists(self.magma_output_prefix + ".genes.out"):
                # Separate runs prevent mixing incomplete batches or batch counts.
                batch_dir = tempfile.mkdtemp(prefix='magma_batches_', dir=self.output_dir)
                batch_prefix = f'{batch_dir}/magma.snp_pvalue.results'
                command = [
                    'magma', '--bfile', self.bfile_prefix, '--gene-annot', self.gene_annot_file,
                    '--pval', self.p_value_file, f'N={self.sample_num}', '--out', batch_prefix,
                    '--gene-model', 'multi=snp-wise', '--gene-settings', 'adap-permp',
                ]
                process_env = os.environ.copy()
                process_env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

                def run_batch(batch_index):
                    log_file = f'{batch_dir}/batch_{batch_index}.stdout.log'
                    with open(log_file, 'w') as log:
                        subprocess.run(
                            command + ['--batch', str(batch_index), str(self.max_workers)],
                            stdout=log, stderr=subprocess.STDOUT, env=process_env, check=True,
                        )

                # MAGMA reports its effective count before processing genes.
                # Keep batch 1 running while launching only the valid batches.
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    first_future = executor.submit(run_batch, 1)
                    batch_count = None
                    while batch_count is None:
                        log_files = [f'{batch_dir}/batch_1.stdout.log'] + glob(f'{batch_prefix}.batch1_*.log')
                        for log_file in log_files:
                            if not os.path.exists(log_file):
                                continue
                            with open(log_file) as log:
                                for line in log:
                                    if 'analysing batch 1 of ' in line:
                                        batch_count = int(line.split('analysing batch 1 of ', 1)[1].split(',', 1)[0])
                                        break
                            if batch_count is not None:
                                break
                        if batch_count is None and first_future.done():
                            try:
                                first_future.result()
                            except subprocess.CalledProcessError as error:
                                raise RuntimeError(f'MAGMA batch 1 failed; see {batch_dir}/batch_1.stdout.log') from error
                            first_batch_files = glob(f'{batch_prefix}.batch1_*.genes.out')
                            if len(first_batch_files) != 1:
                                raise RuntimeError(f'MAGMA batch count is missing; see {batch_dir}/batch_1.stdout.log')
                            batch_count = int(first_batch_files[0].removesuffix('.genes.out').rsplit('_', 1)[1])
                        if batch_count is None:
                            sleep(0.2)

                    print(f'{self.name}: MAGMA {batch_count} effective batches; process limit {self.max_workers}; logs: {batch_dir}', flush=True)
                    futures = {first_future: 1}
                    futures.update({executor.submit(run_batch, i): i for i in range(2, batch_count + 1)})
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f'MAGMA {self.name}', unit='batch'):
                        try:
                            future.result()
                        except subprocess.CalledProcessError as error:
                            raise RuntimeError(f'MAGMA batch {futures[future]} failed; see {batch_dir}/batch_{futures[future]}.stdout.log') from error

                # Auto-detection accounts for MAGMA's reduced effective batch count.
                merged_prefix = f'{batch_dir}/merged'
                with open(f'{batch_dir}/merge.stdout.log', 'w') as log:
                    try:
                        subprocess.run(
                            ['magma', '--merge', batch_prefix, '--out', merged_prefix],
                            stdout=log, stderr=subprocess.STDOUT, env=process_env, check=True,
                        )
                    except subprocess.CalledProcessError as error:
                        raise RuntimeError(f'MAGMA merge failed; see {batch_dir}/merge.stdout.log') from error
                if not all(os.path.exists(f'{merged_prefix}.{suffix}') for suffix in ['genes.out', 'genes.raw']):
                    raise RuntimeError(f'MAGMA merge is incomplete; see {batch_dir}/merge.stdout.log')
                for suffix in ['genes.raw', 'genes.out']:
                    os.replace(f'{merged_prefix}.{suffix}', f'{self.magma_output_prefix}.{suffix}')

    def run(self, model='snp_pvalue'):
        """Run gene analysis or reuse its saved HDF table.

        Parameters
        ----------
        model : str
            Supported analysis is snp_pvalue, using MAGMA's SNP-wise model.

        Returns
        -------
        None
            Save gene associations to gwas_df_h5 with key df. Inputs and
            annotations are prepared only on a result cache miss.
            Verify phenotype row count and available GEMMA sample-count logs.
            Existing outputs with a different N or unverified coordinate
            convention raise without overwriting; magma_run.json records both.
        """
        self.output_dir = f"{self.work_dir}/{self.name}.magma"
        os.makedirs(self.output_dir, exist_ok=True)

        if self.phenotype_file is None:
            raise ValueError('MAGMA requires the phenotype file used by GEMMA to determine N.')
        reference_sample_num = len(get_sample_list_from_vcf(self.input_vcf_file))
        phenotype_df = pd.read_csv(self.phenotype_file, header=None, skip_blank_lines=False)
        if phenotype_df.shape != (reference_sample_num, 1):
            raise ValueError('MAGMA phenotype must have one column and one row per VCF sample.')
        phenotype = phenotype_df[0].to_numpy(dtype=float)
        # GEMMA accepts NA and -9 as missing phenotypes in this single-trait input.
        self.sample_num = int(np.count_nonzero(np.isfinite(phenotype) & (phenotype != -9)))
        if self.sample_num == 0:
            raise ValueError('No observed environmental phenotypes are available for MAGMA.')

        gemma_dir = os.path.dirname(self.site_gwas_df_h5)
        gemma_log_file = f'{gemma_dir}/output/gemma_output.log.txt'
        if not os.path.exists(gemma_log_file):
            gemma_log_file = next(iter(glob(f'{gemma_dir}/blocks/*.gemma/output/gemma_output.log.txt')), None)
        if gemma_log_file is not None:
            with open(gemma_log_file) as source:
                gemma_sample_num = next(
                    int(line.rsplit('=', 1)[1]) for line in source if 'number of analyzed individuals' in line
                )
            if gemma_sample_num != self.sample_num:
                raise ValueError(f'GEMMA analyzed {gemma_sample_num} samples, but the phenotype file gives N={self.sample_num}.')

        metadata_file = f'{self.output_dir}/magma_run.json'
        if os.path.exists(self.gwas_df_h5) or os.path.exists(f'{self.output_dir}/magma.snp_pvalue.results.genes.out'):
            # Earlier outputs used the complete VCF sample count for N.
            previous_sample_num = reference_sample_num
            previous_coordinates = None
            if os.path.exists(metadata_file):
                with open(metadata_file) as source:
                    previous_metadata = json.load(source)
                previous_sample_num = previous_metadata['gwas_sample_size']
                previous_coordinates = previous_metadata.get('gene_coordinate_system')
            if previous_sample_num != self.sample_num:
                raise ValueError(
                    f'Existing MAGMA results use N={previous_sample_num}; current GWAS uses N={self.sample_num}. '
                    f'Preserve {self.output_dir} and use a new MAGMA output directory before recomputing.'
                )
            if previous_coordinates != '1-based-inclusive':
                raise ValueError(
                    'Existing MAGMA results do not verify 1-based inclusive gene coordinates. '
                    f'Preserve {self.output_dir} and use a new MAGMA output directory before recomputing.'
                )
        print(f'{self.name}: MAGMA GWAS sample size N={self.sample_num}; LD-reference samples={reference_sample_num}', flush=True)

        if os.path.exists(self.gwas_df_h5):
            print(
                f'MAGMA GWAS result already exists.')
        else:
            self.prepare()
            self.annotate()
            self.gene_analysis(model=model)

            with open(metadata_file, 'w') as target:
                json.dump({'gwas_sample_size': self.sample_num,
                           'gene_coordinate_system': '1-based-inclusive'}, target)


            gwas_df_h5 = f'{self.work_dir}/{self.name}.magma/{self.name}.magma.gwas_df.h5'
            if not os.path.exists(gwas_df_h5):
                gwas_df = magma_results_parser(
                    f'{self.magma_output_prefix}.genes.out', self.gff_file, model=model)
                gwas_df.to_hdf(
                    gwas_df_h5, key='df', mode='w')

            self.gwas_df_h5 = gwas_df_h5

    def __str__(self):
        return f"MAGMA JOB: {self.name}"


def magma_results_parser(magma_output_genes_out, input_gff3_file, model='snp_pvalue'):
    """Parse MAGMA multi-model permutation P values and annotate gene ranges.

    Parameters
    ----------
    magma_output_genes_out : str
        MAGMA genes.out containing GENE, NSNPS and PERMP_MULTI.
    input_gff3_file : str
        Gene GFF for project chromosome coordinates.
    model : str
        Supported value is snp_pvalue.

    Returns
    -------
    pandas.DataFrame
        Gene IDs, SNP counts, P_VALUE, Benjamini–Hochberg FDR and coordinates.
    """
    gene_range_df = get_gene_range_df(input_gff3_file)
    gene_range_df = gene_range_df.rename(columns={'GENE': 'ID'})

    if model == 'snp_pvalue':
        magma_output_genes_df = pd.read_csv(
            magma_output_genes_out, skiprows=1, sep=r'\s+')
        magma_output_genes_df = magma_output_genes_df.rename(
            columns={'GENE': 'ID', 'PERMP_MULTI': 'P_VALUE'})
        gwas_df = magma_output_genes_df[['ID', 'NSNPS', 'P_VALUE']]
        gwas_df['FDR'] = multipletests(
            gwas_df['P_VALUE'], method='fdr_bh')[1]
        gwas_df = pd.merge(gwas_df, gene_range_df, on='ID', how='left')

    return gwas_df


def get_chr_len_df(fasta_file):
    chrom_len_csv = f'{fasta_file}.ChrLen.csv'
    if os.path.exists(chrom_len_csv):
        chrom_len_df = pd.read_csv(chrom_len_csv)
    else:
        fa_dict = read_fasta_by_faidx(fasta_file)
        chrom_len = {}
        for i in fa_dict:
            chrom_len[i] = fa_dict[i].len()
        chrom_len_df = pd.DataFrame(
            chrom_len.items(), columns=['CHROM', 'LEN'])
        chrom_len_df.to_csv(chrom_len_csv, index=False)
    return chrom_len_df


def genomewide_general_plotter(data_df, chr_length_df, ax=None, style='scatter', c1="#617EB8", c2="#84B8D0", min_chr_length=1e6, max_y=None, ylabel=""):
    """
    support style: scatter, line, shaded

    data_df: DataFrame with columns: CHROM, POS, VALUE
    chr_length_df: DataFrame with columns: chr, length. Chrom rank in chr_length_df will be used to determine the plot order
    style: scatter, line, shaded
    """

    chr_length_df = chr_length_df.loc[chr_length_df['LEN'] > min_chr_length]

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        fig = ax.get_figure()

    chr_coord_dict = {}
    total_length = 0
    chr_list = chr_length_df['CHROM'].tolist()
    for i, r in chr_length_df.iterrows():
        chr_coord_dict[r['CHROM']] = total_length
        total_length += r['LEN']


    data_df = data_df.dropna()
    data_df['CHROM'] = data_df['CHROM'].astype(str)
    data_df['POS'] = data_df['POS'].astype(int)
    data_df['VALUE'] = data_df['VALUE'].astype(float)

    ax.set_xlim(0 - total_length*0.05, total_length*1.05)
    max_y = np.max(data_df['VALUE']) if max_y is None else max_y
    ax.set_ylim(0, max_y*1.1)


    ax.set_xlabel('Chromosome')
    ax.set_ylabel(ylabel)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    c1 = "#617EB8"
    c2 = "#84B8D0"
    n = 0
    chr_ticks = []
    chr_labels = []
    for chr_id in chr_list:
        c = c1 if n % 2 == 0 else c2
        chr_df = data_df.loc[data_df['CHROM'] == chr_id]
        if style == 'scatter':
            ax.scatter((chr_df['POS'] + chr_df['CHROM'].map(chr_coord_dict)
                        ).values, chr_df['VALUE'].values, s=5, c=c)
        elif style == 'line':
            ax.plot((chr_df['POS'] + chr_df['CHROM'].map(chr_coord_dict)
                     ).values, chr_df['VALUE'].values, c=c)
        elif style == 'shaded':
            ax.fill_between((chr_df['POS'] + chr_df['CHROM'].map(chr_coord_dict)
                             ).values, chr_df['VALUE'].values, color=c, alpha=0.5)

        chr_ticks.append(
            (chr_df['POS'] + chr_df['CHROM'].map(chr_coord_dict)).mean())
        chr_labels.append(chr_id)
        n += 1


    ax.set_xticks(chr_ticks)
    ax.set_xticklabels(chr_labels)

    if ax is None:
        plt.show()


def genomewide_gwas_plot(gwas_df, chr_length_df, threshold_fdr=0.05, ax=None, pval_lim=1e-20, plot_fdr=True):
    """
    绘制 Manhattan 图
    gwas_df should have columns: CHROM, POS, P_VALUE, FDR
    chr_list: list of chromosome names
    chr_length_dict: dict of chromosome length
    """
    # data preparation
    gwas_df.loc[gwas_df['P_VALUE'] <= pval_lim, 'P_VALUE'] = pval_lim

    data_df = gwas_df.copy()
    data_df['VALUE'] = -np.log10(data_df['P_VALUE'])

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        fig = ax.get_figure()

    genomewide_general_plotter(data_df, chr_length_df, ax=ax,
                               style='scatter', ylabel='$-log_{10}(P)$')



    if plot_fdr:

        gwas_df = gwas_df.copy()
        gwas_df.loc[gwas_df['FDR'] == 0, 'FDR'] = pval_lim

        sig_mask = gwas_df['FDR'] < threshold_fdr

        if sig_mask.any():

            p_at_thr = gwas_df.loc[sig_mask, 'P_VALUE'].max()
            threshold = -np.log10(max(p_at_thr, pval_lim))
        else:

            y0, y1 = ax.get_ylim()
            threshold = y1 * 0.98

        ax.axhline(
            y=threshold,
            color='r',
            linestyle='--',
            linewidth=1.0,
            # label=f'FDR ≤ {threshold_fdr:.2f}'
        )

        # ax.text(start + x_add, -np.log10(eqtl_pval_threshold) + y_add, f'FDR = 0.05', fontsize=12, color='black', va='bottom', ha='left', bbox=dict(facecolor='white', alpha=1, linewidth=0), zorder=2)
        ax.text(0 - ax.get_xlim()[1]*0.04, threshold * 1.02, f'FDR = {threshold_fdr:.2f}', fontsize=10,
                color='red', va='bottom', ha='left', bbox=dict(facecolor='white', alpha=1, linewidth=0), zorder=2)

    if ax is None:
        plt.show()


def _qq_included(p_value, filter_zero_point=True):
    """Select finite probabilities for QQ plotting; retain P=1."""
    p_value = np.asarray(p_value, dtype=float)
    mask = np.isfinite(p_value) & (p_value >= 0) & (p_value <= 1)
    if filter_zero_point:
        mask &= p_value > 0
    return mask


def quantile_quantile_plot(p_value, title=None, ax=None, filter_zero_point=True):
    """Plot observed and expected P values on logarithmic axes.

    By default exclude zero, nonfinite and out-of-range P values. P=1 is
    included. With filter_zero_point=False, display zero at the smallest
    positive floating-point value without changing the source probabilities.
    """
    p_value = np.asarray(p_value, dtype=float)
    p_value = p_value[_qq_included(p_value, filter_zero_point)]
    show_plot = ax is None
    if ax is None:
        fig, ax = plt.subplots()

    n = len(p_value)
    x_max = y_max = 1.0
    if n:
        expected = -np.log10(np.linspace(1/n, 1, n))
        observed = -np.log10(np.maximum(np.sort(p_value), np.nextafter(0.0, 1.0)))
        ax.scatter(expected, observed, s=10, c="#000000")
        x_max = max(1.0, float(np.max(expected)))
        y_max = max(1.0, float(np.max(observed)))

    max_value = max(x_max, y_max)
    ax.plot([-max_value*0.1, max_value*1.1],
            [-max_value*0.1, max_value*1.1], c="#FB0324")
    ax.set_xlim(-x_max*0.1, x_max*1.1)
    ax.set_ylim(-y_max*0.1, y_max*1.1)
    ax.set_xlabel('Expected $-log_{10}(P)$')
    ax.set_ylabel('Observed $-log_{10}(P)$')
    if title is not None:
        ax.set_title(title)
    if show_plot:
        plt.show()
    return ax


def _shrink_axis_font(ax,
                      label_size=9,
                      tick_size=8,
                      title_size=10,
                      xlabel_pad=4,
                      ylabel_pad=4):
    """
    Adjust axis font size and label spacing.

    Parameters
    ----------
    ax : matplotlib axis
    label_size : int
        Font size for axis labels
    tick_size : int
        Font size for tick labels
    title_size : int
        Font size for title
    xlabel_pad : float
        Distance between x label and axis
    ylabel_pad : float
        Distance between y label and axis
    """

    # label font size
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)

    # label distance
    ax.xaxis.labelpad = xlabel_pad
    ax.yaxis.labelpad = ylabel_pad

    # tick size
    ax.tick_params(axis='both', labelsize=tick_size)

    # title
    if ax.get_title():
        ax.title.set_size(title_size)


def plot_gwas_summary_panel(gwas_df_h5, genome_fasta_file,
                            threshold_fdr=0.05,
                            bins=50,
                            title=None,
                            figsize=(14, 6),
                            save=True):
    """
    Plot GWAS summary panel:
        Left: Manhattan plot (large)
        Right-top: QQ plot
        Right-bottom: P-value histogram

    Parameters
    ----------
    gwas_df_h5 : str
        Path to GWAS result HDF5 file.
    genome_fasta_file : str
        Genome FASTA file for chromosome length extraction.
    threshold_fdr : float
        FDR threshold for Manhattan.
    bins : int
        Histogram bin number.
    title : str or None
        Title for QQ plot and histogram.
    figsize : tuple
        Figure size.
    save : bool
        Whether to save PNG and PDF.

    Returns
    -------
    tuple
        Figure and panel Axes. Read the saved GWAS HDF without running
        associations; export the actual figure values as CSV.gz alongside PNG/PDF.
        P_VALUE retains the original probabilities; QQ_INCLUDED marks finite
        values in (0, 1] used in the QQ panel. MANHATTAN_P_VALUE records
        the display floor used in the Manhattan panel. Call plt.show().
    """

    set_plot_style()

    # -----------------------------
    # Load data
    # -----------------------------
    gwas_df = pd.read_hdf(gwas_df_h5, key='df')
    chr_len_df = get_chr_len_df(genome_fasta_file)

    manhattan_df = gwas_df[['CHROM', 'POS', 'P_VALUE', 'FDR']].copy()
    p_values = gwas_df['P_VALUE'].values
    plot_data_df = gwas_df[['ID', 'CHROM', 'POS', 'P_VALUE', 'FDR']].copy()
    plot_data_df['MANHATTAN_P_VALUE'] = plot_data_df['P_VALUE'].clip(lower=1e-20)
    plot_data_df['QQ_INCLUDED'] = _qq_included(p_values)

    if title is None:
        title = gwas_df_h5.split('/')[-1].replace('.h5', '')

    # -----------------------------
    # Create layout
    # -----------------------------
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[4, 1],
        height_ratios=[1, 1],
        wspace=0.1,
        hspace=0.25
    )

    ax_manhattan = fig.add_subplot(gs[:, 0])
    ax_qq = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 1])

    # -----------------------------
    # Manhattan plot
    # -----------------------------
    genomewide_gwas_plot(
        manhattan_df,
        chr_len_df,
        threshold_fdr=threshold_fdr,
        ax=ax_manhattan
    )

    ax_manhattan.set_title(None)

    ax_manhattan.set_xlabel('')

    # -----------------------------
    # QQ plot
    # -----------------------------
    ax_qq = quantile_quantile_plot(
        p_values,
        # title="QQ Plot",
        ax=ax_qq,
        filter_zero_point=True
    )
    _shrink_axis_font(ax_qq, label_size=10, tick_size=10,
                      title_size=9, xlabel_pad=2)

    # -----------------------------
    # Histogram
    # -----------------------------
    ax_hist.hist(
        p_values,
        bins=bins,
        edgecolor='gray',
        linewidth=0.1,
        alpha=0.7,
        density=True
    )
    ax_hist.axhline(y=1, linestyle='--')
    ax_hist.set_xlabel(r"$P$ value")
    ax_hist.set_ylabel("Density")
    # ax_hist.set_title("P-value Distribution")
    ax_hist.grid(True, linestyle='--', alpha=0.6)

    _shrink_axis_font(ax_hist, label_size=10, tick_size=10,
                      title_size=9, xlabel_pad=2)

    # -----------------------------
    # Overall formatting
    # -----------------------------
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    # -----------------------------
    # Save
    # -----------------------------
    if save:
        # output_file = gwas_df_h5.replace('.h5', '_gwas_summary.pdf')
        # fig.savefig(output_file, format='pdf', facecolor='none', edgecolor='none', bbox_inches='tight')

        output_file = gwas_df_h5.replace('.h5', '_gwas_summary.png')
        fig.savefig(
            output_file,
            dpi=600,
            facecolor='white',
            bbox_inches='tight'
        )

    if save:
        fig.savefig(gwas_df_h5.replace('.h5', '_gwas_summary.pdf'), bbox_inches='tight')
        plot_data_df.to_csv(gwas_df_h5.replace('.h5', '_gwas_summary.csv.gz'), index=False, compression='gzip')
    return fig, (ax_manhattan, ax_qq, ax_hist)

    # return fig

if __name__ == "__main__":

    from statsmodels.sandbox.stats.multicomp import multipletests
    import pandas as pd
    import os
    env_id_list = ["bio12", "bio3", "annualPET", "bio8", "topoWet", "NTO", "TPH", "CLYPPT"]
    for env_id in env_id_list:
        gwas_df_csv = f"/lustre/home/xuyuxing/Work/Jesse/local_adaptation.3.0/2.envGWAS/{env_id}/{env_id}.gemma/output/gemma_output.assoc.txt"
        file_with_fdr = gwas_df_csv.replace(".assoc.txt", ".assoc.fdr.txt.gz")
        if os.path.exists(file_with_fdr):
            print(f"scp cn04_acc2:{file_with_fdr} {env_id}.gemma.assoc.fdr.txt.gz")
            continue
        gwas_df = pd.read_csv(gwas_df_csv, sep="\t")
        gwas_df['fdr'] = multipletests(gwas_df['p_wald'], method='fdr_bh')[1]
        print(gwas_df[gwas_df['fdr'] < 0.05].shape[0])
        gwas_df.to_csv(gwas_df_csv.replace(".assoc.txt", ".assoc.fdr.txt.gz"), sep="\t", index=False, compression="gzip")
        
