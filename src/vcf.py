from cyvcf2 import VCF, Writer
import numpy as np
import pandas as pd
import os
import gc
import time
import random
import subprocess
import tempfile
from .utils import cmd_run, log_print, have_file, multiprocess_running
from .feature import read_gff_file
from interlap import InterLap


def _window_maf_chunk(args):
    """Read one indexed region; each worker owns its VCF handle."""
    vcf_file, chrom, start, end, window_size, min_maf = args
    first_bin = (start - 1) // window_size
    n_bins = (end - 1) // window_size - first_bin + 1
    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=np.int64)
    snp_counts = np.zeros(n_bins, dtype=np.int64)
    rare_snp_counts = np.zeros(n_bins, dtype=np.int64)
    skipped_multiallelic = skipped_missing = 0
    vcf = VCF(vcf_file)
    try:
        for variant in vcf(f'{chrom}:{start}-{end}'):
            # Indexed queries may also return variants overlapping the boundary.
            # Assign by POS so spanning variants cannot be counted twice.
            if not start <= variant.POS <= end:
                continue
            bin_index = (variant.POS - 1) // window_size - first_bin
            is_snp = variant.is_snp
            if is_snp:
                snp_counts[bin_index] += 1
            if len(variant.ALT) != 1:
                skipped_multiallelic += 1
                continue
            # The final column is the phase flag; negative entries are missing
            # allele copies or padding for mixed ploidy.
            alleles = variant.genotype.array()[:, :-1]
            called_count = np.count_nonzero(alleles >= 0)
            if called_count == 0:
                skipped_missing += 1
                continue
            if np.any(alleles > 1):
                raise ValueError(f'Invalid biallelic genotype at {chrom}:{variant.POS}')
            alt_frequency = np.count_nonzero(alleles == 1) / called_count
            site_maf = min(alt_frequency, 1 - alt_frequency)
            if site_maf < min_maf:
                if is_snp:
                    rare_snp_counts[bin_index] += 1
                continue
            sums[bin_index] += site_maf
            counts[bin_index] += 1
    finally:
        vcf.close()
    return (chrom, first_bin, sums, counts, snp_counts, rare_snp_counts,
            skipped_multiallelic, skipped_missing)


def calculate_window_maf(vcf_file, chromosome_lengths, window_size=250_000,
                         n_jobs=8, chunk_size=5_000_000, min_maf=0.05):
    """Compute common-site mean MAF and total/rare SNP counts in parallel.

    All samples and biallelic variant types are included. Missing copies and
    mixed-ploidy padding are excluded; fully missing/non-biallelic sites are
    skipped for MAF. Only sites with MAF >= min_maf contribute to mean MAF
    (default 0.05). Windows without qualifying sites retain MAF=NaN.
    SNP_COUNT counts all SNP records, including multiallelic and missing sites.
    RARE_SNP_COUNT counts biallelic SNPs with called alleles and MAF < min_maf.
    Non-SNP variants contribute to mean MAF but neither SNP count.
    chromosome_lengths maps chromosome IDs to their plotted lengths in bp.
    Chunks align to window boundaries and return only sums/counts, never full
    genotype matrices. n_jobs=1 runs serially. A VCF index is required.
    Returns CHROM, BIN (zero-based), MAF, MAF_SITE_COUNT (qualifying sites),
    SNP_COUNT and RARE_SNP_COUNT; exclusion counts
    are stored in DataFrame.attrs. tqdm reports completed genomic chunks.

    按基因组区间并行读取有索引的 VCF，计算各窗口二等位位点的平均 MAF。
    MAF 使用全部样品，忽略缺失拷贝，跳过全缺失及非二等位位点；仅纳入
    MAF >= min_maf（默认 0.05）的位点，无符合条件的位点时均值为 NaN。
    SNP_COUNT 为全部 SNP 记录数；RARE_SNP_COUNT 为可计算 MAF 且
    MAF < min_maf 的二等位 SNP 数。非 SNP 位点只参与 MAF，不计入 SNP 数。
    n_jobs 控制进程数，chunk_size 控制区间大小（自动对齐绘图窗口）。
    n_jobs=1 时串行执行；tqdm 显示已完成的区间数。
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    from operator import index
    from tqdm.auto import tqdm

    window_size, n_jobs, chunk_size = map(index, (window_size, n_jobs, chunk_size))
    if min(window_size, n_jobs, chunk_size) < 1:
        raise ValueError('window_size, n_jobs and chunk_size must be positive integers.')
    min_maf = float(min_maf)
    if not np.isfinite(min_maf) or not 0 <= min_maf <= 0.5:
        raise ValueError('min_maf must be between 0 and 0.5.')
    lengths = {str(chrom): index(length) for chrom, length in chromosome_lengths.items()}
    if not lengths or min(lengths.values()) < 1:
        raise ValueError('chromosome_lengths must contain positive lengths.')
    vcf_file = os.fspath(vcf_file)
    vcf = VCF(vcf_file)
    try:
        # Fail before dispatch if the index is absent.
        vcf.num_records
        available_chromosomes = set(vcf.seqnames)
    finally:
        vcf.close()
    if not available_chromosomes.intersection(lengths):
        raise ValueError('VCF and plotted chromosomes have no matching chromosome IDs.')

    aligned_chunk_size = max(1, chunk_size // window_size) * window_size
    tasks = [(vcf_file, chrom, start, min(start + aligned_chunk_size - 1, length), window_size, min_maf)
             for chrom, length in lengths.items() if chrom in available_chromosomes
             for start in range(1, length + 1, aligned_chunk_size)]
    sums = {chrom: np.zeros((length + window_size - 1) // window_size, dtype=float)
            for chrom, length in lengths.items()}
    counts = {chrom: np.zeros(len(values), dtype=np.int64) for chrom, values in sums.items()}
    snp_counts = {chrom: np.zeros(len(values), dtype=np.int64) for chrom, values in sums.items()}
    rare_snp_counts = {chrom: np.zeros(len(values), dtype=np.int64) for chrom, values in sums.items()}
    skipped_multiallelic = skipped_missing = 0

    def collect(result):
        nonlocal skipped_multiallelic, skipped_missing
        (chrom, first_bin, chunk_sums, chunk_counts, chunk_snps, chunk_rare_snps,
         multiallelic, missing) = result
        destination = slice(first_bin, first_bin + len(chunk_sums))
        sums[chrom][destination] = chunk_sums
        counts[chrom][destination] = chunk_counts
        snp_counts[chrom][destination] = chunk_snps
        rare_snp_counts[chrom][destination] = chunk_rare_snps
        skipped_multiallelic += multiallelic
        skipped_missing += missing

    if n_jobs == 1:
        for task in tqdm(tasks, desc='Calculating window MAF', unit='chunk'):
            collect(_window_maf_chunk(task))
    else:
        # Spawn avoids inheriting active Jupyter threads or open htslib handles.
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks)),
                                 mp_context=get_context('spawn')) as executor:
            futures = [executor.submit(_window_maf_chunk, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc='Calculating window MAF', unit='chunk'):
                collect(future.result())

    frames = []
    for chrom in lengths:
        means = np.divide(sums[chrom], counts[chrom],
                          out=np.full(len(sums[chrom]), np.nan), where=counts[chrom] > 0)
        frames.append(pd.DataFrame({'CHROM': chrom, 'BIN': np.arange(len(means)),
                                    'MAF': means, 'MAF_SITE_COUNT': counts[chrom],
                                    'SNP_COUNT': snp_counts[chrom],
                                    'RARE_SNP_COUNT': rare_snp_counts[chrom]}))
    result = pd.concat(frames, ignore_index=True)
    result.attrs['skipped_multiallelic'] = skipped_multiallelic
    result.attrs['skipped_missing'] = skipped_missing
    return result


def extract_samples_vcf(input_vcf_file, sample_list, output_vcf_file):
    """
    Extract an explicitly ordered sample panel with bcftools.

    Parameters
    ----------
    input_vcf_file : str
        Indexed VCF; sample IDs must be present.
    sample_list : list of str
        Nonempty requested sample IDs in output order.
    output_vcf_file : str
        Output path; append .gz when absent. Must differ from the input.

    Returns
    -------
    str
        BGZF VCF path, with a TBI index. Uses bcftools view -c 1 to retain
        records with at least one ALT allele. Writes the selected VCF and
        index; existing output paths can be overwritten, so the manual caller
        owns reuse and baseline preservation. No scheduler submission.
    """
    input_vcf_file = os.fspath(input_vcf_file)
    output_vcf_file = os.fspath(output_vcf_file)
    if not output_vcf_file.endswith('.gz'):
        output_vcf_file += '.gz'
    if os.path.realpath(input_vcf_file) == os.path.realpath(output_vcf_file):
        raise ValueError('Input and output VCF paths must be different.')
    if isinstance(sample_list, str) or not sample_list:
        raise ValueError('sample_list must be a non-empty list of sample names.')
    if any(not isinstance(sample, str) or not sample or
           any(char.isspace() for char in sample) for sample in sample_list):
        raise ValueError('Sample names must be non-empty strings without whitespace.')

    # A sample file avoids command-line length limits for large sample lists.
    with tempfile.TemporaryDirectory(prefix='vcf_samples_') as temp_dir:
        sample_file = os.path.join(temp_dir, 'samples.txt')
        with open(sample_file, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(sample_list) + '\n')
        subprocess.run([
            'bcftools', 'view', '-S', sample_file, '-c', '1',
            '-Oz', '-o', output_vcf_file, input_vcf_file,
        ], check=True)

    subprocess.run(['bcftools', 'index', '-t', '-f', output_vcf_file], check=True)
    return output_vcf_file


def merge_snp_indel_vcf(snp_vcf_file, indel_vcf_file, output_vcf_file):
    """
    Combine coordinate-sorted SNP and indel inputs, one record per position.

    Workflow
    --------
    1. Check identical sample identities and ordering.
    2. Concatenate and sort using bcftools.
    3. Retain the first SNP per position, or first record if no SNP exists.
    4. Index and publish the completed VCF, cleaning temporary files.

    Parameters
    ----------
    snp_vcf_file, indel_vcf_file : str
        Coordinate-sorted input VCFs with identical sample names and order.
    output_vcf_file : str
        Distinct BGZF destination; append .gz when absent.

    Returns
    -------
    str
        Output VCF path and associated TBI index. No variant normalization,
        quality filtering, or genotype/allele merging is performed. REF, ALT,
        genotypes and ANN are retained from the chosen record. Existing output
        is replaced only after successful preparation; caller owns reuse.
        External bcftools must be available. No scheduler submission.
    """
    snp_vcf_file = os.fspath(snp_vcf_file)
    indel_vcf_file = os.fspath(indel_vcf_file)
    output_vcf_file = os.fspath(output_vcf_file)
    if not output_vcf_file.endswith('.gz'):
        output_vcf_file += '.gz'
    for input_vcf_file in (snp_vcf_file, indel_vcf_file):
        if os.path.realpath(input_vcf_file) == os.path.realpath(output_vcf_file):
            raise ValueError('Input and output VCF paths must be different.')
    if get_sample_list_from_vcf(snp_vcf_file) != get_sample_list_from_vcf(indel_vcf_file):
        raise ValueError('SNP and indel VCFs must have identical sample names/order.')

    with tempfile.TemporaryDirectory(
            prefix='vcf_merge_', dir=os.path.dirname(os.path.abspath(output_vcf_file))) as temp_dir:
        concat_vcf_file = os.path.join(temp_dir, 'concat.vcf.gz')
        sorted_vcf_file = os.path.join(temp_dir, 'sorted.vcf.gz')
        merged_vcf_file = os.path.join(temp_dir, 'merged.vcf.gz')
        subprocess.run([
            'bcftools', 'concat', '-a', '-d', 'snps', '-Oz', '-o', concat_vcf_file,
            snp_vcf_file, indel_vcf_file,
        ], check=True)
        subprocess.run([
            'bcftools', 'sort', '-T', os.path.join(temp_dir, 'sort'),
            '-Oz', '-o', sorted_vcf_file, concat_vcf_file,
        ], check=True)

        vcf_reader = VCF(sorted_vcf_file)
        try:
            vcf_writer = Writer(merged_vcf_file, vcf_reader, mode='wz')
            try:
                kept_record = None
                for record in vcf_reader:
                    if kept_record is None:
                        kept_record = record
                    elif (record.CHROM, record.POS) != (kept_record.CHROM, kept_record.POS):
                        vcf_writer.write_record(kept_record)
                        kept_record = record
                    elif record.is_snp and not kept_record.is_snp:
                        kept_record = record
                if kept_record is not None:
                    vcf_writer.write_record(kept_record)
            finally:
                vcf_writer.close()
        finally:
            vcf_reader.close()

        subprocess.run(['bcftools', 'index', '-t', merged_vcf_file], check=True)
        os.replace(merged_vcf_file, output_vcf_file)
        os.replace(merged_vcf_file + '.tbi', output_vcf_file + '.tbi')
        if os.path.exists(output_vcf_file + '.csi'):
            os.remove(output_vcf_file + '.csi')
    return output_vcf_file


def extract_cds_vcf(input_vcf_file, ref_gff_file, output_vcf_file, min_maf=0.01, max_het=0.1, max_miss=0.5):
    """
    Extract longest-transcript CDS records using genotype-derived QC.

    Parameters
    ----------
    input_vcf_file : str
        Diploid VCF; the existing dosage formula assumes a biallelic record.
    ref_gff_file : str
        Gene GFF with transcript qualifier longest=1. CDS coordinates are
        1-based inclusive after parsing; overlap is determined by record POS.
    output_vcf_file : str
        Destination VCF path, compressed using bgzip and indexed with tabix.
    min_maf, max_het, max_miss : float
        Inclusive lower MAF and upper heterozygosity/missingness bounds.
        MAF and heterozygosity use called diploid genotypes; missingness uses
        all samples. Defaults are 0.01, 0.1, and 0.5 respectively.

    Returns
    -------
    str
        Compressed VCF path with TBI index. Reads GFF/VCF and writes output;
        this function has no cache policy. The manual caller preserves existing
        outputs. Multiallelic and zero-called-genotype behavior remain legacy
        assumptions for scientific review; this migration does not repair them.
    """
    gff_dict = read_gff_file(ref_gff_file)['gene']

    cds_interlap = {}
    for gene_id in gff_dict:
        gene = gff_dict[gene_id]
        cds_interlap.setdefault(gene.chr_id, InterLap())
        for mRNA in gene.sub_features:
            if int(mRNA.qualifiers['longest'][0]) != 1:
                continue
            for cds in mRNA.sub_features:
                if cds.type == 'CDS':
                    cds_interlap[gene.chr_id].add((cds.start, cds.end))

    if output_vcf_file.endswith('.gz'):
        output_vcf_file = output_vcf_file[:-3]

    vcf_reader = VCF(input_vcf_file)
    vcf_writer = Writer(output_vcf_file, vcf_reader)

    for record in vcf_reader:
        chr_id = record.CHROM
        pos = record.POS
        if chr_id not in cds_interlap:
            continue
        if len(list(cds_interlap[chr_id].find((pos, pos)))) < 1:
            continue
        het_num = record.num_het
        hom_ref_num = record.num_hom_ref
        hom_alt_num = record.num_hom_alt
        maf = (het_num + 2*hom_alt_num) / \
            (2*(hom_ref_num + het_num + hom_alt_num))
        maf = min(maf, 1-maf)
        het_freq = het_num / (het_num + hom_ref_num + hom_alt_num)
        het_exp_freq = 2 * maf * (1 - maf)

        if maf < min_maf or het_freq > max_het or record.call_rate < 1 - max_miss:
            continue

        vcf_writer.write_record(record)

    vcf_writer.close()
    vcf_reader.close()

    cmd_run(f'bgzip {output_vcf_file}')
    cmd_run(f'tabix -f -p vcf {output_vcf_file}.gz')

    return output_vcf_file + '.gz'


def quality_control_vcf(input_vcf_file, output_vcf_file, min_maf=0.05,
                        max_het=0.1, max_miss=0.5, *, genome_fasta_file,
                        only_snps=False, biallelic=False, variant_types=None,
                        n_jobs=40, window_size=10_000_000, show_progress=True):
    """Filter a coordinate-sorted, diploid VCF using genotype-based QC.

    genome_fasta_file is required; chromosome names/reference must match the
    VCF. Reuse read_fasta_by_faidx for lengths without loading the sequences.
    min_maf, max_het and max_miss accept None to disable that filter. MAF and
    heterozygosity exclude missing genotypes; missingness uses all samples.
    Sites without called genotypes are excluded when MAF/heterozygosity is
    requested. Thresholds are inclusive. Do not trust existing INFO/AC/AN.

    only_snps=True keeps SNP records; biallelic=True requires exactly one ALT.
    Otherwise keep all variant types/allele counts unless variant_types is
    supplied (comma-separated string or iterable of bcftools variant types).
    only_snps=True cannot be combined with non-SNP variant_types. Multiallelic
    MAF is the second most frequent allele, recomputed with +fill-tags; this
    branch adds/replaces INFO/MAF and ignores partially missing genotypes.

    n_jobs controls concurrent windows (default 40), window_size their length
    (default 10 Mb). Workers run bcftools; multiallelic MAF uses a two-process
    pipe per window. Assign by POS to avoid duplicating spanning indels, then
    concatenate in VCF header order. Parallel input must be indexed BGZF VCF
    or BCF; create a missing index. n_jobs=1 also supports uncompressed VCF.
    Return the absolute BGZF VCF path with a TBI index. Append '.gz' when the
    output path lacks it. Replace existing outputs only after QC succeeds;
    empty results are allowed, and temporary files are cleaned up.

    通用 VCF 质控：FASTA 必填，各阈值设为 None 可关闭对应过滤；可独立
    控制 SNP、二等位限制及变异类型。默认按 10 Mb 窗口、40 个任务并行，
    保留样品和记录顺序，返回压缩 VCF 路径并创建索引。
    """
    import shutil
    from operator import index
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .feature import read_fasta_by_faidx

    input_vcf_file = os.path.abspath(os.fspath(input_vcf_file))
    if not os.path.isfile(input_vcf_file):
        raise FileNotFoundError(input_vcf_file)
    genome_fasta_file = os.path.abspath(os.fspath(genome_fasta_file))
    if not os.path.isfile(genome_fasta_file):
        raise FileNotFoundError(genome_fasta_file)
    n_jobs, window_size = map(index, (n_jobs, window_size))
    if min(n_jobs, window_size) < 1:
        raise ValueError('Worker counts and chunk/window sizes must be positive integers.')
    fasta = read_fasta_by_faidx(genome_fasta_file)
    chromosome_lengths = {chromosome: record.len() for chromosome, record in fasta.items()}
    del fasta
    reader = VCF(input_vcf_file)
    try:
        chromosomes = list(reader.seqnames)
    finally:
        reader.close()
    if not chromosomes:
        raise ValueError('No chromosomes found in the input VCF header.')
    missing = [chromosome for chromosome in chromosomes if chromosome not in chromosome_lengths]
    if missing:
        raise ValueError(f'VCF chromosomes absent from the reference FASTA: {missing}')
    if any(chromosome_lengths[chromosome] < 1 for chromosome in chromosomes):
        raise ValueError('Reference chromosomes must have positive lengths.')
    thresholds = []
    for name, value, maximum in (('min_maf', min_maf, 0.5),
                                 ('max_het', max_het, 1), ('max_miss', max_miss, 1)):
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f'{name} must be None or in [0, {maximum}].')
        thresholds.append(value)
    min_maf, max_het, max_miss = thresholds
    if shutil.which('bcftools') is None:
        raise FileNotFoundError('Executable not found: bcftools')
    if isinstance(variant_types, str):
        variant_types = variant_types.split(',')
    if variant_types is not None:
        variant_types = tuple(variant_types)
        allowed = {'snps', 'indels', 'mnps', 'ref', 'bnd', 'other', 'overlap'}
        if not variant_types or not set(variant_types) <= allowed:
            raise ValueError(f'variant_types must contain bcftools types from {sorted(allowed)}.')
    if only_snps:
        if variant_types is not None and set(variant_types) != {'snps'}:
            raise ValueError('only_snps=True conflicts with non-SNP variant_types.')
        variant_types = ('snps',)
    output_vcf_file = os.path.abspath(os.fspath(output_vcf_file))
    if not output_vcf_file.endswith('.gz'):
        output_vcf_file += '.gz'
    if os.path.realpath(input_vcf_file) == os.path.realpath(output_vcf_file):
        raise ValueError('Input VCF and output paths must differ.')
    output_dir = os.path.dirname(output_vcf_file)
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vcf_qc_', dir=output_dir) as temp_dir:
        filtered_vcf = os.path.join(temp_dir, 'filtered.vcf.gz')
        command = ['bcftools', 'view', '-Oz', '--no-version']
        if biallelic:
            command += ['-m2', '-M2']
        # Recompute from GT rather than trusting potentially stale INFO/AC/AN.
        called = 'N_PASS(GT!="mis")'
        het = 'N_PASS(GT="het" & GT!="mis")'
        alt_frequency = f'({het}+2*N_PASS(GT="AA"))/(2*{called})'
        conditions = [f'{called}>0'] if min_maf is not None or max_het is not None else []
        if min_maf is not None:
            if biallelic:
                conditions += [f'{alt_frequency}>={min_maf}',
                               f'{alt_frequency}<={1-min_maf}']
            else:
                conditions.append(f'INFO/MAF>={min_maf}')
        if max_het is not None:
            conditions.append(f'{het}/{called}<={max_het}')
        if max_miss is not None:
            conditions.append(f'(N_SAMPLES-{called})/N_SAMPLES<={max_miss}')
        if conditions:
            command += ['-i', ' && '.join(f'({condition})' for condition in conditions)]
        if variant_types:
            command += ['-v', ','.join(variant_types)]

        def filter_region(part_file, region=None):
            region_options = ['-r', region, '--regions-overlap', '0'] if region else []
            if biallelic or min_maf is None:
                subprocess.run(command + region_options + [
                    '-o', part_file, input_vcf_file], check=True)
            else:
                # Multiallelic MAF is the second most frequent allele. Recompute
                # from GT; drop partially missing calls consistently with QC.
                annotation = ['bcftools', '+fill-tags', '--no-version', '-Ou']
                annotation += region_options + [input_vcf_file, '--', '-t', 'MAF', '-d']
                with subprocess.Popen(annotation, stdout=subprocess.PIPE) as producer:
                    try:
                        subprocess.run(command + ['-o', part_file, '-'],
                                       stdin=producer.stdout, check=True)
                    finally:
                        producer.stdout.close()
                    if producer.wait() != 0:
                        raise subprocess.CalledProcessError(producer.returncode, annotation)
        if n_jobs == 1:
            filter_region(filtered_vcf)
        else:
            if not any(os.path.isfile(input_vcf_file + suffix) for suffix in ('.tbi', '.csi')):
                subprocess.run(['bcftools', 'index', input_vcf_file], check=True)
            regions = [f'{chromosome}:{start}-{min(start + window_size - 1, chromosome_lengths[chromosome])}'
                       for chromosome in chromosomes
                       for start in range(1, chromosome_lengths[chromosome] + 1,
                                          window_size)]
            part_files = [os.path.join(temp_dir, f'part_{i:06d}.vcf.gz')
                          for i in range(len(regions))]

            if show_progress:
                print(f'VCF QC: filtering {len(regions)} windows with '
                      f'{min(n_jobs, len(regions))} workers', flush=True)
            with ThreadPoolExecutor(max_workers=min(n_jobs, len(regions))) as executor:
                futures = {executor.submit(filter_region, path, region): region
                           for region, path in zip(regions, part_files)}
                for completed, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if show_progress:
                        print(f'VCF QC: {completed}/{len(regions)} windows '
                              f'complete ({futures[future]})', flush=True)
            file_list = os.path.join(temp_dir, 'parts.txt')
            with open(file_list, 'w', encoding='utf-8') as handle:
                handle.write('\n'.join(part_files) + '\n')
            # Every part comes from the same header. Naive concatenation avoids
            # decompressing/recompressing all records after parallel filtering.
            subprocess.run(['bcftools', 'concat', '--naive', '-f', file_list,
                            '-o', filtered_vcf], check=True)
        subprocess.run(['bcftools', 'index', '-t', filtered_vcf], check=True)
        os.replace(filtered_vcf, output_vcf_file)
        os.replace(filtered_vcf + '.tbi', output_vcf_file + '.tbi')
        if os.path.exists(output_vcf_file + '.csi'):
            os.remove(output_vcf_file + '.csi')
    return output_vcf_file


def get_chr_id_list_from_vcf(vcf_file):
    """
    Get chromosome ID list from a VCF file.
    """

    vcf = VCF(vcf_file)
    chr_id_list = vcf.seqnames
    vcf.close()

    return chr_id_list


def get_sample_list_from_vcf(vcf_file):
    """
    Get sample list from a VCF file.
    """

    vcf = VCF(vcf_file)
    sample_list = vcf.samples
    vcf.close()

    return sample_list


def get_var_stat(genotype_matrix):
    # Count the number of each element in the matrix
    counts = np.apply_along_axis(lambda x: np.histogram(
        x, bins=[-np.inf, -0.5, 0.5, 1.5, 2.5])[0], 1, genotype_matrix)

    # Calculate the number of each type of element
    mis_num = counts[:, 0]
    hom_ref_num = counts[:, 1]
    het_num = counts[:, 2]
    hom_alt_num = counts[:, 3]

    maf = (het_num + 2*hom_alt_num)/(2*(hom_ref_num + het_num + hom_alt_num))
    maf = np.minimum(maf, 1-maf)

    het = het_num/(hom_ref_num + het_num + hom_alt_num)
    mis = mis_num/(mis_num + hom_ref_num + het_num + hom_alt_num)

    return mis, maf, het, hom_ref_num, het_num, hom_alt_num


def get_var_stat_num_parallel(genotype_matrix, chunk_size=1000, n_jobs=8):
    # Split the genotype_matrix into chunks
    mis, maf, het, hom_ref_num, het_num, hom_alt_num = [], [], [], [], [], []

    total_chunks = genotype_matrix.shape[0] // chunk_size + \
        (1 if genotype_matrix.shape[0] % chunk_size != 0 else 0)
    batch_size = n_jobs * 100
    processed_chunks = 0

    for start_chunk in range(0, total_chunks, batch_size):
        end_chunk = min(start_chunk + batch_size, total_chunks)

        args_dict = {}
        for k in range(start_chunk, end_chunk):
            k = k * chunk_size
            k_end = min(k + chunk_size, genotype_matrix.shape[0])
            genotype_matrix_chunk = genotype_matrix[k:k_end]
            args_dict[k] = (genotype_matrix_chunk,)

        mlt_dict = multiprocess_running(
            get_var_stat, args_dict, n_jobs, silence=True)

        for k in range(start_chunk, end_chunk):
            k = k * chunk_size
            mis_chunk, maf_chunk, het_chunk, hom_ref_num_chunk, het_num_chunk, hom_alt_num_chunk = mlt_dict[
                k]['output']
            mis.extend(mis_chunk)
            maf.extend(maf_chunk)
            het.extend(het_chunk)
            hom_ref_num.extend(hom_ref_num_chunk)
            het_num.extend(het_num_chunk)
            hom_alt_num.extend(hom_alt_num_chunk)

        log_print("processed %d/%d chunks, %.2f%%" % (processed_chunks,
                                                      total_chunks, processed_chunks/total_chunks*100))

        processed_chunks += batch_size

    return np.array(mis), np.array(maf), np.array(het), np.array(hom_ref_num), np.array(het_num), np.array(hom_alt_num)


def build_var_stat_table(input_vcf_file, output_h5_file, *, n_jobs=8):
    """
    Write the established chromosome-keyed variant statistics format.

    Workflow
    --------
    1. Create a missing TBI index with tabix.
    2. Traverse chromosomes and load one chromosome genotype matrix.
    3. Calculate genotype statistics and append its HDF table.

    Parameters
    ----------
    input_vcf_file : str
        VCF with genotype calls and variant IDs; existing diploid/biallelic
        dosage assumptions apply. REF and first ALT are recorded. Negative or
        partially missing alleles produce missing dosage.
    output_h5_file : str
        HDF path with one key per nonempty chromosome, preserving record order.
        This legacy layout is retained for existing LD/GWAS consumers.
    n_jobs : int
        Number of chunk workers; default 8. One full chromosome genotype
        matrix remains in memory regardless of worker count.

    Returns
    -------
    str
        Output path. Writes chromosome tables incrementally, not atomically;
        an interrupted file may contain only some chromosomes. The manual caller
        must inspect completeness and retain baseline results before reuse.
        Displays stage progress and never submits scheduler jobs.
    """
    if not have_file(input_vcf_file + ".tbi"):
        cmd_run(f'tabix -p vcf {input_vcf_file}')

    chr_id_list = get_chr_id_list_from_vcf(input_vcf_file)

    for chr_id in chr_id_list:
        log_print(f'Processing {chr_id}')

        genotype_matrix = []
        vcf = VCF(input_vcf_file)
        records = []
        for record in vcf(f"{chr_id}"):
            # chr_var_df = chr_var_df.append({
            #     'ID': record.ID,
            #     'CHROM': record.CHROM,
            #     'POS': record.POS,
            #     'REF': record.REF,
            #     'ALT': record.ALT[0],
            # }, ignore_index=True)
            records.append({
                'ID': record.ID,
                'CHROM': record.CHROM,
                'POS': record.POS,
                'REF': record.REF,
                'ALT': record.ALT[0],
            })

            # cyvcf2 represents missing alleles as negative integers. Treat
            # partially missing diploid calls as missing, too.
            variant_genotypes = [
                gt[0] + gt[1]
                if gt[0] is not None and gt[1] is not None
                and gt[0] >= 0 and gt[1] >= 0 else -1
                for gt in record.genotypes
            ]
            genotype_matrix.append(variant_genotypes)

        # Filtered VCF headers may list chromosomes with no remaining records.
        # Empty tables would coerce POS to float when LD concatenates them.
        if not records:
            vcf.close()
            continue
        chr_var_df = pd.DataFrame(records)

        genotype_matrix = np.array(genotype_matrix)

        mis, maf, het, hom_ref_num, het_num, hom_alt_num = get_var_stat_num_parallel(
            genotype_matrix, chunk_size=1000, n_jobs=n_jobs)

        chr_var_df['MISSF'] = mis
        chr_var_df['MAF'] = maf
        chr_var_df['HETF'] = het
        chr_var_df['HOM_REF'] = hom_ref_num
        chr_var_df['HET'] = het_num
        chr_var_df['HOM_ALT'] = hom_alt_num

        chr_var_df.to_hdf(output_h5_file, key=chr_id, mode='a')

        vcf.close()
        del genotype_matrix
        del chr_var_df
        gc.collect()

    return output_h5_file


def convert_vcf_to_bimbam(input_vcf_file, output_bimbam_file):
    """
    Convert a VCF file to a BIMBAM file.
    """

    vcf = VCF(input_vcf_file)
    bimbam_file_handle = open(output_bimbam_file, 'w')

    num = 0
    for record in vcf:
        variant_genotypes = [
            gt[0] + gt[1] if gt[0] is not None and gt[1] is not None else -1 for gt in record.genotypes]
        variant_genotypes = [str(x) for x in variant_genotypes]
        bimbam_file_handle.write(
            ",".join([record.ID, record.ALT[0], record.REF] + variant_genotypes) + '\n')
        num += 1

        if num % 10000 == 0:
            print(f'Processed {num} variants.')

    vcf.close()
    bimbam_file_handle.close()

    return output_bimbam_file


def read_hdf_with_retry(file_path, key, max_retries=5):
    """
    尝试读取 HDF5 文件，如果失败则重试。

    参数:
    file_path (str): HDF5 文件路径。
    key (str): 要读取的键。
    max_retries (int): 最大重试次数。
    wait_time (int): 每次重试前等待的时间（秒）。

    返回:
    pd.DataFrame: 读取的 DataFrame。
    """
    for attempt in range(max_retries):
        try:
            df = pd.read_hdf(file_path, key=key)
            return df
        except:
            print(f"Attempt {attempt + 1} failed")
            time.sleep(random.uniform(1, 10))
    raise Exception(f"Failed to read HDF5 file after {max_retries} attempts")


def get_genotype_matrix_from_vcf(vcf_file, chr_id=None, start=None, end=None):
    """Read diploid allele-index sums; any missing allele gives dosage -1.

    读取二倍体基因型剂量；任一等位基因缺失时记为 -1，并关闭 VCF 句柄。
    """

    vcf = VCF(vcf_file)
    genotype_matrix = []

    try:
        if chr_id is not None and start is not None and end is not None:
            records = vcf(f'{chr_id}:{start}-{end}')
        elif chr_id is not None and start is None and end is None:
            records = vcf(f'{chr_id}')
        else:
            records = vcf
        for record in records:
            genotypes = record.genotypes
            if any(len(gt) != 3 for gt in genotypes):
                raise ValueError(f'Diploid genotypes required at {record.CHROM}:{record.POS}.')
            genotype_matrix.append([
                gt[0] + gt[1] if gt[0] is not None and gt[1] is not None
                and gt[0] >= 0 and gt[1] >= 0 else -1 for gt in genotypes
            ])
        return (np.array(genotype_matrix) if genotype_matrix
                else np.empty((0, len(vcf.samples)), dtype=int))
    finally:
        vcf.close()


def get_genotype_matrix_from_vcf_by_pos_list(vcf_file, chr_id, pos_list, *, vcf_reader=None):
    """
    Get genotype matrix from a VCF file by position list.
    pos_list: a list of positions to extract genotypes from the VCF file.
    vcf_reader: optional open reader owned and closed by the caller.
    VCF record order and multiple records at the same position are preserved.
    """

    positions = {int(pos) for pos in pos_list}
    owns_reader = vcf_reader is None
    vcf = VCF(vcf_file) if owns_reader else vcf_reader
    try:
        genotype_matrix = []
        used_pos_list = []
        if positions:
            for record in vcf(f'{chr_id}:{min(positions)}-{max(positions)}'):
                if record.POS in positions:
                    variant_genotypes = [
                        gt[0] + gt[1] if gt[0] is not None and gt[1] is not None else -1 for gt in record.genotypes]
                    genotype_matrix.append(variant_genotypes)
                    used_pos_list.append(record.POS)
        if genotype_matrix:
            genotype_matrix = np.array(genotype_matrix)
        else:
            genotype_matrix = np.empty((0, len(vcf.samples)), dtype=int)
        return genotype_matrix, chr_id, used_pos_list
    finally:
        if owns_reader:
            vcf.close()


def get_genotype_df_from_vcf_by_site_df(vcf_file, site_df):
    """
    Get genotype DataFrame from a VCF file by site DataFrame.
    site_df: a DataFrame with columns 'CHROM' and 'POS' specifying the sites to extract genotypes for.
    """
    sample_id_list = get_sample_list_from_vcf(vcf_file)
    chr_id_list = site_df['CHROM'].unique().tolist()
    genotype_df = pd.DataFrame(columns=sample_id_list)
    for chr_id in chr_id_list:
        pos_list = site_df.loc[site_df['CHROM'] == chr_id, 'POS'].tolist()
        id_list = site_df.loc[site_df['CHROM'] == chr_id, 'ID'].tolist()
        genotype_matrix, chr_id, used_pos_list = get_genotype_matrix_from_vcf_by_pos_list(vcf_file, chr_id, pos_list)
        used_id_list = [id_list[pos_list.index(pos)] for pos in used_pos_list]

        genotype_df = pd.concat([
            genotype_df,
            pd.DataFrame(
                genotype_matrix,
                columns=sample_id_list,
                index=used_id_list
            )
        ])
    return genotype_df
