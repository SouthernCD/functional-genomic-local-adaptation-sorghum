from joblib import Parallel, delayed
import numpy as np
import time

"""
This module provides functions to calculate IBS matrix and IBS distance matrix.
"""


def process_chunk_for_IBS(k, chunk_size, genotype_matrix, skip_missing=False):
    """
    处理单个数据块的函数。
    """
    k_end = min(k + chunk_size, genotype_matrix.shape[0])
    genotype_pairs = np.abs(
        genotype_matrix[k:k_end, :, None] - genotype_matrix[k:k_end, None, :]
    )
    block_result = 1 - genotype_pairs / 2
    if skip_missing:
        called = genotype_matrix[k:k_end] >= 0
        valid_pairs = called[:, :, None] & called[:, None, :]
        return np.where(valid_pairs, block_result, 0).sum(axis=0), valid_pairs.sum(axis=0)
    return block_result.sum(axis=0)  # 返回当前块处理后的结果之和，减少内存使用


def get_IBS_matrix_broadcasting_chunk_parallel(genotype_matrix, chunk_size=100, n_jobs=8,
                                              *, skip_missing=False):
    """
    优化内存使用的并行处理计算IBS矩阵。
    skip_missing=True 时，每对样品仅使用双方都非缺失的位点；无共同
    非缺失位点时返回 NaN。默认保留旧版将缺失值直接参与计算的行为。
    """
    # 首先对genotype_matrix进行求和处理以简化数据
    # genotype_matrix = np.sum(genotype_matrix, axis=2)
    n_samples = genotype_matrix.shape[1]

    # 初始化结果矩阵
    results = np.zeros((n_samples, n_samples))
    counts = np.zeros((n_samples, n_samples), dtype=np.int64) if skip_missing else None

    total_chunks = genotype_matrix.shape[0] // chunk_size + \
        (1 if genotype_matrix.shape[0] % chunk_size != 0 else 0)
    batch_size = n_jobs * 100  # 每次处理20个chunk
    processed_chunks = 0

    for start_chunk in range(0, total_chunks, batch_size):
        end_chunk = min(start_chunk + batch_size, total_chunks)
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(process_chunk_for_IBS)(
                k * chunk_size, chunk_size, genotype_matrix, skip_missing)
            for k in range(start_chunk, end_chunk)
        )

        # 将这批次的结果累加到最终结果中
        for block_result in all_results:
            if skip_missing:
                results += block_result[0]
                counts += block_result[1]
            else:
                results += block_result

        processed_chunks += len(all_results)
        print("Time: %s, processed %d/%d chunks, %.2f%%" % (time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()), processed_chunks, total_chunks, processed_chunks/total_chunks*100))

    # 计算最终结果
    if skip_missing:
        results = np.divide(results, counts, out=np.full_like(results, np.nan), where=counts > 0)
    else:
        results /= genotype_matrix.shape[0]

    return results


def get_IBS_distance_matrix(IBS_matrix):
    """
    计算IBS距离矩阵。
    """
    ibs_distance_matrix = 1 - IBS_matrix
    return ibs_distance_matrix


def build_ibs_nj_tree(input_vcf_file, genome_fasta_file, output_prefix=None, *, only_snps=True,
                      min_maf=0.01, max_het=0.01, max_miss=0.5,
                      ld_window_size=15000, r2_threshold=0.5, random_seed=42,
                      n_jobs=40, chunk_size=100, prefilter_window_size=10_000_000,
                      fastme='fastme', show_progress=True):
    """Prune a VCF, write IBS/distance matrices and infer a pure NJ tree.

    Mixed SNP/indel input is accepted. Keep biallelic SNPs by default;
    only_snps=False also includes biallelic indels. Reuse ld.prune_vcf_by_ld
    for QC and seeded random greedy pruning, and this module's IBS functions
    with pairwise deletion of missing calls. Input must be coordinate-sorted
    with diploid genotypes. All samples and their VCF order are preserved.
    Pairs without jointly called sites raise an error before tree inference.
    Reuse vcf.quality_control_vcf for genotype-based QC before LD;
    prune_vcf_by_ld checks the same thresholds again on its fresh statistics.

    output_prefix is a basename without extensions/parameter suffixes; when
    omitted, use the input basename plus '_snp_pruned' or '_pruned'. Append
    '.win{window}.miss{miss}.maf{maf}.het{het}.rq{r2}' to output names.
    Return a dict of absolute paths: pruned_vcf, ibs_matrix,
    ibs_distance_matrix, ibs_distance_phylip and nj_tree. The VCF has a TBI
    index. Temporary filtering/statistics files are not reused across calls.
    Existing outputs at the same prefix are replaced after a successful run.
    bcftools and FastME 2.x must be available; fastme may be an executable path.
    FastME is called with '-m N -w n' (NJ lengths, no NNI postprocessing).
    n_jobs is the single worker setting (default 40) for QC, variant
    statistics, LD pruning, IBS and FastME. Parallel LD requires fork.
    genome_fasta_file is required and must match the VCF reference/contig names.
    Reuse feature.read_fasta_by_faidx to obtain chromosome lengths without
    loading sequences. QC uses n_jobs concurrent bcftools processes;
    prefilter_window_size defaults to 10 Mb (independent of the LD window).
    Filter nonoverlapping windows by record POS and concatenate in VCF header
    order. Indexed BGZF VCF/BCF is required for parallel filtering; a missing
    index is created. Use n_jobs=1 for a plain uncompressed VCF.

    从可含 SNP/indel 的 VCF 完成质控、LD pruning、IBS 和距离矩阵及 NJ
    建树。默认仅用二等位 SNP；only_snps=False 时也纳入二等位 indel。
    复用项目的随机贪婪 pruning，random_seed 控制可重复性；IBS 按每对
    样品共同非缺失位点计算。样品名及顺序取自 VCF。明确使用纯 NJ，
    不沿用旧笔记中的 NNI 优化。返回各结果的绝对路径字典，VCF 带索引。
    先按基因型做相同阈值的预过滤，以减少统计和 LD 的内存开销。
    genome_fasta_file 必填，复用 read_fasta_by_faidx 获取染色体长度。
    n_jobs 默认 40，统一控制 QC、位点统计、LD、IBS 和 FastME 并行度。
    QC 默认按 10 Mb 窗口预过滤；LD 窗口独立设置。
    按 POS 分配窗口并按头部染色体顺序拼接；并行需要输入索引，
    缺失时自动创建。未压缩 VCF 请设置 n_jobs=1。
    """
    import os
    import shutil
    import subprocess
    import tempfile
    from operator import index
    from .ld import prune_vcf_by_ld
    from .vcf import (VCF, get_genotype_matrix_from_vcf, get_sample_list_from_vcf,
                      quality_control_vcf)

    input_vcf_file = os.path.abspath(os.fspath(input_vcf_file))
    if not os.path.isfile(input_vcf_file):
        raise FileNotFoundError(input_vcf_file)
    n_jobs, chunk_size = map(index, (n_jobs, chunk_size))
    if min(n_jobs, chunk_size) < 1:
        raise ValueError('n_jobs and chunk_size must be positive integers.')
    # Normalize threshold values used in output names; QC validates their range.
    min_maf, max_het, max_miss = [None if value is None else float(value)
                                for value in (min_maf, max_het, max_miss)]
    for executable in ('bcftools', os.fspath(fastme)):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f'Executable not found: {executable}')
    if output_prefix is None:
        basename = input_vcf_file
        for suffix in ('.vcf.gz', '.vcf', '.bcf'):
            if basename.endswith(suffix):
                basename = basename[:-len(suffix)]
                break
        output_prefix = basename + ('_snp_pruned' if only_snps else '_pruned')
    output_prefix = os.path.abspath(os.fspath(output_prefix))
    suffix = (f'.win{ld_window_size}.miss{max_miss}.maf{min_maf}'
              f'.het{max_het}.rq{r2_threshold}')
    prefix = output_prefix + suffix
    outputs = {
        'pruned_vcf': prefix + '.vcf.gz',
        'ibs_matrix': prefix + '.ibs.matrix',
        'ibs_distance_matrix': prefix + '.ibs_dist.matrix',
        'ibs_distance_phylip': prefix + '.ibs_dist.phylip',
        'nj_tree': prefix + '.ibs_dist.nj.tree',
    }
    if os.path.realpath(input_vcf_file) in map(os.path.realpath, outputs.values()):
        raise ValueError('Input VCF and output paths must differ.')
    output_dir = os.path.dirname(output_prefix)
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='ibs_nj_', dir=output_dir) as temp_dir:
        filtered_vcf = os.path.join(temp_dir, 'biallelic.vcf.gz')
        quality_control_vcf(
            input_vcf_file, filtered_vcf, genome_fasta_file=genome_fasta_file,
            min_maf=min_maf, max_het=max_het, max_miss=max_miss,
            only_snps=only_snps, biallelic=True,
            variant_types=None if only_snps else ('snps', 'indels'),
            n_jobs=n_jobs, window_size=prefilter_window_size,
            show_progress=show_progress,
        )
        reader = VCF(filtered_vcf)
        try:
            if reader.num_records == 0 or len(reader.samples) < 2:
                raise ValueError('At least one eligible variant and two samples are required.')
        finally:
            reader.close()
        temporary = {key: os.path.join(temp_dir, os.path.basename(path))
                     for key, path in outputs.items()}
        prune_vcf_by_ld(
            filtered_vcf, temporary['pruned_vcf'],
            ld_window_size=ld_window_size, r2_threshold=r2_threshold,
            random_seed=random_seed, min_maf=min_maf, max_het=max_het,
            max_miss=max_miss, pool_num=n_jobs, show_progress=show_progress,
        )
        sample_names = get_sample_list_from_vcf(temporary['pruned_vcf'])
        if any(any(char.isspace() for char in name) for name in sample_names):
            raise ValueError('PHYLIP sample names must not contain whitespace.')
        genotypes = get_genotype_matrix_from_vcf(temporary['pruned_vcf'])
        if genotypes.size == 0:
            raise ValueError('No variants remain after QC and LD pruning.')
        if np.any(genotypes > 2):
            raise ValueError('IBS requires diploid biallelic genotype dosages.')
        ibs_matrix = get_IBS_matrix_broadcasting_chunk_parallel(
            genotypes, chunk_size=chunk_size, n_jobs=n_jobs, skip_missing=True,
        )
        del genotypes
        if not np.isfinite(ibs_matrix).all():
            raise ValueError('Some sample pairs have no jointly called sites; cannot infer NJ.')
        distance_matrix = get_IBS_distance_matrix(ibs_matrix)
        np.savetxt(temporary['ibs_matrix'], ibs_matrix, fmt='%.6f', delimiter=', ')
        np.savetxt(temporary['ibs_distance_matrix'], distance_matrix,
                   fmt='%.6f', delimiter=', ')
        with open(temporary['ibs_distance_phylip'], 'w', encoding='utf-8') as handle:
            handle.write(f'{len(sample_names)}\n')
            for name, row in zip(sample_names, distance_matrix):
                handle.write(name + ' ' + ' '.join(f'{value:.6f}' for value in row) + '\n')
        subprocess.run([
            os.fspath(fastme), '-i', temporary['ibs_distance_phylip'],
            '-m', 'N', '-w', 'n', '-T', str(n_jobs), '-o', temporary['nj_tree'],
        ], check=True)
        from Bio import Phylo
        tree = Phylo.read(temporary['nj_tree'], 'newick')
        if sorted(tip.name for tip in tree.get_terminals()) != sorted(sample_names):
            raise ValueError('NJ tree tip names differ from the VCF sample names.')
        for key, path in outputs.items():
            os.replace(temporary[key], path)
        os.replace(temporary['pruned_vcf'] + '.tbi', outputs['pruned_vcf'] + '.tbi')
        if os.path.exists(outputs['pruned_vcf'] + '.csi'):
            os.remove(outputs['pruned_vcf'] + '.csi')
    return outputs
