from joblib import Parallel, delayed
import numpy as np
import time

"""
This module provides functions to calculate IBS matrix and IBS distance matrix.
"""


def process_chunk_for_IBS(k, chunk_size, genotype_matrix):
    """
    处理单个数据块的函数。
    """
    k_end = min(k + chunk_size, genotype_matrix.shape[0])
    genotype_pairs = np.abs(
        genotype_matrix[k:k_end, :, None] - genotype_matrix[k:k_end, None, :]
    )
    block_result = 1 - genotype_pairs / 2
    return block_result.sum(axis=0)  # 返回当前块处理后的结果之和，减少内存使用


def get_IBS_matrix_broadcasting_chunk_parallel(genotype_matrix, chunk_size=100, n_jobs=8):
    """
    优化内存使用的并行处理计算IBS矩阵。
    """
    # 首先对genotype_matrix进行求和处理以简化数据
    # genotype_matrix = np.sum(genotype_matrix, axis=2)
    n_samples = genotype_matrix.shape[1]

    # 初始化结果矩阵
    results = np.zeros((n_samples, n_samples))

    total_chunks = genotype_matrix.shape[0] // chunk_size + \
        (1 if genotype_matrix.shape[0] % chunk_size != 0 else 0)
    batch_size = n_jobs * 100  # 每次处理20个chunk
    processed_chunks = 0

    for start_chunk in range(0, total_chunks, batch_size):
        end_chunk = min(start_chunk + batch_size, total_chunks)
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(process_chunk_for_IBS)(
                k * chunk_size, chunk_size, genotype_matrix)
            for k in range(start_chunk, end_chunk)
        )

        # 将这批次的结果累加到最终结果中
        for block_result in all_results:
            results += block_result

        processed_chunks += len(all_results)
        print("Time: %s, processed %d/%d chunks, %.2f%%" % (time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()), processed_chunks, total_chunks, processed_chunks/total_chunks*100))

    # 计算最终结果
    results /= genotype_matrix.shape[0]

    return results


def get_IBS_distance_matrix(IBS_matrix):
    """
    计算IBS距离矩阵。
    """
    ibs_distance_matrix = 1 - IBS_matrix
    return ibs_distance_matrix


if __name__ == '__main__':
    from yxquantgene.utils.vcf import get_genotype_matrix_from_vcf
    from yxquantgene.utils.utils import write_matrix_to_file

    genotype_matrix = get_genotype_matrix_from_vcf(vcf_file)
    ibs_matrix = get_IBS_matrix_broadcasting_chunk_parallel(genotype_matrix)
    ibs_distance_matrix = get_IBS_distance_matrix(ibs_matrix)
    write_matrix_to_file(ibs_matrix, ibs_file)
    write_matrix_to_file(ibs_distance_matrix, ibs_distance_file)
    ibs_distance_file = "/lustre/home/xuyuxing/Work/Jesse/local_adaptation/0.reference/Data/reseq/landraces/Sbv5.1.landraces.snp.win10000.maf0.10.miss0.50.rq0.50.ld.nr.rep.ibs_dist.matrix"
