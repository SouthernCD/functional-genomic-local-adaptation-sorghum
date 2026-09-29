from cyvcf2 import VCF, Writer
import numpy as np
import pandas as pd
import os
import gc
import time
import random
from .utils import cmd_run, log_print, have_file, multiprocess_running
from .feature import read_gff_file
from interlap import InterLap


def extract_cds_vcf(input_vcf_file, ref_gff_file, output_vcf_file, min_maf=0.01, max_het=0.1, max_miss=0.5):
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


def quality_control_vcf(input_vcf_file, output_vcf_file, min_maf=0.05, max_het=0.1, max_miss=0.5):
    if output_vcf_file.endswith('.gz'):
        output_vcf_file = output_vcf_file[:-3]

    vcf_reader = VCF(input_vcf_file)
    vcf_writer = Writer(output_vcf_file, vcf_reader)

    for record in vcf_reader:
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
        x, bins=[-1.5, -0.5, 0.5, 1.5, 2.5])[0], 1, genotype_matrix)

    # Calculate the number of each type of element
    mis_num = counts[:, 0]
    hom_ref_num = counts[:, 1]
    het_num = counts[:, 2]
    hom_alt_num = counts[:, 3]

    maf = (het_num + 2*hom_alt_num)/(2*(hom_ref_num + het_num + hom_alt_num))
    maf = np.minimum(maf, 1-maf)

    het = het_num/(hom_ref_num + het_num + hom_alt_num)
    mis = mis_num/(hom_ref_num + het_num + hom_alt_num + hom_alt_num)

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


def build_var_stat_table(input_vcf_file, output_h5_file):
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

            variant_genotypes = [
                gt[0] + gt[1] if gt[0] is not None and gt[1] is not None else -1 for gt in record.genotypes]
            genotype_matrix.append(variant_genotypes)

        chr_var_df = pd.DataFrame(records)

        genotype_matrix = np.array(genotype_matrix)

        mis, maf, het, hom_ref_num, het_num, hom_alt_num = get_var_stat_num_parallel(
            genotype_matrix, chunk_size=1000, n_jobs=8)

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
    """
    Get genotype matrix from a VCF file.
    """

    vcf = VCF(vcf_file)
    genotype_matrix = []

    if chr_id is not None and start is not None and end is not None:
        for record in vcf(f'{chr_id}:{start}-{end}'):
            variant_genotypes = [
                gt[0] + gt[1] if gt[0] is not None and gt[1] is not None else -1 for gt in record.genotypes]
            genotype_matrix.append(variant_genotypes)
    elif chr_id is not None and start is None and end is None:
        for record in vcf(f'{chr_id}'):
            variant_genotypes = [
                gt[0] + gt[1] if gt[0] is not None and gt[1] is not None else -1 for gt in record.genotypes]
            genotype_matrix.append(variant_genotypes)
    else:
        for record in vcf:
            variant_genotypes = [
                gt[0] + gt[1] if gt[0] is not None and gt[1] is not None else -1 for gt in record.genotypes]
            genotype_matrix.append(variant_genotypes)

    genotype_matrix = np.array(genotype_matrix)

    return genotype_matrix


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