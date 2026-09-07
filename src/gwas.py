from .utils import mkdir, have_file, cmd_run
from .vcf import build_var_stat_table, convert_vcf_to_bimbam, get_chr_id_list_from_vcf, read_hdf_with_retry, get_sample_list_from_vcf
from .feature import read_gff_file, get_gene_range_df, read_fasta_by_faidx
from .plot import set_plot_style
from statsmodels.sandbox.stats.multicomp import multipletests
import pandas as pd
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def prepare_gemma_phenotype_file(env_name, phenotype_df, phenotype_file):
    with open(phenotype_file, 'w') as f:
        for i in phenotype_df[env_name].to_list():
            if np.isnan(i):
                f.write('NA\n')
            else:
                f.write(str(i) + '\n')
    return phenotype_file


def gemma_result_parser(gemma_output_assoc_file, var_stat_h5, chr_list):
    """
    从GEMMA的输出文件中解析出结果，并与var_stat_h5中的信息进行合并
    """

    print(f'Parsing {gemma_output_assoc_file}...')
    print(f'Loading gemma_output_assoc_file...')
    gemma_output_assoc_df = pd.read_csv(gemma_output_assoc_file, sep='\t')
    gemma_output_assoc_df = gemma_output_assoc_df[[
        'rs', 'af', 'beta', 'se', 'logl_H1', 'l_remle', 'l_mle', 'p_wald', 'p_lrt', 'p_score']]
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

    return merged_df


def gemma_run(name, work_dir, bimbam_file, phenotype_file, kinship_file, input_vcf_file=None, genome_chr_list=None, retry=3, dry_run=False, vcf_var_stat_h5=None):
    # 创建输出目录
    mkdir(work_dir)

    # 运行GEMMA
    gemma_output_assoc_file = f'{work_dir}/output/gemma_output.assoc.txt'
    if have_file(gemma_output_assoc_file):
        print(
            f'GEMMA GWAS result already exists.')
    elif dry_run:
        with open(f'{work_dir}/gemma_cmd.sh', 'w') as f:
            f.write(
                f"gemma -g {bimbam_file} -k {kinship_file} -lmm 4 -miss 0.1 -p {phenotype_file} -o gemma_output -outdir {work_dir}")
    else:
        with open(f'{work_dir}/gemma_cmd.sh', 'w') as f:
            f.write(
                f"gemma -g {bimbam_file} -k {kinship_file} -lmm 4 -miss 0.1 -p {phenotype_file} -o gemma_output -outdir {work_dir}")

        f, o, e = cmd_run(
            f"gemma -g {bimbam_file} -k {kinship_file} -lmm 4 -miss 0.1 -p {phenotype_file} -o gemma_output", cwd=work_dir)

        retry_count = 0
        if not have_file(gemma_output_assoc_file):
            print(f, o, e)
            retry_count += 1
            if retry_count < retry:
                print(
                    f'GEMMA GWAS failed, retrying {retry_count}...')
                sleep(30)
                cmd_run(
                    f"gemma -g {bimbam_file} -k {kinship_file} -lmm 4 -miss 0.1 -p {phenotype_file} -o gemma_output", cwd=work_dir)
            else:
                raise FileNotFoundError(
                    f'GEMMA GWAS failed: {gemma_output_assoc_file}')

    if input_vcf_file is None and vcf_var_stat_h5 is None:
        return gemma_output_assoc_file

    # 解析GEMMA结果为DataFrame
    gwas_df_h5 = f'{work_dir}/{name}.gemma.gwas_df.h5'
    if have_file(gwas_df_h5):
        return gwas_df_h5
    else:
        gwas_df = gemma_result_parser(
            gemma_output_assoc_file, f'{input_vcf_file}.var_stat.h5' if vcf_var_stat_h5 is None else vcf_var_stat_h5, genome_chr_list)
        gwas_df.to_hdf(
            gwas_df_h5, key='df', mode='w')
        return gwas_df_h5

    gwas_df = pd.read_hdf(gwas_df_h5, key='df')


class GEMMA_JOB():
    def __init__(self, name, work_dir, input_vcf_file, phenotype_file, kinship_file):
        self.name = name
        self.work_dir = work_dir
        self.input_vcf_file = input_vcf_file
        self.kinship_file = kinship_file
        self.phenotype_file = phenotype_file

        self.gwas_df_h5 = f'{self.work_dir}/{self.name}.gemma/{self.name}.gemma.gwas_df.h5'

    def prepare(self):
        self.var_stat_h5 = f'{self.input_vcf_file}.var_stat.h5'
        if not have_file(self.var_stat_h5):
            print(f'Building variant stat table...')
            build_var_stat_table(self.input_vcf_file, self.var_stat_h5)
        self.bimbam_file = f'{self.input_vcf_file}'.replace(
            '.vcf.gz', '.bimbam')
        if not have_file(self.bimbam_file):
            print(f'Converting VCF to BIMBAM...')
            convert_vcf_to_bimbam(self.input_vcf_file, self.bimbam_file)
        self.genome_chr_list = get_chr_id_list_from_vcf(self.input_vcf_file)

    def run(self, retry=3, dry_run=False):
        if have_file(self.gwas_df_h5):
            print(
                f'GEMMA GWAS result already exists.')
        else:
            self.prepare()
            self.gwas_df_h5 = gemma_run(self.name, f'{self.work_dir}/{self.name}.gemma', self.bimbam_file,
                                        self.phenotype_file, self.kinship_file, self.input_vcf_file, self.genome_chr_list, retry=retry, dry_run=dry_run)


def get_genefile(input_gff3_file, genefile, chr_id_map_file=None):
    if chr_id_map_file:
        chr_id_map_dict = {}
        with open(chr_id_map_file) as f:
            for line in f:
                old_name, new_name = line.strip().split()
                chr_id_map_dict[old_name] = new_name

    gene_dict = read_gff_file(input_gff3_file)['gene']
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
                    f"{gene_id} {chr_id_map_dict[gene.chr_id]} {gene.start - 1} {gene.end - 1} {gene.strand}\n")
            else:
                f.write(
                    f"{gene_id} {gene.chr_id} {gene.start - 1} {gene.end - 1} {gene.strand}\n")
    return genefile


class MAGMA_JOB():
    def __init__(self, name, work_dir, input_vcf_file, gff_file, site_gwas_df_h5=None, phenotype_file=None):
        self.name = name
        self.work_dir = work_dir
        self.input_vcf_file = input_vcf_file
        self.gff_file = gff_file
        self.phenotype_file = phenotype_file
        self.site_gwas_df_h5 = site_gwas_df_h5

        self.gwas_df_h5 = f'{self.work_dir}/{self.name}.magma/{self.name}.magma.gwas_df.h5'

    def prepare(self):
        chr_id_map_file = self.input_vcf_file.replace(".vcf.gz", ".chr_id_map")
        renamed_vcf_file = self.input_vcf_file.replace(
            ".vcf.gz", ".chr_rename.vcf.gz")
        bfile_prefix = renamed_vcf_file.replace(".vcf.gz", "")

        if not (have_file(bfile_prefix + ".bed") and have_file(chr_id_map_file) and have_file(renamed_vcf_file)):
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
                f"plink --vcf {renamed_vcf_file} --make-bed --out {bfile_prefix}")

        self.chr_id_map_file = chr_id_map_file
        self.renamed_vcf_file = renamed_vcf_file
        self.bfile_prefix = bfile_prefix
        self.sample_num = len(get_sample_list_from_vcf(self.renamed_vcf_file))

    def make_p_value_file(self):
        site_gwas_df = pd.read_hdf(self.site_gwas_df_h5)
        p_value_file = self.site_gwas_df_h5.replace(".h5", ".p_value.txt")
        site_gwas_df[['ID', 'P_VALUE']].to_csv(
            p_value_file, sep='\t', index=False, header=False)
        self.p_value_file = p_value_file

    def annotate(self):
        bfile_bim_file = self.bfile_prefix + ".bim"
        self.gene_annot_file = self.bfile_prefix + ".genes.annot"

        if not have_file(self.gene_annot_file):
            gene_loc_file = self.bfile_prefix + ".ref_gene_loc"
            get_genefile(self.gff_file, gene_loc_file, self.chr_id_map_file)
            cmd_run(
                f"magma --annotate nonhuman window=2,2 --snp-loc {bfile_bim_file} --gene-loc {gene_loc_file} --out {self.bfile_prefix}")

    def gene_analysis(self, model='snp_pvalue'):
        if model == 'snp_pvalue':
            self.make_p_value_file()
            self.magma_output_prefix = f"{self.output_dir}/magma.snp_pvalue.results"
            if not have_file(self.magma_output_prefix + ".genes.out"):
                cmd_run(f"magma --bfile {self.bfile_prefix} --gene-annot {self.gene_annot_file} --pval {self.p_value_file} N={self.sample_num} --out {self.magma_output_prefix} --gene-model multi=snp-wise --gene-settings adap-permp")

    def run(self, model='snp_pvalue'):
        self.output_dir = f"{self.work_dir}/{self.name}.magma"
        mkdir(self.output_dir)

        if have_file(self.gwas_df_h5):
            print(
                f'MAGMA GWAS result already exists.')
        else:
            self.prepare()
            self.annotate()
            self.gene_analysis(model=model)

            # 解析结果
            gwas_df_h5 = f'{self.work_dir}/{self.name}.magma/{self.name}.magma.gwas_df.h5'
            if not have_file(gwas_df_h5):
                gwas_df = magma_results_parser(
                    f'{self.magma_output_prefix}.genes.out', self.gff_file, model=model)
                gwas_df.to_hdf(
                    gwas_df_h5, key='df', mode='w')

            self.gwas_df_h5 = gwas_df_h5

    def __str__(self):
        return f"MAGMA JOB: {self.name}"


def magma_results_parser(magma_output_genes_out, input_gff3_file, model='snp_pvalue'):
    gene_range_df = get_gene_range_df(input_gff3_file)
    gene_range_df = gene_range_df.rename(columns={'GENE': 'ID'})

    if model == 'snp_pvalue':
        magma_output_genes_df = pd.read_csv(
            magma_output_genes_out, skiprows=1, delim_whitespace=True)
        magma_output_genes_df = magma_output_genes_df.rename(
            columns={'GENE': 'ID', 'PERMP_MULTI': 'P_VALUE'})
        gwas_df = magma_output_genes_df[['ID', 'NSNPS', 'P_VALUE']]
        gwas_df['FDR'] = multipletests(
            gwas_df['P_VALUE'], method='fdr_bh')[1]
        gwas_df = pd.merge(gwas_df, gene_range_df, on='ID', how='left')

    return gwas_df


def get_chr_len_df(fasta_file):
    chrom_len_csv = f'{fasta_file}.ChrLen.csv'
    if have_file(chrom_len_csv):
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

    # 设置ax
    data_df = data_df.dropna()
    data_df['CHROM'] = data_df['CHROM'].astype(str)
    data_df['POS'] = data_df['POS'].astype(int)
    data_df['VALUE'] = data_df['VALUE'].astype(float)

    ax.set_xlim(0 - total_length*0.05, total_length*1.05)
    max_y = np.max(data_df['VALUE']) if max_y is None else max_y
    ax.set_ylim(0, max_y*1.1)

    # 设置 x 轴和 y 轴的标签
    ax.set_xlabel('Chromosome')
    ax.set_ylabel(ylabel)

    # 隐藏坐标轴的上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 绘图
    c1 = "#617EB8"
    c2 = "#84B8D0"
    n = 0
    chr_ticks = []  # 用于存储每个染色体中点的位置
    chr_labels = []  # 用于存储染色体的编号
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
        # 计算染色体的中点位置并添加到列表中
        chr_ticks.append(
            (chr_df['POS'] + chr_df['CHROM'].map(chr_coord_dict)).mean())
        chr_labels.append(chr_id)
        n += 1

    # 设置 x 轴的刻度标签
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

    # 找到qval阈值对应的-log10(pval)
    # 找到 FDR 阈值对应的 -log10(p)
    if plot_fdr:
        # 防止 FDR=0 导致 -log10(inf) 的逻辑问题（你这里用 pval_lim 兜底）
        gwas_df = gwas_df.copy()
        gwas_df.loc[gwas_df['FDR'] == 0, 'FDR'] = pval_lim

        sig_mask = gwas_df['FDR'] < threshold_fdr

        if sig_mask.any():
            # 有显著点：用“显著点中最大的 P_VALUE”（最不显著的显著点）来定义阈值线
            p_at_thr = gwas_df.loc[sig_mask, 'P_VALUE'].max()
            threshold = -np.log10(max(p_at_thr, pval_lim))
        else:
            # 没有显著点：把线画在当前 y 轴顶部（不改变 ylim）
            y0, y1 = ax.get_ylim()
            threshold = y1 * 0.98  # 或者 y1 * 0.999 让它不压到边框上（看你审美）

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


def quantile_quantile_plot(p_value, title=None, ax=None, filter_zero_point=True):
    p_value = np.array(p_value)

    if ax is None:
        fig, ax = plt.subplots()

    if filter_zero_point:
        # 计算 -np.log10(p_value)
        log_p_value = -np.log10(p_value)
        # 找出 -np.log10(p_value) 不等于 0 的值
        mask = log_p_value != 0
        # 使用 mask 来筛选 p_value
        p_value = p_value[mask]

    # 计算理论分位数
    n = len(p_value)

    if n == 0:
        ax.plot([0, 1], [0, 1], c="#FB0324")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
    else:
        expected = -np.log10(np.linspace(1/n, 1, n))

        max_value = max(np.max(expected), np.max(-np.log10(p_value)))

        # 绘图
        ax.scatter(expected, -np.log10(np.sort(p_value)), s=10, c="#000000")

        lim = (0 - max_value*0.1, max_value*1.1)
        ax.plot(lim, lim, c="#FB0324")

        # 设置 x 轴和 y 轴的范围
        ax.set_xlim(0 - np.max(expected)*0.1, np.max(expected)*1.1)
        ax.set_ylim(0 - np.max(-np.log10(p_value)) *
                    0.1, np.max(-np.log10(p_value))*1.1)

    # 设置 x 轴和 y 轴的标签
    ax.set_xlabel('Expected $-log_{10}(P)$')
    ax.set_ylabel('Observed $-log_{10}(P)$')

    # 设置标题
    if title is not None:
        ax.set_title(title)

    if ax is None:
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
        Whether to save output PNG.
    """

    set_plot_style()

    # -----------------------------
    # Load data
    # -----------------------------
    gwas_df = pd.read_hdf(gwas_df_h5, key='df')
    chr_len_df = get_chr_len_df(genome_fasta_file)

    manhattan_df = gwas_df[['CHROM', 'POS', 'P_VALUE', 'FDR']]
    p_values = gwas_df['P_VALUE'].values

    if title is None:
        title = gwas_df_h5.split('/')[-1].replace('.h5', '')

    # -----------------------------
    # Create layout
    # -----------------------------
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[4, 1],   # 左侧更宽
        height_ratios=[1, 1],
        wspace=0.1,
        hspace=0.25
    )

    ax_manhattan = fig.add_subplot(gs[:, 0])   # 左边占两行
    ax_qq = fig.add_subplot(gs[0, 1])          # 右上
    ax_hist = fig.add_subplot(gs[1, 1])        # 右下

    # -----------------------------
    # Manhattan plot
    # -----------------------------
    genomewide_gwas_plot(
        manhattan_df,
        chr_len_df,
        threshold_fdr=threshold_fdr,
        ax=ax_manhattan
    )
    # 关闭标题
    ax_manhattan.set_title(None)
    # 删除x轴标签
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
    ax_hist.axhline(y=1, linestyle='--')  # 理论均匀分布密度
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

    plt.close(fig)

    # return fig
