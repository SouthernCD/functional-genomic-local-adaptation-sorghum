#!/usr/bin/env python
# coding: utf-8

# # 03. Functional enrichment
# 
# Compare environmental association signals across genomic annotations and gene properties, and examine their relationships with population-genetic variation.
# 

# # 03. 功能富集
# 
# 比较不同基因组注释和基因属性中的环境关联信号，并考察其与群体遗传变异的关系。
# 

# ### Environment setup
# 

# ### 环境设置
# 

# In[21]:


import warnings
warnings.filterwarnings('ignore')

import os
import socket
import sys
import json
import gzip
import pickle
from datetime import datetime
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from tqdm.auto import tqdm

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

with open('config/project.yaml') as handle:
    project_config = yaml.safe_load(handle)
project_dir = project_config['remote']['project_root']
if os.path.realpath(os.getcwd()) != os.path.realpath(project_dir):
    raise ValueError(f'Working directory {os.getcwd()} must match configured project root {project_dir}.')
data_dir = f'{project_dir}/data'
source_dir = f'{data_dir}/source'
results_dir = f'{data_dir}/results'
run_dir = f'{project_dir}/run'
enrich_dir = f'{results_dir}/enrichment'
os.makedirs(enrich_dir, exist_ok=True)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

print(f'Host: {socket.gethostname()}')
print(f'Working directory: {os.getcwd()}')
print(f'Python: {sys.executable}')
print(f"Environment prefix: {os.environ.get('CONDA_PREFIX', '<not set>')}")


# ## 1. GEA enrichment
# 
# Evaluate four environmental variables: precipitation of the warmest quarter, growing season length, extractable aluminium and Striga HS score. Site comparisons use nominal Wald P values; gene comparisons use MAGMA gene P values.
# 
# For sites, compare the upper-tail association scores using the ratio of foreground and remaining-background 95th percentiles of −log10(P). Gene comparisons measure Jaccard overlap with the top 5% of associated genes. Chromosome-wise circular shifts provide permutation distributions and two-sided empirical P values.
# 

# ## 1. GEA富集
# 
# 分析四个环境变量：最暖季度降水量、生长季长度、可提取铝含量和Striga HS score。位点级对比使用nominal Wald P值，基因级对比使用MAGMA基因P值。
# 
# 位点级分析比较上尾关联强度，以前景位点与背景其余位点的−log10(P)第95百分位数之比作为统计量。基因级分析计算目标集合与关联最强5%基因的Jaccard重叠。通过染色体内循环移位获得置换分布和双侧经验P值。
# 

# In[3]:


from src.plot import set_plot_style
from src.enrich import site_contrast_runner, plot_enrichment_section
from src.ld import prepare_ld_pairs, calculate_contrast_ld_overlap
set_plot_style()


# ### Association panels and shared parameters
# 
# Genome-wide variants have MAF ≥0.05; the CDS panel retains variants with MAF ≥0.01. Both use heterozygosity ≤0.05. Environmental scores are matched to variants or genes by ID. Use 1,000 permutations, and stratify genome-wide site comparisons into five MAF classes in addition to the combined panel.
# 

# ### 关联数据与共同参数
# 
# 全基因组变异采用MAF ≥0.05，CDS数据采用MAF ≥0.01，两者均要求杂合率 ≤0.05。按ID将环境关联分数与位点或基因对齐。采用1,000次置换；全基因组位点对比除合并分析外，还划分为五个MAF类别。
# 

# In[4]:


la_enrich_dir = f'{enrich_dir}/local_adaptation'
gwas_dir = f'{results_dir}/gwas'
os.makedirs(la_enrich_dir, exist_ok=True)
env_id_list = ['bio18', 'gsl', 'al', 'striga']
env_metadata_df = pd.read_csv(f'{source_dir}/envir_metadata.csv')
permutations = 1000
top_percent = 0.05
pool_num = project_config['resources']['interactive']['cores']
ld_window_size = 15000
min_r2 = 0.5
bin_width = 0.05
random_seed = 42
force_recompute = False

dataset_files = {
    'genome': f'{results_dir}/site_contexts.genome.maf0.05.h5',
    'cds': f'{results_dir}/site_contexts.cds.maf0.01.h5',
    'gene': f'{results_dir}/gene_contexts.h5',
}
vcf_files = {
    'genome': f'{results_dir}/landraces_qc_maf0.05_het0.05.vcf.gz',
    'cds': f'{results_dir}/landraces_cds_maf0.01_het0.05.vcf.gz',
}
with open(f'{project_dir}/config/contexts_settings.json') as handle:
    contexts_settings = json.load(handle)
with open(f'{project_dir}/config/contrasts_settings.json') as handle:
    contrasts_settings = json.load(handle)
stratify_label_map = [
    ['all', 'All MAF'], ['maf_0.05_0.1', '0.05–0.10'],
    ['maf_0.1_0.2', '0.10–0.20'], ['maf_0.2_0.3', '0.20–0.30'],
    ['maf_0.3_0.4', '0.30–0.40'], ['maf_0.4_0.5', '0.40–0.50'],
]


# In[5]:


stage_start = perf_counter()
print(f'[{datetime.now():%H:%M:%S}] Preparing association panels')
datasets = {}
for dataset, path in tqdm(dataset_files.items(), desc='Context panels'):
    data_df = pd.read_hdf(path, key='df')
    if data_df['ID'].isna().any() or not data_df['ID'].is_unique:
        raise ValueError(f'{dataset}: context IDs must be nonmissing and unique.')
    data_df = data_df.reset_index(drop=True)
    data_df[dataset] = True
    datasets[dataset] = data_df

for env_id in tqdm(env_id_list, desc='Environmental association scores'):
    for dataset, method, suffix, p_column in [
        ('genome', 'gemma', '', 'P_WALD'),
        ('cds', 'gemma', '_cds', 'P_WALD'),
        ('gene', 'magma', '', 'P_VALUE'),
    ]:
        name = f'{env_id}{suffix}'
        gwas_h5 = f'{gwas_dir}/{env_id}/{name}.{method}/{name}.{method}.gwas_df.h5'
        associations_df = pd.read_hdf(gwas_h5, key='df')[['ID', p_column]]
        associations_df[env_id] = -np.log10(associations_df[p_column])
        node_ids = datasets[dataset]['ID']
        datasets[dataset] = datasets[dataset].merge(
            associations_df[['ID', env_id]], on='ID', how='left', validate='one_to_one',
        )
        if not datasets[dataset]['ID'].equals(node_ids):
            raise ValueError(f'{dataset}: association merge changed the LD node order.')
for dataset in datasets:
    datasets[dataset] = datasets[dataset].reset_index(drop=True)
print(f'[{datetime.now():%H:%M:%S}] Association panels ready ({perf_counter() - stage_start:.1f}s)')
pd.DataFrame({dataset: {'rows': len(data_df), 'columns': len(data_df.columns)}
              for dataset, data_df in datasets.items()}).T


# ### Linkage between foreground and background signals
# 
# Calculate LD within 15 kb using the corresponding QC genotypes. Rank foreground and remaining-background sites separately by association strength and compare matching 5%-wide rank bins. Linkage summarizes the maximum opposite-group r² per site, scaled above r² =0.5, with a shuffled-score reference. LD indicates linked signals rather than a shared causal variant.
# 

# ### 前景与背景信号之间的连锁联系
# 
# 使用对应的质控基因型计算15 kb内的LD。前景与背景其余位点分别按关联强度排序，比较对应的5%宽度排名区间。连锁指标汇总各位点与另一组位点间的最大r²，并在r² =0.5以上缩放，以打乱关联分数作为参照。LD反映信号之间的连锁联系，不能确定共享因果位点。
# 

# In[6]:


stage_start = perf_counter()
ld_pairs_files = {
    dataset: f'{la_enrich_dir}/ld_pairs.{dataset}.by_index.win{ld_window_size}.r2{min_r2}.h5'
    for dataset in vcf_files
}
ld_dict = {}
for dataset, vcf_file in tqdm(vcf_files.items(), desc='LD panels'):
    legacy_pairs_h5 = f'{la_enrich_dir}/ld_pairs.{dataset}.by_id.win{ld_window_size}.r2{min_r2}.h5'
    ld_dict[dataset] = prepare_ld_pairs(
        datasets[dataset][['ID', 'CHROM', 'POS']], vcf_file, ld_pairs_files[dataset],
        legacy_pairs_h5=legacy_pairs_h5, ld_window_size=ld_window_size,
        min_r2=min_r2, pool_num=pool_num, force_recompute=force_recompute,
    )
print(f'[{datetime.now():%H:%M:%S}] LD panels ready ({perf_counter() - stage_start:.1f}s)')


# ### 1.1 Genomic compartments
# 
# Compare Genic, TE and Other intergenic sites using the genome-wide background, and Distal TEs and Proximal TEs within their respective Distal and Proximal backgrounds. Genic includes gene bodies and 2 kb flanks. Proximal includes gene bodies and 5 kb flanks; Distal lies outside that range. Other intergenic excludes both Genic and TE regions. In each comparison, the background defines the tested population and the foreground is compared with its remaining sites.
# 

# ### 1.1 基因组区室
# 
# 以全基因组为背景比较Genic、TE和Other intergenic；Distal TEs和Proximal TEs分别在Distal与Proximal背景中比较。Genic包括基因本体及两侧各2 kb；Proximal包括基因本体及两侧各5 kb，Distal为该范围之外的区域。Other intergenic排除Genic与TE区域。各对比以背景定义受检集合，将前景与该背景中的其余位点比较。
# 

# In[7]:


contrast_label_map = [
    ('genic_2kb_vs_genome', 'Genic vs. Genome'),
    ('TE_vs_genome', 'TE vs. Genome'),
    ('proximal_TE_vs_proximal', 'Proximal TEs vs. Proximal'),
    ('distal_TE_vs_distal', 'Distal TEs vs. Distal'),
    ('other_intergenic_vs_genome', 'Other intergenic vs. Genome'),
]
compartment_result_files = {
    contrast_id: {
        'enrichment': f'{la_enrich_dir}/{contrast_id}.enrichment.json.gz',
        'overlap': f'{la_enrich_dir}/{contrast_id}_overlap_results.pkl',
    }
    for contrast_id, _ in contrast_label_map
}
figure_prefix = f'{la_enrich_dir}/genomic_compartments.enrichment_and_overlap'


# In[8]:


stage_start = perf_counter()
for contrast_id, label in tqdm(contrast_label_map, desc='Genomic compartments'):
    contrast = contrasts_settings[contrast_id]
    context_df = datasets[contrast['dataset']]
    site_contrast_runner(
        contrast_id, contrast, context_df, env_id_list, la_enrich_dir,
        permutations=permutations, top_percent=top_percent, pool_num=pool_num,
        force_recompute=force_recompute,
    )
    calculate_contrast_ld_overlap(
        contrast_id, contrast, context_df, env_id_list, ld_dict[contrast['dataset']],
        la_enrich_dir, bin_width=bin_width, min_R2=min_r2, random_seed=random_seed,
        force_recompute=force_recompute,
    )
print(f'[{datetime.now():%H:%M:%S}] Compartment results ready ({perf_counter() - stage_start:.1f}s)')


# #### Display genomic-compartment enrichment
# 
# Each block contains four environmental panels. Upper panels show observed and permutation enrichment Z scores for all variants and each MAF class; stars mark two-sided P <0.05. Lower panels show foreground–background LD linkage by GEA-rank bin and its shuffled-score reference. Read enrichment direction relative to the permutation distribution; the raw association-score ratios are retained in the exported data.
# 

# #### 展示基因组区室富集
# 
# 每组对比包含四个环境变量面板。上半部分展示全部变异及各MAF类别的观测和置换富集Z-score，星号表示双侧P <0.05。下半部分展示各GEA排名区间的前景–背景LD联系及打乱分数后的参照。结合置换分布判断富集方向，原始关联强度比值保留在导出数据中。
# 

# In[9]:


compartment_plot = plot_enrichment_section(
    compartment_result_files, contrast_label_map, env_id_list,
    env_metadata_df, stratify_label_map, figure_prefix,
    figsize=(12, 19), block_gap=0.25, section_title_fontsize=15,
    panel_params=dict(wspace=0.15, hspace=0.35),
    letter_labels=True, env_titles_first_only=True,
)
plt.show()
print(compartment_plot['files'])


# ### 1.2 Gene structure
# 
# Compare GEA signals in upstream 2 kb, 5′ UTR, CDS, introns, 3′ UTR and downstream 2 kb regions. Gene structures follow the longest annotated transcript; upstream and downstream regions follow gene orientation. An annotation can overlap another gene's structure, so these categories need not be mutually exclusive.
# 
# Use two backgrounds: Genic, comprising gene bodies and 2 kb flanks, and Genome, comprising all genome-wide QC variants with MAF ≥0.05. Each structural category is compared with the remaining sites in its background. Both comparisons use the same genome-wide panel, including CDS sites; the lower-MAF CDS panel is not used here.
# 

# ### 1.2 基因结构
# 
# 比较上游2 kb、5′ UTR、CDS、内含子、3′ UTR和下游2 kb区域中的GEA信号。基因结构依据最长的注释转录本，上下游区域按基因方向定义。不同基因的结构注释可以重叠，因此这些类别不一定互斥。
# 
# 使用两种背景：Genic包括基因本体及两侧各2 kb，Genome包括全部MAF ≥0.05的全基因组质控变异。各结构类别分别与相应背景中的其余位点比较。两组均使用同一套全基因组数据，包括其中的CDS位点；此处不使用较低MAF阈值的CDS数据。
# 

# In[10]:


structure_labels = [
    ('upstream_2kb', 'Upstream 2 kb'),
    ('five_prime_UTR', '5′ UTR'),
    ('CDS', 'CDS'),
    ('intron', 'Intron'),
    ('three_prime_UTR', '3′ UTR'),
    ('downstream_2kb', 'Downstream 2 kb'),
]
structure_contrast_labels = {
    background: [(f'{context_id}_vs_{background}', f'{label} vs. {background.capitalize()}')
                 for context_id, label in structure_labels]
    for background in ['genic', 'genome']
}
structure_result_files = {
    contrast_id: {
        'enrichment': f'{la_enrich_dir}/{contrast_id}.enrichment.json.gz',
        'overlap': f'{la_enrich_dir}/{contrast_id}_overlap_results.pkl',
    }
    for labels in structure_contrast_labels.values()
    for contrast_id, _ in labels
}
structure_figure_prefix = f'{la_enrich_dir}/gene_structure'
plot_params = dict(
    figsize=(12, 22.8), block_gap=0.25, section_title_fontsize=15,
    panel_params=dict(wspace=0.15, hspace=0.35),
    letter_labels=True, env_titles_first_only=True,
)


# #### 1.2.1 Genic background
# 
# Compare each structural category with other Genic sites to assess differences within gene-associated regions. The Genic background includes gene bodies and 2 kb flanks on both sides.
# 

# #### 1.2.1 Genic背景
# 
# 将各结构类别与Genic中的其余位点比较，考察基因相关区域内部的关联强度差异。Genic背景包括基因本体及两侧各2 kb。
# 

# In[11]:


stage_start = perf_counter()
for contrast_id, label in tqdm(structure_contrast_labels['genic'], desc='Gene structure: Genic'):
    contrast = contrasts_settings[contrast_id]
    context_df = datasets[contrast['dataset']]
    site_contrast_runner(
        contrast_id, contrast, context_df, env_id_list, la_enrich_dir,
        permutations=permutations, top_percent=top_percent, pool_num=pool_num,
        force_recompute=force_recompute,
    )
    calculate_contrast_ld_overlap(
        contrast_id, contrast, context_df, env_id_list, ld_dict[contrast['dataset']],
        la_enrich_dir, bin_width=bin_width, min_R2=min_r2, random_seed=random_seed,
        force_recompute=force_recompute,
    )
print(f'[{datetime.now():%H:%M:%S}] Genic comparisons ready ({perf_counter() - stage_start:.1f}s)')


# ##### Display structural enrichment within the Genic background
# 
# Six blocks show the structural categories, with environmental column titles appearing once at the top. Upper panels compare observed and permutation enrichment Z scores across MAF classes; stars mark two-sided P <0.05. Lower panels show cross-group LD linkage by GEA-rank bin, with a shuffled-score reference.
# 

# ##### 展示Genic背景中的基因结构富集
# 
# 六组面板分别展示各结构类别，环境变量列标题仅在整图顶部出现。上半部分比较各MAF类别的观测与置换富集Z-score，星号表示双侧P <0.05；下半部分展示各GEA排名区间的跨组LD联系及打乱分数后的参照。
# 

# In[12]:


genic_structure_plot = plot_enrichment_section(
    structure_result_files, structure_contrast_labels['genic'], env_id_list,
    env_metadata_df, stratify_label_map,
    f'{structure_figure_prefix}.genic.enrichment_and_overlap', **plot_params,
)
plt.show()
print(genic_structure_plot['files'])


# #### 1.2.2 Genome background
# 
# Compare the same structural categories with the remaining genome-wide sites to assess their association enrichment relative to the broader genomic background.
# 

# #### 1.2.2 Genome背景
# 
# 将相同结构类别与全基因组中的其余位点比较，考察其相对于更广泛基因组背景的关联富集。
# 

# In[13]:


stage_start = perf_counter()
for contrast_id, label in tqdm(structure_contrast_labels['genome'], desc='Gene structure: Genome'):
    contrast = contrasts_settings[contrast_id]
    context_df = datasets[contrast['dataset']]
    site_contrast_runner(
        contrast_id, contrast, context_df, env_id_list, la_enrich_dir,
        permutations=permutations, top_percent=top_percent, pool_num=pool_num,
        force_recompute=force_recompute,
    )
    calculate_contrast_ld_overlap(
        contrast_id, contrast, context_df, env_id_list, ld_dict[contrast['dataset']],
        la_enrich_dir, bin_width=bin_width, min_R2=min_r2, random_seed=random_seed,
        force_recompute=force_recompute,
    )
print(f'[{datetime.now():%H:%M:%S}] Genome comparisons ready ({perf_counter() - stage_start:.1f}s)')


# ##### Display structural enrichment within the Genome background
# 
# Six blocks show the structural categories, with environmental column titles appearing once at the top. Upper panels compare observed and permutation enrichment Z scores across MAF classes; stars mark two-sided P <0.05. Lower panels show cross-group LD linkage by GEA-rank bin, with a shuffled-score reference.
# 

# ##### 展示Genome背景中的基因结构富集
# 
# 六组面板分别展示各结构类别，环境变量列标题仅在整图顶部出现。上半部分比较各MAF类别的观测与置换富集Z-score，星号表示双侧P <0.05；下半部分展示各GEA排名区间的跨组LD联系及打乱分数后的参照。
# 

# In[14]:


genome_structure_plot = plot_enrichment_section(
    structure_result_files, structure_contrast_labels['genome'], env_id_list,
    env_metadata_df, stratify_label_map,
    f'{structure_figure_prefix}.genome.enrichment_and_overlap', **plot_params,
)
plt.show()
print(genome_structure_plot['files'])


# ### 1.3 Coding mutations
# 
# Compare GEA signals among missense, synonymous and putative loss-of-function (LoF) variants using SnpEff consequence annotations. These are predicted effects; putative LoF does not establish experimentally confirmed loss of function. Use the CDS panel with MAF ≥0.01 and heterozygosity ≤0.05, with environmental associations calculated for that panel.
# 
# First compare each consequence class with the remaining CDS variants. Then compare putative LoF variants in two lower-frequency classes to examine differences within the LoF set.
# 

# ### 1.3 编码变异
# 
# 依据SnpEff后果注释，比较错义、同义和候选功能缺失（putative LoF）变异中的GEA信号。这些类别描述预测后果；候选LoF不能等同于经实验确认的功能缺失。使用MAF ≥0.01、杂合率 ≤0.05的CDS数据及其对应的环境关联结果。
# 
# 先将各后果类别与其余CDS变异比较，再比较两个较低频率类别中的候选LoF变异，考察LoF集合内部的差异。
# 

# In[15]:


coding_contrast_labels = [
    ('missense_vs_CDS', 'Missense vs. CDS'),
    ('synonymous_vs_CDS', 'Synonymous vs. CDS'),
    ('putative_LoF_vs_CDS', 'Putative LoF vs. CDS'),
]
lof_contrast_id = 'putative_LoF_MAF_0.01_0.03_vs_putative_LoF_MAF_0.03_0.05'
lof_contrast_labels = [(lof_contrast_id, 'Putative LoF: MAF 0.01–0.03 vs. 0.03–0.05')]
coding_result_files = {
    contrast_id: {
        'enrichment': f'{la_enrich_dir}/{contrast_id}.enrichment.json.gz',
        'overlap': f'{la_enrich_dir}/{contrast_id}_overlap_results.pkl',
    }
    for contrast_id, _ in coding_contrast_labels + lof_contrast_labels
}
cds_stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.01_0.03_cds_0.01', '0.01–0.03'],
    ['maf_0.03_0.05_cds_0.01', '0.03–0.05'],
    ['maf_0.05_0.1_cds_0.01', '0.05–0.10'],
    ['maf_0.1_0.2_cds_0.01', '0.10–0.20'],
    ['maf_0.2_0.3_cds_0.01', '0.20–0.30'],
    ['maf_0.3_0.4_cds_0.01', '0.30–0.40'],
    ['maf_0.4_0.5_cds_0.01', '0.40–0.50'],
]
coding_figure_prefix = f'{la_enrich_dir}/coding_mutations'
plot_params = dict(
    figsize=(12, 13.2), block_gap=0.25, section_title_fontsize=15,
    panel_params=dict(wspace=0.15, hspace=0.35, height_ratios=(3, 1)),
    letter_labels=True, env_titles_first_only=True,
)


# #### 1.3.1 Coding consequences within the CDS background
# 
# Compare missense, synonymous and putative LoF variants with the remaining CDS sites. Analyze all retained sites and seven MAF classes, including the rarer 0.01–0.03 and 0.03–0.05 classes. This describes whether consequence classes differ in upper-tail GEA strength relative to the same coding background.
# 

# #### 1.3.1 CDS背景中的编码后果
# 
# 将错义、同义及候选LoF变异与其余CDS位点比较。分别分析全部保留位点和七个MAF类别，包括较稀有的0.01–0.03和0.03–0.05类别，考察不同后果类别相对于同一编码背景的上尾GEA关联强度。
# 

# In[16]:


stage_start = perf_counter()
for contrast_id, label in tqdm(coding_contrast_labels, desc='Coding consequences'):
    contrast = contrasts_settings[contrast_id]
    context_df = datasets[contrast['dataset']]
    site_contrast_runner(
        contrast_id, contrast, context_df, env_id_list, la_enrich_dir,
        permutations=permutations, top_percent=top_percent, pool_num=pool_num,
        force_recompute=force_recompute,
    )
    calculate_contrast_ld_overlap(
        contrast_id, contrast, context_df, env_id_list, ld_dict[contrast['dataset']],
        la_enrich_dir, bin_width=bin_width, min_R2=min_r2, random_seed=random_seed,
        force_recompute=force_recompute,
    )
print(f'[{datetime.now():%H:%M:%S}] Coding-consequence results ready ({perf_counter() - stage_start:.1f}s)')


# ##### Display coding-consequence enrichment
# 
# Three lettered blocks share four environmental column titles at the top. Upper panels show observed and permutation enrichment Z scores across MAF classes; stars mark two-sided P <0.05. Lower panels show cross-group LD linkage by GEA-rank bin and its shuffled-score reference.
# 

# ##### 展示编码后果富集
# 
# 三组字母编号面板共用整图顶部的四个环境变量列标题。上半部分展示各MAF类别的观测与置换富集Z-score，星号表示双侧P <0.05；下半部分展示各GEA排名区间的跨组LD联系及打乱分数后的参照。
# 

# In[17]:


coding_plot = plot_enrichment_section(
    coding_result_files, coding_contrast_labels, env_id_list,
    env_metadata_df, cds_stratify_label_map,
    f'{coding_figure_prefix}.consequences.enrichment_and_overlap', **plot_params,
)
plt.show()
print(coding_plot['files'])


# #### 1.3.2 GEA strength across putative LoF frequency classes
# 
# Compare putative LoF variants with 0.01 ≤MAF <0.03 against those with 0.03 ≤MAF <0.05. Both groups contain only putative LoF variants, so the comparison concerns allele-frequency classes within this consequence category. Do not apply additional MAF stratification because frequency defines the two groups.
# 

# #### 1.3.2 不同频率候选LoF变异的GEA强度
# 
# 以0.03 ≤MAF <0.05的候选LoF变异为背景，比较0.01 ≤MAF <0.03的候选LoF变异。两组均只包含候选LoF位点，因此该对比考察同一后果类别内部的等位基因频率差异。频率已用于定义两组，不再额外进行MAF分层。
# 

# In[18]:


stage_start = perf_counter()
contrast = dict(contrasts_settings[lof_contrast_id])
contrast['maf_stratify'] = ['all']
context_df = datasets[contrast['dataset']]
site_contrast_runner(
    lof_contrast_id, contrast, context_df, env_id_list, la_enrich_dir,
    permutations=permutations, top_percent=top_percent, pool_num=pool_num,
    force_recompute=force_recompute,
)
calculate_contrast_ld_overlap(
    lof_contrast_id, contrast, context_df, env_id_list, ld_dict[contrast['dataset']],
    la_enrich_dir, bin_width=bin_width, min_R2=min_r2, random_seed=random_seed,
    force_recompute=force_recompute,
)
print(f'[{datetime.now():%H:%M:%S}] LoF frequency comparison ready ({perf_counter() - stage_start:.1f}s)')


# ##### Display the LoF frequency comparison
# 
# Upper panels show the observed and permutation score ratio for the two LoF frequency classes, displayed as Z scores. Ratios above 1 indicate stronger upper-tail associations in the 0.01–0.03 group; stars mark two-sided P <0.05. Lower panels show linkage between the two frequency groups across GEA-rank bins. Raw ratios are retained in the exported data.
# 

# ##### 展示LoF频率对比
# 
# 上半部分以Z-score展示两个LoF频率类别的观测与置换关联强度比值。原始比值大于1表示0.01–0.03组的上尾关联更强，星号表示双侧P <0.05。下半部分展示两个频率组在各GEA排名区间的连锁联系，原始比值保留在导出数据中。
# 

# In[22]:


lof_frequency_plot = plot_enrichment_section(
    coding_result_files, lof_contrast_labels, env_id_list,
    env_metadata_df, [['all', 'Combined classes']],
    f'{coding_figure_prefix}.lof_frequency.enrichment_and_overlap',
    figsize=(12, 4.0), section_title_fontsize=15,
    panel_params=dict(wspace=0.15, hspace=0.45, height_ratios=(0.8, 1)),
    letter_labels=True, env_titles_first_only=True,
)
plt.show()
print(lof_frequency_plot['files'])


# ### 1.4 Epigenomic features
# 
# Compare GEA signals within accessible chromatin regions (ACRs) and unmethylated regions (UMRs), using their annotated boundaries without additional flanks. Analyze each annotation against the Genome background and within Proximal and Distal regions. Proximal comprises gene bodies and 5 kb flanks on each side; Distal lies outside this range. Each foreground is compared with the remaining sites in its background.
# 
# All six comparisons use genome-wide QC variants with MAF ≥0.05. ACR and UMR membership can overlap other annotations; enrichment describes association patterns rather than demonstrating a regulatory mechanism.
# 

# ### 1.4 表观基因组特征
# 
# 比较开放染色质区域（ACRs）和未甲基化区域（UMRs）中的GEA信号，使用注释边界，不额外增加侧翼范围。每类注释分别在Genome背景，以及Proximal和Distal区域内分析。Proximal包括基因本体及两侧各5 kb，Distal为该范围之外的区域。各前景集合与相应背景中的其余位点比较。
# 
# 六组对比均使用MAF ≥0.05的全基因组质控变异。ACR和UMR可以与其他注释重叠；富集描述关联模式，不能直接证明调控机制。
# 

# In[ ]:


epigenomic_contrast_labels = {
    'ACRs': [
        ('ACR_vs_genome', 'ACRs vs. Genome'),
        ('proximal_ACR_vs_proximal', 'Proximal ACRs vs. Proximal'),
        ('distal_ACR_vs_distal', 'Distal ACRs vs. Distal'),
    ],
    'UMRs': [
        ('UMR_vs_genome', 'UMRs vs. Genome'),
        ('proximal_UMR_vs_proximal', 'Proximal UMRs vs. Proximal'),
        ('distal_UMR_vs_distal', 'Distal UMRs vs. Distal'),
    ],
}
epigenomic_result_files = {
    contrast_id: {
        'enrichment': f'{la_enrich_dir}/{contrast_id}.enrichment.json.gz',
        'overlap': f'{la_enrich_dir}/{contrast_id}_overlap_results.pkl',
    }
    for labels in epigenomic_contrast_labels.values()
    for contrast_id, _ in labels
}
epigenomic_figure_prefix = f'{la_enrich_dir}/epigenomic_features'
plot_params = dict(
    figsize=(12, 11.4), block_gap=0.25, section_title_fontsize=15,
    panel_params=dict(wspace=0.15, hspace=0.35),
    letter_labels=True, env_titles_first_only=True,
)


# #### 1.4.1 Accessible chromatin regions (ACRs)
# 
# Compare ACR-associated variants with the remaining Genome sites, then compare Proximal and Distal ACRs within their corresponding backgrounds. These three contrasts assess association enrichment across different genomic neighborhoods.
# 

# #### 1.4.1 开放染色质区域（ACRs）
# 
# 先将ACR内的变异与其余Genome位点比较，再将Proximal和Distal ACRs分别与相应背景中的其余位点比较，考察不同基因组邻域中的关联富集。
# 

# In[ ]:


stage_start = perf_counter()
for contrast_id, label in tqdm(epigenomic_contrast_labels['ACRs'], desc='ACRs enrichment'):
    contrast = contrasts_settings[contrast_id]
    context_df = datasets[contrast['dataset']]
    site_contrast_runner(
        contrast_id, contrast, context_df, env_id_list, la_enrich_dir,
        permutations=permutations, top_percent=top_percent, pool_num=pool_num,
        force_recompute=force_recompute,
    )
    calculate_contrast_ld_overlap(
        contrast_id, contrast, context_df, env_id_list, ld_dict[contrast['dataset']],
        la_enrich_dir, bin_width=bin_width, min_R2=min_r2, random_seed=random_seed,
        force_recompute=force_recompute,
    )
print(f'[{datetime.now():%H:%M:%S}] ACRs comparisons ready ({perf_counter() - stage_start:.1f}s)')


# ##### Display ACRs enrichment
# 
# Three lettered blocks display the Genome, Proximal and Distal comparisons, with four environmental column titles shared at the top. Upper panels show observed and permutation enrichment Z scores for all variants and five MAF classes; stars mark two-sided P <0.05. Lower panels show cross-group LD linkage across GEA-rank bins and its shuffled-score reference.
# 

# ##### 展示ACRs富集
# 
# 三组字母编号面板分别展示Genome、Proximal和Distal对比，共用整图顶部的四个环境变量列标题。上半部分展示全部变异及五个MAF类别的观测与置换富集Z-score，星号表示双侧P <0.05；下半部分展示各GEA排名区间的跨组LD联系及打乱分数后的参照。
# 

# In[ ]:


acrs_plot = plot_enrichment_section(
    epigenomic_result_files, epigenomic_contrast_labels['ACRs'], env_id_list,
    env_metadata_df, stratify_label_map,
    f'{epigenomic_figure_prefix}.acrs.enrichment_and_overlap', **plot_params,
)
plt.show()
print(acrs_plot['files'])


# #### 1.4.2 Unmethylated regions (UMRs)
# 
# Apply the same three backgrounds to UMR-associated variants. Compare genome-wide enrichment with the Proximal and Distal contrasts to describe how UMR association signals vary across genomic neighborhoods.
# 

# #### 1.4.2 未甲基化区域（UMRs）
# 
# 对UMR内的变异采用相同的三种背景。结合全基因组、Proximal和Distal对比，描述UMR关联信号在不同基因组邻域中的分布。
# 

# In[ ]:


stage_start = perf_counter()
for contrast_id, label in tqdm(epigenomic_contrast_labels['UMRs'], desc='UMRs enrichment'):
    contrast = contrasts_settings[contrast_id]
    context_df = datasets[contrast['dataset']]
    site_contrast_runner(
        contrast_id, contrast, context_df, env_id_list, la_enrich_dir,
        permutations=permutations, top_percent=top_percent, pool_num=pool_num,
        force_recompute=force_recompute,
    )
    calculate_contrast_ld_overlap(
        contrast_id, contrast, context_df, env_id_list, ld_dict[contrast['dataset']],
        la_enrich_dir, bin_width=bin_width, min_R2=min_r2, random_seed=random_seed,
        force_recompute=force_recompute,
    )
print(f'[{datetime.now():%H:%M:%S}] UMRs comparisons ready ({perf_counter() - stage_start:.1f}s)')


# ##### Display UMRs enrichment
# 
# Three lettered blocks display the Genome, Proximal and Distal comparisons, with four environmental column titles shared at the top. Upper panels show observed and permutation enrichment Z scores for all variants and five MAF classes; stars mark two-sided P <0.05. Lower panels show cross-group LD linkage across GEA-rank bins and its shuffled-score reference.
# 

# ##### 展示UMRs富集
# 
# 三组字母编号面板分别展示Genome、Proximal和Distal对比，共用整图顶部的四个环境变量列标题。上半部分展示全部变异及五个MAF类别的观测与置换富集Z-score，星号表示双侧P <0.05；下半部分展示各GEA排名区间的跨组LD联系及打乱分数后的参照。
# 

# In[ ]:


umrs_plot = plot_enrichment_section(
    epigenomic_result_files, epigenomic_contrast_labels['UMRs'], env_id_list,
    env_metadata_df, stratify_label_map,
    f'{epigenomic_figure_prefix}.umrs.enrichment_and_overlap', **plot_params,
)
plt.show()
print(umrs_plot['files'])


# ### Combined genomic annotations
# 

# ### 组合基因组注释
# 

# #### Proximal ACR-only vs. TE-only
# 

# #### Proximal中仅ACR与仅TE的区域比较
# 

# In[61]:


from src.enrich import site_contrast_runner
from src.ld import calculate_contrast_ld_overlap
import pickle

contrast_id = "proximal_ACR_not_TE_vs_proximal_TE_not_ACR"
permutations = 1000
top_percent = 0.05
pool_num = 40
contrast = contrasts_settings[contrast_id]
print(f"Running enrichment for contrast: {contrast_id}")
print(f"Using dataset: {contrast['dataset']}")
print(f"Using permutations: {permutations}, top_percent: {top_percent}, pool_num: {pool_num}")
print(f"  Foreground set: {contrast['foreground']}")
print(f"  Background set: {contrast['background']}")
print(f"  MAF stratify: {contrast['maf_stratify']}")
context_df = datasets[contrast["dataset"]]
results_json_file = site_contrast_runner(contrast_id, contrast, context_df, env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)
ld_overlap_pkl_file = calculate_contrast_ld_overlap(contrast_id, contrast, context_df, env_id_list, ld_dict[contrast["dataset"]], la_enrich_dir, bin_width=0.05, min_R2=0.5, random_seed=42)
with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
    results_json = json.load(f)
with open(ld_overlap_pkl_file, 'rb') as f:
    ld_overlap_dict = pickle.load(f)


# In[62]:


from src.enrich import plot_enrichment_and_overlap

stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50'],
]
print(f"Enrichment results for contrast: {contrast_id}")
save_path = os.path.join(la_enrich_dir, f"{contrast_id}.enrichment_and_overlap.pdf")
fig, axs = plot_enrichment_and_overlap(results_json, ld_overlap_dict, env_id_list, env_metadata_df, stratify_label_map)
fig.savefig(save_path, bbox_inches='tight')


# #### Proximal ACR–TE–UMR intersection
# 

# #### Proximal中的ACR、TE与UMR交集
# 

# In[5]:


from src.enrich import site_contrast_runner
from src.ld import calculate_contrast_ld_overlap
import pickle

contrast_id = "proximal_ACR_TE_UMR_vs_proximal"
permutations = 1000
top_percent = 0.05
pool_num = 40
contrast = contrasts_settings[contrast_id]
print(f"Running enrichment for contrast: {contrast_id}")
print(f"Using dataset: {contrast['dataset']}")
print(f"Using permutations: {permutations}, top_percent: {top_percent}, pool_num: {pool_num}")
print(f"  Foreground set: {contrast['foreground']}")
print(f"  Background set: {contrast['background']}")
print(f"  MAF stratify: {contrast['maf_stratify']}")
context_df = datasets[contrast["dataset"]]
results_json_file = site_contrast_runner(contrast_id, contrast, context_df, env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)
ld_overlap_pkl_file = calculate_contrast_ld_overlap(contrast_id, contrast, context_df, env_id_list, ld_dict[contrast["dataset"]], la_enrich_dir, bin_width=0.05, min_R2=0.5, random_seed=42)
with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
    results_json = json.load(f)
with open(ld_overlap_pkl_file, 'rb') as f:
    ld_overlap_dict = pickle.load(f)


# In[6]:


from src.enrich import plot_enrichment_and_overlap

stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50'],
]
print(f"Enrichment results for contrast: {contrast_id}")
save_path = os.path.join(la_enrich_dir, f"{contrast_id}.enrichment_and_overlap.pdf")
fig, axs = plot_enrichment_and_overlap(results_json, ld_overlap_dict, env_id_list, env_metadata_df, stratify_label_map)
fig.savefig(save_path, bbox_inches='tight')


# ### riboSNitches
# 

# ### riboSNitches
# 

# riboSNitch_vs_gene_body

# In[43]:


from src.enrich import site_contrast_runner
from src.ld import calculate_contrast_ld_overlap
import pickle

contrast_id = "riboSNitch_vs_gene_body"
permutations = 1000
top_percent = 0.05
pool_num = 40
contrast = contrasts_settings[contrast_id]
print(f"Running enrichment for contrast: {contrast_id}")
print(f"Using dataset: {contrast['dataset']}")
print(f"Using permutations: {permutations}, top_percent: {top_percent}, pool_num: {pool_num}")
print(f"  Foreground set: {contrast['foreground']}")
print(f"  Background set: {contrast['background']}")
print(f"  MAF stratify: {contrast['maf_stratify']}")
context_df = datasets[contrast["dataset"]]
results_json_file = site_contrast_runner(contrast_id, contrast, context_df, env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)
ld_overlap_pkl_file = calculate_contrast_ld_overlap(contrast_id, contrast, context_df, env_id_list, ld_dict[contrast["dataset"]], la_enrich_dir, bin_width=0.05, min_R2=0.5, random_seed=42)
with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
    results_json = json.load(f)
with open(ld_overlap_pkl_file, 'rb') as f:
    ld_overlap_dict = pickle.load(f)


# In[44]:


from src.enrich import plot_enrichment_and_overlap

stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50'],
]
print(f"Enrichment results for contrast: {contrast_id}")
save_path = os.path.join(la_enrich_dir, f"{contrast_id}.enrichment_and_overlap.pdf")
fig, axs = plot_enrichment_and_overlap(results_json, ld_overlap_dict, env_id_list, env_metadata_df, stratify_label_map)
fig.savefig(save_path, bbox_inches='tight')


# ### 1.5 Evolutionary conservation & Population genetics

# #### 1.5.1 Nucleotide diversity (π)

# ##### 1.5.1.1 low_pi

# In[63]:


from src.enrich import site_contrast_runner
from src.ld import calculate_contrast_ld_overlap
import pickle

contrast_id = "low_pi"
permutations = 1000
top_percent = 0.05
pool_num = 40
contrast = contrasts_settings[contrast_id]
print(f"Running enrichment for contrast: {contrast_id}")
print(f"Using dataset: {contrast['dataset']}")
print(f"Using permutations: {permutations}, top_percent: {top_percent}, pool_num: {pool_num}")
print(f"  Foreground set: {contrast['foreground']}")
print(f"  Background set: {contrast['background']}")
print(f"  MAF stratify: {contrast['maf_stratify']}")
context_df = datasets[contrast["dataset"]]
results_json_file = site_contrast_runner(contrast_id, contrast, context_df, env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)
ld_overlap_pkl_file = calculate_contrast_ld_overlap(contrast_id, contrast, context_df, env_id_list, ld_dict[contrast["dataset"]], la_enrich_dir, bin_width=0.05, min_R2=0.5, random_seed=42)
with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
    results_json = json.load(f)
with open(ld_overlap_pkl_file, 'rb') as f:
    ld_overlap_dict = pickle.load(f)


# In[64]:


from src.enrich import plot_enrichment_and_overlap

stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50'],
]
print(f"Enrichment results for contrast: {contrast_id}")
save_path = os.path.join(la_enrich_dir, f"{contrast_id}.enrichment_and_overlap.pdf")
fig, axs = plot_enrichment_and_overlap(results_json, ld_overlap_dict, env_id_list, env_metadata_df, stratify_label_map)
fig.savefig(save_path, bbox_inches='tight')


# ##### 1.5.1.2 med_pi

# In[65]:


from src.enrich import site_contrast_runner
from src.ld import calculate_contrast_ld_overlap
import pickle

contrast_id = "med_pi"
permutations = 1000
top_percent = 0.05
pool_num = 40
contrast = contrasts_settings[contrast_id]
print(f"Running enrichment for contrast: {contrast_id}")
print(f"Using dataset: {contrast['dataset']}")
print(f"Using permutations: {permutations}, top_percent: {top_percent}, pool_num: {pool_num}")
print(f"  Foreground set: {contrast['foreground']}")
print(f"  Background set: {contrast['background']}")
print(f"  MAF stratify: {contrast['maf_stratify']}")
context_df = datasets[contrast["dataset"]]
results_json_file = site_contrast_runner(contrast_id, contrast, context_df, env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)
ld_overlap_pkl_file = calculate_contrast_ld_overlap(contrast_id, contrast, context_df, env_id_list, ld_dict[contrast["dataset"]], la_enrich_dir, bin_width=0.05, min_R2=0.5, random_seed=42)
with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
    results_json = json.load(f)
with open(ld_overlap_pkl_file, 'rb') as f:
    ld_overlap_dict = pickle.load(f)


# In[66]:


from src.enrich import plot_enrichment_and_overlap

stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50'],
]
print(f"Enrichment results for contrast: {contrast_id}")
save_path = os.path.join(la_enrich_dir, f"{contrast_id}.enrichment_and_overlap.pdf")
fig, axs = plot_enrichment_and_overlap(results_json, ld_overlap_dict, env_id_list, env_metadata_df, stratify_label_map)
fig.savefig(save_path, bbox_inches='tight')


# ##### 1.5.1.3 high_pi

# In[67]:


from src.enrich import site_contrast_runner
from src.ld import calculate_contrast_ld_overlap
import pickle

contrast_id = "high_pi"
permutations = 1000
top_percent = 0.05
pool_num = 40
contrast = contrasts_settings[contrast_id]
print(f"Running enrichment for contrast: {contrast_id}")
print(f"Using dataset: {contrast['dataset']}")
print(f"Using permutations: {permutations}, top_percent: {top_percent}, pool_num: {pool_num}")
print(f"  Foreground set: {contrast['foreground']}")
print(f"  Background set: {contrast['background']}")
print(f"  MAF stratify: {contrast['maf_stratify']}")
context_df = datasets[contrast["dataset"]]
results_json_file = site_contrast_runner(contrast_id, contrast, context_df, env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)
ld_overlap_pkl_file = calculate_contrast_ld_overlap(contrast_id, contrast, context_df, env_id_list, ld_dict[contrast["dataset"]], la_enrich_dir, bin_width=0.05, min_R2=0.5, random_seed=42)
with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
    results_json = json.load(f)
with open(ld_overlap_pkl_file, 'rb') as f:
    ld_overlap_dict = pickle.load(f)


# In[68]:


from src.enrich import plot_enrichment_and_overlap

stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50'],
]
print(f"Enrichment results for contrast: {contrast_id}")
save_path = os.path.join(la_enrich_dir, f"{contrast_id}.enrichment_and_overlap.pdf")
fig, axs = plot_enrichment_and_overlap(results_json, ld_overlap_dict, env_id_list, env_metadata_df, stratify_label_map)
fig.savefig(save_path, bbox_inches='tight')


# #### 1.5.2 Minor allele frequency (MAF)

# In[69]:


from src.enrich import site_contrast_runner

permutations = 1000
top_percent = 0.05
pool_num = 40

contrast_id = "maf_0.05_0.1"
contrast = contrasts_settings[contrast_id]
contrast['maf_stratify'] = ['all']
maf_0_05_0_1_results_json_file = site_contrast_runner(contrast_id, contrast, datasets[contrast["dataset"]], env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)

contrast_id = "maf_0.1_0.2"
contrast = contrasts_settings[contrast_id]
contrast['maf_stratify'] = ['all']
maf_0_1_0_2_results_json_file = site_contrast_runner(contrast_id, contrast, datasets[contrast["dataset"]], env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)

contrast_id = "maf_0.2_0.3"
contrast = contrasts_settings[contrast_id]
contrast['maf_stratify'] = ['all']
maf_0_2_0_3_results_json_file = site_contrast_runner(contrast_id, contrast, datasets[contrast["dataset"]], env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)

contrast_id = "maf_0.3_0.4"
contrast = contrasts_settings[contrast_id]
contrast['maf_stratify'] = ['all']
maf_0_3_0_4_results_json_file = site_contrast_runner(contrast_id, contrast, datasets[contrast["dataset"]], env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)

contrast_id = "maf_0.4_0.5"
contrast = contrasts_settings[contrast_id]
contrast['maf_stratify'] = ['all']
maf_0_4_0_5_results_json_file = site_contrast_runner(contrast_id, contrast, datasets[contrast["dataset"]], env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)


# In[70]:


from src.enrich import plot_set_enrichment

set_label_map = [
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50']
]

set_results_json = {}
for contrast_id, results_json_file in [("maf_0.05_0.1", maf_0_05_0_1_results_json_file), ("maf_0.1_0.2", maf_0_1_0_2_results_json_file), ("maf_0.2_0.3", maf_0_2_0_3_results_json_file), ("maf_0.3_0.4", maf_0_3_0_4_results_json_file), ("maf_0.4_0.5", maf_0_4_0_5_results_json_file)]:
    with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
        results_json = json.load(f)
    set_results_json[contrast_id] = results_json

plot_set_enrichment(set_results_json, env_id_list, env_metadata_df, set_label_map, significance_threshold=0.05)


# #### 1.5.3 Phylogenetic conservation (PhyloP)

# ##### 1.5.3.1 non_CEs

# In[5]:


from src.enrich import site_contrast_runner
from src.ld import calculate_contrast_ld_overlap
import pickle

contrast_id = "non_CEs"
permutations = 1000
top_percent = 0.05
pool_num = 40
contrast = contrasts_settings[contrast_id]
print(f"Running enrichment for contrast: {contrast_id}")
print(f"Using dataset: {contrast['dataset']}")
print(f"Using permutations: {permutations}, top_percent: {top_percent}, pool_num: {pool_num}")
print(f"  Foreground set: {contrast['foreground']}")
print(f"  Background set: {contrast['background']}")
print(f"  MAF stratify: {contrast['maf_stratify']}")
context_df = datasets[contrast["dataset"]]
results_json_file = site_contrast_runner(contrast_id, contrast, context_df, env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)
ld_overlap_pkl_file = calculate_contrast_ld_overlap(contrast_id, contrast, context_df, env_id_list, ld_dict[contrast["dataset"]], la_enrich_dir, bin_width=0.05, min_R2=0.5, random_seed=42)
with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
    results_json = json.load(f)
with open(ld_overlap_pkl_file, 'rb') as f:
    ld_overlap_dict = pickle.load(f)


# In[6]:


from src.enrich import plot_enrichment_and_overlap

stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50'],
]
print(f"Enrichment results for contrast: {contrast_id}")
save_path = os.path.join(la_enrich_dir, f"{contrast_id}.enrichment_and_overlap.pdf")
fig, axs = plot_enrichment_and_overlap(results_json, ld_overlap_dict, env_id_list, env_metadata_df, stratify_label_map)
fig.savefig(save_path, bbox_inches='tight')


# ##### 1.5.3.2 conserved_sites

# In[7]:


from src.enrich import site_contrast_runner
from src.ld import calculate_contrast_ld_overlap
import pickle

contrast_id = "conserved_sites"
permutations = 1000
top_percent = 0.05
pool_num = 40
contrast = contrasts_settings[contrast_id]
print(f"Running enrichment for contrast: {contrast_id}")
print(f"Using dataset: {contrast['dataset']}")
print(f"Using permutations: {permutations}, top_percent: {top_percent}, pool_num: {pool_num}")
print(f"  Foreground set: {contrast['foreground']}")
print(f"  Background set: {contrast['background']}")
print(f"  MAF stratify: {contrast['maf_stratify']}")
context_df = datasets[contrast["dataset"]]
results_json_file = site_contrast_runner(contrast_id, contrast, context_df, env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)
ld_overlap_pkl_file = calculate_contrast_ld_overlap(contrast_id, contrast, context_df, env_id_list, ld_dict[contrast["dataset"]], la_enrich_dir, bin_width=0.05, min_R2=0.5, random_seed=42)
with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
    results_json = json.load(f)
with open(ld_overlap_pkl_file, 'rb') as f:
    ld_overlap_dict = pickle.load(f)


# In[8]:


from src.enrich import plot_enrichment_and_overlap

stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50'],
]
print(f"Enrichment results for contrast: {contrast_id}")
save_path = os.path.join(la_enrich_dir, f"{contrast_id}.enrichment_and_overlap.pdf")
fig, axs = plot_enrichment_and_overlap(results_json, ld_overlap_dict, env_id_list, env_metadata_df, stratify_label_map)
fig.savefig(save_path, bbox_inches='tight')


# ##### 1.5.3.3 neutral_sites

# In[9]:


from src.enrich import site_contrast_runner
from src.ld import calculate_contrast_ld_overlap
import pickle

contrast_id = "neutral_sites"
permutations = 1000
top_percent = 0.05
pool_num = 40
contrast = contrasts_settings[contrast_id]
print(f"Running enrichment for contrast: {contrast_id}")
print(f"Using dataset: {contrast['dataset']}")
print(f"Using permutations: {permutations}, top_percent: {top_percent}, pool_num: {pool_num}")
print(f"  Foreground set: {contrast['foreground']}")
print(f"  Background set: {contrast['background']}")
print(f"  MAF stratify: {contrast['maf_stratify']}")
context_df = datasets[contrast["dataset"]]
results_json_file = site_contrast_runner(contrast_id, contrast, context_df, env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)
ld_overlap_pkl_file = calculate_contrast_ld_overlap(contrast_id, contrast, context_df, env_id_list, ld_dict[contrast["dataset"]], la_enrich_dir, bin_width=0.05, min_R2=0.5, random_seed=42)
with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
    results_json = json.load(f)
with open(ld_overlap_pkl_file, 'rb') as f:
    ld_overlap_dict = pickle.load(f)


# In[10]:


from src.enrich import plot_enrichment_and_overlap

stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50'],
]
print(f"Enrichment results for contrast: {contrast_id}")
save_path = os.path.join(la_enrich_dir, f"{contrast_id}.enrichment_and_overlap.pdf")
fig, axs = plot_enrichment_and_overlap(results_json, ld_overlap_dict, env_id_list, env_metadata_df, stratify_label_map)
fig.savefig(save_path, bbox_inches='tight')


# ##### 1.5.3.4 accelerated_sites

# In[11]:


from src.enrich import site_contrast_runner
from src.ld import calculate_contrast_ld_overlap
import pickle

contrast_id = "accelerated_sites"
permutations = 1000
top_percent = 0.05
pool_num = 40
contrast = contrasts_settings[contrast_id]
print(f"Running enrichment for contrast: {contrast_id}")
print(f"Using dataset: {contrast['dataset']}")
print(f"Using permutations: {permutations}, top_percent: {top_percent}, pool_num: {pool_num}")
print(f"  Foreground set: {contrast['foreground']}")
print(f"  Background set: {contrast['background']}")
print(f"  MAF stratify: {contrast['maf_stratify']}")
context_df = datasets[contrast["dataset"]]
results_json_file = site_contrast_runner(contrast_id, contrast, context_df, env_id_list, la_enrich_dir, permutations=permutations, top_percent=top_percent, pool_num=pool_num)
ld_overlap_pkl_file = calculate_contrast_ld_overlap(contrast_id, contrast, context_df, env_id_list, ld_dict[contrast["dataset"]], la_enrich_dir, bin_width=0.05, min_R2=0.5, random_seed=42)
with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
    results_json = json.load(f)
with open(ld_overlap_pkl_file, 'rb') as f:
    ld_overlap_dict = pickle.load(f)


# In[12]:


from src.enrich import plot_enrichment_and_overlap

stratify_label_map = [
    ['all', 'All MAF'],
    ['maf_0.05_0.1', '0.05-0.10'],
    ['maf_0.1_0.2', '0.10-0.20'],
    ['maf_0.2_0.3', '0.20-0.30'],
    ['maf_0.3_0.4', '0.30-0.40'],
    ['maf_0.4_0.5', '0.40-0.50'],
]
print(f"Enrichment results for contrast: {contrast_id}")
save_path = os.path.join(la_enrich_dir, f"{contrast_id}.enrichment_and_overlap.pdf")
fig, axs = plot_enrichment_and_overlap(results_json, ld_overlap_dict, env_id_list, env_metadata_df, stratify_label_map)
fig.savefig(save_path, bbox_inches='tight')


# ### 1.6 Flowering-time genes
# 
# The gene-level GEA enrichment analysis is recorded in [notebook 02, Section 4.1](02_environmental_gwas.ipynb#4.1-Gene-level-GEA-enrichment).
# 

# ### 1.6 开花时间基因
# 
# 基因级GEA富集分析记录在[笔记02第4.1节](02_environmental_gwas.ipynb#4.1-Gene-level-GEA-enrichment)。
# 

# ### 1.7 Gene Expression

# In[18]:


from src.enrich import gene_contrast_runner

permutations = 1000
top_percent = 0.05

set_label_map = [
    ['stably_expressed_genes', 'Genes with stable expr.'],
    ['high_variance_genes', 'Genes with high var. expr.'],
    ['no_expression_genes', 'Genes with no detected expr.']
]

results_json_files = {}
for contrast_id in ["no_expression_genes", "high_variance_genes", "stably_expressed_genes"]:
    contrast = contrasts_settings[contrast_id]
    results_json_file = gene_contrast_runner(contrast_id, contrast, datasets[contrast["dataset"]], env_id_list, la_enrich_dir, permutations=1000, top_percent=0.05)
    results_json_files[contrast_id] = results_json_file


# In[19]:


from src.enrich import plot_set_enrichment

set_results_json = {}
for contrast_id, results_json_file in results_json_files.items():
    with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
        results_json = json.load(f)
    set_results_json[contrast_id] = results_json

set_id = "gene_expression"
save_path = os.path.join(la_enrich_dir, f"{set_id}.enrichment.pdf")
fig, axs = plot_set_enrichment(set_results_json, env_id_list, env_metadata_df, set_label_map, significance_threshold=0.05, height_per_set=0.7)
fig.savefig(save_path, bbox_inches='tight')


# ### 1.8 Gene Regulatory Network

# In[20]:


from src.enrich import gene_contrast_runner

permutations = 1000
top_percent = 0.05

set_label_map = [
    ['module_hub_genes', 'Module hubs'],
    ['module_genes', 'Module genes'],
    ['non_module_genes', 'Non-module genes'],
]

results_json_files = {}
for contrast_id, _ in set_label_map:
    contrast = contrasts_settings[contrast_id]
    results_json_file = gene_contrast_runner(contrast_id, contrast, datasets[contrast["dataset"]], env_id_list, la_enrich_dir, permutations=1000, top_percent=0.05)
    results_json_files[contrast_id] = results_json_file


# In[21]:


from src.enrich import plot_set_enrichment

set_results_json = {}
for contrast_id, results_json_file in results_json_files.items():
    with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
        results_json = json.load(f)
    set_results_json[contrast_id] = results_json

set_id = "gene_regulatory_network"
save_path = os.path.join(la_enrich_dir, f"{set_id}.enrichment.pdf")
fig, axs = plot_set_enrichment(set_results_json, env_id_list, env_metadata_df, set_label_map, significance_threshold=0.05, height_per_set=0.7)
fig.savefig(save_path, bbox_inches='tight')


# ### 1.9 Gene copy redundancy

# In[24]:


from src.enrich import gene_contrast_runner

permutations = 1000
top_percent = 0.05

set_label_map = [
    ['single_copy_genes', 'Single-copy genes'],
    ['multi_copy_genes', 'Multi-copy genes'],
    ['orphan_genes', 'Orphan genes'],    
]

results_json_files = {}
for contrast_id, _ in set_label_map:
    contrast = contrasts_settings[contrast_id]
    results_json_file = gene_contrast_runner(contrast_id, contrast, datasets[contrast["dataset"]], env_id_list, la_enrich_dir, permutations=1000, top_percent=0.05)
    results_json_files[contrast_id] = results_json_file


# In[ ]:


from src.enrich import plot_set_enrichment

set_results_json = {}
for contrast_id, results_json_file in results_json_files.items():
    with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
        results_json = json.load(f)
    set_results_json[contrast_id] = results_json

set_id = "gene_copy_redundancy"
save_path = os.path.join(la_enrich_dir, f"{set_id}.enrichment.pdf")
fig, axs = plot_set_enrichment(set_results_json, env_id_list, env_metadata_df, set_label_map, significance_threshold=0.05, height_per_set=0.7)
fig.savefig(save_path, bbox_inches='tight')


# ### 1.10 Recombination rate

# In[26]:


from src.enrich import gene_contrast_runner

permutations = 1000
top_percent = 0.05

set_label_map = [
    ['high_recombination_genes', 'Genes in high recomb. regions'],
    ['med_recombination_genes', 'Genes in med. recomb. regions'],
    ['low_recombination_genes', 'Genes in low recomb. regions'],    
]

results_json_files = {}
for contrast_id, _ in set_label_map:
    contrast = contrasts_settings[contrast_id]
    results_json_file = gene_contrast_runner(contrast_id, contrast, datasets[contrast["dataset"]], env_id_list, la_enrich_dir, permutations=1000, top_percent=0.05)
    results_json_files[contrast_id] = results_json_file


# In[ ]:


from src.enrich import plot_set_enrichment

set_results_json = {}
for contrast_id, results_json_file in results_json_files.items():
    with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
        results_json = json.load(f)
    set_results_json[contrast_id] = results_json

set_id = "gene_recombination_rate"
save_path = os.path.join(la_enrich_dir, f"{set_id}.enrichment.pdf")
fig, axs = plot_set_enrichment(set_results_json, env_id_list, env_metadata_df, set_label_map, significance_threshold=0.05, height_per_set=0.7)
fig.savefig(save_path, bbox_inches='tight')


# ## 2. Enrichment Analysis of consistent selection

# In[2]:


cs_enrich_dir = os.path.join(enrich_dir, "consistent_selection")
os.makedirs(cs_enrich_dir, exist_ok=True)


# Read annotation fractions, π and Tajima’s D from the single 500 bp HDF table produced by notebook 02. Keep its numeric values unchanged in `genome_windows_df`; convert only annotation fractions exceeding 0.75 to foreground membership in a separate analysis copy.
# 

# 从笔记 02 生成的唯一 500 bp HDF 表读取注释比例、π 和 Tajima’s D。在 `genome_windows_df` 中保留原始数值；仅在独立的分析副本中，将大于 0.75 的注释比例转为前景成员标记。
# 

# In[ ]:


genome_windows_contexts_h5 = f'{results_dir}/genome_windows_contexts.b500.h5'
context_threshold = 0.75

genome_windows_df = pd.read_hdf(genome_windows_contexts_h5, key='df')
genome_windows_contexts_df = genome_windows_df.copy()
genome_windows_contexts_df['ID'] = (
    genome_windows_contexts_df['CHROM'] + ':'
    + genome_windows_contexts_df['START'].astype(str) + '-'
    + genome_windows_contexts_df['END'].astype(str)
)
statistics_columns = ['VARIANT_COUNT', 'SNP_COUNT', 'PI', 'TajimaD', 'meanCmPerMb']
context_columns = [
    column for column in genome_windows_contexts_df.columns
    if column not in ['ID', 'CHROM', 'START', 'END'] + statistics_columns
]
for context_id in context_columns:
    genome_windows_contexts_df[context_id] = (
        genome_windows_contexts_df[context_id] > context_threshold
    )
genome_windows_contexts_df = genome_windows_contexts_df[
    ['ID'] + [column for column in genome_windows_contexts_df.columns if column != 'ID']
]
genome_windows_contexts_df['genome'] = True
genome_windows_contexts_df


# In[23]:


import os
import json

contrasts_settings_json_file = os.path.join(os.getcwd(), "config", "contrasts_settings.json")
with open(contrasts_settings_json_file, "r") as f:
    contrasts_settings = json.load(f)


# In[24]:


trait_id_list = ['TajimaD', 'PI']


# In[25]:


contrast_ids = ['genic_2kb_vs_genome',
 'TE_vs_genome',
 'proximal_TE_vs_proximal',
 'distal_TE_vs_distal',
 'other_intergenic_vs_genome',
 'upstream_2kb_vs_genic',
 'five_prime_UTR_vs_genic',
 'CDS_vs_genic',
 'intron_vs_genic',
 'three_prime_UTR_vs_genic',
 'downstream_2kb_vs_genic',
 'upstream_2kb_vs_genome',
 'five_prime_UTR_vs_genome',
 'CDS_vs_genome',
 'intron_vs_genome',
 'three_prime_UTR_vs_genome',
 'downstream_2kb_vs_genome',
 'ACR_vs_genome',
 'proximal_ACR_vs_proximal',
 'distal_ACR_vs_distal',
 'UMR_vs_genome',
 'proximal_UMR_vs_proximal',
 'distal_UMR_vs_distal',
 'proximal_ACR_TE_UMR_vs_proximal',
 'not_proximal_ACR_TE_UMR_vs_proximal',
 'proximal_ACR_not_TE_vs_proximal_TE_not_ACR',
]


# In[30]:


from src.enrich import win_contrast_runner, build_enrich_data_df
import numpy as np

results_json_files = {}

for contrast_id in contrast_ids:

    contrast_json = contrasts_settings[contrast_id]
    contrast_json['target'] = 'foreground'
    contrast_json

    contexts_df = build_enrich_data_df(genome_windows_contexts_df, contrast_json, trait_id_list, other_columns=[])
    quant = 0.05
    for trait_id in trait_id_list:
        lower_quantile = np.quantile(contexts_df[trait_id].dropna(), quant)
        nonnan_mask = contexts_df[trait_id].notna()
        contexts_df.loc[nonnan_mask & (contexts_df[trait_id] <= lower_quantile), f'{trait_id}_lower_quantile'] = True
        contexts_df.loc[nonnan_mask & (contexts_df[trait_id] > lower_quantile), f'{trait_id}_lower_quantile'] = False
        contexts_df.drop(columns=[f'{trait_id}'], inplace=True)
        contexts_df.rename(columns={f'{trait_id}_lower_quantile': f'{trait_id}'}, inplace=True)

    output_dir = f'{cs_enrich_dir}/{contrast_id}'
    os.makedirs(output_dir, exist_ok=True)

    results_json_file = win_contrast_runner(contrast_id, contrast_json, contexts_df, trait_id_list, output_dir, permutations=1000)
    results_json_files[contrast_id] = results_json_file


# In[33]:


from src.enrich import plot_set_enrichment
import gzip

set_results_json = {}
for contrast_id, results_json_file in results_json_files.items():
    with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
        results_json = json.load(f)
    set_results_json[contrast_id] = results_json

set_label_map = [[contrast_id, contrast_id] for contrast_id in contrast_ids]
trait_metadata_df = pd.DataFrame([{"env_id": "TajimaD", "description": "Low Tajima's D"}, 
                                  {"env_id": "PI", "description": "Low Nucleotide Diversity (π)"}])

fig, axs = plot_set_enrichment(set_results_json, trait_id_list, trait_metadata_df, set_label_map, significance_threshold=0.05, height_per_set=0.7)


# In[ ]:





# In[49]:


cs_enrich_dir = os.path.join(enrich_dir, "consistent_selection_site")
os.makedirs(cs_enrich_dir, exist_ok=True)


# In[50]:


contrast_ids = ['genic_2kb_vs_genome',
 'TE_vs_genome',
 'proximal_TE_vs_proximal',
 'distal_TE_vs_distal',
 'other_intergenic_vs_genome',
 'upstream_2kb_vs_genic',
 'five_prime_UTR_vs_genic',
 'CDS_vs_genic',
 'intron_vs_genic',
 'three_prime_UTR_vs_genic',
 'downstream_2kb_vs_genic',
 'upstream_2kb_vs_genome',
 'five_prime_UTR_vs_genome',
 'CDS_vs_genome',
 'intron_vs_genome',
 'three_prime_UTR_vs_genome',
 'downstream_2kb_vs_genome',
 'missense_vs_CDS',
 'synonymous_vs_CDS',
 'putative_LoF_vs_CDS',
 'putative_LoF_MAF_0.01_0.03_vs_putative_LoF_MAF_0.03_0.05',
 'ACR_vs_genome',
 'proximal_ACR_vs_proximal',
 'distal_ACR_vs_distal',
 'UMR_vs_genome',
 'proximal_UMR_vs_proximal',
 'distal_UMR_vs_distal',
 'riboSNitch_vs_gene_body',
 'proximal_ACR_TE_UMR_vs_proximal',
 'not_proximal_ACR_TE_UMR_vs_proximal',
 'proximal_ACR_not_TE_vs_proximal_TE_not_ACR',
 'low_pi',
 'med_pi',
 'high_pi',
 'non_CEs',
 'conserved_sites',
 'neutral_sites',
 'accelerated_sites',
]


# In[51]:


import json
import gzip

datasets = {
    'genome': pd.read_hdf(os.path.join(data_dir, "results", "site_contexts.genome.maf0.05.h5"), key="df"), # All variants with MAF >= 0.05
    'cds': pd.read_hdf(os.path.join(data_dir, "results", "site_contexts.cds.maf0.01.h5"), key="df"), # All variants in coding regions with MAF >= 0.01
}

datasets['genome']['genome'] = True
datasets['cds']['cds'] = True

contexts_settings_json_file = os.path.join(os.getcwd(), "config", "contexts_settings.json")
with open(contexts_settings_json_file, "r") as f:
    contexts_settings = json.load(f)

trait_id_list = ['MAF']    


# In[52]:


from src.enrich import site_jaccard_contrast_runner

results_json_files = {}
for contrast_id in contrast_ids:

    contrast_json = contrasts_settings[contrast_id]
    contrast_json['target'] = 'foreground'
    contrast_json

    contexts_df = build_enrich_data_df(datasets[contrast_json['dataset']], contrast_json, trait_id_list, other_columns=[])
    quant = 0.05

    for trait_id in trait_id_list:
        lower_quantile = np.quantile(contexts_df[trait_id].dropna(), quant)
        nonnan_mask = contexts_df[trait_id].notna()
        contexts_df.loc[nonnan_mask & (contexts_df[trait_id] <= lower_quantile), f'{trait_id}_lower_quantile'] = True
        contexts_df.loc[nonnan_mask & (contexts_df[trait_id] > lower_quantile), f'{trait_id}_lower_quantile'] = False
        contexts_df.drop(columns=[f'{trait_id}'], inplace=True)
        contexts_df.rename(columns={f'{trait_id}_lower_quantile': f'{trait_id}'}, inplace=True)

    output_dir = f'{cs_enrich_dir}/{contrast_id}'
    os.makedirs(output_dir, exist_ok=True)

    results_json_file = site_jaccard_contrast_runner(contrast_id, contrast_json, contexts_df, trait_id_list, output_dir, permutations=1000)
    results_json_files[contrast_id] = results_json_file


# In[54]:


from src.enrich import plot_set_enrichment
import gzip

set_results_json = {}
for contrast_id, results_json_file in results_json_files.items():
    with gzip.open(results_json_file, 'rt', encoding='utf-8') as f:
        results_json = json.load(f)
    set_results_json[contrast_id] = results_json

set_label_map = [[contrast_id, contrast_id] for contrast_id in contrast_ids]
trait_metadata_df = pd.DataFrame([{"env_id": "MAF", "description": "Low Minor Allele Frequency (MAF)"}])

fig, axs = plot_set_enrichment(set_results_json, trait_id_list, trait_metadata_df, set_label_map, significance_threshold=0.05, height_per_set=0.7)

