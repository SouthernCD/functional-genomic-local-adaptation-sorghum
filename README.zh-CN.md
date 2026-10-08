# 全球高粱局部适应的功能基因组图谱

本仓库为论文 **“Functional genomic map of local adaptation in global sorghum”** 提供配套代码。仓库包含七本依次运行的 notebook，以及从经过整理、可直接分析的基因型、环境、表达、功能基因组和表型输入复现论文核心分析所需的共用代码。

[English README](README.md)

## 分析 notebooks

| Notebook | 内容 | 主要输出 |
|---|---|---|
| `01_define_and_build_genomic_contexts.ipynb` | 样本审计及 site/gene genomic contexts | site 和 gene context 表格 |
| `01_environment_and_population_structure.ipynb` | 环境变量和群体结构 | Figure 1 |
| `02_environmental_gwas.ipynb` | envGWAS、诊断、MAGMA 和重点位点 | Figure 2 |
| `03_functional_enrichment.ipynb` | 位点及基因富集和敏感性分析 | Figure 3 |
| `04_spatial_distribution_around_genes.ipynb` | TSS/TTS 周围的 strand-aware profiles | Figure 4A 数据 |
| `05_variance_partitioning.ipynb` | context-specific GRM 和 LDAK REML | Figure 4 |
| `06_genomic_prediction.ipynb` | GxE panel 中的基因组预测 | Figure 5 |

Notebook 说明文字采用相邻的英文和中文 Markdown cells。代码、注释、标识符、表格、文件名和图中文字仅使用英文。

## 准备项目目录

如果计划复现本研究，请直接将仓库克隆为项目目录，并在 `config/project.yaml` 中记录其绝对路径：

```bash
git clone <repository-url> /absolute/path/to/functional-genomic-local-adaptation-sorghum
cd /absolute/path/to/functional-genomic-local-adaptation-sorghum
cp config/project.example.yaml config/project.yaml
```

将 `remote.project_root` 设置为 notebook kernel 所看到的项目路径。科学数据保存在项目目录内，但由 Git 忽略：

```text
functional-genomic-local-adaptation-sorghum/
├── config/
├── notebooks/
├── src/
└── data/
    ├── source/
    └── results/
```

- `source/` 保存由项目外部来源提供的数据。本项目的 source 既包括公开的原始数据，也包括在本文核心分析开始前已经整理好并记录来源的 analysis-ready derivatives。Notebook 将这些文件视为只读输入，不会覆盖它们。
- `results/` 保存分析产生或重用的全部文件，包括中间表格、拟合模型、association statistics、genomic-context 表格、外部程序输出、作图数据、图件和补充表。可以按分析内容设置子目录，但这些子目录不再代表不同数据类别。

[`config/data_files.yaml`](config/data_files.yaml) 说明可下载的 source 文件及其相对于 `<project_root>/data` 的预期位置。随研究发布的文件使用该文件中记录的 Figshare collection URL；collection 发布后将替换当前占位符。所有 source 文件直接放在 `data/source`，不再设置内容分类子目录。例如：

```text
<project_root>/data/source/landraces.vcf.gz
<project_root>/data/source/RNAseqPanel.TPMs.csv.gz
<project_root>/data/source/Groover_2026_nbt.PsbS.csv.gz
```

`config/` 只包含项目路径、文件位置、程序环境、远程 kernel 设置和资源意图等技术配置。科学阈值及分析选择均在实际使用它们的 notebook 中说明和设置。

## 复现起点

source collection 从足以支持论文核心分析的公共数据加工版本开始：

- 443 个 landraces、318 个 RNA-seq accessions 和 176 个 GxE accessions 的质控后完整及 LD-pruned VCF；
- accession、环境变量及分析就绪的 GxE phenotype metadata；
- TPM matrix、准备好的 GCN edges、ClusterONE modules、SNPfold results 和 Poaceae orthogroups；
- reference assembly、gene/repeat annotations、ACRs、UMRs、phyloP、recombination map 和 flowering-time genes；
- PsbS、Raf1 和 SBPase 的 barcode-level promoter-mutagenesis measurements。

原始 RNA-seq reads、环境 raster、Striga occurrence modelling、初始 variant calling、原始田间记录，以及 OrthoFinder、ClusterONE 或 SNPfold 的重新运行均不属于本仓库范围。其方法和 provenance 由论文及 source 文件说明记录。Striga HS、Striga HS (Crop)、expression-network summaries、SNPfold results、orthogroups 和 derived GxE traits 均作为 analysis-ready source variables 使用。

## 运行 notebooks

如有需要，可创建公共环境：

```bash
conda env create -f environment.yml
conda activate laenrich
```

在本地打开 notebook，并连接到科学数据所在位置运行的 kernel。按编号顺序运行。每个计算步骤都在 `data/results` 下定义自己的预期输出，并在计算开始前检查该输出是否已经存在。如果预期文件已存在，则直接重用并跳过该步计算；如果不存在，则运行该步骤并保存结果，供后续 notebook session 使用。因此，项目不再需要单独的 checkpoint 数据类别，也不再使用全局运行模式。

如果需要重新运行某一步，应先移动或删除该步骤明确列出的输出文件，再重新运行相应 cell。已有结果不应被默默覆盖。由于文件存在本身不能证明它由当前输入和参数产生，notebook 在继续之前应显示被重用结果的路径及关键 metadata。

Notebook 01 保留上游准备 cells，详见下方说明。耗时 envGWAS、MAGMA、LD 和 LDAK 的预期人工交接方式仍需与后续 notebook 的现有实现核对。Notebook 会给出其输入、预期输出及继续位置，但不会自动提交 scheduler jobs。这些可重用输出同样保存在 `data/results` 下。Notebook 01 将 `site_contexts.genome.maf0.05.h5`、`site_contexts.cds.maf0.01.h5` 及 `gene_contexts.h5` 写入该目录，供后续 notebooks 使用。

Notebook 03 记录审稿人要求的 conditional-background、LD-aware 和 MAF/LoF sensitivity analyses。global-selection/derived-allele-frequency 部分仍仅为方法占位，不报告未经确认的结果。

## 基因组 context 构建

Notebook 01 包含 VCF 合并、样本提取、全基因组及 CDS 质控，以及位点和基因级 context 构建。输入和参数未变时，计算 cells 复用已有结果。Context 表采用 HDF key `df`；变异统计表按染色体组织。

最终汇总保存为 `data/results/contexts.{genome,cds,gene}.summary.csv.gz`。每份表报告 context 中位点或基因的数量和比例，分母为对应数据集总数。Contexts 可以重叠。

## 引用与许可

代码使用 MIT License。各 source dataset 保留原始许可和引用要求；重新分发前请查阅相应原始论文及发布 metadata。
