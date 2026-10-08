# Functional genomic map of local adaptation in global sorghum

This repository accompanies the manuscript **“Functional genomic map of local adaptation in global sorghum”**. It contains seven ordered notebooks and the reusable analysis code needed to reproduce the paper-level analyses from curated, analysis-ready genotype, environmental, expression, functional-genomic, and phenotype inputs.

[中文说明](README.zh-CN.md)

## Analysis notebooks

| Notebook | Purpose | Main output |
|---|---|---|
| `01_define_and_build_genomic_contexts.ipynb` | Accession audits and site- and gene-level genomic contexts | Site and gene context tables |
| `01_environment_and_population_structure.ipynb` | Environmental variables and population structure | Figure 1 |
| `02_environmental_gwas.ipynb` | envGWAS, diagnostics, MAGMA, and focal loci | Figure 2 |
| `03_functional_enrichment.ipynb` | Site- and gene-level enrichment and sensitivity analyses | Figure 3 |
| `04_spatial_distribution_around_genes.ipynb` | Strand-aware profiles around TSS and TTS | Figure 4A data |
| `05_variance_partitioning.ipynb` | Context-specific GRMs and LDAK REML | Figure 4 |
| `06_genomic_prediction.ipynb` | Genomic prediction in the GxE panel | Figure 5 |

Notebook narratives are written in paired English and Chinese Markdown cells. Code, comments, identifiers, tables, filenames, and figure labels are English only.

## Preparing the project directory

If you intend to reproduce this study, clone the repository directly as the project directory and record its absolute path in `config/project.yaml`:

```bash
git clone <repository-url> /absolute/path/to/functional-genomic-local-adaptation-sorghum
cd /absolute/path/to/functional-genomic-local-adaptation-sorghum
cp config/project.example.yaml config/project.yaml
```

Set `remote.project_root` to the project path visible to the notebook kernel. Scientific data are stored inside the project directory but excluded from Git:

```text
functional-genomic-local-adaptation-sorghum/
├── config/
├── notebooks/
├── src/
└── data/
    ├── source/
    └── results/
```

- `source/` contains data supplied to the project from external sources. In this repository, source data include both primary public datasets and documented, analysis-ready derivatives prepared before the analyses presented here. Files in this directory are treated as inputs and are not overwritten by the notebooks.
- `results/` contains every file produced or reused by the analyses, including intermediate tables, fitted models, association statistics, genomic-context tables, external-program outputs, figure data, figures, and supplementary tables. Subdirectories may be used to organize results by analysis, but they do not represent separate data classes.

[`config/data_files.yaml`](config/data_files.yaml) describes downloadable source files and their expected locations relative to `<project_root>/data`. Files released with this study use the Figshare collection URL recorded in that file; the placeholder will be replaced when the collection is published. Source files are placed directly in `data/source`; no content-specific subdirectories are used. For example:

```text
<project_root>/data/source/landraces.vcf.gz
<project_root>/data/source/RNAseqPanel.TPMs.csv.gz
<project_root>/data/source/Groover_2026_nbt.PsbS.csv.gz
```

The `config/` directory contains only technical settings: project paths, file locations, execution environments, remote-kernel settings, and resource intent. Scientific thresholds and analysis choices are stated and set in the notebook where they are used.

## Reproducibility entry point

The source collection starts from processed public-data derivatives that are sufficient for the central analyses of the paper:

- quality-controlled full and LD-pruned VCFs for 443 landraces, 318 RNA-seq accessions, and 176 GxE accessions;
- accession, environmental, and analysis-ready GxE phenotype metadata;
- a TPM matrix, prepared GCN edges, ClusterONE modules, SNPfold results, and Poaceae orthogroups;
- the reference assembly, gene and repeat annotations, ACRs, UMRs, phyloP scores, a recombination map, and flowering-time genes;
- barcode-level promoter-mutagenesis measurements for PsbS, Raf1, and SBPase.

Raw RNA-seq reads, environmental rasters, Striga occurrence modelling, primary variant calling, raw field observations, and reruns of OrthoFinder, ClusterONE, or SNPfold are outside the repository boundary. Their methods and provenance are described in the manuscript and source-file descriptions. Striga HS, Striga HS (Crop), expression-network summaries, SNPfold results, orthogroups, and derived GxE traits are treated as analysis-ready source variables.

## Running the notebooks

Create the public environment if needed:

```bash
conda env create -f environment.yml
conda activate laenrich
```

Open the notebooks locally and connect them to a kernel running where the data reside. Run the notebooks in numerical order. Each computational step defines its expected output under `data/results` and checks whether that output already exists before starting the calculation. If the expected file is present, the step reuses it and is skipped; if it is absent, the step is run and its output is saved for subsequent notebook sessions. This makes it possible to resume the analysis without maintaining a separate checkpoint data class or selecting a global run mode.

To repeat a particular step, first move or remove only the output files identified by that step, then rerun the corresponding cell. Existing outputs should not be overwritten silently. Because a file's presence alone does not establish that it was generated with the current inputs and settings, each notebook should display the paths and key metadata of reused results before continuing.

Notebook 01 retains its upstream preparation cells as described below. The intended manual handoff for long-running envGWAS, MAGMA, LD, and LDAK commands still needs reconciliation with the existing later notebooks. The notebooks show their inputs, expected outputs, and continuation point and do not submit scheduler jobs automatically. Their reusable outputs are also stored under `data/results`. Notebook 01 writes `site_contexts.genome.maf0.05.h5`, `site_contexts.cds.maf0.01.h5`, and `gene_contexts.h5` there for subsequent notebooks.

Reviewer-requested conditional-background, LD-aware, and MAF/LoF sensitivity analyses are recorded in notebook 03. The global-selection/derived-allele-frequency section remains a method placeholder and does not report unconfirmed results.

## Genomic context construction

Notebook 01 contains VCF merging, sample extraction, genome-wide and CDS quality control, and construction of site- and gene-level contexts. Computational cells reuse existing results when inputs and parameters are unchanged. Context tables use HDF key `df`; variant-statistics tables are organized by chromosome.

The final summaries are saved as `data/results/contexts.{genome,cds,gene}.summary.csv.gz`. Each table reports the count and fraction of sites or genes in a context, with the corresponding dataset total as its denominator. Contexts may overlap.

## Citation and license

Code is released under the MIT License. Source datasets retain their original licenses and citation requirements; consult their original publications and the release metadata before redistribution.
