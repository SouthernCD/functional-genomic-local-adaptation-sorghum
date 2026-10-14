# Functional genomic map of local adaptation in global sorghum

This repository contains the source code and research notebooks for the paper **“Functional genomic map of local adaptation in global sorghum”**.

[中文说明](README.zh-CN.md)

## Publication

- Preprint: [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.05.17.725773v1.full).
- Published article: **[Journal, citation and DOI/link to be added]**.

## Reproducing the analyses

[`environment.yml`](environment.yml) specifies the software environment and package versions used for the analyses. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate laenrich
```

The external command-line tools listed in the comments of `environment.yml` must also be available on `PATH` when starting Jupyter.

The source input files and this project's output results are archived together on **Figshare: [DOI/link to be added]**.

[`config/data_files.yaml`](config/data_files.yaml) describes the input files needed for the analyses. Paths are relative to the project's `data/` directory. Download the inputs from Figshare and place them at the listed paths under `data/source/`, preserving any subdirectories. Analysis outputs are stored under `data/results/`.

Copy `config/project.example.yaml` to `config/project.yaml`, set `remote.project_root` to the project directory visible to the notebook kernel, and run either the English notebooks in `notebooks/` or the Chinese notebooks in `notebooks.zh-CN/` in numerical order using the environment above. Both versions contain the same analysis code.
