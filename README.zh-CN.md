# Functional genomic map of local adaptation in global sorghum

本项目包含论文 **“Functional genomic map of local adaptation in global sorghum”** 的源代码和科研笔记。

[English README](README.md)

## 发表信息

- 预印本：[bioRxiv](https://www.biorxiv.org/content/10.64898/2026.05.17.725773v1.full)。
- 正式发表版本：**[期刊、引用信息及 DOI/链接待补充]**。

## 如何复现

[`environment.yml`](environment.yml) 记录分析使用的软件环境和包版本。创建并激活环境：

```bash
conda env create -f environment.yml
conda activate laenrich
```

启动 Jupyter 时，还需确保 `environment.yml` 注释中列出的外部命令行工具可通过 `PATH` 调用。

本项目的原始输入文件和输出结果统一存放于 **Figshare：[DOI/链接待补充]**。

[`config/data_files.yaml`](config/data_files.yaml) 中说明了分析所需的输入文件。路径相对于项目的 `data/` 目录。从 Figshare 下载输入文件后，按清单中的路径放入 `data/source/`，保留相应子目录。分析输出保存在 `data/results/`。

将 `config/project.example.yaml` 复制为 `config/project.yaml`，把 `remote.project_root` 设置为 notebook kernel 可访问的项目目录，再使用上述环境按编号顺序运行笔记。项目提供两套笔记：`notebooks/` 为英文版，`notebooks.zh-CN/` 为中文版，两套版本的分析代码相同，选择其中一套即可。
