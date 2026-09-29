# PhenotypeSimulator 适配层

`simulate_phenotype_with_kinship` 接收调用方已确定的 causal genotype 和 effect size；
它不选择 SNP，也不抽取 beta。R backend 使用 PhenotypeSimulator 0.3.4 生成 kinship
和 residual 分量，再将三个分量相加。R 包安装在 dobby 的独立环境中；适配代码保存在本项目的
`src/sim/`。

输入：

- `causal_genotype_df`：样本 × causal SNP，剂量在 `[0, 2]`，样本 ID 为索引。
- `causal_effect_df`：`SNP_ID` 和 `beta_raw`；ID 必须与 genotype 列完全一致。SNP 会按 effect 表的顺序显式对齐。
- `kinship_df`：方阵，两轴必须与 genotype 样本顺序完全一致；不会自动调整样本顺序。
- 三个非负且总和为 1 的方差目标，以及整数随机种子。

适配层拒绝缺失 genotype、单态 causal SNP 和非正定 kinship。输出包含带样本名的
phenotype、分量、causal effect、backend 信息和 metadata，以及顺序相同的单列
`.gemma.pheno` 文件。输出检查通过后才移入目标位置；已有同名前缀会报错。

方差目标指**实现后的样本方差**（`ddof=1`）：PhenotypeSimulator 的
`rescaleVariance()` 对每次抽样分别缩放。kinship 分量保留 K 诱导的方向，但这样缩放后，
其无条件分布不再严格是 `N(0, sigma_K² K)`。最终 phenotype 不再归一化；由于有限样本下
各分量协方差不严格为零，其实现方差可能偏离 1。metadata 同时记录分量及 phenotype
的方差。若要求严格的高斯方差分量抽样，需要单独决定是否取消这种缩放。

在 dobby 上，`Rscript` 默认使用独立的 PhenotypeSimulator 环境。Notebook 使用本项目的
`dobby-base` Python kernel，并通过 `from src.sim import simulate_phenotype_with_kinship`
导入；适配层本身不会运行 notebook。现有 443 × 443 IBS 矩阵没有表头，调用前应从
LD-pruned VCF 添加样本 ID，并核对其与 GWAS VCF 的顺序。适配层只接收已选择的 causal
genotype 列，不改动完整 GWAS genotype。

```python
from src.sim import simulate_phenotype_with_kinship

result = simulate_phenotype_with_kinship(
    causal_genotype_df, causal_effect_df, kinship_df,
    output_prefix="data/results/simulation/rep001",
    causal_var=0.10, kinship_var=0.40, residual_var=0.50, seed=12345,
)
print(result["gemma_phenotype_file"], result["metadata"])
```

在 dobby 的 `rock01`，已使用真实 443 样本 GWAS VCF 中的 3 个完整 SNP 和本项目的
443 × 443 IBS 矩阵完成小规模测试。生成的 `.gemma.pheno` 被 GEMMA 0.98.5
的 `-lmm 4` 接受，并得到 3 行 association 结果。这验证了适配层与输出格式，
尚未评估 enrichment power 或全基因组运行时间。
