# Sorghum GWFM RC causal-variant recovery benchmark

This benchmark uses the 443-sample sorghum VCF as its sole genotype background. It reuses `simulate_phenotype_with_kinship()` only through phenotype generation, then runs the existing GEMMA LMM (`-lmm 1`, IBS kinship, `-miss 0.1`, no extra covariates) and GCTB 2.5.5 `--gwfm RC` with a single `SNP Intercept` column. It never invokes `SIMULATION_JOB.run()` or enrichment analysis. MAF/LD are used only to choose truth in the LDMS scenario, never as GCTB annotations.

## Fixed data and output layout

- Source VCF: `data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz` on rock01.
- Shared primary panel: `data/results/gwfm_rc_benchmark/panel_all_variants/`. Its script is `examples/gwfm_rc_prepare_panel.sh`. It keeps biallelic SNPs **and InDels** with MAF ≥ 0.05 and F_MISSING ≤ 0.1, assigns `CHROM:POS:REF:ALT` IDs, renames chromosomes `Chr01`–`Chr10` to `1`–`10`, and checks exact sample and variant order across VCF and PLINK. No GWAS significance filter is used. `TYPE` in `variants.tsv` distinguishes SNP and InDel. `SHA256SUMS`, `panel.properties`, `qc_exclusions.tsv`, `samples.txt`, `variants.tsv`, and `COMPLETE` document the frozen input. The older SNP-only `panel/` is retained only as a technical baseline.
- LD block map and boundary audit: `blocks_all_variants_15kb.pos`, `boundary_audit_all_variants_15kb.tsv`. Blocks are non-overlapping 15,000-bp physical intervals as specified by the user, retaining only intervals with panel variants. The audit examines 25 variants on each side of up to 100 sampled boundaries per chromosome and reuses `src/ld.py`'s `calculate_LD()` for cross-boundary r², supplying ID-matched genotype arrays so variants at the same coordinate remain distinct. The project's existing `ld_window_size=15000` is a sliding pair-search distance, whereas the GCTB map consists of non-overlapping intervals; they share the physical scale but have different semantics. The other `src/ld.py` pipelines retrieve records by position and produce r², which loses the correlation sign needed by GCTB. GCTB therefore constructs the primary signed LD/eigen LD from the ID-preserving PLINK panel.
- Diagnostic pilot: `pilot/` has 3,000 Chr01 SNPs, real 443-sample GEMMA and GCTB outputs, complete truth, and a null-control run.
- Shared primary full LD: `ld_all_variants_15kb/`, containing block and eigen LD, `snp.info`, `ldm.info`, `intercept_only.txt`, LDMS scores, checksums, and `COMPLETE`. `examples/gwfm_rc_full_ld.sh` records the boundary audit as a diagnostic and uses four GCTB threads. A separate memory guard stops GCTB at 400 GiB resident memory.
- Fixed 260-run matrix: `design_matrix.tsv`, made by `examples/gwfm_rc_run_matrix.py` without `--execute`. The same script with `--execute` requires `ld/COMPLETE` and runs one replicate at a time, resuming runs marked `COMPLETE`. A failure writes `FAILED`, preserves logs and outputs, and stops the queue.

## Architectures

Every nonnull setting uses target causal component variance 0.05, 0.15, or 0.30, kinship component 0.50, and residual variance 0.45, 0.35, or 0.20. Each has 10 seeds. At variance 0.15, all five nonnull settings have 30 seeds total. Ten additional null replicates have zero causal variance. Seeds are 10001–10030, with phenotype seed `seed + 2`.

| Setting | Causal variants | Effect design |
| --- | ---: | --- |
| Gaussian | 50, 500, 5,000 | Normal effects scaled by genotype SD |
| Large effects | 500 | Ten designated large SNPs contribute 20% of the sum of within-group genetic component variances, excluding cross-group covariance; remaining 490 contribute 80% |
| LDMS | 500 | Causal SNPs sampled from the joint upper MAF and genotype-derived LD-score quartiles; if that stratum has fewer than 500 SNPs, both cutoffs fall in fixed 0.05-quantile steps to no lower than the median, with the applied quantile saved in `truth.tsv` |
| Null | 0 | Kinship plus residual only |

Each run saves `design.json`, `truth.tsv` (all true variant IDs, SNP/InDel type, positions, REF/ALT, raw and phenotype-scaled effects, effect class), phenotype and component tables, raw GEMMA association output, GCTB summary and result files, `inputs.json`, `metrics.json`, `calibration.tsv`, logs, and `COMPLETE` only after validation. GEMMA `allele1`, `allele0`, and `af` are harmonized to VCF ALT; `N` comes from `n_obs` or `443 - n_miss`. The raw `af` is never replaced by folded MAF.

## Evaluation and current status

`src/gwfm_benchmark.py` reports true SNP retention through GEMMA, LD, and GCTB; PIP ≥ 0.1/0.5/0.9 recall, FDR, and nearest true-SNP distance; local credible-set coverage and size; Gelman–Rubin R diagnostics; and calibration bins. `aggregate` writes run and calibration tables plus a figure with SNP counts by PIP bin. Bins with fewer than 30 SNPs are marked as insufficient evidence.

As of 2026-09-28, the 1,500-SNP map has 996/1000 audited boundaries with at least one cross-boundary pair at r² > 0.5 (980/1000 > 0.8); the 10,000-SNP map has 587/588 > 0.5 (582/588 > 0.8). Median fractions of all 625 cross-boundary SNP pairs with r² > 0.5 are 0.1872 and 0.184, respectively. The original script treated more than 5% of boundaries with *any* such pair as a hard failure. That cutoff was arbitrary and too strict: a block-wise LD approximation does not require every cross-boundary pair to be uncorrelated. [GCTB's documentation](https://gctbhub.cloud.edu.au/software/gctb/) uses *approximately* independent blocks, and the [LD block paper](https://academic.oup.com/bioinformatics/article/32/2/283/1743626) notes that ordinary breakpoints can split strong LD. The hard gate has been removed. The observed correlations warrant a sensitivity analysis of fine-mapping results to alternative boundary placements; they do not alone establish that the full panel is infeasible. No smaller SNP panel is substituted for the primary analysis.

The old full-panel 1,500-SNP build also failed before writing block data: GCTB 2.5.5 reported `double free or corruption` while reading the 5,930,896-SNP BED with 16 threads, at 3.41 GiB peak RSS. This was a software failure, not a 400-GiB memory-limit event. Its incomplete `ld/` directory is retained as a failure record. The 15-kb build uses a separate output directory and four threads after a Chr01 test.

The historical SNP-only 15-kb map covers all 5,930,896 SNPs in 47,630 nonempty blocks (median 106, maximum 1,004 SNPs per block). In its 1,000 sampled boundaries, 997 have at least one cross-boundary pair with r² > 0.5 and the median fraction of such pairs is 0.2064. These are diagnostics of omitted cross-block LD, not a pass/fail threshold. The SNP-only full LD construction was stopped and marked `SUPERSEDED` after the decision to include InDels.

The all-SNP Chr01 resource diagnostic completed: 672,581 SNPs in 448 blocks, 5.0 GiB of LD/eigen files; block LD took 175 seconds at 0.46 GiB peak RSS and eigen LD took 121 seconds at 0.52 GiB peak RSS (4 threads). The 3,000-SNP pilot recovered all 50 truth SNPs in GEMMA, LD, and GCTB, but found no true SNP at PIP ≥ 0.1 and formed no local credible sets. This is a technical pilot, not an estimate of genome-wide power. A phenotype-only 500-SNP large-effect test produced 10 designated large effects and a measured 0.20 within-group causal-variance share. The phenotype-only LDMS test produced 500 truth SNPs; the local pilot needed the documented median/median fallback because the joint upper quartile contained none.

The 3,000-SNP null control completed phenotype generation, GEMMA, and the RC SNP-result stage, with all 3,000 partial PIPs below 0.001. GCTB then spent over 20 minutes after its pairwise-LD message without advancing or writing a completion marker, versus 13 seconds for the Gaussian pilot. That diagnostic process was terminated and its run is marked `FAILED`; the partial null files are not counted as a completed experiment. This is an additional GCTB stability/performance concern to resolve before a full run.

The mixed-variant compatibility pilot (`pilot_all_variants/`) uses 1,694 real Chr01 variants, including 137 InDels. GEMMA, GCTB LD, and RC each retained all 1,694 IDs. Four of 50 simulated causal variants were InDels, and their REF/ALT strings and effect-allele direction matched the GCTB summary; all 50 truth IDs reached GCTB. This verifies tool compatibility only. The full mixed panel was started on rock01 as a background job; its completion and exact variant counts still need verification.

On rock01, from the repository root:

```bash
# Already prepared once; safe to rerun when COMPLETE exists.
bash benchmark_scripts/gwfm_rc_prepare_panel.sh
/lustre/home/dobby/Program/agent_program/bin/python3 benchmark_scripts/gwfm_rc_run_matrix.py

# Mixed-variant compatibility pilot and 15-kb block map.
bash benchmark_scripts/gwfm_rc_indel_pilot.sh
/lustre/home/dobby/Program/agent_program/bin/python3 -m src.gwfm_benchmark bp-block-map \
  --output data/results/gwfm_rc_benchmark/blocks_all_variants_15kb.pos --width-bp 15000
/lustre/home/dobby/Program/agent_program/bin/python3 -m src.gwfm_benchmark audit-boundaries \
  --blocks data/results/gwfm_rc_benchmark/blocks_all_variants_15kb.pos \
  --output data/results/gwfm_rc_benchmark/boundary_audit_all_variants_15kb.tsv

# Full LD construction; boundary statistics are reported, not used as a hard stop.
bash benchmark_scripts/gwfm_rc_full_ld.sh
# Start the replicate queue only after reviewing the LD and pilot sensitivity.
/lustre/home/dobby/Program/agent_program/bin/python3 benchmark_scripts/gwfm_rc_run_matrix.py --execute
```
