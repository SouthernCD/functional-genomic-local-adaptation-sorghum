# bigsnpr LD block wrapper

`run_ldsplit()` takes a PLINK 1 bed/bim/fam prefix and returns a pandas block
table. It runs `bigsnpr::snp_cor()` and `bigsnpr::snp_ldsplit()` in the separate
R environment on dobby. VCF conversion, genotype QC, and choosing final block
parameters remain outside this wrapper.

```python
from src.bigsnpr import run_ldsplit

blocks = run_ldsplit(
    plink_prefix="/path/to/sorghum",
    chromosome=1,
    size_kb=100,
    thr_r2=0.3,
    max_r2=0.5,
    ncores=16,
    output_dir="/path/to/ldsplit_chr1",
)
print(blocks.head())
```

Use `chromosome=None` to process all chromosomes, one at a time in R. The
default output directory is `<plink_prefix>.ldsplit_chr<chromosome>` (or
`<plink_prefix>.ldsplit_chrall`); it must be empty. Use a
different directory for each parameter set. The wrapper keeps the imported
`genotypes.rds/.bk`, a `corr.rds` sparse correlation matrix per chromosome
(unless `save_corr=False`), all candidate `solutions.tsv/.rds`, the chosen
`selected_solution.tsv`, `blocks.tsv`, `parameters.json`, `stdout.log`, and
`stderr.log`.

`size_kb` (or the `size` alias) is passed to `snp_cor()` together with physical
positions, so it is measured in kilobases. `min_size` and `max_size` are
variant counts. `pos_scaled=None` uses the bigsnpr zero vector. A scalar
`pos_scaled=10_000_000` divides each position in bp by 10 Mb; a sequence gives
the exact R vector in the selected BIM order. The chosen solution minimizes
`cost2 * sqrt(5 + cost)`, following the package example; inspect all solutions
before treating the resulting boundaries as final.

`max_r2` is a hard bound on squared correlations across blocks. A long chain
of high-LD pairs can make it mathematically impossible to honor both
`max_r2` and `max_size`. The wrapper then writes `feasibility.tsv` with the
largest forced block and raises a specific error before running the split.
`max_cost=None` uses bigsnpr's default (`n_SNP / 200`); pass `max_cost=float("inf")`
to disable that additional cost limit during exploration. This can be slow
and does not guarantee a solution.

If a split fails after `corr.rds` was saved, a later call can reuse it without
re-importing the PLINK data or recomputing correlations. Keep `plink_prefix`,
`chromosome`, `size_kb`, and `thr_r2` the same, and use a new output directory:

```python
blocks = run_ldsplit(
    plink_prefix="/path/to/sorghum",
    chromosome=1,
    size_kb=100,
    thr_r2=0.3,
    max_size=2000,
    max_r2=1.0,
    max_cost=float("inf"),
    corr_source="/path/to/previous_run/chr_1/corr.rds",
    output_dir="/path/to/resplit_chr1",
)
```

The block table has `Block`, `Chrom`, `StartBP`, `EndBP`, and `NSNP` columns.
The start and end positions are the first and last observed SNP in each block.
The Python and R layers both verify SNP order, IDs, chromosomes, and positions
against the BIM file. Missing or mismatched outputs raise an exception.

On dobby, the default Rscript is
`/lustre/home/dobby/Program/agent_program/micromamba-root/envs/bigsnpr/bin/Rscript`.
Pass `rscript=` to use another installation.

## Reproduce the QC-panel full-genome block map on rock01

`ldsplit.py` provides a reusable full-genome function; its `__main__` block
contains the successful QC-panel example. Running that example processes
chromosomes 1–10 with `size_kb=100`, `thr_r2=0.3`,
`min_size=10`, `max_size=2000`, `max_r2=1`, `max_K=600`, `max_cost=Inf`, and
`pos_scaled=None`. It runs two chromosomes concurrently with eight R cores
each and a four-hour limit per chromosome. It then selects
`ceil(1.1 * ceil(chromosome_SNPs / 2000))` blocks from each chromosome's
saved candidate solutions.

```bash
python3 src/bigsnpr/ldsplit.py
```

The example's PLINK prefix is
`data/results/landraces_qc_miss0.5_maf0.05_het0.05`. The source calculations
are retained in `data/results/bigsnpr_full_genome_100kb_v1/`, and the
GCTB-compatible `blocks.pos`, `blocks.tsv`, `summary.tsv`, and completion
metadata are written to `data/results/bigsnpr_full_genome_100kb_k1p1_v1/`.
The source's `standalone_results.tsv` tracks chromosome progress. Completed
chromosome calculations are verified and reused; partial ones are archived
before recomputation. To recompute from scratch, change the example's
`source_root` and `output_root` arguments to fresh directories.
