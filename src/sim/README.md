# PhenotypeSimulator adapter

`simulate_phenotype_with_kinship` takes causal genotypes and effect sizes already
chosen by the caller. It does not select SNPs or draw beta. The R backend uses
PhenotypeSimulator 0.3.4 for the kinship and residual draws, then combines the
three components. The installed R package lives in the isolated dobby environment;
these adapter files live in this project's `src/sim/`.

Inputs:

- `causal_genotype_df`: samples × causal SNPs, additive dosage in `[0, 2]`, sample IDs as index.
- `causal_effect_df`: `SNP_ID`, `beta_raw`; IDs must match genotype columns. SNPs are explicitly reordered to effect-table order.
- `kinship_df`: square matrix, with both axes exactly matching genotype sample order. Sample order is never changed.
- Three nonnegative variance targets summing to 1, plus an integer seed.

The adapter rejects missing calls, monomorphic causal SNPs, and nonpositive-definite
kinship. It writes named phenotype, component, causal-effect, backend and metadata
files, plus a single-column `.gemma.pheno` file in the same sample order. Outputs
are checked before they are moved into place, and an existing output prefix is
rejected.

The targets are **realized sample variances** (`ddof=1`): PhenotypeSimulator's
`rescaleVariance()` scales each draw to the chosen value. The resulting kinship
component retains the direction induced by K, but after this per-replicate scaling
its unconditional distribution is not exactly `N(0, sigma_K² K)`. The phenotype
is not normalized again; its realized variance can differ from 1 because the
components have nonzero finite-sample covariances. The metadata reports both
component and phenotype variances. If exact Gaussian variance-component sampling
is required, this scaling rule needs a separate design decision.

On dobby, `Rscript` defaults to the dedicated PhenotypeSimulator environment.
For notebook imports, activate the project's `dobby-base` Python kernel and use
`from src.sim import simulate_phenotype_with_kinship`. No notebook is run by this
adapter. The actual 443 × 443 IBS matrix is headerless, so attach sample IDs from
the LD-pruned VCF before calling it; verify that its sample order matches the GWAS
VCF. The adapter accepts only the selected causal genotype columns, leaving the
full GWAS genotype unchanged.

```python
from src.sim import simulate_phenotype_with_kinship

result = simulate_phenotype_with_kinship(
    causal_genotype_df, causal_effect_df, kinship_df,
    output_prefix="data/results/simulation/rep001",
    causal_var=0.10, kinship_var=0.40, residual_var=0.50, seed=12345,
)
print(result["gemma_phenotype_file"], result["metadata"])
```

On dobby `rock01`, a smoke run using three complete SNPs from the actual
443-sample GWAS VCF and the project's 443 × 443 IBS matrix passed. The resulting
`.gemma.pheno` was accepted by GEMMA 0.98.5 with `-lmm 4` and produced three
association rows. This verifies the adapter and output contract on a small
real-data subset; it does not assess enrichment power or full-genome runtime.
