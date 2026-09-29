#!/usr/bin/env bash
# Full Striga GEMMA -> GCTB GWFM RC run on rock01.
# Run from any directory: bash examples/gwfm_striga_full.sh
# The VCF-derived eigen LD cache is retained next to the VCF, keyed by the
# supplied block file's SHA-256 digest.
set -euo pipefail

project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
software=/lustre/home/dobby/Program/agent_program/bin
export PATH="$software:$PATH"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"

exec "$software/python3" -m src.gwfm \
  --gwas "$project/data/results/gwas/striga/striga.gemma/output/gemma_output.assoc.txt" \
  --vcf "$project/data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz" \
  --block-file "$project/data/results/bigsnpr_full_genome_100kb_k1p1_v1/blocks.pos" \
  --threads 4 \
  --output "$project/data/results/gwas/striga/gwfm_rc_full"
