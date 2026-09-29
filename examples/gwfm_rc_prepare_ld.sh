#!/usr/bin/env bash
# Create mixed-variant 15 kb blocks, audit boundaries, then build GCTB LD.
set -euo pipefail
project=/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum
bench="$project/data/results/gwfm_rc_benchmark"
python=/lustre/home/dobby/Program/agent_program/bin/python3
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
test -s "$bench/panel_all_variants/COMPLETE"
if [ ! -s "$bench/blocks_all_variants_15kb.pos" ]; then
  "$python" -m src.gwfm_benchmark bp-block-map \
    --panel "$bench/panel_all_variants" \
    --output "$bench/blocks_all_variants_15kb.pos" --width-bp 15000
fi
if [ ! -s "$bench/boundary_audit_all_variants_15kb.tsv" ]; then
  "$python" -m src.gwfm_benchmark audit-boundaries \
    --panel "$bench/panel_all_variants" \
    --blocks "$bench/blocks_all_variants_15kb.pos" \
    --output "$bench/boundary_audit_all_variants_15kb.tsv"
fi
bash "$project/benchmark_scripts/gwfm_rc_full_ld.sh"
