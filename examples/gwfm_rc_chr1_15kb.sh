#!/usr/bin/env bash
# Test the user's 15 kb physical LD blocks on all QC-passed Chr01 SNPs.
set -euo pipefail

project=/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum
software=/lustre/home/dobby/Program/agent_program/bin
bench="$project/data/results/gwfm_rc_benchmark"
work="$bench/chr1_resource"
ld="$work/ld_15kb"
test -s "$bench/blocks_15kb.pos"
test -s "$work/panel/panel.bed"
mkdir -p "$ld"
cd "$work"
awk 'NR==1 || $2==1' "$bench/blocks_15kb.pos" > blocks_15kb.pos
test "$(($(wc -l < blocks_15kb.pos)-1))" -gt 0

if [ ! -s "$ld/ldm.info" ]; then
  "$software/python3" "$project/benchmark_scripts/gwfm_resource_guard.py" \
    --log "$work/make_ldm_15kb.log" \
    --resources "$work/make_ldm_15kb.resources.json" -- \
    "$software/gctb" --bfile "$work/panel/panel" \
    --make-block-ldm --block-info "$work/blocks_15kb.pos" \
    --thread 4 --out "$ld"
  if grep -Eq 'Error:|Error opening|double free|corruption' "$work/make_ldm_15kb.log"; then
    exit 1
  fi
  test -s "$ld/ldm.info"
fi
if [ ! -s "$ld/block1.eigen.bin" ]; then
  "$software/python3" "$project/benchmark_scripts/gwfm_resource_guard.py" \
    --log "$work/make_eigen_15kb.log" \
    --resources "$work/make_eigen_15kb.resources.json" -- \
    "$software/gctb" --ldm "$ld" --make-ldm-eigen \
    --thread 4 --out "$work/eigen_15kb"
  if grep -Eq 'Error:|Error opening|double free|corruption' "$work/make_eigen_15kb.log"; then
    exit 1
  fi
  test -s "$ld/block1.eigen.bin"
fi
date -u +%FT%TZ > "$work/COMPLETE_15kb"
