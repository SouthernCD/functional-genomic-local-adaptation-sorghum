#!/usr/bin/env bash
# Diagnostic only: all QC-passed Chr01 SNPs, using the same fixed LD blocks.
set -euo pipefail

project=/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum
software=/lustre/home/dobby/Program/agent_program/bin
bench="$project/data/results/gwfm_rc_benchmark"
work="$bench/chr1_resource"
mkdir -p "$work/panel" "$work/ld"
test -s "$bench/panel/COMPLETE"
cd "$work"

if [ ! -s panel/panel.bed ]; then
  "$software/plink" --bfile "$bench/panel/panel" --chr 1 \
    --make-bed --out panel/panel > plink.log 2>&1
fi
awk 'NR==1 || $2==1' "$bench/blocks.pos" > blocks.pos
test "$(($(wc -l < blocks.pos)-1))" -gt 0

if [ ! -s ld/ldm.info ]; then
  "$software/python3" "$project/benchmark_scripts/gwfm_resource_guard.py" \
    --log make_ldm.log --resources make_ldm.resources.json -- \
    "$software/gctb" --bfile panel/panel --make-block-ldm \
    --block-info blocks.pos --thread 4 --out ld
  if grep -Eq 'Error:|Error opening' make_ldm.log; then exit 1; fi
  test -s ld/ldm.info
fi
if [ ! -s ld/block1.eigen.bin ]; then
  "$software/python3" "$project/benchmark_scripts/gwfm_resource_guard.py" \
    --log make_eigen.log --resources make_eigen.resources.json -- \
    "$software/gctb" --ldm ld --make-ldm-eigen --thread 4 --out eigen
  if grep -Eq 'Error:|Error opening' make_eigen.log; then exit 1; fi
  test -s ld/block1.eigen.bin
fi
date -u +%FT%TZ > COMPLETE
