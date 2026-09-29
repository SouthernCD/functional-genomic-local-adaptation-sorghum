#!/usr/bin/env bash
# Shared full-panel LD using 15 kb physical blocks, as specified by the user.
set -euo pipefail

project=/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum
software=/lustre/home/dobby/Program/agent_program/bin
bench="$project/data/results/gwfm_rc_benchmark"
panel="$bench/panel_all_variants"
ld="$bench/ld_all_variants_15kb"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
test -s "$panel/COMPLETE"
test -s "$bench/blocks_all_variants_15kb.pos"
test -s "$bench/boundary_audit_all_variants_15kb.tsv"

"$software/python3" - "$bench/boundary_audit_all_variants_15kb.tsv" <<'PY'
import pandas as pd
import sys
x = pd.read_csv(sys.argv[1], sep='\t')
if x.empty:
    raise SystemExit('boundary audit is empty')
print('Boundary diagnostic (not a pass/fail criterion):')
print(f'  sampled boundaries: {len(x)}')
print(f'  fraction with at least one pair r² > 0.5: {(x.max_cross_r2 > 0.5).mean():.4f}')
print(f'  median fraction of 25x25 pairs with r² > 0.5: '
      f'{x.fraction_cross_r2_gt_0_5.median():.4f}')
PY

mkdir -p "$ld"
if [ ! -f "$ld/make_ldm.COMPLETE" ]; then
  if [ -e "$ld/ldm.info" ]; then
    echo "Partial LD output exists without completion marker: $ld" >&2
    exit 1
  fi
  "$software/python3" "$project/benchmark_scripts/gwfm_resource_guard.py" \
    --log "$ld/make_ldm.log" --resources "$ld/make_ldm.resources.json" -- \
    "$software/gctb" --bfile "$panel/panel" --make-block-ldm \
    --block-info "$bench/blocks_all_variants_15kb.pos" --thread 4 --out "$ld"
  if grep -Eq 'Error:|Error opening|double free|corruption' "$ld/make_ldm.log"; then exit 1; fi
  grep -q 'Analysis finished:' "$ld/make_ldm.log"
  test -s "$ld/ldm.info"
  test -s "$ld/snp.info"
  test -s "$ld/rsq0.5.pwld"
  date -u +%FT%TZ > "$ld/make_ldm.COMPLETE"
fi
if [ ! -f "$ld/make_eigen.COMPLETE" ]; then
  if [ -e "$ld/block1.eigen.bin" ]; then
    echo "Partial eigen LD output exists without completion marker: $ld" >&2
    exit 1
  fi
  "$software/python3" "$project/benchmark_scripts/gwfm_resource_guard.py" \
    --log "$ld/make_eigen.log" --resources "$ld/make_eigen.resources.json" -- \
    "$software/gctb" --ldm "$ld" --make-ldm-eigen \
    --thread 4 --out "$ld/eigen"
  if grep -Eq 'Error:|Error opening|double free|corruption' "$ld/make_eigen.log"; then exit 1; fi
  grep -q 'Analysis finished:' "$ld/make_eigen.log"
  test -s "$ld/block1.eigen.bin"
  date -u +%FT%TZ > "$ld/make_eigen.COMPLETE"
fi
awk 'BEGIN {print "SNP Intercept"} {print $2, 1}' "$panel/panel.bim" \
  > "$ld/intercept_only.txt"

"$software/python3" - "$panel/variants.tsv" "$ld/snp.info" \
  "$ld/ldm.info" "$ld/intercept_only.txt" <<'PY'
import sys
from pathlib import Path
from itertools import zip_longest
variants, info, blocks, annotation = map(Path, sys.argv[1:])
with variants.open() as a, info.open() as b, annotation.open() as c:
    next(a); next(b); next(c)
    count = 0
    for panel_line, ld_line, annot_line in zip_longest(a, b, c):
        if not all((panel_line, ld_line, annot_line)):
            raise SystemExit('panel, LD, and annotation SNP counts differ')
        snp = panel_line.split('\t', 1)[0]
        if snp != ld_line.split()[1] or annot_line.split() != [snp, '1']:
            raise SystemExit(f'SNP or annotation mismatch at record {count + 1}')
        count += 1
    if count == 0:
        raise SystemExit('empty variant panel')
print(f'validated {count} variants across panel, LD, and intercept annotation')
if not blocks.is_file():
    raise SystemExit('LD block index absent')
PY
"$software/python3" -m src.gwfm_benchmark ld-scores \
  --pairs "$ld/rsq0.5.pwld" --output "$ld/ld_scores.tsv"
sha256sum "$bench/blocks_all_variants_15kb.pos" "$ld/ldm.info" "$ld/snp.info" \
  "$ld/intercept_only.txt" > "$ld/SHA256SUMS"
date -u +%FT%TZ > "$ld/COMPLETE"
