#!/usr/bin/env bash
# Build GCTB block LD matrices from the verified bigsnpr QC-panel block map.
set -euo pipefail

project=/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum
root="$project/data/results/bigsnpr_full_genome_100kb_k1p1_v1"
panel="$project/data/results/landraces_qc_miss0.5_maf0.05_het0.05"
run="${GCTB_RUN_DIR:-$root/gctb_full_ldm}"
gctb=/lustre/home/dobby/Program/agent_program/bin/gctb
threads="${GCTB_THREADS:-16}"
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$run/ld"
if [[ -e "$run/COMPLETE.json" ]]; then
  echo "Full GCTB LD result already completed: $run" >&2
  exit 0
fi
if [[ -e "$run/ld/ldm.info" ]]; then
  echo "Partial GCTB LD output already exists; inspect before restarting: $run" >&2
  exit 1
fi
for suffix in bed bim fam; do
  test -s "$panel.$suffix"
done
test -s "$root/blocks.pos"
test -s "$root/COMPLETE.json"

python3 - "$run" "$panel" "$root/blocks.pos" "$gctb" "$threads" <<'PY'
from pathlib import Path
import json,sys
from datetime import datetime,timezone
run,panel,blocks,gctb,threads=sys.argv[1:]
(Path(run)/'run_config.json').write_text(json.dumps({
    'started_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    'bfile': panel, 'block_info': blocks, 'gctb': gctb,
    'threads': int(threads), 'operation': '--make-block-ldm',
},indent=2)+'\n')
PY

on_exit() {
  code=$?
  if (( code != 0 )); then
    printf 'exit_code=%s\nfinished_utc=%s\n' "$code" "$(date -u +%FT%TZ)" \
      > "$run/FAILED.txt"
  fi
}
trap on_exit EXIT

echo "$(date -u +%FT%TZ) Running GCTB on 3560 blocks with $threads threads"
"$gctb" --bfile "$panel" --make-block-ldm \
  --block-info "$root/blocks.pos" --thread "$threads" \
  --out "$run/ld" > "$run/gctb.stdout.log" 2> "$run/gctb.stderr.log"

python3 - "$run" "$root/blocks.tsv" <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import csv,json,sys
run=Path(sys.argv[1]);blocks_path=Path(sys.argv[2]);ld=run/'ld'
with blocks_path.open(newline='') as f:
    expected=list(csv.DictReader(f,delimiter='\t'))
with (ld/'ldm.info').open() as f:
    next(f)
    observed=[line.split() for line in f if line.strip()]
if len(expected)!=3560 or len(observed)!=3560:
    raise ValueError(f'expected 3560 LD blocks, found {len(observed)}')
for index,(source,result) in enumerate(zip(expected,observed),1):
    if int(result[0])!=index or int(result[1])!=int(source['Chrom']) \
       or int(result[-1])!=int(source['NSNP']):
        raise ValueError(f'GCTB ldm.info differs from block {index}: {result}')
    binary=ld/f'block{index}.ldm.bin'
    if not binary.is_file() or binary.stat().st_size==0:
        raise ValueError(f'missing or empty LD matrix: {binary}')
with (ld/'snp.info').open() as f:
    n_snp=sum(1 for _ in f)-1
if n_snp!=6452578:
    raise ValueError(f'expected 6452578 SNPs in GCTB output, found {n_snp}')
(run/'COMPLETE.json').write_text(json.dumps({
    'completed_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
    'n_blocks':len(observed),'n_snps':n_snp,
    'bfile':json.loads((run/'run_config.json').read_text())['bfile'],
    'ldm':str(ld),
},indent=2)+'\n')
print(f'Validated {len(observed)} GCTB LD matrices and {n_snp} SNPs')
PY

echo "$(date -u +%FT%TZ) COMPLETE: $run"
