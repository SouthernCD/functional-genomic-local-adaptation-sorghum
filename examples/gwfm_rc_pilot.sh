#!/usr/bin/env bash
# Real-genotype end-to-end pilot. Run on rock01 after panel/COMPLETE exists.
set -euo pipefail

project=/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum
root=/lustre/home/dobby/Program/agent_program
bench="$project/data/results/gwfm_rc_benchmark"
full="$bench/panel"
pilot="$bench/pilot"
bcf="$root/bin/bcftools"
plink="$root/bin/plink"
gctb="$root/bin/gctb"
python="$root/bin/python3"
export PATH="$root/bin:$PATH"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"

test -f "$full/COMPLETE"
mkdir -p "$pilot/panel" "$pilot/ld"
cd "$pilot"
if [ ! -f panel/COMPLETE ]; then
  if [ -e panel/panel.vcf.gz ]; then
    echo 'Incomplete pilot panel exists; inspect before retrying' >&2
    exit 1
  fi
  end_bp=$(awk '$1==1 { n++; if (n==3000) { print $4; exit } }' "$full/panel.bim")
  test -n "$end_bp"
  "$bcf" view -r "1:1-$end_bp" -Oz -o panel/panel.vcf.gz "$full/panel.vcf.gz"
  "$bcf" index -t panel/panel.vcf.gz
  "$bcf" query -l panel/panel.vcf.gz > panel/samples.txt
  "$bcf" query -f '%ID\t%CHROM\t%POS\t%REF\t%ALT\t%MAF\t%F_MISSING\n' panel/panel.vcf.gz \
    | { printf 'SNP\tCHROM\tPOS\tREF\tALT\tMAF\tF_MISSING\n'; cat; } \
    > panel/variants.tsv
  "$bcf" view -i 'INFO/F_MISSING=0' -H panel/panel.vcf.gz | cut -f3 \
    > panel/complete_genotype_snp_ids.txt
  "$plink" --vcf panel/panel.vcf.gz --double-id --keep-allele-order \
    --make-bed --out panel/panel > panel/plink.log 2>&1
  test "$(wc -l < panel/samples.txt)" -eq 443
  date -u +%FT%TZ > panel/COMPLETE
fi

"$python" -m src.gwfm_benchmark block-map --panel "$pilot/panel" \
  --output "$pilot/blocks.pos" --max-snps 500
if [ ! -f ld/ldm.info ]; then
  "$gctb" --bfile panel/panel --make-block-ldm --block-info blocks.pos \
    --thread 4 --out ld > make_ldm.log 2>&1
  if grep -E 'Error:|Error opening' make_ldm.log; then exit 1; fi
  test -s ld/ldm.info
fi
if [ ! -f ld/block1.eigen.bin ]; then
  "$gctb" --ldm ld --make-ldm-eigen --thread 4 --out eigen \
    > make_eigen.log 2>&1
  if grep -E 'Error:|Error opening' make_eigen.log; then exit 1; fi
  test -s ld/block1.eigen.bin
fi
awk 'BEGIN {print "SNP Intercept"} {print $2, 1}' panel/panel.bim \
  > ld/intercept_only.txt

"$python" -m src.gwfm_benchmark run-one \
  --panel "$pilot/panel" --ldm "$pilot/ld" \
  --output "$pilot/run_gaussian50_h015_seed42" \
  --architecture gaussian --count 50 --causal-var 0.15 --seed 42 \
  --gemma-workers 4 --gctb-threads 4
