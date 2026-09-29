#!/usr/bin/env bash
# Compatibility pilot with real biallelic SNPs and InDels, on rock01.
set -euo pipefail
project=/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum
root=/lustre/home/dobby/Program/agent_program/bin
bench="$project/data/results/gwfm_rc_benchmark"
source_vcf="$project/data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz"
pilot="$bench/pilot_all_variants"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$root:$PATH"
mkdir -p "$pilot/panel" "$pilot/ld"
cd "$pilot"
if [ ! -f panel/COMPLETE ]; then
  test ! -e panel/panel.vcf.gz
  printf 'Chr01\t1\n' > panel/chromosome_map.tsv
  "$root/bcftools" view -r Chr01:1-300000 -m2 -M2 -v snps,indels -Ou "$source_vcf" \
    | "$root/bcftools" +fill-tags -Ou -- -t F_MISSING,MAF \
    | "$root/bcftools" view -i 'INFO/F_MISSING<=0.1 && INFO/MAF>=0.05' -Ou \
    | "$root/bcftools" annotate --rename-chrs panel/chromosome_map.tsv -Ou \
    | "$root/bcftools" annotate --set-id '%CHROM:%POS:%REF:%ALT' -Ou \
    | "$root/bcftools" norm -d exact -Oz -o panel/panel.vcf.gz
  "$root/bcftools" index -t panel/panel.vcf.gz
  "$root/bcftools" query -l panel/panel.vcf.gz > panel/samples.txt
  "$root/bcftools" query -f '%ID\t%CHROM\t%POS\t%REF\t%ALT\t%MAF\t%F_MISSING\n' panel/panel.vcf.gz \
    | awk 'BEGIN {OFS="\t"; print "SNP","CHROM","POS","REF","ALT","MAF","F_MISSING","TYPE"} {print $0,(length($4)==1 && length($5)==1 ? "SNP" : "INDEL")}' \
    > panel/variants.tsv
  "$root/bcftools" view -i 'INFO/F_MISSING=0' -H panel/panel.vcf.gz \
    | cut -f3 > panel/complete_genotype_variant_ids.txt
  "$root/plink" --vcf panel/panel.vcf.gz --double-id --keep-allele-order \
    --make-bed --out panel/panel > panel/plink.log 2>&1
  awk 'NR>1 {count[$8]++} END {if (count["INDEL"]<10) exit 1; for (t in count) print t,count[t]}' panel/variants.tsv
  date -u +%FT%TZ > panel/COMPLETE
fi
python3 -m src.gwfm_benchmark bp-block-map --panel "$pilot/panel" \
  --output "$pilot/blocks_15kb.pos" --width-bp 15000
if [ ! -s ld/ldm.info ]; then
  "$root/gctb" --bfile panel/panel --make-block-ldm \
    --block-info blocks_15kb.pos --thread 4 --out ld > make_ldm.log 2>&1
  test -s ld/ldm.info
  if grep -Eq 'Error:|Error opening' make_ldm.log; then exit 1; fi
fi
if [ ! -s ld/block1.eigen.bin ]; then
  "$root/gctb" --ldm ld --make-ldm-eigen --thread 4 --out eigen > make_eigen.log 2>&1
  test -s ld/block1.eigen.bin
  if grep -Eq 'Error:|Error opening' make_eigen.log; then exit 1; fi
fi
awk 'BEGIN {print "SNP Intercept"} {print $2, 1}' panel/panel.bim > ld/intercept_only.txt
python3 -m src.gwfm_benchmark run-one --panel "$pilot/panel" --ldm "$pilot/ld" \
  --output "$pilot/run_gaussian50_h015_seed42" --architecture gaussian \
  --count 50 --causal-var 0.15 --seed 42 --gemma-workers 4 --gctb-threads 4
