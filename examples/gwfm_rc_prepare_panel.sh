#!/usr/bin/env bash
# Build the immutable, phenotype-blind SNP + InDel panel shared by GEMMA and GCTB.
set -euo pipefail

project=/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum
root=/lustre/home/dobby/Program/agent_program
source_vcf="$project/data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz"
panel="$project/data/results/gwfm_rc_benchmark/panel_all_variants"
bcf="$root/bin/bcftools"
plink="$root/bin/plink"

if [ -e "$panel/COMPLETE" ]; then
  echo "Panel already complete: $panel"
  exit 0
fi
mkdir -p "$panel"
cd "$panel"
if [ -e panel.vcf.gz ] || [ -e panel.bed ]; then
  echo "Incomplete panel exists; inspect $panel before retrying" >&2
  exit 1
fi

for chromosome in {1..10}; do
  printf 'Chr%02d\t%d\n' "$chromosome" "$chromosome"
done > chromosome_map.tsv

# Filtering is independent of all simulated phenotypes and GWAS results.
# Match GEMMA's -miss 0.1 and retain the input VCF's MAF >= 0.05 rule.
"$bcf" view -m2 -M2 -v snps,indels -Ou "$source_vcf" \
  | "$bcf" +fill-tags -Ou -- -t F_MISSING,MAF \
  | "$bcf" view -i 'INFO/F_MISSING<=0.1 && INFO/MAF>=0.05' -Ou \
  | "$bcf" annotate --rename-chrs chromosome_map.tsv -Ou \
  | "$bcf" annotate --set-id '%CHROM:%POS:%REF:%ALT' -Ou \
  | "$bcf" norm -d exact -Oz -o panel.vcf.gz
"$bcf" index -t panel.vcf.gz
"$bcf" query -l panel.vcf.gz > samples.txt
"$bcf" query -f '%ID\t%CHROM\t%POS\t%REF\t%ALT\t%MAF\t%F_MISSING\n' panel.vcf.gz \
  | awk 'BEGIN {OFS="\t"; print "SNP","CHROM","POS","REF","ALT","MAF","F_MISSING","TYPE"} {print $0,(length($4)==1 && length($5)==1 ? "SNP" : "INDEL")}' \
  > variants.tsv
"$bcf" view -i 'INFO/F_MISSING=0' -H panel.vcf.gz \
  | cut -f3 > complete_genotype_variant_ids.txt

"$plink" --vcf panel.vcf.gz --double-id --keep-allele-order \
  --make-bed --out panel > plink.log 2>&1

python3 - <<'PY'
from pathlib import Path
samples = Path('samples.txt').read_text().splitlines()
fam = [line.split()[1] for line in Path('panel.fam').read_text().splitlines()]
if samples != fam or len(samples) != 443:
    raise SystemExit('VCF/PLINK sample order mismatch or sample count changed')
with Path('variants.tsv').open() as source:
    next(source)
    ids = [line.split('\t', 1)[0] for line in source]
if not ids or len(ids) != len(set(ids)):
    raise SystemExit('Panel variant IDs are empty or duplicated')
bim = [line.split()[1] for line in Path('panel.bim').read_text().splitlines()]
if ids != bim:
    raise SystemExit('VCF/PLINK variant order or IDs differ')
print(f'Validated {len(samples)} samples and {len(ids)} variants')
PY

source_total=$("$bcf" index -n "$source_vcf")
source_selected=$("$bcf" view -v snps,indels -H "$source_vcf" | wc -l)
source_biallelic=$("$bcf" view -v snps,indels -m2 -M2 -H "$source_vcf" | wc -l)
retained=$(($(wc -l < variants.tsv)-1))
{
  printf 'category\tcount\n'
  printf 'source_total\t%d\n' "$source_total"
  printf 'other_variant_type\t%d\n' "$((source_total-source_selected))"
  printf 'non_biallelic_snp_or_indel\t%d\n' "$((source_selected-source_biallelic))"
  printf 'failed_maf_missingness_or_exact_duplicate\t%d\n' "$((source_biallelic-retained))"
  printf 'retained_biallelic_snp_or_indel\t%d\n' "$retained"
} > qc_exclusions.tsv

sha256sum "$source_vcf" panel.vcf.gz panel.bed panel.bim panel.fam \
  variants.tsv samples.txt chromosome_map.tsv qc_exclusions.tsv > SHA256SUMS
{
  echo 'source_vcf='"$source_vcf"
  echo 'bcftools='"$("$bcf" --version | sed -n '1p')"
  echo 'plink='"$("$plink" --version)"
  echo 'qc=biallelic SNP or InDel; MAF>=0.05; F_MISSING<=0.1; exact duplicate removal'
  echo 'variant_count='"$(($(wc -l < variants.tsv)-1))"
  echo 'causal_candidate_count='"$(wc -l < complete_genotype_variant_ids.txt)"
  echo 'sample_count='"$(wc -l < samples.txt)"
} > panel.properties
date -u +%FT%TZ > COMPLETE
