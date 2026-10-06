#!/usr/bin/env bash

set -euo pipefail

###############################################################################
# Parameters
###############################################################################

ld_project=/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum
ld_vcf="$ld_project/data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz"
ld_outdir="$ld_project/data/results/landraces_qc_miss0.5_maf0.05_het0.05.plink_blocks"

ld_plink=plink
ld_maf=0.05
ld_block_max_kb=500

# Preprocessing steps can use multiple threads.
ld_threads=8

# PLINK --blocks itself is single-threaded.
# Ten chromosomes are run as ten independent PLINK processes.
# Chr4 required >8 GB in the current dataset, so use 64 GB/process.
# With 10 concurrent processes, the theoretical maximum is ~640 GB.
ld_block_memory_mb=64000
ld_preprocess_memory_mb=16000

ld_chromosomes=Chr01,Chr02,Chr03,Chr04,Chr05,Chr06,Chr07,Chr08,Chr09,Chr10

###############################################################################
# Setup
###############################################################################

command -v "$ld_plink" >/dev/null

if [[ ! -f "$ld_vcf" ]]; then
    echo "Missing input VCF: $ld_vcf" >&2
    exit 1
fi

mkdir -p "$ld_outdir"

ld_import="$ld_outdir/landraces.maf${ld_maf}.biallelic.import"
ld_panel="$ld_outdir/landraces.maf${ld_maf}.biallelic"
ld_blocks="$ld_outdir/landraces.maf${ld_maf}.biallelic.blocks_max${ld_block_max_kb}kb"
merged_det="$ld_blocks.project_chr.blocks.det"

###############################################################################
# Step 1: VCF -> PLINK binary
#
# Input:
#   unpruned QC VCF
#
# Filters:
#   biallelic A/C/G/T SNPs
#   MAF >= 0.05
###############################################################################

echo "Step 1: importing VCF..."

"$ld_plink" \
  --vcf "$ld_vcf" \
  --allow-extra-chr \
  --chr-set 10 no-x no-y no-xy no-mt \
  --chr "$ld_chromosomes" \
  --double-id \
  --biallelic-only strict \
  --snps-only just-acgt \
  --maf "$ld_maf" \
  --set-missing-var-ids '@:#:$1:$2' \
  --keep-allele-order \
  --make-bed \
  --threads "$ld_threads" \
  --memory "$ld_preprocess_memory_mb" \
  --out "$ld_import"

###############################################################################
# Step 2: map Chr01-Chr10 -> PLINK chromosomes 1-10
#
# Physical positions are unchanged.
###############################################################################

echo "Step 2: remapping chromosome names..."

awk 'BEGIN {OFS="\t"} {
    c=$1
    sub(/^Chr/, "", c)
    print $2, c+0
}' "$ld_import.bim" \
  > "$ld_outdir/variant_chr_for_plink.tsv"

awk 'BEGIN {
    OFS="\t"
    for(i=1;i<=10;i++)
        print i, sprintf("Chr%02d",i)
}' > "$ld_outdir/plink_to_project_chr.tsv"

"$ld_plink" \
  --bfile "$ld_import" \
  --allow-extra-chr \
  --chr-set 10 no-x no-y no-xy no-mt \
  --update-chr "$ld_outdir/variant_chr_for_plink.tsv" 2 1 \
  --keep-allele-order \
  --make-bed \
  --threads "$ld_threads" \
  --memory "$ld_preprocess_memory_mb" \
  --out "$ld_panel"

###############################################################################
# Step 3: PLINK haplotype blocks
#
# --blocks is single-threaded, so chromosomes are run in parallel.
# Gabriel/Haploview block definition as implemented by PLINK 1.9.
#
# no-small-max-span removes the additional default 20 kb / 30 kb maximum
# span restrictions for 2-SNP / 3-SNP blocks.
###############################################################################

echo "Step 3: calculating LD blocks for 10 chromosomes..."

pids=()

for chr in {1..10}; do
    chr_out="${ld_blocks}.chr${chr}"

    echo "Starting chromosome ${chr} ..."

    "$ld_plink" \
      --bfile "$ld_panel" \
      --chr-set 10 no-x no-y no-xy no-mt \
      --chr "$chr" \
      --blocks no-pheno-req no-small-max-span \
      --blocks-min-maf "$ld_maf" \
      --blocks-max-kb "$ld_block_max_kb" \
      --memory "$ld_block_memory_mb" \
      --out "$chr_out" \
      > "${chr_out}.stdout.log" 2>&1 &

    pids+=("$!")
done

###############################################################################
# Step 4: wait for chromosome jobs
###############################################################################

failed=0

for i in "${!pids[@]}"; do
    chr=$((i + 1))

    if wait "${pids[$i]}"; then
        echo "Chr${chr} finished."
    else
        echo "ERROR: Chr${chr} failed." >&2
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "At least one chromosome failed. Stop before merging." >&2
    exit 1
fi

###############################################################################
# Step 5: merge .blocks.det and restore project chromosome names
###############################################################################

echo "Step 5: merging chromosome block files..."

head -n 1 "${ld_blocks}.chr1.blocks.det" > "$merged_det"

for chr in {1..10}; do
    awk -v chr="$chr" '
        BEGIN {OFS="\t"}
        NR > 1 {
            $1=sprintf("Chr%02d", chr)
            print
        }
    ' "${ld_blocks}.chr${chr}.blocks.det" >> "$merged_det"
done

###############################################################################
# Step 6: QC
###############################################################################

echo
echo "Final block file:"
echo "$merged_det"

echo
awk 'NR>1 {n++} END {print "BLOCK_TOTAL=" n+0}' "$merged_det"

echo
echo "First rows:"
head -n 6 "$merged_det"

echo
echo "Blocks per chromosome:"
awk '
    NR > 1 {n[$1]++}
    END {
        for (c in n)
            print c, n[c]
    }
' "$merged_det" | sort

echo
echo "Done."