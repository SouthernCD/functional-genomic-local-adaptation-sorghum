#!/usr/bin/env bash

set -euo pipefail

# PLINK 1.9 Gabriel/Haploview blocks for the project's Chr01-Chr10 VCF.
# Run from the notebook on the configured compute node.
usage() {
    cat <<'EOF'
Usage: plink_block.sh --vcf FILE --out-dir DIR [options]
  --prefix NAME                 Output basename (default: landraces)
  --maf FLOAT                   SNP/block minimum MAF (default: 0.05)
  --max-kb INT                  Maximum block span in kb (default: 500)
  --threads INT                 Preprocessing threads (default: 8)
  --parallel INT                Concurrent chromosome processes (default: 4)
  --block-memory-mb INT         PLINK workspace per chromosome (default: 64000)
  --preprocess-memory-mb INT    Preprocessing workspace (default: 16000)
  --plink COMMAND              PLINK 1.9 executable (default: plink)
  --help                        Show this help

Output: DIR/NAME.mafMAF.biallelic.blocks_maxMAXkb.project_chr.blocks.det
The caller owns result reuse. Running this script recomputes the supplied output.
EOF
}

ld_vcf=
ld_outdir=
ld_prefix=landraces
ld_plink=plink
ld_maf=0.05
ld_block_max_kb=500
ld_threads=8
ld_parallel=4
ld_block_memory_mb=64000
ld_preprocess_memory_mb=16000
ld_chromosomes=Chr01,Chr02,Chr03,Chr04,Chr05,Chr06,Chr07,Chr08,Chr09,Chr10

while [[ $# -gt 0 ]]; do
    if [[ "$1" == --help ]]; then usage; exit 0; fi
    if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2; exit 1
    fi
    case "$1" in
        --vcf) ld_vcf=$2 ;;
        --out-dir) ld_outdir=$2 ;;
        --prefix) ld_prefix=$2 ;;
        --maf) ld_maf=$2 ;;
        --max-kb) ld_block_max_kb=$2 ;;
        --threads) ld_threads=$2 ;;
        --parallel) ld_parallel=$2 ;;
        --block-memory-mb) ld_block_memory_mb=$2 ;;
        --preprocess-memory-mb) ld_preprocess_memory_mb=$2 ;;
        --plink) ld_plink=$2 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift 2
done
if [[ -z "$ld_vcf" || -z "$ld_outdir" ]]; then
    usage >&2; exit 1
fi
for value in "$ld_threads" "$ld_parallel" "$ld_block_memory_mb" "$ld_preprocess_memory_mb" "$ld_block_max_kb"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Thread, memory and span parameters must be positive integers." >&2
        exit 1
    fi
done

###############################################################################
# Setup
###############################################################################

command -v "$ld_plink" >/dev/null

if [[ ! -f "$ld_vcf" ]]; then
    echo "Missing input VCF: $ld_vcf" >&2
    exit 1
fi

mkdir -p "$ld_outdir"

ld_import="${ld_outdir}/${ld_prefix}.maf${ld_maf}.biallelic.import"
ld_panel="${ld_outdir}/${ld_prefix}.maf${ld_maf}.biallelic"
ld_blocks="${ld_outdir}/${ld_prefix}.maf${ld_maf}.biallelic.blocks_max${ld_block_max_kb}kb"
merged_det="$ld_blocks.project_chr.blocks.det"

###############################################################################
# Step 1: VCF -> PLINK binary
#
# Input:
#   unpruned QC VCF
#
# Filters:
#   biallelic A/C/G/T SNPs
#   MAF >= supplied --maf
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

# Limit simultaneous single-threaded --blocks processes to the resource budget.
pids=()
chromosome_jobs=()
wait_for_blocks() {
    local failed=0 i
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "Chr${chromosome_jobs[$i]} finished."
        else
            echo "ERROR: Chr${chromosome_jobs[$i]} failed; see its stdout log." >&2
            failed=1
        fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "At least one chromosome failed. Stop before merging." >&2
        exit 1
    fi
    pids=()
    chromosome_jobs=()
}

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
      --threads 1 \
      --memory "$ld_block_memory_mb" \
      --out "$chr_out" \
      > "${chr_out}.stdout.log" 2>&1 &
    pids+=("$!")
    chromosome_jobs+=("$chr")
    if [[ ${#pids[@]} -ge "$ld_parallel" ]]; then
        wait_for_blocks
    fi
done
if [[ ${#pids[@]} -gt 0 ]]; then
    wait_for_blocks
fi

###############################################################################
# Step 5: merge .blocks.det and restore project chromosome names
###############################################################################

echo "Step 5: merging chromosome block files..."

temporary_det="${merged_det}.tmp"
head -n 1 "${ld_blocks}.chr1.blocks.det" > "$temporary_det"

for chr in {1..10}; do
    awk -v chr="$chr" '
        BEGIN {OFS="\t"}
        NR > 1 {
            $1=sprintf("Chr%02d", chr)
            print
        }
    ' "${ld_blocks}.chr${chr}.blocks.det" >> "$temporary_det"
done

mv "$temporary_det" "$merged_det"

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
