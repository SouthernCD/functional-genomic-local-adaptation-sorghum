"""
Gene-centered TSS/TTS profile database.

This module calculates per-gene profiles first, without applying any GEA
top-gene filtering. The resulting matrices can later be subset using any
top-gene list or random gene sets.

Default profile settings:
    QRE/pi/ACR/UMR/LD: flank=10000, bin_size=500, step=10

Output matrices use:
    rows    = gene IDs
    columns = relative window positions in transcription direction
"""

import gzip
import json
from math import ceil
from pathlib import Path

import allel
import numpy as np
import pandas as pd
from cyvcf2 import VCF
from interlap import InterLap

from .feature import get_gene_range_df
from .ld import calculate_ld_matrix
from .utils import multiprocess_running


DEFAULT_CONFIG = {
    "qre": {"flank": 10000, "bin_size": 500, "step": 10},
    "pi": {"flank": 10000, "bin_size": 500, "step": 10},
    "acr": {"flank": 10000, "bin_size": 500, "step": 10},
    "umr": {"flank": 10000, "bin_size": 500, "step": 10},
    "ld": {"flank": 10000, "bin_size": 500, "step": 10},
}


# ---------------------------------------------------------------------
# Basic window helpers
# ---------------------------------------------------------------------

def windows_index_to_range(index, step, window, start=1):
    """Return the start and end coordinates of one sliding window."""
    window_start = index * step + start
    window_end = window_start + window - 1
    return window_start, window_end


def total_windows(end, step, window, start=1):
    """Return the number of sliding windows for the given region."""
    seq_len = end - start
    return ceil((seq_len - window + 1) / step) + 1


def split_sequence_to_windows(end, step, window, start=1):
    """Yield window index, start, and end for a genomic region."""
    for i in range(total_windows(end, step, window, start)):
        window_start, window_end = windows_index_to_range(
            i, step, window, start
        )
        window_start = round(window_start)
        window_end = round(window_end)

        if window_start > end:
            continue
        if window_end > end:
            window_end = end

        yield i, window_start, window_end


def _relative_window_positions(flank, step, bin_size):
    """
    Return the relative window coordinates used by the original QRE/pi code.

    With flank=10000, step=10, bin_size=500:
        -9999, -9989, ..., 9501

    These are relative window positions, not window centers.
    """
    start = -flank
    end = flank - 1

    positions = []
    for _, window_start, _ in split_sequence_to_windows(
        end, step, bin_size, start=start
    ):
        positions.append(abs(window_start - start) + 1 - flank)

    return np.asarray(positions, dtype=int)


# ---------------------------------------------------------------------
# QRE
# ---------------------------------------------------------------------

def get_region_bin_QREs(
    chr_id,
    start,
    end,
    site_info_df,
    step=10,
    bin_size=500,
    adjust_flank=None,
    strand="+",
):
    """Calculate QRE in sliding windows for one region."""
    bin_start_positions = []

    for _, window_start, window_end in split_sequence_to_windows(
        end, step, bin_size, start=start
    ):
        if strand == "-":
            window_start = start + end - window_start
            window_end = start + end - window_end
            bin_start_positions.append(window_start)
        else:
            bin_start_positions.append(window_start)

    bin_start_positions = np.asarray(bin_start_positions)

    if adjust_flank is not None:
        if strand == "-":
            bin_start_positions = abs(bin_start_positions - end) + 1
        else:
            bin_start_positions = abs(bin_start_positions - start) + 1
        bin_start_positions = bin_start_positions - adjust_flank

    region_site_df = site_info_df[
        (site_info_df["CHROM"] == chr_id)
        & (site_info_df["POS"] >= start)
        & (site_info_df["POS"] <= end)
    ]

    if len(region_site_df) < 4:
        return pd.DataFrame(
            np.ones(len(bin_start_positions)),
            index=bin_start_positions,
            columns=["QRE"],
        )

    region_q25 = np.percentile(region_site_df["P_WALD"], 25)
    bin_qre = []

    for _, window_start, window_end in split_sequence_to_windows(
        end, step, bin_size, start=start
    ):
        if strand == "-":
            window_start = start + end - window_start
            window_end = start + end - window_end
            bin_snp_df = region_site_df[
                (region_site_df["POS"] >= window_end)
                & (region_site_df["POS"] <= window_start)
            ]
        else:
            bin_snp_df = region_site_df[
                (region_site_df["POS"] >= window_start)
                & (region_site_df["POS"] <= window_end)
            ]

        if len(bin_snp_df) < 4:
            window_q25 = region_q25
        else:
            window_q25 = np.percentile(bin_snp_df["P_WALD"], 25)

        qre = -np.log10(window_q25) / -np.log10(region_q25)
        bin_qre.append(qre)

    return pd.DataFrame(
        bin_qre,
        index=bin_start_positions,
        columns=["QRE"],
    )


def get_bin_QREs_for_many_pos_flank_regions(
    pos_df,
    gemma_gwas_df_h5,
    flank=10000,
    step=10,
    bin_size=500,
    threads=80,
):
    """
    Calculate QRE profiles around many genomic positions.

    pos_df must contain CHROM, POS and STRAND.
    Returned rows are relative window positions and columns are anchors.
    """
    gemma_gwas_df = pd.read_hdf(gemma_gwas_df_h5, key="df")

    chr_gwas = {
        chrom: sub[["CHROM", "POS", "P_WALD"]].copy()
        for chrom, sub in gemma_gwas_df.groupby("CHROM")
    }

    args_dict = {}

    for _, row in pos_df.iterrows():
        chr_id = row["CHROM"]
        pos = int(row["POS"])
        strand = row["STRAND"]
        anchor_id = f"{chr_id}:{pos}_{strand}"

        if anchor_id in args_dict:
            continue

        region_start = pos - flank
        region_end = pos + flank - 1

        if chr_id in chr_gwas:
            site_df = chr_gwas[chr_id]
            site_df = site_df[
                (site_df["POS"] >= region_start)
                & (site_df["POS"] <= region_end)
            ]
        else:
            site_df = pd.DataFrame(columns=["CHROM", "POS", "P_WALD"])

        args_dict[anchor_id] = (
            chr_id,
            region_start,
            region_end,
            site_df,
            step,
            bin_size,
            flank,
            strand,
        )

    results = multiprocess_running(
        get_region_bin_QREs,
        args_dict,
        threads,
    )

    profile_df = pd.DataFrame()

    for anchor_id, result in results.items():
        anchor_profile = result["output"]["QRE"]
        profile_df[anchor_id] = anchor_profile

    return profile_df


# ---------------------------------------------------------------------
# Nucleotide diversity
# ---------------------------------------------------------------------

def load_genotypes(vcf_file, chr_id, range_list):
    """Load genotype calls from one or more genomic intervals."""
    range_interval = InterLap(range_list)

    minimum_start = min(start for start, _ in range_list)
    maximum_end = max(end for _, end in range_list)
    region = f"{chr_id}:{minimum_start}-{maximum_end}"

    vcf = VCF(vcf_file)

    genotypes = []
    positions = []

    for variant in vcf(region):
        if any(range_interval.find((variant.POS, variant.POS))):
            genotypes.append(variant.genotypes)
            positions.append(variant.POS)

    vcf.close()

    if len(genotypes) == 0:
        return None, None

    # Keep the two allele calls and discard the phasing flag.
    genotypes = np.asarray(genotypes)[:, :, :2]

    return genotypes, positions



def load_range_genotypes_and_convert(
    vcf_file,
    chr_id,
    start,
    end,
    strand,
):
    """Load genotypes and convert positions to region-relative coordinates."""
    genotypes, positions = load_genotypes(
        vcf_file,
        chr_id,
        [(start, end)],
    )

    if genotypes is None:
        return None, None

    positions = np.asarray(positions)

    if strand == "-":
        positions = abs(positions - end) + 1
        positions = np.flip(positions)
        genotypes = np.flip(genotypes, axis=0)
    else:
        positions = positions - start + 1

    return genotypes, positions


def get_pi_values(
    genotypes,
    positions,
    window_size,
    start,
    end,
    step=None,
):
    """Calculate nucleotide diversity with scikit-allel."""
    genotypes = allel.GenotypeArray(genotypes)
    allele_counts = genotypes.count_alleles()

    if step is None:
        step = window_size // 2

    pi_values, windows, n_bases, counts = allel.windowed_diversity(
        positions,
        allele_counts,
        size=window_size,
        start=start,
        stop=end,
        step=step,
    )

    return counts, pi_values, windows



def get_region_bin_pi(
    chr_id,
    start,
    end,
    vcf_file,
    step=10,
    bin_size=500,
    adjust_flank=None,
    strand="+",
):
    """Calculate nucleotide diversity in sliding windows for one region."""
    bin_start_positions = []

    for _, window_start, window_end in split_sequence_to_windows(
        end, step, bin_size, start=start
    ):
        if strand == "-":
            window_start = start + end - window_start
            window_end = start + end - window_end
            bin_start_positions.append(window_start)
        else:
            bin_start_positions.append(window_start)

    bin_start_positions = np.asarray(bin_start_positions)

    if adjust_flank is not None:
        if strand == "-":
            bin_start_positions = abs(bin_start_positions - end) + 1
        else:
            bin_start_positions = abs(bin_start_positions - start) + 1
        bin_start_positions = bin_start_positions - adjust_flank

    genotypes, positions = load_range_genotypes_and_convert(
        vcf_file,
        chr_id,
        start,
        end,
        strand,
    )

    if genotypes is None:
        pi_values = np.zeros(len(bin_start_positions))
    else:
        _, pi_values, _ = get_pi_values(
            genotypes,
            positions,
            bin_size,
            1,
            end - start + 2,
            step=step,
        )

    return pd.DataFrame(
        pi_values,
        index=bin_start_positions,
        columns=["pi"],
    )


def get_bin_pi_for_many_pos_flank_regions(
    pos_df,
    vcf_file,
    flank=10000,
    step=10,
    bin_size=500,
    threads=80,
):
    """
    Calculate pi profiles around many genomic positions.

    pos_df must contain CHROM, POS and STRAND.
    Returned rows are relative window positions and columns are anchors.
    """
    args_dict = {}

    for _, row in pos_df.iterrows():
        chr_id = row["CHROM"]
        pos = int(row["POS"])
        strand = row["STRAND"]
        anchor_id = f"{chr_id}:{pos}_{strand}"

        if anchor_id in args_dict:
            continue

        region_start = pos - flank
        region_end = pos + flank - 1

        args_dict[anchor_id] = (
            chr_id,
            region_start,
            region_end,
            vcf_file,
            step,
            bin_size,
            flank,
            strand,
        )

    results = multiprocess_running(
        get_region_bin_pi,
        args_dict,
        threads,
    )

    profile_df = pd.DataFrame()

    for anchor_id, result in results.items():
        anchor_profile = result["output"]["pi"]
        profile_df[anchor_id] = anchor_profile

    return profile_df


# ---------------------------------------------------------------------
# Gene information
# ---------------------------------------------------------------------

def build_gene_info(
    gff_file,
    output_file=None,
    chromosome_filter=None,
    overwrite=False,
):
    """Build the gene table shared by all profile databases.

    Gene coordinates are loaded through feature.get_gene_range_df(), so GFF
    parsing stays centralized in feature.py. TSS/TTS are then derived from
    gene START/END according to transcription strand.
    """
    if output_file is not None:
        output_file = Path(output_file)
        if output_file.exists() and not overwrite:
            return pd.read_hdf(output_file, key="gene_info")

    gene_range_df = get_gene_range_df(gff_file).copy()

    required_columns = {"GENE", "CHROM", "START", "END", "STRAND"}
    missing_columns = required_columns - set(gene_range_df.columns)
    if missing_columns:
        raise ValueError(
            "get_gene_range_df() output is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    gene_range_df = gene_range_df[
        gene_range_df["STRAND"].isin(["+", "-"])
    ].copy()

    if chromosome_filter is not None:
        chromosome_filter = set(chromosome_filter)
        gene_range_df = gene_range_df[
            gene_range_df["CHROM"].isin(chromosome_filter)
        ].copy()

    gene_range_df["START"] = gene_range_df["START"].astype(int)
    gene_range_df["END"] = gene_range_df["END"].astype(int)

    plus_strand = gene_range_df["STRAND"] == "+"
    gene_range_df["TSS"] = np.where(
        plus_strand, gene_range_df["START"], gene_range_df["END"]
    )
    gene_range_df["TTS"] = np.where(
        plus_strand, gene_range_df["END"], gene_range_df["START"]
    )
    gene_range_df["gene_length"] = (
        gene_range_df["END"] - gene_range_df["START"]
    ).abs() + 1

    gene_info = gene_range_df.rename(
        columns={
            "GENE": "gene_id",
            "CHROM": "chrom",
            "START": "start",
            "END": "end",
            "STRAND": "strand",
        }
    ).set_index("gene_id")
    gene_info.index.name = "gene_id"
    gene_info = gene_info[
        ["chrom", "strand", "start", "end", "TSS", "TTS", "gene_length"]
    ]

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        gene_info.to_hdf(
            output_file,
            key="gene_info",
            mode="w",
            format="fixed",
        )

    return gene_info

def make_anchor_table(gene_info, anchor):
    """Create CHROM/POS/STRAND input for a TSS or TTS calculation."""
    anchor = anchor.upper()

    if anchor not in {"TSS", "TTS"}:
        raise ValueError("anchor must be 'TSS' or 'TTS'")

    return pd.DataFrame(
        {
            "CHROM": gene_info["chrom"],
            "POS": gene_info[anchor].astype(int),
            "STRAND": gene_info["strand"],
        },
        index=gene_info.index,
    )


# ---------------------------------------------------------------------
# Profile file helpers
# ---------------------------------------------------------------------

def _save_profile(
    profile_df,
    output_file,
    track,
    anchor,
    flank,
    bin_size,
    step,
    source=None,
    label=None,
    extra_metadata=None,
):
    """Save one profile matrix and its basic calculation parameters."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    profile_df = profile_df.copy()
    profile_df.index.name = "gene_id"
    profile_df.columns = pd.Index(
        [int(x) for x in profile_df.columns],
        name="relative_position",
    )

    metadata = {
        "track": track,
        "anchor": anchor,
        "flank": int(flank),
        "bin_size": int(bin_size),
        "step": int(step),
        "orientation": "transcription_direction",
    }

    if source is not None:
        metadata["source"] = str(source)
    if label is not None:
        metadata["label"] = str(label)
    if extra_metadata is not None:
        metadata.update(extra_metadata)

    with pd.HDFStore(output_file, mode="w") as store:
        store.put("profile", profile_df, format="fixed")
        store.get_storer("profile").attrs.metadata = json.dumps(metadata)

    return output_file


def read_profile(output_file):
    """Read a profile database and its metadata."""
    with pd.HDFStore(output_file, mode="r") as store:
        profile_df = store["profile"]
        metadata = json.loads(
            getattr(store.get_storer("profile").attrs, "metadata", "{}")
        )

    return profile_df, metadata


def _restore_gene_ids(raw_profile, anchor_df):
    """
    Convert anchor-based output to a gene x position matrix.

    Multiple genes may share the same TSS/TTS. The QRE/pi engines calculate
    each unique genomic anchor once; this function maps that profile back to
    every corresponding gene.
    """
    anchor_profile = raw_profile.T.copy()
    anchor_profile.index = anchor_profile.index.astype(str)

    gene_profiles = []

    for gene_id, row in anchor_df.iterrows():
        anchor_id = f"{row['CHROM']}:{int(row['POS'])}_{row['STRAND']}"

        if anchor_id in anchor_profile.index:
            profile = anchor_profile.loc[anchor_id]
            if isinstance(profile, pd.DataFrame):
                profile = profile.iloc[0]
            profile = profile.copy()
        else:
            profile = pd.Series(
                np.nan,
                index=raw_profile.index,
            )

        profile.name = gene_id
        gene_profiles.append(profile)

    profile_df = pd.DataFrame(gene_profiles)
    profile_df.index.name = "gene_id"

    return profile_df


# ---------------------------------------------------------------------
# QRE and pi profile databases
# ---------------------------------------------------------------------

def build_qre_profiles(
    gene_info,
    qre_source,
    output_dir,
    label,
    flank=10000,
    bin_size=500,
    step=10,
    threads=80,
    overwrite=False,
):
    """Build TSS and TTS QRE databases for one GEA environment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    for anchor in ("TSS", "TTS"):
        output_file = output_dir / f"{anchor}.QRE.{label}.h5"
        output_files[anchor] = output_file

        if output_file.exists() and not overwrite:
            continue

        anchor_df = make_anchor_table(gene_info, anchor)

        raw_profile = get_bin_QREs_for_many_pos_flank_regions(
            anchor_df,
            qre_source,
            flank=flank,
            step=step,
            bin_size=bin_size,
            threads=threads,
        )

        profile_df = _restore_gene_ids(raw_profile, anchor_df)

        _save_profile(
            profile_df,
            output_file,
            track="QRE",
            anchor=anchor,
            flank=flank,
            bin_size=bin_size,
            step=step,
            source=qre_source,
            label=label,
        )

    return output_files


def build_pi_profiles(
    gene_info,
    vcf_file,
    output_dir,
    flank=10000,
    bin_size=500,
    step=10,
    threads=80,
    overwrite=False,
):
    """Build TSS and TTS nucleotide-diversity databases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    for anchor in ("TSS", "TTS"):
        output_file = output_dir / f"{anchor}.pi.h5"
        output_files[anchor] = output_file

        if output_file.exists() and not overwrite:
            continue

        anchor_df = make_anchor_table(gene_info, anchor)

        raw_profile = get_bin_pi_for_many_pos_flank_regions(
            anchor_df,
            vcf_file,
            flank=flank,
            step=step,
            bin_size=bin_size,
            threads=threads,
        )

        profile_df = _restore_gene_ids(raw_profile, anchor_df)

        _save_profile(
            profile_df,
            output_file,
            track="pi",
            anchor=anchor,
            flank=flank,
            bin_size=bin_size,
            step=step,
            source=vcf_file,
        )

    return output_files


# ---------------------------------------------------------------------
# LD profiles
# ---------------------------------------------------------------------

def _load_variant_sites(vcf_file, chr_id, start, end):
    """Return unique VCF variant coordinates in one genomic interval."""
    start = max(1, int(start))
    end = int(end)

    if end < start:
        return pd.DataFrame(columns=["CHROM", "POS"])

    vcf = VCF(vcf_file)
    try:
        records = [(variant.CHROM, int(variant.POS))
                   for variant in vcf(f"{chr_id}:{start}-{end}")]
    finally:
        vcf.close()

    if not records:
        return pd.DataFrame(columns=["CHROM", "POS"])

    return (
        pd.DataFrame(records, columns=["CHROM", "POS"])
        .drop_duplicates(["CHROM", "POS"])
        .reset_index(drop=True)
    )


def _anchor_window_relative_position(bin_size):
    """Return the relative start of a window centered on TSS/TTS."""
    return 1 - bin_size // 2


def _resolve_ld_window_size(flank, bin_size, ld_window_size):
    """Choose an LD distance cutoff that covers the complete profile range."""
    minimum = int(flank) + (int(bin_size) + 1) // 2

    if ld_window_size is None:
        return minimum

    ld_window_size = int(ld_window_size)
    if ld_window_size < minimum:
        raise ValueError(
            f"ld_window_size must be >= {minimum} for flank={flank} and "
            f"bin_size={bin_size}; otherwise distal profile windows would "
            "be truncated by the LD distance cutoff"
        )

    return ld_window_size


def get_region_bin_ld(
    chr_id,
    anchor_pos,
    strand,
    vcf_file,
    flank=10000,
    step=10,
    bin_size=500,
    ld_window_size=None,
    exclude_self=True,
    min_pairs=1,
):
    """
    Calculate mean R2 between an anchor window and each sliding window.

    The reference window is centered on the TSS/TTS. For every sliding
    window, all finite pairwise R2 values between variants in the reference
    window and variants in the sliding window are averaged. Identical-site
    self pairs are excluded by default because they have trivial R2=1.
    """
    anchor_pos = int(anchor_pos)
    flank = int(flank)
    step = int(step)
    bin_size = int(bin_size)
    min_pairs = int(min_pairs)

    if min(flank, step, bin_size, min_pairs) < 1:
        raise ValueError("flank, step, bin_size and min_pairs must all be >= 1")

    relative_positions = _relative_window_positions(flank, step, bin_size)
    scan_windows = [
        _relative_to_genomic_window(
            anchor_pos, strand, relative_position, bin_size
        )
        for relative_position in relative_positions
    ]

    reference_relative_position = _anchor_window_relative_position(bin_size)
    reference_start, reference_end = _relative_to_genomic_window(
        anchor_pos, strand, reference_relative_position, bin_size
    )

    region_start = min(reference_start, min(start for start, _ in scan_windows))
    region_end = max(reference_end, max(end for _, end in scan_windows))
    region_sites = _load_variant_sites(
        vcf_file, chr_id, region_start, region_end
    )

    values = np.full(len(relative_positions), np.nan, dtype=float)
    if region_sites.empty:
        return pd.DataFrame(values, index=relative_positions, columns=["LD"])

    reference_sites = region_sites[
        (region_sites["POS"] >= reference_start)
        & (region_sites["POS"] <= reference_end)
    ].copy()

    if reference_sites.empty:
        return pd.DataFrame(values, index=relative_positions, columns=["LD"])

    ld_window_size = _resolve_ld_window_size(
        flank, bin_size, ld_window_size
    )

    ld_matrix = calculate_ld_matrix(
        reference_sites,
        vcf_file,
        ld_window_size=ld_window_size,
        site_info_df2=region_sites,
        output_format="matrix",
        pool_num=1,
        show_progress=False,
    )

    r2 = ld_matrix.to_numpy(dtype=float, copy=True)
    reference_positions = reference_sites["POS"].to_numpy()
    region_positions = region_sites["POS"].to_numpy()

    if exclude_self:
        r2[reference_positions[:, None] == region_positions[None, :]] = np.nan

    # Collapse the reference-site dimension first, retaining pairwise
    # weighting. Prefix sums then make the highly overlapping 10-bp sliding
    # windows cheap to summarize.
    pair_sums = np.nansum(r2, axis=0)
    pair_counts = np.isfinite(r2).sum(axis=0).astype(np.int64)

    order = np.argsort(region_positions)
    region_positions = region_positions[order]
    pair_sums = pair_sums[order]
    pair_counts = pair_counts[order]

    cumulative_sum = np.concatenate(([0.0], np.cumsum(pair_sums)))
    cumulative_count = np.concatenate(([0], np.cumsum(pair_counts)))

    for i, (window_start, window_end) in enumerate(scan_windows):
        left = np.searchsorted(region_positions, window_start, side="left")
        right = np.searchsorted(region_positions, window_end, side="right")
        n_pairs = cumulative_count[right] - cumulative_count[left]

        if n_pairs >= min_pairs:
            r2_sum = cumulative_sum[right] - cumulative_sum[left]
            values[i] = r2_sum / n_pairs

    return pd.DataFrame(values, index=relative_positions, columns=["LD"])


def get_bin_ld_for_many_pos_flank_regions(
    pos_df,
    vcf_file,
    flank=10000,
    step=10,
    bin_size=500,
    threads=8,
    ld_window_size=None,
    exclude_self=True,
    min_pairs=1,
):
    """
    Calculate LD profiles around many genomic positions.

    pos_df must contain CHROM, POS and STRAND. Returned rows are relative
    window positions and columns are anchors. Parallelism is across anchors;
    calculate_ld_matrix() itself therefore runs serially within each task.
    """
    args_dict = {}

    for _, row in pos_df.iterrows():
        chr_id = row["CHROM"]
        pos = int(row["POS"])
        strand = row["STRAND"]
        anchor_id = f"{chr_id}:{pos}_{strand}"

        if anchor_id in args_dict:
            continue

        args_dict[anchor_id] = (
            chr_id,
            pos,
            strand,
            vcf_file,
            flank,
            step,
            bin_size,
            ld_window_size,
            exclude_self,
            min_pairs,
        )

    results = multiprocess_running(
        get_region_bin_ld,
        args_dict,
        threads,
    )

    relative_positions = _relative_window_positions(flank, step, bin_size)
    profile_df = pd.DataFrame(index=relative_positions)

    for anchor_id, result in results.items():
        profile_df[anchor_id] = result["output"]["LD"]

    return profile_df


def build_ld_profiles(
    gene_info,
    vcf_file,
    output_dir,
    flank=10000,
    bin_size=500,
    step=10,
    threads=8,
    ld_window_size=None,
    exclude_self=True,
    min_pairs=1,
    overwrite=False,
):
    """Build TSS and TTS anchor-window LD databases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    for anchor in ("TSS", "TTS"):
        output_file = output_dir / f"{anchor}.LD.h5"
        output_files[anchor] = output_file

        if output_file.exists() and not overwrite:
            continue

        anchor_df = make_anchor_table(gene_info, anchor)
        raw_profile = get_bin_ld_for_many_pos_flank_regions(
            anchor_df,
            vcf_file,
            flank=flank,
            step=step,
            bin_size=bin_size,
            threads=threads,
            ld_window_size=ld_window_size,
            exclude_self=exclude_self,
            min_pairs=min_pairs,
        )
        profile_df = _restore_gene_ids(raw_profile, anchor_df)

        _save_profile(
            profile_df,
            output_file,
            track="LD",
            anchor=anchor,
            flank=flank,
            bin_size=bin_size,
            step=step,
            source=vcf_file,
            extra_metadata={
                "reference_window": "centered_on_anchor",
                "reference_relative_position": int(
                    _anchor_window_relative_position(bin_size)
                ),
                "exclude_self": bool(exclude_self),
                "min_pairs": int(min_pairs),
                "ld_window_size": int(
                    _resolve_ld_window_size(flank, bin_size, ld_window_size)
                ),
                "aggregation": "mean_finite_pairwise_r2",
            },
        )

    return output_files


# ---------------------------------------------------------------------
# Interval feature profiles: ACR and UMR
# ---------------------------------------------------------------------

def _load_intervals(bed_file):
    """Load a BED or gzipped BED file into chromosome-specific InterLap objects."""
    intervals = {}
    bed_file = Path(bed_file)
    opener = gzip.open if bed_file.suffix == ".gz" else open

    with opener(bed_file, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue

            fields = line.rstrip().split("\t")
            if len(fields) < 3:
                continue

            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])

            if chrom not in intervals:
                intervals[chrom] = InterLap()

            # BED uses [start, end); InterLap is queried with closed intervals.
            intervals[chrom].add((start, max(start, end - 1)))

    return intervals


def _relative_to_genomic_window(
    anchor_pos,
    strand,
    relative_position,
    bin_size,
):
    """
    Convert one transcription-oriented relative window to genomic coordinates.

    relative_position follows the same convention as the QRE/pi functions:
    the first window at flank=10000 is -9999.
    """
    if strand == "+":
        window_start = anchor_pos + relative_position - 1
        window_end = window_start + bin_size - 1
    else:
        window_end = anchor_pos - relative_position
        window_start = window_end - bin_size + 1

    return int(window_start), int(window_end)


def build_feature_profiles(
    gene_info,
    bed_file,
    output_dir,
    track_name,
    flank=10000,
    bin_size=500,
    step=10,
    overwrite=False,
):
    """
    Build TSS and TTS binary overlap databases for a BED interval feature.

    Each value is:
        1: the 500-bp window overlaps at least one feature
        0: no overlap
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    relative_positions = _relative_window_positions(
        flank,
        step,
        bin_size,
    )
    intervals = _load_intervals(bed_file)

    output_files = {}

    for anchor in ("TSS", "TTS"):
        output_file = output_dir / f"{anchor}.{track_name}.h5"
        output_files[anchor] = output_file

        if output_file.exists() and not overwrite:
            continue

        values = np.zeros(
            (len(gene_info), len(relative_positions)),
            dtype=np.uint8,
        )

        for i, (_, gene) in enumerate(gene_info.iterrows()):
            chrom = gene["chrom"]

            if chrom not in intervals:
                continue

            anchor_pos = int(gene[anchor])
            strand = gene["strand"]
            chrom_intervals = intervals[chrom]

            for j, relative_position in enumerate(relative_positions):
                window_start, window_end = _relative_to_genomic_window(
                    anchor_pos,
                    strand,
                    relative_position,
                    bin_size,
                )

                values[i, j] = int(
                    any(
                        chrom_intervals.find(
                            (window_start, window_end)
                        )
                    )
                )

        profile_df = pd.DataFrame(
            values,
            index=gene_info.index,
            columns=relative_positions,
        )

        _save_profile(
            profile_df,
            output_file,
            track=track_name,
            anchor=anchor,
            flank=flank,
            bin_size=bin_size,
            step=step,
            source=bed_file,
        )

    return output_files


def build_acr_profiles(
    gene_info,
    acr_bed_file,
    output_dir,
    flank=10000,
    bin_size=500,
    step=10,
    overwrite=False,
):
    """Build TSS and TTS ACR overlap databases."""
    return build_feature_profiles(
        gene_info,
        acr_bed_file,
        output_dir,
        track_name="ACR",
        flank=flank,
        bin_size=bin_size,
        step=step,
        overwrite=overwrite,
    )


def build_umr_profiles(
    gene_info,
    umr_bed_file,
    output_dir,
    flank=10000,
    bin_size=500,
    step=10,
    overwrite=False,
):
    """Build TSS and TTS UMR overlap databases."""
    return build_feature_profiles(
        gene_info,
        umr_bed_file,
        output_dir,
        track_name="UMR",
        flank=flank,
        bin_size=bin_size,
        step=step,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------

def build_gene_profile_database(
    gff_file,
    output_dir,
    qre_sources=None,
    vcf_file=None,
    acr_bed_file=None,
    umr_bed_file=None,
    tracks=("qre", "pi", "acr", "umr"),
    config=None,
    chromosome_filter=None,
    threads=80,
    ld_threads=8,
    ld_window_size=None,
    ld_exclude_self=True,
    ld_min_pairs=1,
    overwrite=False,
):
    """
    Build all requested per-gene profile databases.

    qre_sources example:
        {
            "bio18": "/path/bio18.gemma.gwas_df.h5",
            "gsl": "/path/gsl.gemma.gwas_df.h5",
        }

    config can override DEFAULT_CONFIG, for example:
        {
            "pi": {"flank": 6000}
        }

    LD is supported as track="ld" but is intentionally not included in the
    default track tuple because it is substantially more expensive to build.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        track: values.copy()
        for track, values in DEFAULT_CONFIG.items()
    }

    if config is not None:
        for track, values in config.items():
            if track not in cfg:
                raise ValueError(f"Unknown track in config: {track}")
            cfg[track].update(values)

    tracks = [track.lower() for track in tracks]
    valid_tracks = {"qre", "pi", "acr", "umr", "ld"}

    unknown_tracks = set(tracks) - valid_tracks
    if unknown_tracks:
        raise ValueError(
            f"Unknown tracks: {sorted(unknown_tracks)}"
        )

    gene_info_file = output_dir / "gene_info.h5"

    gene_info = build_gene_info(
        gff_file,
        output_file=gene_info_file,
        chromosome_filter=chromosome_filter,
        overwrite=overwrite,
    )

    outputs = {
        "gene_info": gene_info_file,
    }

    if "qre" in tracks:
        if qre_sources is None:
            raise ValueError(
                "qre_sources is required when building QRE profiles"
            )

        outputs["qre"] = {}

        for label, qre_source in qre_sources.items():
            p = cfg["qre"]

            outputs["qre"][label] = build_qre_profiles(
                gene_info,
                qre_source,
                output_dir,
                label=label,
                flank=p["flank"],
                bin_size=p["bin_size"],
                step=p["step"],
                threads=threads,
                overwrite=overwrite,
            )

    if "pi" in tracks:
        if vcf_file is None:
            raise ValueError(
                "vcf_file is required when building pi profiles"
            )

        p = cfg["pi"]

        outputs["pi"] = build_pi_profiles(
            gene_info,
            vcf_file,
            output_dir,
            flank=p["flank"],
            bin_size=p["bin_size"],
            step=p["step"],
            threads=threads,
            overwrite=overwrite,
        )

    if "ld" in tracks:
        if vcf_file is None:
            raise ValueError(
                "vcf_file is required when building LD profiles"
            )

        p = cfg["ld"]

        outputs["ld"] = build_ld_profiles(
            gene_info,
            vcf_file,
            output_dir,
            flank=p["flank"],
            bin_size=p["bin_size"],
            step=p["step"],
            threads=ld_threads,
            ld_window_size=ld_window_size,
            exclude_self=ld_exclude_self,
            min_pairs=ld_min_pairs,
            overwrite=overwrite,
        )

    if "acr" in tracks:
        if acr_bed_file is None:
            raise ValueError(
                "acr_bed_file is required when building ACR profiles"
            )

        p = cfg["acr"]

        outputs["acr"] = build_acr_profiles(
            gene_info,
            acr_bed_file,
            output_dir,
            flank=p["flank"],
            bin_size=p["bin_size"],
            step=p["step"],
            overwrite=overwrite,
        )

    if "umr" in tracks:
        if umr_bed_file is None:
            raise ValueError(
                "umr_bed_file is required when building UMR profiles"
            )

        p = cfg["umr"]

        outputs["umr"] = build_umr_profiles(
            gene_info,
            umr_bed_file,
            output_dir,
            flank=p["flank"],
            bin_size=p["bin_size"],
            step=p["step"],
            overwrite=overwrite,
        )

    return outputs


if __name__ == "__main__":
    # Example: build all gene-centered profiles with the default settings
    # (flank=10 kb, bin_size=500 bp, step=10 bp).
    project_dir = Path("/path/to/project")

    gff_file = project_dir / "genome.gff3"
    vcf_file = project_dir / "genotype.vcf.gz"
    acr_bed_file = project_dir / "ACR.bed.gz"
    umr_bed_file = project_dir / "UMR.bed.gz"
    output_dir = project_dir / "gene_profiles"

    qre_sources = {
        "bio18": project_dir / "bio18.gemma.gwas_df.h5",
        "gsl": project_dir / "gsl.gemma.gwas_df.h5",
    }

    outputs = build_gene_profile_database(
        gff_file=gff_file,
        output_dir=output_dir,
        qre_sources=qre_sources,
        vcf_file=vcf_file,
        acr_bed_file=acr_bed_file,
        umr_bed_file=umr_bed_file,
        tracks=("qre", "pi", "acr", "umr", "ld"),
        threads=80,
        ld_threads=16,
        overwrite=False,
    )

    print(outputs)
