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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from cyvcf2 import VCF
from interlap import InterLap

from .feature import get_gene_range_df, read_gff_file
from .ld import calculate_ld_matrix
from .utils import multiprocess_running
from .plot import set_plot_style


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

def _get_gene_utr_lengths(gff_file, gene_ids):
    """Return 5' and 3' UTR lengths using the first transcript of each gene.

    This follows the original intragene notebook definition: the 5' UTR is
    the distance from TSS to the first CDS boundary, and the 3' UTR is the
    distance from the last CDS boundary to TTS. Genes without a usable first
    transcript/CDS are returned as NaN.
    """
    gene_dict = read_gff_file(gff_file).get("gene", {})
    utr_5 = {}
    utr_3 = {}

    for gene_id in gene_ids:
        gene = gene_dict.get(gene_id)
        if gene is None:
            utr_5[gene_id] = np.nan
            utr_3[gene_id] = np.nan
            continue

        transcripts = getattr(gene, "sub_features", None) or []
        if not transcripts:
            utr_5[gene_id] = np.nan
            utr_3[gene_id] = np.nan
            continue

        transcript = transcripts[0]
        sub_features = getattr(transcript, "sub_features", None) or []
        cds_features = [feature for feature in sub_features if feature.type == "CDS"]
        if not cds_features:
            utr_5[gene_id] = np.nan
            utr_3[gene_id] = np.nan
            continue

        if gene.strand == "+":
            tss, tts = gene.start, gene.end
            cds_start = min(feature.start for feature in cds_features)
            cds_end = max(feature.end for feature in cds_features)
        else:
            tss, tts = gene.end, gene.start
            cds_start = max(feature.end for feature in cds_features)
            cds_end = min(feature.start for feature in cds_features)

        utr_5[gene_id] = abs(cds_start - tss)
        utr_3[gene_id] = abs(tts - cds_end)

    return pd.Series(utr_5, dtype=float), pd.Series(utr_3, dtype=float)


def build_gene_info(
    gff_file,
    output_file=None,
    chromosome_filter=None,
    overwrite=False,
):
    """Build the gene table shared by all profile databases.

    Gene coordinates are loaded through feature.get_gene_range_df(), while
    UTR lengths reuse feature.read_gff_file() and the first transcript/CDS
    structure used by the original intragene notebook. TSS/TTS are derived
    from gene START/END according to transcription strand.
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

    gene_range_df["GENE"] = gene_range_df["GENE"].astype(str)
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

    utr_5, utr_3 = _get_gene_utr_lengths(
        gff_file, gene_range_df["GENE"].tolist()
    )
    gene_range_df["5utr_length"] = gene_range_df["GENE"].map(utr_5)
    gene_range_df["3utr_length"] = gene_range_df["GENE"].map(utr_3)

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
        [
            "chrom", "strand", "start", "end", "TSS", "TTS",
            "gene_length", "5utr_length", "3utr_length",
        ]
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
        raise ValueError(
            "flank, step, bin_size and min_pairs must all be >= 1")

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

    region_start = min(reference_start, min(
        start for start, _ in scan_windows))
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


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
PLOT_TRACK_ALIASES = {
    "qre": "QRE",
    "pi": "pi",
    "acr": "ACR",
    "acrs": "ACR",
    "umr": "UMR",
    "umrs": "UMR",
    "ld": "LD",
    "high_effect_index": "high_effect_index",
}


def _normalize_plot_track(track):
    key = str(track).strip().lower()
    if key not in PLOT_TRACK_ALIASES:
        raise ValueError(
            f"Unknown plot track: {track}. Supported tracks are "
            "'pi', 'ACR', 'UMR', 'LD' and 'high_effect_index'."
        )
    return PLOT_TRACK_ALIASES[key]


def _get_plot_profile_path(profile_db, env_name, track, anchor):
    """Resolve one profile file from build_gene_profile_database() output."""
    anchor = anchor.upper()
    if anchor not in {"TSS", "TTS"}:
        raise ValueError("anchor must be 'TSS' or 'TTS'")

    if track == "QRE":
        try:
            return Path(profile_db["qre"][env_name][anchor])
        except KeyError as exc:
            raise KeyError(
                f"QRE profile for environment '{env_name}' and {anchor} is "
                "not present in profile_db"
            ) from exc

    output_key = {"pi": "pi", "ACR": "acr", "UMR": "umr", "LD": "ld"}[track]
    try:
        return Path(profile_db[output_key][anchor])
    except KeyError as exc:
        raise KeyError(
            f"{track} profile for {anchor} is not present in profile_db"
        ) from exc


def _load_gene_info_from_profile_db(profile_db):
    if "gene_info" not in profile_db:
        raise KeyError("profile_db must contain the 'gene_info' output")
    return pd.read_hdf(profile_db["gene_info"], key="gene_info")


def _prepare_plot_gene_sets(
    magma_gwas_df_h5,
    gene_universe,
    top_gene_quantile=0.05,
    random_num=1000,
    seed=42,
):
    """Select top MAGMA genes and matched-size random sets from all other genes."""
    top_gene_quantile = float(top_gene_quantile)
    random_num = int(random_num)
    if not 0 < top_gene_quantile < 1:
        raise ValueError("top_gene_quantile must be between 0 and 1")
    if random_num < 1:
        raise ValueError("random_num must be >= 1")

    magma_df = pd.read_hdf(magma_gwas_df_h5, key="df")
    required = {"ID", "P_VALUE"}
    missing = required - set(magma_df.columns)
    if missing:
        raise ValueError(
            "MAGMA GWAS dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    magma_df = magma_df[["ID", "P_VALUE"]].dropna().copy()
    magma_df["ID"] = magma_df["ID"].astype(str)
    magma_df = magma_df.sort_values(
        "P_VALUE", kind="stable").drop_duplicates("ID")

    gene_universe = pd.Index(gene_universe).astype(str)
    magma_df = magma_df[magma_df["ID"].isin(gene_universe)].copy()
    if magma_df.empty:
        raise ValueError("No MAGMA genes overlap the gene profile database")

    top_gene_num = max(1, int(len(magma_df) * top_gene_quantile))
    top_genes = magma_df.head(top_gene_num)["ID"].to_numpy()
    background_genes = magma_df.loc[
        ~magma_df["ID"].isin(top_genes), "ID"
    ].to_numpy()

    if len(background_genes) < top_gene_num:
        raise ValueError(
            f"Only {len(background_genes)} non-top genes are available, but "
            f"each random set requires {top_gene_num} genes"
        )

    rng = np.random.default_rng(seed)
    random_indices = np.empty((random_num, top_gene_num), dtype=np.int32)
    for i in range(random_num):
        random_indices[i] = rng.choice(
            len(background_genes), size=top_gene_num, replace=False
        )

    return {
        "top_genes": top_genes,
        "background_genes": background_genes,
        "random_indices": random_indices,
        "top_gene_num": top_gene_num,
        "eligible_gene_num": len(magma_df),
    }


def _nanmean_axis1(values):
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    sums = np.nansum(values, axis=1)
    return np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=float),
        where=counts > 0,
    )


def _summarize_profile_for_plot(
    profile_file,
    gene_sets,
    plot_xlim,
    ci_percentiles=(5, 95),
    random_batch_size=8,
):
    """Summarize one gene x position profile for top and random gene sets."""
    profile_df, metadata = read_profile(profile_file)
    profile_df.index = profile_df.index.astype(str)
    profile_df.columns = pd.Index(
        [int(x) for x in profile_df.columns], name="relative_position"
    )

    bin_size = int(metadata.get("bin_size", 500))
    x = profile_df.columns.to_numpy(dtype=float) + bin_size / 2
    keep = (x >= plot_xlim[0]) & (x <= plot_xlim[1])
    if not keep.any():
        raise ValueError(
            f"No profile windows in {profile_file} overlap plot_xlim={plot_xlim}"
        )

    profile_df = profile_df.loc[:, keep]
    x = x[keep]
    top_genes = gene_sets["top_genes"]
    background_genes = gene_sets["background_genes"]

    missing_top = pd.Index(top_genes).difference(profile_df.index)
    missing_background = pd.Index(
        background_genes).difference(profile_df.index)
    if len(missing_top) or len(missing_background):
        raise ValueError(
            f"Profile {profile_file} does not contain the complete MAGMA gene "
            f"universe ({len(missing_top)} top and {len(missing_background)} "
            "background genes missing)"
        )

    top_values = profile_df.loc[top_genes].to_numpy(dtype=float, copy=False)
    top_mean = _nanmean_axis1(top_values.T)

    background_values = profile_df.loc[background_genes].to_numpy(
        dtype=float, copy=False
    )
    random_indices = gene_sets["random_indices"]
    random_num = len(random_indices)
    random_batch_size = max(1, int(random_batch_size))
    random_means = np.full((random_num, len(x)), np.nan, dtype=float)

    for start in range(0, random_num, random_batch_size):
        end = min(start + random_batch_size, random_num)
        block = background_values[random_indices[start:end]]
        finite = np.isfinite(block)
        counts = finite.sum(axis=1)
        sums = np.nansum(block, axis=1)
        random_means[start:end] = np.divide(
            sums,
            counts,
            out=np.full(sums.shape, np.nan, dtype=float),
            where=counts > 0,
        )

    with np.errstate(invalid="ignore"):
        random_low = np.nanpercentile(random_means, ci_percentiles[0], axis=0)
        random_high = np.nanpercentile(random_means, ci_percentiles[1], axis=0)

    return {
        "x": x,
        "top_mean": top_mean,
        "random_low": random_low,
        "random_high": random_high,
        "metadata": metadata,
    }


def load_env_track_data(
    env_name,
    magma_gwas_df_h5,
    config,
    top_gene_quantile=0.05,
    random_num=1000,
    seed=42,
    track_order=None,
):
    """Load and summarize the profile database for one GEA environment."""
    profile_db = config["profile_db"]
    plot_xlim = tuple(config.get("plot_xlim", (-5500, 5500)))
    ci_percentiles = tuple(config.get("ci_percentiles", (5, 95)))
    random_batch_size = config.get("random_batch_size", 8)

    if track_order is None:
        track_order = config.get("track_order", ("pi", "ACR", "UMR", "LD"))
    track_order = [_normalize_plot_track(track) for track in track_order]
    if "QRE" in track_order:
        raise ValueError(
            "QRE is plotted automatically and must not be in track_order")
    if len(track_order) != len(set(track_order)):
        raise ValueError("track_order contains duplicate tracks")

    gene_info = _load_gene_info_from_profile_db(profile_db)
    gene_info.index = gene_info.index.astype(str)
    gene_sets = _prepare_plot_gene_sets(
        magma_gwas_df_h5,
        gene_info.index,
        top_gene_quantile=top_gene_quantile,
        random_num=random_num,
        seed=seed,
    )

    data = {"gene_info": gene_info, "gene_sets": gene_sets}
    database_tracks = ["QRE"] + [
        track for track in track_order if track != "high_effect_index"
    ]

    for track in database_tracks:
        data[track] = {}
        for anchor in ("TSS", "TTS"):
            profile_file = _get_plot_profile_path(
                profile_db, env_name, track, anchor
            )
            data[track][anchor] = _summarize_profile_for_plot(
                profile_file,
                gene_sets,
                plot_xlim=plot_xlim,
                ci_percentiles=ci_percentiles,
                random_batch_size=random_batch_size,
            )

    return data, track_order


def _set_left_track_label(ax, label, config):
    ax.set_ylabel(
        label,
        fontsize=config.get("label_fontsize", 11),
        rotation=config.get("left_track_label_rotation", 90),
        ha=config.get("left_track_label_ha", "center"),
        va=config.get("left_track_label_va", "center"),
    )
    ax.yaxis.set_label_coords(config.get("left_track_label_x", -0.15), 0.5)


def _hide_top_right_bottom_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def _hide_right_panel_y_axis(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(
        axis="y", which="both", left=False, right=False,
        labelleft=False, labelright=False,
    )


def _hide_x_axis(ax):
    ax.tick_params(axis="x", which="both", bottom=False,
                   top=False, labelbottom=False)


def _add_site_line(ax, site_label, config):
    ax.axvline(
        0,
        color=config.get("site_line_color", "red"),
        linestyle=config.get("site_line_style", "--"),
        linewidth=config.get("site_line_width", 1),
        label=site_label,
    )


def _set_site_xticks(ax, site_label, config):
    ax.tick_params(axis="x", which="both", bottom=True,
                   top=False, labelbottom=True)
    ax.xaxis.set_major_locator(
        ticker.MultipleLocator(config.get("x_tick_interval", 2500))
    )
    ticks = ax.get_xticks()
    labels = [site_label if np.isclose(
        tick, 0) else str(int(tick)) for tick in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _add_scale_bar(ax, config):
    xlim = config.get("plot_xlim", (-5500, 5500))
    bar_frac = config.get("scale_bp", 500) / (xlim[1] - xlim[0])
    x1 = config.get("scale_bar_right", 0.72)
    x0 = x1 - bar_frac
    y = config.get("scale_bar_y", 0.95)
    ax.plot(
        [x0, x1], [y, y], transform=ax.transAxes,
        color="black", linewidth=config.get("scale_bar_linewidth", 2),
        solid_capstyle="butt", clip_on=False,
    )
    ax.text(
        (x0 + x1) / 2,
        y + config.get("scale_bar_label_offset", 0.025),
        config.get("scale_bar_label", "500-bp (10-bp step)"),
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=config.get("scale_fontsize", 9),
        clip_on=False,
    )


DEFAULT_LENGTH_BAR_CMAP = LinearSegmentedColormap.from_list(
    "length_bar_cmap", ["#FFFFFF", "#5580B0"]
)


def _plot_length_bar(
    fig,
    anchor_ax,
    gene_info_df,
    feature_col,
    label,
    y,
    invert_x,
    config,
):
    """Plot the fraction of top genes whose feature length reaches each bp."""
    if feature_col not in gene_info_df.columns:
        raise ValueError(
            f"gene_info is missing '{feature_col}'. Rebuild gene_info.h5 "
            "with the current gene_profiles.py before plotting."
        )

    values = pd.to_numeric(gene_info_df[feature_col], errors="coerce")
    values = values[np.isfinite(values)].to_numpy(dtype=float)
    if len(values) == 0:
        return

    length_bar_max = int(config.get("length_bar_max", 5500))
    site_x = anchor_ax.transData.transform((0, 0))[0]
    site_x = fig.transFigure.inverted().transform((site_x, 0))[0]
    target_x = -length_bar_max if invert_x else length_bar_max
    bar_end = anchor_ax.transData.transform((target_x, 0))[0]
    bar_end = fig.transFigure.inverted().transform((bar_end, 0))[0]
    bar_length = abs(bar_end - site_x)

    lengths = np.arange(length_bar_max)
    dist = (values[:, None] >= lengths).mean(axis=0)
    bar_y = float(y)
    bar_height = config.get("length_bar_height", 0.02)
    if invert_x:
        ax_bar = fig.add_axes([site_x - bar_length, bar_y, bar_length, bar_height])
    else:
        ax_bar = fig.add_axes([site_x, bar_y, bar_length, bar_height])

    ax_bar.set_axis_off()
    ax_bar.imshow(
        dist.reshape(1, -1),
        aspect="auto",
        cmap=config.get("length_bar_cmap", DEFAULT_LENGTH_BAR_CMAP),
        extent=[0, length_bar_max, 0, 1],
        vmin=0,
        vmax=1,
    )
    label_offset = config.get("length_bar_label_offset_bp", 200)
    ax_bar.text(
        -label_offset,
        0.6,
        label,
        va="center",
        ha="left" if invert_x else "right",
        fontsize=config.get("bar_label_fontsize", 10),
        clip_on=False,
    )
    median_target = config.get("length_bar_median_target", 0.5)
    median_idx = int(np.argmin(np.abs(dist - median_target)))
    ax_bar.axvline(
        median_idx,
        color="black",
        linewidth=config.get("length_bar_median_linewidth", 1),
    )
    if invert_x:
        ax_bar.invert_xaxis()


def _add_gene_utr_bars(fig, qre_tss_ax, qre_tts_ax, data, config):
    """Add gene-length and 5'/3' UTR distributions for the top GEA genes."""
    if not config.get("show_length_bars", True):
        return

    top_genes = pd.Index(data["gene_sets"]["top_genes"]).astype(str)
    gene_info = data["gene_info"].copy()
    gene_info.index = gene_info.index.astype(str)
    top_gene_info = gene_info.reindex(top_genes)

    _plot_length_bar(
        fig, qre_tss_ax, top_gene_info,
        feature_col="gene_length", label="Gene length",
        y=config.get("gene_bar_y", 0.65), invert_x=False, config=config,
    )
    _plot_length_bar(
        fig, qre_tss_ax, top_gene_info,
        feature_col="5utr_length", label="5' UTR length",
        y=config.get("utr_bar_y", 0.61), invert_x=False, config=config,
    )
    _plot_length_bar(
        fig, qre_tts_ax, top_gene_info,
        feature_col="gene_length", label="Gene length",
        y=config.get("gene_bar_y", 0.65), invert_x=True, config=config,
    )
    _plot_length_bar(
        fig, qre_tts_ax, top_gene_info,
        feature_col="3utr_length", label="3' UTR length",
        y=config.get("utr_bar_y", 0.61), invert_x=True, config=config,
    )


def _plot_profile_track(
    ax_left,
    ax_right,
    track_data,
    label,
    config,
    is_bottom_track=False,
    show_gene_set_labels=False,
):
    main_color = config.get("main_color", "#5580B0")
    random_color = config.get("random_color", "gray")
    random_alpha = config.get("random_alpha", 0.2)

    for ax, anchor in ((ax_left, "TSS"), (ax_right, "TTS")):
        summary = track_data[anchor]
        ax.plot(summary["x"], summary["top_mean"], color=main_color)
        ax.fill_between(
            summary["x"], summary["random_low"], summary["random_high"],
            color=random_color, alpha=random_alpha, edgecolor="none",
        )
        _add_site_line(ax, anchor, config)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _set_left_track_label(ax_left, label, config)
    _hide_right_panel_y_axis(ax_right)

    if show_gene_set_labels:
        top_pct = config["_top_gene_quantile"] * 100
        random_num = config["_random_num"]
        ci_low, ci_high = config.get("ci_percentiles", (5, 95))
        ax_left.text(
            0.04, 0.90, f"Top {top_pct:g}% GEA genes",
            transform=ax_left.transAxes, ha="left", va="top",
            fontsize=config.get("legend_fontsize", 10), color=main_color,
        )
        ax_left.text(
            0.04, 0.80,
            f"Random gene sets\n({ci_low}–{ci_high}%)",
            transform=ax_left.transAxes, ha="left", va="top",
            fontsize=config.get("legend_fontsize", 10), color=random_color,
        )

    if is_bottom_track:
        _set_site_xticks(ax_left, "TSS", config)
        _set_site_xticks(ax_right, "TTS", config)
    else:
        ax_left.spines["bottom"].set_visible(False)
        ax_right.spines["bottom"].set_visible(False)
        _hide_x_axis(ax_left)
        _hide_x_axis(ax_right)


def _qre_label(env_desc):
    if "striga" in env_desc.lower():
        return "Relative GEA signal for\n" + r"$\mathit{Striga}$ HS score"
    return f"Relative GEA signal for\n{env_desc.lower()}"


def _load_high_effect_index(config):
    if "high_effect_index_csv" not in config:
        raise KeyError(
            "high_effect_index_csv is required when high_effect_index is plotted"
        )
    df = pd.read_csv(config["high_effect_index_csv"]).sort_values("center")
    xlim = config.get("plot_xlim", (-5500, 5500))
    return df[df["center"].between(*xlim)].copy()


def _plot_high_effect_index_track(ax_left, ax_right, high_effect_df, config, is_bottom_track=False):
    ax_left.plot(
        high_effect_df["center"],
        high_effect_df["relative_probability_index"],
        color=config.get("high_effect_color",
                         config.get("main_color", "#5580B0")),
    )
    ax_left.axhline(
        1,
        color=config.get("baseline_color", "gray"),
        linestyle=config.get("baseline_line_style", ":"),
        linewidth=config.get("baseline_line_width", 1),
    )
    _add_site_line(ax_left, "TSS", config)
    _add_site_line(ax_right, "TTS", config)
    _set_left_track_label(ax_left, "High-effect\nindex", config)
    _hide_right_panel_y_axis(ax_right)

    if config.get("show_high_effect_annotation", True):
        ax_left.figure.text(
            *config.get("high_effect_annotation_fig_xy", (0.35, 0.19)),
            config.get(
                "high_effect_annotation_text",
                "Frequency of high impact CRISPR mutations",
            ),
            fontsize=config.get("high_effect_annotation_fontsize", 9),
            color=config.get("high_effect_annotation_color", "#000000"),
            ha=config.get("high_effect_annotation_ha", "left"),
            va=config.get("high_effect_annotation_va", "center"),
        )

    for ax in (ax_left, ax_right):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if is_bottom_track:
        _set_site_xticks(ax_left, "TSS", config)
        _set_site_xticks(ax_right, "TTS", config)
    else:
        ax_left.spines["bottom"].set_visible(False)
        ax_right.spines["bottom"].set_visible(False)
        _hide_x_axis(ax_left)
        _hide_x_axis(ax_right)


DEFAULT_PLOT_CONFIG = {
    # ------------------------------------------------------------
    # Profile database
    # ------------------------------------------------------------
    # build_gene_profile_database() 的返回值
    # "profile_db": outputs,

    # QRE 固定画在第一行，不需要写进这里
    # 可选: "pi", "ACR", "UMR", "LD", "high_effect_index"
    "track_order": ["LD", "pi", "ACR", "UMR", "high_effect_index"],

    # 数据库可以计算到 ±10 kb，这里只控制实际读取/绘制的范围
    "plot_xlim": (-5500, 5500),

    # ------------------------------------------------------------
    # Random gene sets
    # ------------------------------------------------------------
    # 随机基因集 percentile 区间
    "ci_percentiles": (5, 95),

    # 一次处理多少个 random sets，主要控制内存
    # 不影响最终结果
    "random_batch_size": 8,

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    "fig_size": (11, 6.5),

    # QRE 第一行相对更高
    "qre_height_ratio": 3,

    # 其余 track 的统一高度
    "track_height_ratio": 1,

    # TSS / TTS 两列之间的距离
    "grid_wspace": 0.05,

    # tracks 之间的距离
    "grid_hspace": 0.20,

    # ------------------------------------------------------------
    # Main profile style
    # ------------------------------------------------------------
    # Top GEA genes
    "main_color": "#5580B0",

    # Random gene sets CI
    "random_color": "gray",
    "random_alpha": 0.20,

    # ------------------------------------------------------------
    # TSS / TTS vertical reference line
    # ------------------------------------------------------------
    "site_line_color": "red",
    "site_line_style": "--",
    "site_line_width": 1,

    # ------------------------------------------------------------
    # Y-axis track labels
    # ------------------------------------------------------------
    "label_fontsize": 11,

    # 左侧 ylabel 的位置
    "left_track_label_x": -0.15,

    "left_track_label_rotation": 90,
    "left_track_label_ha": "center",
    "left_track_label_va": "center",

    # ------------------------------------------------------------
    # Top/random annotation
    # ------------------------------------------------------------
    "legend_fontsize": 10,

    # ------------------------------------------------------------
    # X axis
    # ------------------------------------------------------------
    "x_tick_interval": 2500,

    # ------------------------------------------------------------
    # Scale bar
    # ------------------------------------------------------------
    # scale bar 表示的 genomic length
    "scale_bp": 500,

    "scale_bar_label": "500-bp\n(10-bp step)",

    # "TSS" 或 "TTS"
    "scale_bar_panel": "TSS",

    # axes fraction coordinates
    "scale_bar_y": 0.95,
    "scale_bar_right": 0.72,

    "scale_bar_linewidth": 2,
    "scale_bar_label_offset": 0.025,
    "scale_fontsize": 9,

    # ------------------------------------------------------------
    # Gene / UTR length bars
    # ------------------------------------------------------------
    "show_length_bars": True,
    "length_bar_max": 5500,
    "length_bar_height": 0.02,
    "gene_bar_y": 0.65,
    "utr_bar_y": 0.61,
    "length_bar_label_offset_bp": 200,
    "length_bar_median_target": 0.5,
    "length_bar_median_linewidth": 1,
    "bar_label_fontsize": 10,
    "length_bar_cmap": DEFAULT_LENGTH_BAR_CMAP,

    # ------------------------------------------------------------
    # Optional high-effect index track
    # ------------------------------------------------------------
    # 只有 track_order 中加入 "high_effect_index" 才会使用
    #
    # "high_effect_index_csv":
    #     project_dir / "high_effect_index.csv",

    "high_effect_color": "#5580B0",

    # relative_probability_index = 1 的 baseline
    "baseline_color": "gray",
    "baseline_line_style": ":",
    "baseline_line_width": 1,

    # high-effect track 上的说明文字
    "show_high_effect_annotation": True,

    "high_effect_annotation_text": (
        'Frequency of high impact CRISPR mutations\n'
        r'(Groover $\it{et}$ $\it{al.,}$ 2026)'
    ),

    # figure coordinates，不是 axes coordinates
    "high_effect_annotation_fig_xy": (0.35, 0.19),

    "high_effect_annotation_fontsize": 9,
    "high_effect_annotation_color": "#000000",
    "high_effect_annotation_ha": "left",
    "high_effect_annotation_va": "center",

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    # "plot_output_dir": project_dir / "figures",

    "save_fig": False,
    "show_fig": True,

    "save_kwargs": {
        "format": "pdf",
        "facecolor": "none",
        "edgecolor": "none",
        "bbox_inches": "tight",
    },
}


def plot_env_tracks(
    env_name,
    env_desc,
    magma_gwas_df_h5,
    config,
    top_gene_quantile=0.05,
    random_num=1000,
    seed=42,
    track_order=None,
    save_path=None,
):
    """
    Plot TSS/TTS tracks from build_gene_profile_database() output.

    Top genes are selected from magma_gwas_df_h5 by ascending P_VALUE.
    Each random set contains the same number of genes sampled without
    replacement from the remaining non-top MAGMA genes. The same random
    sets are used for every anchor and every track.
    """
    set_plot_style()

    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    plot_config = dict(config)
    plot_config["_top_gene_quantile"] = float(top_gene_quantile)
    plot_config["_random_num"] = int(random_num)

    data, track_order = load_env_track_data(
        env_name,
        magma_gwas_df_h5,
        plot_config,
        top_gene_quantile=top_gene_quantile,
        random_num=random_num,
        seed=seed,
        track_order=track_order,
    )

    all_rows = ["QRE"] + list(track_order)
    n_rows = len(all_rows)
    height_ratios = [plot_config.get("qre_height_ratio", 3)] + [
        plot_config.get("track_height_ratio", 1)
    ] * (n_rows - 1)

    fig = plt.figure(figsize=plot_config.get("fig_size", (11, 6)))
    gs = gridspec.GridSpec(
        n_rows,
        2,
        height_ratios=height_ratios,
        wspace=plot_config.get("grid_wspace", 0.05),
        hspace=plot_config.get("grid_hspace", 0.1),
    )

    axes = {}
    for row_idx, track in enumerate(all_rows):
        if row_idx == 0:
            left_ax = fig.add_subplot(gs[row_idx, 0])
            right_ax = fig.add_subplot(gs[row_idx, 1], sharey=left_ax)
        else:
            left_ax = fig.add_subplot(gs[row_idx, 0], sharex=axes["QRE"][0])
            right_ax = fig.add_subplot(
                gs[row_idx, 1], sharex=axes["QRE"][1], sharey=left_ax
            )
        axes[track] = (left_ax, right_ax)

    _plot_profile_track(
        *axes["QRE"],
        data["QRE"],
        _qre_label(env_desc),
        plot_config,
        is_bottom_track=(n_rows == 1),
        show_gene_set_labels=True,
    )

    scale_bar_panel = plot_config.get("scale_bar_panel", "TSS")
    _add_scale_bar(
        axes["QRE"][0 if scale_bar_panel == "TSS" else 1],
        plot_config,
    )

    track_labels = {
        "pi": "π",
        "ACR": "ACRs",
        "UMR": "UMRs",
        # "LD": r"LD ($r^2$)",
        "LD": ("LD\n" r"($r^2$)"),
    }
    high_effect_df = None

    for i, track in enumerate(track_order):
        is_bottom = i == len(track_order) - 1
        if track == "high_effect_index":
            if high_effect_df is None:
                high_effect_df = _load_high_effect_index(plot_config)
            _plot_high_effect_index_track(
                *axes[track], high_effect_df, plot_config,
                is_bottom_track=is_bottom,
            )
        else:
            _plot_profile_track(
                *axes[track],
                data[track],
                track_labels[track],
                plot_config,
                is_bottom_track=is_bottom,
            )

    for ax in axes["QRE"]:
        ax.set_xlim(*plot_config.get("plot_xlim", (-5500, 5500)))

    _add_gene_utr_bars(
        fig, axes["QRE"][0], axes["QRE"][1], data, plot_config
    )

    if save_path is None and plot_config.get("save_fig", True):
        output_dir = Path(plot_config.get("plot_output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{env_name}.gene_profiles.pdf"

    if plot_config.get("save_fig", True):
        save_kwargs = plot_config.get(
            "save_kwargs",
            dict(format="pdf", facecolor="none",
                 edgecolor="none", bbox_inches="tight"),
        )
        fig.savefig(save_path, **save_kwargs)
    if plot_config.get("show_fig", True):
        plt.show()

    return fig, axes


if __name__ == "__main__":
    # Example 1: build all gene-centered profile databases.
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

    # Example 2: plot one environment from the database returned above.
    # The database is calculated to +/-10 kb, while plot_xlim selects the
    # displayed range here without rebuilding the profiles.
    plot_config = {
        "profile_db": outputs,
        "track_order": ["pi", "ACR", "UMR", "LD"],
        "plot_xlim": (-5500, 5500),
        "plot_output_dir": project_dir / "figures",
        "fig_size": (11, 6.5),
        "qre_height_ratio": 3,
        "track_height_ratio": 1,
        "grid_wspace": 0.05,
        "grid_hspace": 0.10,
        "main_color": "#5580B0",
        "random_color": "gray",
        "random_alpha": 0.2,
        "site_line_color": "red",
        "site_line_style": "--",
        "site_line_width": 1,
        "label_fontsize": 11,
        "legend_fontsize": 10,
        "left_track_label_x": -0.15,
        "x_tick_interval": 2500,
        "ci_percentiles": (5, 95),
        "random_batch_size": 8,
        "scale_bp": 500,
        "scale_bar_label": "500-bp (10-bp step)",
        "scale_bar_panel": "TSS",
        "scale_bar_y": 0.95,
        "scale_bar_right": 0.72,
        "show_length_bars": True,
        "length_bar_max": 5500,
        "length_bar_height": 0.02,
        "gene_bar_y": 0.65,
        "utr_bar_y": 0.61,
        "length_bar_label_offset_bp": 200,
        "length_bar_median_target": 0.5,
        "length_bar_median_linewidth": 1,
        "bar_label_fontsize": 10,
        "length_bar_cmap": DEFAULT_LENGTH_BAR_CMAP,
        "save_fig": True,
        "show_fig": True,
        "save_kwargs": dict(
            format="pdf", facecolor="none", edgecolor="none", bbox_inches="tight"
        ),
    }

    fig, axes = plot_env_tracks(
        env_name="bio18",
        env_desc="Precipitation of Warmest Quarter",
        magma_gwas_df_h5=project_dir / "bio18.magma.gwas_df.h5",
        config=plot_config,
        top_gene_quantile=0.05,
        random_num=1000,
        seed=42,
    )
