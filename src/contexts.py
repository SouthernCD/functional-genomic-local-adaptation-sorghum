import time
import os
from datetime import datetime
import gzip
import random
import pandas as pd
import numpy as np
from cyvcf2 import VCF
from tqdm.auto import tqdm
import statsmodels.api as sm
from scipy import interpolate
import allel
import igraph as ig
from interlap import InterLap
from .interval import interval_minus_set, merge_intervals
from .feature import read_gff_file, read_fasta_by_faidx
from .vcf import get_chr_id_list_from_vcf


# site level contexts

def get_chr_list_from_var_stat_h5(var_stat_h5_file, max_retries=5):
    """
    Read legacy chromosome keys in HDF order with the existing retry policy.

    Parameters
    ----------
    var_stat_h5_file : str
        Existing chromosome-keyed variant statistics HDF.
    max_retries : int
        Maximum read attempts; failed attempts retain historical random waits.

    Returns
    -------
    list of str
        HDF chromosome keys without the leading slash. Opens read-only;
        does not create files or validate completeness of chromosome tables.
    """
    for attempt in range(max_retries):
        try:
            with pd.HDFStore(var_stat_h5_file, mode='r') as store:
                chr_list = store.keys()
            chr_list = [chr_id.split('/')[1] for chr_id in chr_list]
            return chr_list
        except:
            print(f"Attempt {attempt + 1} failed")
            time.sleep(random.uniform(1, 10))
    raise Exception(f"Failed to read HDF5 file after {max_retries} attempts")


def load_site_info_df(site_info_df_h5):
    """
    Load chromosome-keyed variant statistics without changing row order.

    Parameters
    ----------
    site_info_df_h5 : str
        Legacy HDF produced by src.vcf.build_var_stat_table; one key per nonempty chromosome.

    Returns
    -------
    pandas.DataFrame
        Concatenated chromosome tables with reset integer index. Reads HDF keys in their
        existing order; no files are written.
    """
    chr_id_list = get_chr_list_from_var_stat_h5(site_info_df_h5)
    site_info_df = pd.DataFrame()
    for chr_id in chr_id_list:
        chr_site_info_df = pd.read_hdf(site_info_df_h5, key=chr_id)
        site_info_df = pd.concat([site_info_df, chr_site_info_df])
    site_info_df.reset_index(drop=True, inplace=True)
    return site_info_df


def get_gene_struc_contexts(site_contexts_df, ref_gff_file, gene_flank_size=2000, structure_types=['upstream_2kb', "five_prime_UTR", 'CDS', 'intron', "three_prime_UTR", 'downstream_2kb']):
    """
    Annotate sites with strand-aware longest-transcript structure.

    Parameters
    ----------
    site_contexts_df : pandas.DataFrame
        Site table with CHROM and 1-based POS; updated in place.
    ref_gff_file : str
        GFF/GFF.gz parsed into 1-based inclusive features; requires longest=1 transcripts.
    gene_flank_size : int
        Upstream/downstream length in bp; default 2000.
    structure_types : sequence of str
        Requested standard structure names; default names retain the 2kb suffix.

    Returns
    -------
    pandas.DataFrame
        Input table with boolean structure columns; index and row order retained. Reads
        annotations only; no result files are written.
    """
    contexts_interlap_dict = {t: {} for t in structure_types}

    gff_dict = read_gff_file(ref_gff_file)['gene']
    for gene_id in gff_dict:
        gene = gff_dict[gene_id]
        gene_structure = get_gene_structure(
            gene, gene_flank_size=gene_flank_size)
        for t in gene_structure:
            contexts_interlap_dict[t].setdefault(gene.chr_id, InterLap())
            for interval in gene_structure[t]:
                contexts_interlap_dict[t][gene.chr_id].add(interval)

    for t in structure_types:
        print(f"Calculating gene structure context: {t}")
        site_contexts_df[t] = site_contexts_df.apply(lambda row: _if_site_in_context(
            row['CHROM'], row['POS'], contexts_interlap_dict[t]), axis=1)

    return site_contexts_df


def get_gene_structure(gene, gene_flank_size=2000, structure_types=['upstream_2kb', "five_prime_UTR", 'CDS', 'intron', "three_prime_UTR", 'downstream_2kb']):

    gene_structure = {t: [] for t in structure_types}

    for mRNA in gene.sub_features:
        if int(mRNA.qualifiers['longest'][0]) != 1:
            continue

        exon_list = []
        for i in mRNA.sub_features:
            if i.type == 'CDS':
                gene_structure['CDS'].append((i.start, i.end))
            elif i.type == 'exon':
                exon_list.append((i.start, i.end))
            elif i.type == 'five_prime_UTR':
                gene_structure["five_prime_UTR"].append((i.start, i.end))
            elif i.type == 'three_prime_UTR':
                gene_structure["three_prime_UTR"].append((i.start, i.end))

        if len(exon_list) > 0:
            intron_list = interval_minus_set((mRNA.start, mRNA.end), exon_list)
        else:
            intron_list = []

        for i in intron_list:
            gene_structure['intron'].append(i)

        if gene.strand == '+':
            gene_structure['upstream_2kb'].append(
                (mRNA.start - gene_flank_size, mRNA.start - 1))
            gene_structure['downstream_2kb'].append(
                (mRNA.end + 1, mRNA.end + gene_flank_size))
        else:
            gene_structure['upstream_2kb'].append(
                (mRNA.end + 1, mRNA.end + gene_flank_size))
            gene_structure['downstream_2kb'].append(
                (mRNA.start - gene_flank_size, mRNA.start - 1))

    return gene_structure


def _if_site_in_context(chr_id, pos, contexts_interlap_dict):
    if chr_id not in contexts_interlap_dict:
        return False
    if len(list(contexts_interlap_dict[chr_id].find((pos, pos)))) < 1:
        return False
    return True


def get_genomic_composition_contexts(site_contexts_df, ref_gff_file, ref_repeat_gff_file, structure_types=['gene_body', 'genic_2kb', 'genic_proximal', "genic_15kb", "TE", "other_intergenic"]):
    """
    Annotate gene neighborhoods, repeats and intergenic membership.

    Parameters
    ----------
    site_contexts_df : pandas.DataFrame
        Site table with CHROM and 1-based POS; updated in place.
    ref_gff_file : str
        Gene annotations with 1-based inclusive feature coordinates.
    ref_repeat_gff_file : str
        Repeat annotations; TE classes DNA, LINE, LTR and SINE are read from similarity
        features.
    structure_types : sequence of str
        Standard context names. Neighborhoods use fixed 0/2000/5000/15000 bp flanks.

    Returns
    -------
    pandas.DataFrame
        Input with overlapping boolean contexts. repeat includes all feature types;
        other_intergenic excludes genic_2kb and TE. No files are written.
    """
    contexts_interlap_dict = {t: {} for t in structure_types}

    gene_dict = read_gff_file(ref_gff_file)['gene']
    flank_sizes = {
        'gene_body': 0,
        'genic_2kb': 2000,
        'genic_proximal': 5000,
        'genic_15kb': 15000
    }

    for gene_id in gene_dict:
        gene = gene_dict[gene_id]
        for t in ['gene_body', 'genic_2kb', 'genic_proximal', 'genic_15kb']:
            flank_size = flank_sizes[t]
            contexts_interlap_dict[t].setdefault(gene.chr_id, InterLap())
            contexts_interlap_dict[t][gene.chr_id].add(
                (gene.start - flank_size, gene.end + flank_size))

    repeat_gff = read_gff_file(ref_repeat_gff_file)
    contexts_interlap_dict['repeat'] = {}
    # Include every repeat annotation, regardless of feature type or class.
    repeat_features = [r for features in repeat_gff.values()
                       for r in features.values()]
    while repeat_features:
        r = repeat_features.pop()
        contexts_interlap_dict['repeat'].setdefault(r.chr_id, InterLap())
        contexts_interlap_dict['repeat'][r.chr_id].add((r.start, r.end))
        repeat_features.extend(getattr(r, 'sub_features', None) or [])

    repeat_dict = repeat_gff['similarity']
    TEs = ['DNA', 'LINE', 'LTR', 'SINE']
    for repeat_id in repeat_dict:
        r = repeat_dict[repeat_id]
        if r.qualifiers['class'][0] in TEs or r.qualifiers['class'][0].split('/')[0] in TEs:
            contexts_interlap_dict['TE'].setdefault(r.chr_id, InterLap())
            contexts_interlap_dict['TE'][r.chr_id].add((r.start, r.end))

    for t in ['gene_body', 'genic_2kb', 'genic_proximal', 'genic_15kb', 'TE', 'repeat']:
        print(f"Calculating genomic composition context: {t}")
        site_contexts_df[t] = site_contexts_df.apply(lambda row: _if_site_in_context(
            row['CHROM'], row['POS'], contexts_interlap_dict[t]), axis=1)

    site_contexts_df['other_intergenic'] = site_contexts_df.apply(
        lambda row: not any([row[t] for t in ['genic_2kb', 'TE']]), axis=1)

    return site_contexts_df


def classify_aa_mutation(effect, impact):
    if impact == 'HIGH':
        return 1
    elif impact == 'MODERATE' and 'missense_variant' in effect:
        return 2
    elif impact == 'LOW' and 'synonymous_variant' in effect:
        return 3
    else:
        return np.nan


def get_coding_mutation_contexts(site_contexts_df, cds_snpeff_vcf_file, dataset=''):

    """
    Assign the prioritized SnpEff coding consequence by variant ID.

    Parameters
    ----------
    site_contexts_df : pandas.DataFrame
        Site table with ID; updated in place, retaining index/order.
    cds_snpeff_vcf_file : str
        VCF with ANN entries; IDs must match the caller table.
    dataset : str
        Suffix appended to missense, synonymous and putative_LoF columns.

    Returns
    -------
    pandas.DataFrame
        Input with boolean consequence columns. Priority is HIGH, MODERATE missense, LOW
        synonymous; all ANN entries compete per ID. Unmatched IDs are false. Reads VCF; no
        files are written.
    """
    vcf = VCF(cds_snpeff_vcf_file)
    site_info_list = []
    for r in vcf:
        for info in r.INFO['ANN'].split(","):
            info = info.split("|")
            site_info_list.append((r.ID, r.CHROM, r.POS, info[1], info[2]))
    vcf.close()

    site_info_df = pd.DataFrame(site_info_list, columns=[
                                'ID', 'CHROM', 'POS', 'EFFECT', 'IMPACT'])
    site_info_df['MUT_TYPE'] = site_info_df.apply(
        lambda x: classify_aa_mutation(x['EFFECT'], x['IMPACT']), axis=1)
    site_info_df.dropna(subset=['MUT_TYPE'], inplace=True, ignore_index=True)
    # Keep the highest-priority consequence per variant ID.
    site_info_df = site_info_df.sort_values(
        by=['MUT_TYPE'], ascending=True).drop_duplicates(subset=['ID'], keep='first')
    # site_info_df = site_info_df[['ID', 'CHROM', 'POS', 'MUT_TYPE']]
    site_info_df = site_info_df.sort_values(
        by=['CHROM', 'POS']).reset_index(drop=True)

    mut_type_dict = {1: 'LoF', 2: 'Missense', 3: 'Synonymous'}

    site_info_df['MUT_TYPE'] = site_info_df['MUT_TYPE'].apply(
        lambda x: mut_type_dict[x])

    missense_id_list = set(
        site_info_df[site_info_df['MUT_TYPE'] == 'Missense']['ID'].tolist())
    site_contexts_df[f'missense{dataset}'] = site_contexts_df['ID'].apply(
        lambda x: x in missense_id_list)

    synonymous_id_list = set(
        site_info_df[site_info_df['MUT_TYPE'] == 'Synonymous']['ID'].tolist())
    site_contexts_df[f'synonymous{dataset}'] = site_contexts_df['ID'].apply(
        lambda x: x in synonymous_id_list)

    lof_id_list = set(
        site_info_df[site_info_df['MUT_TYPE'] == 'LoF']['ID'].tolist())
    site_contexts_df[f'putative_LoF{dataset}'] = site_contexts_df['ID'].apply(
        lambda x: x in lof_id_list)

    return site_contexts_df


def get_bed_contexts(site_contexts_df, bed_file, context_name):
    """
    Match sites to the project interval table without coordinate conversion.

    Parameters
    ----------
    site_contexts_df : pandas.DataFrame
        Site table with CHROM and POS; updated in place.
    bed_file : str
        Three-column tab-delimited CHROM/START/END. Both endpoints are inclusive in this
        existing implementation, even if the filename ends in bed.
    context_name : str
        Boolean output column name.

    Returns
    -------
    pandas.DataFrame
        Input with context_name; original index/order retained. Caller must establish
        coordinate harmonization. No result files are written.
    """
    bed_df = pd.read_csv(bed_file, sep='\t', header=None,
                         names=['CHROM', 'START', 'END'])
    bed_interlap_dict = {}
    for chr_id in bed_df['CHROM'].unique():
        bed_interlap_dict[chr_id] = InterLap()
        chr_bed_df = bed_df[bed_df['CHROM'] == chr_id]
        for _, row in chr_bed_df.iterrows():
            bed_interlap_dict[chr_id].add((row['START'], row['END']))

    site_contexts_df[context_name] = site_contexts_df.apply(
        lambda row: _if_site_in_context(row['CHROM'], row['POS'], bed_interlap_dict), axis=1)

    return site_contexts_df


def get_ID_list_contexts(site_contexts_df, ID_list, context_name):
    """
    Annotate membership in an explicit identifier set.

    Parameters
    ----------
    site_contexts_df : pandas.DataFrame
        Table with ID; updated in place.
    ID_list : iterable
        Identifiers to match exactly; duplicates collapse to a set.
    context_name : str
        Boolean output column name.

    Returns
    -------
    pandas.DataFrame
        Input with exact-ID membership, preserving row order/index. No files are read or
        written.
    """
    ID_list = set(ID_list)
    site_contexts_df[context_name] = site_contexts_df['ID'].apply(
        lambda x: x in ID_list)
    return site_contexts_df


def get_window_contexts(site_contexts_df, windows_df, context_name):
    """
    Match sites to inclusive genomic windows.

    Parameters
    ----------
    site_contexts_df : pandas.DataFrame
        Site table with CHROM and 1-based POS; updated in place.
    windows_df : pandas.DataFrame
        CHROM, START and END with inclusive endpoints; any overlapping window gives membership.
    context_name : str
        Boolean output column name.

    Returns
    -------
    pandas.DataFrame
        Input with context_name, preserving row order/index. No files are read or written.
    """
    windows_interlap_dict = {}
    for chr_id in windows_df['CHROM'].unique():
        windows_interlap_dict[chr_id] = InterLap()
        chr_windows_df = windows_df[windows_df['CHROM'] == chr_id]
        for _, row in chr_windows_df.iterrows():
            windows_interlap_dict[chr_id].add((row['START'], row['END']))

    site_contexts_df[context_name] = site_contexts_df.apply(
        lambda row: _if_site_in_context(row['CHROM'], row['POS'], windows_interlap_dict), axis=1)

    return site_contexts_df


def get_gene_contexts(site_contexts_df, gene_IDs, ref_gff_file, context_name, flank_size=2000):
    """
    Match sites to catalog genes and their flanks.

    Parameters
    ----------
    site_contexts_df : pandas.DataFrame
        Site table with CHROM and 1-based POS; updated in place.
    gene_IDs : iterable of str
        Catalog IDs matched to GFF gene IDs; absent genes are skipped.
    ref_gff_file : str
        GFF/GFF.gz with 1-based inclusive gene coordinates.
    context_name : str
        Boolean output column name.
    flank_size : int
        Length added to both gene ends in bp, without chromosome clipping.

    Returns
    -------
    pandas.DataFrame
        Input with interval membership; original index/order retained. Reads GFF; no result
        files are written.
    """
    gene_dict = read_gff_file(ref_gff_file)['gene']
    contexts_interlap_dict = {}
    for gene_id in gene_IDs:
        if gene_id not in gene_dict:
            continue
        gene = gene_dict[gene_id]
        contexts_interlap_dict.setdefault(gene.chr_id, InterLap())
        contexts_interlap_dict[gene.chr_id].add(
            (gene.start - flank_size, gene.end + flank_size))
    site_contexts_df[context_name] = site_contexts_df.apply(
        lambda row: _if_site_in_context(row['CHROM'], row['POS'], contexts_interlap_dict), axis=1)

    return site_contexts_df


def get_window_value_contexts(site_contexts_df, windows_df, context_name):
    """
    Assign each site's matching window value to ``context_name``.

    windows_df must contain CHROM, START, END, and context_name; sites must
    contain CHROM and POS. Coordinates are inclusive, as in get_window_contexts.
    Unmatched sites receive NaN. For overlapping windows, the first matching
    row in windows_df supplies the value. Update and return site_contexts_df
    in place, preserving its index and row order.

    windows_df 必须包含 CHROM、START、END 和 context_name 列，位点表必须
    包含 CHROM、POS。区间包含起止点，与 get_window_contexts 一致。
    未匹配位点填 NaN；窗口重叠时取 windows_df 中最先出现的匹配行的值。
    原地更新并返回 site_contexts_df，保留其索引和行顺序。
    """
    for name, df, columns in [
        ('windows_df', windows_df, ['CHROM', 'START', 'END', context_name]),
        ('site_contexts_df', site_contexts_df, ['CHROM', 'POS']),
    ]:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f'{name} is missing required columns: {missing}')

    windows_interlap_dict = {}
    window_values = windows_df[context_name].tolist()
    for order, (chr_id, start, end) in enumerate(
        windows_df[['CHROM', 'START', 'END']].itertuples(
            index=False, name=None)
    ):
        windows_interlap_dict.setdefault(chr_id, InterLap()).add(
            (start, end, order))

    values = []
    for chr_id, pos in site_contexts_df[['CHROM', 'POS']].itertuples(
        index=False, name=None
    ):
        intervals = windows_interlap_dict.get(chr_id)
        matches = intervals.find((pos, pos)) if intervals is not None else ()
        first_match = min(
            matches, key=lambda interval: interval[2], default=None)
        values.append(window_values[first_match[2]]
                      if first_match is not None else np.nan)

    site_contexts_df[context_name] = values
    return site_contexts_df


def get_pi_statistics(ref_gff_file, ref_genome_fa, vcf_file, window_size=500, gene_flank_size=15000, min_snp_count=3):
    """
    Calculate windowed diversity in merged gene neighborhoods.

    Parameters
    ----------
    ref_gff_file : str
        GFF gene annotations, converted to 1-based inclusive intervals.
    ref_genome_fa : str
        Indexed reference FASTA used to clip neighborhoods at chromosome boundaries.
    vcf_file : str
        Indexed landrace VCF; all records/samples contribute, with no additional MAF filter.
    window_size : int
        Window length in bp, local to each merged neighborhood.
    gene_flank_size : int
        Flank length in bp; default 15000.
    min_snp_count : int
        Minimum variant record count per output window; default 3.

    Returns
    -------
    pandas.DataFrame
        CHROM, START, END, SNP_COUNT, PI in traversal order. PI is allel.windowed_diversity
        per-base diversity; windows with too few records are excluded. Reads FASTA/GFF/VCF,
        displays progress, writes no files.
    """
    genome_dict = read_fasta_by_faidx(ref_genome_fa)
    chr_len_dict = {chr_id: genome_dict[chr_id].len()
                    for chr_id in genome_dict}
    chr_id_list = get_chr_id_list_from_vcf(vcf_file)
    chr_len_dict = {chr_id: chr_len_dict[chr_id]
                    for chr_id in chr_id_list if chr_id in chr_len_dict}
    gene_dict = read_gff_file(ref_gff_file)['gene']
    gene_regions_dict = {}
    for gene_id in gene_dict:
        gene = gene_dict[gene_id]
        if gene.chr_id not in chr_len_dict:
            continue
        gene_regions_dict.setdefault(gene.chr_id, [])
        gene_regions_dict[gene.chr_id].append(
            (max(1, gene.start - gene_flank_size), min(chr_len_dict[gene.chr_id], gene.end + gene_flank_size)))
    for chr_id in gene_regions_dict:
        gene_regions_dict[chr_id] = merge_intervals(gene_regions_dict[chr_id])

    vcf = VCF(vcf_file)
    results_list = []
    for chr_id in gene_regions_dict:
        for start, end in tqdm(gene_regions_dict[chr_id], desc=f"Processing {chr_id}"):
            region = f"{chr_id}:{start}-{end}"
            # chr_id = 'Chr01'
            # start = 70290334
            # end = 70425121

            positions = []
            genotypes = []

            for variant in vcf(region):
                positions.append(variant.POS)
                genotypes.append(variant.genotypes)

            # covert to numpy array and remove phase information (keep only two alleles)
            genotypes = np.array(genotypes)[:, :, :2]
            positions = np.array(positions)

            g = allel.GenotypeArray(genotypes)
            ac = g.count_alleles()
            pi_values, windows, n_bases, counts = allel.windowed_diversity(
                positions, ac, size=window_size, start=start, stop=end)

            # Append the results to the list
            for pi, win, count in zip(pi_values, windows, counts):
                results_list.append((chr_id, win[0], win[1], count, pi))

    results_df = pd.DataFrame(results_list, columns=[
                              'CHROM', 'START', 'END', 'SNP_COUNT', 'PI'])
    results_df = results_df[results_df['SNP_COUNT']
                            >= min_snp_count].reset_index(drop=True)
    return results_df


def _window_repeat_fraction(windows, repeat_intervals):
    """Fraction covered by the repeat union; coordinates are 1-based inclusive."""
    windows = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
    if not repeat_intervals or len(windows) == 0:
        return np.zeros(len(windows), dtype=float)
    merged = np.asarray(merge_intervals(
        repeat_intervals, int=True), dtype=np.int64)
    starts, ends = merged[:, 0], merged[:, 1]
    lengths = ends - starts + 1
    prefix_lengths = np.r_[0, np.cumsum(lengths)]

    # Covered bases up to each endpoint, minus coverage before each window.
    # Search sorted interval ends rather than scanning every repeat per window.
    endpoints = windows.copy()
    endpoints[:, 0] -= 1
    completed = np.searchsorted(ends, endpoints, side='right')
    active = completed < len(ends)
    next_interval = np.minimum(completed, len(ends) - 1)
    partial = np.clip(endpoints - starts[next_interval] + 1,
                      0, lengths[next_interval])
    covered_to_endpoint = prefix_lengths[completed] + \
        np.where(active, partial, 0)
    covered_bp = covered_to_endpoint[:, 1] - covered_to_endpoint[:, 0]
    return covered_bp / (windows[:, 1] - windows[:, 0] + 1)


def get_genome_windows_contexts(ref_genome_fa, ref_gff_file, ref_repeat_gff_file,
                                bed_files=None, window_contexts=None,
                                recombination_file=None, window_size=500,
                                bed_zero_based=False):
    """Build continuous annotation coverage on every reference-genome window.

    Parameters
    ----------
    ref_genome_fa : str
        Reference FASTA with an existing .fai; retain every reference sequence.
    ref_gff_file : str
        Gene GFF used for gene-body, flanking and gene-structure intervals.
    ref_repeat_gff_file : str
        Repeat GFF similarity features. TE uses DNA/LINE/LTR/SINE classes;
        REPEAT_FRACTION uses all repeat similarity annotations.
    bed_files : dict or None
        Annotation names mapped to BED paths, such as ACR and UMR.
    window_contexts : dict or None
        Optional annotation names mapped to CHROM/START/END interval tables.
    recombination_file : str or None
        Optional map for recombination-class interval coverage.
    window_size : int
        Window size in bp. Coordinates are 1-based inclusive; the final
        window uses its actual length, including bases without variants.
    bed_zero_based : bool
        False retains the project's inclusive BED convention. True converts
        standard 0-based half-open BED intervals to 1-based inclusive intervals.

    Returns
    -------
    pandas.DataFrame
        CHROM, START, END and continuous interval-union coverage fractions.
        Overlapping bases count once per annotation; absent coverage is zero.
        Does not read genotypes, convert fractions to booleans or save files.
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError('window_size must be a positive integer in bp.')
    fai_df = pd.read_csv(str(ref_genome_fa) + '.fai', sep='\t', header=None,
                         usecols=[0, 1], names=['CHROM', 'LENGTH'],
                         dtype={'CHROM': str, 'LENGTH': np.int64})
    if fai_df.empty or fai_df['CHROM'].duplicated().any() or (fai_df['LENGTH'] <= 0).any():
        raise ValueError(
            'Reference index must contain unique chromosomes with positive lengths.')

    flank_sizes = {'gene_body': 0, 'genic_2kb': 2000,
                   'genic_proximal': 5000, 'genic_15kb': 15000}
    structure_types = ['upstream_2kb', 'five_prime_UTR', 'CDS', 'intron',
                       'three_prime_UTR', 'downstream_2kb']
    contexts = {name: {}
                for name in list(flank_sizes) + structure_types + ['TE']}
    for gene in read_gff_file(ref_gff_file)['gene'].values():
        for name, flank in flank_sizes.items():
            contexts[name].setdefault(gene.chr_id, []).append(
                (gene.start - flank, gene.end + flank))
        for name, intervals in get_gene_structure(gene).items():
            contexts[name].setdefault(gene.chr_id, []).extend(intervals)

    for repeat in read_gff_file(ref_repeat_gff_file)['similarity'].values():
        repeat_class = repeat.qualifiers['class'][0].split('/')[0]
        if repeat_class in ['DNA', 'LINE', 'LTR', 'SINE']:
            contexts['TE'].setdefault(repeat.chr_id, []).append(
                (repeat.start, repeat.end))

    for name, bed_file in (bed_files or {}).items():
        bed_df = pd.read_csv(bed_file, sep='\t', header=None, comment='#',
                             usecols=[0, 1, 2], names=['CHROM', 'START', 'END'],
                             dtype={'CHROM': str})
        if bed_zero_based:
            bed_df = bed_df[bed_df['END'] > bed_df['START']].copy()
            bed_df['START'] += 1
        contexts[name] = {
            chrom: list(frame[['START', 'END']].itertuples(
                index=False, name=None))
            for chrom, frame in bed_df.groupby('CHROM', sort=False)}

    for name, frame in (window_contexts or {}).items():
        contexts[name] = {
            chrom: list(group[['START', 'END']].itertuples(
                index=False, name=None))
            for chrom, group in frame.groupby('CHROM', sort=False)}

    if recombination_file is not None:
        recombination_df = pd.read_csv(recombination_file).dropna()
        recombination_df['meanCmPerMb'] = recombination_df['meanCmPerBp'] * 1e6
        rate = recombination_df['meanCmPerMb']
        # Match get_site_recombination_contexts thresholds and 1,000 bp intervals.
        p75 = rate[rate > 3].quantile(0.75)
        masks = {'low_recombination': rate < 3,
                 'med_recombination': (rate >= 3) & (rate < p75),
                 'high_recombination': rate >= p75}
        for name, mask in masks.items():
            contexts[name] = {}
            for chrom, bp in recombination_df.loc[mask, ['chr', 'bp']].itertuples(index=False, name=None):
                contexts[name].setdefault(chrom, []).append(
                    (int(bp), int(bp) + 999))

    frames = []
    for chrom, length in tqdm(fai_df.itertuples(index=False, name=None),
                              total=len(fai_df), desc='Calculating window contexts'):
        starts = np.arange(1, length + 1, window_size, dtype=np.int64)
        frame = pd.DataFrame({'CHROM': chrom, 'START': starts,
                              'END': np.minimum(starts + window_size - 1, length)})
        windows = frame[['START', 'END']].to_numpy()
        for name, intervals_by_chrom in contexts.items():
            intervals = intervals_by_chrom.get(chrom, [])
            if any(end < start for start, end in intervals):
                raise ValueError(f'Invalid interval in {name} on {chrom}.')
            frame[name] = _window_repeat_fraction(windows, intervals)
        excluded = contexts['genic_2kb'].get(
            chrom, []) + contexts['TE'].get(chrom, [])
        frame['other_intergenic'] = 1 - \
            _window_repeat_fraction(windows, excluded)
        frames.append(frame)
    windows_df = pd.concat(frames, ignore_index=True)
    return get_window_repeat_contexts(windows_df, ref_repeat_gff_file)


def get_window_repeat_contexts(windows_df, ref_repeat_gff_file):
    """
    Add REPEAT_FRACTION using all repeat similarity annotations.

    Reuse get_genome_allele_statistics readers and union coverage calculation.
    Coordinates are 1-based inclusive; overlaps count once and empty coverage
    is zero. Update and return the window table, preserving its row order/index.
    复用 get_genome_allele_statistics 的工具，计算全部 repeat similarity
    注释的并集覆盖比例。使用 1-based 闭区间，重叠只计一次，无覆盖为零。
    原地更新并返回窗口表，保留行顺序及索引。
    """
    repeat_intervals = _read_repeat_intervals(ref_repeat_gff_file)
    fractions = np.zeros(len(windows_df), dtype=float)
    for chrom, positions in windows_df.groupby('CHROM', sort=False).indices.items():
        windows = windows_df.iloc[positions][['START', 'END']].to_numpy()
        fractions[positions] = _window_repeat_fraction(
            windows, repeat_intervals.get(chrom, []))
    windows_df['REPEAT_FRACTION'] = fractions
    return windows_df


def _read_repeat_intervals(ref_repeat_gff_file):
    """Load 1-based inclusive similarity annotations by chromosome."""
    repeat_gff = read_gff_file(ref_repeat_gff_file)
    if 'similarity' not in repeat_gff:
        raise ValueError('Repeat GFF contains no similarity feature type.')
    intervals = {}
    for repeat in repeat_gff['similarity'].values():
        start, end = int(repeat.start), int(repeat.end)
        if start < 1 or end < start:
            raise ValueError(
                'Repeat coordinates must be 1-based with START <= END.')
        intervals.setdefault(repeat.chr_id, []).append((start, end))
    return intervals


def _genome_allele_statistics_chunk(task):
    """Calculate one chunk with a process-local VCF reader; return only rows."""
    (chunk_index, vcf_file, chr_id, chunk_start, chunk_end,
     window_size, min_snp_count, ld_min_maf, mac) = task
    results_list = []
    vcf = VCF(vcf_file)
    try:
        region = f"{chr_id}:{chunk_start}-{chunk_end}"

        positions = []
        genotypes = []
        biallelic_snps = []

        for variant in vcf(region):
            variant_genotypes = variant.genotypes
            if mac > 0:
                alt_carriers = sum(
                    any(allele > 0 for allele in genotype[:-1])
                    for genotype in variant_genotypes
                )
                if alt_carriers < mac:
                    continue
            positions.append(variant.POS)
            genotypes.append(variant_genotypes)
            biallelic_snps.append(
                variant.is_snp and len(variant.ALT) == 1
            )

        n_variants = len(positions)

        # No variants in this chunk
        if n_variants == 0:
            return chunk_index, chr_id, chunk_start, chunk_end, 0, []

        positions = np.asarray(
            positions,
            dtype=np.int64,
        )

        # cyvcf2 genotype:
        # [allele1, allele2, phased]
        genotypes = np.asarray(genotypes)[:, :, :2]

        g = allel.GenotypeArray(genotypes)
        ac = g.count_alleles()

        pi_values, windows, n_bases, counts = (
            allel.windowed_diversity(
                positions,
                ac,
                size=window_size,
                start=chunk_start,
                stop=chunk_end,
            )
        )

        # Filter here instead of building all empty / low-SNP windows
        keep = counts >= min_snp_count

        # Reuse PI windows so all statistics have identical boundaries.
        tajima_d_values, _, _ = allel.windowed_tajima_d(
            positions,
            ac,
            windows=windows,
            min_sites=3,
        )

        theta_w_values, _, _, _ = allel.windowed_watterson_theta(
            positions,
            ac,
            windows=windows,
        )

        ld_r2_values = np.full(len(windows), np.nan)
        # For biallelic SNPs, MAF is min(p_ref, 1 - p_ref).
        allele_numbers = ac.sum(axis=1)
        ref_frequency = np.divide(
            ac[:, 0],
            allele_numbers,
            out=np.full(len(ac), np.nan),
            where=allele_numbers > 0,
        )
        maf = np.minimum(ref_frequency, 1.0 - ref_frequency)
        ld_sites = (
            np.asarray(biallelic_snps, dtype=bool)
            & ac.is_segregating()
            & (maf >= ld_min_maf)
        )
        ld_positions = positions[ld_sites]
        ld_genotypes = g.compress(ld_sites, axis=0).to_n_alt(
            fill=-1, dtype="i1"
        )

        # Handle singleton windows and undefined LD pairs explicitly.
        # windowed_r_squared uses percentile without these guards.
        for i in np.flatnonzero(keep):
            left = np.searchsorted(ld_positions, windows[i, 0])
            right = np.searchsorted(
                ld_positions, windows[i, 1], side="right"
            )
            if right - left < 2:
                continue
            r_squared = np.atleast_1d(
                allel.rogers_huff_r(ld_genotypes[left:right])
            ) ** 2
            valid = r_squared[np.isfinite(r_squared)]
            if valid.size:
                ld_r2_values[i] = np.median(
                    np.clip(valid, 0.0, 1.0))

        for pi, tajima_d, theta_w, ld_r2, win, count in zip(
            pi_values[keep],
            tajima_d_values[keep],
            theta_w_values[keep],
            ld_r2_values[keep],
            windows[keep],
            counts[keep],
        ):
            results_list.append(
                (
                    chr_id,
                    int(win[0]),
                    int(win[1]),
                    int(count),
                    float(pi),
                    float(tajima_d),
                    float(theta_w),
                    float(ld_r2),
                )
            )

    finally:
        vcf.close()
    return chunk_index, chr_id, chunk_start, chunk_end, n_variants, results_list


def get_genome_allele_statistics(
    ref_genome_fa,
    vcf_file,
    recombination_file,
    window_size=500,
    min_snp_count=3,
    chunk_size=1_000_000,
    ld_min_maf=0.0,
    mac=0,
    ref_repeat_gff_file=None,
    n_jobs=40,
):
    """
    Return windowed PI, TajimaD, ThetaW (per base), and median LD_r2.

    TajimaD is NaN with fewer than three segregating sites. LD uses only
    segregating biallelic SNPs and diploid genotype dosages; undefined pairs
    are excluded, and windows with no valid pairs have LD_r2 = NaN.
    ld_min_maf (0 to 0.5, inclusive) applies only to LD. MAF is calculated
    across all VCF samples using called alleles.
    mac is the minimum number of individuals carrying at least one ALT allele
    (heterozygotes and ALT homozygotes each count as one). Sites below mac
    are excluded from all statistics and SNP counts; 0 disables this filter.

    ref_repeat_gff_file is a repeat GFF/GFF.gz path. As in the example below,
    annotations are read from the 'similarity' features using read_gff_file.
    REPEAT_FRACTION is the fraction of each output window's bases covered by
    their union (0 to 1, e.g. 0.8 means 80%). Coordinates include both endpoints;
    overlapping repeats count once and terminal windows use their actual size.
    Windows without repeat annotations have fraction 0. If no repeat GFF is
    supplied, REPEAT_FRACTION is NaN. Existing SNP-window filters are retained.
    n_jobs is the number of worker processes (default 40; 1 runs serially).
    Each worker opens its own indexed VCF reader for one chunk; progress is
    updated as chunks finish, while output retains the original chunk order.
    n_jobs 为并行进程数，默认 40；设为 1 则串行运行。各进程独立读取
    VCF 分块，进度条按完成分块更新，输出保留原始分块顺序。
    输入重复序列 GFF；输出 REPEAT_FRACTION 为窗口内重复区间并集覆盖比例。
    端点计入，重叠碱基仅计一次；未提供文件时为 NaN，无重复注释时为 0。
    """
    from multiprocessing import get_context
    from numbers import Integral

    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    if not 0.0 <= ld_min_maf <= 0.5:
        raise ValueError("ld_min_maf must be between 0.0 and 0.5 inclusive")

    repeat_intervals_by_chr = (_read_repeat_intervals(ref_repeat_gff_file)
                               if ref_repeat_gff_file is not None else {})

    # Genome chromosome lengths
    genome_dict = read_fasta_by_faidx(ref_genome_fa)

    chr_len_dict = {
        chr_id: genome_dict[chr_id].len()
        for chr_id in genome_dict
    }

    # Keep only chromosomes present in the VCF
    chr_id_list = get_chr_id_list_from_vcf(vcf_file)

    chr_len_dict = {
        chr_id: chr_len_dict[chr_id]
        for chr_id in chr_id_list
        if chr_id in chr_len_dict
    }

    # Total number of chunks for tqdm
    total_chunks = sum(
        (chr_len + chunk_size - 1) // chunk_size
        for chr_len in chr_len_dict.values()
    )

    # Small task descriptors; genotype arrays stay inside each worker.
    tasks = [
        (index, str(vcf_file), chr_id, chunk_start, chunk_end,
         window_size, min_snp_count, ld_min_maf, mac)
        for index, (chr_id, chunk_start, chunk_end) in enumerate(
            (chr_id, chunk_start, min(chunk_start + chunk_size - 1, chr_len))
            for chr_id, chr_len in chr_len_dict.items()
            for chunk_start in range(1, chr_len + 1, chunk_size)
        )
    ]
    chunk_results = {}
    with tqdm(total=total_chunks, desc="Genome statistics") as pbar:
        def collect(result):
            index, chr_id, chunk_start, chunk_end, n_variants, rows = result
            chunk_results[index] = rows
            pbar.set_postfix(chrom=chr_id,
                             region=f"{chunk_start}-{chunk_end}", snps=n_variants)
            pbar.update(1)

        if n_jobs == 1:
            for task in tasks:
                collect(_genome_allele_statistics_chunk(task))
        elif tasks:
            # Spawn avoids inheriting native VCF/HDF5 handles from Jupyter.
            with get_context("spawn").Pool(processes=min(n_jobs, total_chunks)) as pool:
                for result in pool.imap_unordered(
                    _genome_allele_statistics_chunk, tasks, chunksize=1
                ):
                    collect(result)

    # Preserve serial chunk order (including the returned DataFrame index).
    results_list = [row for index in range(total_chunks)
                    for row in chunk_results[index]]

    results_df = pd.DataFrame(
        results_list,
        columns=[
            "CHROM",
            "START",
            "END",
            "SNP_COUNT",
            "PI",
            "TajimaD",
            "ThetaW",
            "LD_r2",
        ],
    )

    results_df['REPEAT_FRACTION'] = np.nan
    if ref_repeat_gff_file is not None:
        for chr_id, chromosome_windows in results_df.groupby('CHROM', sort=False):
            results_df.loc[chromosome_windows.index, 'REPEAT_FRACTION'] = _window_repeat_fraction(
                chromosome_windows[['START', 'END']].to_numpy(),
                repeat_intervals_by_chr.get(chr_id, []),
            )

    recombination_df = pd.read_csv(recombination_file, sep=",", header=0)
    recombination_df.dropna(inplace=True, ignore_index=True)
    recombination_df['meanCmPerMb'] = recombination_df['meanCmPerBp'] * 1e6

    results_df["mid"] = ((results_df.START + results_df.END) // 2).astype(int)

    results_df = pd.merge_asof(
        results_df.sort_values(["mid", "CHROM"]),
        recombination_df.sort_values(["bp", "chr"]),
        left_on="mid", right_on="bp",
        left_by="CHROM", right_by="chr",
        direction="backward",
        tolerance=999
    )

    results_df.sort_values(["CHROM", "START"], inplace=True)

    results_df.drop(columns=["mid", "bp", "chr",
                    "meanCmPerBp"], inplace=True, errors='ignore')

    return results_df


def _genome_window_statistics_chunk(task):
    """Count all records and summarize diversity in aligned reference windows."""
    chr_id, start, end, vcf_file, window_size, min_variant_count = task
    starts = np.arange(start, end + 1, window_size, dtype=np.int64)
    windows = np.column_stack([starts, np.minimum(starts + window_size - 1, end)])
    positions, genotypes, snp_positions = [], [], []
    reader = VCF(vcf_file)
    try:
        for variant in reader(f'{chr_id}:{start}-{end}'):
            # Assign records by POS even when an indel overlaps a chunk boundary.
            if not start <= variant.POS <= end:
                continue
            positions.append(variant.POS)
            genotypes.append(variant.genotypes)
            if variant.is_snp:
                snp_positions.append(variant.POS)
    finally:
        reader.close()

    variant_counts = np.zeros(len(windows), dtype=np.int64)
    snp_counts = np.zeros(len(windows), dtype=np.int64)
    pi = np.full(len(windows), np.nan)
    tajima_d = np.full(len(windows), np.nan)
    if positions:
        positions = np.asarray(positions, dtype=np.int64)
        alleles = allel.GenotypeArray(np.asarray(genotypes)[:, :, :2]).count_alleles()
        pi, _, _, variant_counts = allel.windowed_diversity(
            positions, alleles, windows=windows,
        )
        tajima_d, _, _ = allel.windowed_tajima_d(
            positions, alleles, windows=windows, min_sites=3,
        )
        insufficient = variant_counts < min_variant_count
        pi[insufficient] = np.nan
        tajima_d[insufficient] = np.nan
        snp_counts = np.bincount(
            (np.asarray(snp_positions, dtype=np.int64) - start) // window_size,
            minlength=len(windows),
        )
    return pd.DataFrame({
        'CHROM': chr_id, 'START': windows[:, 0], 'END': windows[:, 1],
        'VARIANT_COUNT': variant_counts, 'SNP_COUNT': snp_counts,
        'PI': pi, 'TajimaD': tajima_d,
    })


def get_genome_window_statistics(ref_genome_fa, vcf_file, recombination_file,
                                 window_size=500, min_variant_count=3,
                                 chunk_size=1_000_000, n_jobs=1):
    """Calculate diversity on every aligned reference-genome window.

    Workflow
    --------
    1. Build all 1-based inclusive windows from the reference FASTA index.
    2. Count SNP/all-variant records and calculate PI and TajimaD in VCF chunks.
    3. Assign recombination rates to window midpoints using the source map.

    Parameters
    ----------
    ref_genome_fa : str
        Reference FASTA with an existing .fai; all reference sequences are kept.
    vcf_file : str
        Indexed, coordinate-sorted diploid VCF. Include SNPs and indels without
        additional MAF, heterozygosity or allele-count filtering.
    recombination_file : str
        CSV with chr, bp and meanCmPerBp. Match midpoints backward within
        999 bp and convert rates to cM/Mb.
    window_size : int
        Window length in bp; the terminal window uses its actual length.
    min_variant_count : int
        Minimum variant-record count for PI/TajimaD; default 3. TajimaD also
        requires at least three segregating sites. Undefined values remain NaN.
    chunk_size : int
        VCF task size in bp, rounded down to a positive window-size multiple.
    n_jobs : int
        Worker count; 1 is serial, otherwise use spawn with independent readers.

    Returns
    -------
    pandas.DataFrame
        CHROM, START, END, VARIANT_COUNT (all records), SNP_COUNT (SNP records),
        PI (per bp), TajimaD and meanCmPerMb. Empty windows have zero counts
        and NaN diversity. Reference sequences absent from VCF are retained.
        No LD, ThetaW or DAF is calculated; no files are saved.
    """
    from multiprocessing import get_context
    from operator import index

    window_size, min_variant_count, chunk_size, n_jobs = map(
        index, (window_size, min_variant_count, chunk_size, n_jobs),
    )
    if min(window_size, min_variant_count, chunk_size, n_jobs) < 1:
        raise ValueError('Window, count, chunk and worker settings must be positive.')
    chunk_size = max(window_size, chunk_size // window_size * window_size)
    reference = pd.read_csv(
        f'{ref_genome_fa}.fai', sep='\t', header=None, usecols=[0, 1],
        names=['CHROM', 'LENGTH'], dtype={'CHROM': str},
    )
    if reference['CHROM'].duplicated().any() or (reference['LENGTH'] <= 0).any():
        raise ValueError('Reference sequences must have unique names and positive lengths.')
    vcf_chromosomes = set(get_chr_id_list_from_vcf(vcf_file))
    frames, tasks = [], []
    for chr_id, length in reference.itertuples(index=False, name=None):
        starts = np.arange(1, length + 1, window_size, dtype=np.int64)
        frames.append(pd.DataFrame({
            'CHROM': chr_id, 'START': starts,
            'END': np.minimum(starts + window_size - 1, length),
        }))
        if chr_id in vcf_chromosomes:
            tasks.extend(
                (chr_id, start, min(start + chunk_size - 1, length),
                 vcf_file, window_size, min_variant_count)
                for start in range(1, length + 1, chunk_size)
            )
    windows = pd.concat(frames, ignore_index=True)
    if n_jobs == 1:
        statistics = [
            _genome_window_statistics_chunk(task)
            for task in tqdm(tasks, desc='Genome window statistics', unit='chunk')
        ]
    elif tasks:
        with get_context('spawn').Pool(processes=min(n_jobs, len(tasks))) as pool:
            statistics = list(tqdm(
                pool.imap(_genome_window_statistics_chunk, tasks), total=len(tasks),
                desc='Genome window statistics', unit='chunk',
            ))
    else:
        statistics = []
    if statistics:
        windows = windows.merge(
            pd.concat(statistics, ignore_index=True), on=['CHROM', 'START', 'END'],
            how='left', validate='one_to_one',
        )
    else:
        for column in ['VARIANT_COUNT', 'SNP_COUNT', 'PI', 'TajimaD']:
            windows[column] = np.nan
    for column in ['VARIANT_COUNT', 'SNP_COUNT']:
        windows[column] = windows[column].fillna(0).astype(np.int64)

    rates = pd.read_csv(recombination_file).dropna()
    rates['meanCmPerMb'] = rates['meanCmPerBp'] * 1e6
    windows['mid'] = ((windows['START'] + windows['END']) // 2).astype(np.int64)
    windows = pd.merge_asof(
        windows.sort_values(['mid', 'CHROM']), rates.sort_values(['bp', 'chr']),
        left_on='mid', right_on='bp', left_by='CHROM', right_by='chr',
        direction='backward', tolerance=999,
    )
    return windows.sort_values(['CHROM', 'START']).reset_index(drop=True)[
        ['CHROM', 'START', 'END', 'VARIANT_COUNT', 'SNP_COUNT', 'PI', 'TajimaD', 'meanCmPerMb']
    ]


def build_genome_tracks(allele_stat_df, ld_blocks, vcf_file, gff_file,
                        ref_genome_fa, window_size=1_000_000, n_jobs=8,
                        chunk_size=5_000_000, rare_maf_threshold=0.05,
                        ref_repeat_gff_file=None, gene_flank_size=2000,
                        bed_files=None, bed_zero_based=False):
    """Build raw genome-landscape statistics on reference-length plotting windows.

    Workflow
    --------
    1. Assign 500 bp statistics and full PLINK blocks to bins by midpoint.
    2. Calculate annotation-union coverage directly in the plotting bins.
    3. Count variants and calculate MAF in one VCF pass; return the table.

    Parameters
    ----------
    allele_stat_df : pandas.DataFrame
        CHROM, START, END, PI, TajimaD, meanCmPerMb. Average valid input
        window values by midpoint without weighting or standardization.
    ld_blocks : pandas.DataFrame
        PLINK CHR, BP1, BP2, KB; average full block lengths by midpoint,
        converting KB to bp. Bins without blocks have NaN lengths.
    vcf_file : str
        Indexed VCF. Count all variant records, including SNPs and indels.
        Rare counts include callable biallelic sites below rare_maf_threshold;
        mean MAF includes callable biallelic sites at or above that threshold.
    gff_file : str
        Gene GFF with 1-based inclusive gene intervals.
    ref_genome_fa : str
        Reference FASTA with an existing .fai. Retain every plotting bin on
        chromosomes in allele_stat_df; clip terminal bins to reference length.
    window_size, chunk_size : int
        Plotting window width (default 1 Mb) and parallel VCF task width in bp.
    n_jobs : int
        VCF worker count; 1 is serial.
    rare_maf_threshold : float
        Shared rare-variant and mean-MAF cutoff; default 0.05.
    ref_repeat_gff_file : str or None
        Repeat similarity GFF. REPEAT_FRACTION includes all similarity
        intervals; TE_FRACTION includes DNA, LINE, LTR and SINE classes.
    gene_flank_size : int
        Extend gene bodies on both sides, default 2000 bp. Clip intervals
        to chromosome bounds before calculating GENIC_FRACTION.
    bed_files : dict or None
        Annotation names mapped to BED paths, such as ACR and UMR. No flanks
        are added. Each annotation produces a <name>_FRACTION column.
    bed_zero_based : bool
        False retains the project's 1-based inclusive BED convention. True
        converts standard 0-based half-open intervals to 1-based inclusive.

    Returns
    -------
    pandas.DataFrame
        CHROM/START/END/BIN and raw tracks in plotting order. Annotation
        fractions are union-covered bp / actual bin bp; overlaps count once
        and absent coverage is zero. GENIC_BP and MAF_SITE_COUNT provide
        supporting counts. Undefined means are NaN. No files are written;
        sources and settings are recorded in attrs. No z-score is applied.
    """
    import re
    from .vcf import calculate_window_maf

    track_window_size = window_size
    maf_n_jobs, maf_chunk_size = n_jobs, chunk_size
    ld_unfiltered_vcf_file, ref_gff_file = vcf_file, gff_file
    genome_path = os.fspath(ref_genome_fa)
    fai_path = os.fspath(str(genome_path) + '.fai')
    if not os.path.isfile(genome_path):
        raise FileNotFoundError(f'Reference genome not found: {genome_path}')
    if not os.path.isfile(fai_path):
        raise FileNotFoundError(
            f'Reference genome index not found: {fai_path}')
    if not isinstance(track_window_size, int) or track_window_size <= 0:
        raise ValueError('track_window_size must be a positive integer in bp.')
    if not isinstance(gene_flank_size, int) or gene_flank_size < 0:
        raise ValueError('gene_flank_size must be a nonnegative integer in bp.')
    annotation_intervals = {}
    if ref_repeat_gff_file is not None:
        annotation_intervals = {'TE_FRACTION': {}, 'REPEAT_FRACTION': {}}
        for repeat in read_gff_file(ref_repeat_gff_file)['similarity'].values():
            interval = (repeat.start, repeat.end)
            annotation_intervals['REPEAT_FRACTION'].setdefault(
                repeat.chr_id, []).append(interval)
            if repeat.qualifiers['class'][0].split('/')[0] in ['DNA', 'LINE', 'LTR', 'SINE']:
                annotation_intervals['TE_FRACTION'].setdefault(
                    repeat.chr_id, []).append(interval)
    for name, bed_file in (bed_files or {}).items():
        bed_df = pd.read_csv(bed_file, sep='\t', header=None, comment='#',
                             usecols=[0, 1, 2], names=['CHROM', 'START', 'END'],
                             dtype={'CHROM': str})
        if bed_zero_based:
            bed_df = bed_df[bed_df['END'] > bed_df['START']].copy()
            bed_df['START'] += 1
        if (bed_df['END'] < bed_df['START']).any():
            raise ValueError(f'Invalid interval in {name}.')
        annotation_intervals[f'{name}_FRACTION'] = {
            chrom: list(frame[['START', 'END']].itertuples(index=False, name=None))
            for chrom, frame in bed_df.groupby('CHROM', sort=False)}
    metric_columns = ['PI', 'TajimaD', 'meanCmPerMb']
    required_columns = ['CHROM', 'START', 'END'] + metric_columns
    missing_columns = sorted(set(required_columns) -
                             set(allele_stat_df.columns))
    if missing_columns:
        raise ValueError(f'Missing allele_stat_df columns: {missing_columns}')
    if allele_stat_df.empty:
        raise ValueError('allele_stat_df contains no windows to plot.')

    track_source_df = allele_stat_df[required_columns].copy()
    track_source_df['CHROM'] = track_source_df['CHROM'].astype(str)
    track_source_df[['START', 'END']] = track_source_df[['START', 'END']].apply(
        pd.to_numeric, errors='raise'
    )
    if (track_source_df[['START', 'END']].isna().any().any()
            or (track_source_df['START'] < 1).any()
            or (track_source_df['END'] < track_source_df['START']).any()):
        raise ValueError(
            'Input windows must have valid 1-based START/END coordinates.')
    track_source_df[metric_columns] = track_source_df[metric_columns].apply(
        pd.to_numeric, errors='coerce'
    ).replace([np.inf, -np.inf], np.nan)

    # Assign each source window by its midpoint to 1-based, non-overlapping bins.
    source_midpoints = (track_source_df['START'] + track_source_df['END']) / 2
    track_source_df['BIN'] = ((source_midpoints - 1) //
                              track_window_size).astype(int)
    track_groups = track_source_df.groupby(['CHROM', 'BIN'], sort=False)
    track_mean_df = track_groups[metric_columns].mean()

    # Assign each LD block once, by its midpoint; average full block lengths and convert kb to bp.
    block_columns = ['CHR', 'BP1', 'BP2', 'KB']
    missing_block_columns = sorted(set(block_columns) - set(ld_blocks.columns))
    if missing_block_columns:
        raise ValueError(f'Missing ld_blocks columns: {missing_block_columns}')
    track_blocks_df = ld_blocks[block_columns].copy()
    track_blocks_df['CHR'] = track_blocks_df['CHR'].astype(str)
    track_blocks_df[['BP1', 'BP2', 'KB']] = track_blocks_df[['BP1', 'BP2', 'KB']].apply(
        pd.to_numeric, errors='raise'
    )
    if (not np.isfinite(track_blocks_df[['BP1', 'BP2', 'KB']].to_numpy()).all()
            or (track_blocks_df['BP1'] < 1).any()
            or (track_blocks_df['BP2'] < track_blocks_df['BP1']).any()
            or (track_blocks_df['KB'] <= 0).any()):
        raise ValueError(
            'LD blocks must have valid 1-based BP1/BP2 and positive KB lengths.')
    block_midpoints = (track_blocks_df['BP1'] + track_blocks_df['BP2']) / 2
    track_blocks_df['BIN'] = ((block_midpoints - 1) //
                              track_window_size).astype(int)
    track_blocks_df.rename(columns={'CHR': 'CHROM'}, inplace=True)
    block_mean_lengths = (track_blocks_df.groupby(['CHROM', 'BIN'])[
                          'KB'].mean() * 1000).rename('LD_BLOCK_BP')
    # Outer join preserves block-only bins even when allele statistics are missing.
    track_mean_df = track_mean_df.join(block_mean_lengths, how='outer')

    # Reuse the project GFF reader: its gene coordinates are 1-based, inclusive.
    track_gene_dict = read_gff_file(ref_gff_file).get('gene', {})
    if not track_gene_dict:
        raise ValueError('No gene features found in ref_gff_file.')
    track_gene_df = pd.DataFrame(
        [(gene_id, str(gene.chr_id), int(gene.start), int(gene.end))
         for gene_id, gene in track_gene_dict.items()],
        columns=['GENE_ID', 'CHROM', 'START', 'END'],
    )
    if ((track_gene_df['START'] < 1).any()
            or (track_gene_df['END'] < track_gene_df['START']).any()):
        raise ValueError(
            'GFF genes must have valid 1-based START/END coordinates.')

    # Natural chromosome order: Chr01, Chr02, ..., Chr10.
    chromosome_order = sorted(
        track_source_df['CHROM'].unique(),
        key=lambda chrom: [int(part) if part.isdigit() else part.lower()
                           for part in re.split(r'(\d+)', chrom)],
    )
    chromosome_lengths = track_source_df.groupby(
        'CHROM')['END'].max().astype(int).to_dict()
    for chrom, block_end in track_blocks_df.groupby('CHROM')['BP2'].max().items():
        if chrom in chromosome_lengths:
            chromosome_lengths[chrom] = max(
                chromosome_lengths[chrom], int(block_end))
    # Check annotated gene ends against the reference chromosome lengths below.
    for chrom, gene_end in track_gene_df.groupby('CHROM')['END'].max().items():
        if chrom in chromosome_lengths:
            chromosome_lengths[chrom] = max(
                chromosome_lengths[chrom], int(gene_end))
    if not set(chromosome_order).intersection(track_gene_df['CHROM']):
        raise ValueError(
            'GFF and plotted chromosomes have no matching chromosome IDs.')

    fai_df = pd.read_csv(fai_path, sep='\t', header=None, usecols=[0, 1],
                         names=['CHROM', 'LENGTH'], dtype={'CHROM': str})
    if fai_df['CHROM'].duplicated().any() or (fai_df['LENGTH'] <= 0).any():
        raise ValueError(
            'Reference index must contain unique chromosomes with positive lengths.')
    reference_lengths = fai_df.set_index(
        'CHROM')['LENGTH'].astype(int).to_dict()
    unmatched = sorted(set(chromosome_order) - set(reference_lengths))
    if unmatched:
        raise ValueError(
            f'Plotted chromosomes missing from reference index: {unmatched}')
    for chrom in chromosome_order:
        if reference_lengths[chrom] < chromosome_lengths[chrom]:
            raise ValueError(
                f'Input coordinates exceed reference length for {chrom}.')
        chromosome_lengths[chrom] = reference_lengths[chrom]

    # Reindex to include empty bins as NaN; never connect across missing data.
    track_frames = []
    for chrom in chromosome_order:
        chrom_length = chromosome_lengths[chrom]
        n_bins = (chrom_length + track_window_size - 1) // track_window_size
        chrom_track = track_mean_df.xs(
            chrom, level='CHROM').reindex(range(n_bins)).copy()
        chrom_track['CHROM'] = chrom
        chrom_track['START'] = np.arange(n_bins) * track_window_size + 1
        chrom_track['END'] = np.minimum(
            (np.arange(n_bins) + 1) * track_window_size, chrom_length)
        chrom_track['BIN'] = np.arange(n_bins)
        track_frames.append(chrom_track.reset_index(drop=True))
    # Count each annotated base once; divide by actual terminal-window length.
    for chrom, chrom_track in zip(chromosome_order, track_frames):
        windows = chrom_track[['START', 'END']].to_numpy(dtype=np.int64)
        chrom_length = chromosome_lengths[chrom]
        genic_intervals = [
            (max(1, int(start) - gene_flank_size),
             min(chrom_length, int(end) + gene_flank_size))
            for start, end in track_gene_df.loc[
                track_gene_df['CHROM'] == chrom, ['START', 'END']
            ].itertuples(index=False, name=None)]
        chrom_track['GENIC_FRACTION'] = _window_repeat_fraction(windows, genic_intervals)
        chrom_track['GENIC_BP'] = np.rint(
            chrom_track['GENIC_FRACTION'] * (windows[:, 1] - windows[:, 0] + 1)
        ).astype(np.int64)
        for column, intervals_by_chr in annotation_intervals.items():
            chrom_track[column] = _window_repeat_fraction(
                windows, intervals_by_chr.get(chrom, []))

    # Compute Variant counts and common-site mean MAF together in one parallel VCF pass.
    track_maf_df = calculate_window_maf(
        ld_unfiltered_vcf_file, chromosome_lengths, window_size=track_window_size,
        n_jobs=maf_n_jobs, chunk_size=maf_chunk_size, min_maf=rare_maf_threshold,
    )
    track_maf_lookup = track_maf_df.set_index(['CHROM', 'BIN'])
    for chrom, chrom_track in zip(chromosome_order, track_frames):
        chrom_maf = track_maf_lookup.xs(
            chrom, level='CHROM').reindex(range(len(chrom_track)))
        for column in ['MAF_SITE_COUNT', 'MAF', 'VARIANT_COUNT', 'RARE_VARIANT_COUNT']:
            chrom_track[column] = chrom_maf[column].to_numpy()
    print(f"MAF excluded {track_maf_df.attrs['skipped_multiallelic']:,} non-biallelic and "
          f"{track_maf_df.attrs['skipped_missing']:,} fully missing sites on plotted chromosomes.")
    track_df = pd.concat(track_frames, ignore_index=True)
    output_columns = ['CHROM', 'START', 'END', 'BIN', 'GENIC_FRACTION']
    output_columns += list(annotation_intervals)
    output_columns += ['VARIANT_COUNT', 'RARE_VARIANT_COUNT', 'MAF', 'PI',
                       'TajimaD', 'meanCmPerMb', 'LD_BLOCK_BP',
                       'GENIC_BP', 'MAF_SITE_COUNT']
    track_df = track_df[output_columns]
    track_df.attrs.update(
        window_size=track_window_size, rare_maf_threshold=rare_maf_threshold,
        gene_flank_size=gene_flank_size, bed_zero_based=bed_zero_based,
        chromosome_lengths=chromosome_lengths,
        sources={'allele_statistics': 'allele_stat_df', 'ld_blocks': 'ld_blocks',
                 'vcf': str(vcf_file), 'gff': str(gff_file),
                 'reference_genome': str(genome_path), 'reference_index': str(fai_path)},
        maf_exclusions=dict(track_maf_df.attrs),
    )
    if ref_repeat_gff_file is not None:
        track_df.attrs['sources']['repeat_gff'] = str(ref_repeat_gff_file)
    track_df.attrs['sources']['beds'] = dict(bed_files or {})
    return track_df


def plot_genome_tracks(track_df, output_prefix=None, title=None,
                       rare_maf_threshold=None, maf_ymin=0.05,
                       ld_ylim=(1e2, 1e6), figsize=None,
                       colors=('#879DD0', '#A0CFDF'), hspace=0.18, h_pad=0.14,
                       label_fontsize=10, tick_fontsize=9, title_fontsize=12):
    """Plot raw prepared genome tracks and optionally export PDF/PNG/CSV.gz.

    Parameters
    ----------
    track_df : pandas.DataFrame
        build_genome_tracks output or its CSV reloaded with pandas. Requires
        CHROM, START, END, GENIC_FRACTION, VARIANT_COUNT, RARE_VARIANT_COUNT,
        MAF, PI, TajimaD, meanCmPerMb and LD_BLOCK_BP. Optional TE_FRACTION,
        REPEAT_FRACTION, ACR_FRACTION and UMR_FRACTION follow Genic in that order.
        Values remain in their original units; no statistics are recomputed.
    output_prefix : str or None
        If supplied, export PDF, 300 dpi PNG and the actual plotted CSV.gz.
    title : str or None
        Optional title; None leaves the figure untitled.
    rare_maf_threshold : float or None
        Legend cutoff, defaulting to table attrs or 0.05 for reloaded CSV.
    maf_ymin : float or None
        MAF lower limit, default 0.05; None enables autoscaling.
    ld_ylim : tuple
        Positive lower/upper bounds in bp. Only LD-block length uses a log axis;
        its exported values remain bp. Default bounds are 100 and 1,000,000.
    figsize : tuple or None
        Figure size; default height is 1.55 inches per track.
    colors : tuple
        Two alternating chromosome colors; rare counts use pale variants.
    hspace, h_pad : float
        Fractional vertical gap and vertical padding in inches.
    label_fontsize, tick_fontsize, title_fontsize : float
        Independent font sizes for labels, ticks and title.

    Returns
    -------
    dict
        figure, axes, plot_data (columns in track order) and exported files.
        Density tracks are covered-base fractions; the pale rare-variant
        area is a subset of the total. No VCF/GFF reads or automatic show.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator, LogLocator, FixedLocator, FuncFormatter
    from src.plot import set_plot_style

    set_plot_style()

    if rare_maf_threshold is None:
        rare_maf_threshold = track_df.attrs.get('rare_maf_threshold', 0.05)
    if not np.isfinite(rare_maf_threshold) or not 0 <= rare_maf_threshold <= 0.5:
        raise ValueError('rare_maf_threshold must be between 0 and 0.5.')
    if maf_ymin is not None and (not np.isfinite(maf_ymin) or not 0 <= maf_ymin <= 0.5):
        raise ValueError('maf_ymin must be None or between 0 and 0.5.')
    ld_ymin, ld_ymax = ld_ylim
    if not (np.isfinite([ld_ymin, ld_ymax]).all() and 0 < ld_ymin < ld_ymax):
        raise ValueError(
            'ld_ylim must contain two increasing positive values.')
    if len(colors) != 2:
        raise ValueError(
            'colors must contain two alternating chromosome colors.')
    track_specs = [('GENIC_FRACTION', 'Genic density')]
    for column, label in [
        ('TE_FRACTION', 'TE density'),
        ('REPEAT_FRACTION', 'Repeat density'),
        ('ACR_FRACTION', 'ACR density'),
        ('UMR_FRACTION', 'UMR density'),
    ]:
        if column in track_df.columns:
            track_specs.append((column, label))
    track_specs += [
        ('VARIANT_COUNT', 'Variant count'),
        ('MAF', f'Mean MAF\n(≥ {rare_maf_threshold:g})'),
        ('PI', r'Mean $\pi$'),
        ('TajimaD', "Tajima's D"),
        ('meanCmPerMb', 'Recombination\n(cM/Mb)'),
        ('LD_BLOCK_BP', 'LD block\nlength (bp)'),
    ]
    plot_data_columns = ['CHROM', 'START', 'END']
    for metric, _ in track_specs:
        plot_data_columns.append(metric)
        if metric == 'VARIANT_COUNT':
            plot_data_columns.append('RARE_VARIANT_COUNT')
    missing_columns = sorted(set(plot_data_columns) - set(track_df.columns))
    if missing_columns:
        raise ValueError(f'Missing track_df columns: {missing_columns}')
    if track_df.empty:
        raise ValueError('track_df contains no windows to plot.')
    track_plot_df = track_df[plot_data_columns].copy()
    track_plot_df['CHROM'] = track_plot_df['CHROM'].astype(str)
    numeric_columns = plot_data_columns[1:]
    track_plot_df[numeric_columns] = track_plot_df[numeric_columns].apply(
        pd.to_numeric, errors='raise')
    if (not np.isfinite(track_plot_df[['START', 'END']].to_numpy()).all()
            or (track_plot_df['START'] < 1).any()
            or (track_plot_df['END'] < track_plot_df['START']).any()):
        raise ValueError(
            'Window coordinates must be finite, 1-based and ordered.')
    rare = track_plot_df['RARE_VARIANT_COUNT']
    total = track_plot_df['VARIANT_COUNT']
    if (rare < 0).any() or (total < 0).any() or (rare > total).any():
        raise ValueError(
            'Rare Variant counts must lie between zero and total Variant counts.')
    track_plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    chromosome_order = track_plot_df['CHROM'].drop_duplicates().tolist()
    track_frames, chromosome_ticks, chromosome_boundaries = [], [], []
    chromosome_offset = 0
    for chrom in chromosome_order:
        chrom_track = track_plot_df.loc[track_plot_df['CHROM'] == chrom].sort_values(
            'START').copy()
        chrom_length = int(chrom_track['END'].max())
        chrom_track['X_LEFT_MB'] = (
            chromosome_offset + chrom_track['START'] - 1) / 1e6
        chrom_track['X_RIGHT_MB'] = (
            chromosome_offset + chrom_track['END']) / 1e6
        track_frames.append(chrom_track)
        chromosome_ticks.append((chromosome_offset + chrom_length / 2) / 1e6)
        chromosome_offset += chrom_length
        chromosome_boundaries.append(chromosome_offset / 1e6)
    # Preserve rows in the same order as the plot, including reloaded CSV input.
    track_plot_df = pd.concat(track_frames, ignore_index=True)[
        plot_data_columns]
    track_colors = colors
    rare_track_colors = [tuple(0.35 * np.array(to_rgb(color)) + 0.65)
                         for color in track_colors]
    if figsize is None:
        figsize = (14, 1.55 * len(track_specs))
    files = {}
    # Override individual sizes too: the project style may set axes/ticks separately.
    with plt.rc_context({
        'font.size': label_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'figure.titlesize': title_fontsize,
        'legend.fontsize': tick_fontsize - 0.5,
        'axes.linewidth': 0.9,
        'pdf.fonttype': 42,
    }):
        track_fig, track_axes = plt.subplots(
            len(track_specs), 1, figsize=figsize, sharex=True,
            layout='constrained',
        )
        track_fig.set_constrained_layout_pads(h_pad=h_pad, hspace=hspace)
        for ax, (metric, ylabel) in zip(track_axes, track_specs):
            if metric == 'LD_BLOCK_BP':
                fill_baseline = ld_ymin
            elif metric == 'MAF' and maf_ymin is not None:
                fill_baseline = maf_ymin
            else:
                fill_baseline = 0
            if metric == 'LD_BLOCK_BP':
                ax.set_yscale('log', base=10)
                ax.set_ylim(ld_ymin, ld_ymax)
            for chrom_index, chrom_track in enumerate(track_frames):
                x = np.r_[chrom_track['X_LEFT_MB'].to_numpy(),
                          chrom_track['X_RIGHT_MB'].iloc[-1]]
                values = chrom_track[metric].to_numpy(dtype=float)
                y = np.r_[values, values[-1]]
                ax.fill_between(x, fill_baseline, y, step='post', color=track_colors[chrom_index % 2],
                                alpha=1.0, linewidth=0)
                ax.step(x, y, where='post',
                        color=track_colors[chrom_index % 2], linewidth=0.8)
                if metric == 'VARIANT_COUNT':
                    rare_values = chrom_track['RARE_VARIANT_COUNT'].to_numpy(
                        dtype=float)
                    rare_y = np.r_[rare_values, rare_values[-1]]
                    rare_color = rare_track_colors[chrom_index % 2]
                    # Pale area is a subset of the total, not an additional count.
                    ax.fill_between(x, 0, rare_y, step='post', color=rare_color,
                                    alpha=1.0, linewidth=0)
                    ax.step(x, rare_y, where='post',
                            color=rare_color, linewidth=0.8)
            for boundary in chromosome_boundaries[:-1]:
                ax.axvline(boundary, color='0.85', linewidth=0.5, zorder=0)
            # Horizontal two-line labels avoid collisions between adjacent tracks.
            ax.set_ylabel(ylabel, fontsize=label_fontsize, rotation=0,
                          ha='right', va='center', linespacing=1.25)
            ax.yaxis.set_label_coords(-0.045, 0.5)
            if metric == 'LD_BLOCK_BP':
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            if metric == 'VARIANT_COUNT':
                ax.legend(handles=[
                    Patch(facecolor=track_colors[0],
                          alpha=1.0, label='Total variants'),
                    Patch(facecolor=rare_track_colors[0],
                          label=f'Rare variants\nMAF < {rare_maf_threshold:g}'),
                ], loc='lower right', bbox_to_anchor=(1, 1.05), ncol=2,
                    frameon=False, fontsize=tick_fontsize - 0.5,
                    handlelength=1.5, columnspacing=1.4, borderaxespad=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='both', direction='out', width=0.9,
                           length=3, pad=3, labelsize=tick_fontsize)
            if metric == 'TajimaD':
                ax.axhline(0, color='0.4', linewidth=0.6)
            if metric not in ('TajimaD', 'LD_BLOCK_BP'):
                lower_bound = fill_baseline
                if metric != 'MAF' or maf_ymin is not None:
                    upper_bound = ax.get_ylim()[1]
                    if upper_bound <= lower_bound:
                        upper_bound = lower_bound + \
                            max(0.01, abs(lower_bound) * 0.1)
                    ax.set_ylim(lower_bound, upper_bound)
                    # Include the exact lower bound, avoiding a crowded nearby tick.
                    ticks = MaxNLocator(nbins=3, min_n_ticks=2).tick_values(
                        lower_bound, upper_bound)
                    ticks = ticks[(ticks > lower_bound + 0.12 * (upper_bound - lower_bound))
                                  & (ticks <= upper_bound)]
                    ax.yaxis.set_major_locator(
                        FixedLocator(np.r_[lower_bound, ticks]))
                    if metric == 'MAF':
                        ax.yaxis.set_major_formatter(FuncFormatter(
                            lambda value, position: f'{value:g}'))
            if metric == 'VARIANT_COUNT':
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda value, position: f'{value / 1000:g}k' if abs(value) >= 1000 else f'{value:g}'))
            if metric == 'PI':
                ax.ticklabel_format(axis='y', style='sci',
                                    scilimits=(-3, 3), useMathText=True)
        track_axes[-1].set_xlim(0, chromosome_offset / 1e6)
        track_axes[-1].set_xticks(chromosome_ticks, chromosome_order)
        track_axes[-1].set_xlabel('Chromosome',
                                  fontsize=label_fontsize, labelpad=8)
        if title:
            window_size = track_df.attrs.get('window_size', int(
                (track_plot_df['END'] - track_plot_df['START'] + 1).max()))
            track_fig.suptitle(f'{title} | {window_size / 1e6:g} Mb windows',
                               fontsize=title_fontsize, fontweight='normal')
        if output_prefix is not None:
            prefix = os.fspath(output_prefix)
            os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)
            files = {kind: f'{prefix}{suffix}' for kind, suffix in
                     [('pdf', '.pdf'), ('png', '.png'), ('csv', '.csv.gz')]}
            track_plot_df.to_csv(files['csv'], index=False, compression='gzip')
            track_fig.savefig(files['pdf'], bbox_inches='tight')
            track_fig.savefig(files['png'], dpi=300, bbox_inches='tight')
    return {'figure': track_fig, 'axes': track_axes, 'plot_data': track_plot_df, 'files': files}

# gene level contexts


def load_gene_info_df(ref_gff_file):
    """
    Load chromosome-assigned gene identities and coordinates.

    Parameters
    ----------
    ref_gff_file : str
        GFF/GFF.gz; retain top-level gene features on names starting with Chr.

    Returns
    -------
    pandas.DataFrame
        ID, CHROM, START, END, STRAND in annotation order, using 1-based inclusive
        endpoints. Reads GFF; no files are written.
    """
    gff_dict = read_gff_file(ref_gff_file)['gene']
    genes_in_chrs = [gene_id for gene_id in gff_dict.keys(
    ) if gff_dict[gene_id].chr_id.startswith('Chr')]
    gene_contexts_df = pd.DataFrame({'ID': genes_in_chrs, 'CHROM': [gff_dict[gene_id].chr_id for gene_id in genes_in_chrs], 'START': [gff_dict[gene_id].start for gene_id in genes_in_chrs], 'END': [
                                    gff_dict[gene_id].end for gene_id in genes_in_chrs], 'STRAND': [gff_dict[gene_id].strand for gene_id in genes_in_chrs]})
    return gene_contexts_df


def get_expression_contexts(gene_contexts_df, TPMs_csv_file, min_expr=2.0, high_var_threshold=1.0):

    """
    Classify expression using thresholded TPM and LOWESS residuals.

    Parameters
    ----------
    gene_contexts_df : pandas.DataFrame
        Gene table with ID; updated in place.
    TPMs_csv_file : str
        CSV/CSV.gz with gene IDs in the first column and one numeric TPM column per sample.
    min_expr : float
        TPM values <= this threshold become zero; default 2.0.
    high_var_threshold : float
        Standardized residual cutoff; high > cutoff, stable < cutoff; default 1.0.

    Returns
    -------
    pandas.DataFrame
        Input with no_expression_genes, high_variance_genes and stably_expressed_genes.
        LOWESS fraction is 0.1 on log2 mean/variance; stable genes also require expression
        in every sample. Genes absent from TPM are false in all categories. Reads TPM;
        writes no files.
    """
    gene_tpm_df = pd.read_csv(TPMs_csv_file, sep=",", header=0, index_col=0)
    gene_tpm_df = gene_tpm_df[gene_tpm_df.index.isin(
        set(gene_contexts_df['ID'].values))]

    gene_tpm_df = gene_tpm_df[gene_tpm_df > min_expr]
    gene_tpm_df.fillna(0, inplace=True)
    no_expr_gene_list = gene_tpm_df.index[gene_tpm_df.sum(
        axis=1) == 0].tolist()

    mean_values = np.mean(gene_tpm_df, axis=1)
    var_values = np.var(gene_tpm_df, axis=1)
    log_mean_values = np.log2(mean_values + 1)
    log_var_values = np.log2(var_values + 1)

    loess = sm.nonparametric.lowess(
        log_var_values, log_mean_values, frac=0.1)  # LOWESS smoothing fraction.
    loess_mean = loess[:, 0]
    loess_var = loess[:, 1]

    loess_expected_func = interpolate.interp1d(
        loess_mean, loess_var, kind='linear', fill_value='extrapolate')
    # Standardize residuals.
    # Predict log variance from log mean.
    loess_predicted_var = loess_expected_func(log_mean_values)
    # Subtract the LOWESS expectation.
    residuals = log_var_values - loess_predicted_var
    # Scale by the residual standard deviation.
    residuals_std = np.std(residuals)
    # Standardize residuals.
    standardized_residuals = residuals / residuals_std

    # Select high-variance genes.
    high_residual_genes = standardized_residuals > high_var_threshold
    high_residual_genes = gene_tpm_df.index[high_residual_genes].tolist()

    # Retain genes below the residual cutoff.
    low_residual_genes = standardized_residuals < high_var_threshold
    # Require positive retained mean expression.
    good_expr_genes = log_mean_values > 0

    # Require retained expression in every sample.
    gene_exp_sample_num = (gene_tpm_df > 0).sum(axis=1)
    all_expr_gene_list = gene_exp_sample_num[gene_exp_sample_num ==
                                             gene_tpm_df.shape[1]].index.tolist()
    all_expr_gene_list = set(all_expr_gene_list)
    all_expr_genes = gene_tpm_df.index.isin(all_expr_gene_list)

    housekeeping_genes = low_residual_genes & good_expr_genes & all_expr_genes
    housekeeping_genes = gene_tpm_df.index[housekeeping_genes].tolist()

    gene_contexts_df['no_expression_genes'] = gene_contexts_df['ID'].apply(
        lambda x: x in no_expr_gene_list)
    gene_contexts_df['high_variance_genes'] = gene_contexts_df['ID'].apply(
        lambda x: x in high_residual_genes)
    gene_contexts_df['stably_expressed_genes'] = gene_contexts_df['ID'].apply(
        lambda x: x in housekeeping_genes)

    return gene_contexts_df


def get_GCNs_contexts(gene_contexts_df, gcns_edge_file, gcns_module_file, cut_off=0.1):

    """
    Classify module membership and within-module network hubs.

    Parameters
    ----------
    gene_contexts_df : pandas.DataFrame
        Gene table with ID; updated in place.
    gcns_edge_file : str
        Headerless TSV/TSV.gz: gene1, gene2, numeric weight.
    gcns_module_file : str
        Gzip text with tab-separated gene IDs, one module per line.
    cut_off : float
        Minimum retained edge weight, inclusive; default 0.1.

    Returns
    -------
    pandas.DataFrame
        Input with module_hub_genes, module_genes and non_module_genes. Graphs with >=5
        vertices qualify; hubs have degree >2 and >=2 centrality z-scores >1. Module genes
        exclude hubs. Reads network/module files, displays module progress, writes no files.
    """
    all_genes = set(gene_contexts_df['ID'].tolist())

    # Read the supplied network.
    # Load weighted edges.
    edge_df = pd.read_csv(gcns_edge_file, sep="\t", header=None, names=[
                          "gene1", "gene2", "weight"])
    edge_df["weight"] = edge_df["weight"].astype(float)

    # Apply the edge-weight threshold.
    filtered_edge_df = edge_df[edge_df["weight"] >= cut_off]

    # Load module memberships for undirected within-module graphs.
    with gzip.open(gcns_module_file, 'rt') as f:
        module_lines = f.readlines()

    module_dict = {}
    num = 0
    for line in module_lines:
        genes = line.strip().split('\t')
        module_dict[f'Module_{num}'] = set(genes)
        num += 1

    # module_gene_num = {k: len(v) for k, v in module_dict.items()}
    # module_gene_num_df = pd.DataFrame.from_dict(module_gene_num, orient='index', columns=['GeneNum'])
    # module_gene_num_df['GeneNum'].describe()

    # Calculate hubs independently within each module.
    hub_gene_list = []

    for module_name, gene_set in tqdm(module_dict.items(), desc="Classifying network modules", unit="module"):
        module_edge_df = filtered_edge_df[
            (filtered_edge_df["gene1"].isin(gene_set)) &
            (filtered_edge_df["gene2"].isin(gene_set))
        ]

        # Compute centralities on the retained within-module graph.
        edges = list(zip(module_edge_df["gene1"], module_edge_df["gene2"]))
        g_module = ig.Graph.TupleList(edges, directed=False)

        if len(g_module.vs) < 5:
            continue

        # Degree centrality (normalized)
        degree_module = g_module.degree(mode="all")
        degree_norm_module = [d / (len(g_module.vs) - 1)
                              for d in degree_module]  # Normalize by the maximum possible degree.
        degree_zscore_module = (
            degree_norm_module - np.mean(degree_norm_module)) / np.std(degree_norm_module)

        # Closeness centrality (normalized)
        closeness_module = g_module.closeness(normalized=True)
        closeness_zscore_module = (
            closeness_module - np.mean(closeness_module)) / np.std(closeness_module)

        # Betweenness centrality (normalized)
        betweenness_module = g_module.betweenness(directed=False)
        betweenness_norm_module = [
            b / ((len(g_module.vs) - 1) * (len(g_module.vs) - 2) / 2) for b in betweenness_module]
        betweenness_zscore_module = (betweenness_norm_module - np.mean(
            betweenness_norm_module)) / np.std(betweenness_norm_module)

        # Eigenvector centrality
        eigenvector_module = g_module.eigenvector_centrality()
        # zscore normalization
        eigenvector_norm_module = (
            eigenvector_module - np.mean(eigenvector_module)) / np.std(eigenvector_module)

        module_df = pd.DataFrame({
            "gene": g_module.vs["name"],
            "degree": degree_module,
            "degree_zscore": degree_zscore_module,
            "closeness_zscore": closeness_zscore_module,
            "betweenness_zscore": betweenness_zscore_module,
            "eigenvector_zscore": eigenvector_norm_module
        })

        # Hubs require two centrality z-scores >1 and degree >2; graphs have >=5 vertices.
        hub_df = module_df[(module_df[['degree_zscore', 'closeness_zscore',
                            'betweenness_zscore', 'eigenvector_zscore']] > 1).sum(axis=1) >= 2]
        hub_df = hub_df[hub_df['degree'] > 2]

        hub_gene_list.extend(hub_df['gene'].tolist())

    hub_genes = set(hub_gene_list) & all_genes

    other_module_genes = set()
    for module_name, gene_set in module_dict.items():
        other_module_genes = other_module_genes.union(gene_set)
    other_module_genes = other_module_genes - hub_genes
    other_module_genes = other_module_genes & all_genes

    other_genes = all_genes - hub_genes - other_module_genes

    print(f"Hub genes: {len(hub_genes)}")
    print(f"Other module genes: {len(other_module_genes)}")
    print(f"Other genes: {len(other_genes)}")

    gene_contexts_df = get_ID_list_contexts(
        gene_contexts_df, list(hub_genes), context_name='module_hub_genes')
    gene_contexts_df = get_ID_list_contexts(gene_contexts_df, list(
        other_module_genes), context_name='module_genes')
    gene_contexts_df = get_ID_list_contexts(
        gene_contexts_df, list(other_genes), context_name='non_module_genes')

    return gene_contexts_df


def get_gene_family_contexts(gene_contexts_df, gene_family_file):
    """
    Classify sorghum genes from hierarchical orthogroup assignments.

    Parameters
    ----------
    gene_contexts_df : pandas.DataFrame
        Gene table with ID; updated in place.
    gene_family_file : str
        TSV/TSV.gz with HOG and Sbic; Sbic lists genes separated by comma-space.

    Returns
    -------
    pandas.DataFrame
        Input with orphan_genes, single_copy_genes and multi_copy_genes. Multiple-copy
        classification takes precedence across groups; unassigned genes are orphan. Reads
        HOG; writes no files.
    """
    poaceae_OG_df = pd.read_csv(gene_family_file, sep="\t")
    Sbic_OG_df = poaceae_OG_df[['HOG', 'Sbic']]
    Sbic_OG_df.dropna(inplace=True)
    Sbic_OG_df['GeneCount'] = Sbic_OG_df['Sbic'].apply(
        lambda x: len(x.split(', ')))
    multiple_copy_genes = []
    for _, row in Sbic_OG_df.iterrows():
        if row['GeneCount'] > 1:
            genes = row['Sbic'].split(', ')
            multiple_copy_genes.extend(genes)
    multiple_copy_genes = list(set(multiple_copy_genes))

    all_og_genes = []
    for _, row in Sbic_OG_df.iterrows():
        genes = row['Sbic'].split(', ')
        all_og_genes.extend(genes)
    all_og_genes = list(set(all_og_genes))

    single_copy_genes = list(set(all_og_genes) - set(multiple_copy_genes))

    all_genes = set(gene_contexts_df['ID'].tolist())
    orphan_genes = all_genes - \
        set(single_copy_genes) - set(multiple_copy_genes)

    gene_contexts_df['orphan_genes'] = gene_contexts_df['ID'].apply(
        lambda x: True if x in orphan_genes else False)
    gene_contexts_df['single_copy_genes'] = gene_contexts_df['ID'].apply(
        lambda x: True if x in single_copy_genes else False)
    gene_contexts_df['multi_copy_genes'] = gene_contexts_df['ID'].apply(
        lambda x: True if x in multiple_copy_genes else False)

    return gene_contexts_df


def get_gene_recombination_contexts(gene_contexts_df, recombination_file):

    """
    Classify genes by mean overlapping recombination-map rates.

    Parameters
    ----------
    gene_contexts_df : pandas.DataFrame
        Gene table with ID, CHROM, START, END; inclusive coordinates, updated in place.
    recombination_file : str
        CSV/CSV.gz containing chr, bp and meanCmPerBp; drop missing rows and convert to cM/Mb.

    Returns
    -------
    pandas.DataFrame
        Input with low/med/high_recombination_genes. Mean over intersecting bp..bp+999
        intervals is unweighted. Quartiles use mapped genes; low <Q25, medium [Q25,Q75),
        high >=Q75. Unmapped genes are false. Reads map, displays progress, writes no files.
    """
    recombination_df = pd.read_csv(recombination_file, sep=",", header=0)
    recombination_df.dropna(inplace=True, ignore_index=True)
    recombination_df['meanCmPerMb'] = recombination_df['meanCmPerBp'] * 1e6

    recombination_dict = {}
    for i, r in tqdm(recombination_df.iterrows(), total=len(recombination_df), desc="Building recombination intervals"):
        recombination_dict.setdefault(r['chr'], InterLap()).add(
            (r['bp'], r['bp']+999, r['meanCmPerMb']))

    gene_avg_recombination_rates = {}
    for i, r in tqdm(gene_contexts_df.iterrows(), total=len(gene_contexts_df), desc="Calculating gene average recombination rates"):
        gene_id = r['ID']
        gene_chr = r['CHROM']
        gene_start = r['START']
        gene_end = r['END']
        if gene_chr in recombination_dict:
            overlapping_intervals = list(
                recombination_dict[gene_chr].find((gene_start, gene_end)))
            if len(overlapping_intervals) > 0:
                avg_recombination_rate = np.mean(
                    [interval[2] for interval in overlapping_intervals])
                gene_avg_recombination_rates[gene_id] = float(
                    avg_recombination_rate)

    gene_avg_recombination_df = pd.DataFrame(list(
        gene_avg_recombination_rates.items()), columns=['ID', 'avg_recombination_rate'])
    p25 = gene_avg_recombination_df['avg_recombination_rate'].quantile(0.25)
    p75 = gene_avg_recombination_df['avg_recombination_rate'].quantile(0.75)
    low_recombination_genes = gene_avg_recombination_df[gene_avg_recombination_df['avg_recombination_rate'] < p25]['ID'].tolist(
    )
    med_recombination_genes = gene_avg_recombination_df[(gene_avg_recombination_df['avg_recombination_rate'] >= p25) & (
        gene_avg_recombination_df['avg_recombination_rate'] < p75)]['ID'].tolist()
    high_recombination_genes = gene_avg_recombination_df[
        gene_avg_recombination_df['avg_recombination_rate'] >= p75]['ID'].tolist()

    gene_contexts_df = get_ID_list_contexts(
        gene_contexts_df, low_recombination_genes, context_name='low_recombination_genes')
    gene_contexts_df = get_ID_list_contexts(
        gene_contexts_df, med_recombination_genes, context_name='med_recombination_genes')
    gene_contexts_df = get_ID_list_contexts(
        gene_contexts_df, high_recombination_genes, context_name='high_recombination_genes')

    return gene_contexts_df


def get_site_recombination_contexts(site_contexts_df, recombination_file, min_CmPerMb=3):

    """
    Classify sites using recombination-map intervals and thresholds.

    Parameters
    ----------
    site_contexts_df : pandas.DataFrame
        Site table with CHROM/POS; updated in place.
    recombination_file : str
        CSV/CSV.gz with chr, bp, meanCmPerBp; inclusive bp..bp+999 intervals.
    min_CmPerMb : float
        Low/medium boundary in cM/Mb; default 3.0. Q75 is calculated from map rows strictly
        above this threshold.

    Returns
    -------
    pandas.DataFrame
        Input with low (<threshold), medium [threshold,Q75), high (>=Q75) booleans.
        Overlapping interval rates are averaged; unmapped sites are false. Reads map, writes
        no files.
    """
    recombination_df = pd.read_csv(recombination_file, sep=",", header=0)
    recombination_df.dropna(inplace=True, ignore_index=True)
    recombination_df['meanCmPerMb'] = recombination_df['meanCmPerBp'] * 1e6

    p75 = recombination_df[recombination_df['meanCmPerMb']
                           > min_CmPerMb]['meanCmPerMb'].quantile(0.75)

    recombination_dict = {}
    for i, r in tqdm(recombination_df.iterrows(), total=len(recombination_df), desc="Building recombination intervals"):
        recombination_dict.setdefault(r['chr'], InterLap()).add(
            (r['bp'], r['bp']+999, r['meanCmPerMb']))

    site_contexts_df['recombination_rate'] = site_contexts_df.apply(lambda row: np.mean([interval[2] for interval in recombination_dict[row['CHROM']].find(
        (row['POS'], row['POS']))]) if row['CHROM'] in recombination_dict and len(list(recombination_dict[row['CHROM']].find((row['POS'], row['POS'])))) > 0 else np.nan, axis=1)

    site_contexts_df['low_recombination'] = site_contexts_df['recombination_rate'] < min_CmPerMb
    site_contexts_df['med_recombination'] = (site_contexts_df['recombination_rate'] >= min_CmPerMb) & (
        site_contexts_df['recombination_rate'] < p75)
    site_contexts_df['high_recombination'] = site_contexts_df['recombination_rate'] >= p75

    site_contexts_df.drop(columns=['recombination_rate'], inplace=True)

    return site_contexts_df


if __name__ == "__main__":
    data_dir = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum"
    ref_repeat_gff_file = f"{data_dir}/data/source/Sbicolor_730_v5.1.repeatmasked_assembly_v5.0.gff3.gz"
    ref_genome_file = f"{data_dir}/data/source/Sbicolor_730_v5.0.fa.gz"

    from src.feature import read_gff_file, read_fasta_by_faidx
    from interlap import InterLap

    ref_genome = read_fasta_by_faidx(ref_genome_file)

    repeat_features = read_gff_file(ref_repeat_gff_file)['similarity']
    chr_repeat_interlap = {}
    for repeat_id in repeat_features:
        repeat = repeat_features[repeat_id]
        chr_repeat_interlap.setdefault(
            repeat.chr_id, InterLap()).add((repeat.start, repeat.end))

    list(chr_repeat_interlap['Chr01'])

    for chr_id in chr_repeat_interlap:
        repeat_len_sum = sum(end - start for start,
                             end in chr_repeat_interlap[chr_id])
        chr_len = ref_genome[chr_id].len()
        print(f"{chr_id}: {repeat_len_sum}/{chr_len} {repeat_len_sum/chr_len:.2%}")

    pass


def save_context_table(table_df, output_file):
    """
    Save a complete context or window table with atomic replacement.

    Parameters
    ----------
    table_df : pandas.DataFrame
        One complete table, retaining its original row order and index.
    output_file : str
        Destination HDF path. The parent directory must already exist.

    Returns
    -------
    str
        Destination path. Writes key ``df`` to a temporary sibling, then
        replaces the destination after successful completion. Existing
        annotations are retained by the caller; this function does not
        validate cache compatibility or recompute missing annotations.
    """
    temporary_file = f'{output_file}.tmp'
    table_df.to_hdf(temporary_file, key='df', mode='w')
    os.replace(temporary_file, output_file)
    print(f'[{datetime.now():%H:%M:%S}] Saved: {output_file}; shape={table_df.shape}', flush=True)
    return output_file


# Genomic feature and environmental association landscapes.

def _landscape_windows(track_df, window_size):
    """Select and validate complete, ordered reference-length chromosome bins."""
    chromosomes = [f'Chr{i:02d}' for i in range(1, 11)]
    windows = track_df.loc[track_df['CHROM'].isin(chromosomes), ['CHROM', 'START', 'END']].copy()
    windows = windows.sort_values(['CHROM', 'START']).reset_index(drop=True)
    if windows.duplicated(['CHROM', 'START', 'END']).any():
        raise ValueError('Duplicate landscape windows would make alignment ambiguous.')
    if set(windows['CHROM']) != set(chromosomes):
        raise ValueError('Landscape inputs must contain Chr01 through Chr10.')
    for chrom, frame in windows.groupby('CHROM', sort=False):
        starts, ends = frame['START'].to_numpy(), frame['END'].to_numpy()
        if not np.array_equal(starts, 1 + np.arange(len(frame)) * window_size):
            raise ValueError(f'Windows on {chrom} must be consecutive {window_size} bp bins.')
        if (ends < starts).any() or (ends - starts + 1 > window_size).any():
            raise ValueError(f'Invalid window boundaries on {chrom}.')
        if not np.array_equal(ends[:-1], starts[1:] - 1):
            raise ValueError(f'Nonterminal windows on {chrom} must have full width.')
    return windows


def _landscape_zscore(values):
    """Standardize finite values with population SD, retaining missing values."""
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if valid.any():
        std = values[valid].std(ddof=0)
        result[valid] = (values[valid] - values[valid].mean()) / std if std > 0 else 0
    return result


def _save_landscape_csv(frame, file_path):
    """Publish a complete small table by atomic replacement."""
    temporary_file = f'{file_path}.tmp'
    frame.to_csv(temporary_file, index=False, compression='gzip')
    os.replace(temporary_file, file_path)


def _landscape_counts(frame, positions, columns, windows, window_size):
    """Count boolean categories, retaining NaN in bins without background objects."""
    selected = frame.loc[frame['CHROM'].isin(windows['CHROM']), ['CHROM'] + columns].copy()
    selected_positions = positions.loc[selected.index].to_numpy()
    chromosome_ends = windows.groupby('CHROM')['END'].max()
    if (not np.isfinite(selected_positions).all() or (selected_positions < 1).any()
            or (selected_positions > selected['CHROM'].map(chromosome_ends).to_numpy()).any()):
        raise ValueError('Context coordinates fall outside reference chromosome bounds.')
    selected['START'] = ((selected_positions - 1) // window_size) * window_size + 1
    groups = selected.groupby(['CHROM', 'START'], sort=False)
    counts = groups[columns].sum().reindex(pd.MultiIndex.from_frame(windows[['CHROM', 'START']]))
    background = groups.size().reindex(counts.index)
    counts.loc[background.isna(), :] = np.nan
    # A coordinate beyond the last reference bin must not silently disappear.
    if not groups.size().index.isin(counts.index).all():
        raise ValueError('Context coordinates fall outside the landscape windows.')
    return counts.reset_index(drop=True)


def prepare_genomic_feature_landscape(track_data_csv, gene_contexts_h5, site_contexts_h5,
                                      flowering_genes_csv, output_prefix,
                                      window_size=1_000_000, force_recompute=False):
    """Prepare raw and standardized feature tracks and cloned-gene markers.

    Workflow
    --------
    1. Reuse a complete output set, unless force_recompute is requested.
    2. Read existing 02 landscape fractions and population statistics unchanged.
    3. Count gene classes by midpoint and QC site classes by position.
    4. Match Cloned=1 genes to body coordinates and mark every overlapping bin.
    5. Standardize each feature across all plotted bins and save small CSVs.

    Parameters
    ----------
    track_data_csv : str
        Existing 02 genome tracks CSV.gz with CHROM/START/END, GENIC_FRACTION,
        TE_FRACTION, ACR_FRACTION, UMR_FRACTION, VARIANT_COUNT, MAF, PI and
        meanCmPerMb. Population statistics and BED coordinates are not recalculated.
    gene_contexts_h5, site_contexts_h5 : str
        Project context HDF files, key 'df'. Genes use ID/CHROM/START/END;
        QC sites use CHROM/POS and boolean context columns. Coordinates are
        1-based inclusive. Conservation counts are restricted to genic_15kb.
    flowering_genes_csv : str
        Source CSV containing unique Gene_IDs, Gene Name and binary Cloned.
        Only Cloned=1 bodies, without flanks, generate markers.
    output_prefix : str
        Saves .features.csv.gz, .cloned_genes.csv.gz and .cloned_bins.csv.gz.
    window_size : int
        Existing track resolution, default 1 Mb; retain Chr01 through Chr10.
    force_recompute : bool
        Rebuild all feature outputs. Reuse assumes inputs/parameters have not
        changed; explicitly recompute after changing contexts or the cloned list.

    Returns
    -------
    dict
        features, cloned_genes, cloned_bins DataFrames and files mapping.
        Features contain PANEL/GROUP/TRACK/LABEL, coordinates, RAW_VALUE and
        Z_SCORE. Empty category backgrounds are NaN; present backgrounds with
        no category members are zero. Constant finite tracks standardize to zero.
        Reads context tables one at a time; writes outputs atomically per file.
    """
    from time import perf_counter

    started = perf_counter()
    files = {name: f'{output_prefix}.{name}.csv.gz'
             for name in ['features', 'cloned_genes', 'cloned_bins']}
    if not force_recompute and all(os.path.exists(path) for path in files.values()):
        print(f'[{datetime.now():%H:%M:%S}] Reusing feature landscape: {output_prefix}', flush=True)
        return {**{name: pd.read_csv(path) for name, path in files.items()}, 'files': files}
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    print(f'[{datetime.now():%H:%M:%S}] Preparing feature landscape', flush=True)
    track_df = pd.read_csv(track_data_csv)
    windows = _landscape_windows(track_df, window_size)
    aligned = windows.merge(track_df, on=['CHROM', 'START', 'END'], validate='one_to_one')
    specs = [
        ('main', 'Genomic structure', 'GENIC_FRACTION', 'Genic', 'Oranges'),
        ('main', 'Genomic structure', 'TE_FRACTION', 'TEs', 'Oranges'),
        ('main', 'Epigenomic features', 'ACR_FRACTION', 'ACRs', 'Reds'),
        ('main', 'Epigenomic features', 'UMR_FRACTION', 'UMRs', 'Reds'),
        ('main', 'Gene expression', 'no_expression_genes', 'Genes with no detected expr.', 'Blues'),
        ('main', 'Gene expression', 'high_variance_genes', 'Genes with high var. expr.', 'Blues'),
        ('main', 'Gene expression', 'stably_expressed_genes', 'Genes with stable expr.', 'Blues'),
        ('main', 'Gene regulatory networks', 'non_module_genes', 'Non-module gene', 'Greens'),
        ('main', 'Gene regulatory networks', 'module_genes', 'Module gene', 'Greens'),
        ('main', 'Gene regulatory networks', 'module_hub_genes', 'Module hub', 'Greens'),
        ('main', 'riboSNitches', 'riboSNitch', 'riboSNitches', 'Purples'),
        ('supp', 'Recombination rate', 'meanCmPerMb', 'Recombination', 'Oranges'),
        ('supp', 'Population genetic', 'VARIANT_COUNT', '#Variants', 'YlOrBr'),
        ('supp', 'Population genetic', 'MAF', 'MAF', 'YlOrBr'),
        ('supp', 'Population genetic', 'PI', 'π', 'YlOrBr'),
        ('supp', 'Phylogenetic conservation', 'non_CEs', 'Non-evaluable', 'BuGn'),
        ('supp', 'Phylogenetic conservation', 'accelerated_sites', 'Accelerated', 'BuGn'),
        ('supp', 'Phylogenetic conservation', 'neutral_sites', 'Neutral', 'BuGn'),
        ('supp', 'Phylogenetic conservation', 'conserved_sites', 'Conserved', 'BuGn'),
    ]
    raw = {name: aligned[name].to_numpy() for name in
           ['GENIC_FRACTION', 'TE_FRACTION', 'ACR_FRACTION', 'UMR_FRACTION',
            'meanCmPerMb', 'VARIANT_COUNT', 'MAF', 'PI']}
    gene_df = pd.read_hdf(gene_contexts_h5, key='df')
    gene_columns = [spec[2] for spec in specs[4:10]]
    gene_counts = _landscape_counts(gene_df, (gene_df['START'] + gene_df['END']) // 2,
                                    gene_columns, windows, window_size)
    raw.update({name: gene_counts[name].to_numpy() for name in gene_columns})
    flowering_df = pd.read_csv(flowering_genes_csv)
    if not flowering_df['Cloned'].isin([0, 1]).all():
        raise ValueError('Cloned must contain only 0 or 1.')
    cloned = flowering_df.loc[flowering_df['Cloned'].eq(1), ['Gene_ID', 'Gene Name']]
    if cloned['Gene_ID'].duplicated().any() or gene_df['ID'].duplicated().any():
        raise ValueError('Duplicate gene IDs make cloned-coordinate matching ambiguous.')
    cloned = cloned.merge(gene_df[['ID', 'CHROM', 'START', 'END']], left_on='Gene_ID',
                          right_on='ID', how='left', validate='one_to_one').drop(columns='ID')
    if (cloned[['CHROM', 'START', 'END']].isna().any().any()
            or (cloned['START'] < 1).any() or (cloned['START'] > cloned['END']).any()
            or not cloned['CHROM'].isin(windows['CHROM']).all()):
        raise ValueError('Every cloned gene must match valid plotted gene coordinates.')
    chromosome_ends = windows.groupby('CHROM')['END'].max()
    if (cloned['END'] > cloned['CHROM'].map(chromosome_ends)).any():
        raise ValueError('Cloned gene coordinates exceed reference chromosome bounds.')
    cloned_bins = windows.copy()
    overlaps = []
    for row in windows.itertuples(index=False):
        matches = cloned.loc[(cloned['CHROM'] == row.CHROM)
                             & (cloned['START'] <= row.END) & (cloned['END'] >= row.START), 'Gene_ID']
        overlaps.append(';'.join(matches))
    cloned_bins['GENE_IDS'] = overlaps
    cloned_bins['IS_CLONED'] = cloned_bins['GENE_IDS'].ne('').astype(int)
    matched_ids = set(';'.join(overlaps).split(';')) - {''}
    if matched_ids != set(cloned['Gene_ID']):
        raise ValueError('Some cloned genes do not overlap any reference window.')
    del gene_df, gene_counts
    print(f'[{datetime.now():%H:%M:%S}] Counting QC site contexts', flush=True)
    site_df = pd.read_hdf(site_contexts_h5, key='df')
    site_counts = _landscape_counts(site_df, site_df['POS'], ['riboSNitch'], windows, window_size)
    raw['riboSNitch'] = site_counts['riboSNitch'].to_numpy()
    conservation_columns = [spec[2] for spec in specs[-4:]]
    genic_sites = site_df.loc[site_df['genic_15kb']]
    conservation_counts = _landscape_counts(genic_sites, genic_sites['POS'],
                                            conservation_columns, windows, window_size)
    raw.update({name: conservation_counts[name].to_numpy() for name in conservation_columns})
    del site_df, genic_sites, site_counts, conservation_counts
    feature_frames = []
    for order, (panel, group, track, label, cmap) in enumerate(tqdm(specs, desc='Preparing feature tracks', unit='track')):
        frame = windows.copy()
        frame['PANEL'], frame['GROUP'], frame['TRACK'] = panel, group, track
        frame['LABEL'], frame['CMAP'], frame['TRACK_ORDER'] = label, cmap, order
        frame['RAW_VALUE'] = raw[track]
        frame['Z_SCORE'] = _landscape_zscore(raw[track])
        feature_frames.append(frame)
    features = pd.concat(feature_frames, ignore_index=True)
    result = {'features': features, 'cloned_genes': cloned, 'cloned_bins': cloned_bins}
    for name, frame in result.items():
        _save_landscape_csv(frame, files[name])
    print(f'[{datetime.now():%H:%M:%S}] Saved {len(windows):,} bins, {len(specs)} tracks and '
          f'{len(cloned)} cloned genes ({perf_counter() - started:.1f}s): {output_prefix}', flush=True)
    return {**result, 'files': files}


def prepare_gea_landscape(track_data_csv, gwas_dir, environment_metadata_csv,
                          output_prefix, window_size=1_000_000, top_fraction=0.05,
                          force_recompute=False):
    """Prepare top-association hypergeometric enrichment on landscape bins.

    Parameters
    ----------
    track_data_csv : str
        Existing 02 genome tracks defining the complete reference-length bins.
    gwas_dir : str
        Genome-wide GEMMA root. Results are read from
        {gwas_dir}/{env_id}/{env_id}.gemma/{env_id}.gemma.gwas_df.h5, key 'df'.
        Required fields are CHROM/POS/P_VALUE (the project's Wald P values).
    environment_metadata_csv : str
        Source metadata with unique env_id values; all listed variables are used.
    output_prefix : str
        Save .gea.csv.gz with env_id, coordinates, N_TESTED, N_TOP, TOTAL_TESTED,
        TOTAL_TOP, TOP_P_THRESHOLD, P_VALUE and NEG_LOG10_P.
    window_size : int
        Existing track resolution, default 1 Mb.
    top_fraction : float
        Genome-wide P-value quantile, default 0.05; include ties. The background
        is all rows of each association table, including unplotted sequences.
    force_recompute : bool
        Rebuild the result. Cached inputs/parameters are assumed unchanged;
        explicitly recompute after changing GWAS inputs or top_fraction.

    Returns
    -------
    dict
        gea DataFrame and files mapping. Process one environmental table at a
        time with progress; save once after all variables finish. Empty bins
        have P=1. Zero P values map to NEG_LOG10_P=500. No GWAS is launched.

    Workflow
    --------
    Read the existing window grid, select top sites per environment, count
    tested/top sites in each bin, evaluate hypergeom.sf(k-1, M, K, n), and save.
    """
    from scipy.stats import hypergeom
    from time import perf_counter

    output_file = f'{output_prefix}.gea.csv.gz'
    if not force_recompute and os.path.exists(output_file):
        print(f'[{datetime.now():%H:%M:%S}] Reusing GEA landscape: {output_file}', flush=True)
        return {'gea': pd.read_csv(output_file), 'files': {'gea': output_file}}
    if not 0 < top_fraction < 1:
        raise ValueError('top_fraction must be between zero and one.')
    started = perf_counter()
    windows = _landscape_windows(pd.read_csv(track_data_csv), window_size)
    metadata = pd.read_csv(environment_metadata_csv)
    if metadata['env_id'].duplicated().any():
        raise ValueError('Environmental IDs must be unique.')
    window_index = pd.MultiIndex.from_frame(windows[['CHROM', 'START']])
    frames = []
    for env_id in tqdm(metadata['env_id'], desc='Summarizing GEA enrichment', unit='environment'):
        file_path = f'{gwas_dir}/{env_id}/{env_id}.gemma/{env_id}.gemma.gwas_df.h5'
        gwas = pd.read_hdf(file_path, key='df')[['CHROM', 'POS', 'P_VALUE']]
        if gwas.empty or not np.isfinite(gwas['P_VALUE']).all() or not gwas['P_VALUE'].between(0, 1).all():
            raise ValueError(f'GWAS P values for {env_id} must be finite and in [0, 1].')
        threshold = gwas['P_VALUE'].quantile(top_fraction)
        gwas['IS_TOP'] = gwas['P_VALUE'].le(threshold)
        total_tested, total_top = len(gwas), int(gwas['IS_TOP'].sum())
        plotted = gwas.loc[gwas['CHROM'].isin(windows['CHROM'])].copy()
        chromosome_ends = windows.groupby('CHROM')['END'].max()
        if (not np.isfinite(plotted['POS']).all() or (plotted['POS'] < 1).any()
                or (plotted['POS'] > plotted['CHROM'].map(chromosome_ends)).any()):
            raise ValueError(f'GWAS coordinates for {env_id} exceed reference chromosome bounds.')
        plotted['START'] = ((plotted['POS'] - 1) // window_size) * window_size + 1
        grouped = plotted.groupby(['CHROM', 'START']).agg(N_TESTED=('POS', 'size'), N_TOP=('IS_TOP', 'sum'))
        if not grouped.index.isin(window_index).all():
            raise ValueError(f'GWAS coordinates for {env_id} fall outside reference windows.')
        counts = grouped.reindex(window_index, fill_value=0)
        frame = windows.copy()
        frame['env_id'] = env_id
        frame[['N_TESTED', 'N_TOP']] = counts[['N_TESTED', 'N_TOP']].to_numpy(dtype=int)
        frame['TOTAL_TESTED'], frame['TOTAL_TOP'] = total_tested, total_top
        frame['TOP_P_THRESHOLD'] = threshold
        frame['P_VALUE'] = hypergeom.sf(frame['N_TOP'] - 1, total_tested, total_top, frame['N_TESTED'])
        pvalues = frame['P_VALUE'].to_numpy()
        transformed = np.full(len(frame), 500.0)
        positive = pvalues > 0
        transformed[positive] = -np.log10(pvalues[positive])
        frame['NEG_LOG10_P'] = np.maximum(transformed, 0)
        frames.append(frame)
        del gwas, plotted, grouped, counts
    gea = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    _save_landscape_csv(gea, output_file)
    print(f'[{datetime.now():%H:%M:%S}] Saved GEA landscape ({perf_counter() - started:.1f}s): {output_file}', flush=True)
    return {'gea': gea, 'files': {'gea': output_file}}


def plot_genomic_gea_landscape(feature_data_csv, gea_data_csv, cloned_bins_csv,
                               environment_metadata_csv, environment_corr_csv,
                               output_prefix, panel='main', window_size=1_000_000,
                               figsize=(29.7 * 0.7, 21), label_fontsize=16,
                               title_fontsize=24, chromosome_fontsize=18,
                               hspace=0.275, wspace=0.15, highlight_env_ids=None):
    """Plot aligned genomic feature and GEA landscapes from saved statistics.

    Parameters
    ----------
    feature_data_csv, gea_data_csv, cloned_bins_csv : str
        CSV.gz outputs of the two prepare functions. No context, VCF or GWAS
        source is read and missing statistics are never recalculated.
    environment_metadata_csv, environment_corr_csv : str
        Metadata with env_id/env_abbr and a square Pearson correlation matrix
        indexed by the same environmental IDs. Cluster distance is 1-abs(r)
        with average linkage. Labels use original IDs before leaf reordering.
    output_prefix : str
        Export .pdf, .png (300 dpi) and actual plotted .csv.gz. The latter
        includes raw/transformed values, DISPLAY_ROW (zero at the top), and
        color limits, plus cloned IDs and the displayed window markers.
    panel : {'main', 'supp'}
        Feature selection; both figures share GEA colors and ordering.
    window_size : int
        Existing bin resolution in bp; terminal bins retain their true END.
    figsize : tuple
        Canvas width/height in inches, matching the old figure by default.
    label_fontsize, title_fontsize, chromosome_fontsize : float
        Label, panel-title and chromosome-title sizes.
    hspace, wspace : float
        GridSpec gaps; the left label/tree column has width weight 80.
    highlight_env_ids : sequence of str or None
        Environmental IDs whose custom abbreviation labels use bold red text.

    Returns
    -------
    dict
        figure, axes (groups plus GEA tree/heatmaps and cloned marker axes),
        plot_data and files. Does not call show or alter global plot style.
        Z-score limits are -1/2.5; GEA uses the shared 5th/95th percentiles.
        Gray bands span three bins, are centered and clipped to each chromosome.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    if panel not in ['main', 'supp']:
        raise ValueError("panel must be 'main' or 'supp'.")
    features = pd.read_csv(feature_data_csv)
    gea = pd.read_csv(gea_data_csv)
    cloned_bins = pd.read_csv(cloned_bins_csv, keep_default_na=False)
    metadata = pd.read_csv(environment_metadata_csv).set_index('env_id')
    corr = pd.read_csv(environment_corr_csv, index_col=0)
    if not metadata.index.is_unique or not corr.index.is_unique or not corr.columns.is_unique:
        raise ValueError('Environmental IDs must be unique.')
    environment_ids = metadata.index.tolist()
    highlight_env_ids = set(highlight_env_ids or [])
    if (set(corr.index) != set(environment_ids) or set(corr.columns) != set(environment_ids)
            or set(gea['env_id']) != set(environment_ids)):
        raise ValueError('GEA, metadata and environmental correlations must have identical IDs.')
    corr = corr.loc[environment_ids, environment_ids]
    correlations = corr.to_numpy(dtype=float)
    if (not np.isfinite(correlations).all()
            or not np.allclose(correlations, correlations.T)
            or not np.allclose(np.diag(correlations), 1)
            or (np.abs(correlations) > 1 + 1e-10).any()):
        raise ValueError('Environmental correlations must be finite, symmetric and have diagonal one.')
    distance = 1 - np.clip(np.abs(correlations), 0, 1)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance), method='average')
    leaf_ids = dendrogram(tree, labels=environment_ids, no_plot=True)['ivl']
    windows = _landscape_windows(cloned_bins, window_size)
    window_keys = ['CHROM', 'START', 'END']
    window_index = pd.MultiIndex.from_frame(windows[window_keys])
    if gea.duplicated(['env_id'] + window_keys).any():
        raise ValueError('Duplicate GEA bins would make plotted alignment ambiguous.')
    gea_matrix = gea.pivot(index='env_id', columns=window_keys, values='NEG_LOG10_P')
    if set(gea_matrix.columns) != set(window_index):
        raise ValueError('GEA and feature windows must match exactly.')
    gea_matrix = gea_matrix.reindex(index=leaf_ids, columns=window_index)
    if not np.isfinite(gea_matrix.to_numpy()).all():
        raise ValueError('GEA statistics must exist for every environment and window.')
    gea_limits = np.percentile(gea_matrix.to_numpy(), [5, 95])
    feature_norm = Normalize(vmin=-1, vmax=2.5)
    gea_norm = Normalize(vmin=gea_limits[0], vmax=gea_limits[1])
    selected = features.loc[features['PANEL'].eq(panel)].sort_values('TRACK_ORDER')
    if selected.empty:
        raise ValueError(f'No feature tracks found for {panel}.')
    groups = list(selected.groupby('GROUP', sort=False))
    chromosomes = windows['CHROM'].drop_duplicates().tolist()
    sizes = windows.groupby('CHROM', sort=False).size().tolist()
    heights = [frame['TRACK'].nunique() for _, frame in groups] + [len(leaf_ids), 1]
    fig = plt.figure(figsize=figsize)
    grid = GridSpec(len(heights), len(chromosomes) + 1, figure=fig,
                    width_ratios=[80] + sizes, height_ratios=heights,
                    hspace=hspace, wspace=wspace, left=0.125, right=0.99,
                    bottom=0.035, top=0.97)
    axes, plot_frames = {}, []
    display_offset = 0

    def add_title(label, row_axes):
        fig.text(0.015, row_axes[0].get_position().y1 + 0.002, label,
                 ha='left', va='bottom', fontsize=title_fontsize)

    def add_dividers(row_axes, bottom=None, top=None):
        y0 = row_axes[0].get_position().y0 if bottom is None else bottom
        y1 = row_axes[0].get_position().y1 if top is None else top
        for left, right in zip(row_axes[:-1], row_axes[1:]):
            x = (left.get_position().x1 + right.get_position().x0) / 2
            fig.add_artist(plt.Line2D([x, x], [y0, y1], transform=fig.transFigure,
                                     color='black', linewidth=1, zorder=10))

    for row, (group, frame) in enumerate(groups):
        tracks = frame['TRACK'].drop_duplicates().tolist()
        if frame.duplicated(['TRACK'] + window_keys).any():
            raise ValueError(f'Duplicate feature bins in {group}.')
        for track in tracks:
            track_index = pd.MultiIndex.from_frame(frame.loc[frame['TRACK'].eq(track), window_keys])
            if set(track_index) != set(window_index):
                raise ValueError(f'Feature track {track} has incompatible windows.')
        matrix = frame.pivot(index='TRACK', columns=window_keys, values='Z_SCORE')
        matrix = matrix.reindex(index=tracks, columns=window_index)
        labels = frame.drop_duplicates('TRACK').set_index('TRACK')['LABEL']
        cmap = frame['CMAP'].iloc[0]
        row_axes = []
        for column, chrom in enumerate(chromosomes, start=1):
            ax = fig.add_subplot(grid[row, column])
            data = matrix.xs(chrom, axis=1, level='CHROM').to_numpy(dtype=float)
            ax.pcolormesh(np.ma.masked_invalid(data), cmap=cmap, norm=feature_norm,
                          shading='flat', edgecolors='none')
            ax.set(xlim=(0, data.shape[1]), ylim=(len(tracks), 0))
            ax.axis('off')
            for y in range(len(tracks) + 1):
                ax.axhline(y, color='white', linewidth=4)
            if row == 0:
                ax.set_title(chrom, fontsize=chromosome_fontsize, pad=10)
            row_axes.append(ax)
        label_ax = fig.add_subplot(grid[row, 0])
        label_ax.set(xlim=(0, 1), ylim=(len(tracks), 0))
        label_ax.axis('off')
        for index, track in enumerate(tracks):
            label_ax.text(0.95, index + 0.5, labels[track], ha='right', va='center', fontsize=label_fontsize)
        axes[group] = {'labels': label_ax, 'heatmaps': row_axes}
        add_title(group, row_axes)
        add_dividers(row_axes)
        exported = frame.copy()
        exported['VALUE'] = exported['Z_SCORE']
        exported['DISPLAY_ROW'] = exported['TRACK'].map({track: display_offset + i for i, track in enumerate(tracks)})
        exported['COLOR_MIN'], exported['COLOR_MAX'] = -1, 2.5
        plot_frames.append(exported)
        display_offset += len(tracks)

    gea_row = len(groups)
    tree_ax = fig.add_subplot(grid[gea_row, 0])
    dendrogram(tree, orientation='left', labels=environment_ids, ax=tree_ax, color_threshold=0.5)
    # Dendrogram coordinates increase from bottom to top; row zero is the bottom leaf.
    leaf_positions = tree_ax.get_yticks()
    tree_ax.tick_params(axis='both', length=0, labelbottom=False, labeltop=False,
                        labelleft=False, labelright=False)
    tree_ax.set_yticks([])
    x_left, x_right = tree_ax.get_xlim()
    label_space = (x_left - x_right) * 0.7
    for y, env_id in zip(leaf_positions, leaf_ids):
        tree_ax.text(x_right - label_space * 0.95, y, metadata.loc[env_id, 'env_abbr'],
                     ha='right', va='center', fontsize=label_fontsize,
                     color='red' if env_id in highlight_env_ids else 'black',
                     fontweight='bold' if env_id in highlight_env_ids else 'normal')
    tree_ax.set_xlim(x_left, x_right - label_space)
    for spine in tree_ax.spines.values():
        spine.set_visible(False)
    gea_axes = []
    for column, chrom in enumerate(chromosomes, start=1):
        ax = fig.add_subplot(grid[gea_row, column])
        data = gea_matrix.xs(chrom, axis=1, level='CHROM').to_numpy(dtype=float)
        ax.pcolormesh(data, cmap='Reds', norm=gea_norm, shading='flat', edgecolors='none')
        ax.set(xlim=(0, data.shape[1]), ylim=(0, len(leaf_ids)))
        ax.axis('off')
        gea_axes.append(ax)
    axes['GEA signals'] = {'tree': tree_ax, 'heatmaps': gea_axes}
    add_title('GEA signals', gea_axes)
    add_dividers(gea_axes)
    gea_export = gea.copy()
    gea_export['PANEL'], gea_export['GROUP'] = panel, 'GEA signals'
    gea_export['TRACK'] = gea_export['env_id']
    gea_export['LABEL'] = gea_export['env_id'].map(metadata['env_abbr'])
    gea_export['RAW_VALUE'], gea_export['VALUE'] = gea_export['P_VALUE'], gea_export['NEG_LOG10_P']
    gea_export['DISPLAY_ROW'] = gea_export['env_id'].map(
        {env_id: display_offset + len(leaf_ids) - 1 - i for i, env_id in enumerate(leaf_ids)})
    gea_export['COLOR_MIN'], gea_export['COLOR_MAX'] = gea_limits
    plot_frames.append(gea_export)

    cloned_row = gea_row + 1
    band_top = grid[0, 1].get_position(fig).y1
    band_bottom = grid[cloned_row, 1].get_position(fig).y0
    cloned_axes = []
    marker_matrix = cloned_bins.set_index(window_keys).reindex(window_index)
    for column, chrom in enumerate(chromosomes, start=1):
        ax = fig.add_subplot(grid[cloned_row, column])
        ax.set(xlim=(0, sizes[column - 1]), ylim=(0, 1))
        ax.axis('off')
        box = ax.get_position()
        markers = marker_matrix.xs(chrom, level='CHROM')['IS_CLONED'].to_numpy(dtype=int)
        nbin = len(markers)
        for index in np.flatnonzero(markers):
            # Bin centers are c+0.5; clip the three-bin band to chromosome edges.
            left = max(0, index + 0.5 - 1.5)
            right = min(nbin, index + 0.5 + 1.5)
            rect = Rectangle((box.x0 + left / nbin * box.width, band_bottom),
                             (right - left) / nbin * box.width, band_top - band_bottom,
                             transform=fig.transFigure, facecolor='black', edgecolor='none', alpha=0.2, zorder=3)
            fig.add_artist(rect)
        ax.scatter(np.flatnonzero(markers) + 0.5, np.full(markers.sum(), 0.5),
                   s=28, color='black', clip_on=True, zorder=4)
        cloned_axes.append(ax)
    label_ax = fig.add_subplot(grid[cloned_row, 0])
    label_ax.set(xlim=(0, 1), ylim=(0, 1))
    label_ax.axis('off')
    label_ax.text(0.95, 0.5, 'Cloned Flowering Genes', ha='right', va='center', fontsize=label_fontsize)
    axes['Flowering genes'] = {'labels': label_ax, 'markers': cloned_axes}
    add_title('Flowering genes', cloned_axes)
    add_dividers(cloned_axes, bottom=band_bottom, top=band_top)
    cloned_export = marker_matrix.reset_index()
    cloned_export['PANEL'], cloned_export['GROUP'] = panel, 'Flowering genes'
    cloned_export['TRACK'], cloned_export['LABEL'] = 'Cloned', 'Cloned Flowering Genes'
    cloned_export['RAW_VALUE'], cloned_export['VALUE'] = cloned_export['IS_CLONED'], cloned_export['IS_CLONED']
    cloned_export['DISPLAY_ROW'] = display_offset + len(leaf_ids)
    cloned_export['BAND_WIDTH_BINS'] = 3
    plot_frames.append(cloned_export)
    plot_data = pd.concat(plot_frames, ignore_index=True).sort_values(['DISPLAY_ROW', 'CHROM', 'START']).reset_index(drop=True)
    files = {kind: f'{output_prefix}.{extension}' for kind, extension in
             [('pdf', 'pdf'), ('png', 'png'), ('csv', 'csv.gz')]}
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    fig.savefig(files['pdf'], format='pdf', facecolor='none', edgecolor='none')
    fig.savefig(files['png'], dpi=300, facecolor='white')
    _save_landscape_csv(plot_data, files['csv'])
    print(f'[{datetime.now():%H:%M:%S}] Exported {panel} landscape: {output_prefix}', flush=True)
    return {'figure': fig, 'axes': axes, 'plot_data': plot_data, 'files': files}
