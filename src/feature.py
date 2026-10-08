import gzip
from collections import OrderedDict
from BCBio import GFF
from pyfaidx import Fasta
import re
from .utils import cmd_run, have_file
import pandas as pd


def complement_seq(seq):
    """just complement"""
    if 'U' in seq:
        complement = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A'}
    else:
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    bases = list(seq)
    bases = [complement.get(base, base) for base in bases]
    bases = ''.join(bases)
    return bases


def reverse_complement(seq, RNA=False):
    """
    #>>> seq = "TCGGinsGCCC"
    #>>> print "Reverse Complement:"
    #>>> print(reverse_complement(seq))
    #GGGCinsCCGA
    """
    alt_map = {'ins': '0'}
    if RNA:
        complement = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A'}
    else:
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    # for k, v in alt_map.iteritems():
    for k, v in alt_map.items():
        seq = seq.replace(k, v)
    bases = list(seq)
    bases = reversed([complement.get(base, base) for base in bases])
    bases = ''.join(bases)
    # for k, v in alt_map.iteritems():
    for k, v in alt_map.items():
        bases = bases.replace(v, k)
    return bases


def reverse_seq(seq):
    """just reverse"""
    return "".join(reversed(seq))


def sub_seq(seq, site_start, site_end, strand, RNA):
    """get sub set of a seq"""
    start = min(site_start, site_end)
    end = max(site_start, site_end)
    sub_seq_out = seq[start - 1:end]

    if RNA is True:
        sub_seq_out = sub_seq_out.replace("T", "U")
    if strand == "-":
        sub_seq_out = reverse_complement(sub_seq_out, RNA)

    return sub_seq_out


def faidx2xyx(pyfaidx_record, write_seq_mem):
    try:
        seq_name = pyfaidx_record.long_name
    except:
        seq_name = pyfaidx_record.name

    if write_seq_mem:
        return FastaRecord(seq_name, str(pyfaidx_record), pyfaidx_record)
    else:
        return FastaRecord(seq_name, None, pyfaidx_record)


def read_fasta_by_faidx(file_name, write_seq_mem=False):
    if file_name.split(".")[-1] == 'gz':
        f, o, e = cmd_run("samtools faidx %s" % file_name)
        if f is False:
            if 'please use bgzip' in e:
                f1, _, _ = cmd_run(f"gunzip {file_name}")
                f2, _, _ = cmd_run(f"bgzip {file_name[:-3]}")
                f3, _, _ = cmd_run(f"samtools faidx {file_name}")
                if f1 is False or f2 is False or f3 is False:
                    raise EnvironmentError("fasta file not suit for pyfaidx")

    try:
        output_dict = OrderedDict()
        for i in Fasta(file_name):
            output_dict[i.name] = faidx2xyx(i, write_seq_mem)
        return output_dict
    except:
        try:
            cmd_run("samtools faidx %s" % file_name)
            output_dict = OrderedDict()
            for i in Fasta(file_name):
                output_dict[i.name] = faidx2xyx(i, write_seq_mem)
            return output_dict
        except:
            raise EnvironmentError("fasta file still not suit for pyfaidx")


class BioSeq(object):
    def __init__(self, seq, seqname=None, seq_type='nucl'):
        self.seqname = seqname
        self.seq = seq
        self.seq_type = seq_type

    nucl = ["A", "T", "C", "G", "U"]
    nucl_degenerate = ["M", "S", "W", "B", "D", "R", "H", "Y", "V", "K"]
    any_nucl = ["N"]
    prot = ["A", "P", "B", "Q", "C", "R", "D", "S", "E", "T", "F", "U", "G", "V", "H", "W", "I", "Y", "K", "Z", "L",
            "M", "N"]
    any_prot = ["X"]
    stop_site = ["*"]
    gap_site = ['-']

    def __str__(self):
        return self.seq

    def seqs_length(self):
        return len(self.seq)

    def seq_clean(self, degenerate_sites_allowed=False, gap_allowed=False,
                  translation_stop_allowed=False, replace_flag=True):
        # make code
        if self.seq_type == 'nucl':
            seq_type_flag = 0b0001
        elif self.seq_type == 'prot':
            seq_type_flag = 0b0000

        dsa_flag = degenerate_sites_allowed
        if dsa_flag:
            dsa_flag = 0b0010

        ga_flag = gap_allowed
        if ga_flag:
            ga_flag = 0b0100

        tsa_flag = translation_stop_allowed
        if tsa_flag:
            tsa_flag = 0b1000

        bit_field = seq_type_flag + dsa_flag + ga_flag + tsa_flag

        if bit_field == 0b0111:
            all_allowed = self.nucl + self.any_nucl + self.nucl_degenerate + self.gap_site
            safe_remove = []
            safe_replace = self.any_nucl
        elif bit_field == 0b0011:
            all_allowed = self.nucl + self.any_nucl + self.nucl_degenerate
            safe_remove = self.gap_site
            safe_replace = self.any_nucl
        elif bit_field == 0b0101:
            all_allowed = self.nucl + self.any_nucl + self.gap_site
            safe_remove = []
            safe_replace = self.any_nucl
        elif bit_field == 0b0001:
            all_allowed = self.nucl + self.any_nucl
            safe_remove = self.gap_site
            safe_replace = self.any_nucl
        elif bit_field == 0b0000:
            all_allowed = self.prot + self.any_prot
            safe_remove = self.gap_site + self.stop_site
            safe_replace = self.any_prot
        elif bit_field == 0b0100:
            all_allowed = self.prot + self.any_prot + self.gap_site
            safe_remove = self.stop_site
            safe_replace = self.any_prot
        elif bit_field == 0b1100:
            all_allowed = self.prot + self.any_prot + self.gap_site + self.stop_site
            safe_remove = []
            safe_replace = self.any_prot
        elif bit_field == 0b1000:
            all_allowed = self.prot + self.any_prot + self.stop_site
            safe_remove = self.gap_site
            safe_replace = self.any_prot
        else:
            raise ValueError("Bad input flag")

        new_seq = self.seq
        char_site = set(new_seq)
        bad_flag = False
        for i in char_site:
            if i in all_allowed:
                continue
            elif i in safe_remove:
                if i == "*":
                    new_seq = re.sub("\*", "", new_seq)
                else:
                    new_seq = re.sub(i, "", new_seq)
            else:
                if replace_flag:
                    if i == "*":
                        new_seq = re.sub("\*", "", new_seq)
                    else:
                        new_seq = re.sub(i, safe_replace[0], new_seq)
                bad_flag = True

        if replace_flag:
            self.seq = new_seq
        else:
            if bad_flag:
                return False
            else:
                return True

    def seqs_upper(self):
        self.seq = self.seq.upper()

    def seqs_lower(self):
        self.seq = self.seq.lower()

    def no_wrap(self):
        self.seq = re.sub("\n", "", self.seq)

    def wrap(self, line_length=75):
        raw_seq = str(self.seq)
        raw_seq = re.sub("\n", "", raw_seq)
        new_seq = ""

        for j in (raw_seq[x: x + line_length] for x in range(0, len(raw_seq), line_length)):
            new_seq += j + "\n"

        self.seq = new_seq

        return new_seq

    def sub(self, site_start, site_end, strand, RNA):
        return sub_seq(self.seq, site_start, site_end, strand, RNA)

    def RNA(self):
        return self.seq.replace("T", "U")

    def reverse(self):
        return reverse_seq(self.seq)

    def complement(self):
        return complement_seq(self.seq)

    def reverse_complement(self):
        return complement_seq(reverse_seq(self.seq))

    def write_to_file(self, output_file, wrap_length=75):
        raw_seq = str(self.seq)
        with open(output_file, 'a') as f:
            f.write(">%s\n" % self.seqname)
            for x in range(0, len(raw_seq), wrap_length):
                # print(self.seqname, x)
                i = raw_seq[x: x + wrap_length]
                f.write(i + "\n")

    # def write_to_file(self, output_file, wrap_length=75):
    #     print(self.seqname, len(self.seq))
    #     self.wrap(line_length=wrap_length)
    #     print('ok wrap')
    #     # self.seq = re.sub("\n", "", self.seq)
    #     with open(output_file, 'a') as f:
    #         f.write(">%s\n%s" % (self.seqname, self.seq))


class FastaRecord(BioSeq):
    def __init__(self, seqname, seq=None, faidx=None):
        super(FastaRecord, self).__init__(seq)
        self.seqname = seqname
        self.faidx = faidx
        self.qualifiers = re.findall('\[([^\[\]]+)=([^\[\]]+)\]', seqname)

    def seqname_short(self):
        name_short = re.search('^(\S+)', self.seqname).group(1)
        return name_short

    @property
    def seq(self):
        if self.faidx is not None:
            return str(self.faidx)
        else:
            return self._seq

    @seq.setter
    def seq(self, value):
        self._seq = value

    def read_seq_to_mem(self):
        self._seq = str(self.faidx)

    def sub(self, site_start, site_end, strand, RNA):
        if self.faidx is not None:
            output = str(self.faidx[site_start - 1:site_end])
            if RNA is True:
                output = output.replace("T", "U")
            if strand == "-":
                output = reverse_complement(output, RNA)
            return output
        else:
            return sub_seq(self.seq, site_start, site_end, strand, RNA)

    def len(self):
        if hasattr(self, 'faidx'):
            return len(self.faidx)
        else:
            return len(self.seq)

    def get_GC_ratio(self):
        gc_now = (self.seq.count('C') + self.seq.count('G') +
                  self.seq.count('c') + self.seq.count('g'))/self.len()
        return gc_now


class ChrLoci(object):
    def __init__(self, chr_id=None, strand=None, start=None, end=None, sp_id=None):
        self.chr_id = chr_id
        self.sp_id = sp_id

        if strand is None:
            self.strand = strand
        elif strand == "+" or str(strand) == '1':
            self.strand = "+"
        elif strand == "-" or str(strand) == '-1':
            self.strand = "-"
        else:
            self.strand = None

        if end is not None and start is not None:
            self.start = min(int(start), int(end))
            self.end = max(int(start), int(end))
            self._range = (self.start, self.end)
            self.length = abs(self.end - self.start) + 1

    # def __eq__(self, other):
    #     """Implement equality by comparing all the location attributes."""
    #     if not isinstance(other, ChrLoci):
    #         return False
    #     return self.start == other.start and \
    #            self.end == other.end and \
    #            self.strand == other.strand and \
    #            self.chr_id == other.chr_id

    def get_fancy_name(self):
        if self.strand is None:
            self.fancy_name = "%s:%d-%d" % (self.chr_id, self.start, self.end)
        else:
            self.fancy_name = "%s:%d-%d:%s" % (self.chr_id,
                                               self.start, self.end, self.strand)
        return self.fancy_name

    def __str__(self):
        try:
            self.get_fancy_name()
            if self.sp_id:
                return "%s %s" % (self.sp_id, self.fancy_name)
            else:
                return self.fancy_name
        except:
            return "No detail range"

    __repr__ = __str__

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, value):
        start, end = value
        if end is not None and start is not None:
            self.start = min(int(start), int(end))
            self.end = max(int(start), int(end))
            self._range = (self.start, self.end)

    def get_sequence(self, data_fasta_file, RNA=False):
        sequence = read_fasta_by_faidx(data_fasta_file)[self.chr_id].sub(self._range[0], self._range[1], self.strand,
                                                                         RNA)
        self.get_fancy_name()
        return BioSeq(sequence, self.fancy_name)

    def get_sequence_quick(self, seq_dict, RNA=False):
        """
        seq_dict should load sequence into memory, and Bioseq not have faidx attr
        """
        sequence = seq_dict[self.chr_id].sub(
            self._range[0], self._range[1], self.strand, RNA)
        self.get_fancy_name()
        return BioSeq(sequence, self.fancy_name)

    def len(self):
        return abs(self.end - self.start) + 1

    def __eq__(self, other):
        return self.chr_id == other.chr_id and self.strand == other.strand and self.start == other.start and self.end == other.end and self.sp_id == other.sp_id

    def __hash__(self):
        return hash(id(self))


class GenomeFeature(ChrLoci):
    def __init__(self, id=None, type=None, chr_loci=None, qualifiers={}, sub_features=None, chr_id=None, strand=None, start=None, end=None, sp_id=None):
        if chr_loci:
            super(GenomeFeature, self).__init__(chr_id=chr_loci.chr_id, strand=chr_loci.strand, start=chr_loci.start,
                                                end=chr_loci.end, sp_id=chr_loci.sp_id)
        else:
            super(GenomeFeature, self).__init__(
                chr_id=chr_id, strand=strand, start=start, end=end, sp_id=sp_id)

        self.id = id
        self.type = type
        if chr_loci:
            self.chr_loci = chr_loci
        else:
            self.chr_loci = ChrLoci(chr_id=chr_id, strand=strand,
                                    start=start, end=end, sp_id=sp_id)
        self.sub_features = sub_features
        self.qualifiers = qualifiers

    def sgf_len(self):
        self.sgf_len_dir = {i: 0 for i in list(
            set([sgf.type for sgf in self.sub_features]))}

        for sgf in self.sub_features:
            sgf_len = abs(sgf.start - sgf.end) + 1
            self.sgf_len_dir[sgf.type] += sgf_len

    def get_bottom_subfeatures(self):
        bgf_list = []
        if sgf is None or len(self.sub_features) == 0:
            bgf_list.append(sgf)
            return bgf_list
        else:
            for sgf in sgf.sub_features:
                bgf_list.extend(self.get_bottom_subfeatures(sgf))
        return bgf_list

    def __eq__(self, other):
        return self.id == other.id and self.chr_loci == other.chr_loci and self.type == other.type

    def __hash__(self):
        return hash(id(self))


def ft2cl(feature_location, chr_id):
    """
    create ChrLoci by FeatureLocation from BCBio
    """
    return ChrLoci(chr_id=chr_id, strand=feature_location.strand, start=feature_location.start + 1,
                   end=feature_location.end)


def sf2gf(sf, chr_id):
    """
    create GenomeFeature by SeqFeature from BCBio
    """
    sf_cl = ft2cl(sf.location, chr_id)
    gf = GenomeFeature(id=sf.id, type=sf.type, chr_loci=sf_cl)
    gf.qualifiers = sf.qualifiers

    # parse sub_feature
    if hasattr(sf, 'sub_features') and len(sf.sub_features) != 0:
        gf.sub_features = []
        for sub_sf in sf.sub_features:
            gf.sub_features.append(sf2gf(sub_sf, chr_id))

    return gf


def read_gff_file(gff_file):
    """
    Read GFF/GFF.gz into ordered feature dictionaries.

    Parameters
    ----------
    gff_file : str
        GFF path parsed by BCBio; gzip is selected by the .gz suffix.

    Returns
    -------
    collections.OrderedDict
        Top-level feature type to ID to GenomeFeature, with recursive
        subfeatures and qualifiers preserved. Coordinates are converted
        from BCBio half-open locations to 1-based inclusive endpoints.
        Missing IDs receive sequential NoID names. Reads only; no files saved.
    """

    feature_dict = OrderedDict()
    no_id = 0
    opener = gzip.open if str(gff_file).endswith(".gz") else open
    mode = "rt" if str(gff_file).endswith(".gz") else "r"

    with opener(gff_file, mode) as in_handle:
        for rec in GFF.parse(in_handle):
            for feature in rec.features:
                new_feature = sf2gf(feature, rec.id)
                feature_dict.setdefault(new_feature.type, OrderedDict())
                if new_feature.id == "":
                    new_feature.id = f"NoID_{no_id}"
                    no_id += 1
                feature_dict[new_feature.type][new_feature.id] = new_feature

    return feature_dict


def get_gene_range_df(gff_file):
    gene_range_csv = f'{gff_file}.GeneRange.csv'
    if have_file(gene_range_csv):
        gene_range_df = pd.read_csv(gene_range_csv)
    else:
        gene_dict = read_gff_file(gff_file)['gene']
        gene_id_list = list(gene_dict.keys())
        gene_range_df = pd.DataFrame()
        gene_range_df['GENE'] = gene_id_list
        gene_range_df['CHROM'] = [gene_dict[i].chr_id for i in gene_id_list]
        gene_range_df['START'] = [gene_dict[i].start for i in gene_id_list]
        gene_range_df['END'] = [gene_dict[i].end for i in gene_id_list]
        gene_range_df['STRAND'] = [gene_dict[i].strand for i in gene_id_list]
        gene_range_df.to_csv(gene_range_csv, index=False)
    return gene_range_df
