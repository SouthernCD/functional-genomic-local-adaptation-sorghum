"""Run one GCTB GWFM RC job from a GEMMA association file and a VCF.

The VCF provides LD genotypes, variant IDs, REF/ALT alleles, and the GWAS
sample count. Building a new LD cache requires a GCTB block file with columns
``Block Chrom StartBP EndBP`` (optional fifth column ``NSNP``). A complete
matching LD cache can be reused without a block file.

Example block file (whitespace or tab separated)::

    Block Chrom StartBP EndBP
    1 1 1 100000
    2 1 100001 200000

Blocks must be ordered, nonoverlapping, nonempty, and cover every SNP in the
VCF-derived PLINK BIM. If NSNP is supplied, it must match the BIM count.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess


REQUIRED_LD = ("ldm.info", "snp.info", "rsq0.5.pwld", "block1.eigen.bin")
REQUIRED_RESULT = (".snpRes", ".parRes", ".gcs", ".gcsRes", ".lcs", ".lcsRes")


def _vcf_stem(path: Path) -> str:
    name = path.name
    for suffix in (".vcf.gz", ".vcf.bgz", ".vcf"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError("--vcf must end in .vcf, .vcf.gz, or .vcf.bgz")


def _source_identity(path: Path) -> dict:
    stat = path.stat()
    return {"vcf": str(path.resolve()), "vcf_bytes": stat.st_size,
            "vcf_mtime_ns": stat.st_mtime_ns}


def _command(args: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        completed = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT,
                                   check=False)
    if completed.returncode:
        raise RuntimeError(f"command exited {completed.returncode}; inspect {log}")
    contents = log.read_text(errors="replace")
    if "Analysis finished:" not in contents and Path(args[0]).name == "gctb":
        raise RuntimeError(f"GCTB did not report completion; inspect {log}")


BLOCK_FORMAT = ("header: Block Chrom StartBP EndBP [NSNP]; rows: 1 1 1 100000; "
                "ordered, nonoverlapping blocks covering every BIM SNP")


def _chromosome(value: str) -> int:
    match = re.fullmatch(r"[Cc]hr0*([1-9][0-9]*)", value)
    if match:
        return int(match.group(1))
    if value.isdecimal() and int(value) >= 1:
        return int(value)
    raise ValueError(f"invalid chromosome label {value!r}; expected a positive number or Chr01")


def _read_blocks(path: Path) -> list[tuple[int, int, int, int, int | None]]:
    """Read a GCTB block map and normalize chromosome labels to integers."""
    blocks: list[tuple[int, int, int, int, int | None]] = []
    with path.open() as source:
        header = source.readline().split()
        if header not in (["Block", "Chrom", "StartBP", "EndBP"],
                          ["Block", "Chrom", "StartBP", "EndBP", "NSNP"]):
            raise ValueError(f"invalid block-file header in {path}; expected {BLOCK_FORMAT}")
        has_count = len(header) == 5
        previous_chrom, previous_end = 0, 0
        for line_number, line in enumerate(source, 2):
            fields = line.split()
            if len(fields) != len(header):
                raise ValueError(f"block-file row {line_number} needs {len(header)} columns: {BLOCK_FORMAT}")
            try:
                block = int(fields[0])
                chrom = _chromosome(fields[1])
                start, end = int(fields[2]), int(fields[3])
                count = int(fields[4]) if has_count else None
            except ValueError as exc:
                raise ValueError(f"invalid block-file row {line_number}: {exc}") from exc
            if block != len(blocks) + 1 or start < 1 or end < start:
                raise ValueError(f"block-file row {line_number} needs consecutive Block IDs and positive StartBP <= EndBP")
            if chrom < previous_chrom or (chrom == previous_chrom and start <= previous_end):
                raise ValueError(f"block-file row {line_number} is out of order or overlaps a preceding block")
            if count is not None and count < 1:
                raise ValueError(f"block-file row {line_number} has nonpositive NSNP")
            blocks.append((block, chrom, start, end, count))
            previous_chrom, previous_end = chrom, end
    if not blocks:
        raise ValueError(f"block file is empty: {path}; expected {BLOCK_FORMAT}")
    return blocks


def _write_validated_blocks(blocks: list[tuple[int, int, int, int, int | None]],
                            bim: Path, output: Path) -> None:
    """Require every BIM variant in exactly one nonempty block, then write GCTB format."""
    counts = [0] * len(blocks)
    index = 0
    previous_chrom, previous_pos = 0, 0
    with bim.open() as source:
        for row_number, line in enumerate(source, 1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"malformed PLINK BIM row {row_number}")
            chrom, pos = _chromosome(fields[0]), int(fields[3])
            if chrom < previous_chrom or (chrom == previous_chrom and pos < previous_pos):
                raise ValueError(f"PLINK BIM is not sorted at row {row_number}")
            previous_chrom, previous_pos = chrom, pos
            while index < len(blocks) and (blocks[index][1] < chrom or
                                           (blocks[index][1] == chrom and blocks[index][3] < pos)):
                index += 1
            if index == len(blocks) or blocks[index][1] != chrom or pos < blocks[index][2]:
                raise ValueError(f"block file does not cover BIM SNP at {chrom}:{pos} (row {row_number})")
            counts[index] += 1
    if not any(counts):
        raise ValueError("PLINK BIM has no variants")
    for block, observed in zip(blocks, counts):
        if not observed or (block[4] is not None and observed != block[4]):
            raise ValueError(f"block {block[0]} has {observed} BIM SNPs; expected {block[4] or 'at least one'}")
    with output.open("w") as destination:
        destination.write("Block Chrom StartBP EndBP\n")
        for number, chrom, start, end, _ in blocks:
            destination.write(f"{number} {chrom} {start} {end}\n")


def _cache_matches(directory: Path, identity: dict) -> bool:
    marker = directory / "COMPLETE.json"
    if not marker.is_file() or not all((directory / name).is_file()
                                       for name in REQUIRED_LD):
        return False
    try:
        recorded = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    return all(recorded.get(key) == value for key, value in identity.items())


def _normalize_bim_chromosomes(bim: Path) -> None:
    """Map sorghum Chr01..Chr10 to numeric labels without changing SNP IDs."""
    temporary = bim.with_suffix(".bim.partial")
    with bim.open() as source, temporary.open("w") as out:
        for line in source:
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"malformed PLINK BIM row: {line[:100]}")
            chrom = fields[0]
            match = re.fullmatch(r"[Cc]hr0*([1-9][0-9]*)", chrom)
            if match:
                fields[0] = str(int(match.group(1)))
            elif not chrom.isdecimal() or int(chrom) < 1:
                raise ValueError(f"unsupported chromosome label for GCTB: {chrom}")
            out.write("\t".join(fields) + "\n")
    temporary.replace(bim)


def _ld_ids(ld: Path):
    with (ld / "snp.info").open() as source:
        next(source)
        for line in source:
            fields = line.split()
            if len(fields) < 2:
                raise ValueError("malformed LD snp.info")
            yield fields[1]


def _make_intercept(ld: Path, target: Path) -> Path:
    with target.open("w") as out:
        out.write("SNP Intercept\n")
        for snp in _ld_ids(ld):
            out.write(f"{snp} 1\n")
    return target


def _open_vcf(vcf: Path):
    return gzip.open(vcf, "rt") if vcf.suffix in (".gz", ".bgz") else vcf.open()


def _variant_database(vcf: Path, database: Path) -> tuple[sqlite3.Connection, int]:
    """Build an on-disk ID/allele lookup, avoiding a genome-wide Python dict."""
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE variant (snp TEXT PRIMARY KEY, ref TEXT, alt TEXT)")
    sample_count = 0
    batch = []
    with _open_vcf(vcf) as source:
        for line in source:
            if line.startswith("#CHROM"):
                sample_count = len(line.rstrip("\n").split("\t")) - 9
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t", 5)
            if len(fields) < 5:
                raise ValueError("malformed VCF record")
            snp, ref, alt = fields[2:5]
            if not snp or snp == "." or "," in alt:
                raise ValueError(f"VCF needs unique nonempty IDs and biallelic records: {snp}")
            batch.append((snp, ref, alt))
            if len(batch) == 50_000:
                connection.executemany("INSERT INTO variant VALUES (?, ?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO variant VALUES (?, ?, ?)", batch)
    connection.commit()
    if sample_count < 1:
        raise ValueError("VCF header has no samples")
    return connection, sample_count


def _summary(gwas: Path, vcf: Path, destination: Path) -> dict:
    import pandas as pd

    db_path = destination.with_suffix(".variants.sqlite")
    connection, vcf_samples = _variant_database(vcf, db_path)
    sample_count = vcf_samples
    seen: set[str] = set()
    total = 0
    try:
        with destination.open("w") as out:
            out.write("SNP A1 A2 freq b se p N\n")
            for chunk in pd.read_csv(gwas, sep="\t", chunksize=50_000,
                                     dtype={"rs": str}):
                required = {"rs", "allele1", "allele0", "af", "beta", "se", "p_wald"}
                if not required.issubset(chunk):
                    raise ValueError(f"GEMMA columns missing: {required - set(chunk)}")
                if "n_miss" not in chunk:
                    raise ValueError("GEMMA requires n_miss to calculate N from VCF samples")
                if chunk.rs.duplicated().any() or any(snp in seen for snp in chunk.rs):
                    raise ValueError("duplicate GEMMA SNP ID")
                seen.update(chunk.rs)
                lookups = {}
                ids = chunk.rs.tolist()
                for offset in range(0, len(ids), 500):
                    batch = ids[offset:offset + 500]
                    marks = ",".join("?" for _ in batch)
                    lookups.update((snp, (ref, alt)) for snp, ref, alt in
                                   connection.execute(
                                       f"SELECT snp, ref, alt FROM variant WHERE snp IN ({marks})",
                                       batch))
                for row in chunk.itertuples(index=False):
                    if row.rs not in lookups:
                        raise ValueError(f"GEMMA SNP absent from VCF: {row.rs}")
                    ref, alt = lookups[row.rs]
                    if row.allele1 == alt and row.allele0 == ref:
                        freq, beta = float(row.af), float(row.beta)
                    elif row.allele1 == ref and row.allele0 == alt:
                        freq, beta = 1 - float(row.af), -float(row.beta)
                    else:
                        raise ValueError(f"GEMMA/VCF allele mismatch: {row.rs}")
                    se, p = float(row.se), float(row.p_wald)
                    n = sample_count - int(row.n_miss)
                    if "n_obs" in chunk and not pd.isna(row.n_obs) and int(row.n_obs) != n:
                        raise ValueError(f"n_obs differs from VCF samples - n_miss: {row.rs}")
                    if not (0 < freq < 1 and math.isfinite(beta) and
                            math.isfinite(se) and se > 0 and
                            math.isfinite(p) and 0 <= p <= 1 and n > 0):
                        raise ValueError(f"invalid GEMMA statistic: {row.rs}")
                    out.write(f"{row.rs} {alt} {ref} {freq:.9g} {beta:.9g} "
                              f"{se:.9g} {max(p, 5e-324):.9g} {n}\n")
                    total += 1
    finally:
        connection.close()
        db_path.unlink(missing_ok=True)
    if not total:
        raise ValueError("GEMMA association file is empty")
    return {"gwas_variants": total, "vcf_samples": vcf_samples,
            "assumed_gwas_samples": sample_count}


@dataclass
class GWFMJob:
    gwas: Path
    vcf: Path
    output: Path
    ldm_eigen: Path | None = None
    block_file: Path | None = None
    annot: Path | None = None
    threads: int = 4
    plink: str = "plink"
    gctb: str = "gctb"

    def _ld(self) -> Path:
        if self.ldm_eigen:
            ld = self.ldm_eigen.resolve()
            if not all((ld / name).is_file() for name in REQUIRED_LD):
                raise ValueError(f"--ldm-eigen must be an existing, complete GCTB eigen LD directory: {ld}; "
                                 f"to build LD from the VCF, omit --ldm-eigen and provide --block-file ({BLOCK_FORMAT})")
            return ld
        stem = _vcf_stem(self.vcf)
        vcf_identity = _source_identity(self.vcf)
        if self.block_file is None:
            candidates = [directory for directory in sorted(
                self.vcf.parent.glob(f"{stem}.ldm_eigen*"))
                if directory.is_dir() and _cache_matches(directory, vcf_identity)]
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise ValueError("multiple completed LD caches match this VCF; "
                                 "select one with --ldm-eigen or supply --block-file")
            raise ValueError(f"no completed LD cache found for {self.vcf}; "
                             f"provide --block-file to build one. Expected format: {BLOCK_FORMAT}")
        block_file = self.block_file.resolve()
        if not block_file.is_file():
            raise FileNotFoundError(f"--block-file does not exist: {block_file}; expected {BLOCK_FORMAT}")
        blocks = _read_blocks(block_file)
        digest = hashlib.sha256(block_file.read_bytes()).hexdigest()
        identity = {**vcf_identity, "block_sha256": digest}
        ld = self.vcf.with_name(f"{stem}.ldm_eigen_{digest[:12]}")
        marker = ld / "COMPLETE.json"
        if marker.is_file():
            if not _cache_matches(ld, identity):
                raise ValueError(f"LD cache does not match VCF or is incomplete: {ld}")
            return ld
        if ld.exists() and any(ld.iterdir()):
            raise ValueError(f"partial LD cache needs inspection: {ld}")
        ld.mkdir(parents=True, exist_ok=True)
        panel = ld / "panel"
        _command([self.plink, "--vcf", str(self.vcf), "--double-id",
                  "--keep-allele-order", "--allow-extra-chr", "--make-bed",
                  "--out", str(panel)],
                 ld / "plink.log")
        _normalize_bim_chromosomes(panel.with_suffix(".bim"))
        _write_validated_blocks(blocks, panel.with_suffix(".bim"), ld / "blocks.pos")
        # GCTB 2.5.5 can abort in multithreaded block LD construction.
        _command([self.gctb, "--bfile", str(panel), "--make-block-ldm",
                  "--block-info", str(ld / "blocks.pos"), "--thread",
                  "1", "--out", str(ld)], ld / "make_ldm.log")
        _command([self.gctb, "--ldm", str(ld), "--make-ldm-eigen",
                  "--thread", str(self.threads), "--out", str(ld / "eigen")],
                 ld / "make_eigen.log")
        if not all((ld / name).is_file() for name in REQUIRED_LD):
            raise RuntimeError(f"GCTB LD construction left missing files: {ld}")
        marker.write_text(json.dumps({**identity, "block_file": str(block_file)}, indent=2) + "\n")
        return ld

    def run(self) -> Path:
        self.gwas = self.gwas.resolve()
        self.vcf = self.vcf.resolve()
        self.output = self.output.resolve()
        if not self.gwas.is_file() or not self.vcf.is_file():
            raise FileNotFoundError("--gwas and --vcf must be existing files")
        if self.threads < 1:
            raise ValueError("--threads must be positive")
        marker = self.output / "COMPLETE.json"
        if marker.exists():
            raise FileExistsError(f"completed job already exists: {self.output}")
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {self.output}")
        ld = self._ld()
        self.output.mkdir(parents=True, exist_ok=True)
        annotation = (self.annot.resolve() if self.annot else
                      _make_intercept(ld, self.output / "intercept_only.txt"))
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        info = _summary(self.gwas, self.vcf, self.output / "sumstats.ma")
        args = [self.gctb, "--gwfm", "RC", "--ldm-eigen", str(ld),
                "--gwas-summary", str(self.output / "sumstats.ma"),
                "--annot", str(annotation), "--thread", str(self.threads),
                "--out", str(self.output / "rc")]
        _command(args, self.output / "gctb.log")
        if not all((self.output / f"rc{suffix}").is_file()
                   for suffix in REQUIRED_RESULT):
            raise RuntimeError("GCTB reported success but GWFM result files are missing")
        marker.write_text(json.dumps({"gwas": str(self.gwas), "vcf": str(self.vcf),
                                      "ldm_eigen": str(ld), "annot": str(annotation),
                                      "block_file": str(self.block_file.resolve()) if self.block_file else None,
                                      **info}, indent=2) + "\n")
        return self.output / "rc.snpRes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gwas", type=Path, required=True,
                        help="GEMMA gemma_output.assoc.txt")
    parser.add_argument("--vcf", type=Path, required=True,
                        help="genotypes matching the GEMMA variant IDs and alleles")
    parser.add_argument("--output", type=Path, required=True,
                        help="new, empty job output directory")
    parser.add_argument("--ldm-eigen", type=Path,
                        help="use an existing GCTB eigen LD directory")
    parser.add_argument("--block-file", type=Path,
                        help="required to build LD when no complete cache exists; "
                             "header: Block Chrom StartBP EndBP [NSNP]; "
                             "ordered, nonoverlapping blocks covering every BIM SNP")
    parser.add_argument("--annot", type=Path,
                        help="custom annotation; default is an all-one Intercept")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plink", default="plink")
    parser.add_argument("--gctb", default="gctb")
    print(GWFMJob(**vars(parser.parse_args())).run())


if __name__ == "__main__":
    # Example on rock01, from the repository root:
    # python3 -m src.gwfm \
    #   --gwas data/results/gwas/striga/striga.gemma/output/gemma_output.assoc.txt \
    #   --vcf data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz \
    #   --block-file data/results/bigsnpr_full_genome_100kb_k1p1_v1/blocks.pos \
    #   --threads 4 \
    #   --output data/results/gwas/striga/gwfm_rc
    # Per-SNP N is the VCF sample count minus GEMMA n_miss.
    main()
