"""Run bigsnpr::snp_cor and snp_ldsplit on PLINK bed/bim/fam data.

One R process imports only the selected chromosome(s), then handles each
chromosome separately. The result is a pandas block table; all candidate
solutions and the sparse correlations are retained in the output directory.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
import subprocess
from typing import Sequence

import pandas as pd


DEFAULT_RSCRIPT = Path(
    "/lustre/home/dobby/Program/agent_program/micromamba-root/"
    "envs/bigsnpr/bin/Rscript"
)
BACKEND = Path(__file__).with_name("ldsplit_backend.R")
BLOCK_COLUMNS = ["Block", "Chrom", "StartBP", "EndBP", "NSNP"]


def _prefix(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value.suffix in {".bed", ".bim", ".fam"}:
        value = value.with_suffix("")
    for suffix in (".bed", ".bim", ".fam"):
        if not Path(f"{value}{suffix}").is_file():
            raise FileNotFoundError(f"missing PLINK file: {value}{suffix}")
    return value


def _bim(prefix: Path, chromosome: str | int | None) -> list[dict]:
    rows: list[dict] = []
    selected = None if chromosome is None else str(chromosome)
    seen_chrom: set[str] = set()
    last_chrom: str | None = None
    last_pos = -1
    with Path(f"{prefix}.bim").open() as source:
        for index, line in enumerate(source, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"BIM row {index} does not have six columns")
            chrom, snp, _, pos_text, _, _ = fields
            pos = int(pos_text)
            if not snp or pos < 1:
                raise ValueError(f"invalid SNP ID or position in BIM row {index}")
            if chrom != last_chrom:
                if chrom in seen_chrom:
                    raise ValueError(f"chromosome {chrom} occurs in multiple BIM segments")
                seen_chrom.add(chrom)
                last_pos = -1
            if pos < last_pos:
                raise ValueError(f"BIM is not position-sorted on chromosome {chrom}")
            last_chrom, last_pos = chrom, pos
            if selected is None or chrom == selected:
                rows.append({"Index": index, "Chrom": chrom,
                             "SNP": snp, "Position": pos})
    if not rows:
        raise ValueError(f"no BIM variants for chromosome {chromosome!r}")
    if len({row["SNP"] for row in rows}) != len(rows):
        raise ValueError("selected BIM SNP IDs are not unique")
    return rows


def _check_real(value: float, name: str, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _write_expected(rows: list[dict], pos_scaled: float | Sequence[float] | None,
                    destination: Path) -> None:
    if pos_scaled is None:
        scaled = [0.0] * len(rows)
    elif isinstance(pos_scaled, Real):
        divisor = _check_real(pos_scaled, "pos_scaled")
        scaled = ([0.0] * len(rows) if divisor == 0 else
                  [row["Position"] / divisor for row in rows])
    else:
        scaled = [_check_real(value, "pos_scaled") for value in pos_scaled]
        if len(scaled) != len(rows):
            raise ValueError("pos_scaled vector must match selected BIM variant count")
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["Index", "Chrom", "SNP", "Position", "PosScaled"])
        for row, value in zip(rows, scaled):
            writer.writerow([row["Index"], row["Chrom"], row["SNP"],
                             row["Position"], format(value, ".17g")])


def _verify_order(rows: list[dict], directory: Path) -> None:
    path = directory / "variant_order.tsv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["Chrom", "SNP", "Position"]:
            raise ValueError(f"unexpected R variant-order columns: {path}")
        for index, expected in enumerate(rows):
            observed = next(reader, None)
            if observed is None or (observed["Chrom"], observed["SNP"],
                                    int(observed["Position"])) != (
                                        expected["Chrom"], expected["SNP"],
                                        expected["Position"]):
                raise ValueError(f"R/PLINK BIM SNP order or position mismatch at {index + 1}")
        if next(reader, None) is not None:
            raise ValueError("R returned more variants than the selected BIM")


def _blocks(rows: list[dict], directory: Path) -> pd.DataFrame:
    result = []
    cursor = 0
    number = 0
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["Chrom"], []).append(row)
    for chrom, chr_rows in groups.items():
        size_file = directory / f"chr_{chrom}" / "selected_sizes.tsv"
        with size_file.open() as stream:
            sizes = [int(line.strip()) for line in stream if line.strip()]
        if not sizes or any(size < 1 for size in sizes) or sum(sizes) != len(chr_rows):
            raise ValueError(f"LD split does not cover all BIM variants on chromosome {chrom}")
        offset = 0
        for size in sizes:
            segment = chr_rows[offset:offset + size]
            if not segment or len(segment) != size:
                raise ValueError(f"invalid LD block size on chromosome {chrom}")
            number += 1
            result.append((number, chrom, segment[0]["Position"],
                           segment[-1]["Position"], size))
            offset += size
        cursor += sum(sizes)
    if cursor != len(rows):
        raise ValueError("LD blocks do not cover selected BIM rows")
    frame = pd.DataFrame(result, columns=BLOCK_COLUMNS)
    frame.to_csv(directory / "blocks.tsv", sep="\t", index=False)
    return frame


def run_ldsplit(
    plink_prefix: str | Path,
    chromosome: str | int | None = None,
    *,
    size_kb: float = 100,
    size: float | None = None,
    thr_r2: float = 0.05,
    min_size: int = 10,
    max_size: int = 2000,
    max_r2: float = 0.3,
    max_K: int = 500,
    max_cost: float | None = None,
    pos_scaled: float | Sequence[float] | None = None,
    ncores: int = 1,
    output_dir: str | Path | None = None,
    corr_source: str | Path | None = None,
    rscript: str | Path = DEFAULT_RSCRIPT,
    save_corr: bool = True,
    timeout_seconds: float | None = None,
) -> pd.DataFrame:
    """Return LD blocks and write candidate solutions, correlations, and logs.

    ``size_kb``/``size`` is the physical LD window in kb. ``pos_scaled=None``
    uses bigsnpr's zero vector; a scalar divides physical positions in bp
    (e.g. 10_000_000 limits block span to about 10 Mb), while a sequence
    supplies the exact R ``pos_scaled`` vector in selected BIM order.

    bigsnpr can return many feasible partitions. This wrapper selects the
    documented example's minimum ``cost2 * sqrt(5 + cost)`` solution and
    saves all alternatives in ``chr_*/solutions.tsv`` and ``solutions.rds``.
    """
    prefix = _prefix(plink_prefix)
    rows = _bim(prefix, chromosome)
    window = _check_real(size_kb if size is None else size, "size", positive=True)
    threshold = _check_real(thr_r2, "thr_r2")
    maximum_r2 = _check_real(max_r2, "max_r2")
    if threshold > 1 or maximum_r2 > 1:
        raise ValueError("thr_r2 and max_r2 must be at most 1")
    for value, name in ((min_size, "min_size"), (max_size, "max_size"),
                        (max_K, "max_K"), (ncores, "ncores")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_size > max_size:
        raise ValueError("min_size cannot exceed max_size")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_cost is not None and (math.isnan(float(max_cost)) or float(max_cost) < 0):
        raise ValueError("max_cost must be nonnegative or None")
    source = None
    if corr_source is not None:
        source = Path(corr_source).expanduser().resolve()
        if not source.is_file() or source.name != "corr.rds" or chromosome is None:
            raise ValueError("corr_source must be a chr_*/corr.rds file for one chromosome")
        prior_dir = source.parent.parent
        prior = json.loads((prior_dir / "parameters.json").read_text())
        if (prior["plink_prefix"] != str(prefix) or
                str(prior["chromosome"]) != str(chromosome) or
                prior["size_kb"] != window or prior["thr_r2"] != threshold or
                prior["n_variants"] != len(rows)):
            raise ValueError("corr_source was generated from different PLINK data or LD settings")
        _verify_order(rows, prior_dir)
    label = "all" if chromosome is None else str(chromosome)
    directory = (Path(output_dir).expanduser().resolve() if output_dir else
                 prefix.with_name(f"{prefix.name}.ldsplit_chr{label}"))
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"LD split output is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    _write_expected(rows, pos_scaled, directory / "expected.tsv")
    (directory / "parameters.json").write_text(json.dumps({
        "plink_prefix": str(prefix), "chromosome": chromosome,
        "size_kb": window, "thr_r2": threshold, "min_size": min_size,
        "max_size": max_size, "max_r2": maximum_r2, "max_K": max_K,
        "max_cost": max_cost,
        "pos_scaled": ("zero" if pos_scaled is None else
                       "vector in expected.tsv" if not isinstance(pos_scaled, Real)
                       else float(pos_scaled)),
        "ncores": ncores, "save_corr": save_corr, "rscript": str(rscript),
        "corr_source": str(source) if source else None,
        "n_variants": len(rows),
    }, indent=2) + "\n")
    command = [str(rscript), str(BACKEND), str(prefix), str(directory),
               str(window), str(threshold), str(min_size), str(max_size),
               str(maximum_r2), str(max_K), str(ncores),
               "TRUE" if save_corr else "FALSE",
               "DEFAULT" if max_cost is None else str(max_cost),
               str(source) if source else "NONE", "1"]
    try:
        completed = subprocess.run(command, text=True, capture_output=True,
                                   check=False, timeout=timeout_seconds)
    except OSError as exc:
        raise RuntimeError(f"could not start Rscript: {rscript}") from exc
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = exc.stdout or "", exc.stderr or ""
        (directory / "stdout.log").write_text(
            stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout)
        (directory / "stderr.log").write_text(
            stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr)
        raise RuntimeError(f"bigsnpr R backend exceeded {timeout_seconds} seconds; "
                           f"inspect {directory / 'stderr.log'}") from exc
    (directory / "stdout.log").write_text(completed.stdout)
    (directory / "stderr.log").write_text(completed.stderr)
    if completed.returncode:
        detail = "\n".join(completed.stderr.splitlines()[-10:])
        raise RuntimeError(f"bigsnpr R backend exited {completed.returncode}; "
                           f"inspect {directory / 'stderr.log'}\n{detail}")
    _verify_order(rows, directory)
    return _blocks(rows, directory)



def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_tsv(path: Path, columns: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    lower = int(point)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (point - lower)


def _chromosome_counts(prefix: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    previous_chrom, previous_pos = 0, 0
    with Path(f"{prefix}.bim").open() as stream:
        for row_number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"PLINK BIM row {row_number} needs six columns")
            chrom, pos = int(fields[0]), int(fields[3])
            if chrom < 1 or pos < 1 or chrom < previous_chrom or (
                    chrom == previous_chrom and pos < previous_pos):
                raise ValueError(f"PLINK BIM is not sorted at row {row_number}")
            counts[chrom] = counts.get(chrom, 0) + 1
            previous_chrom, previous_pos = chrom, pos
    if not counts:
        raise ValueError("PLINK BIM contains no variants")
    return counts


def _case_directory(source: Path, chrom: int, size_kb: float,
                    max_r2: float) -> Path:
    name = f"chr{chrom:02d}_window{size_kb:g}kb_maxr2{max_r2:g}"
    return source / "cases" / name / "run"


def _run_chromosome(prefix: Path, source: Path, chrom: int, n_snp: int,
                    settings: dict) -> dict:
    """Run one chromosome or verify and reuse its completed calculation."""
    directory = _case_directory(source, chrom, settings["size_kb"],
                                settings["max_r2"])
    rows = _bim(prefix, chrom)
    if len(rows) != n_snp:
        raise ValueError(f"chromosome {chrom}: BIM count changed")
    params_file = directory / "parameters.json"
    frame = None
    if params_file.is_file():
        params = json.loads(params_file.read_text())
        expected = {
            "plink_prefix": str(prefix), "chromosome": chrom,
            "size_kb": settings["size_kb"], "thr_r2": settings["thr_r2"],
            "min_size": settings["min_size"], "max_size": settings["max_size"],
            "max_r2": settings["max_r2"], "max_K": settings["max_K"],
            "max_cost": settings["max_cost"],
            "pos_scaled": ("zero" if settings["pos_scaled"] is None
                           else float(settings["pos_scaled"])),
            "n_variants": n_snp, "corr_source": None,
        }
        if any(params.get(key) != value for key, value in expected.items()):
            raise ValueError(f"existing chromosome {chrom} run uses different parameters: {directory}")
        needed = ("expected.tsv", "blocks.tsv", "variant_order.tsv",
                  f"chr_{chrom}/solutions.tsv", f"chr_{chrom}/selected_solution.tsv")
        if all((directory / name).is_file() for name in needed):
            _verify_order(rows, directory)
            frame = pd.read_csv(directory / "blocks.tsv", sep="\t")
    if frame is None:
        if directory.exists() and any(directory.iterdir()):
            archive = directory.with_name(
                f"interrupted_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}")
            directory.rename(archive)
            print(f"{_timestamp()} Chr{chrom}: saved partial run in {archive}", flush=True)
        frame = run_ldsplit(
            plink_prefix=prefix, chromosome=chrom,
            size_kb=settings["size_kb"], thr_r2=settings["thr_r2"],
            min_size=settings["min_size"], max_size=settings["max_size"],
            max_r2=settings["max_r2"], max_K=settings["max_K"],
            max_cost=settings["max_cost"], pos_scaled=settings["pos_scaled"],
            ncores=settings["ncores"], output_dir=directory, save_corr=True,
            timeout_seconds=settings["timeout_seconds"],
            rscript=settings["rscript"],
        )
    if list(frame.columns) != BLOCK_COLUMNS or int(frame.NSNP.sum()) != n_snp:
        raise ValueError(f"chromosome {chrom}: incomplete selected blocks")
    return {"chrom": chrom, "n_snp": n_snp, "status": "success",
            "n_blocks": len(frame), "case_dir": str(directory.parent), "error": ""}


def _select_chromosome(prefix: Path, source: Path, chrom: int, n_snp: int,
                       target_k: int, first_block: int, settings: dict) -> tuple[list[dict], dict]:
    """Read one saved snp_ldsplit solution with the requested block count."""
    run = _case_directory(source, chrom, settings["size_kb"], settings["max_r2"])
    with (run / f"chr_{chrom}" / "solutions.tsv").open(newline="") as stream:
        candidates = [row for row in csv.DictReader(stream, delimiter="\t")
                      if int(row["n_block"]) == target_k]
    if len(candidates) != 1:
        raise ValueError(f"chromosome {chrom}: expected one K={target_k} solution; found {len(candidates)}")
    chosen = candidates[0]
    sizes = [int(value) for value in chosen["all_size"].split(",")]
    if (len(sizes) != target_k or not sizes
            or min(sizes) < settings["min_size"]
            or max(sizes) > settings["max_size"] or sum(sizes) != n_snp):
        raise ValueError(f"chromosome {chrom}: invalid selected block sizes")
    bim_rows = _bim(prefix, chrom)
    positions: list[int] = []
    with (run / "expected.tsv").open(newline="") as stream:
        expected_rows = csv.DictReader(stream, delimiter="\t")
        for bim, observed in zip(bim_rows, expected_rows):
            if (int(observed["Index"]) != bim["Index"]
                    or observed["Chrom"] != bim["Chrom"]
                    or observed["SNP"] != bim["SNP"]
                    or int(observed["Position"]) != bim["Position"]):
                raise ValueError(f"chromosome {chrom}: expected.tsv differs from BIM")
            positions.append(bim["Position"])
        if next(expected_rows, None) is not None:
            raise ValueError(f"chromosome {chrom}: extra expected.tsv variants")
    if len(positions) != n_snp:
        raise ValueError(f"chromosome {chrom}: missing expected.tsv variants")
    blocks: list[dict] = []
    widths: list[int] = []
    cursor = 0
    for offset, size in enumerate(sizes):
        start, end = positions[cursor], positions[cursor + size - 1]
        if blocks and start <= blocks[-1]["EndBP"]:
            raise ValueError(f"chromosome {chrom}: block coordinates overlap")
        blocks.append({"Block": first_block + offset, "Chrom": chrom,
                       "StartBP": start, "EndBP": end, "NSNP": size})
        widths.append(end - start + 1)
        cursor += size
    if cursor != n_snp:
        raise ValueError(f"chromosome {chrom}: blocks do not cover the BIM")
    small = settings["small_block_threshold"]
    maximum = settings["max_size"]
    summary = {
        "Chrom": chrom, "NSNP": n_snp, "KMinimum": math.ceil(n_snp / maximum),
        "KTarget": target_k, "NBlock": len(blocks),
        "MinSNP": min(sizes), "MedianSNP": _percentile(sizes, 0.5),
        "P90SNP": _percentile(sizes, 0.9), "P95SNP": _percentile(sizes, 0.95),
        "MaxSNP": max(sizes),
        f"NLe{small}SNP": sum(size <= small for size in sizes),
        f"NEq{maximum}SNP": sum(size == maximum for size in sizes),
        "MedianBP": _percentile(widths, 0.5),
        "P95BP": _percentile(widths, 0.95), "MaxBP": max(widths),
        "PercKept": chosen["perc_kept"], "Cost": chosen["cost"],
        "Cost2": chosen["cost2"],
    }
    return blocks, summary


def run_full_genome_ldsplit(
    plink_prefix: str | Path,
    source_root: str | Path,
    output_root: str | Path,
    *,
    size_kb: float,
    thr_r2: float,
    min_size: int,
    max_size: int,
    max_r2: float,
    max_K: int,
    max_cost: float | None,
    pos_scaled: float | None,
    k_factor: float,
    small_block_threshold: int,
    parallel_chromosomes: int,
    selection_workers: int,
    ncores: int,
    timeout_seconds: float | None,
    rscript: str | Path = DEFAULT_RSCRIPT,
) -> Path:
    """Run each BIM chromosome and select a requested-K solution per chromosome.

    Completed chromosome runs are verified and reused. Source correlations and
    candidate solutions remain in source_root; output_root receives a combined
    GCTB blocks.pos, detailed block tables, summary.tsv and completion metadata.
    """
    prefix = _prefix(plink_prefix)
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if (parallel_chromosomes < 1 or selection_workers < 1 or ncores < 1
            or min_size < 1 or max_size < min_size or max_K < 1
            or small_block_threshold < 1 or not math.isfinite(k_factor)
            or k_factor < 1 or output == source or source in output.parents):
        raise ValueError("invalid LD split settings or overlapping output directories")
    settings = {
        "size_kb": float(size_kb), "thr_r2": float(thr_r2),
        "min_size": min_size, "max_size": max_size, "max_r2": float(max_r2),
        "max_K": max_K, "max_cost": max_cost, "pos_scaled": pos_scaled,
        "k_factor": k_factor, "small_block_threshold": small_block_threshold,
        "parallel_chromosomes": parallel_chromosomes,
        "selection_workers": selection_workers, "ncores": ncores,
        "timeout_seconds": timeout_seconds, "rscript": str(rscript),
    }
    counts = _chromosome_counts(prefix)
    chroms = sorted(counts)
    if (output / "COMPLETE.json").is_file():
        selection = output / "selection.json"
        if not (output / "blocks.pos").is_file() or not selection.is_file():
            raise ValueError(f"completed output is missing files: {output}")
        completion = json.loads((output / "COMPLETE.json").read_text())
        if (completion.get("n_chromosomes") != len(chroms)
                or completion.get("n_snps") != sum(counts.values())):
            raise ValueError(f"completed output does not match the current BIM: {output}")
        recorded = json.loads(selection.read_text())
        if (recorded.get("source") != str(source)
                or recorded.get("k_factor") != k_factor
                or recorded.get("max_size") != max_size
                or ("plink_prefix" in recorded and recorded["plink_prefix"] != str(prefix))
                or ("settings" in recorded and recorded["settings"] != settings)):
            raise ValueError(f"completed output uses different inputs or parameters: {output}")
        first_params = _case_directory(source, chroms[0], size_kb, max_r2) / "parameters.json"
        params = json.loads(first_params.read_text())
        for key in ("plink_prefix", "size_kb", "thr_r2", "min_size", "max_size",
                    "max_r2", "max_K", "max_cost", "pos_scaled"):
            expected = (str(prefix) if key == "plink_prefix" else
                        "zero" if key == "pos_scaled" and pos_scaled is None else
                        settings[key])
            if params.get(key) != expected:
                raise ValueError(f"completed output uses different LD settings: {output}")
        return output / "blocks.pos"
    if output.exists() and {path.name for path in output.iterdir()} - {"driver.log", "launcher.pid"}:
        raise FileExistsError(f"selected output is not empty: {output}")
    source.mkdir(parents=True, exist_ok=True)
    columns = ["chrom", "n_snp", "status", "n_blocks", "case_dir", "error"]
    progress = {chrom: {"chrom": chrom, "n_snp": counts[chrom], "status": "pending",
                        "n_blocks": "", "case_dir": str(_case_directory(
                            source, chrom, size_kb, max_r2).parent), "error": ""}
                for chrom in chroms}
    def save_progress() -> None:
        _write_tsv(source / "standalone_results.tsv", columns,
                   [progress[chrom] for chrom in chroms])
    save_progress()
    print(f"{_timestamp()} START {len(chroms)} chromosomes, "
          f"{parallel_chromosomes} concurrent", flush=True)
    errors: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=parallel_chromosomes) as pool:
        futures = {pool.submit(_run_chromosome, prefix, source, chrom,
                               counts[chrom], settings): chrom for chrom in chroms}
        for future in as_completed(futures):
            chrom = futures[future]
            try:
                progress[chrom] = future.result()
                print(f"{_timestamp()} Chr{chrom} complete", flush=True)
            except Exception as exc:
                errors[chrom] = f"{type(exc).__name__}: {exc}"
                progress[chrom].update(status="failed", error=errors[chrom])
                print(f"{_timestamp()} Chr{chrom} failed: {errors[chrom]}", flush=True)
            save_progress()
    if errors:
        raise RuntimeError(f"LD split failed on chromosomes {sorted(errors)}; "
                           f"see {source / 'standalone_results.tsv'}")
    targets = {chrom: math.ceil(k_factor * math.ceil(counts[chrom] / max_size))
               for chrom in chroms}
    offsets: dict[int, int] = {}
    first_block = 1
    for chrom in chroms:
        offsets[chrom] = first_block
        first_block += targets[chrom]
    selected: dict[int, tuple[list[dict], dict]] = {}
    with ThreadPoolExecutor(max_workers=selection_workers) as pool:
        futures = {pool.submit(_select_chromosome, prefix, source, chrom,
                               counts[chrom], targets[chrom], offsets[chrom],
                               settings): chrom for chrom in chroms}
        for future in as_completed(futures):
            selected[futures[future]] = future.result()
    blocks = [block for chrom in chroms for block in selected[chrom][0]]
    summaries = [selected[chrom][1] for chrom in chroms]
    if ([row["Block"] for row in blocks] != list(range(1, len(blocks) + 1))
            or sum(row["NSNP"] for row in blocks) != sum(counts.values())):
        raise ValueError("combined block map does not cover all BIM variants")
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps({
        "source": str(source), "plink_prefix": str(prefix),
        "created_utc": _timestamp(), "k_factor": k_factor,
        "max_size": max_size, "targets": targets,
        "workers": selection_workers, "settings": settings,
    }, indent=2) + "\n")
    for chrom in chroms:
        directory = output / f"chr_{chrom}"
        directory.mkdir()
        _write_tsv(directory / "blocks.tsv", BLOCK_COLUMNS, selected[chrom][0])
    summary_columns = [
        "Chrom", "NSNP", "KMinimum", "KTarget", "NBlock", "MinSNP",
        "MedianSNP", "P90SNP", "P95SNP", "MaxSNP",
        f"NLe{small_block_threshold}SNP", f"NEq{max_size}SNP",
        "MedianBP", "P95BP", "MaxBP", "PercKept", "Cost", "Cost2",
    ]
    _write_tsv(output / "blocks.tsv", BLOCK_COLUMNS, blocks)
    _write_tsv(output / "blocks.pos", BLOCK_COLUMNS[:4], blocks)
    _write_tsv(output / "summary.tsv", summary_columns, summaries)
    (output / "COMPLETE.json").write_text(json.dumps({
        "completed_utc": _timestamp(), "n_chromosomes": len(chroms),
        "n_blocks": len(blocks), "n_snps": sum(counts.values()),
        f"n_blocks_le_{small_block_threshold}_snp":
            sum(row["NSNP"] <= small_block_threshold for row in blocks),
    }, indent=2) + "\n")
    print(f"{_timestamp()} COMPLETE {len(blocks)} blocks: {output / 'blocks.pos'}",
          flush=True)
    return output / "blocks.pos"


if __name__ == "__main__":
    # Reproduce the sorghum landrace example on rock01. These paths and LD
    # settings belong to this invocation; run_full_genome_ldsplit is reusable.
    project = Path("/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum")
    print(run_full_genome_ldsplit(
        plink_prefix=project / "data/results/landraces_qc_miss0.5_maf0.05_het0.05",
        source_root=project / "data/results/bigsnpr_full_genome_100kb_v1",
        output_root=project / "data/results/bigsnpr_full_genome_100kb_k1p1_v1",
        size_kb=100, thr_r2=0.3, min_size=10, max_size=2000,
        max_r2=1.0, max_K=600, max_cost=float("inf"), pos_scaled=None,
        k_factor=1.1, small_block_threshold=100,
        parallel_chromosomes=2, selection_workers=8, ncores=8,
        timeout_seconds=4 * 3600,
    ))
