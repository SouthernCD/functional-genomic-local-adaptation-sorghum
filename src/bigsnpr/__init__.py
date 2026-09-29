"""Small Python entry point for bigsnpr LD calculations."""

from .ldsplit import run_full_genome_ldsplit, run_ldsplit

__all__ = ["run_ldsplit", "run_full_genome_ldsplit"]
