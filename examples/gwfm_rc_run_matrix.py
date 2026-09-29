"""Fixed, resumable GWFM RC benchmark matrix for rock01.

Without --execute, write the complete matrix only. Execution requires the
shared LD feasibility gate and LD build to have produced ld/COMPLETE.
"""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path('/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum')
BENCH = PROJECT / 'data/results/gwfm_rc_benchmark'
PYTHON = Path('/lustre/home/dobby/Program/agent_program/bin/python3')


def designs():
    settings = [('gaussian', n) for n in (50, 500, 5000)]
    settings += [('large', 500), ('ldms', 500)]
    for architecture, count in settings:
        for variance in (0.05, 0.15, 0.30):
            for replicate in range(1, 31 if variance == 0.15 else 11):
                seed = 10_000 + replicate
                name = f'{architecture}_{count}_h{variance:.2f}_s{seed}'
                yield name, architecture, count, variance, seed
    for replicate in range(1, 11):
        seed = 10_000 + replicate
        yield f'null_0_h0.00_s{seed}', 'null', 0, 0.0, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--gemma-workers', type=int, default=40)
    parser.add_argument('--gctb-threads', type=int, default=32)
    args = parser.parse_args()
    BENCH.mkdir(parents=True, exist_ok=True)
    runs = BENCH / 'runs_all_variants'
    runs.mkdir(exist_ok=True)
    entries = list(designs())
    with (BENCH / 'design_matrix.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(('run', 'architecture', 'n_causal', 'causal_var', 'seed'))
        writer.writerows(entries)
    print(f'Wrote {len(entries)} fixed designs to {BENCH / "design_matrix.tsv"}')
    if not args.execute:
        return
    ld = BENCH / 'ld_all_variants_15kb'
    if not (ld / 'COMPLETE').is_file():
        raise SystemExit('STOP: full mixed-variant LD is not complete')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT) + ':' + env.get('PYTHONPATH', '')
    for name, architecture, count, variance, seed in entries:
        out = runs / name
        if (out / 'COMPLETE').is_file():
            print(f'skip complete: {name}', flush=True)
            continue
        out.mkdir(exist_ok=True)
        command = [str(PYTHON), '-m', 'src.gwfm_benchmark', 'run-one',
                   '--ldm', str(ld), '--output', str(out),
                   '--architecture', architecture, '--count', str(count),
                   '--causal-var', str(variance), '--seed', str(seed),
                   '--gemma-workers', str(args.gemma_workers),
                   '--gctb-threads', str(args.gctb_threads)]
        if architecture == 'ldms':
            command += ['--ld-scores', str(ld / 'ld_scores.tsv')]
        (out / 'command.txt').write_text(' '.join(command) + '\n')
        with (out / 'driver.log').open('a') as stream:
            result = subprocess.run(command, cwd=PROJECT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode or not (out / 'COMPLETE').is_file():
            (out / 'FAILED').write_text(f'command exit code: {result.returncode}\n')
            raise SystemExit(f'run failed: {name}; inspect {out / "driver.log"}')
        (out / 'FAILED').unlink(missing_ok=True)
        print(f'complete: {name}', flush=True)
    subprocess.run([str(PYTHON), '-m', 'src.gwfm_benchmark', 'aggregate',
                    '--output-root', str(runs)], cwd=PROJECT, env=env,
                   check=True)


if __name__ == '__main__':
    main()
