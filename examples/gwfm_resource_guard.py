"""Run one GCTB command and record/limit its resident memory on rock01."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def rss_kib(pid):
    try:
        with Path(f'/proc/{pid}/status').open() as source:
            for line in source:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])
    except FileNotFoundError:
        return 0
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--max-rss-gib', type=float, default=400)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('missing command')
    start = time.monotonic()
    peak = 0
    exceeded = False
    with args.log.open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                peak = max(peak, rss_kib(proc.pid))
                if peak > args.max_rss_gib * 1024**2:
                    exceeded = True
                    proc.terminate()
                    break
                time.sleep(1)
            code = proc.wait(timeout=30)
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    status = dict(command=command, exit_code=code,
                  elapsed_seconds=time.monotonic() - start,
                  max_rss_gib=peak / 1024**2,
                  memory_limit_gib=args.max_rss_gib,
                  memory_limit_exceeded=exceeded)
    args.resources.write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)
    if code or exceeded:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
