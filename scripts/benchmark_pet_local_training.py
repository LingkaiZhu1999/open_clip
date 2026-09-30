#!/usr/bin/env python3
"""Bounded real-PET timing runs on two idle GPUs; no checkpoints or external tracking."""
import argparse
import csv
import json
import re
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--chunks', nargs='+', type=int, default=[4, 16, 32])
    parser.add_argument('--global-only', action='store_true', help='Measure the original global-only model at the same batch')
    args = parser.parse_args()
    if args.steps < 2 or args.batch_size < 1 or min(args.chunks) < 1:
        parser.error('Need positive sizes and at least two steps (first is warmup)')
    root = Path(__file__).resolve().parents[1]
    output = root / '.local/pet/local-loss-profile' / time.strftime('training-%Y%m%d-%H%M%S')
    output.mkdir(parents=True)
    results = dict(batch_size=args.batch_size, steps=args.steps, global_only=args.global_only, chunks={})
    for chunk in ([0] if args.global_only else args.chunks):
        logfile = output / f'chunk{chunk}.log'
        memoryfile = output / f'chunk{chunk}_gpu.csv'
        launcher = 'train_pet_2gpu.sh' if args.global_only else 'train_pet_local_2gpu.sh'
        region_args = [] if args.global_only else ['--pet-region-chunk-size', str(chunk)]
        command = ['bash', str(root / 'scripts' / launcher), str(args.data_root),
                   '--batch-size', str(args.batch_size), *region_args,
                   '--pet-limit-per-split', str(args.steps * args.batch_size * 2),
                   '--workers', '0', '--pet-cache', 'none', '--epochs', '1', '--warmup', '0',
                   '--val-frequency', '0', '--logs', 'none', '--report-to', '',
                   '--log-every-n-steps', '1', '--log-metric-every-n-steps', '1',
                   '--name', f'local-timing-{output.name}-chunk{chunk}']
        with memoryfile.open('w') as memory, logfile.open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits', '-lms', '250'], stdout=memory)
            try:
                completed = subprocess.run(command, cwd=root / '.local/pet', stdout=log, stderr=subprocess.STDOUT)
            finally:
                monitor.terminate()
                monitor.wait()
        if completed.returncode:
            raise RuntimeError(f'Training benchmark failed: {logfile}')
        times = [float(t) for t in re.findall(r'Train Epoch: .*?Batch \(t\): ([0-9.]+)', logfile.read_text())]
        data = [float(t) for t in re.findall(r'Train Epoch: .*?Data \(t\): ([0-9.]+)', logfile.read_text())]
        if len(times) != args.steps:
            raise RuntimeError(f'Expected {args.steps} step timings, got {times}')
        peaks = {}
        for row in csv.reader(memoryfile.read_text().splitlines()):
            if len(row) == 3 and row[1].strip().isdigit():
                gpu, used = int(row[0]), int(row[1])
                peaks[gpu] = max(peaks.get(gpu, 0), used)
        result = dict(batch_seconds=times, data_seconds=data,
                      median_after_first_seconds=statistics.median(times[1:]),
                      sampled_peak_device_mib=peaks,
                      log=str(logfile.relative_to(root)),
                      command=command)
        results['chunks'][str(chunk)] = result
        (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f'chunk {chunk}: {result["median_after_first_seconds"]:.3f} s median after first step; peak {peaks}', flush=True)
    print(f'Saved {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
