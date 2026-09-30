#!/usr/bin/env python3
"""Isolated synthetic loss benchmark; no dataset, encoders, or training job changes.

Run only on idle GPUs. Example:
PYTHONPATH=src NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=2 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
scripts/benchmark_pet_local_loss.py --device cuda --batch-size 128 --chunks 4 16 32

CPU mode is useful for functional/profiler checks, not GPU speed predictions.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.profiler import profile, ProfilerActivity

from open_clip.local_region import RegionClipLoss
from open_clip.loss import ClipLoss, gather_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--chunks', nargs='+', type=int, default=[4, 16, 32])
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--regions', type=int, default=98)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--profile', action='store_true', help='Profile one additional step per chunk (can be expensive)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.tokens, args.regions, args.dim, args.steps, *args.chunks) < 1 or args.warmup < 0:
        parser.error('Sizes/steps must be positive and warmup nonnegative')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0'))) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    try:
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        torch.manual_seed(19 + rank)
        shapes = [(args.batch_size, 512), (args.batch_size, 512),
                  (args.batch_size, args.regions, args.dim), (args.batch_size, args.tokens, args.dim)]
        values = [F.normalize(torch.randn(shape, device=device), dim=-1).to(dtype).requires_grad_()
                  for shape in shapes]
        valid = torch.ones(args.batch_size, args.tokens, device=device, dtype=torch.bool)
        scale = torch.tensor(14.286, device=device, requires_grad=True)
        options = dict(rank=rank, world_size=world, local_loss=True, gather_with_grad=True)
        global_loss = ClipLoss(**options)

        def sync():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        def step(function):
            for x in [*values, scale]:
                x.grad = None
            if world > 1:
                dist.barrier()
            sync()
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                result = function()
            sync()
            forward = time.perf_counter() - start
            start = time.perf_counter()
            result.backward()
            sync()
            backward = time.perf_counter() - start
            elapsed = torch.tensor([forward, backward], device=device)
            if world > 1:
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if not torch.isfinite(result):
                raise RuntimeError('Nonfinite benchmark loss')
            return elapsed.tolist()

        def measure(function):
            for _ in range(args.warmup):
                step(function)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            timings = [step(function) for _ in range(args.steps)]
            result = dict(forward_ms=1000 * statistics.median(x[0] for x in timings),
                          backward_ms=1000 * statistics.median(x[1] for x in timings),
                          total_ms=1000 * statistics.median(sum(x) for x in timings))
            result['samples_ms'] = [[1000 * value for value in sample] for sample in timings]
            if device.type == 'cuda':
                peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**20, device=device)
                if world > 1:
                    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                result['peak_allocated_mib'] = peak.item()
            return result

        def gather_only():
            if world == 1:
                return values[2].float().sum() + values[3].float().sum()
            r, t = gather_features(values[2], values[3], **{k:v for k,v in options.items() if k != 'cache_labels'})
            masks = [torch.empty_like(valid) for _ in range(world)]
            dist.all_gather(masks, valid)
            return r.float().square().sum() + t.float().square().sum()

        output = dict(device=str(device), world_size=world, per_rank_batch=args.batch_size,
                      dtype=str(dtype), regions=args.regions, tokens=args.tokens, dim=args.dim,
                      note='Synthetic loss only; excludes encoder/optimizer/DDP parameter-gradient costs. Timings synchronize stage boundaries.',
                      global_loss=measure(lambda: global_loss(values[0], values[1], scale)),
                      local_feature_gather=measure(gather_only), chunks={})
        for chunk in args.chunks:
            config = dict(chunk_size=chunk, weight=1., attention_temperature=.1,
                          word_temperature=.1, contrastive_temperature=.07)
            criterion = RegionClipLoss(config, **options)
            function = lambda: criterion(values[0], values[1], scale, values[2], values[3], valid)
            result = measure(function)
            if args.profile:
                activities = [ProfilerActivity.CPU]
                if device.type == 'cuda':
                    activities.append(ProfilerActivity.CUDA)
                with profile(activities=activities) as prof:
                    step(function)
                events = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
                result['top_cpu_events'] = [dict(name=e.key, calls=e.count,
                    self_cpu_ms=e.self_cpu_time_total/1000,
                    self_device_ms=getattr(e, 'self_device_time_total', 0)/1000) for e in events[:15]]
                result['operator_calls'] = sum(e.count for e in events if e.key.startswith('aten::'))
            output['chunks'][str(chunk)] = result
            if rank == 0:
                print(f'chunk={chunk}: {result["total_ms"]:.1f} ms forward+backward', flush=True)
        if rank == 0:
            encoded = json.dumps(output, indent=2)
            print(encoded)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(encoded + '\n')
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
