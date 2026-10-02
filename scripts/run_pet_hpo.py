#!/usr/bin/env python3
"""Sequential single-GPU PET ablations, retaining one best checkpoint per experiment."""
import argparse
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    'A': {'model': 'PET-ResNet34-BioClinicalBERT', 'pretrained_image': False, 'local_weight': None},
    'B': {'model': 'PET-ResNet34-BioClinicalBERT-Local', 'pretrained_image': False, 'local_weight': 0.1},
    'C': {'model': 'PET-ResNet34-BioClinicalBERT', 'pretrained_image': True, 'local_weight': None},
}


def write_status(root, status):
    tmp = root / 'status.json.tmp'
    tmp.write_text(json.dumps(status, indent=2, allow_nan=False) + '\n')
    tmp.replace(root / 'status.json')
    lines = ['# PET learning-rate and warmup search', '',
             f"Status: **{status['state']}**. Updated {datetime.now().isoformat(timespec='seconds')}", '',
             'Selection: minimum full-validation `clip_val_loss` (global loss in all experiments).',
             'Batch: 256 × 2 accumulation steps on one GPU = 512 contrastive pairs; pretrained Bio_ClinicalBERT; context 1024; seed 0.',
             'Each trial: max 20 epochs, patience 3, same split/preprocessing/schedule horizon.',
             'Grid: LR {1e-5, 5e-5} × warmup {20, 60} optimizer steps. Test split is unused.', '',
             '| Experiment | LR | Warmup | Best epoch | Global validation loss | Status |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for t in status['trials']:
        score = t.get('best_value')
        lines.append(f"| {t['experiment']} | {t['lr']} | {t['warmup']} | {t.get('best_epoch', '—')} | "
                     f"{f'{score:.6f}' if score is not None else '—'} | {t['state']} |")
    lines += ['', '## Selected checkpoints', '']
    for key, t in status['winners'].items():
        lines.append(f"- {key}: LR {t['lr']}, warmup {t['warmup']}, epoch {t['best_epoch']}, "
                     f"global loss {t['best_value']:.6f}; `{key}/best.pt` (trial `{t['name']}`).")
    if status.get('error'):
        lines += ['', 'Failure: ' + status['error']]
    lines += ['', 'These are validation-selected results from a small single-seed grid, not an exhaustive optimum.',
              'Each experiment retains one checkpoint; atomic replacement temporarily needs one extra checkpoint worth of space.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/mnt/snotra1/ida/datasets/lymphoma_mskcc_deid')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'status.json').exists():
            raise SystemExit('Use a new output directory; refusing to overwrite an existing search.')
        status = dict(state='running', trials=[], winners={}, pid=os.getpid(), gpu=args.gpu, accum_freq=2)
        env = os.environ.copy()
        env.update(PET_LOG_DIR=str(root / 'trials'), WANDB_DIR=str(root),
                   HF_HOME=str(REPO / '.local/pet/hf-cache'), TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   CUDA_VISIBLE_DEVICES=str(args.gpu), NPROC_PER_NODE='1',
                   PYTHON_BIN=str(REPO / '.local/pet-cuda-venv/bin/python'), OMP_NUM_THREADS='4')
        (root / 'code.patch').write_text(subprocess.check_output(['git', 'diff'], cwd=REPO, text=True))
        (root / 'git_revision.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True))
        # Preserve untracked implementation/config files too; git diff alone misses them.
        snapshot = root / 'source_snapshot'
        sources = [Path(__file__).resolve(), REPO / 'scripts/train_pet_2gpu.sh',
                   REPO / 'scripts/train_pet_scratch.sh', REPO / 'src/open_clip_train/main.py',
                   REPO / 'src/open_clip_train/params.py', REPO / 'src/open_clip_train/best_checkpoint.py',
                   REPO / 'src/open_clip_train/train.py', REPO / 'src/open_clip_train/pet_accum.py',
                   REPO / 'src/open_clip/factory.py', REPO / 'src/open_clip/local_region.py',
                   REPO / 'src/open_clip/hf_model.py', REPO / 'src/open_clip/model.py']
        sources += list((REPO / 'src/open_clip/model_configs').glob('PET*BioClinicalBERT*.json'))
        for source in sources:
            dest = snapshot / source.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        write_status(root, status)
        try:
            # Interleave experiments so all conditions receive initial results early.
            for lr, warmup in itertools.product((1e-5, 5e-5), (20, 60)):
                for key, cfg in EXPERIMENTS.items():
                    if shutil.disk_usage(root).free < 4 * 1024**3:
                        raise RuntimeError('Less than 4 GiB free; stopping before another trial.')
                    name = f'{root.name}-{key}-lr{lr:g}-warm{warmup}'
                    winner = status['winners'].get(key)
                    threshold = winner['best_value'] if winner else math.inf
                    cmd = ['bash', str(REPO / 'scripts/train_pet_scratch.sh'), args.data,
                           '--model', cfg['model'], '--lr', str(lr), '--warmup', str(warmup),
                           '--epochs', '20', '--batch-size', '256', '--accum-freq', '2',
                           '--logs', str(root / 'trials'),
                           '--seed', '0', '--workers', '4', '--force-context-length', '1024',
                           '--lr-scheduler', 'cosine', '--precision', 'amp_bf16', '--save-best-only',
                           '--best-metric', 'clip_val_loss', '--best-checkpoint-path', str(root / key / 'best.pt'),
                           '--best-checkpoint-threshold', str(threshold), '--early-stopping-patience', '3',
                           '--log-metric-every-n-steps', '1', '--name', name,
                           '--wandb-project-name', 'pet-bioclinicalbert-hpo', '--report-to', 'wandb']
                    if cfg['local_weight'] is not None:
                        cmd += ['--pet-region-loss-weight', str(cfg['local_weight']),
                                '--pet-region-chunk-size', '32', '--pet-region-max-tokens', '128']
                    if cfg['pretrained_image']:
                        cmd += ['--pretrained-image']
                    trial = dict(experiment=key, lr=lr, warmup=warmup, name=name, state='running', command=cmd)
                    status['trials'].append(trial)
                    write_status(root, status)
                    print(f'Starting {name}', flush=True)
                    with (root / f'{name}.terminal.log').open('w') as log:
                        result = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                    trial['returncode'] = result.returncode
                    if result.returncode:
                        trial['state'] = 'failed'
                        raise RuntimeError(f'{name} exited {result.returncode}; see its terminal log.')
                    records_path = root / 'trials' / name / 'checkpoints/results.jsonl'
                    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
                    if not records or any(not math.isfinite(r['clip_val_loss']) for r in records):
                        raise RuntimeError(f'{name} has missing/nonfinite validation results.')
                    best = min(records, key=lambda r: r['clip_val_loss'])
                    trial.update(state='completed', best_value=best['clip_val_loss'], best_epoch=best['epoch'],
                                 completed_epochs=records[-1]['epoch'], best_metrics=best)
                    if trial['best_value'] < threshold:
                        status['winners'][key] = {k: trial[k] for k in ('name', 'lr', 'warmup', 'best_value', 'best_epoch', 'best_metrics')}
                    write_status(root, status)
                    print(f"Completed {name}: best {trial['best_value']:.6f}, epoch {trial['best_epoch']}", flush=True)
            status['state'] = 'completed'
            write_status(root, status)
        except BaseException as exc:
            status.update(state='failed', error=str(exc))
            write_status(root, status)
            raise


if __name__ == '__main__':
    main()
