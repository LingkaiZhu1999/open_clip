#!/usr/bin/env python3
"""Run ImageNet/global+local ablation D using the selected C trial's settings."""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]


def build_command(reference, root):
    winner = reference['winners']['C']
    trial = next(t for t in reference['trials'] if t['name'] == winner['name'])
    if reference['state'] != 'completed' or trial['returncode'] != 0:
        raise ValueError('Reference C campaign must be completed successfully')
    command = trial['command'].copy()
    if '--pretrained-image' not in command or '--resume' in command:
        raise ValueError('Reference C must start from ImageNet weights without checkpoint resume')
    replacements = {
        '--model': 'PET-ResNet34-BioClinicalBERT-Local',
        '--logs': str(root / 'trials'),
        '--name': root.name + '-D',
        '--best-checkpoint-path': str(root / 'D/best.pt'),
        # C's threshold belongs to its earlier grid trial; D must select its own best.
        '--best-checkpoint-threshold': 'inf',
    }
    for flag, value in replacements.items():
        command[command.index(flag) + 1] = value
    command += ['--pet-region-loss-weight', '0.1', '--pet-region-chunk-size', '32',
                '--pet-region-max-tokens', '128']
    return command, winner


def write_status(root, status):
    status['updated_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    temporary = root / 'status.json.tmp'
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + '\n')
    temporary.replace(root / 'status.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', default=str(REPO / '.local/pet/logs/hpo-20261001-abc-1gpu'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reference_root = Path(args.reference).resolve()
    reference = json.loads((reference_root / 'status.json').read_text())
    with (root / 'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'status.json').exists():
            raise SystemExit('Use a fresh output directory for D')
        if shutil.disk_usage(root).free < 4 * 1024**3:
            raise SystemExit('Need at least 4 GiB free for best-checkpoint atomic replacement')
        command, winner = build_command(reference, root)
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), NPROC_PER_NODE='1',
                   PYTHON_BIN=str(REPO / '.local/pet-cuda-venv/bin/python'),
                   PET_LOG_DIR=str(root / 'trials'), WANDB_DIR=str(root),
                   HF_HOME=str(REPO / '.local/pet/hf-cache'), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                   TZ='Europe/Stockholm')
        # A small source snapshot captures the exact launch implementation without data copies.
        (root / 'code.patch').write_text(subprocess.check_output(['git', 'diff'], cwd=REPO, text=True))
        (root / 'git_revision.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True))
        sources = [Path(__file__), REPO / 'scripts/train_pet_scratch.sh']
        sources += list((REPO / 'src/open_clip_train').glob('*.py'))
        sources += list((REPO / 'src/open_clip').glob('*.py'))
        sources += list((REPO / 'src/open_clip/model_configs').glob('PET-ResNet34*BioClinicalBERT*.json'))
        for source in sources:
            dest = root / 'source_snapshot' / source.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        status = dict(state='running', experiment='D', pid=os.getpid(), gpu=args.gpu,
                      name=command[command.index('--name') + 1], command=command,
                      reference_campaign=str(reference_root), reference_C=winner,
                      initialization='ImageNet ResNet34 + pretrained Bio_ClinicalBERT; fresh local projections',
                      local_weight=0.1, lr=winner['lr'], warmup=winner['warmup'],
                      batch_size=256, accum_freq=2)
        write_status(root, status)
        try:
            with (root / 'terminal.log').open('w') as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                status['training_pid'] = child.pid
                write_status(root, status)
                result = child.wait()
            status['returncode'] = result
            if result:
                raise RuntimeError(f'D exited {result}; see terminal.log')
            path = root / 'trials' / status['name'] / 'checkpoints/results.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not records or any(not math.isfinite(r['clip_val_loss']) for r in records):
                raise RuntimeError('Missing or nonfinite D validation results')
            best = min(records, key=lambda r: r['clip_val_loss'])
            if not (root / 'D/best.pt').is_file():
                raise RuntimeError('D has no retained checkpoint')
            status.update(state='completed', best_epoch=best['epoch'], best_value=best['clip_val_loss'],
                          best_metrics=best, completed_epochs=records[-1]['epoch'])
            write_status(root, status)
        except BaseException as exc:
            status.update(state='failed', error=str(exc))
            write_status(root, status)
            raise


if __name__ == '__main__':
    main()
