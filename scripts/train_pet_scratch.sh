#!/usr/bin/env bash
set -euo pipefail
# Pass the original dataset root; images and reports are read in place. Optional remaining args override training defaults.
DATA_DIR="$(realpath "${1:?Usage: train_pet_scratch.sh DATASET_ROOT [extra training args]}")"
shift
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
for required in PET_binaries data_scan_did.csv data_Scan_Report.csv data_splits.csv; do
  if [[ ! -e "$DATA_DIR/$required" ]]; then
    echo "Missing source dataset component: $required" >&2
    exit 1
  fi
done
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer" >&2
  exit 1
fi
launch=("$PYTHON_BIN" -m open_clip_train.main)
distributed_args=()
if (( NPROC_PER_NODE > 1 )); then
  "$PYTHON_BIN" - "$NPROC_PER_NODE" <<'PYCHECK'
import sys
import torch
count = int(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() < count:
    raise SystemExit(f"Need {count} visible CUDA GPUs; found {torch.cuda.device_count()} (torch {torch.__version__})")
if not torch.distributed.is_nccl_available():
    raise SystemExit("This PyTorch build does not support NCCL")
print(f"Launching {count} GPU workers using PyTorch {torch.__version__}", flush=True)
PYCHECK
  launch=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1
          --nproc_per_node="$NPROC_PER_NODE" --module -- open_clip_train.main)
  distributed_args=(--device cuda --dist-backend nccl --local-loss --gather-with-grad)
fi
# Both towers initialize randomly: no --pretrained or --pretrained-image.
# Compilation is opt-in: compiled text backward OOMs at batch 256 with 2560 tokens.
training_args=(
  --model PET-ResNet34
  --dataset-type pet --pet-split-column split0 --pet-cache ram --pet-report-sections findings_impression
  --train-data "$DATA_DIR" --val-data "$DATA_DIR"
  --force-context-length 2560 --force-image-size 224
  --pet-canvas-size 310 --pet-image-size 224 --pet-val-image-size 310 --pet-intensity-max 30
  --batch-size 256 --workers 16 --epochs 50 --warmup 200
  --lr 5e-4 --wd 0.1 --precision amp_bf16 --grad-checkpointing --report-to wandb --wandb-project-name pet-resnet34-concat-scratch
  --seed 0 --val-frequency 1 --zeroshot-frequency 0 --save-frequency 1 --delete-previous-checkpoint
  --logs "$REPO_DIR/.local/pet/logs" --name pet-resnet34-concat-scratch
)
exec "${launch[@]}" "${training_args[@]}" "${distributed_args[@]}" "$@"
