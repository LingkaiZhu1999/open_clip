#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:?Usage: train_pet_attention_1gpu.sh DATASET_ROOT [extra training args]}"
shift
export PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.local/pet-cuda-venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE=1
export HF_HOME="${HF_HOME:-$REPO_DIR/.local/pet/hf-cache}"
PET_LOG_DIR="${PET_LOG_DIR:-$REPO_DIR/.local/pet/logs}"
mkdir -p "$PET_LOG_DIR"
export WANDB_DIR="${WANDB_DIR:-$PET_LOG_DIR}"
image_init=()
case "${PET_PRETRAINED_IMAGE:-1}" in
  1) image_init=(--pretrained-image) ;;
  0) ;; # Backbone and fusion from scratch; Bio_ClinicalBERT stays pretrained.
  *) echo 'PET_PRETRAINED_IMAGE must be 0 or 1' >&2; exit 1 ;;
esac
# New architecture: these are starting settings, not measured optimal settings.
# 256 x 2 = 512 contrastive pairs/update. Benchmark memory before a full run;
# callers can override with e.g. --batch-size 128 --accum-freq 4 if necessary.
exec bash "$REPO_DIR/scripts/train_pet_scratch.sh" "$DATA_DIR" \
  --model PET-ResNet50-BioClinicalBERT-Attention "${image_init[@]}" \
  --force-context-length 1024 --batch-size 256 --accum-freq 2 --workers 4 \
  --lr 1e-5 --warmup 20 --epochs 20 --precision amp_bf16 \
  --save-best-only --best-metric clip_val_loss --early-stopping-patience 3 \
  --logs "$PET_LOG_DIR" --wandb-project-name pet-resnet50-attention \
  --name "pet-resnet50-attention-$(date +%Y%m%d-%H%M%S-%N)-$$" "$@"
