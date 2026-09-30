#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:?Usage: train_pet_bioclinicalbert_2gpu.sh DATASET_ROOT [extra training args]}"
shift
# Store checkpoints and experiment logs in the original repository location.
PET_LOG_DIR="${PET_LOG_DIR:-$REPO_DIR/.local/pet/logs}"
mkdir -p "$PET_LOG_DIR"
export WANDB_DIR="${WANDB_DIR:-$PET_LOG_DIR}"
export HF_HOME="${HF_HOME:-$REPO_DIR/.local/pet/hf-cache}"
# Pretrained, trainable Bio_ClinicalBERT; random PET ResNet and alignment projections.
# Reports retain 1024 tokens, encoded in native 512-position BERT windows.
exec bash "$REPO_DIR/scripts/train_pet_local_2gpu.sh" "$DATA_DIR" \
  --model PET-ResNet34-BioClinicalBERT-Local --force-context-length 1024 \
  --batch-size 256 --pet-region-chunk-size 32 --lr 5e-5 \
  --epochs 10 \
  --save-frequency 1 --logs "$PET_LOG_DIR" \
  --wandb-project-name pet-resnet34-bioclinicalbert \
  --name "pet-bioclinicalbert-2gpu-$(date +%Y%m%d-%H%M%S-%N)-$$" "$@"
