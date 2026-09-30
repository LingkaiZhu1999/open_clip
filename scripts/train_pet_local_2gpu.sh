#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:?Usage: train_pet_local_2gpu.sh DATASET_ROOT [extra training args]}"
shift
# Conservative initial batch: local matching is quadratic in global pair count.
# Both encoders remain random-initialized; additional user arguments override defaults.
exec bash "$REPO_DIR/scripts/train_pet_2gpu.sh" "$DATA_DIR" \
  --model PET-ResNet34-Local --batch-size 256 --workers 4 \
  --pet-image-size 224 --pet-val-image-size 224 \
  --pet-region-loss-weight 1.0 --pet-region-max-tokens 128 --pet-region-chunk-size 4 \
  --wandb-project-name pet-resnet34-global-local-scratch \
  --name "pet-resnet34-global-local-2gpu-$(date +%Y%m%d-%H%M%S-%N)-$$" "$@"
