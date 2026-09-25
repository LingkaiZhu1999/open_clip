#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.local/pet-cuda-venv/bin/python}"
export NPROC_PER_NODE=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# Direct PCIe P2P stalls on this host; verified NCCL SHM fallback works.
# Override only after validating peer transfers on the target machine.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "CUDA Python not found: $PYTHON_BIN. Set PYTHON_BIN to a CUDA-enabled environment." >&2
  exit 1
fi
DATA_DIR="${1:?Usage: train_pet_2gpu.sh DATASET_ROOT [extra training args]}"
shift
# Generate once before torchrun so both ranks use the same unique run name.
RUN_NAME="pet-resnet34-concat-2gpu-$(date +%Y%m%d-%H%M%S-%N)-$$"
exec bash "$REPO_DIR/scripts/train_pet_scratch.sh" "$DATA_DIR" \
  --name "$RUN_NAME" "$@"
