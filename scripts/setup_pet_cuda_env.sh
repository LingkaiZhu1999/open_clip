#!/usr/bin/env bash
set -euo pipefail
# Reuse a working CUDA environment read-only to avoid duplicating GPU libraries.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_PYTHON="${1:?Usage: setup_pet_cuda_env.sh /path/to/existing/cuda-env/bin/python}"
TARGET="$REPO_DIR/.local/pet-cuda-venv"
"$SHARED_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Source environment must support CUDA"'
uv venv --allow-existing --python "$SHARED_PYTHON" "$TARGET"
SHARED_SITE="$("$SHARED_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
"$TARGET/bin/python" - "$SHARED_SITE" <<'PY'
import site, sys
from pathlib import Path
Path(site.getsitepackages()[0], 'shared_cuda_runtime.pth').write_text(sys.argv[1] + '\n')
PY
uv pip install --python "$TARGET/bin/python" --no-deps --no-cache -r "$REPO_DIR/scripts/requirements-pet-runtime.txt"
PYTHONPATH="$REPO_DIR/src" "$TARGET/bin/python" -c 'import open_clip, open_clip_train.main, torch; print("CUDA runtime ready:", torch.__version__, "GPUs:", torch.cuda.device_count())'
