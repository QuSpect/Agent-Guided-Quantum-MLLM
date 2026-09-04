#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
SRC_DIR="$PROJECT_DIR/src"
RECORD_DIR="$PROJECT_DIR/records"
CACHE_DIR="$PROJECT_DIR/cache"
BASE_PYTHON=python3

mkdir -p "$SRC_DIR" "$RECORD_DIR" "$CACHE_DIR/pip" "$CACHE_DIR/huggingface"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export HF_HOME="$CACHE_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/huggingface/hub"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ ! -d "$SRC_DIR/transformers/.git" ]]; then
  if ! git clone --depth 1 https://github.com/huggingface/transformers.git "$SRC_DIR/transformers"; then
    source /etc/network_turbo
    git clone --depth 1 https://github.com/huggingface/transformers.git "$SRC_DIR/transformers"
    unset http_proxy https_proxy
  fi
fi
python -m pip install --editable "$SRC_DIR/transformers"

PACKAGES=(
  accelerate
  peft
  datasets
  "huggingface_hub[hf_xet]"
  safetensors
  sentencepiece
  pillow
  qwen-vl-utils
  evaluate
  scikit-learn
  scipy
  pandas
  pyyaml
  pytest
  psutil
  pynvml
  ninja
  packaging
  rich
  tensorboard
)
python -m pip install "${PACKAGES[@]}"

git -C "$SRC_DIR/transformers" rev-parse HEAD > "$RECORD_DIR/transformers_commit.txt"
python -m pip freeze > "$RECORD_DIR/pip_freeze_bootstrap.txt"

python - <<'PY' > "$RECORD_DIR/bootstrap_check.json"
import json
import platform

import accelerate
import datasets
import evaluate
import huggingface_hub
import peft
import torch
import transformers

result = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "peft": peft.__version__,
    "datasets": datasets.__version__,
    "evaluate": evaluate.__version__,
    "huggingface_hub": huggingface_hub.__version__,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

echo "BOOTSTRAP_V2_OK"
