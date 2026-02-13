#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$ROOT/configs/config_mkspan_default_gnn.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

echo "[gpu] config: $CONFIG_PATH"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu] nvidia-smi -L"
  nvidia-smi -L || true
else
  echo "[gpu] nvidia-smi not found"
fi

python - <<'PY' "$CONFIG_PATH"
import sys
from omegaconf import OmegaConf
import torch

from meta_heuristic.diff_gnn_utils_schedule import get_device

cfg = OmegaConf.load(sys.argv[1])

diff_cfg = dict(cfg.get("diffgnn", {}))
if "device" in diff_cfg:
    cfg["device"] = diff_cfg.get("device")
elif "device" not in cfg:
    cfg["device"] = "gpu"

device = get_device(cfg)

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    except Exception:
        pass
    try:
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

print(f"selected_device: {device}")
print(f"using_gpu: {device == 'cuda'}")
PY
