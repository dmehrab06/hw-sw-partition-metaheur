#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$ROOT/configs"
OUTDIR="$ROOT/outputs/logs"
mkdir -p "$OUTDIR"

export PYTHONNOUSERSITE=1

PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
CONFIG_GLOB="${CONFIG_GLOB:-$CONFIG_DIR/config_mkspan_default_gnn.yaml}"
METHODS_ENV="${HWSW_METHODS:-${METHODS:-}}"

cd "$ROOT"

mapfile -t CONFIGS < <(ls $CONFIG_GLOB 2>/dev/null | sort || true)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No config files matched: $CONFIG_GLOB"
  exit 1
fi

if [[ -n "$METHODS_ENV" ]]; then
  echo "Running gnn_main.py on ${#CONFIGS[@]} configs (methods=$METHODS_ENV)"
else
  echo "Running gnn_main.py on ${#CONFIGS[@]} configs (methods=default)"
fi

for config in "${CONFIGS[@]}"; do
  config_base="$(basename "$config" .yaml)"
  log_file="$OUTDIR/gnn_main_${config_base}.log"

  echo "---- [GNN] $config_base ----"
  if [[ -n "$METHODS_ENV" ]]; then
    HWSW_METHODS="$METHODS_ENV" "$PYTHON" gnn_main.py -c "$config" >"$log_file" 2>&1 || {
      echo "gnn_main.py failed for $config (see $log_file)"
      continue
    }
  else
    "$PYTHON" gnn_main.py -c "$config" >"$log_file" 2>&1 || {
      echo "gnn_main.py failed for $config (see $log_file)"
      continue
    }
  fi

  result_prefix=$("$PYTHON" - <<'PY' "$config"
from omegaconf import OmegaConf
import sys
cfg = OmegaConf.load(sys.argv[1])
print(cfg.get('result-file-prefix', 'makespan_opt_gnn'))
PY
)

  out_src="$ROOT/outputs/${result_prefix}-result-summary-soda-graphs-config.csv"
  if [[ -f "$out_src" ]]; then
    cp "$out_src" "$OUTDIR/${result_prefix}-result-summary-soda-graphs-config.csv"
  fi

done

echo "GNN batch complete. CSV copies are in $OUTDIR"


# commands
# ./run_all_gnn_configs.sh
# CONFIG_GLOB="configs/config_mkspan_area_*_hw_*_seed_*.yaml" ./run_all_gnn_configs.sh

# HWSW_METHODS="pso,dbpso,clpso,ccpso,gl25,esa,shade,jade,random,greedy,gnn,diff_gnn,non_diffgnn" \
# CONFIG_GLOB="configs/config_mkspan_area_*_hw_*_seed_*.yaml" \
# ./run_all_gnn_configs.sh

# HWSW_METHODS="pso,dbpso,clpso,ccpso,gl25,esa,shade,jade,random,greedy,diff_gnn" \
# CONFIG_GLOB="configs/config_mkspan_area_*_hw_*_seed_*.yaml" \
# ./run_all_gnn_configs.sh
