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
RESULT_CSV_ENV="${HWSW_RESULT_CSV:-${RESULT_CSV:-}}"
RESULT_PREFIX_ENV="${HWSW_RESULT_PREFIX:-${RESULT_PREFIX:-}}"
CSV_DIR_ENV="${HWSW_CSV_DIR:-${CSV_DIR:-}}"

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
if [[ -n "$RESULT_CSV_ENV" ]]; then
  echo "CSV output override: $RESULT_CSV_ENV"
fi
if [[ -n "$RESULT_PREFIX_ENV" ]]; then
  echo "CSV prefix override: $RESULT_PREFIX_ENV"
fi
if [[ -n "$CSV_DIR_ENV" ]]; then
  echo "CSV directory override: $CSV_DIR_ENV"
fi

for config in "${CONFIGS[@]}"; do
  config_base="$(basename "$config" .yaml)"
  log_file="$OUTDIR/gnn_main_${config_base}.log"

  echo "---- [GNN] $config_base ----"
  run_env=( )
  if [[ -n "$METHODS_ENV" ]]; then
    run_env+=(HWSW_METHODS="$METHODS_ENV")
  fi
  if [[ -n "$RESULT_CSV_ENV" ]]; then
    run_env+=(HWSW_RESULT_CSV="$RESULT_CSV_ENV")
  fi
  if [[ -n "$RESULT_PREFIX_ENV" ]]; then
    run_env+=(HWSW_RESULT_PREFIX="$RESULT_PREFIX_ENV")
  fi
  if [[ -n "$CSV_DIR_ENV" ]]; then
    run_env+=(HWSW_CSV_DIR="$CSV_DIR_ENV")
  fi

  if [[ ${#run_env[@]} -gt 0 ]]; then
    env "${run_env[@]}" "$PYTHON" gnn_main.py -c "$config" >"$log_file" 2>&1 || {
      echo "gnn_main.py failed for $config (see $log_file)"
      continue
    }
  else
    "$PYTHON" gnn_main.py -c "$config" >"$log_file" 2>&1 || {
      echo "gnn_main.py failed for $config (see $log_file)"
      continue
    }
  fi

  out_src=$(env \
    HWSW_CSV_DIR="${CSV_DIR_ENV}" \
    HWSW_RESULT_CSV="${RESULT_CSV_ENV}" \
    HWSW_RESULT_PREFIX="${RESULT_PREFIX_ENV}" \
    "$PYTHON" - <<'PY' "$config" "$ROOT"
from omegaconf import OmegaConf
import os
import sys
cfg = OmegaConf.load(sys.argv[1])
root = sys.argv[2]
out_dir = os.getenv("HWSW_CSV_DIR") or cfg.get('output-dir', 'outputs')
csv_override = os.getenv("HWSW_RESULT_CSV") or cfg.get("result-csv") or cfg.get("result-csv-name")
result_prefix = os.getenv("HWSW_RESULT_PREFIX") or cfg.get('result-file-prefix', 'results')
if csv_override:
    out_path = csv_override if os.path.isabs(csv_override) else os.path.join(out_dir, csv_override)
else:
    out_path = os.path.join(out_dir, f"{result_prefix}-result-summary-soda-graphs-config.csv")
if not os.path.isabs(out_path):
    out_path = os.path.join(root, out_path)
print(out_path)
PY
)

  if [[ -f "$out_src" ]]; then
    out_copy="$OUTDIR/$(basename "$out_src")"
    if [[ "$(realpath "$out_src")" != "$(realpath "$out_copy" 2>/dev/null || echo "")" ]]; then
      cp "$out_src" "$out_copy"
    fi
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

# Custom output file name (appends all configs to one CSV in output-dir)
# HWSW_RESULT_CSV="my_custom_results.csv" \
# CONFIG_GLOB="configs/config_mkspan_area_*_hw_*_seed_*.yaml" \
# ./run_all_gnn_configs.sh

# Custom output directory + file name
# HWSW_CSV_DIR="outputs/analysis_outputs" \
# HWSW_RESULT_CSV="my_custom_results.csv" \
# ./run_all_gnn_configs.sh
