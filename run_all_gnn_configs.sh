#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$ROOT/configs"
OUTDIR="${OUTDIR:-$ROOT/outputs/logs}"
mkdir -p "$OUTDIR"

export PYTHONNOUSERSITE=1

PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
CONFIG_GLOB="${CONFIG_GLOB:-$CONFIG_DIR/config_mkspan_default_gnn.yaml}"
METHODS_ENV="${HWSW_METHODS:-${METHODS:-}}"
RESULT_CSV_ENV="${HWSW_RESULT_CSV:-${RESULT_CSV:-}}"
RESULT_PREFIX_ENV="${HWSW_RESULT_PREFIX:-${RESULT_PREFIX:-}}"
CSV_DIR_ENV="${HWSW_CSV_DIR:-${CSV_DIR:-}}"
RUN_TAG_ENV="${HWSW_RUN_TAG:-${RUN_TAG:-}}"
INPROC_RUNNER="$ROOT/tools/run_gnn_configs_inproc.py"

cd "$ROOT"

mapfile -t CONFIGS < <(ls $CONFIG_GLOB 2>/dev/null | sort || true)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No config files matched: $CONFIG_GLOB"
  exit 1
fi

if [[ -n "$METHODS_ENV" ]]; then
  echo "Running gnn_main.py on ${#CONFIGS[@]} configs (selected methods=$METHODS_ENV)"
else
  echo "Running gnn_main.py on ${#CONFIGS[@]} configs (selected methods=default)"
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
if [[ -n "$RUN_TAG_ENV" ]]; then
  echo "Run tag: $RUN_TAG_ENV"
fi

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
if [[ -n "$RUN_TAG_ENV" ]]; then
  run_env+=(HWSW_RUN_TAG="$RUN_TAG_ENV")
fi

if [[ ${#run_env[@]} -gt 0 ]]; then
  env "${run_env[@]}" "$PYTHON" "$INPROC_RUNNER" --root "$ROOT" --outdir "$OUTDIR" "${CONFIGS[@]}"
else
  "$PYTHON" "$INPROC_RUNNER" --root "$ROOT" --outdir "$OUTDIR" "${CONFIGS[@]}"
fi


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
