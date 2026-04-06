#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"

PROFILE="${PROFILE:-pilot}"
SEEDS_STR="${SEEDS:-42 43 44}"
AREA="${AREA:-0.5}"
SQUEEZE_AREAS_STR="${SQUEEZE_AREAS:-0.1 0.3 0.5 0.7}"
SEED_FILTER_STR="${SEED_FILTER:-}"
AREA_FILTER_STR="${AREA_FILTER:-}"
MAX_CONFIGS="${MAX_CONFIGS:-0}"

GENERATE_CONFIGS="${GENERATE_CONFIGS:-1}"
RUN_GNN="${RUN_GNN:-1}"
RUN_MIP="${RUN_MIP:-1}"
RUN_PLOT="${RUN_PLOT:-1}"
CLEAR_RESULTS="${CLEAR_RESULTS:-1}"

FAST_MIP="${FAST_MIP:-1}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-120}"
TIMEOUT_KILL_AFTER_SEC="${TIMEOUT_KILL_AFTER_SEC:-15}"

METHODS_ENV="${HWSW_METHODS:-random,greedy,diff_gnn,diff_gnn_order,gcps,pso,dbpso,clpso,ccpso,esa,shade,jade,gl25}"

CONFIG_PROFILE_ROOT="$ROOT/inputs/task_graph_topology_config/$PROFILE"
MANIFEST="$CONFIG_PROFILE_ROOT/squeeze_net_tosa_area_sweep/manifest.csv"
SUITE_NAME="squeezenet_area_sweep"
RESULT_PREFIX="${SUITE_NAME}_${PROFILE}"
OUTDIR="$ROOT/outputs/$SUITE_NAME/$PROFILE"
GNN_CSV="$OUTDIR/${RESULT_PREFIX}-result-summary-soda-graphs-config.csv"
MIP_CSV="$OUTDIR/mip_${RESULT_PREFIX}-result-summary-soda-graphs-config.csv"

TMP_ROOT="$(mktemp -d "$ROOT/outputs/.squeezenet_sweep_${PROFILE}.XXXXXX")"
trap 'rm -rf "$TMP_ROOT"' EXIT

read -ra SEED_ARGS <<<"$SEEDS_STR"
read -ra SQUEEZE_AREAS <<<"$SQUEEZE_AREAS_STR"
read -ra SEED_FILTER <<<"$SEED_FILTER_STR"
read -ra AREA_FILTER <<<"$AREA_FILTER_STR"

mkdir -p "$OUTDIR"

if [[ "$CLEAR_RESULTS" == "1" ]]; then
  rm -f \
    "$GNN_CSV" \
    "$MIP_CSV" \
    "$OUTDIR/${RESULT_PREFIX}_summary.csv" \
    "$OUTDIR/${RESULT_PREFIX}_reported_barchart.png" \
    "$OUTDIR/${RESULT_PREFIX}_reported_barchart.pdf" \
    "$OUTDIR/${RESULT_PREFIX}_normalized_barchart.png" \
    "$OUTDIR/${RESULT_PREFIX}_normalized_barchart.pdf"
fi

if [[ "$GENERATE_CONFIGS" == "1" ]]; then
  "$PYTHON" "$ROOT/tools/generate_task_graph_topology_configs.py" \
    --profile "$PROFILE" \
    --area "$AREA" \
    --seeds "${SEED_ARGS[@]}" \
    --squeeze-areas "${SQUEEZE_AREAS[@]}"
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

COMMON_SELECT_ARGS=(--manifest "$MANIFEST" --max-configs "$MAX_CONFIGS")
if [[ ${#SEED_FILTER[@]} -gt 0 ]]; then
  COMMON_SELECT_ARGS+=(--seeds "${SEED_FILTER[@]}")
fi
if [[ ${#AREA_FILTER[@]} -gt 0 ]]; then
  COMMON_SELECT_ARGS+=(--areas "${AREA_FILTER[@]}")
fi

GNN_CFG_DIR="$TMP_ROOT/gnn_configs"
"$PYTHON" "$ROOT/tools/select_configs_from_manifest.py" \
  "${COMMON_SELECT_ARGS[@]}" \
  --out-dir "$GNN_CFG_DIR"

if [[ "$RUN_GNN" == "1" ]]; then
  CONFIG_GLOB="$GNN_CFG_DIR/*.yaml" \
  HWSW_METHODS="$METHODS_ENV" \
  HWSW_CSV_DIR="$OUTDIR" \
  HWSW_RESULT_PREFIX="$RESULT_PREFIX" \
  PYTHON="$PYTHON" \
  "$ROOT/run_all_gnn_configs.sh"
fi

if [[ "$RUN_MIP" == "1" ]]; then
  CONFIG_GLOB="$GNN_CFG_DIR/*.yaml" \
  OUTDIR="$OUTDIR" \
  FAST_MIP="$FAST_MIP" \
  RUN_TIMEOUT_SEC="$RUN_TIMEOUT_SEC" \
  TIMEOUT_KILL_AFTER_SEC="$TIMEOUT_KILL_AFTER_SEC" \
  PYTHON="$PYTHON" \
  "$ROOT/run_all_mip_configs.sh"
fi

if [[ "$RUN_PLOT" == "1" ]]; then
  "$PYTHON" "$ROOT/tools/plot_task_graph_barcharts.py" \
    --manifest "$MANIFEST" \
    --gnn-csv "$GNN_CSV" \
    --mip-csv "$MIP_CSV" \
    --mode area_sweep \
    --output-dir "$OUTDIR" \
    --tag "${RESULT_PREFIX}"
fi

echo "SqueezeNet area sweep complete."
