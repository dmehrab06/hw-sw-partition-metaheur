#!/bin/bash
set -euo pipefail

# Sample one random config list and launch parallel batches on the same configs:
#   1) diff_gnn_order
#   2) diff_gnn
#   3) comparison methods (default: gcps,gl25)
#   4) mip batch (via run_all_mip_configs.sh / mip_eval.py)
#
# Usage:
#   ./run_two_methods_same_random_configs.sh
#
# Optional env overrides:
#   COUNT=10
#   CONFIG_PATTERN="configs/config_mkspan_area_*_hw_*_seed_*.yaml"
#   ORDER_CSV="dff_gnn_order_10_results.csv"
#   DIFF_CSV="dff_gnn_10_results.csv"
#   COMPARE_CSV="compare_gcps_gl25_10_results.csv"
#   COMPARE_METHODS="gcps,gl25"
#   RUNNER_LOG_DIR="outputs/logs"
#   RUN_TAG="mytag"
#   MIP_SOLVER_TOOL="cvxpy"
#   MIP_EVAL_PY="mip_eval.py"   # fallback to milp_eval.py if missing
#   MIP_FAST=1
#   MIP_RUN_TIMEOUT_SEC=120
#   MIP_OUTDIR="outputs/logs/mip_mytag"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

COUNT="${COUNT:-15}"
CONFIG_PATTERN="${CONFIG_PATTERN:-configs/config_mkspan_area_*_hw_*_seed_*.yaml}"
RUNNER_LOG_DIR="${RUNNER_LOG_DIR:-outputs/logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

ORDER_CSV="${ORDER_CSV:-dff_gnn_order_${COUNT}_results.csv}"
DIFF_CSV="${DIFF_CSV:-dff_gnn_${COUNT}_results.csv}"
COMPARE_CSV="${COMPARE_CSV:-compare_gcps_gl25_${COUNT}_results.csv}"
COMPARE_METHODS="${COMPARE_METHODS:-gcps,gl25}"

MIP_SOLVER_TOOL="${MIP_SOLVER_TOOL:-cvxpy}"
MIP_EVAL_PY="${MIP_EVAL_PY:-mip_eval.py}"
MIP_FAST="${MIP_FAST:-1}"
MIP_RUN_TIMEOUT_SEC="${MIP_RUN_TIMEOUT_SEC:-120}"
MIP_OUTDIR="${MIP_OUTDIR:-${RUNNER_LOG_DIR}/mip_${RUN_TAG}}"

ORDER_LOG="${RUNNER_LOG_DIR}/diff_gnn_order_${RUN_TAG}.log"
DIFF_LOG="${RUNNER_LOG_DIR}/diff_gnn_${RUN_TAG}.log"
COMPARE_LOG="${RUNNER_LOG_DIR}/compare_${RUN_TAG}.log"
MIP_LOG="${RUNNER_LOG_DIR}/mip_${RUN_TAG}.log"
CONFIG_LIST_FILE="${RUNNER_LOG_DIR}/shared_random_configs_${RUN_TAG}.txt"

mkdir -p "$RUNNER_LOG_DIR" "$MIP_OUTDIR"

mapfile -t ALL_CONFIGS < <(ls $CONFIG_PATTERN 2>/dev/null | sort || true)
if [[ ${#ALL_CONFIGS[@]} -eq 0 ]]; then
  echo "No config files matched: $CONFIG_PATTERN"
  exit 1
fi

if (( COUNT <= 0 )); then
  echo "COUNT must be > 0 (got $COUNT)"
  exit 1
fi

if (( COUNT > ${#ALL_CONFIGS[@]} )); then
  echo "COUNT=$COUNT is larger than matched configs=${#ALL_CONFIGS[@]}"
  exit 1
fi

mapfile -t PICKED_CONFIGS < <(printf '%s\n' "${ALL_CONFIGS[@]}" | shuf -n "$COUNT")
printf '%s\n' "${PICKED_CONFIGS[@]}" > "$CONFIG_LIST_FILE"
CONFIG_GLOB_VALUE="$(tr '\n' ' ' < "$CONFIG_LIST_FILE" | sed 's/[[:space:]]*$//')"

launch_gnn_batch() {
  local methods="$1"
  local csv_file="$2"
  local log_file="$3"
  nohup env \
    HWSW_RESULT_CSV="$csv_file" \
    HWSW_METHODS="$methods" \
    CONFIG_GLOB="$CONFIG_GLOB_VALUE" \
    ./run_all_gnn_configs.sh > "$log_file" 2>&1 &
  echo $!
}

ORDER_PID="$(launch_gnn_batch "diff_gnn_order" "$ORDER_CSV" "$ORDER_LOG")"
DIFF_PID="$(launch_gnn_batch "diff_gnn" "$DIFF_CSV" "$DIFF_LOG")"
COMPARE_PID="$(launch_gnn_batch "$COMPARE_METHODS" "$COMPARE_CSV" "$COMPARE_LOG")"

nohup env \
  CONFIG_GLOB="$CONFIG_GLOB_VALUE" \
  SOLVER_TOOL="$MIP_SOLVER_TOOL" \
  MIP_EVAL_PY="$MIP_EVAL_PY" \
  FAST_MIP="$MIP_FAST" \
  RUN_TIMEOUT_SEC="$MIP_RUN_TIMEOUT_SEC" \
  OUTDIR="$MIP_OUTDIR" \
  ./run_all_mip_configs.sh > "$MIP_LOG" 2>&1 &
MIP_PID=$!

echo "Launched runs on shared random config set:"
echo "  Config list file: $CONFIG_LIST_FILE"
echo "  Number of configs: $COUNT"
echo "  diff_gnn_order PID: $ORDER_PID | log: $ORDER_LOG | csv: $ORDER_CSV"
echo "  diff_gnn       PID: $DIFF_PID | log: $DIFF_LOG | csv: $DIFF_CSV"
echo "  compare($COMPARE_METHODS) PID: $COMPARE_PID | log: $COMPARE_LOG | csv: $COMPARE_CSV"
echo "  mip            PID: $MIP_PID | log: $MIP_LOG | outdir: $MIP_OUTDIR"
echo
echo "Manual MIP command on the same sampled configs:"
echo "  CONFIG_GLOB=\"$CONFIG_GLOB_VALUE\" SOLVER_TOOL=\"$MIP_SOLVER_TOOL\" MIP_EVAL_PY=\"$MIP_EVAL_PY\" OUTDIR=\"$MIP_OUTDIR\" ./run_all_mip_configs.sh"
