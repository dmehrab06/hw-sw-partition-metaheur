#!/bin/bash
set -euo pipefail

# Sample one random config list and launch two independent runs:
#   1) diff_gnn_order
#   2) diff_gnn
# Both runs use the exact same sampled configs.
#
# Usage:
#   ./run_two_methods_same_random_configs.sh
#
# Optional env overrides:
#   COUNT=10
#   CONFIG_PATTERN="configs/config_mkspan_area_*_hw_*_seed_*.yaml"
#   ORDER_CSV="dff_gnn_order_10_results.csv"
#   DIFF_CSV="dff_gnn_10_results.csv"
#   RUNNER_LOG_DIR="outputs/logs"
#   RUN_TAG="mytag"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

COUNT="${COUNT:-10}"
CONFIG_PATTERN="${CONFIG_PATTERN:-configs/config_mkspan_area_*_hw_*_seed_*.yaml}"
RUNNER_LOG_DIR="${RUNNER_LOG_DIR:-outputs/logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

ORDER_CSV="${ORDER_CSV:-dff_gnn_order_${COUNT}_results.csv}"
DIFF_CSV="${DIFF_CSV:-dff_gnn_${COUNT}_results.csv}"

ORDER_LOG="${RUNNER_LOG_DIR}/diff_gnn_order_${RUN_TAG}.log"
DIFF_LOG="${RUNNER_LOG_DIR}/diff_gnn_${RUN_TAG}.log"
CONFIG_LIST_FILE="${RUNNER_LOG_DIR}/shared_random_configs_${RUN_TAG}.txt"

mkdir -p "$RUNNER_LOG_DIR"

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

nohup env \
  HWSW_RESULT_CSV="$ORDER_CSV" \
  HWSW_METHODS="diff_gnn_order" \
  CONFIG_GLOB="$CONFIG_GLOB_VALUE" \
  ./run_all_gnn_configs.sh > "$ORDER_LOG" 2>&1 &
ORDER_PID=$!

nohup env \
  HWSW_RESULT_CSV="$DIFF_CSV" \
  HWSW_METHODS="diff_gnn" \
  CONFIG_GLOB="$CONFIG_GLOB_VALUE" \
  ./run_all_gnn_configs.sh > "$DIFF_LOG" 2>&1 &
DIFF_PID=$!

echo "Launched runs on shared random config set:"
echo "  Config list file: $CONFIG_LIST_FILE"
echo "  Number of configs: $COUNT"
echo "  diff_gnn_order PID: $ORDER_PID | log: $ORDER_LOG | csv: $ORDER_CSV"
echo "  diff_gnn       PID: $DIFF_PID | log: $DIFF_LOG | csv: $DIFF_CSV"

