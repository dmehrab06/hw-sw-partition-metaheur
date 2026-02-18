#!/bin/bash
set -euo pipefail

# Run diff_gnn and diff_gnn_order on a set of configs (default: all YAMLs in configs/).
# Results go to outputs/diff_gnn_order/makespan_opt-result-summary-soda-graphs-config.csv
# Logs go to outputs/logs_diff_gnn_order/
# Override OUTDIR/LOGDIR to point at a custom prefix.
# Override HWSW_METHODS to run a custom method list.
# Optional: set CSV_OUT=/path/to/file.csv to copy the aggregated CSV there.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$ROOT/configs"
OUTDIR="${OUTDIR:-$ROOT/outputs/diff_gnn_order}"
LOGDIR="${LOGDIR:-$ROOT/outputs/logs_diff_gnn_order}"

PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
CONFIG_GLOB="${CONFIG_GLOB:-$CONFIG_DIR/*.yaml}"
METHODS_ENV="${HWSW_METHODS:-diff_gnn,diff_gnn_order}"
export PYTHONNOUSERSITE=1

mkdir -p "$OUTDIR" "$LOGDIR"

# Collect configs
mapfile -t CONFIGS < <(ls $CONFIG_GLOB 2>/dev/null | sort || true)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No config files matched: $CONFIG_GLOB"
  exit 1
fi

echo "Running methods [$METHODS_ENV] on ${#CONFIGS[@]} configs"

for cfg_path in "${CONFIGS[@]}"; do
  cfg_base="$(basename "$cfg_path" .yaml)"
  log_file="$LOGDIR/diff_gnn_order_${cfg_base}.log"
  echo "---- [$METHODS_ENV] $cfg_base ----"
  HWSW_METHODS="$METHODS_ENV" HWSW_CSV_DIR="$OUTDIR" "$PYTHON" "$ROOT/gnn_main.py" -c "$cfg_path" >"$log_file" 2>&1 || {
    echo "Run failed for $cfg_base (see $log_file)"
    continue
  }
done

echo "Done. New CSV is at $OUTDIR/makespan_opt-result-summary-soda-graphs-config.csv"
echo "Logs in $LOGDIR"

if [[ -n "${CSV_OUT:-}" ]]; then
  src="$OUTDIR/makespan_opt-result-summary-soda-graphs-config.csv"
  if [[ -f "$src" ]]; then
    dst="$CSV_OUT"
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "Also copied to $dst"
  else
    echo "CSV_OUT requested but source file not found: $src"
  fi
fi
