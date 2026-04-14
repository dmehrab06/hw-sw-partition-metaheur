#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_COMBOPT_PYTHON="/people/dass304/.conda/envs/combopt/bin/python"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$DEFAULT_COMBOPT_PYTHON" ]]; then
    PYTHON="$DEFAULT_COMBOPT_PYTHON"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  else
    PYTHON="python"
  fi
fi
OUTROOT="${OUTROOT:-$ROOT/BatchExperiments/Ablation/diff_gnn_order_lssp_trace}"
LATEST_ONLY="${LATEST_ONLY:-1}"
EPOCH_STRIDE="${EPOCH_STRIDE:-25}"
RUN_OVERRIDE=""

# Parse simple command-line options: --run RUN_ID to pick a specific run directory,
# --all to disable latest-only behavior.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_OVERRIDE="$2"
      shift 2
      ;;
    --run=*)
      RUN_OVERRIDE="${1#--run=}"
      shift
      ;;
    --all)
      LATEST_ONLY=0
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--run RUN_ID] [--all]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${RUN_OVERRIDE:-}" ]]; then
  # Normalize run dir name: accept both 'run-...' and '...'
  if [[ "$RUN_OVERRIDE" == run-* ]]; then
    RUN_DIR="$RUN_OVERRIDE"
  else
    RUN_DIR="run-$RUN_OVERRIDE"
  fi
  mapfile -t TRACE_CSVS < <(find "$OUTROOT" -type f -path "*/${RUN_DIR}/*_diff_gnn_order_lssp_trace.csv" | sort || true)
elif [[ "$LATEST_ONLY" =~ ^(1|true|yes|on)$ ]]; then
  # For latest-per-dataset behavior, pick the newest run directory per top-level dataset
  mapfile -t TRACE_CSVS < <(
    "$PYTHON" - <<'PY' "$OUTROOT"
from pathlib import Path
import sys

root = Path(sys.argv[1])
grouped = {}
for csv_path in sorted(root.glob("*/*/run-*/*_diff_gnn_order_lssp_trace.csv")):
    try:
        rel = csv_path.relative_to(root)
    except ValueError:
        continue
    parts = rel.parts
    if len(parts) < 4:
        continue
    # group by top-level dataset directory (parts[0])
    group_key = parts[0]
    prev = grouped.get(group_key)
    if prev is None or csv_path.parent.name > prev.parent.name:
        grouped[group_key] = csv_path
for path in sorted(grouped.values()):
    print(path)
PY
  )
else
  mapfile -t TRACE_CSVS < <(find "$OUTROOT" -type f -name '*_diff_gnn_order_lssp_trace.csv' | sort || true)
fi

if [[ "${#TRACE_CSVS[@]}" -eq 0 ]]; then
  echo "No ablation trace CSVs found under $OUTROOT" >&2
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Plotting diff_gnn_order ablation traces"
echo "  Python   : $PYTHON"
echo "  Out root : $OUTROOT"
echo "  Latest   : $LATEST_ONLY"
echo "  Stride   : $EPOCH_STRIDE"
echo "  Traces   : ${#TRACE_CSVS[@]}"

for trace_csv in "${TRACE_CSVS[@]}"; do
  png_path="${trace_csv%.csv}.png"
  pdf_path="${trace_csv%.csv}.pdf"
  echo
  echo "Trace: $trace_csv"
  echo "Plot : $png_path"
  echo "PDF  : $pdf_path"
  "$PYTHON" "$ROOT/tools/plot_diff_gnn_order_ablation_trace.py" \
    --input-csv "$trace_csv" \
    --output-png "$png_path" \
    --epoch-stride "$EPOCH_STRIDE"
done

echo
echo "Finished plotting diff_gnn_order ablation traces."
