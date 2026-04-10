#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"

ALL_METHODS_ORDER=(
  "mip"
  "diff_gnn_order"
  "gl25"
  "gcps"
  "esa"
  "pso"
  "dbpso"
  "clpso"
  "ccpso"
  "shade"
  "jade"
  "random"
  "greedy"
)

METHODS=(
  # "mip"
  "diff_gnn_order"
  "gl25"
  "gcps"
  "esa"
  "pso"
  "dbpso"
  "clpso"
  "ccpso"
  "shade"
  "jade"
  "random"
  "greedy"
)

GRAPH_SIZES=(
  # "10"
  # "15"
  "1000"
  # "10000"
)

if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  read -r -a METHODS <<<"$METHODS_OVERRIDE"
fi
if [[ -n "${GRAPH_SIZES_OVERRIDE:-}" ]]; then
  read -r -a GRAPH_SIZES <<<"$GRAPH_SIZES_OVERRIDE"
fi

DATASETS=()
for size in "${GRAPH_SIZES[@]}"; do
  DATASETS+=("squeezenet_like_${size}")
done

RESULT_TAG="large_scale_area05_10seed"
OUTDIR="$ROOT/BatchExperiments/large_scale_area05"
MANIFEST="$OUTDIR/${RESULT_TAG}_selected_manifest.csv"
MIP_PLOT_METRIC="${MIP_PLOT_METRIC:-lp}"

mkdir -p "$OUTDIR"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST"
  exit 1
fi

PLOT_METHODS=()
for method in "${ALL_METHODS_ORDER[@]}"; do
  if [[ ! " ${METHODS[*]} " =~ " ${method} " ]]; then
    continue
  fi
  for dataset in "${DATASETS[@]}"; do
    if [[ -d "$OUTDIR/$dataset/$method" ]]; then
      PLOT_METHODS+=("$method")
      break
    fi
  done
done

if [[ ${#PLOT_METHODS[@]} -eq 0 ]]; then
  echo "No selected method folders were found under $OUTDIR"
  exit 1
fi

"$PYTHON" "$ROOT/tools/plot_large_scale_results.py" \
  --search-root "$OUTDIR" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTDIR" \
  --methods "${PLOT_METHODS[@]}" \
  --datasets "${DATASETS[@]}" \
  --mip-metric "$MIP_PLOT_METRIC" \
  --tag "$RESULT_TAG"

echo "Finished plotting large-scale synthetic results from $OUTDIR"
