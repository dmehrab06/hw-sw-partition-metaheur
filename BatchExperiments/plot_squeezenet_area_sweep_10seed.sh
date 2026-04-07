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

AREAS=(0.1 0.3 0.5 0.7)

if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  read -r -a METHODS <<<"$METHODS_OVERRIDE"
fi
if [[ -n "${AREAS_OVERRIDE:-}" ]]; then
  read -r -a AREAS <<<"$AREAS_OVERRIDE"
fi

RESULT_TAG="squeezenet_area_sweep_10seed"
OUTDIR="$ROOT/BatchExperiments/squeezenet_area_sweep"
MANIFEST="$OUTDIR/${RESULT_TAG}_selected_manifest.csv"
GNN_CSV="$OUTDIR/${RESULT_TAG}-result-summary-soda-graphs-config.csv"
MIP_CSV="$OUTDIR/mip_${RESULT_TAG}-result-summary-soda-graphs-config.csv"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST"
  exit 1
fi

PLOT_ARGS=(
  "$PYTHON" "$ROOT/tools/plot_batch_method_bars.py"
  --manifest "$MANIFEST"
  --output-dir "$OUTDIR"
  --mode area_sweep
  --methods
)

for method in "${ALL_METHODS_ORDER[@]}"; do
  if [[ " ${METHODS[*]} " =~ " ${method} " ]]; then
    PLOT_ARGS+=("$method")
  fi
done

PLOT_ARGS+=(--areas)
for area in "${AREAS[@]}"; do
  PLOT_ARGS+=("$area")
done

PLOT_ARGS+=(--tag "$RESULT_TAG")

if [[ -f "$GNN_CSV" ]]; then
  PLOT_ARGS+=(--gnn-csv "$GNN_CSV")
fi
if [[ -f "$MIP_CSV" ]]; then
  PLOT_ARGS+=(--mip-csv "$MIP_CSV")
fi

"${PLOT_ARGS[@]}"

echo "Finished plotting SqueezeNet area sweep results from $OUTDIR"
