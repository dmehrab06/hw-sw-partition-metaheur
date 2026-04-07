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
  "10"
  "15"
  # "1000"
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
AGGREGATE_MANIFEST="$OUTDIR/${RESULT_TAG}_plot_manifest.csv"

mkdir -p "$OUTDIR"
rm -f "$AGGREGATE_MANIFEST"

DATASET_MANIFESTS=()
for dataset in "${DATASETS[@]}"; do
  dataset_manifest="$OUTDIR/$dataset/${RESULT_TAG}_${dataset}_selected_manifest.csv"
  if [[ -f "$dataset_manifest" ]]; then
    DATASET_MANIFESTS+=("$dataset_manifest")
  else
    echo "Skipping $dataset: missing manifest $dataset_manifest"
  fi
done

if [[ ${#DATASET_MANIFESTS[@]} -eq 0 ]]; then
  echo "No dataset manifests found under $OUTDIR"
  exit 1
fi

"$PYTHON" - <<'PY' "$AGGREGATE_MANIFEST" "${DATASET_MANIFESTS[@]}"
from pathlib import Path
import sys
import pandas as pd

out_path = Path(sys.argv[1])
parts = []
for manifest_path in sys.argv[2:]:
    path = Path(manifest_path)
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    if frame.empty:
        continue
    parts.append(frame)

if not parts:
    raise SystemExit("No non-empty dataset manifests found.")

merged = pd.concat(parts, ignore_index=True).drop_duplicates()
merged.to_csv(out_path, index=False)
print(f"Wrote aggregate manifest to {out_path}")
PY

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
  --manifest "$AGGREGATE_MANIFEST" \
  --output-dir "$OUTDIR" \
  --methods "${PLOT_METHODS[@]}" \
  --datasets "${DATASETS[@]}" \
  --tag "$RESULT_TAG"

echo "Finished plotting large-scale synthetic results from $OUTDIR"
