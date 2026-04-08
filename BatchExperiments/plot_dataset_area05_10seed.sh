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

DATASETS=(
  "paper_fig3_11node"
  "mobile_net_tosa"
  "rez_net_tosa"
  "squeeze_net_tosa"
  "anomaly_detection_tosa"
  "image_classification_tosa"
  "keyword_spotting_tosa"
  "visual_wake_words_tosa"
)

# Keep plotting aligned with run_dataset_area05_10seed.sh's canonical 10-seed batch.
SEEDS=(42 43 44 45 46 47 48 49 50 51)

if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  read -r -a METHODS <<<"$METHODS_OVERRIDE"
fi
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  read -r -a DATASETS <<<"$DATASETS_OVERRIDE"
fi
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<<"$SEEDS_OVERRIDE"
fi

RESULT_TAG="dataset_area05_10seed"
OUTDIR="$ROOT/BatchExperiments/dataset_area05"
AGGREGATE_MANIFEST="$OUTDIR/${RESULT_TAG}_plot_manifest.csv"
MIP_PLOT_METRIC="${MIP_PLOT_METRIC:-lssp}"

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

"$PYTHON" - <<'PY' "$AGGREGATE_MANIFEST" "${#SEEDS[@]}" "${SEEDS[@]}" "${DATASET_MANIFESTS[@]}"
from pathlib import Path
import sys
import pandas as pd

out_path = Path(sys.argv[1])
num_seeds = int(sys.argv[2])
seed_values = {int(value) for value in sys.argv[3:3 + num_seeds]}
manifest_args = sys.argv[3 + num_seeds:]
parts = []
for manifest_path in manifest_args:
    path = Path(manifest_path)
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    if frame.empty:
        continue
    if "seed" in frame:
        frame["seed"] = frame["seed"].astype(int)
        frame = frame[frame["seed"].isin(seed_values)]
    if frame.empty:
        continue
    parts.append(frame)

if not parts:
    raise SystemExit("No non-empty dataset manifests found.")

merged = pd.concat(parts, ignore_index=True).drop_duplicates()
merged.to_csv(out_path, index=False)
print(f"Wrote aggregate manifest to {out_path}")
PY

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="$OUTDIR/$dataset"
  DATASET_MANIFEST="$DATASET_DIR/${RESULT_TAG}_${dataset}_selected_manifest.csv"

  mapfile -t DATASET_MANIFEST_CANDIDATES < <(find "$DATASET_DIR" -maxdepth 2 -type f -name '*selected_manifest.csv' | sort || true)
  if [[ ${#DATASET_MANIFEST_CANDIDATES[@]} -eq 0 ]]; then
    continue
  fi

  "$PYTHON" - <<'PY' "$DATASET_MANIFEST" "${#SEEDS[@]}" "${SEEDS[@]}" "${DATASET_MANIFEST_CANDIDATES[@]}"
from pathlib import Path
import sys
import pandas as pd

out_path = Path(sys.argv[1])
num_seeds = int(sys.argv[2])
seed_values = {int(value) for value in sys.argv[3:3 + num_seeds]}
manifest_args = sys.argv[3 + num_seeds:]
frames = []
for manifest_path in manifest_args:
    path = Path(manifest_path)
    if not path.exists():
        continue
    frame = pd.read_csv(path)
    if frame.empty:
        continue
    if "seed" in frame:
        frame["seed"] = frame["seed"].astype(int)
        frame = frame[frame["seed"].isin(seed_values)]
    if frame.empty:
        continue
    frames.append(frame)

if not frames:
    raise SystemExit("No non-empty dataset manifests found.")

merged = pd.concat(frames, ignore_index=True)
subset = [
    "graph_name",
    "seed",
    "area_constraint",
    "hw_scale_factor",
    "hw_scale_variance",
    "comm_scale_factor",
]
merged = merged.drop_duplicates(subset=subset, keep="last")
merged.to_csv(out_path, index=False)
print(f"Wrote cumulative dataset manifest to {out_path}")
PY

  PLOT_METHODS=()
  for method in "${ALL_METHODS_ORDER[@]}"; do
    if [[ ! " ${METHODS[*]} " =~ " ${method} " ]]; then
      continue
    fi
    if [[ -d "$DATASET_DIR/$method" ]]; then
      PLOT_METHODS+=("$method")
    fi
  done

  if [[ ${#PLOT_METHODS[@]} -eq 0 ]]; then
    echo "Skipping $dataset: no method folders selected"
    continue
  fi

  "$PYTHON" "$ROOT/tools/plot_batch_method_bars.py" \
    --search-root "$DATASET_DIR" \
    --manifest "$DATASET_MANIFEST" \
    --output-dir "$DATASET_DIR" \
    --mode datasets \
    --methods "${PLOT_METHODS[@]}" \
    --datasets "$dataset" \
    --mip-metric "$MIP_PLOT_METRIC" \
    --tag "${RESULT_TAG}_${dataset}"
done

ROOT_PLOT_METHODS=()
for method in "${ALL_METHODS_ORDER[@]}"; do
  if [[ ! " ${METHODS[*]} " =~ " ${method} " ]]; then
    continue
  fi
  for dataset in "${DATASETS[@]}"; do
    if [[ -d "$OUTDIR/$dataset/$method" ]]; then
      ROOT_PLOT_METHODS+=("$method")
      break
    fi
  done
done

if [[ ${#ROOT_PLOT_METHODS[@]} -gt 0 ]]; then
  "$PYTHON" "$ROOT/tools/plot_batch_method_bars.py" \
    --search-root "$OUTDIR" \
    --manifest "$AGGREGATE_MANIFEST" \
    --output-dir "$OUTDIR" \
    --mode datasets \
    --methods "${ROOT_PLOT_METHODS[@]}" \
    --datasets "${DATASETS[@]}" \
    --mip-metric "$MIP_PLOT_METRIC" \
    --tag "$RESULT_TAG"
fi

echo "Finished plotting dataset-area results from existing method folders in $OUTDIR"
