#!/bin/bash
set -euo pipefail

export HWSW_METHOD_RUNTIME_PROFILE="${HWSW_METHOD_RUNTIME_PROFILE:-balanced}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
source "$ROOT/BatchExperiments2/batch2_common.sh"

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
  "anomaly_detection_tosa"
  "keyword_spotting_tosa"
  "image_classification_tosa"
  "visual_wake_words_tosa"
  "squeeze_net_tosa"
  "rez_net_tosa"
  "mobile_net_tosa"
  "squeezenet_like_1000"
  # "paper_fig3_11node"
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
OUTDIR="$ROOT/BatchExperiments2/dataset_area05"
AGGREGATE_MANIFEST="$OUTDIR/${RESULT_TAG}_plot_manifest.csv"
MIP_PLOT_METRIC="${MIP_PLOT_METRIC:-lp}"
LARGE_SCALE_OUTDIR="${LARGE_SCALE_OUTDIR:-$ROOT/BatchExperiments2/large_scale_area05}"

uses_large_scale_borrowed_results() {
  local dataset="$1"
  [[ "$dataset" == "squeezenet_like_1000" ]]
}

sync_large_scale_plot_inputs() {
  local dataset="$1"
  local dataset_dir="$OUTDIR/$dataset"
  local source_dataset_dir="$LARGE_SCALE_OUTDIR/$dataset"
  local source_manifest="$source_dataset_dir/large_scale_area05_10seed_${dataset}_selected_manifest.csv"
  local target_manifest="$dataset_dir/${RESULT_TAG}_${dataset}_selected_manifest.csv"

  if ! uses_large_scale_borrowed_results "$dataset"; then
    return 0
  fi
  if [[ ! -d "$source_dataset_dir" ]]; then
    return 0
  fi

  mkdir -p "$dataset_dir"
  if [[ -f "$source_manifest" ]]; then
    cp -a "$source_manifest" "$target_manifest"
  fi
  for method in "${METHODS[@]}"; do
    local src_method_dir="$source_dataset_dir/$method"
    local dst_method_dir="$dataset_dir/$method"
    if [[ -d "$src_method_dir" ]]; then
      mkdir -p "$dst_method_dir"
      cp -a "$src_method_dir/." "$dst_method_dir/"
    fi
  done
}

mkdir -p "$OUTDIR"
rm -f "$AGGREGATE_MANIFEST"

for dataset in "${DATASETS[@]}"; do
  sync_large_scale_plot_inputs "$dataset"
done

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
