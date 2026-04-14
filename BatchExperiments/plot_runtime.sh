#!/bin/bash
set -euo pipefail

export HWSW_METHOD_RUNTIME_PROFILE="${HWSW_METHOD_RUNTIME_PROFILE:-arato}"

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
  # "mip"  # Uncomment to add the fixed placeholder MILP runtime bar (default 3600s).
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
  # Comment out any dataset you do not want to plot by default.
  # Use DATASETS_OVERRIDE="dataset_a dataset_b" to replace this list at runtime.
  # "paper_fig3_11node"
  # "anomaly_detection_tosa"
  # "keyword_spotting_tosa"
  # "image_classification_tosa"
  # "visual_wake_words_tosa"
  # "squeeze_net_tosa"
  # "rez_net_tosa"
  "mobile_net_tosa"
  # "squeezenet_like_1000"
)

SEEDS=(42)

if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  read -r -a METHODS <<<"$METHODS_OVERRIDE"
fi
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  read -r -a DATASETS <<<"$DATASETS_OVERRIDE"
fi
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<<"$SEEDS_OVERRIDE"
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No datasets selected. Uncomment entries in DATASETS or use DATASETS_OVERRIDE=\"dataset_a dataset_b\"."
  exit 1
fi

MIP_RUNTIME_PLACEHOLDER_SEC="${MIP_RUNTIME_PLACEHOLDER_SEC:-3600}"

RESULT_TAG="dataset_area05_runtime"
OUTDIR="$ROOT/BatchExperiments/runtime_area05"
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

ROOT_DATASETS=()
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
    if [[ "$method" == "mip" && "$MIP_RUNTIME_PLACEHOLDER_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      PLOT_METHODS+=("$method")
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

  ROOT_DATASETS+=("$dataset")
  "$PYTHON" "$ROOT/tools/plot_runtime_phase_bars.py" \
    --search-root "$DATASET_DIR" \
    --manifest "$DATASET_MANIFEST" \
    --output-dir "$DATASET_DIR" \
    --methods "${PLOT_METHODS[@]}" \
    --datasets "$dataset" \
    --mip-placeholder-sec "$MIP_RUNTIME_PLACEHOLDER_SEC" \
    --tag "${RESULT_TAG}_${dataset}"
done

if [[ ${#ROOT_DATASETS[@]} -eq 0 ]]; then
  echo "No datasets had runtime result folders under $OUTDIR"
  exit 1
fi

ROOT_PLOT_METHODS=()
for method in "${ALL_METHODS_ORDER[@]}"; do
  if [[ ! " ${METHODS[*]} " =~ " ${method} " ]]; then
    continue
  fi
  if [[ "$method" == "mip" && "$MIP_RUNTIME_PLACEHOLDER_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    ROOT_PLOT_METHODS+=("$method")
    continue
  fi
  for dataset in "${ROOT_DATASETS[@]}"; do
    if [[ -d "$OUTDIR/$dataset/$method" ]]; then
      ROOT_PLOT_METHODS+=("$method")
      break
    fi
  done
done

"$PYTHON" "$ROOT/tools/plot_runtime_phase_bars.py" \
  --search-root "$OUTDIR" \
  --manifest "$AGGREGATE_MANIFEST" \
  --output-dir "$OUTDIR" \
  --methods "${ROOT_PLOT_METHODS[@]}" \
  --datasets "${ROOT_DATASETS[@]}" \
  --mip-placeholder-sec "$MIP_RUNTIME_PLACEHOLDER_SEC" \
  --tag "$RESULT_TAG"

echo "Finished plotting runtime breakdowns from method folders in $OUTDIR"
