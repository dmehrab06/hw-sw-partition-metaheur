#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"

# Canonical display / plotting order.
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

# Comment out any method you do not want to run.
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

# Default 8-dataset batch: 7 real SODA graphs + 1 paper sanity-check graph.
# Comment out any dataset you do not want to include.
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

# Edit this array to control the number of seeds.
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

PROFILE="full"
AREA="0.5"
RESULT_TAG="dataset_area05_10seed"
OUTDIR="$ROOT/BatchExperiments/dataset_area05"
PLOT_AFTER_RUN="${PLOT_AFTER_RUN:-0}"
TMPDIR_ROOT="$(mktemp -d "$ROOT/BatchExperiments/.dataset_area05_tmp.XXXXXX")"
trap 'rm -rf "$TMPDIR_ROOT"' EXIT
CONFIG_ROOT="$TMPDIR_ROOT/task_graph_topology_config"

# MIP runtime controls. Edit if needed for larger graphs.
FAST_MIP="${FAST_MIP:-1}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-1800}"
TIMEOUT_KILL_AFTER_SEC="${TIMEOUT_KILL_AFTER_SEC:-30}"

CONFIG_PROFILE_ROOT="$CONFIG_ROOT/$PROFILE"
MANIFEST="$CONFIG_PROFILE_ROOT/graph_suite_area05/manifest.csv"
ROOT_MANIFEST="$OUTDIR/${RESULT_TAG}_selected_manifest.csv"
ROOT_GNN_CSV="$OUTDIR/${RESULT_TAG}-result-summary-soda-graphs-config.csv"
ROOT_MIP_CSV="$OUTDIR/mip_${RESULT_TAG}-result-summary-soda-graphs-config.csv"

mkdir -p "$OUTDIR"
rm -f "$ROOT_GNN_CSV" "$ROOT_MIP_CSV"

# Full profile defaults to 10 seeds, but we pass them explicitly here for clarity.
"$PYTHON" "$ROOT/tools/generate_task_graph_topology_configs.py" \
  --profile "$PROFILE" \
  --config-root "$CONFIG_ROOT" \
  --area "$AREA" \
  --seeds "${SEEDS[@]}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

CFG_DIR="$TMPDIR_ROOT/configs"
"$PYTHON" "$ROOT/tools/select_configs_from_manifest.py" \
  --manifest "$MANIFEST" \
  --out-dir "$CFG_DIR" \
  --graph-names "${DATASETS[@]}" \
  --seeds "${SEEDS[@]}" \
  --areas "$AREA"
"$PYTHON" - <<'PY' "$CFG_DIR/selected_manifest.csv" "$ROOT_MANIFEST"
from pathlib import Path
import sys
import pandas as pd

new_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

frames = []
if out_path.exists():
    frames.append(pd.read_csv(out_path))
frames.append(pd.read_csv(new_path))

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
print(f"Wrote cumulative root manifest to {out_path}")
PY

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="$OUTDIR/$dataset"
  mkdir -p "$DATASET_DIR"

  DATASET_CFG_DIR="$TMPDIR_ROOT/configs_${dataset}"
  "$PYTHON" "$ROOT/tools/select_configs_from_manifest.py" \
    --manifest "$ROOT_MANIFEST" \
    --out-dir "$DATASET_CFG_DIR" \
    --graph-names "$dataset" \
    --seeds "${SEEDS[@]}" \
    --areas "$AREA"

  DATASET_MANIFEST="$DATASET_DIR/${RESULT_TAG}_${dataset}_selected_manifest.csv"
  "$PYTHON" - <<'PY' "$DATASET_CFG_DIR/selected_manifest.csv" "$DATASET_MANIFEST"
from pathlib import Path
import sys
import pandas as pd

new_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

frames = []
if out_path.exists():
    frames.append(pd.read_csv(out_path))
frames.append(pd.read_csv(new_path))

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

  for method in "${METHODS[@]}"; do
    METHOD_DIR="$DATASET_DIR/$method"
    METHOD_CFG_DIR="$TMPDIR_ROOT/configs_${dataset}_${method}"
    METHOD_PREFIX="${RESULT_TAG}_${dataset}_${method}"
    METHOD_MANIFEST="$METHOD_DIR/${METHOD_PREFIX}_selected_manifest.csv"

    rm -rf "$METHOD_CFG_DIR"
    mkdir -p "$METHOD_DIR" "$METHOD_CFG_DIR"

    "$PYTHON" - <<'PY' "$DATASET_CFG_DIR/selected_manifest.csv" "$METHOD_MANIFEST"
from pathlib import Path
import sys
import pandas as pd

new_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

frames = []
if out_path.exists():
    frames.append(pd.read_csv(out_path))
frames.append(pd.read_csv(new_path))

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
print(f"Wrote cumulative method manifest to {out_path}")
PY
    cp -a "$DATASET_CFG_DIR"/. "$METHOD_CFG_DIR"/
    rm -f "$METHOD_CFG_DIR/selected_manifest.csv"

    for cfg in "$METHOD_CFG_DIR"/*.yaml; do
      src_cfg="$(readlink -f "$cfg")"
      rm -f "$cfg"
      "$PYTHON" - <<'PY' "$src_cfg" "$cfg" "$METHOD_DIR" "$METHOD_PREFIX"
from omegaconf import OmegaConf
import sys

src, dst, out_dir, result_prefix = sys.argv[1:]
cfg = OmegaConf.load(src)
cfg["output-dir"] = out_dir
cfg["solution-dir"] = f"{out_dir}/partitions"
cfg["result-file-prefix"] = result_prefix
vis = dict(cfg.get("visualization", {}))
vis["enabled"] = False
cfg["visualization"] = vis
OmegaConf.save(config=cfg, f=dst)
PY
    done

    if [[ "$method" == "mip" ]]; then
      CONFIG_GLOB="$METHOD_CFG_DIR/*.yaml" \
      OUTDIR="$METHOD_DIR" \
      FAST_MIP="$FAST_MIP" \
      RUN_TIMEOUT_SEC="$RUN_TIMEOUT_SEC" \
      TIMEOUT_KILL_AFTER_SEC="$TIMEOUT_KILL_AFTER_SEC" \
      PYTHON="$PYTHON" \
      "$ROOT/run_all_mip_configs.sh"
    else
      CONFIG_GLOB="$METHOD_CFG_DIR/*.yaml" \
      OUTDIR="$METHOD_DIR" \
      HWSW_METHODS="$method" \
      HWSW_CSV_DIR="$METHOD_DIR" \
      HWSW_RESULT_PREFIX="$METHOD_PREFIX" \
      PYTHON="$PYTHON" \
      "$ROOT/run_all_gnn_configs.sh"
    fi
  done

done

if [[ "$PLOT_AFTER_RUN" =~ ^(1|true|yes|on)$ ]]; then
  env \
    PYTHON="$PYTHON" \
    METHODS_OVERRIDE="$(IFS=' '; echo "${METHODS[*]}")" \
    DATASETS_OVERRIDE="$(IFS=' '; echo "${DATASETS[*]}")" \
    "$ROOT/BatchExperiments/plot_dataset_area05_10seed.sh"
fi

echo "Finished dataset-area batch. Outputs are in $OUTDIR"
