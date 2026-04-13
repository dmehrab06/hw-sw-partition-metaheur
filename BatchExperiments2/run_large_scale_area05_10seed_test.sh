#!/bin/bash
set -euo pipefail

export HWSW_METHOD_RUNTIME_PROFILE="${HWSW_METHOD_RUNTIME_PROFILE:-balanced}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
source "$ROOT/BatchExperiments2/batch2_common.sh"

ALL_METHODS_ORDER=(
  # "mip"
  "diff_gnn"
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
  # "diff_gnn"
  # "diff_gnn_order"
  "gl25"
  # "gcps"
  # "esa"
  # "pso"
  # "dbpso"
  # "clpso"
  # "ccpso"
  # "shade"
  # "jade"
  # "random"
  # "greedy"
)

GRAPH_SIZES=(
  # "10"
  # "15"
  "1000"
  # "10000"
)

# Extra non-synthetic datasets that can be run through the same batch path.
# Comment out entries here if you want a synthetic-only large-scale sweep.
EXTRA_DATASETS=(
  "paper_fig3_11node"
)

# SEEDS=(42 43 44 45 46 47 48 49 50 51)
SEEDS=(42)

if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  read -r -a METHODS <<<"$METHODS_OVERRIDE"
fi
if [[ -n "${GRAPH_SIZES_OVERRIDE:-}" ]]; then
  read -r -a GRAPH_SIZES <<<"$GRAPH_SIZES_OVERRIDE"
fi
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<<"$SEEDS_OVERRIDE"
fi

DATASETS=()
for size in "${GRAPH_SIZES[@]}"; do
  DATASETS+=("squeezenet_like_${size}")
done
DATASETS+=("${EXTRA_DATASETS[@]}")
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  read -r -a DATASETS <<<"$DATASETS_OVERRIDE"
fi
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No datasets selected. Enable at least one GRAPH_SIZES entry, set an EXTRA_DATASETS entry, or pass DATASETS_OVERRIDE."
  exit 1
fi

PROFILE="full"
AREA="0.5"
RESULT_TAG="large_scale_area05_10seed"
OUTDIR="$ROOT/BatchExperiments2/large_scale_area05"
CONFIG_ROOT="${CONFIG_ROOT:-$OUTDIR/config_cache}"
FORCE_REGENERATE_CONFIGS="${FORCE_REGENERATE_CONFIGS:-0}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d-%H%M%S')}"
SELECTED_CONFIG_ROOT="${SELECTED_CONFIG_ROOT:-$OUTDIR/selected_configs}"

DEFAULT_FULL_CONFIG_SEEDS=(42 43 44 45 46 47 48 49 50 51)
DEFAULT_PILOT_CONFIG_SEEDS=(42 43 44)
if [[ "$PROFILE" == "full" ]]; then
  CONFIG_CACHE_SEEDS=("${DEFAULT_FULL_CONFIG_SEEDS[@]}")
else
  CONFIG_CACHE_SEEDS=("${DEFAULT_PILOT_CONFIG_SEEDS[@]}")
fi
if [[ -n "${CONFIG_SEEDS_OVERRIDE:-}" ]]; then
  read -r -a CONFIG_CACHE_SEEDS <<<"$CONFIG_SEEDS_OVERRIDE"
fi

FAST_MIP="${FAST_MIP:-1}"
MIP_SOLVER_TOOL="${MIP_SOLVER_TOOL:-cvxpy-scip}"
MIP_SOLVE_MODE="${MIP_SOLVE_MODE:-exact}"
MIP_SW_CONSTRAINT_MODE="${MIP_SW_CONSTRAINT_MODE:-pairwise_topo}"
MIP_USE_REDUCED_SW="${MIP_USE_REDUCED_SW:-false}"
MIP_ACCEPT_NONOPTIMAL="${MIP_ACCEPT_NONOPTIMAL:-false}"
MIP_VERBOSE="${MIP_VERBOSE:-true}"
MIP_TIME_LIMIT_SEC="${MIP_TIME_LIMIT_SEC:-300}"
MIP_GAP="${MIP_GAP:-0.05}"
MIP_NODE_LIMIT="${MIP_NODE_LIMIT:-100000}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-0}"
TIMEOUT_KILL_AFTER_SEC="${TIMEOUT_KILL_AFTER_SEC:-30}"

CONFIG_PROFILE_ROOT="$CONFIG_ROOT/$PROFILE"
MANIFEST="$CONFIG_PROFILE_ROOT/graph_suite_area05/manifest.csv"
ROOT_MANIFEST="$OUTDIR/${RESULT_TAG}_selected_manifest.csv"

print_banner() {
  local message="$1"
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message"
}

join_by_comma() {
  local IFS=', '
  echo "$*"
}

generate_missing_synthetic_graphs() {
  local missing_sizes=()
  for size in "${GRAPH_SIZES[@]}"; do
    local dot_path="$ROOT/inputs/task_graph_topology/synthetic/squeezenet_like_${size}.dot"
    if [[ ! -f "$dot_path" ]]; then
      missing_sizes+=("$size")
    fi
  done

  if [[ ${#missing_sizes[@]} -eq 0 ]]; then
    return 0
  fi

  print_banner "Generating missing synthetic graphs"
  "$PYTHON" "$ROOT/tools/generate_squeezenet_like_synthetic_graph.py" \
    --nodes "${missing_sizes[@]}" \
    --seed 42 \
    --k 0.1 \
    --l 0.5 \
    --mu 1.0 \
    --area-constraint "$AREA"
}

generate_stable_topology_configs() {
  "$PYTHON" "$ROOT/tools/generate_task_graph_topology_configs.py" \
    --profile "$PROFILE" \
    --config-root "$CONFIG_ROOT" \
    --area "$AREA" \
    --seeds "${CONFIG_CACHE_SEEDS[@]}" \
    --strip-method-configs
}

count_manifest_rows() {
  "$PYTHON" - <<'PY' "$1"
from pathlib import Path
import sys
import pandas as pd

path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    print(0)
else:
    print(len(pd.read_csv(path)))
PY
}

count_missing_config_paths() {
  "$PYTHON" - <<'PY' "$1" "$ROOT"
from pathlib import Path
import sys
import pandas as pd

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()

if not manifest_path.exists() or manifest_path.stat().st_size == 0:
    print(0)
    raise SystemExit(0)

df = pd.read_csv(manifest_path)
missing = 0
for value in df.get("config_path", []):
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        missing += 1
print(missing)
PY
}

sanitize_large_scale_configs() {
  "$PYTHON" - <<'PY' "$CONFIG_PROFILE_ROOT/graph_suite_area05" "${DATASETS[@]}"
from pathlib import Path
import sys
import yaml

suite_root = Path(sys.argv[1])
datasets = sys.argv[2:]
updated = 0

for dataset in datasets:
    dataset_dir = suite_root / dataset
    if not dataset_dir.exists():
        continue
    for path in sorted(dataset_dir.glob("*.yaml")):
        with path.open() as handle:
            cfg = yaml.safe_load(handle) or {}

        changed = False
        if "diffgnn" in cfg:
            cfg.pop("diffgnn", None)
            changed = True
        if "diffgnn_order" in cfg:
            cfg.pop("diffgnn_order", None)
            changed = True
        if not changed:
            continue

        with path.open("w") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        updated += 1

print(f"Sanitized {updated} large-scale configs under {suite_root}")
PY
}

mkdir -p "$OUTDIR"
mkdir -p "$SELECTED_CONFIG_ROOT"
bootstrap_batchexperiments2_family "large_scale_area05" "$ROOT" "$PYTHON"

print_banner "Starting large-scale synthetic batch run"
echo "  Profile        : $PROFILE"
echo "  Area constraint: $AREA"
echo "  Output root    : $OUTDIR"
echo "  Config root    : $CONFIG_ROOT"
echo "  Selected cfg dir: $SELECTED_CONFIG_ROOT"
echo "  Run tag        : $RUN_TAG"
echo "  Graph sizes (${#GRAPH_SIZES[@]}): $(join_by_comma "${GRAPH_SIZES[@]}")"
echo "  Datasets (${#DATASETS[@]}): $(join_by_comma "${DATASETS[@]}")"
echo "  Methods  (${#METHODS[@]}): $(join_by_comma "${METHODS[@]}")"
echo "  Seeds    (${#SEEDS[@]}): $(join_by_comma "${SEEDS[@]}")"
echo "  Config cache seeds: $(join_by_comma "${CONFIG_CACHE_SEEDS[@]}")"
echo "  MIP tlimit    : ${MIP_TIME_LIMIT_SEC}s"
if [[ "$RUN_TIMEOUT_SEC" =~ ^[0-9]+$ ]] && (( RUN_TIMEOUT_SEC > 0 )); then
  echo "  MIP watchdog  : ${RUN_TIMEOUT_SEC}s (kill-after ${TIMEOUT_KILL_AFTER_SEC}s)"
else
  echo "  MIP watchdog  : disabled"
fi
echo "  MIP gap       : $MIP_GAP"
echo "  MIP node limit: $MIP_NODE_LIMIT"
echo "  Plot step      : disabled in this script; run plot_large_scale_area05_10seed.sh separately"

generate_missing_synthetic_graphs

if [[ ! -f "$MANIFEST" || "$FORCE_REGENERATE_CONFIGS" =~ ^(1|true|yes|on)$ ]]; then
  print_banner "Generating stable topology configurations"
  generate_stable_topology_configs
else
  print_banner "Reusing existing stable topology configurations"
  echo "  Using manifest: $MANIFEST"
fi

sanitize_large_scale_configs

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

CFG_DIR="$SELECTED_CONFIG_ROOT/root"
print_banner "Selecting requested synthetic dataset/seed subset from manifest"
"$PYTHON" "$ROOT/tools/select_configs_from_manifest.py" \
  --manifest "$MANIFEST" \
  --out-dir "$CFG_DIR" \
  --graph-names "${DATASETS[@]}" \
  --seeds "${SEEDS[@]}" \
  --areas "$AREA"
expected_root_configs=$(( ${#DATASETS[@]} * ${#SEEDS[@]} ))
actual_root_configs="$(count_manifest_rows "$CFG_DIR/selected_manifest.csv")"
missing_root_configs="$(count_missing_config_paths "$CFG_DIR/selected_manifest.csv")"
if [[ "$actual_root_configs" -ne "$expected_root_configs" || "$missing_root_configs" -gt 0 ]]; then
  print_banner "Stable config cache is missing requested rows; regenerating once"
  echo "  Expected rows: $expected_root_configs"
  echo "  Found rows   : $actual_root_configs"
  echo "  Missing cfgs : $missing_root_configs"
  generate_stable_topology_configs
  sanitize_large_scale_configs
  "$PYTHON" "$ROOT/tools/select_configs_from_manifest.py" \
    --manifest "$MANIFEST" \
    --out-dir "$CFG_DIR" \
    --graph-names "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --areas "$AREA"
  actual_root_configs="$(count_manifest_rows "$CFG_DIR/selected_manifest.csv")"
  missing_root_configs="$(count_missing_config_paths "$CFG_DIR/selected_manifest.csv")"
fi
if [[ "$actual_root_configs" -ne "$expected_root_configs" || "$missing_root_configs" -gt 0 ]]; then
  echo "Config selection mismatch after regeneration: expected $expected_root_configs rows, found $actual_root_configs rows, missing $missing_root_configs config files in $CFG_DIR/selected_manifest.csv"
  exit 1
fi
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

dataset_idx=0
total_datasets=${#DATASETS[@]}
total_methods=${#METHODS[@]}
batch_start_sec=$SECONDS

for dataset in "${DATASETS[@]}"; do
  dataset_idx=$((dataset_idx + 1))
  print_banner "Dataset [$dataset_idx/$total_datasets]: $dataset"
  dataset_start_sec=$SECONDS
  DATASET_DIR="$OUTDIR/$dataset"
  mkdir -p "$DATASET_DIR"

  DATASET_CFG_DIR="$SELECTED_CONFIG_ROOT/$dataset/_dataset"
  "$PYTHON" "$ROOT/tools/select_configs_from_manifest.py" \
    --manifest "$ROOT_MANIFEST" \
    --out-dir "$DATASET_CFG_DIR" \
    --graph-names "$dataset" \
    --seeds "${SEEDS[@]}" \
    --areas "$AREA"
  expected_dataset_configs=${#SEEDS[@]}
  actual_dataset_configs="$(count_manifest_rows "$DATASET_CFG_DIR/selected_manifest.csv")"
  missing_dataset_configs="$(count_missing_config_paths "$DATASET_CFG_DIR/selected_manifest.csv")"
  if [[ "$actual_dataset_configs" -ne "$expected_dataset_configs" || "$missing_dataset_configs" -gt 0 ]]; then
    echo "Dataset config mismatch for $dataset: expected $expected_dataset_configs rows, found $actual_dataset_configs rows, missing $missing_dataset_configs config files in $DATASET_CFG_DIR/selected_manifest.csv"
    exit 1
  fi

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

  method_idx=0
  for method in "${METHODS[@]}"; do
    method_idx=$((method_idx + 1))
    print_banner "Dataset [$dataset_idx/$total_datasets] Method [$method_idx/$total_methods]: $dataset / $method"
    method_start_sec=$SECONDS
    METHOD_DIR="$DATASET_DIR/$method"
    METHOD_PREFIX="${RESULT_TAG}_${dataset}_${method}"
    METHOD_MANIFEST="$METHOD_DIR/${METHOD_PREFIX}_selected_manifest.csv"

    mkdir -p "$METHOD_DIR"

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

    if [[ "$method" == "mip" ]]; then
      echo "  Launching MIP batch with shared dataset configs from $DATASET_CFG_DIR"
      CONFIG_GLOB="$DATASET_CFG_DIR/*.yaml" \
      OUTDIR="$METHOD_DIR" \
      SOLVER_TOOL="$MIP_SOLVER_TOOL" \
      FAST_MIP="$FAST_MIP" \
      MIP_SOLVE_MODE="$MIP_SOLVE_MODE" \
      MIP_SW_CONSTRAINT_MODE="$MIP_SW_CONSTRAINT_MODE" \
      MIP_USE_REDUCED_SW="$MIP_USE_REDUCED_SW" \
      MIP_ACCEPT_NONOPTIMAL="$MIP_ACCEPT_NONOPTIMAL" \
      MIP_VERBOSE="$MIP_VERBOSE" \
      MIP_TIME_LIMIT_SEC="$MIP_TIME_LIMIT_SEC" \
      MIP_GAP="$MIP_GAP" \
      MIP_NODE_LIMIT="$MIP_NODE_LIMIT" \
      RUN_TIMEOUT_SEC="$RUN_TIMEOUT_SEC" \
      TIMEOUT_KILL_AFTER_SEC="$TIMEOUT_KILL_AFTER_SEC" \
      HWSW_OUTPUT_DIR="$METHOD_DIR" \
      HWSW_SOLUTION_DIR="$METHOD_DIR/partitions" \
      HWSW_RESULT_PREFIX="$METHOD_PREFIX" \
      HWSW_RUN_TAG="$RUN_TAG" \
      PYTHON="$PYTHON" \
      "$ROOT/run_all_mip_configs.sh"
    else
      echo "  Launching GNN/metaheuristic batch for method $method with shared dataset configs from $DATASET_CFG_DIR"
      CONFIG_GLOB="$DATASET_CFG_DIR/*.yaml" \
      OUTDIR="$METHOD_DIR" \
      HWSW_METHODS="$method" \
      HWSW_OUTPUT_DIR="$METHOD_DIR" \
      HWSW_SOLUTION_DIR="$METHOD_DIR/partitions" \
      HWSW_CSV_DIR="$METHOD_DIR" \
      HWSW_RESULT_PREFIX="$METHOD_PREFIX" \
      HWSW_RUN_TAG="$RUN_TAG" \
      PYTHON="$PYTHON" \
      "$ROOT/run_all_gnn_configs.sh"
    fi
    method_elapsed_sec=$((SECONDS - method_start_sec))
    print_banner "Completed method: $dataset / $method (${method_elapsed_sec}s)"
  done

  dataset_elapsed_sec=$((SECONDS - dataset_start_sec))
  print_banner "Completed dataset: $dataset (${dataset_elapsed_sec}s)"
done

batch_elapsed_sec=$((SECONDS - batch_start_sec))
print_banner "Finished large-scale synthetic batch run (${batch_elapsed_sec}s)"
echo "Results are under $OUTDIR"
echo "Run plotting separately with:"
echo "  $ROOT/BatchExperiments2/plot_large_scale_area05_10seed.sh"
