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
  # "mip"
  "diff_gnn_order"
  # "gl25"
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

# Default 8-dataset batch: 7 real SODA graphs + 1 paper sanity-check graph.
# Comment out any dataset you do not want to include.
DATASETS=(
  # "paper_fig3_11node"
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
# SEEDS=(42)

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
CONFIG_ROOT="${CONFIG_ROOT:-$ROOT/BatchExperiments/dataset_area05_configs}"
FORCE_REGENERATE_CONFIGS="${FORCE_REGENERATE_CONFIGS:-0}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d-%H%M%S')}"
SELECTED_CONFIG_ROOT="${SELECTED_CONFIG_ROOT:-}"
SELECTED_CONFIG_ROOT_WAS_TEMP=0
if [[ -z "$SELECTED_CONFIG_ROOT" ]]; then
  SELECTED_CONFIG_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/${RESULT_TAG}_configs.XXXXXX")"
  SELECTED_CONFIG_ROOT_WAS_TEMP=1
fi

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

# MIP runtime controls for the dataset batch.
# These defaults are intentionally higher than the fast smoke-test settings.
FAST_MIP="${FAST_MIP:-1}"
MIP_TIME_LIMIT_SEC="${MIP_TIME_LIMIT_SEC:-300}"
MIP_GAP="${MIP_GAP:-0.05}"
MIP_NODE_LIMIT="${MIP_NODE_LIMIT:-100000}"
MIP_TIMEOUT_BUFFER_SEC="${MIP_TIMEOUT_BUFFER_SEC:-180}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-$((MIP_TIME_LIMIT_SEC + MIP_TIMEOUT_BUFFER_SEC))}"
TIMEOUT_KILL_AFTER_SEC="${TIMEOUT_KILL_AFTER_SEC:-30}"

CONFIG_PROFILE_ROOT="$CONFIG_ROOT/$PROFILE"
MANIFEST="$CONFIG_PROFILE_ROOT/graph_suite_area05/manifest.csv"
ROOT_MANIFEST="$OUTDIR/${RESULT_TAG}_selected_manifest.csv"
ROOT_GNN_CSV="$OUTDIR/${RESULT_TAG}-result-summary-soda-graphs-config.csv"
ROOT_MIP_CSV="$OUTDIR/mip_${RESULT_TAG}-result-summary-soda-graphs-config.csv"
CPU_COUNT_OVERRIDE="${HWSW_CPU_COUNT_OVERRIDE:-${CPU_COUNT_OVERRIDE:-}}"
CPU_COUNT_SOURCE="auto"
if [[ -n "$CPU_COUNT_OVERRIDE" ]]; then
  CPU_COUNT="$CPU_COUNT_OVERRIDE"
  CPU_COUNT_SOURCE="override"
else
  CPU_COUNT="$(command -v nproc >/dev/null 2>&1 && nproc || getconf _NPROCESSORS_ONLN || echo 1)"
fi
PARALLEL_DATASET_METHODS="${HWSW_PARALLEL_DATASET_METHODS:-0}"
REQUESTED_INNER_CONFIG_JOBS="${HWSW_MAX_PARALLEL_CONFIGS:-${MAX_PARALLEL_CONFIGS:-}}"
if [[ -n "$REQUESTED_INNER_CONFIG_JOBS" ]]; then
  GROUP_CONFIG_PARALLEL_JOBS="$REQUESTED_INNER_CONFIG_JOBS"
elif [[ "$PARALLEL_DATASET_METHODS" =~ ^(1|true|yes|on)$ ]]; then
  GROUP_CONFIG_PARALLEL_JOBS=1
else
  GROUP_CONFIG_PARALLEL_JOBS=1
fi
MAX_PARALLEL_DATASET_METHODS="${HWSW_MAX_PARALLEL_DATASET_METHODS:-0}"
if [[ "$PARALLEL_DATASET_METHODS" =~ ^(1|true|yes|on)$ ]]; then
  if ! [[ "$GROUP_CONFIG_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || (( GROUP_CONFIG_PARALLEL_JOBS < 1 )); then
    GROUP_CONFIG_PARALLEL_JOBS=1
  fi
  if ! [[ "$MAX_PARALLEL_DATASET_METHODS" =~ ^[0-9]+$ ]] || (( MAX_PARALLEL_DATASET_METHODS <= 0 )); then
    MAX_PARALLEL_DATASET_METHODS=$(( CPU_COUNT / GROUP_CONFIG_PARALLEL_JOBS ))
    if (( MAX_PARALLEL_DATASET_METHODS < 1 )); then
      MAX_PARALLEL_DATASET_METHODS=1
    fi
  fi
else
  MAX_PARALLEL_DATASET_METHODS=1
fi

print_banner() {
  local message="$1"
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message"
}

join_by_comma() {
  local IFS=', '
  echo "$*"
}

cleanup() {
  if (( ${#ACTIVE_GROUP_PIDS[@]} > 0 )); then
    for pid in "${ACTIVE_GROUP_PIDS[@]}"; do
      kill "$pid" >/dev/null 2>&1 || true
    done
  fi
  if [[ "$SELECTED_CONFIG_ROOT_WAS_TEMP" == "1" && -n "$SELECTED_CONFIG_ROOT" && -d "$SELECTED_CONFIG_ROOT" ]]; then
    rm -rf "$SELECTED_CONFIG_ROOT"
  fi
}
trap cleanup EXIT

ACTIVE_GROUP_PIDS=()
ACTIVE_GROUP_LABELS=()
ACTIVE_GROUP_LOGS=()
FAILED_GROUPS=0

remove_active_group_at_index() {
  local idx="$1"
  unset 'ACTIVE_GROUP_PIDS[idx]'
  unset 'ACTIVE_GROUP_LABELS[idx]'
  unset 'ACTIVE_GROUP_LOGS[idx]'
  ACTIVE_GROUP_PIDS=("${ACTIVE_GROUP_PIDS[@]}")
  ACTIVE_GROUP_LABELS=("${ACTIVE_GROUP_LABELS[@]}")
  ACTIVE_GROUP_LOGS=("${ACTIVE_GROUP_LOGS[@]}")
}

reap_finished_groups() {
  local idx pid rc label log_path
  for (( idx=${#ACTIVE_GROUP_PIDS[@]}-1; idx>=0; idx-- )); do
    pid="${ACTIVE_GROUP_PIDS[idx]}"
    if kill -0 "$pid" >/dev/null 2>&1; then
      continue
    fi
    label="${ACTIVE_GROUP_LABELS[idx]}"
    log_path="${ACTIVE_GROUP_LOGS[idx]}"
    if wait "$pid"; then
      print_banner "Completed group: $label"
      echo "  Group log: $log_path"
    else
      rc=$?
      FAILED_GROUPS=$((FAILED_GROUPS + 1))
      print_banner "Group failed: $label (exit=$rc)"
      echo "  Group log: $log_path"
    fi
    remove_active_group_at_index "$idx"
  done
}

wait_for_group_slot() {
  while (( ${#ACTIVE_GROUP_PIDS[@]} >= MAX_PARALLEL_DATASET_METHODS )); do
    sleep 1
    reap_finished_groups
  done
}

wait_for_all_groups() {
  while (( ${#ACTIVE_GROUP_PIDS[@]} > 0 )); do
    sleep 1
    reap_finished_groups
  done
}

launch_dataset_method_group() {
  local dataset="$1"
  local method="$2"
  local dataset_cfg_dir="$3"
  local method_dir="$4"
  local method_prefix="$5"
  local group_label="${dataset} / ${method}"
  local group_log="$method_dir/${method_prefix}__batch.log"

  wait_for_group_slot

  if [[ "$method" == "mip" ]]; then
    (
      CONFIG_GLOB="$dataset_cfg_dir/*.yaml" \
      OUTDIR="$method_dir" \
      FAST_MIP="$FAST_MIP" \
      MIP_TIME_LIMIT_SEC="$MIP_TIME_LIMIT_SEC" \
      MIP_GAP="$MIP_GAP" \
      MIP_NODE_LIMIT="$MIP_NODE_LIMIT" \
      RUN_TIMEOUT_SEC="$RUN_TIMEOUT_SEC" \
      TIMEOUT_KILL_AFTER_SEC="$TIMEOUT_KILL_AFTER_SEC" \
      HWSW_OUTPUT_DIR="$method_dir" \
      HWSW_SOLUTION_DIR="$method_dir/partitions" \
      HWSW_RESULT_PREFIX="$method_prefix" \
      HWSW_RUN_TAG="$RUN_TAG" \
      PYTHON="$PYTHON" \
      "$ROOT/run_all_mip_configs.sh"
    ) >"$group_log" 2>&1 &
  else
    (
      CONFIG_GLOB="$dataset_cfg_dir/*.yaml" \
      OUTDIR="$method_dir" \
      HWSW_METHODS="$method" \
      HWSW_OUTPUT_DIR="$method_dir" \
      HWSW_SOLUTION_DIR="$method_dir/partitions" \
      HWSW_CSV_DIR="$method_dir" \
      HWSW_RESULT_PREFIX="$method_prefix" \
      HWSW_RUN_TAG="$RUN_TAG" \
      HWSW_MAX_PARALLEL_CONFIGS="$GROUP_CONFIG_PARALLEL_JOBS" \
      PYTHON="$PYTHON" \
      "$ROOT/run_all_gnn_configs.sh"
    ) >"$group_log" 2>&1 &
  fi

  ACTIVE_GROUP_PIDS+=("$!")
  ACTIVE_GROUP_LABELS+=("$group_label")
  ACTIVE_GROUP_LOGS+=("$group_log")
  print_banner "Launched group: $group_label"
  echo "  PID      : ${ACTIVE_GROUP_PIDS[-1]}"
  echo "  Group log: $group_log"
}

generate_stable_topology_configs() {
  "$PYTHON" "$ROOT/tools/generate_task_graph_topology_configs.py" \
    --profile "$PROFILE" \
    --config-root "$CONFIG_ROOT" \
    --area "$AREA" \
    --seeds "${CONFIG_CACHE_SEEDS[@]}"
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

mkdir -p "$OUTDIR"
mkdir -p "$SELECTED_CONFIG_ROOT"

print_banner "Starting dataset-area batch run"
echo "  Profile        : $PROFILE"
echo "  Area constraint: $AREA"
echo "  Output root    : $OUTDIR"
echo "  Config root    : $CONFIG_ROOT"
echo "  Selected cfg dir: $SELECTED_CONFIG_ROOT"
echo "  Run tag        : $RUN_TAG"
echo "  Datasets (${#DATASETS[@]}): $(join_by_comma "${DATASETS[@]}")"
echo "  Methods  (${#METHODS[@]}): $(join_by_comma "${METHODS[@]}")"
echo "  Seeds    (${#SEEDS[@]}): $(join_by_comma "${SEEDS[@]}")"
echo "  Config cache seeds: $(join_by_comma "${CONFIG_CACHE_SEEDS[@]}")"
echo "  MIP tlimit    : ${MIP_TIME_LIMIT_SEC}s"
echo "  MIP timeout pad: ${MIP_TIMEOUT_BUFFER_SEC}s"
echo "  MIP hard kill : ${RUN_TIMEOUT_SEC}s"
echo "  MIP gap       : $MIP_GAP"
echo "  MIP node limit: $MIP_NODE_LIMIT"
echo "  CPU count     : $CPU_COUNT ($CPU_COUNT_SOURCE)"
echo "  Outer parallel: $PARALLEL_DATASET_METHODS"
echo "  Max group jobs: $MAX_PARALLEL_DATASET_METHODS"
echo "  Inner cfg jobs: $GROUP_CONFIG_PARALLEL_JOBS"
echo "  Plot step      : disabled in this script; run plot_dataset_area05_10seed.sh separately"

if [[ ! -f "$MANIFEST" || "$FORCE_REGENERATE_CONFIGS" =~ ^(1|true|yes|on)$ ]]; then
  print_banner "Generating stable topology configurations"
  generate_stable_topology_configs
else
  print_banner "Reusing existing stable topology configurations"
  echo "  Using manifest: $MANIFEST"
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

CFG_DIR="$SELECTED_CONFIG_ROOT/root"
print_banner "Selecting requested dataset/seed subset from manifest"
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
  if [[ -d "$SELECTED_CONFIG_ROOT/$dataset" ]]; then
    for legacy_cfg_dir in "$SELECTED_CONFIG_ROOT/$dataset"/*; do
      [[ -d "$legacy_cfg_dir" ]] || continue
      [[ "$(basename "$legacy_cfg_dir")" == "_dataset" ]] && continue
      rm -rf "$legacy_cfg_dir"
    done
  fi
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

    if [[ "$PARALLEL_DATASET_METHODS" =~ ^(1|true|yes|on)$ ]]; then
      launch_dataset_method_group "$dataset" "$method" "$DATASET_CFG_DIR" "$METHOD_DIR" "$METHOD_PREFIX"
    else
      if [[ "$method" == "mip" ]]; then
        echo "  Launching MIP batch with shared dataset configs from $DATASET_CFG_DIR"
        CONFIG_GLOB="$DATASET_CFG_DIR/*.yaml" \
        OUTDIR="$METHOD_DIR" \
        FAST_MIP="$FAST_MIP" \
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
        HWSW_MAX_PARALLEL_CONFIGS="$GROUP_CONFIG_PARALLEL_JOBS" \
        PYTHON="$PYTHON" \
        "$ROOT/run_all_gnn_configs.sh"
      fi
      method_elapsed_sec=$((SECONDS - method_start_sec))
      print_banner "Completed method: $dataset / $method (${method_elapsed_sec}s)"
    fi
  done

  dataset_elapsed_sec=$((SECONDS - dataset_start_sec))
  print_banner "Completed dataset: $dataset (${dataset_elapsed_sec}s)"

done

if [[ "$PARALLEL_DATASET_METHODS" =~ ^(1|true|yes|on)$ ]]; then
  print_banner "Waiting for ${#ACTIVE_GROUP_PIDS[@]} active dataset/method groups"
  wait_for_all_groups
  if (( FAILED_GROUPS > 0 )); then
    echo "$FAILED_GROUPS dataset/method groups failed"
    exit 1
  fi
fi

batch_elapsed_sec=$((SECONDS - batch_start_sec))
print_banner "Finished dataset-area batch run (${batch_elapsed_sec}s)"
echo "Results are under $OUTDIR"
echo "Run plotting separately with:"
echo "  $ROOT/BatchExperiments/plot_dataset_area05_10seed.sh"

echo "Finished dataset-area batch. Outputs are in $OUTDIR"
