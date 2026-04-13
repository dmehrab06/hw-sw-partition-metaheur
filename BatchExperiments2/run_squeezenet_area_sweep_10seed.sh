#!/bin/bash
set -euo pipefail

export HWSW_METHOD_RUNTIME_PROFILE="${HWSW_METHOD_RUNTIME_PROFILE:-balanced}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
GRAPH_NAME="squeeze_net_tosa"
source "$ROOT/BatchExperiments2/batch2_common.sh"

# Comment out any method you do not want to run.
METHODS=(
  # "mip"
  # "diff_gnn_order"
  "gl25"
  "gcps"
  "esa"
  "pso"
  "dbpso"
  "clpso"
  "ccpso"
  "shade"
  "jade"
  # "random"
  # "greedy"
)

# Edit this array to control the number of seeds.
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Edit this array to control the area sweep.
AREAS=(0.1 0.3 0.7 0.9)

if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  read -r -a METHODS <<<"$METHODS_OVERRIDE"
fi
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<<"$SEEDS_OVERRIDE"
fi
if [[ -n "${AREAS_OVERRIDE:-}" ]]; then
  read -r -a AREAS <<<"$AREAS_OVERRIDE"
fi

PROFILE="full"
RESULT_TAG="squeezenet_area_sweep_10seed"
OUTDIR="$ROOT/BatchExperiments2/squeezenet_area_sweep"
CONFIG_ROOT="${CONFIG_ROOT:-$OUTDIR/config_cache}"
SELECTED_CONFIG_ROOT="${SELECTED_CONFIG_ROOT:-$OUTDIR/selected_configs}"
FORCE_REGENERATE_CONFIGS="${FORCE_REGENERATE_CONFIGS:-0}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d-%H%M%S')}"

FAST_MIP="${FAST_MIP:-1}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-1800}"
TIMEOUT_KILL_AFTER_SEC="${TIMEOUT_KILL_AFTER_SEC:-30}"

CONFIG_PROFILE_ROOT="$CONFIG_ROOT/$PROFILE"
MANIFEST="$CONFIG_PROFILE_ROOT/squeeze_net_tosa_area_sweep/manifest.csv"
GNN_CSV="$OUTDIR/${RESULT_TAG}-result-summary-soda-graphs-config.csv"
MIP_CSV="$OUTDIR/mip_${RESULT_TAG}-result-summary-soda-graphs-config.csv"

print_banner() {
  local message="$1"
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message"
}

join_by_comma() {
  local IFS=', '
  echo "$*"
}

generate_stable_topology_configs() {
  "$PYTHON" "$ROOT/tools/generate_task_graph_topology_configs.py" \
    --profile "$PROFILE" \
    --config-root "$CONFIG_ROOT" \
    --seeds "${SEEDS[@]}" \
    --squeeze-areas "${AREAS[@]}" \
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

sanitize_squeezenet_configs() {
  local sweep_dir="$CONFIG_PROFILE_ROOT/squeeze_net_tosa_area_sweep/squeeze_net_tosa"
  if [[ ! -d "$sweep_dir" ]]; then
    return 0
  fi
  "$PYTHON" - <<'PY' "$sweep_dir"
from pathlib import Path
import sys
import yaml

sweep_dir = Path(sys.argv[1])
updated = 0
for path in sorted(sweep_dir.glob("*.yaml")):
    with path.open() as handle:
        cfg = yaml.safe_load(handle) or {}

    changed = False
    methods = cfg.get("methods", None)
    if isinstance(methods, list):
        filtered = [m for m in methods if m != "diff_gnn_order"]
        if filtered != methods:
            cfg["methods"] = filtered
            changed = True

    if "diffgnn_order" in cfg:
        cfg.pop("diffgnn_order", None)
        changed = True

    if changed:
        with path.open("w") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        updated += 1

print(f"Sanitized {updated} squeeze-area configs under {sweep_dir}")
PY
}

mkdir -p "$OUTDIR" "$OUTDIR/partitions" "$SELECTED_CONFIG_ROOT"
bootstrap_batchexperiments2_family "squeezenet_area_sweep" "$ROOT" "$PYTHON"

print_banner "Starting SqueezeNet area sweep batch run"
echo "  Graph         : $GRAPH_NAME"
echo "  Profile       : $PROFILE"
echo "  Output root   : $OUTDIR"
echo "  Config root   : $CONFIG_ROOT"
echo "  Selected cfgs : $SELECTED_CONFIG_ROOT"
echo "  Run tag       : $RUN_TAG"
echo "  Methods (${#METHODS[@]}): $(join_by_comma "${METHODS[@]}")"
echo "  Seeds   (${#SEEDS[@]}): $(join_by_comma "${SEEDS[@]}")"
echo "  Areas   (${#AREAS[@]}): $(join_by_comma "${AREAS[@]}")"
if [[ "$RUN_TIMEOUT_SEC" =~ ^[0-9]+$ ]] && (( RUN_TIMEOUT_SEC > 0 )); then
  echo "  MIP watchdog  : ${RUN_TIMEOUT_SEC}s (kill-after ${TIMEOUT_KILL_AFTER_SEC}s)"
else
  echo "  MIP watchdog  : disabled"
fi

if [[ ! -f "$MANIFEST" || "$FORCE_REGENERATE_CONFIGS" =~ ^(1|true|yes|on)$ ]]; then
  print_banner "Generating stable topology configurations"
  generate_stable_topology_configs
else
  print_banner "Reusing existing stable topology configurations"
  echo "  Using manifest: $MANIFEST"
fi

sanitize_squeezenet_configs

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

CFG_DIR="$SELECTED_CONFIG_ROOT/root"
print_banner "Selecting requested area/seed subset from manifest"
"$PYTHON" "$ROOT/tools/select_configs_from_manifest.py" \
  --manifest "$MANIFEST" \
  --out-dir "$CFG_DIR" \
  --graph-names "$GRAPH_NAME" \
  --seeds "${SEEDS[@]}" \
  --areas "${AREAS[@]}"
EXPECTED_CONFIGS=$(( ${#SEEDS[@]} * ${#AREAS[@]} ))
ACTUAL_CONFIGS="$(count_manifest_rows "$CFG_DIR/selected_manifest.csv")"
MISSING_CONFIGS="$(count_missing_config_paths "$CFG_DIR/selected_manifest.csv")"
if [[ "$ACTUAL_CONFIGS" -ne "$EXPECTED_CONFIGS" || "$MISSING_CONFIGS" -gt 0 ]]; then
  print_banner "Stable config cache is missing requested rows; regenerating once"
  echo "  Expected rows: $EXPECTED_CONFIGS"
  echo "  Found rows   : $ACTUAL_CONFIGS"
  echo "  Missing cfgs : $MISSING_CONFIGS"
  generate_stable_topology_configs
  sanitize_squeezenet_configs
  "$PYTHON" "$ROOT/tools/select_configs_from_manifest.py" \
    --manifest "$MANIFEST" \
    --out-dir "$CFG_DIR" \
    --graph-names "$GRAPH_NAME" \
    --seeds "${SEEDS[@]}" \
    --areas "${AREAS[@]}"
  ACTUAL_CONFIGS="$(count_manifest_rows "$CFG_DIR/selected_manifest.csv")"
  MISSING_CONFIGS="$(count_missing_config_paths "$CFG_DIR/selected_manifest.csv")"
fi
if [[ "$ACTUAL_CONFIGS" -ne "$EXPECTED_CONFIGS" || "$MISSING_CONFIGS" -gt 0 ]]; then
  echo "Config selection mismatch after regeneration: expected $EXPECTED_CONFIGS rows, found $ACTUAL_CONFIGS rows, missing $MISSING_CONFIGS config files in $CFG_DIR/selected_manifest.csv"
  exit 1
fi

cp "$CFG_DIR/selected_manifest.csv" "$OUTDIR/${RESULT_TAG}_selected_manifest.csv"

GNN_METHODS=()
RUN_MIP=0
for method in "${METHODS[@]}"; do
  if [[ "$method" == "mip" ]]; then
    RUN_MIP=1
  else
    GNN_METHODS+=("$method")
  fi
done

if [[ ${#GNN_METHODS[@]} -gt 0 ]]; then
  GNN_METHODS_CSV="$(IFS=,; echo "${GNN_METHODS[*]}")"
  CONFIG_GLOB="$CFG_DIR/*.yaml" \
  OUTDIR="$OUTDIR" \
  HWSW_METHODS="$GNN_METHODS_CSV" \
  HWSW_OUTPUT_DIR="$OUTDIR" \
  HWSW_SOLUTION_DIR="$OUTDIR/partitions" \
  HWSW_CSV_DIR="$OUTDIR" \
  HWSW_RESULT_CSV="$GNN_CSV" \
  HWSW_RESULT_PREFIX="$RESULT_TAG" \
  HWSW_RUN_TAG="$RUN_TAG" \
  PYTHON="$PYTHON" \
  "$ROOT/run_all_gnn_configs.sh"
fi

if [[ "$RUN_MIP" == "1" ]]; then
  CONFIG_GLOB="$CFG_DIR/*.yaml" \
  OUTDIR="$OUTDIR" \
  FAST_MIP="$FAST_MIP" \
  RUN_TIMEOUT_SEC="$RUN_TIMEOUT_SEC" \
  TIMEOUT_KILL_AFTER_SEC="$TIMEOUT_KILL_AFTER_SEC" \
  HWSW_OUTPUT_DIR="$OUTDIR" \
  HWSW_SOLUTION_DIR="$OUTDIR/partitions" \
  HWSW_RESULT_CSV="$MIP_CSV" \
  HWSW_RESULT_PREFIX="mip_${RESULT_TAG}" \
  HWSW_RUN_TAG="$RUN_TAG" \
  PYTHON="$PYTHON" \
  "$ROOT/run_all_mip_configs.sh"
fi

echo "Finished SqueezeNet area sweep batch. Outputs are in $OUTDIR"
echo "Run plotting separately with:"
echo "  $ROOT/BatchExperiments2/plot_squeezenet_area_sweep_10seed.sh"
