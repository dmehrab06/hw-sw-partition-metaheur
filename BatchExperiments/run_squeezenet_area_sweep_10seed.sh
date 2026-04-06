#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"

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

# Edit this array to control the number of seeds.
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Edit this array to control the area sweep.
AREAS=(0.1 0.3 0.5 0.7)

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
OUTDIR="$ROOT/BatchExperiments/squeezenet_area_sweep"
TMPDIR_ROOT="$(mktemp -d "$ROOT/BatchExperiments/.squeezenet_sweep_tmp.XXXXXX")"
trap 'rm -rf "$TMPDIR_ROOT"' EXIT
CONFIG_ROOT="$TMPDIR_ROOT/task_graph_topology_config"

FAST_MIP="${FAST_MIP:-1}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-1800}"
TIMEOUT_KILL_AFTER_SEC="${TIMEOUT_KILL_AFTER_SEC:-30}"

CONFIG_PROFILE_ROOT="$CONFIG_ROOT/$PROFILE"
MANIFEST="$CONFIG_PROFILE_ROOT/squeeze_net_tosa_area_sweep/manifest.csv"
GNN_CSV="$OUTDIR/${RESULT_TAG}-result-summary-soda-graphs-config.csv"
MIP_CSV="$OUTDIR/mip_${RESULT_TAG}-result-summary-soda-graphs-config.csv"

mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.csv "$OUTDIR"/*.png "$OUTDIR"/*.pdf "$OUTDIR"/mip_eval_*.log

"$PYTHON" "$ROOT/tools/generate_task_graph_topology_configs.py" \
  --profile "$PROFILE" \
  --config-root "$CONFIG_ROOT" \
  --seeds "${SEEDS[@]}" \
  --squeeze-areas "${AREAS[@]}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

CFG_DIR="$TMPDIR_ROOT/configs"
"$PYTHON" "$ROOT/tools/select_configs_from_manifest.py" \
  --manifest "$MANIFEST" \
  --out-dir "$CFG_DIR" \
  --seeds "${SEEDS[@]}" \
  --areas "${AREAS[@]}"
cp "$CFG_DIR/selected_manifest.csv" "$OUTDIR/${RESULT_TAG}_selected_manifest.csv"

for cfg in "$CFG_DIR"/*.yaml; do
  src_cfg="$(readlink -f "$cfg")"
  rm -f "$cfg"
  "$PYTHON" - <<'PY' "$src_cfg" "$cfg" "$OUTDIR" "$RESULT_TAG"
from omegaconf import OmegaConf
import sys

src, dst, out_dir, result_tag = sys.argv[1:]
cfg = OmegaConf.load(src)
cfg["output-dir"] = out_dir
cfg["solution-dir"] = f"{out_dir}/partitions"
cfg["result-file-prefix"] = result_tag
vis = dict(cfg.get("visualization", {}))
vis["enabled"] = False
cfg["visualization"] = vis
OmegaConf.save(config=cfg, f=dst)
PY
done

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
  HWSW_METHODS="$GNN_METHODS_CSV" \
  HWSW_CSV_DIR="$OUTDIR" \
  HWSW_RESULT_PREFIX="$RESULT_TAG" \
  PYTHON="$PYTHON" \
  "$ROOT/run_all_gnn_configs.sh"
fi

if [[ "$RUN_MIP" == "1" ]]; then
  CONFIG_GLOB="$CFG_DIR/*.yaml" \
  OUTDIR="$OUTDIR" \
  FAST_MIP="$FAST_MIP" \
  RUN_TIMEOUT_SEC="$RUN_TIMEOUT_SEC" \
  TIMEOUT_KILL_AFTER_SEC="$TIMEOUT_KILL_AFTER_SEC" \
  PYTHON="$PYTHON" \
  "$ROOT/run_all_mip_configs.sh"
fi

"$PYTHON" "$ROOT/tools/plot_batch_method_bars.py" \
  --manifest "$OUTDIR/${RESULT_TAG}_selected_manifest.csv" \
  --gnn-csv "$GNN_CSV" \
  --mip-csv "$MIP_CSV" \
  --output-dir "$OUTDIR" \
  --mode area_sweep \
  --methods "${METHODS[@]}" \
  --areas "${AREAS[@]}" \
  --tag "$RESULT_TAG"

echo "Finished SqueezeNet area sweep batch. Outputs are in $OUTDIR"
