#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"

VARIANTS=(
  "diff_gnn"
  "diff_gnn_no_postprocess"
  "diff_gnn_no_refinement"
  # "diff_gnn_no_learned_edge_weights"
  "diff_gnn_order"
  "diff_gnn_order_no_postprocess"
  "diff_gnn_order_no_refinement"
  # "diff_gnn_order_no_learned_edge_weights"
)

DATASETS=(
  # Comment out any dataset you do not want to run by default.
  # Use DATASETS_OVERRIDE="dataset_a dataset_b" to replace this list at runtime.
  # "paper_fig3_11node"
  # "anomaly_detection_tosa"
  # "keyword_spotting_tosa"
  # "image_classification_tosa"
  # "visual_wake_words_tosa"
  "squeeze_net_tosa"
  # "rez_net_tosa"
  # "mobile_net_tosa"
  # "squeezenet_like_1000"
)

# SEEDS=(42 43 44 45 46 47 48 49 50 51)
SEEDS=(42)

if [[ -n "${VARIANTS_OVERRIDE:-}" ]]; then
  read -r -a VARIANTS <<<"$VARIANTS_OVERRIDE"
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

if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  echo "No ablation variants selected. Uncomment entries in VARIANTS or use VARIANTS_OVERRIDE."
  exit 1
fi

PROFILE="full"
AREA="0.5"
RESULT_TAG="ablation_area05"
OUTROOT="${OUTROOT:-$ROOT/BatchExperiments/AblationComponents}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d-%H%M%S')}"
RUN_DIR="$OUTROOT/run-$RUN_TAG"
CONFIG_ROOT="${CONFIG_ROOT:-$ROOT/BatchExperiments/dataset_area05_configs}"
FORCE_REGENERATE_CONFIGS="${FORCE_REGENERATE_CONFIGS:-0}"
SELECTED_CONFIG_ROOT="$RUN_DIR/selected_configs"
CONFIG_PROFILE_ROOT="$CONFIG_ROOT/$PROFILE"
MANIFEST="$CONFIG_PROFILE_ROOT/graph_suite_area05/manifest.csv"
ROOT_MANIFEST="$RUN_DIR/${RESULT_TAG}_selected_manifest.csv"
RUN_MANIFEST="$RUN_DIR/ablation_run_manifest.csv"
GROUP_CONFIG_PARALLEL_JOBS="${HWSW_MAX_PARALLEL_CONFIGS:-${MAX_PARALLEL_CONFIGS:-1}}"

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

print_banner() {
  local message="$1"
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message"
}

join_by_comma() {
  local IFS=', '
  echo "$*"
}

resolve_variant_method() {
  local variant="$1"
  case "$variant" in
    diff_gnn|diff_gnn_no_postprocess|diff_gnn_no_refinement|diff_gnn_no_learned_edge_weights)
      echo "diff_gnn"
      ;;
    diff_gnn_order|diff_gnn_order_no_postprocess|diff_gnn_order_no_refinement|diff_gnn_order_no_learned_edge_weights)
      echo "diff_gnn_order"
      ;;
    *)
      echo "Unsupported ablation variant: $variant" >&2
      exit 1
      ;;
  esac
}

resolve_common_epochs() {
  local dataset="$1"
  if [[ -n "${COMMON_EPOCHS_OVERRIDE:-}" ]]; then
    echo "$COMMON_EPOCHS_OVERRIDE"
    return 0
  fi

  case "$dataset" in
    paper_fig3_11node)
      echo "${ABLATION_EPOCHS_PAPER_FIG3_11NODE:-100}"
      ;;
    squeeze_net_tosa)
      echo "${ABLATION_EPOCHS_SQUEEZE_NET_TOSA:-2500}"
      ;;
    mobile_net_tosa)
      echo "${ABLATION_EPOCHS_MOBILE_NET_TOSA:-1500}"
      ;;
    squeezenet_like_1000)
      echo "${ABLATION_EPOCHS_SQUEEZENET_LIKE_1000:-750}"
      ;;
    squeezenet_like_10000)
      echo "${ABLATION_EPOCHS_SQUEEZENET_LIKE_10000:-500}"
      ;;
    *)
      echo "${COMMON_EPOCHS_DEFAULT:-500}"
      ;;
  esac
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

write_ablation_config() {
  local base_cfg="$1"
  local variant="$2"
  local out_cfg="$3"
  local variant_dir="$4"
  local trace_dir="$5"
  local result_prefix="$6"
  local dataset="$7"
  local epochs="$8"
  "$PYTHON" - <<'PY' "$base_cfg" "$variant" "$out_cfg" "$variant_dir" "$trace_dir" "$result_prefix" "$dataset" "$epochs"
from pathlib import Path
import os
import sys
import yaml

base_cfg, variant, out_cfg, variant_dir, trace_dir, result_prefix, dataset, epochs = sys.argv[1:9]
epochs = int(epochs)
base_path = Path(base_cfg).resolve()
out_path = Path(out_cfg).resolve()
variant_root = Path(variant_dir).resolve()
trace_root = Path(trace_dir).resolve()
trace_root.mkdir(parents=True, exist_ok=True)

with base_path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

seed = int(cfg.get("seed", 42))
graph_file = str(cfg.get("graph-file", "") or "")
graph_name = str(dataset or Path(graph_file).stem or cfg.get("graph_name", "") or "")
method = "diff_gnn_order" if variant.startswith("diff_gnn_order") else "diff_gnn"
hard_eval_default = max(1, epochs // 5)
hard_eval_every = max(1, int(os.environ.get("ABLATION_HARD_EVAL_EVERY", hard_eval_default)))
trace_every = max(1, int(os.environ.get("ABLATION_TRACE_EVERY", epochs)))
progress_every = max(1, int(os.environ.get("ABLATION_PROGRESS_EVERY", max(1, epochs // 10))))
trace_csv = trace_root / f"seed-{seed}__{base_path.stem}__trace.csv"

cfg["output-dir"] = str(variant_root)
cfg["solution-dir"] = str((variant_root / "partitions").resolve())
cfg["result-file-prefix"] = str(result_prefix)
cfg["methods"] = [method]
cfg["ablation_variant"] = str(variant)
cfg["_dataset_name"] = graph_name
cfg["_graph_name"] = graph_name
cfg["_graph_file"] = graph_file
cfg["_source_config_path"] = str(base_path)

ablation_trace = {
    "enabled": True,
    "output_csv": str(trace_csv),
    "compute_every": trace_every,
    "discrete_threshold": float(os.environ.get("ABLATION_TRACE_THRESHOLD", "0.5")),
    "soft_mode": str(os.environ.get("ABLATION_TRACE_SOFT_MODE", "")) or None,
    "include_static_lssp": True,
    "include_learned_swprio_lssp": True,
}
ablation_trace = {k: v for k, v in ablation_trace.items() if v is not None}

diff_cfg = dict(cfg.get("diffgnn", {}) or {})
diff_cfg["iter"] = epochs
diff_cfg["epochs"] = epochs
diff_cfg["hard_eval_every"] = hard_eval_every
diff_cfg["progress_log_every"] = progress_every
diff_cfg.setdefault("selection_metric_train", "queue")
diff_cfg.setdefault("selection_metric_final", diff_cfg.get("selection_metric_train", "queue"))
# Disable learned edge weights by default for the current 6-case ablation.
diff_cfg["learn_edge_weight"] = False
diff_cfg["learned_edge_weight"] = False
diff_cfg["edge_weight_learner"] = "none"
diff_post = dict(diff_cfg.get("postprocess", {}) or {})
diff_post.setdefault("mode", "hybrid")
diff_post.setdefault("eval_mode", "lssp")
diff_cfg["postprocess"] = diff_post
diff_cfg["ablation_trace"] = dict(ablation_trace)

if variant == "diff_gnn_no_postprocess":
    diff_cfg["postprocess"]["enabled"] = False
    diff_cfg["postprocess"]["mode"] = "none"
elif variant == "diff_gnn_no_refinement":
    diff_cfg["postprocess"]["dls_steps"] = 0
    diff_cfg["postprocess"]["dls_fill_decode"] = False
elif variant == "diff_gnn_no_learned_edge_weights":
    diff_cfg["learn_edge_weight"] = False
    diff_cfg["learned_edge_weight"] = False
    diff_cfg["edge_weight_learner"] = "none"

cfg["diffgnn"] = diff_cfg

order_cfg = dict(cfg.get("diffgnn_order", {}) or {})
order_cfg["iter"] = epochs
order_cfg["epochs"] = epochs
order_cfg["early_stop_enabled"] = False
order_cfg["early_stop_min_epochs"] = epochs
order_cfg["hard_eval_every"] = hard_eval_every
order_cfg["progress_log_every"] = progress_every
order_cfg.setdefault("selection_metric_train", "queue")
order_cfg.setdefault("selection_metric_final", order_cfg.get("selection_metric_train", "queue"))
# Disable learned edge weights by default for the current 6-case ablation.
order_cfg["learn_edge_weight"] = False
order_cfg["learned_edge_weight"] = False
order_cfg["edge_weight_learner"] = "none"
order_post = dict(order_cfg.get("postprocess", {}) or {})
order_post.setdefault("mode", "hybrid")
order_post.setdefault("eval_mode", "lssp")
order_cfg["postprocess"] = order_post
order_cfg["ablation_trace"] = dict(ablation_trace)

if variant == "diff_gnn_order_no_postprocess":
    order_cfg["postprocess"]["enabled"] = False
    order_cfg["postprocess"]["mode"] = "none"
elif variant == "diff_gnn_order_no_refinement":
    order_cfg["order_refine_steps"] = 0
elif variant == "diff_gnn_order_no_learned_edge_weights":
    order_cfg["learn_edge_weight"] = False
    order_cfg["learned_edge_weight"] = False
    order_cfg["edge_weight_learner"] = "none"

cfg["diffgnn_order"] = order_cfg

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
}

mkdir -p "$RUN_DIR"
mkdir -p "$SELECTED_CONFIG_ROOT"

print_banner "Starting ablation batch run"
echo "  Profile         : $PROFILE"
echo "  Area constraint : $AREA"
echo "  Output root     : $OUTROOT"
echo "  Run dir         : $RUN_DIR"
echo "  Config root     : $CONFIG_ROOT"
echo "  Selected cfg dir: $SELECTED_CONFIG_ROOT"
echo "  Run tag         : $RUN_TAG"
echo "  Datasets (${#DATASETS[@]}): $(join_by_comma "${DATASETS[@]}")"
echo "  Variants (${#VARIANTS[@]}): $(join_by_comma "${VARIANTS[@]}")"
echo "  Seeds    (${#SEEDS[@]}): $(join_by_comma "${SEEDS[@]}")"
echo "  Config cache seeds: $(join_by_comma "${CONFIG_CACHE_SEEDS[@]}")"
echo "  Inner cfg jobs : $GROUP_CONFIG_PARALLEL_JOBS"

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

cp "$CFG_DIR/selected_manifest.csv" "$ROOT_MANIFEST"

printf 'dataset,variant,method,epochs,variant_dir,result_prefix,summary_csv,trace_dir\n' >"$RUN_MANIFEST"

dataset_idx=0
total_datasets=${#DATASETS[@]}
total_variants=${#VARIANTS[@]}

for dataset in "${DATASETS[@]}"; do
  dataset_idx=$((dataset_idx + 1))
  DATASET_DIR="$RUN_DIR/$dataset"
  DATASET_CFG_DIR="$SELECTED_CONFIG_ROOT/$dataset/_dataset"
  mkdir -p "$DATASET_DIR"

  print_banner "Dataset [$dataset_idx/$total_datasets]: $dataset"
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

  epochs="$(resolve_common_epochs "$dataset")"
  variant_idx=0
  for variant in "${VARIANTS[@]}"; do
    variant_idx=$((variant_idx + 1))
    method="$(resolve_variant_method "$variant")"
    variant_dir="$DATASET_DIR/$variant"
    variant_cfg_dir="$variant_dir/configs"
    trace_dir="$variant_dir/traces"
    result_prefix="${RESULT_TAG}_${dataset}_${variant}"
    summary_csv="$variant_dir/${result_prefix}-result-summary-soda-graphs-config.csv"

    mkdir -p "$variant_cfg_dir" "$trace_dir" "$variant_dir/partitions"

    shopt -s nullglob
    base_cfgs=("$DATASET_CFG_DIR"/*.yaml)
    shopt -u nullglob
    if [[ ${#base_cfgs[@]} -eq 0 ]]; then
      echo "No dataset configs found in $DATASET_CFG_DIR for $dataset"
      exit 1
    fi

    for base_cfg in "${base_cfgs[@]}"; do
      real_cfg="$(readlink -f "$base_cfg")"
      base_stem="$(basename "${real_cfg%.yaml}")"
      out_cfg="$variant_cfg_dir/${base_stem}.${variant}.yaml"
      write_ablation_config "$real_cfg" "$variant" "$out_cfg" "$variant_dir" "$trace_dir" "$result_prefix" "$dataset" "$epochs"
    done

    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$dataset" \
      "$variant" \
      "$method" \
      "$epochs" \
      "$variant_dir" \
      "$result_prefix" \
      "$summary_csv" \
      "$trace_dir" >>"$RUN_MANIFEST"

    print_banner "Dataset [$dataset_idx/$total_datasets] Variant [$variant_idx/$total_variants]: $dataset / $variant"
    echo "  Method       : $method"
    echo "  Common epochs: $epochs"
    echo "  Config dir   : $variant_cfg_dir"
    echo "  Output dir   : $variant_dir"
    echo "  Summary CSV  : $summary_csv"

    CONFIG_GLOB="$variant_cfg_dir/*.yaml" \
    OUTDIR="$variant_dir" \
    HWSW_METHODS="$method" \
    HWSW_OUTPUT_DIR="$variant_dir" \
    HWSW_SOLUTION_DIR="$variant_dir/partitions" \
    HWSW_CSV_DIR="$variant_dir" \
    HWSW_RESULT_PREFIX="$result_prefix" \
    HWSW_RUN_TAG="$RUN_TAG" \
    HWSW_MAX_PARALLEL_CONFIGS="$GROUP_CONFIG_PARALLEL_JOBS" \
    PYTHON="$PYTHON" \
    "$ROOT/run_all_gnn_configs.sh"
  done
done

print_banner "Finished ablation batch run"
echo "Results are under $RUN_DIR"
echo "Summarize with:"
echo "  RUN_DIR=\"$RUN_DIR\" $ROOT/BatchExperiments/run_ablation_stats.sh"
