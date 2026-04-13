#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_COMBOPT_PYTHON="/people/dass304/.conda/envs/combopt/bin/python"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$DEFAULT_COMBOPT_PYTHON" ]]; then
    PYTHON="$DEFAULT_COMBOPT_PYTHON"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  else
    PYTHON="python"
  fi
fi
MANIFEST="${MANIFEST:-$ROOT/BatchExperiments/dataset_area05_configs/full/graph_suite_area05/manifest.csv}"
OUTROOT="${OUTROOT:-$ROOT/BatchExperiments/Ablation/diff_gnn_order_lssp_trace}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
METHOD="${METHOD:-diff_gnn_order}"
AREA_TAG="${AREA_TAG:-area-0.50}"
DEVICE="${DEVICE:-gpu}"
SEEDS_CSV="${SEEDS_CSV:-42}"
DATASETS_CSV="${DATASETS_CSV:-paper_fig3_11node,squeeze_net_tosa}"

IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

mkdir -p "$OUTROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running diff_gnn_order LSSP-trace ablation"
echo "  Python   : $PYTHON"
echo "  Manifest : $MANIFEST"
echo "  Out root : $OUTROOT"
echo "  Run tag  : $RUN_TAG"
echo "  Device   : $DEVICE"
echo "  Datasets : ${DATASETS[*]}"
echo "  Seeds    : ${SEEDS[*]}"

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    case "$dataset" in
      paper_fig3_11node)
        epochs="${PAPER_FIG3_EPOCHS:-40}"
        ;;
      squeeze_net_tosa)
        epochs="${SQUEEZENET_TOSA_EPOCHS:-100}"
        ;;
      *)
        epochs="${DEFAULT_EPOCHS:-100}"
        ;;
    esac

    RUN_DIR="$OUTROOT/$dataset/seed-${seed}/run-${RUN_TAG}"
    CFG_DIR="$RUN_DIR/config"
    mkdir -p "$CFG_DIR"

    BASE_CFG="$("$PYTHON" - <<'PY' "$MANIFEST" "$dataset" "$seed" "$AREA_TAG"
import pandas as pd
import sys

manifest, dataset, seed, area_tag = sys.argv[1:5]
df = pd.read_csv(manifest)
sub = df[df["graph_name"].astype(str) == dataset].copy()
if sub.empty:
    raise SystemExit(f"No manifest rows found for dataset {dataset}")
seed_token = f"seed-{seed}"
area_token = str(area_tag)
sub = sub[
    sub["config_stem"].astype(str).str.contains(seed_token, regex=False)
    & sub["config_stem"].astype(str).str.contains(area_token, regex=False)
]
if sub.empty:
    raise SystemExit(
        f"No config matched dataset={dataset}, seed={seed}, area token={area_token} in {manifest}"
    )
print(str(sub.iloc[0]["config_path"]))
PY
)"
    if [[ "$BASE_CFG" != /* ]]; then
      BASE_CFG="$ROOT/$BASE_CFG"
    fi

    TRACE_CSV="$RUN_DIR/${dataset}_diff_gnn_order_lssp_trace.csv"
    RESULT_PREFIX="ablation_${dataset}_seed-${seed}_${RUN_TAG}"
    ABLATION_CFG="$CFG_DIR/${dataset}_seed-${seed}_ablation.yaml"

    "$PYTHON" - <<'PY' "$BASE_CFG" "$ABLATION_CFG" "$RUN_DIR" "$RESULT_PREFIX" "$TRACE_CSV" "$epochs" "$DEVICE"
from pathlib import Path
import sys
import yaml

base_cfg, out_cfg, run_dir, result_prefix, trace_csv, epochs, device = sys.argv[1:8]
with open(base_cfg, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["output-dir"] = run_dir
cfg["result-file-prefix"] = result_prefix
cfg["methods"] = "diff_gnn_order"
cfg.setdefault("device", device)

diff_cfg = dict(cfg.get("diffgnn_order", {}) or {})
diff_cfg.setdefault("device", device)
diff_cfg["iter"] = int(epochs)
diff_cfg["epochs"] = int(epochs)
diff_cfg["progress_log_every"] = 1
diff_cfg["soft_makespan_exact_every"] = 1
diff_cfg["postprocess"] = dict(diff_cfg.get("postprocess", {}) or {})
diff_cfg["postprocess"]["mode"] = "hybrid"
diff_cfg["postprocess"]["eval_mode"] = "lssp"
diff_cfg["postprocess"]["print_progress"] = True
diff_cfg["postprocess"]["print_every"] = 1
diff_cfg["postprocess"]["final_all_decode_candidates"] = False
diff_cfg["ablation_trace"] = {
    "enabled": True,
    "output_csv": trace_csv,
    "compute_every": 1,
    "discrete_threshold": 0.5,
    "soft_mode": "sequential",
    "include_static_lssp": True,
    "include_learned_swprio_lssp": True,
}
cfg["diffgnn_order"] = diff_cfg

Path(out_cfg).parent.mkdir(parents=True, exist_ok=True)
with open(out_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

    echo
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dataset=$dataset Seed=$seed Epochs=$epochs"
    echo "  Base cfg    : $BASE_CFG"
    echo "  Ablation cfg: $ABLATION_CFG"
    echo "  Run dir     : $RUN_DIR"
    echo "  Trace csv   : $TRACE_CSV"

    (
      cd "$ROOT"
      "$PYTHON" "$ROOT/gnn_main.py" -c "$ABLATION_CFG" --methods "$METHOD"
    ) | tee "$RUN_DIR/gnn_main.log"

    if [[ ! -f "$TRACE_CSV" ]]; then
      echo "Expected trace CSV was not generated: $TRACE_CSV" >&2
      exit 1
    fi
  done
done

echo
echo "Finished diff_gnn_order LSSP-trace ablation runs."
echo "Plot separately with:"
echo "  $ROOT/BatchExperiments/Ablation/plot_diff_gnn_order_lssp_trace.sh"
