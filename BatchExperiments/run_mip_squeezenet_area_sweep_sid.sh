#!/bin/bash
set -euo pipefail

export HWSW_METHOD_RUNTIME_PROFILE="${HWSW_METHOD_RUNTIME_PROFILE:-arato}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BATCH_DIR="$ROOT/BatchExperiments"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-combopt}"
JOB_PYTHON="${JOB_PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"

GRAPH_NAME="${GRAPH_NAME:-squeeze_net_tosa}"
PROFILE="${PROFILE:-full}"
RESULT_TAG="${RESULT_TAG:-squeezenet_area_sweep_10seed}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d-%H%M%S')}"

SEEDS=(42 43 44 45 46 47 48 49 50 51)
AREAS=(0.1 0.3 0.7 0.9)

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<<"$SEEDS_OVERRIDE"
fi
if [[ -n "${AREAS_OVERRIDE:-}" ]]; then
  read -r -a AREAS <<<"$AREAS_OVERRIDE"
fi

OUTDIR="${OUTDIR:-$BATCH_DIR/squeezenet_area_sweep}"
ROOT_MIP_CSV="${ROOT_MIP_CSV:-$OUTDIR/mip_${RESULT_TAG}-result-summary-soda-graphs-config.csv}"
CSV_FRAGMENT_DIR="${CSV_FRAGMENT_DIR:-$OUTDIR/mip_csv_fragments}"

MIP_SOLVER_TOOL="${MIP_SOLVER_TOOL:-cvxpy-scip}"
MIP_SOLVE_MODE="${MIP_SOLVE_MODE:-exact}"
MIP_SW_CONSTRAINT_MODE="${MIP_SW_CONSTRAINT_MODE:-pairwise_topo}"
MIP_USE_REDUCED_SW="${MIP_USE_REDUCED_SW:-false}"
MIP_ACCEPT_NONOPTIMAL="${MIP_ACCEPT_NONOPTIMAL:-false}"
MIP_VERBOSE="${MIP_VERBOSE:-true}"
MIP_TIME_LIMIT_SEC="${MIP_TIME_LIMIT_SEC:-3600}"
MIP_GAP="${MIP_GAP:-0}"
MIP_NODE_LIMIT="${MIP_NODE_LIMIT:-0}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-0}"
TIMEOUT_KILL_AFTER_SEC="${TIMEOUT_KILL_AFTER_SEC:-180}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-encode_optimize}"
SBATCH_PARTITION="${SBATCH_PARTITION:-slurm}"
SBATCH_NTASKS_PER_NODE="${SBATCH_NTASKS_PER_NODE:-1}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-1}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
SBATCH_TIME="${SBATCH_TIME:-2:00:00}"
SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-run_cvxpy_mip_sqarea}"

SBATCH_ROOT="${SBATCH_ROOT:-$BATCH_DIR/slurm_jobs/run_mip_squeezenet_area_sweep_sid}"
SBATCH_OUT_ROOT="${SBATCH_OUT_ROOT:-$BATCH_DIR/slurm_outputs/run_mip_squeezenet_area_sweep_sid}"
CONFIG_ROOT="${CONFIG_ROOT:-$OUTDIR/config_cache}"
SELECTION_ROOT="${SELECTION_ROOT:-$SBATCH_ROOT/selected_${RUN_TAG}}"
DRY_RUN="${DRY_RUN:-0}"

join_by_comma() {
  local IFS=', '
  echo "$*"
}

sanitize_squeezenet_configs() {
  local sweep_dir="$CONFIG_ROOT/$PROFILE/squeeze_net_tosa_area_sweep/squeeze_net_tosa"
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

mkdir -p "$SBATCH_ROOT" "$SBATCH_OUT_ROOT" "$SELECTION_ROOT" "$OUTDIR" "$CSV_FRAGMENT_DIR" "$OUTDIR/partitions"

echo "Preparing SqueezeNet area-sweep MIP submissions"
echo "  Graph        : $GRAPH_NAME"
echo "  Areas        : $(join_by_comma "${AREAS[@]}")"
echo "  Seeds        : $(join_by_comma "${SEEDS[@]}")"
echo "  Profile      : $PROFILE"
echo "  Run tag      : $RUN_TAG"
echo "  Output dir   : $OUTDIR"
echo "  MIP CSV      : $ROOT_MIP_CSV"
echo "  SBATCH root  : $SBATCH_ROOT"
echo "  Output logs  : $SBATCH_OUT_ROOT"
echo "  Config root  : $CONFIG_ROOT"
echo "  Conda env    : $CONDA_ENV_NAME"
echo "  Job python   : $JOB_PYTHON"
echo "  Dry run      : $DRY_RUN"

"$PYTHON" "$ROOT/tools/generate_task_graph_topology_configs.py" \
  --profile "$PROFILE" \
  --config-root "$CONFIG_ROOT" \
  --seeds "${SEEDS[@]}" \
  --squeeze-areas "${AREAS[@]}"

sanitize_squeezenet_configs

MANIFEST="$CONFIG_ROOT/$PROFILE/squeeze_net_tosa_area_sweep/manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

"$PYTHON" "$ROOT/tools/select_configs_from_manifest.py" \
  --manifest "$MANIFEST" \
  --out-dir "$SELECTION_ROOT" \
  --graph-names "$GRAPH_NAME" \
  --seeds "${SEEDS[@]}" \
  --areas "${AREAS[@]}"

cp "$SELECTION_ROOT/selected_manifest.csv" "$OUTDIR/${RESULT_TAG}_selected_manifest.csv"

EXPECTED_CONFIGS=$(( ${#SEEDS[@]} * ${#AREAS[@]} ))
ACTUAL_CONFIGS="$("$PYTHON" - <<'PY' "$SELECTION_ROOT/selected_manifest.csv"
from pathlib import Path
import sys
import pandas as pd

path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    print(0)
else:
    print(len(pd.read_csv(path)))
PY
)"

if [[ "$ACTUAL_CONFIGS" -ne "$EXPECTED_CONFIGS" ]]; then
  echo "Config selection mismatch: expected $EXPECTED_CONFIGS rows, found $ACTUAL_CONFIGS in $SELECTION_ROOT/selected_manifest.csv"
  exit 1
fi

SUBMITTED=0

while IFS=$'\t' read -r CONFIG_PATH AREA_VALUE SEED_VALUE; do
  [[ -n "$CONFIG_PATH" ]] || continue

  CONFIG_BASE="$(basename "$CONFIG_PATH" .yaml)"
  JOB_DIR="$SBATCH_ROOT/$GRAPH_NAME"
  LOG_DIR="$SBATCH_OUT_ROOT/$GRAPH_NAME"
  SBATCH_FILE="$JOB_DIR/${CONFIG_BASE}.sbatch"
  LOG_OUT="$LOG_DIR/${CONFIG_BASE}_%j.out"
  LOG_ERR="$LOG_DIR/${CONFIG_BASE}_%j.err"
  CSV_FRAGMENT="$CSV_FRAGMENT_DIR/${CONFIG_BASE}.csv"
  LOCK_FILE="${ROOT_MIP_CSV}.lock"

  mkdir -p "$JOB_DIR" "$LOG_DIR"

  cat >"$SBATCH_FILE" <<EOF
#!/bin/bash
# Auto-generated by BatchExperiments/run_mip_squeezenet_area_sweep_sid.sh
# Mirrors the 4-area, 10-seed squeeze_net_tosa sweep from
# BatchExperiments/run_squeezenet_area_sweep_10seed.sh, but submits one MIP job per config.

#SBATCH --output=$LOG_OUT
#SBATCH --error=$LOG_ERR
#SBATCH -A $SBATCH_ACCOUNT
#SBATCH -p $SBATCH_PARTITION
#SBATCH --ntasks-per-node=$SBATCH_NTASKS_PER_NODE
#SBATCH --cpus-per-task=$SBATCH_CPUS_PER_TASK
#SBATCH --mem=$SBATCH_MEM
#SBATCH -t $SBATCH_TIME
#SBATCH --job-name=$SBATCH_JOB_NAME

set -euo pipefail

cd "$ROOT"
module load python/miniforge25.3.0
source /share/apps/python/miniforge25.3.0/etc/profile.d/conda.sh

mkdir -p "$OUTDIR" "$OUTDIR/partitions" "$CSV_FRAGMENT_DIR"

conda run -n "$CONDA_ENV_NAME" --no-capture-output env \
  CONFIG_GLOB="$CONFIG_PATH" \
  OUTDIR="$OUTDIR" \
  SOLVER_TOOL="$MIP_SOLVER_TOOL" \
  FAST_MIP="1" \
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
  HWSW_OUTPUT_DIR="$OUTDIR" \
  HWSW_SOLUTION_DIR="$OUTDIR/partitions" \
  HWSW_RESULT_PREFIX="mip_${RESULT_TAG}" \
  HWSW_RESULT_CSV="$CSV_FRAGMENT" \
  HWSW_RUN_TAG="$RUN_TAG" \
  PYTHON="$JOB_PYTHON" \
  bash -lc 'cd "'"$ROOT"'" && ./run_all_mip_configs.sh'

exec 9>"$LOCK_FILE"
flock 9
"$JOB_PYTHON" - <<'PY' "$CSV_FRAGMENT_DIR" "$ROOT_MIP_CSV"
from pathlib import Path
import sys
import pandas as pd

frag_dir = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
paths = sorted(frag_dir.glob("*.csv"))
frames = []
for path in paths:
    try:
        frame = pd.read_csv(path)
    except Exception:
        continue
    if frame.empty:
        continue
    frames.append(frame)

if not frames:
    raise SystemExit(0)

merged = pd.concat(frames, ignore_index=True)
dedupe_cols = [
    col for col in [
        "Config",
        "GraphName",
        "Seed",
        "Area_Percentage",
        "HW_Scale_Factor",
        "HW_Scale_Var",
        "Comm_Scale_Var",
    ] if col in merged.columns
]
if dedupe_cols:
    merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
merged.to_csv(out_csv, index=False)
print(f"Wrote merged MIP CSV to {out_csv}")
PY
EOF

  chmod +x "$SBATCH_FILE"

  if [[ "$DRY_RUN" =~ ^(1|true|yes|on)$ ]]; then
    echo "DRY_RUN sbatch $SBATCH_FILE"
  else
    SUBMIT_OUTPUT="$(sbatch "$SBATCH_FILE")"
    echo "$SUBMIT_OUTPUT :: area=$AREA_VALUE seed=$SEED_VALUE :: $CONFIG_BASE"
    SUBMITTED=$((SUBMITTED + 1))
  fi
done < <(
  "$PYTHON" - <<'PY' "$SELECTION_ROOT/selected_manifest.csv" "$ROOT"
from pathlib import Path
import csv
import sys

manifest = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()

with manifest.open(newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        config_path = Path(row["config_path"])
        if not config_path.is_absolute():
            config_path = root / config_path
        print(f"{config_path}\t{row['area_constraint']}\t{row['seed']}")
PY
)

if [[ "$DRY_RUN" =~ ^(1|true|yes|on)$ ]]; then
  echo "Prepared $ACTUAL_CONFIGS sbatch files under $SBATCH_ROOT"
else
  echo "Submitted $SUBMITTED jobs"
fi
