#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$ROOT/configs"
OUTDIR="${OUTDIR:-$ROOT/outputs/logs}"
mkdir -p "$OUTDIR"

export PYTHONNOUSERSITE=1

PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
SOLVER_TOOL="${SOLVER_TOOL:-${MIP_SOLVER_TOOL:-cvxpy-scip}}"
MIP_EVAL_PY="${MIP_EVAL_PY:-mip_eval.py}"
CONFIG_GLOB="${CONFIG_GLOB:-$CONFIG_DIR/config_mkspan_default_gnn.yaml}"
RESULT_CSV_ENV="${HWSW_RESULT_CSV:-${RESULT_CSV:-}}"
RESULT_PREFIX_OVERRIDE="${HWSW_RESULT_PREFIX:-${RESULT_PREFIX:-}}"
OUTPUT_DIR_OVERRIDE="${HWSW_OUTPUT_DIR:-${OUTPUT_DIR:-}}"
SOLUTION_DIR_OVERRIDE="${HWSW_SOLUTION_DIR:-${SOLUTION_DIR:-}}"
RUN_TAG_ENV="${HWSW_RUN_TAG:-${RUN_TAG:-}}"
PARALLEL_RUNNER="$ROOT/tools/run_mip_configs_parallel.py"
ATTACH_SOLVER_STATS_PY="$ROOT/tools/attach_mip_solver_stats.py"
PARALLEL_CONFIG_JOBS="${HWSW_MAX_PARALLEL_CONFIGS:-${MAX_PARALLEL_CONFIGS:-1}}"

if [[ -f "$MIP_EVAL_PY" ]]; then
  MIP_EVAL_ENTRY="$MIP_EVAL_PY"
elif [[ -f "$ROOT/$MIP_EVAL_PY" ]]; then
  MIP_EVAL_ENTRY="$ROOT/$MIP_EVAL_PY"
elif [[ -f "$ROOT/mip_eval.py" ]]; then
  MIP_EVAL_ENTRY="$ROOT/mip_eval.py"
elif [[ -f "$ROOT/milp_eval.py" ]]; then
  MIP_EVAL_ENTRY="$ROOT/milp_eval.py"
else
  echo "Cannot find MIP evaluator script. Checked: $MIP_EVAL_PY, $ROOT/mip_eval.py, $ROOT/milp_eval.py"
  exit 1
fi

# MIP override defaults (can be overridden via env).
# Set FAST_MIP=0 to run configs exactly as they appear in YAML.
# The default profile below is intentionally main-compat: pairwise-topo SW constraints
# and a SCIP-first backend request.
FAST_MIP="${FAST_MIP:-1}"
MIP_SOLVE_MODE="${MIP_SOLVE_MODE:-exact}"
MIP_SW_CONSTRAINT_MODE="${MIP_SW_CONSTRAINT_MODE:-pairwise_topo}"
MIP_USE_REDUCED_SW="${MIP_USE_REDUCED_SW:-false}"
MIP_TIME_LIMIT_SEC="${MIP_TIME_LIMIT_SEC:-600}"
MIP_GAP="${MIP_GAP:-0}"
MIP_NODE_LIMIT="${MIP_NODE_LIMIT:-0}"
MIP_ACCEPT_NONOPTIMAL="${MIP_ACCEPT_NONOPTIMAL:-false}"
MIP_VERBOSE="${MIP_VERBOSE:-true}"

# Optional extra wall-clock guard for each config run.
# Disabled by default so the solver's internal time-limit-sec can flush incumbent artifacts.
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-0}"
TIMEOUT_KILL_AFTER_SEC="${TIMEOUT_KILL_AFTER_SEC:-15}"

cd "$ROOT"

mapfile -t CONFIGS < <(ls $CONFIG_GLOB 2>/dev/null | sort || true)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No config files matched: $CONFIG_GLOB"
  exit 1
fi

echo "Running MIP solver (${SOLVER_TOOL}) on ${#CONFIGS[@]} configs"
echo "FAST_MIP=$FAST_MIP"
echo "MIP evaluator: $(basename "$MIP_EVAL_ENTRY")"
if [[ "$FAST_MIP" =~ ^(1|true|yes|on)$ ]]; then
  echo "MIP override settings: solver=$SOLVER_TOOL, mode=$MIP_SOLVE_MODE, sw=$MIP_SW_CONSTRAINT_MODE, tlimit=${MIP_TIME_LIMIT_SEC}s, gap=$MIP_GAP, nodes=$MIP_NODE_LIMIT, accept_nonoptimal=$MIP_ACCEPT_NONOPTIMAL"
fi
if [[ -n "$RUN_TAG_ENV" ]]; then
  echo "Run tag: $RUN_TAG_ENV"
fi
if [[ -n "$RESULT_PREFIX_OVERRIDE" ]]; then
  echo "Result prefix override: $RESULT_PREFIX_OVERRIDE"
fi
if [[ -n "$RESULT_CSV_ENV" ]]; then
  echo "CSV output override: $RESULT_CSV_ENV"
fi
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  echo "Output directory override: $OUTPUT_DIR_OVERRIDE"
fi
if [[ -n "$SOLUTION_DIR_OVERRIDE" ]]; then
  echo "Solution directory override: $SOLUTION_DIR_OVERRIDE"
fi
if [[ "$RUN_TIMEOUT_SEC" =~ ^[0-9]+$ ]] && (( RUN_TIMEOUT_SEC > 0 )); then
  echo "External watchdog: ${RUN_TIMEOUT_SEC}s (kill-after ${TIMEOUT_KILL_AFTER_SEC}s)"
else
  echo "External watchdog: disabled; using solver internal time-limit-sec"
fi
if [[ "${PARALLEL_CONFIG_JOBS}" =~ ^[0-9]+$ ]] && (( PARALLEL_CONFIG_JOBS > 1 )); then
  echo "Parallel config jobs: $PARALLEL_CONFIG_JOBS"
fi

if [[ "${PARALLEL_CONFIG_JOBS}" =~ ^[0-9]+$ ]] && (( PARALLEL_CONFIG_JOBS > 1 )); then
  env \
    PYTHON="$PYTHON" \
    SOLVER_TOOL="$SOLVER_TOOL" \
    MIP_EVAL_PY="$MIP_EVAL_PY" \
    FAST_MIP="$FAST_MIP" \
    MIP_SOLVE_MODE="$MIP_SOLVE_MODE" \
    MIP_SW_CONSTRAINT_MODE="$MIP_SW_CONSTRAINT_MODE" \
    MIP_USE_REDUCED_SW="$MIP_USE_REDUCED_SW" \
    MIP_TIME_LIMIT_SEC="$MIP_TIME_LIMIT_SEC" \
    MIP_GAP="$MIP_GAP" \
    MIP_NODE_LIMIT="$MIP_NODE_LIMIT" \
    MIP_ACCEPT_NONOPTIMAL="$MIP_ACCEPT_NONOPTIMAL" \
    MIP_VERBOSE="$MIP_VERBOSE" \
    RUN_TIMEOUT_SEC="$RUN_TIMEOUT_SEC" \
    TIMEOUT_KILL_AFTER_SEC="$TIMEOUT_KILL_AFTER_SEC" \
    HWSW_RUN_TAG="$RUN_TAG_ENV" \
    HWSW_RESULT_PREFIX="$RESULT_PREFIX_OVERRIDE" \
    HWSW_RESULT_CSV="$RESULT_CSV_ENV" \
    HWSW_OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE" \
    HWSW_SOLUTION_DIR="$SOLUTION_DIR_OVERRIDE" \
    "$PYTHON" "$PARALLEL_RUNNER" --root "$ROOT" --outdir "$OUTDIR" --script "$ROOT/run_all_mip_configs.sh" --jobs "$PARALLEL_CONFIG_JOBS" "${CONFIGS[@]}"
  exit $?
fi

batch_start_sec=$SECONDS

append_mip_status_row() {
  local config_path="$1"
  local lookup_config_path="$2"
  local out_csv="$3"
  local runtime_sec="$4"
  local status="$5"
  local validity_note="$6"

  "$PYTHON" - <<'PY' "$config_path" "$lookup_config_path" "$out_csv" "$runtime_sec" "$status" "$validity_note" "$ROOT"
import csv
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from meta_heuristic.partition_schedule_evaluator import synchronize_problem_with_config

config_path = Path(sys.argv[1])
lookup_config_path = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
runtime_sec = float(sys.argv[4])
status = str(sys.argv[5])
validity_note = str(sys.argv[6])
root = Path(sys.argv[7]).resolve()

cfg = OmegaConf.load(config_path)
lookup_cfg = OmegaConf.load(lookup_config_path)

def _resolve_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else root / path

task_graph = None
node_count = np.nan
naive_lb = np.nan
area_budget = np.nan
taskgraph_pickle = lookup_cfg.get("taskgraph-pickle", cfg.get("taskgraph-pickle", None))
if taskgraph_pickle:
    tg_path = _resolve_path(taskgraph_pickle)
    if tg_path.exists():
        try:
            with open(tg_path, "rb") as handle:
                task_graph = pickle.load(handle)
            task_graph = synchronize_problem_with_config(task_graph, cfg)
            node_count = len(task_graph.graph.nodes())
            naive_lb = sum(
                min(task_graph.software_costs[node], task_graph.hardware_costs[node])
                for node in task_graph.graph.nodes()
            )
            total_area = getattr(task_graph, "total_area", None)
            if total_area is not None:
                area_budget = float(total_area) * float(cfg.get("area-constraint", 0.0))
        except Exception:
            pass


def _resolve_solution_dir():
    value = lookup_cfg.get("solution-dir", cfg.get("solution-dir", ""))
    if not value:
        return None
    return _resolve_path(value)


def _config_keys():
    return (
        f"{float(cfg.get('area-constraint', 0.0)):.2f}",
        f"{float(cfg.get('hw-scale-factor', 0.0)):.1f}",
        f"{float(cfg.get('hw-scale-variance', 0.0)):.2f}",
        str(cfg.get("seed", 42)),
    )


solution_dir = _resolve_solution_dir()
partition_pkl = None
partition_json = None
partition_meta = None
json_payload = {}
partition_from_json = None

if solution_dir and solution_dir.exists():
    area_key, hw_key, hwvar_key, seed_key = _config_keys()
    stem = f"*area-{area_key}_hwscale-{hw_key}_hwvar-{hwvar_key}_seed-{seed_key}_assignment-mip"
    matches_pkl = sorted(solution_dir.glob(f"{stem}.pkl"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    matches_json = sorted(solution_dir.glob(f"{stem}.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    matches_meta = sorted(solution_dir.glob(f"{stem}.meta.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    if not matches_pkl:
        matches_pkl = sorted(solution_dir.glob("*assignment-mip.pkl"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    if not matches_json:
        matches_json = sorted(solution_dir.glob("*assignment-mip.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    if not matches_meta:
        matches_meta = sorted(solution_dir.glob("*assignment-mip.meta.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    partition_pkl = matches_pkl[0] if matches_pkl else None
    partition_json = matches_json[0] if matches_json else None
    partition_meta = matches_meta[0] if matches_meta else None

if partition_json and partition_json.exists():
    try:
        json_payload = json.loads(partition_json.read_text())
    except Exception:
        json_payload = {}
    partition_payload = json_payload.get("partition_assignment", None)
    if isinstance(partition_payload, list) and partition_payload and isinstance(partition_payload[0], dict):
        partition_from_json = partition_payload[0]
    elif isinstance(partition_payload, dict):
        partition_from_json = partition_payload

solver_status = status
base_data = {
    "SimTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "RunTag": os.getenv("HWSW_RUN_TAG", ""),
    "Config": config_path.stem,
    "GraphName": cfg.get("graph-file", ""),
    "N": node_count,
    "HW_Scale_Factor": cfg.get("hw-scale-factor", np.nan),
    "HW_Scale_Var": cfg.get("hw-scale-variance", np.nan),
    "Comm_Scale_Var": cfg.get("comm-scale-factor", np.nan),
    "Area_Percentage": cfg.get("area-constraint", np.nan),
    "Seed": cfg.get("seed", np.nan),
    "LB_Naive": naive_lb,
}

model_makespan = np.nan
lp_makespan = np.nan
lssp_makespan = np.nan
makespan = np.nan
partition_cost = np.nan
optimization_time_sec = runtime_sec
postprocess_time_sec = 0.0
algorithm_total_time_sec = runtime_sec
solution_valid = False
initial_solution_valid = False
was_repaired = False
num_repaired_nodes = 0
repair_strategy = ""
area_used = np.nan
timeout_note = "Time limit exceeded; no incumbent solution artifact was recorded."
validity_note_out = validity_note

if json_payload:
    payload_status = str(json_payload.get("status", "")).strip()
    if payload_status:
        solver_status = payload_status
    try:
        model_makespan = float(json_payload.get("makespan", np.nan))
    except Exception:
        pass
    try:
        lp_makespan = float(json_payload.get("lp_makespan", np.nan))
    except Exception:
        pass
    try:
        lssp_makespan = float(json_payload.get("final_lssp_makespan", np.nan))
    except Exception:
        pass
    try:
        optimization_time_sec = float(json_payload.get("solve_time_sec", optimization_time_sec))
    except Exception:
        pass
    try:
        postprocess_time_sec = float(json_payload.get("postprocess_time_sec", np.nan))
    except Exception:
        pass
    try:
        algorithm_total_time_sec = float(json_payload.get("algorithm_total_time_sec", np.nan))
    except Exception:
        pass
    if not np.isfinite(postprocess_time_sec):
        if np.isfinite(algorithm_total_time_sec):
            postprocess_time_sec = max(0.0, float(algorithm_total_time_sec) - float(optimization_time_sec))
        else:
            postprocess_time_sec = 0.0
    if not np.isfinite(algorithm_total_time_sec):
        algorithm_total_time_sec = float(optimization_time_sec) + float(postprocess_time_sec)
    if np.isfinite(lssp_makespan):
        makespan = lssp_makespan
    elif np.isfinite(model_makespan):
        makespan = model_makespan
    timeout_note = "Time limit exceeded; CSV row records the best incumbent artifact written before timeout."
    validity_note_out = "Incumbent solution recorded from a timed-out exact MIP run."

if task_graph is not None and ((partition_pkl and partition_pkl.exists()) or isinstance(partition_from_json, dict)):
    try:
        if partition_pkl and partition_pkl.exists():
            with open(partition_pkl, "rb") as handle:
                partition = pickle.load(handle)
        else:
            partition = {str(k): int(v) for k, v in partition_from_json.items()}
        missing = [n for n in task_graph.graph.nodes() if n not in partition]
        for n in missing:
            partition[n] = 0
        from meta_heuristic.partition_schedule_evaluator import evaluate_partition_lssp
        lssp_result = evaluate_partition_lssp(task_graph, partition)
        makespan = float(lssp_result["makespan"])
        lssp_makespan = makespan
        partition_cost = float(task_graph.evaluate_partition_cost(partition))
        solution_valid = bool(lssp_result.get("is_valid", True))
        was_repaired = bool(lssp_result.get("was_repaired", False))
        initial_solution_valid = (not was_repaired)
        num_repaired_nodes = len(lssp_result.get("repaired_nodes", []))
        repair_strategy = str(lssp_result.get("repair_strategy", "benefit_per_area"))
        area_used = float(lssp_result.get("execution_summary", {}).get("area_used", np.nan))
        if np.isfinite(area_budget):
            area_budget = float(lssp_result.get("execution_summary", {}).get("area_budget", area_budget))
        timeout_note = "Time limit exceeded; CSV row records the best incumbent artifact written before timeout."
        if solution_valid and not was_repaired:
            validity_note_out = "Valid incumbent solution; no area repair needed."
        elif solution_valid and was_repaired:
            validity_note_out = (
                "Incumbent solution was invalid before post-processing; repaired with "
                f"{repair_strategy} greedy fixing to satisfy the area constraint."
            )
        else:
            validity_note_out = (
                "Incumbent solution remained invalid after post-processing; reported makespan "
                "reflects the violation penalty."
            )
    except Exception:
        pass

row = {
    **base_data,
    "mip_status": solver_status,
    "mip_model_makespan": model_makespan,
    "mip_lp_makespan": lp_makespan,
    "mip_lssp_makespan": lssp_makespan,
    "mip_opt_cost": makespan,
    "mip_opt_ratio": (makespan / naive_lb) if np.isfinite(makespan) and np.isfinite(naive_lb) and naive_lb > 0 else np.nan,
    "mip_partition_cost": partition_cost,
    "mip_bb": "milp_eval",
    "mip_makespan": makespan,
    "mip_time": runtime_sec,
    "mip_optimization_time_sec": optimization_time_sec,
    "mip_postprocess_time_sec": postprocess_time_sec,
    "mip_total_runtime_sec": algorithm_total_time_sec,
    "mip_solution_valid": solution_valid,
    "mip_initial_solution_valid": initial_solution_valid,
    "mip_was_repaired": was_repaired,
    "mip_num_repaired_nodes": num_repaired_nodes,
    "mip_repair_strategy": repair_strategy,
    "mip_area_used": area_used,
    "mip_area_budget": area_budget,
    "mip_validity_note": validity_note_out,
    "mip_timeout_note": timeout_note,
}

result_df = pd.DataFrame([row])
out_csv.parent.mkdir(parents=True, exist_ok=True)

if out_csv.exists():
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        result_df.to_csv(out_csv, mode="w", index=False, header=True)
    else:
        existing_cols = list(rows[0])
        ordered_cols = existing_cols + [c for c in result_df.columns if c not in existing_cols]

        row_dicts = []
        for raw in rows[1:]:
            padded = list(raw) + [""] * max(0, len(ordered_cols) - len(raw))
            row_dicts.append(dict(zip(ordered_cols, padded[: len(ordered_cols)])))

        if len(ordered_cols) != len(existing_cols):
            existing_df = pd.DataFrame(row_dicts, columns=ordered_cols)
            result_df = result_df.reindex(columns=ordered_cols)
            combined_df = pd.concat([existing_df, result_df], ignore_index=True)
            combined_df.to_csv(out_csv, mode="w", index=False, header=True)
        else:
            result_df = result_df.reindex(columns=ordered_cols)
            result_df.to_csv(out_csv, mode="a", index=False, header=False)
else:
    result_df.to_csv(out_csv, mode="a", index=False, header=True)
PY
}

resolve_mip_artifacts() {
  local solution_dir="$1"
  local area_key="$2"
  local hw_key="$3"
  local hwvar_key="$4"
  local seed_key="$5"
  local stem=""
  local partition_pkl=""
  local partition_json=""
  local partition_meta=""

  stem="*area-${area_key}_hwscale-${hw_key}_hwvar-${hwvar_key}_seed-${seed_key}_assignment-mip"
  partition_pkl=$(ls -t "$solution_dir"/$stem.pkl 2>/dev/null | head -n1 || true)
  partition_json=$(ls -t "$solution_dir"/$stem.json 2>/dev/null | head -n1 || true)
  partition_meta=$(ls -t "$solution_dir"/$stem.meta.json 2>/dev/null | head -n1 || true)

  if [[ -z "$partition_pkl" ]]; then
    partition_pkl=$(ls -t "$solution_dir"/*assignment-mip.pkl 2>/dev/null | head -n1 || true)
  fi
  if [[ -z "$partition_json" ]]; then
    partition_json=$(ls -t "$solution_dir"/*assignment-mip.json 2>/dev/null | head -n1 || true)
  fi
  if [[ -z "$partition_meta" ]]; then
    partition_meta=$(ls -t "$solution_dir"/*assignment-mip.meta.json 2>/dev/null | head -n1 || true)
  fi

  printf '%s\n%s\n%s\n' "$partition_pkl" "$partition_json" "$partition_meta"
}

attach_mip_solver_stats() {
  local log_file="$1"
  local partition_json="$2"
  local partition_meta="$3"

  if [[ ! -f "$ATTACH_SOLVER_STATS_PY" ]]; then
    return 0
  fi
  if [[ ! -f "$log_file" ]]; then
    return 0
  fi
  if [[ -z "$partition_json" && -z "$partition_meta" ]]; then
    return 0
  fi

  local cmd=( "$PYTHON" "$ATTACH_SOLVER_STATS_PY" --log-file "$log_file" )
  if [[ -n "$partition_json" && -f "$partition_json" ]]; then
    cmd+=( --json-path "$partition_json" )
  fi
  if [[ -n "$partition_meta" && -f "$partition_meta" ]]; then
    cmd+=( --meta-path "$partition_meta" )
  fi
  "${cmd[@]}" >/dev/null 2>&1 || true
}

copy_versioned_mip_artifacts() {
  local partition_pkl="$1"
  if [[ -z "$RUN_TAG_ENV" || -z "$partition_pkl" || ! -f "$partition_pkl" ]]; then
    return 0
  fi

  local versioned_partition="${partition_pkl%.pkl}__run-${RUN_TAG_ENV}.pkl"
  cp -f "$partition_pkl" "$versioned_partition"

  local meta_src="${partition_pkl%.pkl}.meta.json"
  if [[ -f "$meta_src" ]]; then
    cp -f "$meta_src" "${versioned_partition%.pkl}.meta.json"
  fi

  local json_src="${partition_pkl%.pkl}.json"
  if [[ -f "$json_src" ]]; then
    cp -f "$json_src" "${versioned_partition%.pkl}.json"
  fi
}

for config in "${CONFIGS[@]}"; do
  config_base="$(basename "$config" .yaml)"
  if [[ -n "$RUN_TAG_ENV" ]]; then
    log_file="$OUTDIR/mip_eval_${config_base}__run-${RUN_TAG_ENV}.log"
  else
    log_file="$OUTDIR/mip_eval_${config_base}.log"
  fi
  tmp_cfg=""
  run_config="$config"
  config_start_sec=$SECONDS

  echo "---- [MIP] $config_base ----"
  if [[ "$FAST_MIP" =~ ^(1|true|yes|on)$ || -n "$RESULT_PREFIX_OVERRIDE" || -n "$OUTPUT_DIR_OVERRIDE" || -n "$SOLUTION_DIR_OVERRIDE" ]]; then
    tmp_cfg="$(mktemp "$OUTDIR/${config_base}.fast_mip.XXXXXX.yaml")"
    fast_mode_enabled=0
    if [[ "$FAST_MIP" =~ ^(1|true|yes|on)$ ]]; then
      fast_mode_enabled=1
    fi
    "$PYTHON" - <<'PY' "$config" "$tmp_cfg" "$fast_mode_enabled"
import os
import sys
from omegaconf import OmegaConf

src = sys.argv[1]
dst = sys.argv[2]
fast_mode_enabled = sys.argv[3] == "1"

def as_bool(v, default=False):
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default

def as_optional_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    value = float(s)
    return None if value <= 0 else value

def as_optional_int(v):
    numeric = as_optional_float(v)
    return None if numeric is None else int(numeric)

cfg = OmegaConf.load(src)
solver_tool = os.getenv("SOLVER_TOOL", "").strip()
if solver_tool:
    cfg["solver-tool"] = solver_tool
if fast_mode_enabled:
    mip = dict(cfg.get("mip", {}))
    mip["solve-mode"] = os.getenv("MIP_SOLVE_MODE", "exact")
    mip["sw-constraint-mode"] = os.getenv("MIP_SW_CONSTRAINT_MODE", "pairwise_topo")
    mip["use-reduced-sw-constraints"] = as_bool(os.getenv("MIP_USE_REDUCED_SW", "false"), False)
    mip["time-limit-sec"] = as_optional_float(os.getenv("MIP_TIME_LIMIT_SEC", "600"))
    mip["mip-gap"] = as_optional_float(os.getenv("MIP_GAP", "0"))
    mip["node-limit"] = as_optional_int(os.getenv("MIP_NODE_LIMIT", "0"))
    mip["accept-nonoptimal"] = as_bool(os.getenv("MIP_ACCEPT_NONOPTIMAL", "false"), False)
    mip["verbose"] = as_bool(os.getenv("MIP_VERBOSE", "false"), False)
    cfg["mip"] = mip

result_prefix = (os.getenv("HWSW_RESULT_PREFIX", "") or os.getenv("RESULT_PREFIX", "")).strip()
if result_prefix:
    cfg["result-file-prefix"] = result_prefix

output_dir = (os.getenv("HWSW_OUTPUT_DIR", "") or os.getenv("OUTPUT_DIR", "")).strip()
if output_dir:
    cfg["output-dir"] = output_dir

solution_dir = (os.getenv("HWSW_SOLUTION_DIR", "") or os.getenv("SOLUTION_DIR", "")).strip()
if solution_dir:
    cfg["solution-dir"] = solution_dir

OmegaConf.save(config=cfg, f=dst)
PY
    run_config="$tmp_cfg"
  fi

  result_prefix=$("$PYTHON" - <<'PY' "$run_config"
from omegaconf import OmegaConf
import sys
cfg = OmegaConf.load(sys.argv[1])
print(cfg.get('result-file-prefix', 'mip_solver'))
PY
)
  if [[ -n "$RESULT_CSV_ENV" ]]; then
    if [[ "$RESULT_CSV_ENV" = /* ]]; then
      out_csv="$RESULT_CSV_ENV"
    else
      out_csv="$OUTDIR/$RESULT_CSV_ENV"
    fi
  else
    out_csv="$OUTDIR/mip_${result_prefix}-result-summary-soda-graphs-config.csv"
  fi

  mapfile -t cfg_vals < <("$PYTHON" - <<'PY' "$run_config" "$ROOT"
from omegaconf import OmegaConf
from pathlib import Path
import sys
cfg = OmegaConf.load(sys.argv[1])
root = Path(sys.argv[2])
solution_dir = Path(cfg.get('solution-dir', 'makespan-opt-partitions'))
if not solution_dir.is_absolute():
    solution_dir = root / solution_dir
result_prefix = cfg.get('result-file-prefix', 'mip_solver')
print(str(solution_dir))
print(result_prefix)
PY
)
  solution_dir="${cfg_vals[0]}"
  result_prefix="${cfg_vals[1]}"

  mapfile -t cfg_key < <("$PYTHON" - <<'PY' "$run_config"
from omegaconf import OmegaConf
import sys
cfg = OmegaConf.load(sys.argv[1])
print(f"{float(cfg.get('area-constraint', 0.0)):.2f}")
print(f"{float(cfg.get('hw-scale-factor', 0.0)):.1f}")
print(f"{float(cfg.get('hw-scale-variance', 0.0)):.2f}")
print(str(cfg.get('seed', 42)))
PY
)
  area_key="${cfg_key[0]}"
  hw_key="${cfg_key[1]}"
  hwvar_key="${cfg_key[2]}"
  seed_key="${cfg_key[3]}"

  run_cmd=( "$PYTHON" "$MIP_EVAL_ENTRY" -c "$run_config" -t "$SOLVER_TOOL" )
  rc=0
  set +e
  if command -v timeout >/dev/null 2>&1 && [[ "$RUN_TIMEOUT_SEC" =~ ^[0-9]+$ ]] && (( RUN_TIMEOUT_SEC > 0 )); then
    timeout --signal=TERM --kill-after="${TIMEOUT_KILL_AFTER_SEC}s" "${RUN_TIMEOUT_SEC}s" "${run_cmd[@]}" >"$log_file" 2>&1
    rc=$?
  else
    "${run_cmd[@]}" >"$log_file" 2>&1
    rc=$?
  fi
  set -e
  config_elapsed_sec=$((SECONDS - config_start_sec))

  mapfile -t artifact_paths < <(resolve_mip_artifacts "$solution_dir" "$area_key" "$hw_key" "$hwvar_key" "$seed_key")
  partition_pkl="${artifact_paths[0]}"
  partition_json="${artifact_paths[1]}"
  partition_meta="${artifact_paths[2]}"
  attach_mip_solver_stats "$log_file" "$partition_json" "$partition_meta"
  copy_versioned_mip_artifacts "$partition_pkl"
  if (( rc == 0 )) && grep -q "SCIP Status[[:space:]]*: solving was interrupted \\[time limit reached\\]" "$log_file" 2>/dev/null; then
    rc=124
  fi

  if (( rc == 124 || rc == 137 )); then
    if [[ -n "$partition_pkl" || -n "$partition_json" || -n "$partition_meta" ]]; then
      echo "MIP hit a time limit for $config and wrote incumbent artifacts (see $log_file)"
    else
      echo "MIP hit a time limit for $config without writing artifacts (see $log_file)"
    fi
    append_mip_status_row "$config" "$run_config" "$out_csv" "$config_elapsed_sec" "time_limit_exceeded" "Time limit exceeded; no accepted exact MIP solution was recorded."
    [[ -n "$tmp_cfg" ]] && rm -f "$tmp_cfg"
    continue
  fi
  if (( rc != 0 )); then
    echo "MIP solver failed for $config (exit=$rc, see $log_file)"
    append_mip_status_row "$config" "$run_config" "$out_csv" "$config_elapsed_sec" "failed" "MIP solver failed before producing an accepted exact solution."
    [[ -n "$tmp_cfg" ]] && rm -f "$tmp_cfg"
    continue
  fi
  if [[ -z "$partition_pkl" && -z "$partition_json" ]]; then
    echo "No assignment-mip artifact found in $solution_dir (skipping CSV row)"
    [[ -n "$tmp_cfg" ]] && rm -f "$tmp_cfg"
    continue
  fi

  "$PYTHON" - <<'PY' "$config" "$partition_pkl" "$partition_json" "$out_csv" "$config_elapsed_sec"
import os
import pickle
import sys
import json
import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from omegaconf import OmegaConf
import numpy as np
import random

from meta_heuristic import TaskGraph
from meta_heuristic.partition_schedule_evaluator import evaluate_partition_lssp, synchronize_problem_with_config
from utils.partition_utils import ScheduleConstPartitionSolver

config_path = Path(sys.argv[1])
partition_pkl_arg = Path(sys.argv[2]) if sys.argv[2] else None
partition_json_arg = Path(sys.argv[3]) if sys.argv[3] else None
out_csv = Path(sys.argv[4])
runtime_sec = float(sys.argv[5])

cfg = OmegaConf.load(config_path)
seed = cfg.get('seed', 42)

task_graph = None
meta = {}
json_payload = {}


def build_taskgraph_like(graph, area_constraint: float):
    hardware_area = {n: float(graph.nodes[n].get('area_cost', 0.0)) for n in graph.nodes()}
    total_area = float(sum(hardware_area.values()))

    class _TaskGraphLike(SimpleNamespace):
        def violates(self, partition):
            if total_area <= 0:
                return 0
            used_area = sum(hardware_area[n] for n, a in partition.items() if int(a) == 1)
            return int((used_area / total_area) > float(area_constraint))

        def evaluate_partition_cost(self, partition):
            exec_cost = 0.0
            comm_cost = 0.0
            area_used = 0.0
            for node, placement in partition.items():
                if int(placement) == 1:
                    exec_cost += self.hardware_costs[node]
                    area_used += self.hardware_area[node]
                else:
                    exec_cost += self.software_costs[node]
            for (u, v), cost in self.communication_costs.items():
                if int(partition[u]) != int(partition[v]):
                    comm_cost += cost
            if total_area > 0 and (area_used / total_area) > float(area_constraint):
                return float(self.violation_cost)
            return float(exec_cost + comm_cost)

    return _TaskGraphLike(
        graph=graph,
        hardware_area=hardware_area,
        hardware_costs={n: float(graph.nodes[n].get('hardware_time', 0.0)) for n in graph.nodes()},
        software_costs={n: float(graph.nodes[n].get('software_time', 0.0)) for n in graph.nodes()},
        communication_costs={(u, v): float(graph.edges[u, v].get('communication_cost', 0.0)) for u, v in graph.edges()},
        area_constraint=float(area_constraint),
        total_area=total_area,
        violation_cost=1e9,
    )

# Prefer the TaskGraph pickle used by the MIP run (metadata saved alongside partition/json)
meta_path = None
for artifact_path in (partition_pkl_arg, partition_json_arg):
    if not artifact_path:
        continue
    if artifact_path.name.endswith('_assignment-mip.pkl'):
        candidate = artifact_path.with_name(artifact_path.name.replace('_assignment-mip.pkl', '_assignment-mip.meta.json'))
    elif artifact_path.name.endswith('_assignment-mip.json'):
        candidate = artifact_path.with_name(artifact_path.name.replace('_assignment-mip.json', '_assignment-mip.meta.json'))
    else:
        continue
    if candidate.exists():
        meta_path = candidate
        break
tg_pickle = None
if meta_path and meta_path.exists():
    try:
        meta = json.loads(meta_path.read_text())
        tg_pickle = meta.get('taskgraph_pickle_copy') or meta.get('taskgraph_pickle')
    except Exception:
        tg_pickle = None

if partition_json_arg and partition_json_arg.exists():
    try:
        json_payload = json.loads(partition_json_arg.read_text())
    except Exception:
        json_payload = {}

cfg_tg_pickle = cfg.get('taskgraph-pickle', None)
if tg_pickle and not Path(tg_pickle).exists():
    tg_pickle = None
if not tg_pickle and cfg_tg_pickle:
    tg_pickle = cfg_tg_pickle

tg_pickle_used = None
if tg_pickle and Path(tg_pickle).exists():
    with open(tg_pickle, 'rb') as f:
        task_graph = pickle.load(f)
    task_graph = synchronize_problem_with_config(task_graph, cfg)
    graph = task_graph.graph
    tg_pickle_used = str(tg_pickle)
else:
    random.seed(seed)
    np.random.seed(seed)
    solver = ScheduleConstPartitionSolver()
    graph = solver.load_pydot_graph(
        cfg['graph-file'],
        k=cfg['hw-scale-factor'],
        l=cfg['hw-scale-variance'],
        mu=cfg['comm-scale-factor'],
        A_max=100,
    )

partition = None
if partition_pkl_arg and partition_pkl_arg.exists():
    with open(partition_pkl_arg, 'rb') as f:
        partition = pickle.load(f)
else:
    partition_payload = json_payload.get('partition_assignment', None)
    if isinstance(partition_payload, list) and partition_payload and isinstance(partition_payload[0], dict):
        partition = {str(k): int(v) for k, v in partition_payload[0].items()}
    elif isinstance(partition_payload, dict):
        partition = {str(k): int(v) for k, v in partition_payload.items()}

if partition is None:
    raise RuntimeError("No partition assignment found in MIP artifacts")

# Ensure partition covers all nodes (fill missing with software=0)
missing = [n for n in graph.nodes() if n not in partition]
if missing and cfg_tg_pickle and tg_pickle_used and (str(cfg_tg_pickle) != tg_pickle_used):
    # Try fallback to config pickle if meta copy mismatches partition
    try:
        with open(cfg_tg_pickle, 'rb') as f:
            task_graph = pickle.load(f)
        task_graph = synchronize_problem_with_config(task_graph, cfg)
        graph = task_graph.graph
        tg_pickle_used = str(cfg_tg_pickle)
        missing = [n for n in graph.nodes() if n not in partition]
    except Exception:
        pass
if missing:
    for n in missing:
        partition[n] = 0
    print(f"[warn] Filled {len(missing)} missing nodes with software=0: {missing[:5]}")

if task_graph is None:
    task_graph = build_taskgraph_like(graph, cfg['area-constraint'])

naive_lb = sum(min(task_graph.software_costs[n], task_graph.hardware_costs[n]) for n in graph.nodes())
lssp_result = evaluate_partition_lssp(task_graph, partition)
makespan = float(lssp_result['makespan'])
partition_cost = float(task_graph.evaluate_partition_cost(partition))
solver_status = meta.get('solver_status') or json_payload.get('status', 'optimal')
model_makespan = float(meta.get('model_makespan', json_payload.get('makespan', np.nan)))
lp_makespan = float(meta.get('lp_makespan', json_payload.get('lp_makespan', np.nan)))
lssp_makespan = float(meta.get('final_lssp_makespan', json_payload.get('final_lssp_makespan', makespan)))
optimization_time_sec = float(meta.get('solve_time_sec', json_payload.get('solve_time_sec', runtime_sec)))
postprocess_time_sec = float(meta.get('postprocess_time_sec', json_payload.get('postprocess_time_sec', np.nan)))
algorithm_total_time_sec = float(meta.get('algorithm_total_time_sec', json_payload.get('algorithm_total_time_sec', np.nan)))
if not np.isfinite(postprocess_time_sec):
    if np.isfinite(algorithm_total_time_sec):
        postprocess_time_sec = max(0.0, float(algorithm_total_time_sec) - float(optimization_time_sec))
    else:
        postprocess_time_sec = max(0.0, float(runtime_sec) - float(optimization_time_sec))
if not np.isfinite(algorithm_total_time_sec):
    algorithm_total_time_sec = float(optimization_time_sec) + float(postprocess_time_sec)

is_valid = bool(lssp_result.get('is_valid', True))
was_repaired = bool(lssp_result.get('was_repaired', False))
repair_strategy = str(lssp_result.get('repair_strategy', 'benefit_per_area'))
area_used = float(lssp_result.get('execution_summary', {}).get('area_used', 0.0))
area_budget = float(lssp_result.get('execution_summary', {}).get('area_budget', 0.0))

if is_valid and not was_repaired:
    validity_note = "Valid solution; no area repair needed."
elif is_valid and was_repaired:
    validity_note = (
        "Invalid before post-processing; repaired with "
        f"{repair_strategy} greedy fixing to satisfy the area constraint."
    )
else:
    validity_note = (
        "Invalid solution after post-processing; area constraint still violated "
        "and reported makespan reflects the violation penalty."
    )

base_data = {
    'SimTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'RunTag': os.getenv('HWSW_RUN_TAG', ''),
    'Config': config_path.stem,
    'GraphName': cfg['graph-file'],
    'N': len(graph.nodes()),
    'HW_Scale_Factor': cfg['hw-scale-factor'],
    'HW_Scale_Var': cfg['hw-scale-variance'],
    'Comm_Scale_Var': cfg['comm-scale-factor'],
    'Area_Percentage': cfg['area-constraint'],
    'Seed': cfg.get('seed', 42),
    'LB_Naive': naive_lb,
}

method = 'mip'
row = {
    **base_data,
    f'{method}_status': solver_status,
    f'{method}_model_makespan': model_makespan,
    f'{method}_lp_makespan': lp_makespan,
    f'{method}_lssp_makespan': lssp_makespan,
    f'{method}_opt_cost': makespan,
    f'{method}_opt_ratio': (makespan / naive_lb) if naive_lb > 0 else 0,
    f'{method}_partition_cost': partition_cost,
    f'{method}_bb': 'milp_eval',
    f'{method}_makespan': makespan,
    f'{method}_time': runtime_sec,
    f'{method}_optimization_time_sec': optimization_time_sec,
    f'{method}_postprocess_time_sec': postprocess_time_sec,
    f'{method}_total_runtime_sec': algorithm_total_time_sec,
    f'{method}_solution_valid': is_valid,
    f'{method}_initial_solution_valid': (not was_repaired),
    f'{method}_was_repaired': was_repaired,
    f'{method}_num_repaired_nodes': len(lssp_result.get('repaired_nodes', [])),
    f'{method}_repair_strategy': repair_strategy,
    f'{method}_area_used': area_used,
    f'{method}_area_budget': area_budget,
    f'{method}_validity_note': validity_note,
    f'{method}_timeout_note': "",
}

out_csv.parent.mkdir(parents=True, exist_ok=True)
result_df = pd.DataFrame([row])

if out_csv.exists():
    with out_csv.open(newline='') as handle:
        rows = list(csv.reader(handle))

    if not rows:
        result_df.to_csv(out_csv, mode='w', index=False, header=True)
    else:
        existing_cols = list(rows[0])
        ordered_cols = existing_cols + [c for c in result_df.columns if c not in existing_cols]

        row_dicts = []
        for raw in rows[1:]:
            padded = list(raw) + [""] * max(0, len(ordered_cols) - len(raw))
            row_dicts.append(dict(zip(ordered_cols, padded[:len(ordered_cols)])))

        if len(ordered_cols) != len(existing_cols):
            existing_df = pd.DataFrame(row_dicts, columns=ordered_cols)
            result_df = result_df.reindex(columns=ordered_cols)
            combined_df = pd.concat([existing_df, result_df], ignore_index=True)
            combined_df.to_csv(out_csv, mode='w', index=False, header=True)
        else:
            result_df = result_df.reindex(columns=ordered_cols)
            result_df.to_csv(out_csv, mode='a', index=False, header=False)
else:
    result_df.to_csv(out_csv, mode='a', index=False, header=True)
PY

  [[ -n "$tmp_cfg" ]] && rm -f "$tmp_cfg"
  echo "Completed $config_base in ${config_elapsed_sec}s"

done

batch_elapsed_sec=$((SECONDS - batch_start_sec))
echo "MIP batch complete in ${batch_elapsed_sec}s. CSVs are in $OUTDIR"
