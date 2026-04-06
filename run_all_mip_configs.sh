#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$ROOT/configs"
OUTDIR="${OUTDIR:-$ROOT/outputs/logs}"
mkdir -p "$OUTDIR"

export PYTHONNOUSERSITE=1

PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
SOLVER_TOOL="${SOLVER_TOOL:-cvxpy}"
MIP_EVAL_PY="${MIP_EVAL_PY:-mip_eval.py}"
CONFIG_GLOB="${CONFIG_GLOB:-$CONFIG_DIR/config_mkspan_default_gnn.yaml}"

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

# Fast-mode defaults (can be overridden via env).
# Set FAST_MIP=0 to run configs exactly as-is.
FAST_MIP="${FAST_MIP:-1}"
MIP_SOLVE_MODE="${MIP_SOLVE_MODE:-hybrid}"
MIP_SW_CONSTRAINT_MODE="${MIP_SW_CONSTRAINT_MODE:-adjacent}"
MIP_USE_REDUCED_SW="${MIP_USE_REDUCED_SW:-true}"
MIP_TIME_LIMIT_SEC="${MIP_TIME_LIMIT_SEC:-30}"
MIP_GAP="${MIP_GAP:-0.15}"
MIP_NODE_LIMIT="${MIP_NODE_LIMIT:-20000}"
MIP_ACCEPT_NONOPTIMAL="${MIP_ACCEPT_NONOPTIMAL:-true}"
MIP_VERBOSE="${MIP_VERBOSE:-false}"

# Extra wall-clock guard for each config run.
# 0 disables external timeout.
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-120}"
TIMEOUT_KILL_AFTER_SEC="${TIMEOUT_KILL_AFTER_SEC:-15}"

cd "$ROOT"

mapfile -t CONFIGS < <(ls $CONFIG_GLOB 2>/dev/null | sort || true)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No config files matched: $CONFIG_GLOB"
  exit 1
fi

echo "Running MIP solver (${SOLVER_TOOL}) on ${#CONFIGS[@]} configs"
echo "FAST_MIP=$FAST_MIP, RUN_TIMEOUT_SEC=$RUN_TIMEOUT_SEC"
echo "MIP evaluator: $(basename "$MIP_EVAL_ENTRY")"
if [[ "$FAST_MIP" =~ ^(1|true|yes|on)$ ]]; then
  echo "Fast MIP settings: mode=$MIP_SOLVE_MODE, sw=$MIP_SW_CONSTRAINT_MODE, tlimit=${MIP_TIME_LIMIT_SEC}s, gap=$MIP_GAP, nodes=$MIP_NODE_LIMIT"
fi

for config in "${CONFIGS[@]}"; do
  config_base="$(basename "$config" .yaml)"
  log_file="$OUTDIR/mip_eval_${config_base}.log"
  tmp_cfg=""
  run_config="$config"

  echo "---- [MIP] $config_base ----"
  if [[ "$FAST_MIP" =~ ^(1|true|yes|on)$ ]]; then
    tmp_cfg="$(mktemp "$OUTDIR/${config_base}.fast_mip.XXXXXX.yaml")"
    "$PYTHON" - <<'PY' "$config" "$tmp_cfg"
import os
import sys
from omegaconf import OmegaConf

src = sys.argv[1]
dst = sys.argv[2]

def as_bool(v, default=False):
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default

cfg = OmegaConf.load(src)
mip = dict(cfg.get("mip", {}))

mip["solve-mode"] = os.getenv("MIP_SOLVE_MODE", "hybrid")
mip["sw-constraint-mode"] = os.getenv("MIP_SW_CONSTRAINT_MODE", "adjacent")
mip["use-reduced-sw-constraints"] = as_bool(os.getenv("MIP_USE_REDUCED_SW", "true"), True)
mip["time-limit-sec"] = float(os.getenv("MIP_TIME_LIMIT_SEC", "30"))
mip["mip-gap"] = float(os.getenv("MIP_GAP", "0.15"))
mip["node-limit"] = int(float(os.getenv("MIP_NODE_LIMIT", "20000")))
mip["accept-nonoptimal"] = as_bool(os.getenv("MIP_ACCEPT_NONOPTIMAL", "true"), True)
mip["verbose"] = as_bool(os.getenv("MIP_VERBOSE", "false"), False)

cfg["mip"] = mip
OmegaConf.save(config=cfg, f=dst)
PY
    run_config="$tmp_cfg"
  fi

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

  if (( rc == 124 || rc == 137 )); then
    echo "MIP timed out for $config after ${RUN_TIMEOUT_SEC}s (see $log_file)"
    [[ -n "$tmp_cfg" ]] && rm -f "$tmp_cfg"
    continue
  fi
  if (( rc != 0 )); then
    echo "MIP solver failed for $config (exit=$rc, see $log_file)"
    [[ -n "$tmp_cfg" ]] && rm -f "$tmp_cfg"
    continue
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

  mapfile -t cfg_key < <("$PYTHON" - <<'PY' "$config"
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

  partition_pkl=$(ls -t "$solution_dir"/*area-"$area_key"_hwscale-"$hw_key"_hwvar-"$hwvar_key"_seed-"$seed_key"_assignment-mip.pkl 2>/dev/null | head -n1 || true)
  if [[ -z "$partition_pkl" ]]; then
    partition_pkl=$(ls -t "$solution_dir"/*assignment-mip.pkl 2>/dev/null | head -n1 || true)
  fi
  if [[ -z "$partition_pkl" ]]; then
    echo "No assignment-mip.pkl found in $solution_dir (skipping CSV row)"
    [[ -n "$tmp_cfg" ]] && rm -f "$tmp_cfg"
    continue
  fi

  out_csv="$OUTDIR/mip_${result_prefix}-result-summary-soda-graphs-config.csv"

  "$PYTHON" - <<'PY' "$config" "$partition_pkl" "$out_csv"
import os
import pickle
import sys
import json
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
partition_path = Path(sys.argv[2])
out_csv = Path(sys.argv[3])

cfg = OmegaConf.load(config_path)
seed = cfg.get('seed', 42)

task_graph = None


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

# Prefer the TaskGraph pickle used by the MIP run (metadata saved alongside partition)
meta_path = partition_path.with_name(partition_path.name.replace('_assignment-mip.pkl', '_assignment-mip.meta.json'))
tg_pickle = None
if meta_path.exists():
    try:
        meta = json.loads(meta_path.read_text())
        tg_pickle = meta.get('taskgraph_pickle_copy') or meta.get('taskgraph_pickle')
    except Exception:
        tg_pickle = None

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

with open(partition_path, 'rb') as f:
    partition = pickle.load(f)

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
    f'{method}_opt_cost': makespan,
    f'{method}_opt_ratio': (makespan / naive_lb) if naive_lb > 0 else 0,
    f'{method}_partition_cost': partition_cost,
    f'{method}_bb': 'milp_eval',
    f'{method}_makespan': makespan,
    f'{method}_time': 0.0,
    f'{method}_solution_valid': is_valid,
    f'{method}_initial_solution_valid': (not was_repaired),
    f'{method}_was_repaired': was_repaired,
    f'{method}_num_repaired_nodes': len(lssp_result.get('repaired_nodes', [])),
    f'{method}_repair_strategy': repair_strategy,
    f'{method}_area_used': area_used,
    f'{method}_area_budget': area_budget,
    f'{method}_validity_note': validity_note,
}

out_csv.parent.mkdir(parents=True, exist_ok=True)
result_df = pd.DataFrame([row])

if out_csv.exists():
    existing_cols = pd.read_csv(out_csv, nrows=0).columns.tolist()
    ordered_cols = existing_cols + [c for c in result_df.columns if c not in existing_cols]
    result_df = result_df.reindex(columns=ordered_cols)
    result_df.to_csv(out_csv, mode='a', index=False, header=False)
else:
    result_df.to_csv(out_csv, mode='a', index=False, header=True)
PY

  [[ -n "$tmp_cfg" ]] && rm -f "$tmp_cfg"

done

echo "MIP batch complete. CSVs are in $OUTDIR"
