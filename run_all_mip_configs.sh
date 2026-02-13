#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$ROOT/configs"
OUTDIR="$ROOT/outputs/logs"
mkdir -p "$OUTDIR"

export PYTHONNOUSERSITE=1

PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
SOLVER_TOOL="${SOLVER_TOOL:-cvxpy}"
CONFIG_GLOB="${CONFIG_GLOB:-$CONFIG_DIR/config_mkspan_default_gnn.yaml}"

cd "$ROOT"

mapfile -t CONFIGS < <(ls $CONFIG_GLOB 2>/dev/null | sort || true)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No config files matched: $CONFIG_GLOB"
  exit 1
fi

echo "Running MIP solver (${SOLVER_TOOL}) on ${#CONFIGS[@]} configs"

for config in "${CONFIGS[@]}"; do
  config_base="$(basename "$config" .yaml)"
  log_file="$OUTDIR/mip_eval_${config_base}.log"

  echo "---- [MIP] $config_base ----"
  "$PYTHON" milp_eval.py -c "$config" -t "$SOLVER_TOOL" >"$log_file" 2>&1 || {
    echo "MIP solver failed for $config (see $log_file)"
    continue
  }

  mapfile -t cfg_vals < <("$PYTHON" - <<'PY' "$config" "$ROOT"
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

  partition_pkl=$(ls -t "$solution_dir"/*assignment-mip.pkl 2>/dev/null | head -n1 || true)
  if [[ -z "$partition_pkl" ]]; then
    echo "No assignment-mip.pkl found in $solution_dir (skipping CSV row)"
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

import pandas as pd
from omegaconf import OmegaConf
import numpy as np
import random

from meta_heuristic import TaskGraph
from utils.partition_utils import ScheduleConstPartitionSolver
from utils.scheduler_utils import compute_dag_execution_time, compute_dag_makespan

config_path = Path(sys.argv[1])
partition_path = Path(sys.argv[2])
out_csv = Path(sys.argv[3])

cfg = OmegaConf.load(config_path)
seed = cfg.get('seed', 42)

task_graph = None

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
        graph = task_graph.graph
        tg_pickle_used = str(cfg_tg_pickle)
        missing = [n for n in graph.nodes() if n not in partition]
    except Exception:
        pass
if missing:
    for n in missing:
        partition[n] = 0
    print(f"[warn] Filled {len(missing)} missing nodes with software=0: {missing[:5]}")

if task_graph is not None:
    naive_lb = sum(min(task_graph.software_costs[n], task_graph.hardware_costs[n]) for n in graph.nodes())
    if task_graph.violates(partition):
        makespan = task_graph.violation_cost
    else:
        lp_assignment = [1 - partition[n] for n in task_graph.rounak_graph]
        makespan, _ = compute_dag_makespan(task_graph.rounak_graph, lp_assignment)
    partition_cost = task_graph.evaluate_partition_cost(partition)
else:
    node_sw = {n: graph.nodes[n]['software_time'] for n in graph.nodes()}
    node_hw = {n: graph.nodes[n]['hardware_time'] for n in graph.nodes()}
    node_area = {n: graph.nodes[n]['area_cost'] for n in graph.nodes()}
    edge_comm = {(u, v): graph.edges[u, v].get('communication_cost', 0) for u, v in graph.edges()}
    total_area = sum(node_area.values())

    exec_cost = 0.0
    area_used = 0.0
    for n, placement in partition.items():
        if placement <= 0.5:
            exec_cost += node_sw[n]
        else:
            exec_cost += node_hw[n]
            area_used += node_area[n]

    naive_lb = sum(min(node_sw[n], node_hw[n]) for n in graph.nodes())
    if total_area > 0 and (area_used / total_area) > cfg['area-constraint']:
        makespan = 1e9
    else:
        lp_assignment = [1 - partition[n] for n in graph.nodes()]
        makespan, _ = compute_dag_makespan(graph, lp_assignment)
    comm_cost = 0.0
    for (u, v), c in edge_comm.items():
        if (partition[u] <= 0.5) != (partition[v] <= 0.5):
            comm_cost += c

    if total_area > 0 and (area_used / total_area) > cfg['area-constraint']:
        partition_cost = 1e9
    else:
        partition_cost = exec_cost + comm_cost

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

done

echo "MIP batch complete. CSVs are in $OUTDIR"
