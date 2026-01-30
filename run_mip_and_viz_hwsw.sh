#!/bin/bash
set -euo pipefail

# Env
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate combopt
export PYTHONNOUSERSITE=1
PYTHON="/people/dass304/.conda/envs/combopt/bin/python"

ROOT="/people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur"
cd "$ROOT"

CONFIG="$ROOT/configs/config_mkspan_default_gnn.yaml"
DOT="inputs/task_graph_topology/soda-benchmark-graphs/pytorch-graphs/squeeze_net_tosa.dot"
OUTDIR="Figs/hwsw"
mkdir -p "$OUTDIR"

echo "[mip] Running milp_eval with $CONFIG"
$PYTHON milp_eval.py -c "$CONFIG" -t cvxpy

echo "[SID] milp_eval completed."

# Find latest MIP partition pickle produced by milp_eval
PARTITION_PKL=$(ls -t makespan-opt-partitions/*assignment-mip.pkl 2>/dev/null | head -n1)
if [[ -z "$PARTITION_PKL" ]]; then
  echo "Partition pickle not found in makespan-opt-partitions/. Exiting."
  exit 1
fi

OUTPNG="$OUTDIR/partition_overlay.png"
echo "[viz] Drawing partition -> $OUTPNG"
$PYTHON viz_hwsw_partition.py \
  --dot "$DOT" \
  --partition "$PARTITION_PKL" \
  --out "$OUTPNG"

echo "Done."
