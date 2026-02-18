#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
CFG="${CFG:-$ROOT/configs/config_fig3_taskgraph_gnn.yaml}"

export PYTHONNOUSERSITE=1

echo "[1/3] Generating Fig.3 graph + TaskGraph pickle"
"$PYTHON" "$ROOT/tools/generate_fig3_taskgraph.py"

echo "[2/3] Running diff_gnn and diff_gnn_order"
HWSW_METHODS="diff_gnn,diff_gnn_order" "$PYTHON" "$ROOT/gnn_main.py" -c "$CFG"

echo "[3/3] Creating schedule visualizations"
"$PYTHON" "$ROOT/tools/visualize_schedule_from_partitions.py" \
  --config "$CFG" \
  --methods "diff_gnn,diff_gnn_order"

echo "Done. Check $ROOT/outputs/fig3_schedule/"
