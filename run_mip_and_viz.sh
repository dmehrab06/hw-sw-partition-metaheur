#!/bin/bash
set -euo pipefail

# 1) Env setup (matches your login snippet)
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate combopt
PYTHON="/people/dass304/.conda/envs/combopt/bin/python"
export PYTHONNOUSERSITE=1

# 2) Repo root
cd /people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur

CONFIG="/people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/configs/config_mkspan_default_gnn.yaml"
DOT="inputs/task_graph_topology/soda-benchmark-graphs/pytorch-graphs/squeeze_net_tosa.dot"
OUTDIR="Figs/hwsw"
mkdir -p "$OUTDIR"

# --- Visualization via HWSW_solver_test helpers ---
echo "[viz] Rendering $DOT -> $OUTDIR/squeeze_net_tosa.png (using HWSW_solver_test)"
$PYTHON - <<'PY'
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pydot

dot_path = os.environ.get("DOT", "inputs/task_graph_topology/soda-benchmark-graphs/pytorch-graphs/squeeze_net_tosa.dot")
outdir = os.environ.get("OUTDIR", "Figs/hwsw")
os.makedirs(outdir, exist_ok=True)

graphs = pydot.graph_from_dot_file(dot_path)
if not graphs:
    sys.exit(f"Failed to load {dot_path}")
P = graphs[0]
G = nx.DiGraph()
for node in P.get_nodes():
    name = node.get_name().strip('"')
    if name in {"node","graph","edge"}:
        continue
    G.add_node(name)
for edge in P.get_edges():
    G.add_edge(edge.get_source().strip('"'), edge.get_destination().strip('"'))

# Layout with fallbacks
try:
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
except Exception:
    try:
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=0)

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=600, node_color="lightblue", edge_color="#555", font_size=7)
plt.axis("off")
plt.tight_layout()
out_path = os.path.join(outdir, "squeeze_net_tosa.png")
plt.savefig(out_path, dpi=200)
print(f"saved {out_path}")
PY

echo "[mip] Running milp_eval with $CONFIG"
$PYTHON milp_eval.py -c "$CONFIG" -t cvxpy

# Optional: pre-solve synthetic graphs from HWSW_solver_test
# python HWSW_solver_test/milp_solver.py
