#!/usr/bin/env python
"""
Visualize a HW/SW partition assignment using HWSW_solver_test utilities.

Loads a DOT task graph, a partition pickle (node -> 0/1), and saves a PNG with
nodes colored by assignment (SW=blue, HW=red) and edges shown. Uses
visualization_utils (imported from HWSW_solver_test) to keep dependency usage in
line with the existing codebase.
"""

import argparse
import os
import pickle
import pydot
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pull in compute_dag_makespan via visualization_utils to satisfy dependency use
from HWSW_solver_test.utils import visualization_utils as vu


def load_dot_graph(dot_path: str) -> nx.DiGraph:
    graphs = pydot.graph_from_dot_file(dot_path)
    if not graphs:
        raise FileNotFoundError(f"Failed to load DOT graph: {dot_path}")
    P = graphs[0]
    G = nx.DiGraph()
    for node in P.get_nodes():
        name = node.get_name().strip('"')
        if name in {"node", "graph", "edge"}:
            continue
        G.add_node(name)
    for edge in P.get_edges():
        G.add_edge(edge.get_source().strip('"'), edge.get_destination().strip('"'))
    return G


def load_partition(pkl_path: str):
    with open(pkl_path, "rb") as f:
        part = pickle.load(f)
    return part


def color_map(partition, nodes):
    colors = []
    for n in nodes:
        val = partition.get(n, 0)
        colors.append("lightcoral" if val == 1 else "skyblue")
    return colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dot", required=True, help="Path to DOT task graph")
    ap.add_argument("--partition", required=True, help="Partition pickle (node->0/1)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    args = ap.parse_args()

    G = load_dot_graph(args.dot)
    part = load_partition(args.partition)

    # Attempt layout with graphviz, fallback to spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=0)

    node_order = list(G.nodes())
    node_colors = color_map(part, node_order)

    plt.figure(figsize=(12, 9))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=650,
            font_size=8, edge_color="#555")

    # Evaluate makespan if possible (best effort)
    try:
        assignment = [1 - part[n] for n in G.nodes]  # solver uses 1 for HW
        makespan, _ = vu.compute_dag_makespan(G, assignment)
        title_ms = f"Makespan ≈ {makespan:.2f}"
    except Exception:
        title_ms = ""

    plt.title(f"HW/SW Partition" + (f" | {title_ms}" if title_ms else ""))
    plt.axis("off")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

