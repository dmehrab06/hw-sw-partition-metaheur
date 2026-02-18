#!/usr/bin/env python
"""
Generate the paper's Fig.3 task graph (11 tasks) as:
1) DOT topology file
2) TaskGraph pickle with fixed node/edge attributes
3) Optional preview image
"""

import argparse
import os
import pickle
import sys

import networkx as nx
import pydot


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meta_heuristic.task_graph import TaskGraph


# (software_time, hardware_time, hardware_area)
FIG3_NODE_ATTRS = {
    "T1": (4.0, 2.0, 4.0),
    "T2": (5.0, 3.0, 2.0),
    "T3": (4.0, 1.0, 4.0),
    "T4": (5.0, 2.0, 7.0),
    "T5": (3.0, 1.0, 5.0),
    "T6": (16.0, 10.0, 3.0),
    "T7": (3.0, 2.0, 4.0),
    "T8": (4.0, 2.0, 2.0),
    "T9": (14.0, 7.0, 3.0),
    "T10": (5.0, 2.0, 5.0),
    "T11": (20.0, 9.0, 5.0),
}

# (u, v, communication_cost)
FIG3_EDGES = [
    ("T1", "T4", 1.0),
    ("T2", "T4", 2.0),
    ("T2", "T5", 2.0),
    ("T3", "T5", 1.0),
    ("T4", "T6", 4.0),
    ("T4", "T7", 1.0),
    ("T5", "T8", 1.0),
    ("T5", "T9", 2.0),
    ("T6", "T10", 3.0),
    ("T7", "T10", 1.0),
    ("T8", "T11", 1.0),
    ("T9", "T11", 2.0),
]

# Fixed layout for preview plot
FIG3_POS = {
    "T1": (-2.0, 2.0),
    "T2": (0.0, 2.0),
    "T3": (2.0, 2.0),
    "T4": (-1.0, 1.0),
    "T5": (1.0, 1.0),
    "T6": (-2.0, 0.0),
    "T7": (-1.0, 0.0),
    "T8": (1.0, 0.0),
    "T9": (2.0, 0.0),
    "T10": (-1.5, -1.0),
    "T11": (1.5, -1.0),
}


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for node in FIG3_NODE_ATTRS:
        g.add_node(node)
    for u, v, c in FIG3_EDGES:
        g.add_edge(u, v, communication_cost=float(c))
    return g


def write_dot(dot_out: str) -> None:
    os.makedirs(os.path.dirname(dot_out), exist_ok=True)

    dot = pydot.Dot(graph_type="digraph")
    dot.set_rankdir("TB")

    for node, (ts, th, area) in FIG3_NODE_ATTRS.items():
        label = f"{node}\\n({int(ts)},{int(th)},{int(area)})"
        dot.add_node(pydot.Node(node, label=label, shape="circle"))

    for u, v, c in FIG3_EDGES:
        dot.add_edge(pydot.Edge(u, v, label=str(int(c))))

    dot.write_raw(dot_out)


def write_taskgraph_pickle(pkl_out: str, area_constraint: float) -> None:
    os.makedirs(os.path.dirname(pkl_out), exist_ok=True)

    tg = TaskGraph(area_constraint=area_constraint)
    g = build_graph()

    tg.graph = g
    tg.rounak_graph = g

    tg.software_costs = {n: FIG3_NODE_ATTRS[n][0] for n in g.nodes()}
    tg.hardware_costs = {n: FIG3_NODE_ATTRS[n][1] for n in g.nodes()}
    tg.hardware_area = {n: FIG3_NODE_ATTRS[n][2] for n in g.nodes()}
    tg.communication_costs = {(u, v): float(c) for u, v, c in FIG3_EDGES}

    tg.node_to_num = {node: i for i, node in enumerate(g.nodes())}
    tg.num_to_node = {i: node for node, i in tg.node_to_num.items()}
    tg.total_area = float(sum(tg.hardware_area.values()))

    nx.set_node_attributes(tg.rounak_graph, tg.hardware_area, "area_cost")
    nx.set_node_attributes(tg.rounak_graph, tg.hardware_costs, "hardware_time")
    nx.set_node_attributes(tg.rounak_graph, tg.software_costs, "software_time")
    nx.set_edge_attributes(tg.rounak_graph, tg.communication_costs, "communication_cost")

    with open(pkl_out, "wb") as f:
        pickle.dump(tg, f)


def write_preview(preview_out: str) -> None:
    os.makedirs(os.path.dirname(preview_out), exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping preview image")
        return

    g = build_graph()
    plt.figure(figsize=(10, 6))

    nx.draw_networkx_nodes(g, FIG3_POS, node_size=1400, node_color="#f7f7f7", edgecolors="#333333")
    nx.draw_networkx_labels(g, FIG3_POS, font_size=11, font_weight="bold")
    nx.draw_networkx_edges(g, FIG3_POS, arrows=True, arrowsize=20, width=1.8)

    node_labels = {
        n: f"({int(FIG3_NODE_ATTRS[n][0])},{int(FIG3_NODE_ATTRS[n][1])},{int(FIG3_NODE_ATTRS[n][2])})"
        for n in g.nodes()
    }
    for node, (x, y) in FIG3_POS.items():
        plt.text(x - 0.68, y + 0.32, node_labels[node], fontsize=9, color="#444444")

    edge_labels = {(u, v): int(c) for u, v, c in FIG3_EDGES}
    nx.draw_networkx_edge_labels(g, FIG3_POS, edge_labels=edge_labels, font_size=10)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(preview_out, dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dot-out",
        default=os.path.join(ROOT, "inputs/task_graph_topology/custom/paper_fig3_11node.dot"),
    )
    ap.add_argument(
        "--pickle-out",
        default=os.path.join(
            ROOT,
            "inputs/task_graph_complete/taskgraph-paper_fig3_11node-instance-config-config_fig3_taskgraph_gnn.pkl",
        ),
    )
    ap.add_argument(
        "--preview-out",
        default=os.path.join(ROOT, "outputs/fig3_schedule/fig3_taskgraph_reference.png"),
    )
    ap.add_argument("--area-constraint", type=float, default=0.5)
    args = ap.parse_args()

    write_dot(args.dot_out)
    write_taskgraph_pickle(args.pickle_out, area_constraint=float(args.area_constraint))
    write_preview(args.preview_out)

    print("Generated:")
    print(f"  DOT    : {args.dot_out}")
    print(f"  Pickle : {args.pickle_out}")
    print(f"  Preview: {args.preview_out}")


if __name__ == "__main__":
    main()
