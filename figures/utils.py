import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def _safe_filename(s: str) -> str:
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^\w\-_\.]', '', s)
    return s[:220]

def _config_basename(config):
    if not config:
        return f"taskgraph_{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        parts = []
        # prefer human-readable fields if present
        if 'graph-file' in config:
            parts.append(Path(config['graph-file']).stem)
        if 'config' in config:
            parts.append(Path(config['config']).stem)
        for k in ('area-constraint','hw-scale-factor','hw-scale-variance','comm-scale-factor','seed'):
            if k in config:
                parts.append(f"{k.replace('_','')}-{config[k]}")
        if not parts:
            # fallback: full dict string (trimmed)
            parts = [str(config)[:180]]
        return "_".join(map(str, parts))
    except Exception:
        return f"taskgraph_{time.strftime('%Y%m%d-%H%M%S')}"

def visualize_task_graph(config=None, task_graph=None, out_dir='figures', fig_name=None, figsize=(40,32), show=False):
    """
    Visualize a TaskGraph instance and save as a PDF.

    Call signature matches usage in gnn_main.py:
      visualize_task_graph(config=config, task_graph=TG, out_dir='figures', fig_name='initial_task_graph')

    Args:
        config (dict or None): configuration/settings used to create the graph (used to form filename)
        task_graph: TaskGraph instance (expects .graph, .software_costs, .hardware_costs,
                    .hardware_area, .communication_costs, .total_area, .area_constraint)
        out_dir (str): output directory (will be created)
        fig_name (str or None): base filename (without extension). If provided it's used as prefix.
        figsize (tuple): matplotlib figure size
        show (bool): if True, show the figure interactively
    Returns:
        path to saved PDF file
    """

    print("Generating task graph visualization...")

    if task_graph is None or getattr(task_graph, 'graph', None) is None:
        raise ValueError("task_graph with a valid .graph is required")

    G = task_graph.graph

    # determine partition: try greedy heuristic on TaskGraph, else default all-software
    partition = None
    try:
        # some implementations return (cost, solution)
        maybe = getattr(task_graph, 'greedy_heur', None)
        if callable(maybe):
            res = maybe()
            if isinstance(res, tuple) and len(res) >= 2:
                partition = res[1]
            elif isinstance(res, dict):
                partition = res
    except Exception:
        partition = None

    if partition is None:
        # default: all software (0)
        partition = {n: 0 for n in G.nodes()}

    # normalize partition values to 0/1
    normalized_partition = {}
    for n, v in partition.items():
        try:
            normalized_partition[n] = 1 if float(v) > 0.5 else 0
        except Exception:
            try:
                normalized_partition[n] = int(v)
            except Exception:
                normalized_partition[n] = 0

    sw = getattr(task_graph, 'software_costs', {}) or {}
    hw = getattr(task_graph, 'hardware_costs', {}) or {}
    area = getattr(task_graph, 'hardware_area', {}) or {}
    comm = getattr(task_graph, 'communication_costs', {}) or {}

    # Build labels
    labels = {}
    for n in G.nodes():
        parts = [str(n)]
        if n in sw:
            parts.append(f"SW:{sw[n]:.2f}")
        if n in hw:
            parts.append(f"HW:{hw[n]:.2f}")
        if n in area:
            parts.append(f"Area:{area[n]:.1f}")
        labels[n] = "\n".join(parts)

    # Edge costs: comm may use tuple keys or edge->value mapping
    edge_costs = {}
    # normalize comm mapping
    for k, v in comm.items():
        if isinstance(k, tuple) and len(k) >= 2:
            edge_costs[(k[0], k[1])] = float(v)
        else:
            # if key is edge-like string, try to parse, else ignore
            edge_costs[k] = float(v)

    # safe max for scaling
    numeric_edge_values = [v for k,v in edge_costs.items() if isinstance(k, tuple)]
    max_comm = max(numeric_edge_values) if numeric_edge_values else 1.0

    edge_labels = {}
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        val = edge_costs.get((u, v), edge_costs.get((str(u), str(v)), 0.0))
        edge_labels[(u, v)] = f"{val:.2f}"
        # width scaled, min 0.5 max ~6
        width = max(0.5, (val / max_comm) * 6.0) if max_comm > 0 else 0.5
        edge_widths.append(width)
        pu = normalized_partition.get(u, 0)
        pv = normalized_partition.get(v, 0)
        edge_colors.append('red' if pu != pv else '#888888')

    # Node colors
    node_colors = []
    for n in G.nodes():
        node_colors.append('#7fbf7f' if normalized_partition.get(n,0)==1 else '#9ecae1')

    # layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    os.makedirs(out_dir, exist_ok=True)

    # filename composition
    base = fig_name if fig_name else _config_basename(config)
    cfg_part = _config_basename(config)
    #filename = _safe_filename(f"{base}_{cfg_part}_{time.strftime('%Y%m%d-%H%M%S')}.pdf")
    filename = _safe_filename(f"{base}_{cfg_part}.pdf")

    out_path = os.path.join(out_dir, filename)

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title(f"Task Graph (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.95, node_size=1200, linewidths=0.5, edgecolors='k')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrowsize=12)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # legend
    import matplotlib.patches as mpatches
    hw_patch = mpatches.Patch(color='#7fbf7f', label='Hardware (1)')
    sw_patch = mpatches.Patch(color='#9ecae1', label='Software (0)')
    cross_patch = mpatches.Patch(color='red', label='Inter-partition Comm')
    same_patch = mpatches.Patch(color='#888888', label='Intra-partition Comm')
    plt.legend(handles=[hw_patch, sw_patch, cross_patch, same_patch], loc='lower left', fontsize=8)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, format='pdf')
    if show:
        plt.show()
    plt.close()

    print(f"Task graph visualization saved to: {out_path}")

    return out_path
