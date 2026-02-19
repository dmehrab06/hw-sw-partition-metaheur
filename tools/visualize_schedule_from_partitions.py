#!/usr/bin/env python
"""
Create input task-graph and output schedule visualizations from saved partitions.

Usage:
  - CLI mode (standalone):
      python tools/visualize_schedule_from_partitions.py --config <yaml>
  - Programmatic mode (from gnn_main):
      from tools.visualize_schedule_from_partitions import generate_visualizations_for_run
"""

import argparse
import math
import os
import pickle
import re
import sys
from glob import glob
from pathlib import Path
from collections.abc import Mapping

import networkx as nx
from omegaconf import OmegaConf


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_taskgraph(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_partition(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if s in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
    return default


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _run_tag_from_config(config: dict) -> str:
    graph_name = Path(str(config.get("graph-file", "graph"))).stem
    cfg_path = str(config.get("config", "config"))
    cfg_name = Path(cfg_path).stem if cfg_path else "config"
    return (
        f"{_sanitize_name(cfg_name)}_"
        f"{_sanitize_name(graph_name)}_"
        f"area-{float(config.get('area-constraint', 0.0)):.2f}_"
        f"hwscale-{float(config.get('hw-scale-factor', 1.0)):.1f}_"
        f"hwvar-{float(config.get('hw-scale-variance', 0.0)):.2f}_"
        f"comm-{float(config.get('comm-scale-factor', 1.0)):.2f}_"
        f"seed-{int(config.get('seed', 0))}"
    )


def _partition_file_from_config(config, method: str) -> str:
    graph_name = Path(str(config["graph-file"])).stem
    filename = (
        f"taskgraph-{graph_name}_"
        f"area-{float(config['area-constraint']):.2f}_"
        f"hwscale-{float(config['hw-scale-factor']):.1f}_"
        f"hwvar-{float(config['hw-scale-variance']):.2f}_"
        f"comm-{float(config['comm-scale-factor']):.2f}_"
        f"seed-{int(config['seed'])}_"
        f"assignment-{method}.pkl"
    )
    return os.path.join(config["solution-dir"], filename)


def _latest_partition_fallback(solution_dir: str, method: str) -> str | None:
    pattern = os.path.join(solution_dir, f"*assignment-{method}.pkl")
    candidates = glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def _assign_overlap_lanes(tasks: list[tuple[str, float, float]], eps: float = 1e-9) -> list[tuple[str, float, float, int]]:
    """
    Assign each interval task to the first available lane so overlapping tasks
    appear on separate stacked lanes.
    """
    lane_end_times: list[float] = []
    assigned: list[tuple[str, float, float, int]] = []

    for node, start, finish in sorted(tasks, key=lambda t: (t[1], t[2], t[0])):
        lane_idx = None
        for i, lane_end in enumerate(lane_end_times):
            if start >= lane_end - eps:
                lane_idx = i
                lane_end_times[i] = finish
                break
        if lane_idx is None:
            lane_idx = len(lane_end_times)
            lane_end_times.append(finish)
        assigned.append((node, start, finish, lane_idx))

    return assigned


def _dag_layered_layout(graph: nx.DiGraph) -> dict:
    """
    DAG-friendly left-to-right layered layout without extra dependencies.
    Falls back to spring layout if graph is not a DAG.
    """
    try:
        topo = list(nx.topological_sort(graph))
    except Exception:
        return nx.spring_layout(graph, seed=42)

    depth = {n: 0 for n in topo}
    for n in topo:
        preds = list(graph.predecessors(n))
        if preds:
            depth[n] = max(depth[p] + 1 for p in preds)

    layers: dict[int, list] = {}
    for n in topo:
        layers.setdefault(depth[n], []).append(n)

    max_depth = max(depth.values()) if depth else 1
    if max_depth <= 0:
        max_depth = 1

    pos = {}
    for d, nodes in layers.items():
        nodes_sorted = sorted(nodes, key=lambda x: str(x))
        k = len(nodes_sorted)
        for i, n in enumerate(nodes_sorted):
            y = 0.5 if k == 1 else 1.0 - (i / (k - 1))
            x = d / max_depth
            pos[n] = (x, y)
    return pos


def _dag_shape_stats(graph: nx.DiGraph) -> tuple[int, int] | None:
    """
    Return (num_layers, max_layer_width) for a DAG, or None for non-DAG.
    """
    try:
        topo = list(nx.topological_sort(graph))
    except Exception:
        return None

    depth = {n: 0 for n in topo}
    for n in topo:
        preds = list(graph.predecessors(n))
        if preds:
            depth[n] = max(depth[p] + 1 for p in preds)

    layers: dict[int, list] = {}
    for n in topo:
        layers.setdefault(depth[n], []).append(n)

    if not layers:
        return (1, 1)
    return (1 + max(layers.keys()), max(len(v) for v in layers.values()))


def _plot_input_task_graph(task_graph, out_path: str, context: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    context = context or {}
    g = task_graph.graph
    pos = _dag_layered_layout(g)
    num_nodes = len(g.nodes())
    num_edges = len(g.edges())

    dag_stats = _dag_shape_stats(g)
    if dag_stats is None:
        # fallback estimate for non-DAGs
        approx = max(1, int(math.sqrt(max(1, num_nodes))))
        num_layers, max_layer_width = approx, approx
    else:
        num_layers, max_layer_width = dag_stats

    # Adaptive canvas size for larger graphs.
    fig_w = 10.0 + 0.65 * float(num_layers) + 0.010 * float(num_edges)
    fig_h = 6.0 + 0.28 * float(max_layer_width) + 0.004 * float(num_nodes)
    fig_w = min(max(fig_w, 12.0), 52.0)
    fig_h = min(max(fig_h, 8.0), 34.0)

    show_attrs = num_nodes <= 50
    show_edge_labels = num_nodes <= 40
    show_node_labels = num_nodes <= 220

    if num_nodes <= 30:
        node_size = 1500
        node_font_size = 10
    elif num_nodes <= 80:
        node_size = 1000
        node_font_size = 8.5
    elif num_nodes <= 220:
        node_size = 520
        node_font_size = 7
    elif num_nodes <= 600:
        node_size = 240
        node_font_size = 0  # unused when labels hidden
    else:
        node_size = 140
        node_font_size = 0  # unused when labels hidden

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=node_size,
        node_color="#f7f7f7",
        edgecolors="#2d2d2d",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_edges(
        g,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.8,
        edge_color="#3a3a3a",
        ax=ax,
        connectionstyle="arc3,rad=0.02",
    )

    if show_node_labels:
        nx.draw_networkx_labels(
            g,
            pos,
            labels={n: str(n) for n in g.nodes()},
            font_size=node_font_size,
            font_weight="bold",
            ax=ax,
        )

    if show_attrs:
        attr_labels = {}
        for n in g.nodes():
            sw = float(task_graph.software_costs.get(n, 0.0))
            hw = float(task_graph.hardware_costs.get(n, 0.0))
            area = float(task_graph.hardware_area.get(n, 0.0))
            attr_labels[n] = f"({sw:.0f},{hw:.0f},{area:.0f})"
        pos_attr = {n: (x, y - 0.07) for n, (x, y) in pos.items()}
        nx.draw_networkx_labels(g, pos_attr, labels=attr_labels, font_size=8.5, ax=ax)

    if show_edge_labels:
        edge_labels = {}
        for u, v in g.edges():
            c = float(task_graph.communication_costs.get((u, v), task_graph.communication_costs.get((v, u), 0.0)))
            if c > 0:
                edge_labels[(u, v)] = f"{c:.0f}"
        if edge_labels:
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8.5, ax=ax)

    ax.annotate(
        "Dependency direction",
        xy=(0.98, 0.97),
        xytext=(0.76, 0.97),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333333"),
        ha="right",
        va="center",
        fontsize=9,
    )

    title = "Input Task Graph (DAG)"
    run_name = context.get("run_name", "")
    if run_name:
        title += f" - {run_name}"
    ax.set_title(title, pad=14)
    ax.set_axis_off()

    info = (
        f"Nodes={len(g.nodes())}  Edges={len(g.edges())}  "
        f"Area limit={float(getattr(task_graph, 'area_constraint', 0.0)):.4f}  "
        f"Seed={context.get('seed', '-')}"
    )
    fig.text(0.01, 0.02, info, ha="left", va="bottom", fontsize=9, family="monospace")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    if num_nodes <= 200:
        dpi = 220
    elif num_nodes <= 500:
        dpi = 170
    else:
        dpi = 140
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_schedule(task_graph, partition: dict, method: str, out_path: str, context: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    context = context or {}
    result = task_graph.evaluate_makespan(partition)
    starts = result["start_times"]
    finishes = result["finish_times"]
    partition_cost = float(task_graph.evaluate_partition_cost(partition))
    hw_nodes = list(result.get("hardware_nodes", [n for n, p in partition.items() if p == 1]))
    sw_nodes = list(result.get("software_nodes", [n for n, p in partition.items() if p == 0]))
    total_comm_delay = float(result.get("total_communication_delay", 0.0))
    active_comm_edges = list(result.get("active_communication_edges", []))

    total_area = float(getattr(task_graph, "total_area", 0.0))
    area_constraint = float(getattr(task_graph, "area_constraint", 0.0))
    area_budget = area_constraint * total_area if total_area > 0 else 0.0
    area_used = float(sum(task_graph.hardware_area.get(n, 0.0) for n in hw_nodes))
    area_used_frac = (area_used / total_area) if total_area > 0 else 0.0
    area_budget_frac = (area_budget / total_area) if total_area > 0 else 0.0

    # Keep Software/Bus/Hardware groups visually separated, but pack stacked
    # hardware lanes tightly so there is no blank vertical gap between them.
    bar_h = 0.34
    hw_lane_pitch = bar_h
    group_gap = 0.9

    software_tasks = [(n, starts[n], finishes[n]) for n in sw_nodes]
    hardware_tasks = [(n, starts[n], finishes[n]) for n in hw_nodes]
    software_tasks.sort(key=lambda t: t[1])
    hardware_tasks.sort(key=lambda t: t[1])
    hardware_laned = _assign_overlap_lanes(hardware_tasks)
    num_hw_lanes = max(1, 1 + max((lane for _, _, _, lane in hardware_laned), default=0))

    y_hardware_top = 0.0
    y_hardware_by_lane = {lane: (y_hardware_top - lane * hw_lane_pitch) for lane in range(num_hw_lanes)}
    y_bus = y_hardware_top + group_gap
    y_software = y_bus + group_gap

    bus_transfers = []
    for (u, v), comm in task_graph.communication_costs.items():
        if partition[u] != partition[v] and comm > 0:
            start = finishes[u]
            end = start + float(comm)
            bus_transfers.append((u, v, start, end))
    bus_transfers.sort(key=lambda t: t[2])

    fig_h_sched = 5.6 + max(0, num_hw_lanes - 1) * 0.45
    fig, (ax, ax_info) = plt.subplots(
        2,
        1,
        figsize=(17.0, fig_h_sched + 2.0),
        gridspec_kw={"height_ratios": [fig_h_sched, 1.8]},
    )

    for n, s, f in software_tasks:
        ax.barh(y_software, f - s, left=s, height=bar_h, color="#E6D7C8", edgecolor="#333333", linewidth=0.7)
        ax.text(s + (f - s) / 2.0, y_software, n, ha="center", va="center", fontsize=9)

    for n, s, f, lane in hardware_laned:
        y = y_hardware_by_lane[lane]
        ax.barh(y, f - s, left=s, height=bar_h, color="#C5DEB3", edgecolor="#333333", linewidth=0.7)
        ax.text(s + (f - s) / 2.0, y, n, ha="center", va="center", fontsize=9)

    for u, v, s, e in bus_transfers:
        ax.barh(y_bus, e - s, left=s, height=bar_h * 0.75, color="#B6CCE1", edgecolor="#333333", linewidth=0.6)
        ax.text(s + (e - s) / 2.0, y_bus, f"{u[1:]}->{v[1:]}", ha="center", va="center", fontsize=7)

    max_finish = max(finishes.values()) if finishes else 0.0
    max_bus = max((e for _, _, _, e in bus_transfers), default=0.0)
    x_max = max(max_finish, max_bus)

    y_ticks = [y_software, y_bus]
    y_labels = ["Software", "Bus"]
    for lane in range(num_hw_lanes):
        y_ticks.append(y_hardware_by_lane[lane])
        y_labels.append(f"Hardware {lane + 1}")
    y_min = min(y_ticks) - 0.6
    y_max = max(y_ticks) + 0.6
    ax.set_yticks(y_ticks, y_labels)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, max(1.0, x_max * 1.02))
    ax.set_xticks(range(0, int(math.ceil(max(1.0, x_max))) + 1))
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_title(f"Schedule ({method})", pad=12)

    info_line_1 = (
        f"Run={context.get('run_name', '-')} | Method={method} | Seed={context.get('seed', '-')} | "
        f"Makespan={float(result['makespan']):.2f} | PartitionCost={partition_cost:.2f}"
    )
    info_line_2 = (
        f"HW nodes={len(hw_nodes)} SW nodes={len(sw_nodes)} | "
        f"Area used/budget={area_used:.2f}/{area_budget:.2f} ({area_used_frac:.3f}/{area_budget_frac:.3f}) | "
        f"Cross edges={len(active_comm_edges)} Comm delay={total_comm_delay:.2f}"
    )
    info_line_3 = f"Partition file={context.get('partition_file', '-')}"

    ax_info.axis("off")
    ax_info.text(0.01, 0.82, info_line_1, ha="left", va="top", fontsize=8.8, family="monospace", transform=ax_info.transAxes)
    ax_info.text(0.01, 0.49, info_line_2, ha="left", va="top", fontsize=8.8, family="monospace", transform=ax_info.transAxes)
    ax_info.text(0.01, 0.16, info_line_3, ha="left", va="top", fontsize=8.8, family="monospace", transform=ax_info.transAxes)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _parse_methods(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        return [v.strip() for v in raw.split(",") if v.strip()]
    return None


def generate_visualizations_for_run(
    config: dict,
    methods: list[str] | None = None,
    out_dir: str | None = None,
    task_graph=None,
    include_input: bool | None = None,
    include_output: bool | None = None,
    strict_partitions: bool = False,
) -> list[str]:
    """
    Generate visualizations for a completed run.
    Returns list of generated file paths.
    """
    cfg = dict(config)
    vis_cfg_raw = cfg.get("visualization", {})
    vis_cfg = dict(vis_cfg_raw) if isinstance(vis_cfg_raw, Mapping) else {}

    include_input_default = _as_bool(vis_cfg.get("include_input_graph", vis_cfg.get("input_graph", False)), False)
    include_output_default = _as_bool(vis_cfg.get("include_output_schedule", vis_cfg.get("output_schedule", True)), True)
    include_input = include_input_default if include_input is None else bool(include_input)
    include_output = include_output_default if include_output is None else bool(include_output)
    strict_partitions = bool(_as_bool(vis_cfg.get("strict_partitions", strict_partitions), strict_partitions))

    vis_methods = methods
    if vis_methods is None:
        vis_methods = _parse_methods(vis_cfg.get("methods", None))
    if vis_methods is None:
        vis_methods = _parse_methods(cfg.get("methods", None))
    vis_methods = vis_methods or []

    if out_dir:
        # CLI override: keep backward-compatible behavior and write directly to this folder.
        out_root = out_dir
        schedule_dir = out_root
        input_dir = out_root
    else:
        out_root = vis_cfg.get("out_dir") or vis_cfg.get("output_dir") or os.path.join(ROOT, "outputs", "fig3_schedule")
        schedule_dir = vis_cfg.get("schedule_dir", out_root)
        input_dir = vis_cfg.get("input_dir", out_root)

    if task_graph is None:
        tg_pkl = cfg.get("taskgraph-pickle", "")
        if not tg_pkl or not os.path.exists(tg_pkl):
            raise FileNotFoundError(f"TaskGraph pickle not found: {tg_pkl}")
        task_graph = _load_taskgraph(tg_pkl)

    run_tag = _run_tag_from_config(cfg)
    saved_paths: list[str] = []

    if include_input:
        in_path = os.path.join(input_dir, f"{run_tag}_input_taskgraph.png")
        _plot_input_task_graph(
            task_graph,
            in_path,
            context={"run_name": run_tag, "seed": cfg.get("seed", "-")},
        )
        saved_paths.append(in_path)

    if include_output:
        for method in vis_methods:
            part_path = _partition_file_from_config(cfg, method)
            if not os.path.exists(part_path):
                if strict_partitions:
                    continue
                fallback = _latest_partition_fallback(cfg["solution-dir"], method)
                if fallback is None:
                    continue
                part_path = fallback

            partition = _load_partition(part_path)
            out_path = os.path.join(schedule_dir, f"{run_tag}_{method}_schedule.png")
            _plot_schedule(
                task_graph,
                partition,
                method,
                out_path,
                context={
                    "run_name": run_tag,
                    "seed": cfg.get("seed", "-"),
                    "partition_file": os.path.basename(part_path),
                },
            )
            saved_paths.append(out_path)

    return saved_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", default=None, help="Comma-separated methods to visualize. Defaults to config methods.")
    ap.add_argument("--out-dir", default=None, help="Root output dir for generated visualizations.")
    ap.add_argument("--include-input", default="auto", help="auto|true|false")
    ap.add_argument("--include-output", default="auto", help="auto|true|false")
    ap.add_argument("--strict-partitions", action="store_true", help="Skip missing partitions instead of fallback to latest.")
    args = ap.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    config["config"] = args.config
    methods = _parse_methods(args.methods) if args.methods is not None else None

    include_input = None
    if str(args.include_input).strip().lower() != "auto":
        include_input = _as_bool(args.include_input, True)
    include_output = None
    if str(args.include_output).strip().lower() != "auto":
        include_output = _as_bool(args.include_output, True)

    saved = generate_visualizations_for_run(
        config=config,
        methods=methods,
        out_dir=args.out_dir,
        task_graph=None,
        include_input=include_input,
        include_output=include_output,
        strict_partitions=args.strict_partitions,
    )

    if not saved:
        print("[warn] no visualization files generated")
    else:
        for p in saved:
            print(f"saved {p}")


if __name__ == "__main__":
    main()
