#!/usr/bin/env python
"""
Create schedule visualizations (Software / Bus / Hardware lanes) from saved
partition pickles for selected methods.
"""

import argparse
import math
import os
import pickle
import sys
from glob import glob


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from omegaconf import OmegaConf


def _load_taskgraph(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_partition(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _partition_file_from_config(config, method: str) -> str:
    graph_name = os.path.splitext(os.path.basename(config["graph-file"]))[0]
    filename = (
        f"taskgraph-{graph_name}_"
        f"area-{config['area-constraint']:.2f}_"
        f"hwscale-{config['hw-scale-factor']:.1f}_"
        f"hwvar-{config['hw-scale-variance']:.2f}_"
        f"comm-{config['comm-scale-factor']:.2f}_"
        f"seed-{config['seed']}_"
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


def _plot_schedule(task_graph, partition: dict, method: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = task_graph.evaluate_makespan(partition)
    starts = result["start_times"]
    finishes = result["finish_times"]

    y_software = 2.0
    y_bus = 1.0
    y_hardware = 0.0

    bar_h = 0.34

    software_tasks = [(n, starts[n], finishes[n]) for n, p in partition.items() if p == 0]
    hardware_tasks = [(n, starts[n], finishes[n]) for n, p in partition.items() if p == 1]
    software_tasks.sort(key=lambda t: t[1])
    hardware_tasks.sort(key=lambda t: t[1])

    # Bus transfers for cross-partition edges
    bus_transfers = []
    for (u, v), comm in task_graph.communication_costs.items():
        if partition[u] != partition[v] and comm > 0:
            start = finishes[u]
            end = start + float(comm)
            bus_transfers.append((u, v, start, end))
    bus_transfers.sort(key=lambda t: t[2])

    plt.figure(figsize=(12, 4.5))

    for n, s, f in software_tasks:
        plt.barh(y_software, f - s, left=s, height=bar_h, color="#E6D7C8", edgecolor="#333333", linewidth=0.7)
        plt.text(s + (f - s) / 2.0, y_software, n, ha="center", va="center", fontsize=9)

    for n, s, f in hardware_tasks:
        plt.barh(y_hardware, f - s, left=s, height=bar_h, color="#C5DEB3", edgecolor="#333333", linewidth=0.7)
        plt.text(s + (f - s) / 2.0, y_hardware, n, ha="center", va="center", fontsize=9)

    for u, v, s, e in bus_transfers:
        plt.barh(y_bus, e - s, left=s, height=bar_h * 0.75, color="#B6CCE1", edgecolor="#333333", linewidth=0.6)
        plt.text(s + (e - s) / 2.0, y_bus, f"{u[1:]}->{v[1:]}", ha="center", va="center", fontsize=7)

    max_finish = max(finishes.values()) if finishes else 0.0
    max_bus = max((e for _, _, _, e in bus_transfers), default=0.0)
    x_max = max(max_finish, max_bus)

    plt.yticks([y_software, y_bus, y_hardware], ["Software", "Bus", "Hardware"])
    plt.ylim(-0.6, 2.6)
    plt.xlim(0, max(1.0, x_max * 1.02))
    plt.xticks(range(0, int(math.ceil(max(1.0, x_max))) + 1))
    plt.grid(axis="x", linestyle="--", alpha=0.25)
    plt.title(f"Schedule ({method})  |  makespan={result['makespan']:.2f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", default="diff_gnn,diff_gnn_order")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/fig3_schedule"))
    args = ap.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    tg_pkl = config.get("taskgraph-pickle", "")
    if not tg_pkl or not os.path.exists(tg_pkl):
        raise FileNotFoundError(f"TaskGraph pickle not found: {tg_pkl}")

    task_graph = _load_taskgraph(tg_pkl)

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    if not methods:
        raise ValueError("No methods provided")

    config_base = os.path.splitext(os.path.basename(args.config))[0]

    for method in methods:
        part_path = _partition_file_from_config(config, method)
        if not os.path.exists(part_path):
            fallback = _latest_partition_fallback(config["solution-dir"], method)
            if fallback is None:
                print(f"[skip] partition not found for method={method}")
                continue
            part_path = fallback

        partition = _load_partition(part_path)
        out_path = os.path.join(args.out_dir, f"{config_base}_{method}_schedule.png")
        _plot_schedule(task_graph, partition, method, out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
