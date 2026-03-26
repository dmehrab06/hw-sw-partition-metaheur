#!/usr/bin/env python
"""
Demo utility for LSSP scheduling on an existing TaskGraph + partition result.

This script:
1) Loads TaskGraph pickle and partition pickle
2) Computes start/end times and makespan with LSSP
3) Optionally calls the same visualization pipeline used in gnn_main.py
"""

import argparse
from glob import glob
import os
import pickle
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meta_heuristic.partition_schedule_evaluator import evaluate_partition_lssp  # noqa: E402


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg = {}
    try:
        from omegaconf import OmegaConf

        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if isinstance(loaded, dict):
            cfg = dict(loaded)
    except Exception:
        import yaml

        with open(path, "r") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            cfg = dict(loaded)
    cfg["config"] = path
    return cfg


def _partition_file_from_config(config: dict, method: str) -> str:
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


def _resolve_taskgraph_pickle(cfg: dict, explicit_path: str | None) -> str:
    candidate = explicit_path or cfg.get("taskgraph-pickle")
    if not candidate:
        raise ValueError(
            "TaskGraph pickle path missing. Pass --taskgraph-pickle or set taskgraph-pickle in config."
        )
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"TaskGraph pickle not found: {candidate}")
    return candidate


def _resolve_partition_pickle(cfg: dict, method: str, explicit_path: str | None) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Partition pickle not found: {explicit_path}")
        return explicit_path

    if not cfg:
        raise ValueError("Need --partition-pickle when --config is not provided.")

    expected = _partition_file_from_config(cfg, method)
    if os.path.exists(expected):
        return expected

    fallback = _latest_partition_fallback(cfg["solution-dir"], method)
    if fallback and os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        f"Could not find partition for method '{method}'. "
        f"Checked expected path and fallback in solution-dir={cfg.get('solution-dir')}."
    )


def _print_lssp_result(result: dict) -> None:
    makespan = float(result["makespan"])
    starts = result.get("start_times", {}) or {}
    finishes = result.get("finish_times", {}) or {}
    priorities = result.get("static_priorities", {}) or {}

    print("LSSP schedule summary")
    print(f"  makespan: {makespan:.6f}")
    print(f"  hw_nodes: {len(result.get('hardware_nodes', []))}")
    print(f"  sw_nodes: {len(result.get('software_nodes', []))}")
    print("")
    print("Node schedule (sorted by start time)")
    print("node,start,end,priority")
    ordered = sorted(starts.keys(), key=lambda n: (starts[n], finishes.get(n, starts[n]), str(n)))
    for node in ordered:
        print(
            f"{node},{float(starts[node]):.6f},{float(finishes.get(node, starts[node])):.6f},"
            f"{float(priorities.get(node, 0.0)):.6f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config used by gnn_main.py.")
    ap.add_argument("--method", default="diff_gnn_order", help="Method name used in partition filename.")
    ap.add_argument("--taskgraph-pickle", default=None, help="Explicit TaskGraph pickle path.")
    ap.add_argument("--partition-pickle", default=None, help="Explicit partition pickle path.")
    ap.add_argument(
        "--visualize",
        action="store_true",
        help="Also call generate_visualizations_for_run(...) like gnn_main.py.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Optional visualization output dir override.",
    )
    args = ap.parse_args()

    cfg = _load_config(args.config)
    taskgraph_pickle = _resolve_taskgraph_pickle(cfg, args.taskgraph_pickle)
    partition_pickle = _resolve_partition_pickle(cfg, args.method, args.partition_pickle)

    task_graph = _load_pickle(taskgraph_pickle)
    partition = _load_pickle(partition_pickle)

    lssp_result = evaluate_partition_lssp(task_graph, partition)
    _print_lssp_result(lssp_result)

    if args.visualize:
        if not cfg:
            raise ValueError("Visualization mode requires --config so naming and output settings are available.")
        try:
            from tools.visualize_schedule_from_partitions import generate_visualizations_for_run
        except Exception as e:
            raise RuntimeError(
                f"Failed to import visualization helper (tools.visualize_schedule_from_partitions): {e}"
            ) from e

        cfg["taskgraph-pickle"] = taskgraph_pickle
        cfg["solution-dir"] = str(Path(partition_pickle).parent)
        saved_paths = generate_visualizations_for_run(
            config=cfg,
            methods=[args.method],
            out_dir=args.out_dir,
            task_graph=task_graph,
            include_input=True,
            include_output=True,
            strict_partitions=False,
        )
        if saved_paths:
            print("")
            print("Visualization files")
            for path in saved_paths:
                print(f"saved {path}")
        else:
            print("[warn] no visualization files generated")


if __name__ == "__main__":
    main()
