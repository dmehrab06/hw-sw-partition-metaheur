#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

GLOBAL_FONT_SIZE = 32
PANEL_WIDTH = 12.0
PANEL_HEIGHT = 6.0

plt.rcParams["font.size"] = GLOBAL_FONT_SIZE
plt.rcParams["axes.titlesize"] = GLOBAL_FONT_SIZE
plt.rcParams["axes.labelsize"] = GLOBAL_FONT_SIZE
plt.rcParams["xtick.labelsize"] = GLOBAL_FONT_SIZE
plt.rcParams["ytick.labelsize"] = GLOBAL_FONT_SIZE
plt.rcParams["legend.fontsize"] = GLOBAL_FONT_SIZE

DEFAULT_METHOD_ORDER = [
    "mip",
    "diff_gnn",
    "diff_gnn_order",
    "gl25",
    "gcps",
    "esa",
    "pso",
    "dbpso",
    "clpso",
    "ccpso",
    "shade",
    "jade",
    "random",
    "greedy",
]

METHOD_LABELS = {
    "mip": "MILP",
    "diff_gnn": "DIFF-GNN-P",
    "random": "RANDOM",
    "greedy": "GREEDY",
    "gcps": "GCPS",
    "pso": "PSO",
    "dbpso": "DBPSO",
    "clpso": "CLPSO",
    "ccpso": "CCPSO",
    "esa": "ESA",
    "shade": "SHADE",
    "jade": "JADE",
    "gl25": "GL25",
    "diff_gnn_order": "DIFF-GNN",
}

METHOD_COLORS = {
    "mip": "#e15759",
    "diff_gnn": "#4c78a8",
    "random": "#76b7b2",
    "greedy": "#59a14f",
    "gcps": "#edc948",
    "pso": "#b07aa1",
    "dbpso": "#ff9da7",
    "clpso": "#9c755f",
    "ccpso": "#bab0ab",
    "esa": "#4e79a7",
    "shade": "#f28e2b",
    "jade": "#af7aa1",
    "gl25": "#8cd17d",
    "diff_gnn_order": "#2f5597",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot large-scale synthetic results with makespan/runtime panels.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--mip-metric",
        choices=["lssp", "lp", "model"],
        default="lssp",
        help="Metric to use for the MIP bars while leaving other methods unchanged.",
    )
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def _ordered_methods(methods: list[str]) -> list[str]:
    seen = set(methods)
    ordered = [method for method in DEFAULT_METHOD_ORDER if method in seen]
    ordered.extend(method for method in methods if method not in ordered)
    return ordered


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _key_tuple(frame: pd.DataFrame) -> list[tuple]:
    return list(
        zip(
            frame["graph_name"],
            frame["seed"],
            frame["area_constraint"].round(6),
            frame["hw_scale_factor"].round(6),
            frame["hw_scale_variance"].round(6),
            frame["comm_scale_factor"].round(6),
        )
    )


def _load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    for col in ("seed", "nodes", "edges"):
        if col in frame:
            frame[col] = frame[col].astype(int)
    for col in ("area_constraint", "hw_scale_factor", "hw_scale_variance", "comm_scale_factor"):
        if col in frame:
            frame[col] = frame[col].astype(float)
    frame["key"] = _key_tuple(frame)
    return frame


def _reported_makespan(row: pd.Series, method: str) -> float | None:
    static = _safe_float(row.get(f"{method}_lssp_makespan", row.get(f"{method}_makespan")))
    learned = _safe_float(row.get(f"{method}_lssp_swprio_makespan"))
    if method == "diff_gnn_order":
        values = [v for v in (static, learned) if v is not None]
        return min(values) if values else None
    return static


def _reported_mip_makespan(row: pd.Series, mip_metric: str) -> float | None:
    col = {
        "lp": "mip_lp_makespan",
        "model": "mip_model_makespan",
    }.get(str(mip_metric).strip().lower(), "mip_makespan")
    return _safe_float(row.get(col))


def _discover_result_csvs(search_root: Path, methods: list[str]) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    method_set = set(methods)
    for csv_path in sorted(search_root.rglob("*result-summary-soda-graphs-config.csv")):
        matched_method = None
        for parent in csv_path.parents:
            if parent == search_root.parent:
                break
            if parent.name in method_set:
                matched_method = parent.name
                break
        if matched_method is None:
            continue
        pairs.append((matched_method, csv_path))
    return pairs


def _load_results(search_root: Path, methods: list[str], mip_metric: str = "lssp") -> pd.DataFrame:
    rows: list[dict] = []
    for method, path in _discover_result_csvs(search_root, methods):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["graph_name"] = frame["GraphName"].map(lambda value: Path(str(value)).stem)
        frame["seed"] = frame["Seed"].astype(int)
        frame["area_constraint"] = frame["Area_Percentage"].astype(float)
        frame["hw_scale_factor"] = frame["HW_Scale_Factor"].astype(float)
        frame["hw_scale_variance"] = frame["HW_Scale_Var"].astype(float)
        frame["comm_scale_factor"] = frame["Comm_Scale_Var"].astype(float)

        for _, row in frame.iterrows():
            if method == "mip":
                reported = _reported_mip_makespan(row, mip_metric)
                runtime = _safe_float(row.get("mip_time"))
            else:
                reported = _reported_makespan(row, method)
                runtime = _safe_float(row.get(f"{method}_time"))
            if reported is None and runtime is None:
                continue
            rows.append(
                {
                    "graph_name": row["graph_name"],
                    "seed": int(row["seed"]),
                    "area_constraint": float(row["area_constraint"]),
                    "hw_scale_factor": float(row["hw_scale_factor"]),
                    "hw_scale_variance": float(row["hw_scale_variance"]),
                    "comm_scale_factor": float(row["comm_scale_factor"]),
                    "method": method,
                    "reported_makespan": reported,
                    "runtime_sec": runtime,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["key"] = _key_tuple(out)
    return out.drop_duplicates(subset=["key", "method"], keep="last")


def _compute_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    def q10(values):
        return np.percentile(values, 10)

    def q90(values):
        return np.percentile(values, 90)

    summary = (
        frame.groupby(["graph_name", "nodes", "area_constraint", "method"], dropna=False)
        .agg(
            mean_makespan=("reported_makespan", "mean"),
            median_makespan=("reported_makespan", "median"),
            min_makespan=("reported_makespan", "min"),
            max_makespan=("reported_makespan", "max"),
            p10_makespan=("reported_makespan", q10),
            p90_makespan=("reported_makespan", q90),
            mean_runtime_sec=("runtime_sec", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
            min_runtime_sec=("runtime_sec", "min"),
            max_runtime_sec=("runtime_sec", "max"),
            p10_runtime_sec=("runtime_sec", q10),
            p90_runtime_sec=("runtime_sec", q90),
            num_runs=("reported_makespan", "count"),
        )
        .reset_index()
    )
    return summary


def _method_positions(methods: list[str]) -> tuple[np.ndarray, list[str]]:
    labels = [METHOD_LABELS.get(method, method.replace("_", "-").upper()) for method in methods]
    return np.arange(len(methods)), labels


def _pretty_dataset_name(graph_name: str) -> str:
    return graph_name.replace("_", " ").upper()


def _draw_metric_boxplots(ax, frame: pd.DataFrame, methods: list[str], metric_col: str, ylabel: str, title: str) -> None:
    x, labels = _method_positions(methods)
    box_data: list[list[float]] = []
    box_positions: list[int] = []
    mean_positions: list[int] = []
    mean_values: list[float] = []

    for idx, method in enumerate(methods):
        values = frame.loc[frame["method"] == method, metric_col].dropna().astype(float).tolist()
        if not values:
            continue
        box_data.append(values)
        box_positions.append(idx)
        mean_positions.append(idx)
        mean_values.append(float(np.mean(values)))

    if box_data:
        boxplot = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.62,
            patch_artist=True,
            whis=(10, 90),
            showfliers=False,
            medianprops={"color": "#ff7f0e", "linewidth": 1.4},
            whiskerprops={"color": "#222222", "linewidth": 1.1},
            capprops={"color": "#222222", "linewidth": 1.1},
            boxprops={"edgecolor": "#222222", "linewidth": 1.2},
        )
        for patch, pos in zip(boxplot["boxes"], box_positions):
            method = methods[pos]
            patch.set_facecolor(METHOD_COLORS.get(method, "#cccccc"))
            patch.set_alpha(0.55)

        ax.plot(
            mean_positions,
            mean_values,
            color="black",
            marker="o",
            linewidth=1.4,
            markersize=6,
            zorder=3,
        )

    ax.set_title(title, fontsize=GLOBAL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=GLOBAL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.tick_params(axis="both", labelsize=GLOBAL_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.text(
        0.98,
        0.02,
        "Mean = ●",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=GLOBAL_FONT_SIZE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.2},
    )


def _plot_grid(frame: pd.DataFrame, methods: list[str], datasets: list[str], output_path: Path) -> None:
    cols = max(1, len(datasets))
    fig, axes = plt.subplots(2, cols, figsize=(cols * PANEL_WIDTH, 2 * PANEL_HEIGHT), squeeze=False)

    for col, dataset in enumerate(datasets):
        sub = frame[frame["graph_name"] == dataset]
        area = float(sub["area_constraint"].iloc[0]) if not sub.empty else np.nan
        title = f"{_pretty_dataset_name(dataset)} | AREA = {area:.2f}"
        _draw_metric_boxplots(axes[0, col], sub, methods, "reported_makespan", "Makespan", title)
        _draw_metric_boxplots(axes[1, col], sub, methods, "runtime_sec", "Runtime (s)", title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    methods = _ordered_methods([method for method in args.methods if method in METHOD_LABELS])
    manifest = _load_manifest(args.manifest)
    if manifest.empty:
        raise SystemExit("Manifest is empty.")

    results = _load_results(args.search_root, methods, mip_metric=args.mip_metric)
    if results.empty:
        raise SystemExit("No result rows found for plotting.")

    merged = results.merge(
        manifest[["key", "graph_name", "nodes", "edges", "seed", "area_constraint"]],
        on=["key", "graph_name", "seed", "area_constraint"],
        how="inner",
    )
    if merged.empty:
        raise SystemExit("No overlapping manifest/result rows found.")

    datasets = [dataset for dataset in args.datasets if dataset in set(merged["graph_name"])]
    if not datasets:
        raise SystemExit("No requested datasets found in merged results.")
    merged = merged[merged["graph_name"].isin(datasets)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / f"{args.tag}_long_results.csv", index=False)
    summary = _compute_summary(merged)
    summary.to_csv(args.output_dir / f"{args.tag}_summary.csv", index=False)
    _plot_grid(merged, methods, datasets, args.output_dir / f"{args.tag}_grid.png")

    print(f"Wrote large-scale figures and CSV summaries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
