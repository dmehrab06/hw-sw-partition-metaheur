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
    parser = argparse.ArgumentParser(description="Plot method-wise makespan bars for batch experiments.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gnn-csv", type=Path, default=None)
    parser.add_argument("--mip-csv", type=Path, default=None)
    parser.add_argument("--search-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["datasets", "area_sweep"], required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHOD_ORDER))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--areas", nargs="*", type=float, default=None)
    parser.add_argument("--tag", default="batch")
    return parser.parse_args()


def _ordered_methods(methods: list[str]) -> list[str]:
    seen = set(methods)
    ordered = [method for method in DEFAULT_METHOD_ORDER if method in seen]
    ordered.extend(method for method in methods if method not in ordered)
    return ordered


def _as_path_list(value: Path | list[Path] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, list):
        return [path for path in value if path is not None]
    return [value]


def _discover_result_csvs(search_root: Path | None, methods: list[str]) -> tuple[list[Path], list[Path]]:
    if search_root is None or not search_root.exists():
        return [], []

    method_set = set(methods)
    gnn_paths: list[Path] = []
    mip_paths: list[Path] = []

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
        if matched_method == "mip":
            mip_paths.append(csv_path)
        else:
            gnn_paths.append(csv_path)

    return gnn_paths, mip_paths


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


def _first_finite(row: pd.Series, candidates: list[str]) -> float | None:
    for col in candidates:
        if col in row:
            value = _safe_float(row[col])
            if value is not None:
                return value
    return None


def _reported_makespan(row: pd.Series, method: str) -> float | None:
    static = _safe_float(row.get(f"{method}_lssp_makespan", row.get(f"{method}_makespan")))
    learned = _safe_float(row.get(f"{method}_lssp_swprio_makespan"))
    if method == "diff_gnn_order":
        values = [v for v in (static, learned) if v is not None]
        return min(values) if values else None
    return static


def _load_gnn_results(paths: Path | list[Path] | None, methods: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    dag_candidates = [f"{method}_dag_makespan" for method in methods if method != "mip"]

    for path in _as_path_list(paths):
        if not path.exists():
            continue
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
            dag = _first_finite(row, dag_candidates)
            for method in methods:
                if method == "mip":
                    continue
                report = _reported_makespan(row, method)
                if report is None:
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
                        "reported_makespan": float(report),
                        "dag_makespan": dag,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["key"] = _key_tuple(out)
    return out.drop_duplicates(subset=["key", "method"], keep="last")


def _load_mip_results(paths: Path | list[Path] | None, dag_lookup: dict[tuple, float | None]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _as_path_list(paths):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["graph_name"] = frame["GraphName"].map(lambda value: Path(str(value)).stem)
        frame["seed"] = frame["Seed"].astype(int)
        frame["area_constraint"] = frame["Area_Percentage"].astype(float)
        frame["hw_scale_factor"] = frame["HW_Scale_Factor"].astype(float)
        frame["hw_scale_variance"] = frame["HW_Scale_Var"].astype(float)
        frame["comm_scale_factor"] = frame["Comm_Scale_Var"].astype(float)
        frame["key"] = _key_tuple(frame)
        frame["dag_makespan"] = frame["key"].map(dag_lookup)
        out = frame[
            [
                "graph_name",
                "seed",
                "area_constraint",
                "hw_scale_factor",
                "hw_scale_variance",
                "comm_scale_factor",
                "key",
                "dag_makespan",
            ]
        ].copy()
        out["method"] = "mip"
        out["reported_makespan"] = frame["mip_makespan"].astype(float)
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["key", "method"], keep="last")


def _compute_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    def q10(values):
        return np.percentile(values, 10)

    def q90(values):
        return np.percentile(values, 90)

    summary = (
        frame.groupby(group_cols + ["method"], dropna=False)
        .agg(
            mean_makespan=("reported_makespan", "mean"),
            median_makespan=("reported_makespan", "median"),
            min_makespan=("reported_makespan", "min"),
            max_makespan=("reported_makespan", "max"),
            p10_makespan=("reported_makespan", q10),
            p90_makespan=("reported_makespan", q90),
            mean_dag_makespan=("dag_makespan", "mean"),
            num_runs=("reported_makespan", "count"),
        )
        .reset_index()
    )
    summary["mean_over_dag"] = summary["mean_makespan"] / summary["mean_dag_makespan"]
    return summary


def _method_positions(methods: list[str]) -> tuple[np.ndarray, list[str]]:
    labels = [METHOD_LABELS.get(method, method.replace("_org", "").replace("_", "-").upper()) for method in methods]
    return np.arange(len(methods)), labels


def _pretty_dataset_name(graph_name: str) -> str:
    return graph_name.replace("_", " ").upper()


def _figure_size(rows: int, cols: int) -> tuple[float, float]:
    return cols * PANEL_WIDTH, rows * PANEL_HEIGHT


def _draw_method_boxplots(ax, frame: pd.DataFrame, methods: list[str], title: str) -> None:
    x, labels = _method_positions(methods)
    box_data: list[list[float]] = []
    box_positions: list[int] = []
    mean_positions: list[int] = []
    mean_values: list[float] = []

    for idx, method in enumerate(methods):
        values = (
            frame.loc[frame["method"] == method, "reported_makespan"]
            .dropna()
            .astype(float)
            .tolist()
        )
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
            label="Mean",
            zorder=3,
        )
        ax.legend(loc="upper right", fontsize=GLOBAL_FONT_SIZE)

    ax.set_title(title, fontsize=GLOBAL_FONT_SIZE)
    ax.set_xlabel("")
    ax.set_ylabel("Makespan", fontsize=GLOBAL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.tick_params(axis="both", labelsize=GLOBAL_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def _plot_dataset_grid(frame: pd.DataFrame, methods: list[str], datasets: list[str], output_path: Path) -> None:
    if not datasets:
        return
    cols = min(2, max(1, len(datasets)))
    rows = int(np.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=_figure_size(rows, cols), squeeze=False)

    for ax, dataset in zip(axes.flat, datasets):
        sub = frame[frame["graph_name"] == dataset]
        area = float(sub["area_constraint"].iloc[0]) if not sub.empty else np.nan
        title = f"{_pretty_dataset_name(dataset)} | AREA = {area:.2f}"
        _draw_method_boxplots(ax, sub, methods, title)

    for ax in axes.flat[len(datasets):]:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_area_grid(frame: pd.DataFrame, methods: list[str], areas: list[float], output_path: Path) -> None:
    if not areas:
        return
    cols = min(2, max(1, len(areas)))
    rows = int(np.ceil(len(areas) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=_figure_size(rows, cols), squeeze=False)

    for ax, area in zip(axes.flat, areas):
        sub = frame[np.isclose(frame["area_constraint"], area)]
        graph_name = str(sub["graph_name"].iloc[0]) if not sub.empty else "squeeze_net_tosa"
        title = f"{_pretty_dataset_name(graph_name)} | AREA = {area:.2f}"
        _draw_method_boxplots(ax, sub, methods, title)

    for ax in axes.flat[len(areas):]:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    methods = _ordered_methods([method for method in args.methods if method in METHOD_LABELS])
    manifest = _load_manifest(args.manifest)
    if manifest.empty:
        raise SystemExit("Manifest is empty.")

    discovered_gnn, discovered_mip = _discover_result_csvs(args.search_root, methods)
    gnn_sources = discovered_gnn + _as_path_list(args.gnn_csv)
    mip_sources = discovered_mip + _as_path_list(args.mip_csv)

    gnn_results = _load_gnn_results(gnn_sources, methods)
    dag_lookup = dict(zip(gnn_results["key"], gnn_results["dag_makespan"])) if not gnn_results.empty else {}
    mip_results = _load_mip_results(mip_sources, dag_lookup) if "mip" in methods else pd.DataFrame()

    frames = [frame for frame in (gnn_results, mip_results) if not frame.empty]
    if not frames:
        raise SystemExit("No result rows found for plotting.")
    results = pd.concat(frames, ignore_index=True)

    merged = results.merge(
        manifest[["key", "graph_name", "family", "nodes", "edges", "seed", "area_constraint"]],
        on=["key", "graph_name", "seed", "area_constraint"],
        how="inner",
        suffixes=("", "_manifest"),
    )
    if merged.empty:
        raise SystemExit("No overlapping manifest/result rows found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / f"{args.tag}_long_results.csv", index=False)

    if args.mode == "datasets":
        datasets = list(args.datasets) if args.datasets else sorted(merged["graph_name"].unique().tolist())
        merged = merged[merged["graph_name"].isin(datasets)]
        summary = _compute_summary(merged, ["graph_name", "family", "nodes", "area_constraint"])
        summary.to_csv(args.output_dir / f"{args.tag}_dataset_summary.csv", index=False)
        _plot_dataset_grid(merged, methods, datasets, args.output_dir / f"{args.tag}_dataset_method_bars.png")
    else:
        areas = list(args.areas) if args.areas else sorted(merged["area_constraint"].unique().tolist())
        merged = merged[merged["area_constraint"].isin(areas)]
        summary = _compute_summary(merged, ["graph_name", "area_constraint"])
        summary.to_csv(args.output_dir / f"{args.tag}_area_summary.csv", index=False)
        _plot_area_grid(merged, methods, areas, args.output_dir / f"{args.tag}_area_method_bars.png")

    print(f"Wrote batch figures and CSV summaries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
