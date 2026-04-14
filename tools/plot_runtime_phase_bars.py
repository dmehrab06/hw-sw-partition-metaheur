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

GLOBAL_FONT_SIZE = 28
PANEL_WIDTH = 14.0
PANEL_HEIGHT = 6.5
# Match the red/blue family already used in the repo's other batch plots.
OPT_COLOR = "#e15759"
POST_COLOR = "#4e79a7"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot optimization/post-process runtime phase bars.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHOD_ORDER))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--mip-placeholder-sec", type=float, default=None)
    parser.add_argument("--tag", default="runtime")
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


def _attach_result_metadata(frame: pd.DataFrame, source_path: Path, source_index: int) -> pd.DataFrame:
    out = frame.copy()
    if "SimTime" in out:
        out["_result_timestamp"] = pd.to_datetime(out["SimTime"], errors="coerce")
    else:
        out["_result_timestamp"] = pd.NaT
    out["_source_mtime_ns"] = source_path.stat().st_mtime_ns if source_path.exists() else -1
    out["_source_index"] = source_index
    out["_row_index"] = np.arange(len(out), dtype=int)
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


def _discover_result_csvs(search_root: Path, methods: list[str]) -> tuple[list[Path], list[Path]]:
    if not search_root.exists():
        return [], []

    method_set = set(methods)
    gnn_paths: list[Path] = []
    mip_paths: list[Path] = []

    for csv_path in sorted(search_root.rglob("*result-summary-soda-graphs-config.csv")):
        lower_name = csv_path.name.lower()
        if lower_name.startswith("old") or " copy" in lower_name:
            continue

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


def _keep_latest_per_config(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_cols = ["_result_timestamp", "_source_mtime_ns", "_source_index", "_row_index"]
    deduped = (
        frame.sort_values(sort_cols, kind="mergesort", na_position="first")
        .drop_duplicates(subset=["key", "method"], keep="last")
        .copy()
    )
    return deduped.drop(columns=sort_cols, errors="ignore")


def _normalize_runtime_columns(
    row: pd.Series,
    method: str,
) -> tuple[float | None, float | None, float | None]:
    opt = _safe_float(row.get(f"{method}_optimization_time_sec"))
    post = _safe_float(row.get(f"{method}_postprocess_time_sec"))
    total = _safe_float(row.get(f"{method}_total_runtime_sec"))
    legacy = _safe_float(row.get(f"{method}_time"))

    if opt is None and legacy is not None:
        opt = legacy
    if total is None and opt is not None and post is not None:
        total = opt + post
    if total is None and legacy is not None and post is None:
        total = legacy
    if opt is None and total is not None and post is not None:
        opt = max(0.0, total - post)
    if post is None and total is not None and opt is not None:
        post = max(0.0, total - opt)
    if post is None and opt is not None:
        post = 0.0
    if total is None and opt is not None and post is not None:
        total = opt + post

    return opt, post, total


def _load_gnn_results(paths: list[Path], methods: list[str]) -> pd.DataFrame:
    rows: list[dict] = []

    for source_index, path in enumerate(paths):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = _attach_result_metadata(frame, path, source_index)
        frame["graph_name"] = frame["GraphName"].map(lambda value: Path(str(value)).stem)
        frame["seed"] = frame["Seed"].astype(int)
        frame["area_constraint"] = frame["Area_Percentage"].astype(float)
        frame["hw_scale_factor"] = frame["HW_Scale_Factor"].astype(float)
        frame["hw_scale_variance"] = frame["HW_Scale_Var"].astype(float)
        frame["comm_scale_factor"] = frame["Comm_Scale_Var"].astype(float)

        for _, row in frame.iterrows():
            for method in methods:
                if method == "mip":
                    continue
                opt, post, total = _normalize_runtime_columns(row, method)
                if opt is None and post is None and total is None:
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
                        "optimization_time_sec": float(opt or 0.0),
                        "postprocess_time_sec": float(post or 0.0),
                        "total_runtime_sec": float(total or 0.0),
                        "_result_timestamp": row["_result_timestamp"],
                        "_source_mtime_ns": row["_source_mtime_ns"],
                        "_source_index": row["_source_index"],
                        "_row_index": row["_row_index"],
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["key"] = _key_tuple(out)
    return _keep_latest_per_config(out)


def _load_mip_results(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []

    for source_index, path in enumerate(paths):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = _attach_result_metadata(frame, path, source_index)
        frame["graph_name"] = frame["GraphName"].map(lambda value: Path(str(value)).stem)
        frame["seed"] = frame["Seed"].astype(int)
        frame["area_constraint"] = frame["Area_Percentage"].astype(float)
        frame["hw_scale_factor"] = frame["HW_Scale_Factor"].astype(float)
        frame["hw_scale_variance"] = frame["HW_Scale_Var"].astype(float)
        frame["comm_scale_factor"] = frame["Comm_Scale_Var"].astype(float)

        for _, row in frame.iterrows():
            opt, post, total = _normalize_runtime_columns(row, "mip")
            if opt is None and post is None and total is None:
                continue
            rows.append(
                {
                    "graph_name": row["graph_name"],
                    "seed": int(row["seed"]),
                    "area_constraint": float(row["area_constraint"]),
                    "hw_scale_factor": float(row["hw_scale_factor"]),
                    "hw_scale_variance": float(row["hw_scale_variance"]),
                    "comm_scale_factor": float(row["comm_scale_factor"]),
                    "method": "mip",
                    "optimization_time_sec": float(opt or 0.0),
                    "postprocess_time_sec": float(post or 0.0),
                    "total_runtime_sec": float(total or 0.0),
                    "_result_timestamp": row["_result_timestamp"],
                    "_source_mtime_ns": row["_source_mtime_ns"],
                    "_source_index": row["_source_index"],
                    "_row_index": row["_row_index"],
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["key"] = _key_tuple(out)
    return _keep_latest_per_config(out)


def _build_mip_placeholder_results(manifest: pd.DataFrame, placeholder_sec: float) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in manifest.iterrows():
        rows.append(
            {
                "graph_name": row["graph_name"],
                "seed": int(row["seed"]),
                "area_constraint": float(row["area_constraint"]),
                "hw_scale_factor": float(row["hw_scale_factor"]),
                "hw_scale_variance": float(row["hw_scale_variance"]),
                "comm_scale_factor": float(row["comm_scale_factor"]),
                "method": "mip",
                "optimization_time_sec": float(placeholder_sec),
                "postprocess_time_sec": 0.0,
                "total_runtime_sec": float(placeholder_sec),
                "key": row["key"],
            }
        )
    return pd.DataFrame(rows)


def _compute_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    summary = (
        frame.groupby(["graph_name", "family", "nodes", "area_constraint", "method"], dropna=False)
        .agg(
            mean_optimization_time_sec=("optimization_time_sec", "mean"),
            mean_postprocess_time_sec=("postprocess_time_sec", "mean"),
            mean_total_runtime_sec=("total_runtime_sec", "mean"),
            median_total_runtime_sec=("total_runtime_sec", "median"),
            std_total_runtime_sec=("total_runtime_sec", "std"),
            num_runs=("total_runtime_sec", "count"),
        )
        .reset_index()
    )
    return summary


def _pretty_dataset_name(graph_name: str) -> str:
    return graph_name.replace("_", " ").upper()


def _method_positions(methods: list[str]) -> tuple[np.ndarray, list[str]]:
    labels = [METHOD_LABELS.get(method, method.replace("_", "-").upper()) for method in methods]
    return np.arange(len(methods)), labels


def _figure_size(rows: int, cols: int) -> tuple[float, float]:
    return cols * PANEL_WIDTH, rows * PANEL_HEIGHT


def _annotate_no_results(ax) -> None:
    ax.text(
        0.5,
        0.5,
        "No runtime rows",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=GLOBAL_FONT_SIZE,
        color="#666666",
        fontweight="bold",
    )


def _draw_runtime_bars(
    ax,
    summary: pd.DataFrame,
    methods: list[str],
    title: str,
    show_ylabel: bool,
) -> None:
    x, method_labels = _method_positions(methods)
    if summary.empty:
        opt = np.zeros(len(methods), dtype=float)
        post = np.zeros(len(methods), dtype=float)
    else:
        ordered = summary.set_index("method").reindex(methods)
        opt = ordered["mean_optimization_time_sec"].fillna(0.0).astype(float).to_numpy()
        post = ordered["mean_postprocess_time_sec"].fillna(0.0).astype(float).to_numpy()
    total = opt + post

    ax.bar(x, opt, width=0.72, color=OPT_COLOR, alpha=0.70, label="Optimization")
    ax.bar(x, post, width=0.72, bottom=opt, color=POST_COLOR, alpha=0.55, label="Post-process")

    for idx, value in enumerate(total):
        if value <= 0:
            continue
        ax.text(
            idx,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=max(10, int(GLOBAL_FONT_SIZE * 0.45)),
            rotation=0,
        )

    max_total = float(np.max(total)) if total.size else 0.0
    if max_total <= 0.0:
        _annotate_no_results(ax)

    ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel("Runtime (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def _plot_runtime_grid(summary: pd.DataFrame, methods: list[str], datasets: list[str], output_path: Path) -> None:
    if not datasets:
        return
    cols = 1 if len(datasets) == 1 else min(2, len(datasets))
    rows = int(np.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=_figure_size(rows, cols), squeeze=False)

    legend_handles = None
    legend_labels = None

    for index, dataset in enumerate(datasets):
        ax = axes.flat[index]
        sub = summary[summary["graph_name"] == dataset]
        area = float(sub["area_constraint"].iloc[0]) if not sub.empty else np.nan
        area_text = f"{area:.2f}" if np.isfinite(area) else "N/A"
        title = f"{_pretty_dataset_name(dataset)} | AREA = {area_text}"
        _draw_runtime_bars(ax, sub, methods, title, show_ylabel=(index % cols == 0))
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in axes.flat[len(datasets):]:
        ax.axis("off")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            ncol=1,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            bbox_to_anchor=(0.985, 0.985),
        )

    fig.tight_layout(rect=(0, 0, 0.88, 1))
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

    gnn_paths, mip_paths = _discover_result_csvs(args.search_root, methods)
    frames = []
    gnn_results = _load_gnn_results(gnn_paths, methods)
    if not gnn_results.empty:
        frames.append(gnn_results)
    if "mip" in methods:
        if args.mip_placeholder_sec is not None and args.mip_placeholder_sec > 0:
            mip_results = _build_mip_placeholder_results(manifest, float(args.mip_placeholder_sec))
        else:
            mip_results = _load_mip_results(mip_paths)
        if not mip_results.empty:
            frames.append(mip_results)
    if not frames:
        raise SystemExit(f"No runtime result rows found under {args.search_root}")

    results = pd.concat(frames, ignore_index=True)
    merged = results.merge(
        manifest[["key", "graph_name", "seed", "area_constraint", "family", "nodes", "edges"]],
        on=["key", "graph_name", "seed", "area_constraint"],
        how="inner",
    )
    if merged.empty:
        raise SystemExit("No overlapping manifest/runtime rows found.")

    datasets = list(args.datasets) if args.datasets else sorted(merged["graph_name"].unique().tolist())
    merged = merged[merged["graph_name"].isin(datasets)].copy()
    if merged.empty:
        raise SystemExit("No runtime rows matched the requested datasets.")

    summary = _compute_summary(merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / f"{args.tag}_runtime_long_results.csv", index=False)
    summary.to_csv(args.output_dir / f"{args.tag}_runtime_summary.csv", index=False)
    _plot_runtime_grid(summary, methods, datasets, args.output_dir / f"{args.tag}_runtime_phase_bars.png")
    print(f"Wrote runtime plots and CSV summaries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
