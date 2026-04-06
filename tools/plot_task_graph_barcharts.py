#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "mip",
    "random",
    "greedy",
    "gcps",
    "pso",
    "dbpso",
    "clpso",
    "ccpso",
    "esa",
    "shade",
    "jade",
    "gl25",
    "diff_gnn",
    "diff_gnn_order",
]

METHOD_LABELS = {
    "mip": "MILP",
    "random": "Random",
    "greedy": "Greedy",
    "gcps": "GCPS",
    "pso": "PSO",
    "dbpso": "DBPSO",
    "clpso": "CLPSO",
    "ccpso": "CCPSO",
    "esa": "ESA",
    "shade": "SHADE",
    "jade": "JADE",
    "gl25": "GL25",
    "diff_gnn": "Diff-GNN",
    "diff_gnn_order": "Diff-GNN-Order",
}

FAMILY_ORDER = {"custom": 0, "pytorch-graphs": 1, "tflite-graphs": 2, "synthetic": 3}
GNN_METHODS = [m for m in METHOD_ORDER if m != "mip"]
KEY_COLS = [
    "graph_name",
    "seed",
    "area_constraint",
    "hw_scale_factor",
    "hw_scale_variance",
    "comm_scale_factor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate task-graph experiment CSVs and draw grouped bar charts.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gnn-csv", type=Path, required=True)
    parser.add_argument("--mip-csv", type=Path, default=None)
    parser.add_argument("--mode", choices=["graph_suite", "area_sweep"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="", help="Short tag used in output filenames.")
    return parser.parse_args()


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _key_tuple(frame: pd.DataFrame) -> pd.Series:
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
    for col in ("seed", "nodes", "edges"):
        frame[col] = frame[col].astype(int)
    for col in ("area_constraint", "hw_scale_factor", "hw_scale_variance", "comm_scale_factor"):
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


def _reported_makespan(row: pd.Series, method: str) -> tuple[float | None, float | None, float | None]:
    static = _safe_float(row.get(f"{method}_lssp_makespan", row.get(f"{method}_makespan")))
    learned = _safe_float(row.get(f"{method}_lssp_swprio_makespan"))
    if method == "diff_gnn_order":
        vals = [v for v in (static, learned) if v is not None]
        report = min(vals) if vals else None
    else:
        report = static
    return report, static, learned


def _load_gnn_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KEY_COLS + ["method", "reported_makespan", "static_makespan", "learned_makespan", "dag_makespan"])

    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=KEY_COLS + ["method", "reported_makespan", "static_makespan", "learned_makespan", "dag_makespan"])

    frame["graph_name"] = frame["GraphName"].map(lambda value: Path(str(value)).stem)
    frame["seed"] = frame["Seed"].astype(int)
    frame["area_constraint"] = frame["Area_Percentage"].astype(float)
    frame["hw_scale_factor"] = frame["HW_Scale_Factor"].astype(float)
    frame["hw_scale_variance"] = frame["HW_Scale_Var"].astype(float)
    frame["comm_scale_factor"] = frame["Comm_Scale_Var"].astype(float)

    dag_candidates = [f"{method}_dag_makespan" for method in GNN_METHODS]
    rows: list[dict] = []
    for _, row in frame.iterrows():
        dag = _first_finite(row, dag_candidates)
        for method in GNN_METHODS:
            report, static, learned = _reported_makespan(row, method)
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
                    "static_makespan": static,
                    "learned_makespan": learned,
                    "dag_makespan": dag,
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    long["key"] = _key_tuple(long)
    long = long.drop_duplicates(subset=["key", "method"], keep="last")
    return long


def _load_mip_long(path: Path | None, dag_lookup: dict[tuple, float | None]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=KEY_COLS + ["method", "reported_makespan", "static_makespan", "learned_makespan", "dag_makespan", "key"])

    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=KEY_COLS + ["method", "reported_makespan", "static_makespan", "learned_makespan", "dag_makespan", "key"])

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
            "dag_makespan",
            "key",
        ]
    ].copy()
    out["method"] = "mip"
    out["reported_makespan"] = frame["mip_makespan"].astype(float)
    out["static_makespan"] = frame["mip_makespan"].astype(float)
    out["learned_makespan"] = np.nan
    out = out.drop_duplicates(subset=["key", "method"], keep="last")
    return out


def _merge_with_manifest(manifest: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(
        manifest[
            [
                "key",
                "graph_name",
                "family",
                "nodes",
                "edges",
                "area_constraint",
                "seed",
            ]
        ],
        on=["key", "graph_name", "area_constraint", "seed"],
        how="left",
        suffixes=("", "_manifest"),
    )
    merged["reported_over_dag"] = merged["reported_makespan"] / merged["dag_makespan"]
    return merged


def _group_order_graphs(manifest: pd.DataFrame) -> list[str]:
    uniq = manifest[["graph_name", "family", "nodes"]].drop_duplicates()
    uniq["family_order"] = uniq["family"].map(lambda value: FAMILY_ORDER.get(value, 99))
    uniq = uniq.sort_values(["family_order", "nodes", "graph_name"])
    return uniq["graph_name"].tolist()


def _grouped_bar(
    summary: pd.DataFrame,
    groups: list,
    group_col: str,
    value_col: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    methods = [method for method in METHOD_ORDER if method in set(summary["method"])]
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(methods), 1)))
    x = np.arange(len(groups))
    total_width = 0.86
    width = total_width / max(len(methods), 1)

    fig_w = max(16, len(groups) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 7.5))
    for idx, method in enumerate(methods):
        method_df = summary[summary["method"] == method]
        lookup = dict(zip(method_df[group_col], method_df[value_col]))
        values = [lookup.get(group, np.nan) for group in groups]
        offsets = x - total_width / 2 + idx * width + width / 2
        ax.bar(offsets, values, width=width, label=METHOD_LABELS.get(method, method), color=colors[idx])

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=min(5, max(len(methods), 1)), fontsize=9, frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_summary(summary: pd.DataFrame, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / filename, index=False)


def _plot_graph_suite(manifest: pd.DataFrame, merged: pd.DataFrame, output_dir: Path, tag: str) -> None:
    group_order = _group_order_graphs(manifest)
    summary = (
        merged.groupby(["graph_name", "family", "nodes", "method"], dropna=False)
        .agg(
            mean_reported_makespan=("reported_makespan", "mean"),
            mean_reported_over_dag=("reported_over_dag", "mean"),
            mean_dag_makespan=("dag_makespan", "mean"),
            num_runs=("reported_makespan", "count"),
        )
        .reset_index()
    )
    summary["graph_name"] = pd.Categorical(summary["graph_name"], categories=group_order, ordered=True)
    summary = summary.sort_values(["graph_name", "method"])
    _write_summary(summary, output_dir, f"{tag or 'graph_suite'}_summary.csv")
    _grouped_bar(
        summary=summary,
        groups=group_order,
        group_col="graph_name",
        value_col="mean_reported_over_dag",
        y_label="Mean reported makespan / mean DAG makespan",
        title="Area-0.5 topology suite: mean normalized reported makespan",
        output_path=output_dir / f"{tag or 'graph_suite'}_normalized_barchart.png",
    )


def _plot_area_sweep(manifest: pd.DataFrame, merged: pd.DataFrame, output_dir: Path, tag: str) -> None:
    summary = (
        merged.groupby(["area_constraint", "method"], dropna=False)
        .agg(
            mean_reported_makespan=("reported_makespan", "mean"),
            mean_reported_over_dag=("reported_over_dag", "mean"),
            num_runs=("reported_makespan", "count"),
        )
        .reset_index()
        .sort_values(["area_constraint", "method"])
    )
    group_order = [round(float(v), 2) for v in sorted(summary["area_constraint"].dropna().unique().tolist())]
    summary["area_group"] = summary["area_constraint"].map(lambda value: round(float(value), 2))
    _write_summary(summary, output_dir, f"{tag or 'area_sweep'}_summary.csv")
    _grouped_bar(
        summary=summary,
        groups=group_order,
        group_col="area_group",
        value_col="mean_reported_makespan",
        y_label="Mean reported makespan",
        title="SqueezeNet-TOSA area sweep: mean reported makespan",
        output_path=output_dir / f"{tag or 'area_sweep'}_reported_barchart.png",
    )
    _grouped_bar(
        summary=summary,
        groups=group_order,
        group_col="area_group",
        value_col="mean_reported_over_dag",
        y_label="Mean reported makespan / mean DAG makespan",
        title="SqueezeNet-TOSA area sweep: mean normalized reported makespan",
        output_path=output_dir / f"{tag or 'area_sweep'}_normalized_barchart.png",
    )


def main() -> int:
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    gnn_long = _load_gnn_long(args.gnn_csv)
    dag_lookup = dict(zip(gnn_long["key"], gnn_long["dag_makespan"])) if not gnn_long.empty else {}
    mip_long = _load_mip_long(args.mip_csv, dag_lookup)
    frames = [frame for frame in (gnn_long, mip_long) if not frame.empty]
    if frames:
        results = pd.concat(frames, ignore_index=True)
    else:
        results = pd.DataFrame(columns=KEY_COLS + ["method", "reported_makespan", "static_makespan", "learned_makespan", "dag_makespan", "key"])
    merged = _merge_with_manifest(manifest, results)

    if args.mode == "graph_suite":
        _plot_graph_suite(manifest, merged, args.output_dir, args.tag)
    else:
        _plot_area_sweep(manifest, merged, args.output_dir, args.tag)

    print(f"Wrote charts and summaries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
