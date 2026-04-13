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

MIP_TLE_ONLY_GRAPHS = {
    "mobile_net_tosa",
    "squeezenet_like_1000",
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
    parser.add_argument(
        "--mip-metric",
        choices=["lssp", "lp", "model"],
        default="lssp",
        help="Which MILP makespan field to plot for method 'mip'.",
    )
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
        lower_name = csv_path.name.lower()
        # Ignore obvious backup/manual copies that would otherwise pollute plots
        # with stale rows, e.g. "olddataset_..." or "... copy.csv".
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


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _safe_bool(value, default: bool) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        vv = value.strip().lower()
        if vv in {"1", "true", "yes", "on"}:
            return True
        if vv in {"0", "false", "no", "off"}:
            return False
    return bool(value)


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


def _extract_validity_metadata(row: pd.Series, method: str) -> dict[str, object]:
    solution_valid_raw = row.get(f"{method}_solution_valid")
    initial_valid_raw = row.get(f"{method}_initial_solution_valid")
    was_repaired_raw = row.get(f"{method}_was_repaired")
    num_repaired_raw = row.get(f"{method}_num_repaired_nodes")

    solution_valid = _safe_bool(solution_valid_raw, True)
    initial_solution_valid = _safe_bool(initial_valid_raw, True)
    was_repaired = _safe_bool(was_repaired_raw, False)
    num_repaired_nodes = 0 if pd.isna(num_repaired_raw) else int(num_repaired_raw)

    validity_note = row.get(f"{method}_validity_note")
    if pd.isna(validity_note) or validity_note is None:
        if solution_valid and not was_repaired:
            validity_note = "Valid solution; no area repair needed."
        elif solution_valid and was_repaired:
            validity_note = "Invalid before post-processing; repaired to satisfy the area constraint."
        else:
            validity_note = "Invalid solution after post-processing."

    repair_strategy = row.get(f"{method}_repair_strategy")
    if pd.isna(repair_strategy):
        repair_strategy = None

    area_used = _safe_float(row.get(f"{method}_area_used"))
    area_budget = _safe_float(row.get(f"{method}_area_budget"))

    return {
        "solution_valid": solution_valid,
        "initial_solution_valid": initial_solution_valid,
        "was_repaired": was_repaired,
        "num_repaired_nodes": num_repaired_nodes,
        "repair_strategy": repair_strategy,
        "area_used": area_used,
        "area_budget": area_budget,
        "validity_note": str(validity_note),
    }


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


def _load_gnn_results(paths: Path | list[Path] | None, methods: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    dag_candidates = [f"{method}_dag_makespan" for method in methods if method != "mip"]

    for source_index, path in enumerate(_as_path_list(paths)):
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
                        "_result_timestamp": row["_result_timestamp"],
                        "_source_mtime_ns": row["_source_mtime_ns"],
                        "_source_index": row["_source_index"],
                        "_row_index": row["_row_index"],
                        "status": "completed",
                        "is_timeout": False,
                        "is_failed": False,
                        **_extract_validity_metadata(row, method),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["key"] = _key_tuple(out)
    return _keep_latest_per_config(out)


def _load_mip_results(
    paths: Path | list[Path] | None,
    dag_lookup: dict[tuple, float | None],
    mip_metric: str = "lssp",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    metric_col = {
        "lssp": "mip_makespan",
        "lp": "mip_lp_makespan",
        "model": "mip_model_makespan",
    }.get(str(mip_metric).strip().lower(), "mip_makespan")
    for source_index, path in enumerate(_as_path_list(paths)):
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
                "_result_timestamp",
                "_source_mtime_ns",
                "_source_index",
                "_row_index",
            ]
        ].copy()
        out["method"] = "mip"
        out["reported_makespan"] = pd.to_numeric(frame.get(metric_col, np.nan), errors="coerce")
        out["solution_valid"] = frame.get("mip_solution_valid", True).map(lambda value: _safe_bool(value, True)) if "mip_solution_valid" in frame else True
        out["initial_solution_valid"] = frame.get("mip_initial_solution_valid", True).map(lambda value: _safe_bool(value, True)) if "mip_initial_solution_valid" in frame else True
        out["was_repaired"] = frame.get("mip_was_repaired", False).map(lambda value: _safe_bool(value, False)) if "mip_was_repaired" in frame else False
        out["num_repaired_nodes"] = frame.get("mip_num_repaired_nodes", 0)
        out["repair_strategy"] = frame.get("mip_repair_strategy", None)
        out["area_used"] = frame.get("mip_area_used", np.nan)
        out["area_budget"] = frame.get("mip_area_budget", np.nan)
        out["validity_note"] = frame.get("mip_validity_note", "Valid solution; no area repair needed.")
        out["status"] = frame.get("mip_status", "optimal").fillna("optimal").astype(str)
        out["is_timeout"] = out["status"].eq("time_limit_exceeded")
        out["is_failed"] = out["status"].eq("failed")
        tle_only_mask = out["graph_name"].isin(MIP_TLE_ONLY_GRAPHS) & out["is_timeout"]
        out.loc[tle_only_mask, "reported_makespan"] = np.nan
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return _keep_latest_per_config(out)


def _inject_missing_mip_tle_placeholders(
    manifest: pd.DataFrame,
    mip_results: pd.DataFrame,
    dag_lookup: dict[tuple, float | None],
) -> pd.DataFrame:
    if manifest.empty:
        return mip_results

    target = manifest[manifest["graph_name"].isin(MIP_TLE_ONLY_GRAPHS)].copy()
    if target.empty:
        return mip_results

    existing_keys = set()
    if not mip_results.empty:
        existing_keys = set(mip_results["key"].tolist())
    target = target[~target["key"].isin(existing_keys)]
    if target.empty:
        return mip_results

    rows: list[dict] = []
    for _, row in target.iterrows():
        rows.append(
            {
                "graph_name": row["graph_name"],
                "seed": int(row["seed"]),
                "area_constraint": float(row["area_constraint"]),
                "hw_scale_factor": float(row["hw_scale_factor"]),
                "hw_scale_variance": float(row["hw_scale_variance"]),
                "comm_scale_factor": float(row["comm_scale_factor"]),
                "key": row["key"],
                "dag_makespan": dag_lookup.get(row["key"]),
                "method": "mip",
                "reported_makespan": np.nan,
                "solution_valid": False,
                "initial_solution_valid": False,
                "was_repaired": False,
                "num_repaired_nodes": 0,
                "repair_strategy": None,
                "area_used": np.nan,
                "area_budget": np.nan,
                "validity_note": "Time limit exceeded; plotting timeout only.",
                "status": "time_limit_exceeded",
                "is_timeout": True,
                "is_failed": False,
                "_result_timestamp": pd.NaT,
                "_source_mtime_ns": -1,
                "_source_index": -1,
                "_row_index": -1,
            }
        )

    placeholder = pd.DataFrame(rows)
    if mip_results.empty:
        return placeholder
    combined = pd.concat([mip_results, placeholder], ignore_index=True)
    return _keep_latest_per_config(combined)


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
            num_invalid_runs=("solution_valid", lambda values: int((~pd.Series(values).map(lambda value: _safe_bool(value, True))).sum())),
            num_repaired_runs=("was_repaired", lambda values: int(pd.Series(values).map(lambda value: _safe_bool(value, False)).sum())),
            num_timeout_runs=("is_timeout", lambda values: int(pd.Series(values).map(bool).sum())),
            num_failed_runs=("is_failed", lambda values: int(pd.Series(values).map(bool).sum())),
        )
        .reset_index()
    )
    summary["mean_over_dag"] = summary["mean_makespan"] / summary["mean_dag_makespan"]
    note_frame = (
        frame.groupby(group_cols + ["method"], dropna=False)["validity_note"]
        .agg(lambda values: " | ".join(sorted({str(v) for v in values if pd.notna(v) and str(v).strip()})))
        .reset_index(name="validity_note_summary")
    )
    summary = summary.merge(note_frame, on=group_cols + ["method"], how="left")
    return summary


def _method_positions(methods: list[str]) -> tuple[np.ndarray, list[str]]:
    labels = [METHOD_LABELS.get(method, method.replace("_org", "").replace("_", "-").upper()) for method in methods]
    return np.arange(len(methods)), labels


def _pretty_dataset_name(graph_name: str) -> str:
    return graph_name.replace("_", " ").upper()


def _figure_size(rows: int, cols: int) -> tuple[float, float]:
    return cols * PANEL_WIDTH, rows * PANEL_HEIGHT


def _draw_method_boxplots(ax, frame: pd.DataFrame, methods: list[str], title: str) -> None:
    x, method_labels = _method_positions(methods)
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
            zorder=3,
        )

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    if yrange <= 0:
        yrange = 1.0
        ymax = ymin + yrange
        ax.set_ylim(ymin, ymax)
    status_y = ymin + 0.02 * yrange
    status_font = max(10, int(GLOBAL_FONT_SIZE * 0.45))
    for idx, method in enumerate(methods):
        timeout_count = int(frame.loc[frame["method"] == method, "is_timeout"].sum()) if "is_timeout" in frame else 0
        failed_count = int(frame.loc[frame["method"] == method, "is_failed"].sum()) if "is_failed" in frame else 0
        status_labels = []
        if timeout_count:
            status_labels.append(f"TLE {timeout_count}")
        if failed_count:
            status_labels.append(f"FAIL {failed_count}")
        if status_labels:
            ax.text(
                idx,
                status_y,
                "\n".join(status_labels),
                ha="center",
                va="bottom",
                color="#c62828",
                fontsize=status_font,
                fontweight="bold",
                zorder=4,
            )

    ax.set_title(title, fontsize=GLOBAL_FONT_SIZE)
    ax.set_xlabel("")
    ax.set_ylabel("Makespan", fontsize=GLOBAL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=35, ha="right")
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


def _annotate_no_results(ax) -> None:
    ax.text(
        0.5,
        0.5,
        "No results",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=GLOBAL_FONT_SIZE,
        color="#666666",
        fontweight="bold",
    )


def _plot_dataset_grid(frame: pd.DataFrame, methods: list[str], datasets: list[str], output_path: Path) -> None:
    if not datasets:
        return
    cols = min(2, max(1, len(datasets)))
    rows = int(np.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=_figure_size(rows, cols), squeeze=False)

    for ax, dataset in zip(axes.flat, datasets):
        sub = frame[frame["graph_name"] == dataset]
        area = float(sub["area_constraint"].iloc[0]) if not sub.empty else np.nan
        area_text = f"{area:.2f}" if np.isfinite(area) else "N/A"
        title = f"{_pretty_dataset_name(dataset)} | AREA = {area_text}"
        _draw_method_boxplots(ax, sub, methods, title)
        if sub.empty:
            _annotate_no_results(ax)

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
    mip_results = _load_mip_results(mip_sources, dag_lookup, mip_metric=args.mip_metric) if "mip" in methods else pd.DataFrame()
    if "mip" in methods:
        mip_results = _inject_missing_mip_tle_placeholders(manifest, mip_results, dag_lookup)

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
