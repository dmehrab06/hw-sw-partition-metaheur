#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

GLOBAL_FONT_SIZE = 24
FIG_WIDTH = 22.0
FIG_HEIGHT = 10.0
TRAINING_WIDTH = 80.0
POST_GAP = 2.0
POST_WIDTH = 18.0
TOTAL_WIDTH = TRAINING_WIDTH + POST_GAP + POST_WIDTH

plt.rcParams["font.size"] = GLOBAL_FONT_SIZE
plt.rcParams["axes.titlesize"] = GLOBAL_FONT_SIZE
plt.rcParams["axes.labelsize"] = GLOBAL_FONT_SIZE
plt.rcParams["xtick.labelsize"] = GLOBAL_FONT_SIZE
plt.rcParams["ytick.labelsize"] = GLOBAL_FONT_SIZE
plt.rcParams["legend.fontsize"] = GLOBAL_FONT_SIZE

SOFT_COLOR = "#2f5597"
STATIC_COLOR = "#f28e2b"
LEARNED_COLOR = "#59a14f"
POST_COLOR = "#b07aa1"
AREA_COLOR = "#6c757d"
DIVIDER_COLOR = "#7f7f7f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot diff_gnn_order ablation trace.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument(
        "--epoch-stride",
        type=int,
        default=1,
        help="Keep every Nth training epoch in the plot. Post-processing rows are always kept.",
    )
    return parser.parse_args()


def _pretty_dataset_name(graph_name: str) -> str:
    return str(graph_name).replace("_", " ").upper()


def _to_bool(value) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sanitize_metric(series: pd.Series) -> pd.Series:
    out = _to_numeric(series).astype(float)
    out[np.abs(out) >= 1e8] = np.nan
    return out


def _parse_area_constraint(source_config: str | Path | None) -> float | None:
    if not source_config:
        return None
    path = Path(source_config)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(r"(?m)^area-constraint:\s*([0-9]*\.?[0-9]+)\s*$", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _build_train_ticks(train_df: pd.DataFrame, max_ticks: int = 5) -> tuple[list[float], list[str]]:
    if train_df.empty:
        return [], []
    if len(train_df) == 1:
        epoch_value = int(train_df.iloc[0]["epoch_num"]) if pd.notna(train_df.iloc[0]["epoch_num"]) else 1
        return [float(train_df.iloc[0]["plot_x"])], [str(epoch_value)]

    raw_indices = np.linspace(0, len(train_df) - 1, min(max_ticks, len(train_df)), dtype=int)
    indices = list(dict.fromkeys(int(i) for i in raw_indices))
    positions = [float(train_df.iloc[i]["plot_x"]) for i in indices]
    labels = []
    for i in indices:
        epoch_value = train_df.iloc[i]["epoch_num"]
        labels.append(str(int(epoch_value)) if pd.notna(epoch_value) else f"E{i + 1}")
    return positions, labels


def _build_post_ticks(post_df: pd.DataFrame, max_total_labels: int = 4) -> tuple[list[float], list[str]]:
    if post_df.empty:
        return [], []

    count = len(post_df)
    if count <= max_total_labels:
        selected_indices = list(range(count))
    else:
        selected_indices = np.linspace(0, count - 1, int(max_total_labels), dtype=int).tolist()
        selected_indices = list(dict.fromkeys(int(idx) for idx in selected_indices))
        if selected_indices[-1] != count - 1:
            selected_indices[-1] = count - 1

    positions: list[float] = []
    labels: list[str] = []
    for post_idx in selected_indices:
        x = float(post_df.iloc[post_idx]["plot_x"])
        if post_idx == 0:
            x += 0.8
        positions.append(x)
        labels.append(f"A{post_idx + 1}")
    return positions, labels


def _line_with_markers(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    *,
    color: str,
    label: str,
    linewidth: float = 3.0,
) -> Line2D:
    line, = ax.plot(x, y, color=color, linewidth=linewidth, label=label, zorder=2)
    finite_mask = np.isfinite(y)
    feasible_mask = finite_mask & (valid == 1)
    infeasible_mask = finite_mask & (valid == 0)
    if feasible_mask.any():
        ax.scatter(
            x[feasible_mask],
            y[feasible_mask],
            s=85,
            marker="o",
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
    if infeasible_mask.any():
        ax.scatter(
            x[infeasible_mask],
            y[infeasible_mask],
            s=150,
            marker="X",
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
    return line


def main() -> None:
    args = parse_args()
    epoch_stride = max(1, int(args.epoch_stride))
    output_pdf = args.output_pdf if args.output_pdf is not None else args.output_png.with_suffix(".pdf")

    frame = pd.read_csv(args.input_csv)
    if frame.empty:
        raise SystemExit(f"Input CSV is empty: {args.input_csv}")

    frame["epoch_num"] = _to_numeric(frame.get("epoch", pd.Series(dtype=float)))
    frame["iteration_num"] = _to_numeric(frame.get("iteration", pd.Series(dtype=float)))
    frame["operation_index_num"] = _to_numeric(frame.get("operation_index", pd.Series(dtype=float)))
    frame["global_step_num"] = _to_numeric(frame.get("global_step", pd.Series(dtype=float)))
    frame["accepted_bool"] = frame.get("accepted", pd.Series(dtype=object)).map(_to_bool)
    frame["valid_bool"] = frame.get("threshold_partition_valid", pd.Series(dtype=object)).map(_to_bool)

    train_df = frame[(frame["phase"] == "train") & (frame["event"] == "epoch")].copy()
    train_df = train_df.sort_values(["epoch_num", "iteration_num", "global_step_num"], kind="stable").reset_index(drop=True)
    if epoch_stride > 1 and not train_df.empty:
        train_keep_mask = (
            train_df["epoch_num"].eq(1)
            | train_df["training_end"].map(_to_bool).fillna(False)
            | (train_df["epoch_num"] % epoch_stride == 0)
        )
        train_df = train_df[train_keep_mask].copy().reset_index(drop=True)

    post_mask = (
        ((frame["phase"] == "postprocess") & ((frame["event"].isin(["decode_selected", "start", "done"])) | (frame["accepted_bool"] == True)))
        | ((frame["phase"] == "final") & (frame["event"] == "selected_final"))
    )
    post_df = frame[post_mask].copy()
    post_df = post_df.sort_values(["operation_index_num", "iteration_num", "global_step_num"], kind="stable").reset_index(drop=True)

    if train_df.empty:
        raise SystemExit(f"No training rows found in: {args.input_csv}")

    if len(train_df) == 1:
        train_df["plot_x"] = [TRAINING_WIDTH / 2.0]
    else:
        train_df["plot_x"] = np.linspace(0.0, TRAINING_WIDTH, len(train_df))

    if not post_df.empty:
        if len(post_df) == 1:
            post_df["plot_x"] = [TRAINING_WIDTH + POST_GAP + (POST_WIDTH / 2.0)]
        else:
            post_df["plot_x"] = np.linspace(TRAINING_WIDTH + POST_GAP, TOTAL_WIDTH, len(post_df))

    plot_df = pd.concat([train_df, post_df], ignore_index=True, sort=False)
    plot_df = plot_df.sort_values("plot_x", kind="stable").reset_index(drop=True)

    x_all = plot_df["plot_x"].to_numpy(dtype=float)
    soft_y = _sanitize_metric(plot_df.get("soft_seq_makespan", pd.Series(dtype=float))).to_numpy(dtype=float)
    static_y = _sanitize_metric(plot_df.get("threshold_lssp_static", pd.Series(dtype=float))).to_numpy(dtype=float)
    learned_y = _sanitize_metric(plot_df.get("threshold_lssp_learned_swprio", pd.Series(dtype=float))).to_numpy(dtype=float)
    post_y = _sanitize_metric(plot_df.get("postprocess_lssp_cost", pd.Series(dtype=float))).to_numpy(dtype=float)

    valid_series = plot_df.get("valid_bool", pd.Series([None] * len(plot_df)))
    valid_numeric = np.array([1 if value is True else 0 if value is False else -1 for value in valid_series], dtype=int)

    area_values = _to_numeric(plot_df.get("threshold_hw_area", pd.Series(dtype=float))).to_numpy(dtype=float)
    budget_values = _to_numeric(plot_df.get("threshold_budget", pd.Series(dtype=float))).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        area_ratio = area_values / budget_values
    area_ratio[~np.isfinite(area_ratio)] = np.nan

    graph_name = str(plot_df.iloc[0].get("graph_name", "dataset"))
    source_config = plot_df.iloc[0].get("source_config")
    area_constraint = _parse_area_constraint(source_config)
    if area_constraint is None:
        title = _pretty_dataset_name(graph_name)
    else:
        title = f"{_pretty_dataset_name(graph_name)} | AREA = {area_constraint:.2f}"

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax2 = ax.twinx()

    finite_soft = np.isfinite(soft_y)
    if finite_soft.any():
        ax.plot(
            x_all[finite_soft],
            soft_y[finite_soft],
            color=SOFT_COLOR,
            linewidth=3.2,
            label="Soft-makespan",
            zorder=2,
        )

    static_line = _line_with_markers(
        ax,
        x_all,
        static_y,
        valid_numeric,
        color=STATIC_COLOR,
        label=r"LS$_{static}$",
    )
    learned_line = _line_with_markers(
        ax,
        x_all,
        learned_y,
        valid_numeric,
        color=LEARNED_COLOR,
        label=r"LS$_{learn}$",
    )

    finite_post = np.isfinite(post_y)
    post_line = None
    if finite_post.any():
        post_line, = ax.plot(
            x_all[finite_post],
            post_y[finite_post],
            color=POST_COLOR,
            linewidth=3.2,
            marker="s",
            markersize=7,
            label=r"LS$_{best}$",
            zorder=3,
        )

    finite_ratio = np.isfinite(area_ratio)
    area_line = None
    if finite_ratio.any():
        area_line, = ax2.plot(
            x_all[finite_ratio],
            area_ratio[finite_ratio],
            color=AREA_COLOR,
            linewidth=2.6,
            linestyle=(0, (7, 4)),
            label="Area / Budget",
            zorder=1,
        )
    ax2.axhline(
        1.0,
        color=AREA_COLOR,
        linewidth=2.0,
        linestyle=":",
        alpha=0.8,
        zorder=1,
    )

    plotted_metrics = [soft_y, static_y, learned_y, post_y]
    finite_metric_values = np.concatenate([values[np.isfinite(values)] for values in plotted_metrics if np.isfinite(values).any()])
    if finite_metric_values.size == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min = max(0.0, float(np.nanmin(finite_metric_values)) * 0.95)
        y_max = float(np.nanmax(finite_metric_values)) * 1.10
        if not np.isfinite(y_max) or y_max <= y_min:
            y_max = y_min + 1.0
    ax.set_ylim(y_min, y_max)

    if finite_ratio.any():
        ratio_max = float(np.nanmax(area_ratio[finite_ratio]))
        ax2.set_ylim(0.0, max(1.10, ratio_max * 1.08))
    else:
        ax2.set_ylim(0.0, 1.10)

    right_padding = 3.0 if post_df.empty else 4.5
    ax.set_xlim(-1.0, TOTAL_WIDTH + right_padding)
    ax.axvline(TRAINING_WIDTH, color=DIVIDER_COLOR, linewidth=2.0, linestyle="--", alpha=0.85)

    train_tick_positions, train_tick_labels = _build_train_ticks(train_df)
    post_tick_positions, post_tick_labels = _build_post_ticks(post_df)
    xticks = train_tick_positions + post_tick_positions
    xlabels = train_tick_labels + post_tick_labels
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        tick_labels = ax.get_xticklabels()
        if post_tick_positions:
            first_post_label_index = len(train_tick_positions)
            if 0 <= first_post_label_index < len(tick_labels):
                tick_labels[first_post_label_index].set_horizontalalignment("left")
            last_post_label_index = len(train_tick_positions) + len(post_tick_positions) - 1
            if 0 <= last_post_label_index < len(tick_labels):
                tick_labels[last_post_label_index].set_horizontalalignment("right")

    ax.set_xlabel("Epoch / Post-Process Step")
    ax.set_ylabel("Makespan")
    ax2.set_ylabel("Area / Budget")
    ax.set_title(title, pad=18)

    ax.grid(axis="y", linestyle=":", alpha=0.35)

    text_y = y_min + 0.94 * (y_max - y_min)
    ax.text(
        TRAINING_WIDTH / 2.0,
        text_y,
        "TRAINING",
        ha="center",
        va="center",
        fontweight="bold",
        color="#444444",
    )
    if not post_df.empty:
        ax.text(
            TRAINING_WIDTH + POST_GAP + (POST_WIDTH / 2.0),
            text_y,
            "POST-PROCESSING",
            ha="center",
            va="center",
            fontweight="bold",
            color="#444444",
        )

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []

    if finite_soft.any():
        legend_handles.append(Line2D([0], [0], color=SOFT_COLOR, linewidth=3.2))
        legend_labels.append("Soft-makespan")
    legend_handles.append(static_line)
    legend_labels.append(r"LS$_{static}$")
    legend_handles.append(learned_line)
    legend_labels.append(r"LS$_{learn}$")
    if post_line is not None:
        legend_handles.append(post_line)
        legend_labels.append(r"LS$_{best}$")
    legend_handles.append(Line2D([0], [0], marker="o", color="white", markerfacecolor="black", markeredgecolor="black", markersize=10, linewidth=0))
    legend_labels.append("Feasible")
    legend_handles.append(Line2D([0], [0], marker="X", color="white", markerfacecolor="black", markeredgecolor="black", markersize=12, linewidth=0))
    legend_labels.append("Infeasible")
    if area_line is not None:
        legend_handles.append(Line2D([0], [0], color=AREA_COLOR, linewidth=2.6, linestyle=(0, (7, 4))))
        legend_labels.append("Area / Budget")
        legend_handles.append(Line2D([0], [0], color=AREA_COLOR, linewidth=2.0, linestyle=":"))
        legend_labels.append("Budget = 1.0")

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=200, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
