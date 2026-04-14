#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/people/dass304/.conda/envs/combopt/bin/python}"
OUTROOT="${OUTROOT:-$ROOT/BatchExperiments/AblationComponents}"

RUN_DIR_ARG="${1:-}"
RUN_DIR="${RUN_DIR:-$RUN_DIR_ARG}"
if [[ -z "$RUN_DIR" ]]; then
  latest_run="$(find "$OUTROOT" -maxdepth 1 -mindepth 1 -type d -name 'run-*' | sort | tail -n 1 || true)"
  if [[ -z "$latest_run" ]]; then
    echo "No ablation run directory found under $OUTROOT"
    exit 1
  fi
  RUN_DIR="$latest_run"
fi

RUN_MANIFEST="$RUN_DIR/ablation_run_manifest.csv"
if [[ ! -f "$RUN_MANIFEST" ]]; then
  echo "Run manifest not found: $RUN_MANIFEST"
  exit 1
fi

echo "Run dir   : $RUN_DIR"
echo "Manifest  : $RUN_MANIFEST"
echo "Filters   : datasets='${DATASETS_FILTER:-}' variants='${VARIANTS_FILTER:-}'"
echo

"$PYTHON" - <<'PY' "$RUN_MANIFEST" "${DATASETS_FILTER:-}" "${VARIANTS_FILTER:-}"
from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


run_manifest = Path(sys.argv[1]).resolve()
datasets_filter_raw = sys.argv[2]
variants_filter_raw = sys.argv[3]


def parse_filter(text: str) -> set[str]:
    tokens = [token.strip() for token in re.split(r"[\s,]+", text.strip()) if token.strip()]
    return set(tokens)


def to_num(value):
    try:
        out = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(out):
        return np.nan
    return out


def string_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column].astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def fmt_stat(values, digits: int = 2) -> str:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if arr.empty:
        return "-"
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return f"{mean:.{digits}f}+-{std:.{digits}f}"


def fmt_mean(values, digits: int = 2) -> str:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if arr.empty:
        return "-"
    return f"{float(arr.mean()):.{digits}f}"


def fmt_improvement(base_values, improved_values, digits: int = 1) -> str:
    base = pd.to_numeric(pd.Series(list(base_values)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    improved = pd.to_numeric(pd.Series(list(improved_values)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if base.empty or improved.empty:
        return "-"
    base_mean = float(base.mean())
    improved_mean = float(improved.mean())
    if not np.isfinite(base_mean) or abs(base_mean) < 1e-12:
        return "-"
    delta_pct = 100.0 * (base_mean - improved_mean) / base_mean
    direction = "lower" if delta_pct >= 0.0 else "higher"
    return f"{abs(delta_pct):.{digits}f}% {direction}"


def load_trace_metrics(trace_dir: Path, seed: int) -> dict[str, float]:
    candidates = sorted(trace_dir.glob(f"seed-{seed}__*.csv"))
    if not candidates:
        candidates = sorted(trace_dir.glob(f"*seed-{seed}*.csv"))
    if not candidates:
        return {
            "pre_fix_lssp": np.nan,
            "pre_fix_sw": np.nan,
            "post_fix_lssp": np.nan,
            "post_fix_sw": np.nan,
            "completed_epochs": np.nan,
        }

    trace_path = candidates[-1]
    trace_df = pd.read_csv(trace_path)
    if trace_df.empty:
        return {
            "pre_fix_lssp": np.nan,
            "pre_fix_sw": np.nan,
            "post_fix_lssp": np.nan,
            "post_fix_sw": np.nan,
            "completed_epochs": np.nan,
        }

    trace_df["epoch_num"] = numeric_series(trace_df, "epoch")
    trace_df["global_step_num"] = numeric_series(trace_df, "global_step")

    train_rows = trace_df[string_series(trace_df, "phase") == "train"].copy()
    training_end_rows = train_rows[
        string_series(train_rows, "training_end").str.lower().isin({"1", "true", "yes"})
    ].copy()
    if not training_end_rows.empty:
        pre_row = training_end_rows.sort_values(["epoch_num", "global_step_num"]).iloc[-1]
    elif not train_rows.empty:
        pre_row = train_rows.sort_values(["epoch_num", "global_step_num"]).iloc[-1]
    else:
        pre_row = None

    decode_rows = trace_df[string_series(trace_df, "event") == "decode_selected"].copy()
    if not decode_rows.empty:
        post_row = decode_rows.sort_values(["epoch_num", "global_step_num"]).iloc[-1]
    else:
        post_row = None

    completed_epochs = numeric_series(trace_df, "epoch").dropna()
    completed_epoch = float(completed_epochs.max()) if not completed_epochs.empty else np.nan

    return {
        "pre_fix_lssp": to_num(pre_row.get("threshold_lssp_static")) if pre_row is not None else np.nan,
        "pre_fix_sw": to_num(pre_row.get("threshold_lssp_learned_swprio")) if pre_row is not None else np.nan,
        "post_fix_lssp": to_num(post_row.get("threshold_lssp_static")) if post_row is not None else np.nan,
        "post_fix_sw": to_num(post_row.get("threshold_lssp_learned_swprio")) if post_row is not None else np.nan,
        "completed_epochs": completed_epoch,
    }


variant_labels = {
    "diff_gnn_no_postprocess": "Train only | Diff-GNN w/o order",
    "diff_gnn_no_refinement": "Train + Area Improve | Diff-GNN w/o order",
    "diff_gnn": "Train + Area Improve + Refine | Diff-GNN w/o order",
    "diff_gnn_order_no_postprocess": "Train only | Diff-GNN",
    "diff_gnn_order_no_refinement": "Train + Area Improve | Diff-GNN",
    "diff_gnn_order": "Train + Area Improve + Refine | Diff-GNN",
    "diff_gnn_no_learned_edge_weights": "Diff-GNN w/o order | w/o learned weights",
    "diff_gnn_order_no_learned_edge_weights": "Diff-GNN | w/o learned weights",
}
variant_display_order = [
    "diff_gnn_no_postprocess",
    "diff_gnn_no_refinement",
    "diff_gnn",
    "diff_gnn_order_no_postprocess",
    "diff_gnn_order_no_refinement",
    "diff_gnn_order",
    "diff_gnn_no_learned_edge_weights",
    "diff_gnn_order_no_learned_edge_weights",
]
variant_order = {name: idx for idx, name in enumerate(variant_display_order)}

component_comparison_rows = [
    ("Train only", "diff_gnn_no_postprocess", "diff_gnn_order_no_postprocess"),
    ("+ Area improve", "diff_gnn_no_refinement", "diff_gnn_order_no_refinement"),
    ("+ Area improve + refine", "diff_gnn", "diff_gnn_order"),
]

dataset_filter = parse_filter(datasets_filter_raw)
variant_filter = parse_filter(variants_filter_raw)

manifest_df = pd.read_csv(run_manifest)
if dataset_filter:
    manifest_df = manifest_df[manifest_df["dataset"].astype(str).isin(dataset_filter)].copy()
if variant_filter:
    manifest_df = manifest_df[manifest_df["variant"].astype(str).isin(variant_filter)].copy()

if manifest_df.empty:
    raise SystemExit("No run-manifest rows matched the requested filters.")

records: list[dict] = []
warnings: list[str] = []

for row in manifest_df.to_dict(orient="records"):
    dataset = str(row["dataset"])
    variant = str(row["variant"])
    method = str(row["method"])
    summary_csv = Path(str(row["summary_csv"])).resolve()
    trace_dir = Path(str(row["trace_dir"])).resolve()

    if not summary_csv.exists():
        warnings.append(f"Missing summary CSV for {dataset}/{variant}: {summary_csv}")
        continue

    summary_df = pd.read_csv(summary_csv)
    if summary_df.empty:
        warnings.append(f"Empty summary CSV for {dataset}/{variant}: {summary_csv}")
        continue

    prefix = method
    for _, summary_row in summary_df.iterrows():
        seed = int(float(summary_row["Seed"]))
        trace_metrics = load_trace_metrics(trace_dir, seed)
        if np.isnan(trace_metrics["completed_epochs"]):
            warnings.append(f"Missing trace CSV for {dataset}/{variant} seed={seed} under {trace_dir}")

        records.append(
            {
                "dataset": dataset,
                "variant": variant,
                "variant_label": variant_labels.get(variant, variant),
                "seed": seed,
                "epochs": trace_metrics["completed_epochs"],
                "pre_fix_lssp": trace_metrics["pre_fix_lssp"],
                "pre_fix_sw": trace_metrics["pre_fix_sw"],
                "post_fix_lssp": trace_metrics["post_fix_lssp"],
                "post_fix_sw": trace_metrics["post_fix_sw"],
                "final_dag": to_num(summary_row.get(f"{prefix}_dag_makespan")),
                "final_lssp": to_num(summary_row.get(f"{prefix}_lssp_makespan")),
                "final_sw": to_num(summary_row.get(f"{prefix}_lssp_swprio_makespan")),
                "final_report": to_num(summary_row.get(f"{prefix}_best_makespan")),
                "opt_sec": to_num(summary_row.get(f"{prefix}_optimization_time_sec")),
                "post_sec": to_num(summary_row.get(f"{prefix}_postprocess_time_sec")),
                "total_sec": to_num(summary_row.get(f"{prefix}_total_runtime_sec")),
            }
        )

if not records:
    raise SystemExit("No ablation records were found.")

records_df = pd.DataFrame.from_records(records)

dataset_order = list(dict.fromkeys(manifest_df["dataset"].astype(str).tolist()))

for dataset in dataset_order:
    dataset_df = records_df[records_df["dataset"] == dataset].copy()
    if dataset_df.empty:
        continue

    rows = []
    variants = sorted(
        dataset_df["variant"].astype(str).unique().tolist(),
        key=lambda name: variant_order.get(name, 10**6),
    )
    for variant in variants:
        variant_df = dataset_df[dataset_df["variant"] == variant].copy()
        rows.append(
            {
                "Variant": variant_labels.get(variant, variant),
                "Seeds": int(variant_df["seed"].nunique()),
                "Epochs": fmt_stat(variant_df["epochs"], digits=1),
                "PreFix-LSSP": fmt_stat(variant_df["pre_fix_lssp"]),
                "PreFix-SW": fmt_stat(variant_df["pre_fix_sw"]),
                "PostFix-LSSP": fmt_stat(variant_df["post_fix_lssp"]),
                "PostFix-SW": fmt_stat(variant_df["post_fix_sw"]),
                "Final-DAG": fmt_stat(variant_df["final_dag"]),
                "Final-LSSP": fmt_stat(variant_df["final_lssp"]),
                "Final-SW": fmt_stat(variant_df["final_sw"]),
                "Final-Report": fmt_stat(variant_df["final_report"]),
                "Opt(s)": fmt_stat(variant_df["opt_sec"]),
                "Post(s)": fmt_stat(variant_df["post_sec"]),
                "Total(s)": fmt_stat(variant_df["total_sec"]),
            }
        )

    print(f"Dataset: {dataset}")
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    comparison_rows = []
    for setting, left_variant, right_variant in component_comparison_rows:
        left_df = dataset_df[dataset_df["variant"] == left_variant].copy()
        right_df = dataset_df[dataset_df["variant"] == right_variant].copy()
        if left_df.empty and right_df.empty:
            continue
        comparison_rows.append(
            {
                "Setting": setting,
                "Diff-GNN w/o order": fmt_mean(left_df["final_lssp"]),
                "Diff-GNN": fmt_mean(right_df["final_lssp"]),
                "Improvement": fmt_improvement(left_df["final_lssp"], right_df["final_lssp"]),
            }
        )

    if comparison_rows:
        print("Component-addition summary (Final-LSSP):")
        print(pd.DataFrame(comparison_rows).to_string(index=False))
        print()

if warnings:
    print("Warnings:")
    for warning in sorted(dict.fromkeys(warnings)):
        print(f"  - {warning}")
PY
