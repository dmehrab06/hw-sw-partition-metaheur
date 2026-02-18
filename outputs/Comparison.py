#!/usr/bin/env python3
"""
Analyze + visualize makespan results vs a MIP (optimal) baseline.

Inputs (your two "excel sheets" are CSVs here):
  1) mip_makespan_opt-result-summary-soda-graphs-config.csv  (has mip_makespan)
  2) makespan_opt-result-summary-soda-graphs-config.csv      (either wide with <method>_makespan,
     or long with columns like method + makespan [+ time])

Outputs (saved into --out_dir):
  - Bar plots (mean gap/ratio, win counts)
  - Scatter plots (method vs MIP, runtime vs gap)
  - Boxplots + ECDFs for gap distributions
  - Pairwise bar plots (our GNN vs baselines)
  - Heatmaps across (Area_Percentage, HW_Scale_Factor)
  - CSV summaries + LaTeX tables (.tex)

Run:
  python analyze_makespan.py \
    --mip_csv /mnt/data/mip_makespan_opt-result-summary-soda-graphs-config.csv \
    --methods_csv /mnt/data/makespan_opt-result-summary-soda-graphs-config.csv \
    --gnn_method diff_gnn
"""

import argparse
from pathlib import Path
import re
import math
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p)


def detect_methods(df: pd.DataFrame):
    """Methods are inferred from columns like '<method>_makespan'."""
    methods = []
    for c in df.columns:
        m = re.match(r"^(.*)_makespan$", str(c))
        if m:
            name = m.group(1)
            if str(name).lower().startswith("mip"):
                continue
            methods.append(name)
    return sorted(set(methods))


def detect_long_format(df: pd.DataFrame):
    cols = {str(c).lower(): c for c in df.columns}
    if "method" in cols and "makespan" in cols:
        method_col = cols["method"]
        makespan_col = cols["makespan"]
        time_col = None
        for cand in ["time_s", "runtime_s", "time", "runtime"]:
            if cand in cols:
                time_col = cols[cand]
                break
        return method_col, makespan_col, time_col
    return None, None, None


def pick_default_gnn_method(methods):
    # Prefer a method containing 'gnn', else common names, else first method.
    for cand in methods:
        if "gnn" in cand.lower():
            return cand
    for cand in ["diff_gnn", "gnn", "graph_gnn"]:
        if cand in methods:
            return cand
    return methods[0] if methods else None


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def build_long_df(merged: pd.DataFrame, methods):
    rows = []
    for m in methods:
        ms_col = f"{m}_makespan"
        t_col = f"{m}_time"
        if ms_col not in merged.columns:
            continue

        for _, r in merged.iterrows():
            mip = safe_float(r["mip_makespan"])
            ms = safe_float(r[ms_col])
            if not (np.isfinite(mip) and mip > 0 and np.isfinite(ms)):
                continue

            t = safe_float(r[t_col]) if t_col in merged.columns else np.nan
            gap = (ms - mip) / mip
            ratio = ms / mip

            rows.append(
                {
                    "Config": r.get("Config"),
                    "GraphName": r.get("GraphName"),
                    "N": safe_float(r.get("N")),
                    "Area_Percentage": safe_float(r.get("Area_Percentage")),
                    "HW_Scale_Factor": safe_float(r.get("HW_Scale_Factor")),
                    "Seed": r.get("Seed"),
                    "method": m,
                    "makespan": ms,
                    "time_s": t,
                    "mip_makespan": mip,
                    "gap": gap,
                    "gap_%": 100.0 * gap,
                    "ratio": ratio,
                }
            )
    return pd.DataFrame(rows)


def build_long_df_from_long(merged: pd.DataFrame, method_col: str, makespan_col: str, time_col: str | None):
    df = merged.copy()
    df["method"] = df[method_col].astype(str)
    df["makespan"] = df[makespan_col].apply(safe_float)
    df["mip_makespan"] = df["mip_makespan"].apply(safe_float)

    if time_col is not None:
        df["time_s"] = df[time_col].apply(safe_float)
    else:
        df["time_s"] = np.nan

    df = df[np.isfinite(df["mip_makespan"]) & (df["mip_makespan"] > 0) & np.isfinite(df["makespan"])]
    df["gap"] = (df["makespan"] - df["mip_makespan"]) / df["mip_makespan"]
    df["gap_%"] = 100.0 * df["gap"]
    df["ratio"] = df["makespan"] / df["mip_makespan"]

    keep = [
        "Config",
        "GraphName",
        "N",
        "Area_Percentage",
        "HW_Scale_Factor",
        "Seed",
        "method",
        "makespan",
        "time_s",
        "mip_makespan",
        "gap",
        "gap_%",
        "ratio",
    ]
    existing = [c for c in keep if c in df.columns]
    return df[existing].copy()


def build_mip_long(merged: pd.DataFrame):
    if "mip_makespan" not in merged.columns:
        return pd.DataFrame()
    df = merged.copy()
    df["method"] = "MIP"
    df["makespan"] = df["mip_makespan"].apply(safe_float)
    df["mip_makespan"] = df["mip_makespan"].apply(safe_float)
    if "mip_time" in df.columns:
        df["time_s"] = df["mip_time"].apply(safe_float)
    else:
        df["time_s"] = np.nan

    df = df[np.isfinite(df["mip_makespan"]) & (df["mip_makespan"] > 0) & np.isfinite(df["makespan"])]
    df["gap"] = 0.0
    df["gap_%"] = 0.0
    df["ratio"] = 1.0

    keep = [
        "Config",
        "GraphName",
        "N",
        "Area_Percentage",
        "HW_Scale_Factor",
        "Seed",
        "method",
        "makespan",
        "time_s",
        "mip_makespan",
        "gap",
        "gap_%",
        "ratio",
    ]
    existing = [c for c in keep if c in df.columns]
    return df[existing].copy()


def compute_overall_stats(long_df: pd.DataFrame):
    # wins computed over *heuristics/methods in methods_csv* (not counting MIP)
    best_by_cfg = (
        long_df.groupby("Config")["makespan"].min().rename("best_makespan").reset_index()
    )
    tmp = long_df.merge(best_by_cfg, on="Config", how="left")
    tmp["is_win"] = (tmp["makespan"] == tmp["best_makespan"]).astype(int)

    g = tmp.groupby("method")
    out = pd.DataFrame(
        {
            "mean_gap_%": g["gap_%"].mean(),
            "median_gap_%": g["gap_%"].median(),
            "p95_gap_%": g["gap_%"].quantile(0.95),
            "mean_ratio": g["ratio"].mean(),
            "median_ratio": g["ratio"].median(),
            "within_1%": (g["gap"].apply(lambda s: (s <= 0.01).mean()) * 100.0),
            "within_5%": (g["gap"].apply(lambda s: (s <= 0.05).mean()) * 100.0),
            "within_10%": (g["gap"].apply(lambda s: (s <= 0.10).mean()) * 100.0),
            "wins": g["is_win"].sum(),
            "win_rate_%": g["is_win"].mean() * 100.0,
            "mean_time_s": g["time_s"].mean(),
            "median_time_s": g["time_s"].median(),
            "n_instances": g.size(),
        }
    ).reset_index()

    out = out.sort_values("mean_gap_%", ascending=True)
    return out


def compute_pairwise_vs_ours(long_df: pd.DataFrame, ours: str):
    wide = long_df.pivot_table(index="Config", columns="method", values="makespan", aggfunc="first")
    if ours not in wide.columns:
        return pd.DataFrame()

    rows = []
    ours_ms = wide[ours]
    for m in wide.columns:
        if m == ours:
            continue
        ms = wide[m]
        both = pd.concat([ours_ms, ms], axis=1).dropna()
        if both.empty:
            continue
        ours_v = both.iloc[:, 0]
        base_v = both.iloc[:, 1]

        better_pct = (ours_v < base_v).mean() * 100.0
        # Positive means ours is better (lower makespan)
        rel_impr = ((base_v - ours_v) / base_v).replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "baseline": m,
                "ours_better_%": better_pct,
                "mean_rel_impr_%": 100.0 * rel_impr.mean(),
                "median_rel_impr_%": 100.0 * rel_impr.median(),
                "mean_abs_diff": (base_v - ours_v).mean(),
                "median_abs_diff": (base_v - ours_v).median(),
                "n": len(both),
            }
        )
    return pd.DataFrame(rows).sort_values("ours_better_%", ascending=False)


# -------------------- Plot helpers (matplotlib only) --------------------

def savefig(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    # also save a vector copy when possible
    if outpath.suffix.lower() == ".png":
        fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def plot_bar(summary: pd.DataFrame, y_col: str, title: str, ylabel: str, outpath: Path, top_k=None):
    dfp = summary.copy()
    if top_k is not None:
        dfp = dfp.head(top_k)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(dfp))
    ax.bar(x, dfp[y_col].astype(float))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(dfp["method"].astype(str), rotation=45, ha="right")
    savefig(fig, outpath)


def plot_box_gap(long_df: pd.DataFrame, methods_order, outpath: Path, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data = [long_df.loc[long_df["method"] == m, "gap_%"].dropna().values for m in methods_order]
    try:
        ax.boxplot(data, tick_labels=methods_order, showfliers=False)
    except TypeError:
        # matplotlib <3.9 uses 'labels'
        ax.boxplot(data, labels=methods_order, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel("Gap to MIP (%)  (lower is better)")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    savefig(fig, outpath)


def plot_ecdf(long_df: pd.DataFrame, methods_order, outpath: Path, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for m in methods_order:
        vals = np.sort(long_df.loc[long_df["method"] == m, "gap_%"].dropna().values)
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=m)
    ax.set_title(title)
    ax.set_xlabel("Gap to MIP (%)  (lower is better)")
    ax.set_ylabel("ECDF")
    ax.legend()
    savefig(fig, outpath)


def plot_scatter_vs_mip(long_df: pd.DataFrame, method: str, outpath: Path):
    dfm = long_df[long_df["method"] == method].dropna(subset=["mip_makespan", "makespan"])
    if dfm.empty:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dfm["mip_makespan"], dfm["makespan"], alpha=0.7)
    # y = x line
    mn = min(dfm["mip_makespan"].min(), dfm["makespan"].min())
    mx = max(dfm["mip_makespan"].max(), dfm["makespan"].max())
    ax.plot([mn, mx], [mn, mx])
    ax.set_title(f"{method}: makespan vs MIP-opt makespan")
    ax.set_xlabel("MIP makespan (optimal / best-known)")
    ax.set_ylabel(f"{method} makespan")
    savefig(fig, outpath)


def plot_scatter_runtime_gap(long_df: pd.DataFrame, method: str, outpath: Path):
    dfm = long_df[long_df["method"] == method].dropna(subset=["time_s", "gap_%"])
    if dfm.empty:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dfm["time_s"], dfm["gap_%"], alpha=0.7)
    ax.set_title(f"{method}: runtime vs gap to MIP")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Gap to MIP (%)")
    savefig(fig, outpath)


def plot_pareto_methods(summary: pd.DataFrame, outpath: Path):
    dfp = summary.dropna(subset=["mean_time_s", "mean_gap_%"]).copy()
    if dfp.empty:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dfp["mean_time_s"], dfp["mean_gap_%"])
    for _, r in dfp.iterrows():
        ax.annotate(str(r["method"]), (r["mean_time_s"], r["mean_gap_%"]))
    ax.set_title("Method-level tradeoff: mean runtime vs mean gap to MIP")
    ax.set_xlabel("Mean runtime (s)  (lower is better)")
    ax.set_ylabel("Mean gap to MIP (%)  (lower is better)")
    savefig(fig, outpath)


def plot_heatmap_group(long_df: pd.DataFrame, method: str, outpath: Path):
    dfm = long_df[long_df["method"] == method].copy()
    if dfm.empty:
        return
    piv = dfm.pivot_table(
        index="Area_Percentage",
        columns="HW_Scale_Factor",
        values="gap_%",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    if piv.empty:
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_title(f"{method}: mean gap (%) across (Area_Percentage, HW_Scale_Factor)")
    ax.set_xlabel("HW_Scale_Factor")
    ax.set_ylabel("Area_Percentage")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_yticks(range(piv.shape[0]))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticklabels([str(i) for i in piv.index])
    fig.colorbar(im, ax=ax, label="Mean gap (%)")
    savefig(fig, outpath)


# -------------------- LaTeX tables --------------------

def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    def _supports_arg(func, name: str) -> bool:
        try:
            return name in inspect.signature(func).parameters
        except Exception:
            return False

    latex_kwargs = dict(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.3f}",
    )
    if _supports_arg(pd.DataFrame.to_latex, "longtable"):
        latex_kwargs["longtable"] = False
    if _supports_arg(pd.DataFrame.to_latex, "booktabs"):
        latex_kwargs["booktabs"] = True
    elif _supports_arg(pd.DataFrame.to_latex, "hrules"):
        latex_kwargs["hrules"] = True

    tex_body = df.to_latex(**latex_kwargs)
    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"{tex_body}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    return tex


def plot_pairwise_bar(pairwise: pd.DataFrame, y_col: str, title: str, ylabel: str, outpath: Path):
    if pairwise.empty:
        return
    dfp = pairwise.copy().sort_values(y_col, ascending=False)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(len(dfp))
    ax.bar(x, dfp[y_col].astype(float))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(dfp["baseline"].astype(str), rotation=45, ha="right")
    savefig(fig, outpath)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--methods_csv",
        type=str,
        default="/people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/outputs/makespan_opt-result-summary-soda-graphs-config.csv",
        help="CSV with method results (either wide with <method>_makespan or long with method+makespan).",
    )

    ap.add_argument(
        "--mip_csv",
        type=str,
        default="/people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/outputs/mip_makespan_opt-result-summary-soda-graphs-config.csv",
        help="CSV with MIP baseline (must contain mip_makespan).",
    )

    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. If not set, uses <methods_csv_dir>/analysis_outputs",
    )

    ap.add_argument(
        "--gnn_method",
        type=str,
        default="diff_gnn",   # change to your exact method name if different
        help="Name of your GNN-based method (e.g., diff_gnn).",
    )

    ap.add_argument(
        "--top_k_for_ecdf",
        type=int,
        default=6,
        help="How many top methods (by mean gap) to include in ECDF plot.",
    )

    args = ap.parse_args()

    # Default out_dir if not provided
    if args.out_dir is None:
        args.out_dir = str(Path(args.methods_csv).resolve().parent / "analysis_outputs")

    mip = read_table(args.mip_csv)
    methods_df = read_table(args.methods_csv)

    if "Config" not in mip.columns or "Config" not in methods_df.columns:
        raise ValueError("Both files must contain a 'Config' column to join on.")

    if "mip_makespan" not in mip.columns:
        raise ValueError("MIP file must contain 'mip_makespan'.")

    method_col, makespan_col, time_col = detect_long_format(methods_df)
    if method_col and makespan_col:
        methods = sorted(methods_df[method_col].dropna().astype(str).unique().tolist())
    else:
        methods = detect_methods(methods_df)

    if not methods:
        raise ValueError(
            "No methods detected. Expected either columns 'method' + 'makespan' "
            "or wide columns like '<method>_makespan' in methods_csv."
        )

    gnn_method = args.gnn_method or pick_default_gnn_method(methods)
    if gnn_method not in methods:
        raise ValueError(f"--gnn_method '{gnn_method}' not found among detected methods: {methods}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.methods_csv).resolve().parent / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge
    mip_keep = ["Config", "mip_makespan"]
    merged = methods_df.merge(mip[mip_keep], on="Config", how="inner")
    if merged.empty:
        raise ValueError("Join produced 0 rows. Are Config values matching between the two files?")

    # Long-form
    if method_col and makespan_col:
        long_df = build_long_df_from_long(merged, method_col, makespan_col, time_col)
    else:
        long_df = build_long_df(merged, methods)

    # Include MIP baseline in a separate long-form for makespan-only charts
    long_mip = build_mip_long(merged)
    long_all = pd.concat([long_df, long_mip], ignore_index=True) if not long_mip.empty else long_df.copy()

    # Stats
    overall = compute_overall_stats(long_df)
    pairwise = compute_pairwise_vs_ours(long_df, gnn_method)

    # Save CSV summaries
    overall.to_csv(out_dir / "overall_summary.csv", index=False)
    pairwise.to_csv(out_dir / "pairwise_vs_ours.csv", index=False)
    long_df.to_csv(out_dir / "long_results.csv", index=False)

    # Determine plotting order (best to worst by mean gap)
    order = overall["method"].tolist()

    # -------------------- Plots --------------------
    plot_bar(
        overall,
        y_col="mean_gap_%",
        title="Mean gap to MIP (lower is better)",
        ylabel="Mean gap (%)",
        outpath=out_dir / "bar_mean_gap.png",
    )

    # Mean makespan including MIP baseline
    makespan_summary = (
        long_all.groupby("method")["makespan"].mean().reset_index().rename(columns={"makespan": "mean_makespan"})
    )
    makespan_summary = makespan_summary.sort_values("mean_makespan", ascending=True)
    plot_bar(
        makespan_summary,
        y_col="mean_makespan",
        title="Mean makespan (including MIP baseline)",
        ylabel="Mean makespan",
        outpath=out_dir / "bar_mean_makespan_including_mip.png",
    )

    plot_bar(
        overall,
        y_col="win_rate_%",
        title="Win rate across instances (best among non-MIP methods)",
        ylabel="Win rate (%)",
        outpath=out_dir / "bar_win_rate.png",
    )

    plot_box_gap(
        long_df,
        methods_order=order,
        outpath=out_dir / "box_gap.png",
        title="Gap distribution to MIP (boxplot, no fliers)",
    )

    plot_ecdf(
        long_df,
        methods_order=order[: min(args.top_k_for_ecdf, len(order))],
        outpath=out_dir / "ecdf_gap_topk.png",
        title="ECDF of gap to MIP (top methods by mean gap)",
    )

    plot_scatter_vs_mip(long_df, gnn_method, out_dir / f"scatter_vs_mip_{gnn_method}.png")
    plot_scatter_runtime_gap(long_df, gnn_method, out_dir / f"scatter_runtime_gap_{gnn_method}.png")

    plot_pareto_methods(overall, out_dir / "scatter_pareto_methods.png")

    plot_heatmap_group(long_df, gnn_method, out_dir / f"heatmap_gap_{gnn_method}.png")

    # Pairwise visuals: our GNN vs each baseline
    plot_pairwise_bar(
        pairwise,
        y_col="ours_better_%",
        title=f"{gnn_method} better than baseline (% of instances)",
        ylabel="Ours better (%)",
        outpath=out_dir / "bar_ours_better_pct.png",
    )

    plot_pairwise_bar(
        pairwise,
        y_col="mean_rel_impr_%",
        title=f"{gnn_method} mean relative improvement vs baseline",
        ylabel="Mean improvement (%)",
        outpath=out_dir / "bar_ours_mean_rel_impr.png",
    )

    # Optional: also plot the best method (by mean gap) for comparison
    best_method = order[0]
    if best_method != gnn_method:
        plot_scatter_vs_mip(long_df, best_method, out_dir / f"scatter_vs_mip_{best_method}.png")
        plot_heatmap_group(long_df, best_method, out_dir / f"heatmap_gap_{best_method}.png")

    # -------------------- LaTeX tables --------------------
    # Paper-friendly summary table
    table_cols = [
        "method",
        "mean_gap_%",
        "median_gap_%",
        "p95_gap_%",
        "mean_ratio",
        "win_rate_%",
        "mean_time_s",
        "n_instances",
    ]
    summary_for_tex = overall[table_cols].copy()

    tex1 = to_latex_table(
        summary_for_tex,
        caption="Overall makespan performance relative to the MIP baseline (lower is better). Gap/ratio are computed as method vs MIP makespan.",
        label="tab:mkspan_overall_vs_mip",
    )
    (out_dir / "table_overall_vs_mip.tex").write_text(tex1)

    # Pairwise table: ours vs each baseline
    if not pairwise.empty:
        tex2 = to_latex_table(
            pairwise,
            caption=f"Pairwise comparison of our method ({gnn_method}) vs each baseline (positive improvement means lower makespan).",
            label="tab:mkspan_pairwise_ours",
        )
        (out_dir / "table_pairwise_ours.tex").write_text(tex2)

    # Console highlights (so you immediately see “how good” the GNN method is)
    ours_row = overall[overall["method"] == gnn_method]
    if not ours_row.empty:
        r = ours_row.iloc[0].to_dict()
        print("\n=== OUR METHOD (GNN) SUMMARY ===")
        print(f"method            : {gnn_method}")
        print(f"mean_gap_%         : {r['mean_gap_%']:.3f}")
        print(f"median_gap_%       : {r['median_gap_%']:.3f}")
        print(f"mean_ratio         : {r['mean_ratio']:.3f}")
        print(f"win_rate_%         : {r['win_rate_%']:.3f}")
        print(f"mean_time_s        : {r['mean_time_s']:.3f}")
        print(f"outputs written to : {out_dir}")

    print("\nTop methods by mean gap to MIP:")
    print(overall[["method", "mean_gap_%", "median_gap_%", "win_rate_%", "mean_time_s"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
