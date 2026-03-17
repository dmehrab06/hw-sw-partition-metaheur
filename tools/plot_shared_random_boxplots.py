#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import subprocess
from pathlib import Path
from statistics import mean, median, stdev


RAW_METHODS = ("diff_gnn_order", "diff_gnn", "gl25", "gcps")
WITH_MIP_METHODS = ("diff_gnn_order", "mip", "diff_gnn", "gl25", "gcps")

DISPLAY_LABELS = {
    "diff_gnn_order": "diff_gnn_with_order",
    "diff_gnn": "diff_gnn_without_order",
    "gl25": "gl25_org",
    "gcps": "gcps_org",
    "mip": "mip",
}

PLOT_COLORS = {
    "diff_gnn_order": "#8fbc8f",
    "mip": "#d6a5a5",
    "diff_gnn": "#87a9c7",
    "gl25": "#d3d3d3",
    "gcps": "#c8c8c8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot shared-config makespan boxplots in the comparison_with_mip_5configs style."
    )
    parser.add_argument("--config-list", required=True, help="Path to shared_random_configs_*.txt")
    parser.add_argument("--diff-csv", required=True, help="CSV with diff_gnn results")
    parser.add_argument("--order-csv", required=True, help="CSV with diff_gnn_order results")
    parser.add_argument("--compare-csv", required=True, help="CSV with gcps/gl25 results")
    parser.add_argument("--mip-csv", required=True, help="CSV with mip_makespan values")
    parser.add_argument(
        "--out-dir",
        default="hw-sw-partition-metaheur/outputs/analysis_outputs",
        help="Output directory",
    )
    parser.add_argument("--tag", required=True, help="Tag suffix for generated files")
    return parser.parse_args()


def read_config_order(path: Path) -> list[str]:
    configs: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                configs.append(Path(line).stem)
    if not configs:
        raise ValueError(f"No configs found in {path}")
    return configs


def read_csv_map(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cfg = (row.get("Config") or "").strip()
            if not cfg:
                continue
            item = out.setdefault(cfg, {})
            for method in ("diff_gnn", "diff_gnn_order", "gl25", "gcps", "mip"):
                col = f"{method}_makespan"
                raw = (row.get(col) or "").strip()
                if raw:
                    item[method] = float(raw)
    return out


def build_rows(
    config_order: list[str],
    diff_map: dict[str, dict[str, float]],
    order_map: dict[str, dict[str, float]],
    compare_map: dict[str, dict[str, float]],
    mip_map: dict[str, dict[str, float]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for cfg in config_order:
        row: dict[str, float | str] = {"Config": cfg}
        row["diff_gnn"] = diff_map.get(cfg, {}).get("diff_gnn")
        row["diff_gnn_order"] = order_map.get(cfg, {}).get("diff_gnn_order")
        row["gl25"] = compare_map.get(cfg, {}).get("gl25")
        row["gcps"] = compare_map.get(cfg, {}).get("gcps")
        row["mip"] = mip_map.get(cfg, {}).get("mip")
        missing = [method for method in WITH_MIP_METHODS if row.get(method) is None]
        if missing:
            raise ValueError(f"Config {cfg} is missing methods: {', '.join(missing)}")
        rows.append(row)
    return rows


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(values)
    if not vals:
        raise ValueError("Cannot compute percentile of empty list")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return vals[low]
    frac = pos - low
    return vals[low] * (1.0 - frac) + vals[high] * frac


def summarize_methods(rows: list[dict[str, float | str]], methods: tuple[str, ...]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for method in methods:
        vals = [float(row[method]) for row in rows]
        summary.append(
            {
                "method": method,
                "label": DISPLAY_LABELS[method],
                "n": len(vals),
                "mean_makespan": mean(vals),
                "median_makespan": median(vals),
                "min_makespan": min(vals),
                "max_makespan": max(vals),
                "std_makespan": stdev(vals) if len(vals) > 1 else 0.0,
                "q1": percentile(vals, 0.25),
                "q3": percentile(vals, 0.75),
            }
        )
    summary.sort(key=lambda item: float(item["mean_makespan"]))
    return summary


def write_csv(path: Path, rows: list[dict[str, float | str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: float) -> str:
    return f"{value:.1f}"


def nice_top(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for factor in (1, 1.2, 1.5, 2, 2.5, 5, 10):
        candidate = factor * magnitude
        if candidate >= value:
            return candidate
    return 10 * magnitude


def svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    attr_str = " ".join(f'{k}="{html.escape(str(v), quote=True)}"' for k, v in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_str}>{html.escape(text)}</text>'


def y_map(value: float, plot_top: float, plot_height: float, y_max: float) -> float:
    return plot_top + plot_height - (value / y_max) * plot_height


def write_boxplot_svg(
    path: Path,
    summary: list[dict[str, float | str]],
    title: str,
) -> None:
    width = 1280
    height = 768
    left = 85
    right = 15
    top = 40
    bottom = 145
    plot_width = width - left - right
    plot_height = height - top - bottom
    bg = "#ffffff"
    grid = "#e5e7eb"

    y_max = nice_top(max(float(item["max_makespan"]) for item in summary) * 1.12)
    tick_step = 1000 if y_max > 4000 else max(100.0, y_max / 8.0)
    ticks: list[float] = []
    t = 0.0
    while t <= y_max + 1e-9:
        ticks.append(t)
        t += tick_step

    n = len(summary)
    slot = plot_width / max(n, 1)
    box_width = min(120.0, slot * 0.5)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>')

    for tick in ticks:
        y = y_map(tick, top, plot_height, y_max)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="{grid}" stroke-width="1" stroke-dasharray="4 4"/>'
        )
        if tick > 0:
            parts.append(
                svg_text(
                    left - 12,
                    y + 5,
                    str(int(tick)),
                    fill="#222222",
                    **{"font-size": 12, "font-family": "sans-serif", "text-anchor": "end"},
                )
            )

    parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#444444" stroke-width="1.2"/>')

    mean_points: list[tuple[float, float, float]] = []
    for idx, item in enumerate(summary):
        cx = left + slot * idx + slot / 2.0
        q1 = float(item["q1"])
        q3 = float(item["q3"])
        med = float(item["median_makespan"])
        vmin = float(item["min_makespan"])
        vmax = float(item["max_makespan"])
        avg = float(item["mean_makespan"])
        label = str(item["label"])
        method = str(item["method"])

        y_q1 = y_map(q1, top, plot_height, y_max)
        y_q3 = y_map(q3, top, plot_height, y_max)
        y_med = y_map(med, top, plot_height, y_max)
        y_min = y_map(vmin, top, plot_height, y_max)
        y_maxv = y_map(vmax, top, plot_height, y_max)
        y_avg = y_map(avg, top, plot_height, y_max)

        parts.append(f'<line x1="{cx:.2f}" y1="{y_maxv:.2f}" x2="{cx:.2f}" y2="{y_q3:.2f}" stroke="#555555" stroke-width="2"/>')
        parts.append(f'<line x1="{cx:.2f}" y1="{y_q1:.2f}" x2="{cx:.2f}" y2="{y_min:.2f}" stroke="#555555" stroke-width="2"/>')
        cap_half = box_width * 0.25
        parts.append(f'<line x1="{cx-cap_half:.2f}" y1="{y_maxv:.2f}" x2="{cx+cap_half:.2f}" y2="{y_maxv:.2f}" stroke="#555555" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-cap_half:.2f}" y1="{y_min:.2f}" x2="{cx+cap_half:.2f}" y2="{y_min:.2f}" stroke="#555555" stroke-width="2"/>')
        parts.append(
            f'<rect x="{cx - box_width/2:.2f}" y="{y_q3:.2f}" width="{box_width:.2f}" height="{max(1.0, y_q1 - y_q3):.2f}" '
            f'fill="{PLOT_COLORS[method]}" fill-opacity="0.65" stroke="#8c8c8c" stroke-width="2"/>'
        )
        parts.append(f'<line x1="{cx-box_width/2:.2f}" y1="{y_med:.2f}" x2="{cx+box_width/2:.2f}" y2="{y_med:.2f}" stroke="#f39c12" stroke-width="2"/>')

        mean_points.append((cx, y_avg, avg))

        parts.append(
            f'<g transform="translate({cx:.2f},{top + plot_height + 40:.2f}) rotate(-33)">' +
            svg_text(
                0,
                0,
                label,
                fill="#222222",
                **{"font-size": 15, "font-family": "sans-serif", "text-anchor": "end"},
            ) +
            "</g>"
        )

    if mean_points:
        points_attr = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in mean_points)
        parts.append(f'<polyline points="{points_attr}" fill="none" stroke="#111111" stroke-width="2"/>')
        for x, y, avg in mean_points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#000000"/>')
            parts.append(
                svg_text(
                    x + 12,
                    y + 4,
                    format_value(avg),
                    fill="#333333",
                    **{"font-size": 12, "font-family": "sans-serif"},
                )
            )

    title_x = width / 2.0
    parts.append(
        svg_text(
            title_x,
            32,
            title,
            fill="#111111",
            **{"font-size": 26, "font-family": "sans-serif", "text-anchor": "middle"},
        )
    )
    parts.append(
        svg_text(
            28,
            height / 2.0,
            "Makespan (lower is better)",
            fill="#111111",
            **{
                "font-size": 16,
                "font-family": "sans-serif",
                "text-anchor": "middle",
                "transform": f"rotate(-90, 28, {height / 2.0:.2f})",
            },
        )
    )

    leg_x = width - 150
    leg_y = 56
    parts.append(f'<line x1="{leg_x}" y1="{leg_y}" x2="{leg_x+35}" y2="{leg_y}" stroke="#111111" stroke-width="2"/>')
    parts.append(f'<circle cx="{leg_x+18}" cy="{leg_y}" r="4.5" fill="#000000"/>')
    parts.append(
        svg_text(
            leg_x + 48,
            leg_y + 4,
            "Mean",
            fill="#111111",
            **{"font-size": 14, "font-family": "sans-serif"},
        )
    )

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def convert_svg(svg_path: Path, png_path: Path, pdf_path: Path) -> None:
    subprocess.run(
        ["rsvg-convert", "-o", str(png_path), str(svg_path)],
        check=True,
    )
    subprocess.run(
        ["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tag = args.tag

    config_order = read_config_order(Path(args.config_list))
    rows = build_rows(
        config_order,
        read_csv_map(Path(args.diff_csv)),
        read_csv_map(Path(args.order_csv)),
        read_csv_map(Path(args.compare_csv)),
        read_csv_map(Path(args.mip_csv)),
    )

    detailed_rows = []
    for row in rows:
        detailed_rows.append(
            {
                "Config": row["Config"],
                "mip": row["mip"],
                "diff_gnn_with_order": row["diff_gnn_order"],
                "diff_gnn_without_order": row["diff_gnn"],
                "gl25_org": row["gl25"],
                "gcps_org": row["gcps"],
            }
        )

    without_mip_summary = summarize_methods(rows, RAW_METHODS)
    with_mip_summary = summarize_methods(rows, WITH_MIP_METHODS)

    detailed_path = out_dir / f"comparison_with_mip_like_provided_style_15configs_{tag}_detailed.csv"
    without_summary_path = out_dir / f"comparison_like_provided_style_15configs_{tag}_summary.csv"
    with_summary_path = out_dir / f"comparison_with_mip_like_provided_style_15configs_{tag}_summary.csv"

    write_csv(
        detailed_path,
        detailed_rows,
        ["Config", "mip", "diff_gnn_with_order", "diff_gnn_without_order", "gl25_org", "gcps_org"],
    )
    write_csv(
        without_summary_path,
        [
            {
                "method": row["method"],
                "label": row["label"],
                "n": row["n"],
                "mean_makespan": row["mean_makespan"],
                "median_makespan": row["median_makespan"],
                "min_makespan": row["min_makespan"],
                "max_makespan": row["max_makespan"],
            }
            for row in without_mip_summary
        ],
        ["method", "label", "n", "mean_makespan", "median_makespan", "min_makespan", "max_makespan"],
    )
    write_csv(
        with_summary_path,
        [
            {
                "method": row["label"],
                "n": row["n"],
                "mean_makespan": row["mean_makespan"],
                "median_makespan": row["median_makespan"],
                "min_makespan": row["min_makespan"],
                "max_makespan": row["max_makespan"],
                "std_makespan": row["std_makespan"],
            }
            for row in with_mip_summary
        ],
        ["method", "n", "mean_makespan", "median_makespan", "min_makespan", "max_makespan", "std_makespan"],
    )

    without_svg = out_dir / f"comparison_like_provided_style_15configs_{tag}.svg"
    without_png = out_dir / f"comparison_like_provided_style_15configs_{tag}.png"
    without_pdf = out_dir / f"comparison_like_provided_style_15configs_{tag}.pdf"

    with_svg = out_dir / f"comparison_with_mip_like_provided_style_15configs_{tag}.svg"
    with_png = out_dir / f"comparison_with_mip_like_provided_style_15configs_{tag}.png"
    with_pdf = out_dir / f"comparison_with_mip_like_provided_style_15configs_{tag}.pdf"

    write_boxplot_svg(
        without_svg,
        without_mip_summary,
        "DiffGNN (With/Without Order) vs Baseline Methods (Sorted by Mean)",
    )
    write_boxplot_svg(
        with_svg,
        with_mip_summary,
        "MIP vs DiffGNN (With/Without Order) vs Baseline Methods (Sorted by Mean)",
    )
    convert_svg(without_svg, without_png, without_pdf)
    convert_svg(with_svg, with_png, with_pdf)

    print(f"Wrote {without_png}")
    print(f"Wrote {with_png}")
    print(f"Wrote {without_summary_path}")
    print(f"Wrote {with_summary_path}")
    print(f"Wrote {detailed_path}")


if __name__ == "__main__":
    main()
