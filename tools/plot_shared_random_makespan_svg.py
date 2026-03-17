#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from statistics import mean, median


METHODS = ("diff_gnn", "diff_gnn_order", "gcps", "gl25")
METHOD_LABELS = {
    "diff_gnn": "Diff-GNN",
    "diff_gnn_order": "Diff-GNN+Order",
    "gcps": "GCPS",
    "gl25": "GL25",
}
METHOD_COLORS = {
    "diff_gnn": "#1f77b4",
    "diff_gnn_order": "#ff7f0e",
    "gcps": "#2ca02c",
    "gl25": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge shared-config makespan CSVs and emit dependency-free SVG bar charts."
    )
    parser.add_argument("--config-list", required=True, help="Path to shared_random_configs_*.txt")
    parser.add_argument("--diff-csv", required=True, help="CSV containing diff_gnn results")
    parser.add_argument("--order-csv", required=True, help="CSV containing diff_gnn_order results")
    parser.add_argument("--compare-csv", required=True, help="CSV containing gcps/gl25 results")
    parser.add_argument(
        "--out-dir",
        default="outputs/analysis_outputs",
        help="Directory for merged CSVs and SVG outputs",
    )
    parser.add_argument(
        "--tag",
        default="shared_random_makespan",
        help="Filename prefix for generated artifacts",
    )
    return parser.parse_args()


def read_config_order(path: Path) -> list[str]:
    configs: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            configs.append(Path(line).stem)
    if not configs:
        raise ValueError(f"No configs found in {path}")
    return configs


def read_makespan_rows(path: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "Config" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing required 'Config' column")
        for row in reader:
            cfg = (row.get("Config") or "").strip()
            if not cfg:
                continue
            bucket = results.setdefault(cfg, {})
            for method in METHODS:
                col = f"{method}_makespan"
                raw = (row.get(col) or "").strip()
                if raw:
                    bucket[method] = float(raw)
    return results


def merge_results(config_order: list[str], csv_maps: list[dict[str, dict[str, float]]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    expected = set(config_order)
    observed = set()
    for csv_map in csv_maps:
        observed.update(csv_map.keys())
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "Config mismatch between shared config list and CSV outputs. "
            f"Missing={missing}, Extra={extra}"
        )

    for index, cfg in enumerate(config_order, start=1):
        row: dict[str, object] = {
            "config_index": index,
            "Config": cfg,
        }
        for method in METHODS:
            value = None
            for csv_map in csv_maps:
                if cfg in csv_map and method in csv_map[cfg]:
                    value = csv_map[cfg][method]
                    break
            row[f"{method}_makespan"] = value
        missing_methods = [m for m in METHODS if row[f"{m}_makespan"] is None]
        if missing_methods:
            raise ValueError(f"Config {cfg} is missing makespan values for: {', '.join(missing_methods)}")
        merged.append(row)
    return merged


def compute_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_counts = {method: 0 for method in METHODS}
    for row in rows:
        row_values = {method: float(row[f"{method}_makespan"]) for method in METHODS}
        best_value = min(row_values.values())
        for method, value in row_values.items():
            if abs(value - best_value) < 1e-9:
                best_counts[method] += 1

    summary: list[dict[str, object]] = []
    for method in METHODS:
        values = [float(row[f"{method}_makespan"]) for row in rows]
        summary.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "mean_makespan": mean(values),
                "median_makespan": median(values),
                "best_count": best_counts[method],
            }
        )
    summary.sort(key=lambda item: float(item["mean_makespan"]))
    return summary


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def nice_axis_max(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for factor in (1, 2, 5, 10):
        candidate = factor * magnitude
        if candidate >= value:
            return float(candidate)
    return float(10 * magnitude)


def format_number(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}"


def svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    attr_str = " ".join(f'{key}="{html.escape(str(val), quote=True)}"' for key, val in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_str}>{html.escape(text)}</text>'


def write_grouped_svg(path: Path, rows: list[dict[str, object]], title: str, subtitle: str) -> None:
    width = max(1500, 110 * len(rows) + 260)
    height = 760
    left = 85
    right = 30
    top = 90
    bottom = 180
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = nice_axis_max(
        max(float(row[f"{method}_makespan"]) for row in rows for method in METHODS) * 1.08
    )
    tick_count = 6
    group_width = plot_width / max(len(rows), 1)
    bar_width = min(22.0, group_width / (len(METHODS) + 1.25))
    group_inner = bar_width * len(METHODS)
    group_offset = (group_width - group_inner) / 2.0

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}">'
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(svg_text(left, 36, title, fill="#111111", **{"font-size": 24, "font-family": "sans-serif", "font-weight": 700}))
    parts.append(svg_text(left, 62, subtitle, fill="#4b5563", **{"font-size": 13, "font-family": "sans-serif"}))

    for idx in range(tick_count + 1):
        value = y_max * idx / tick_count
        y = top + plot_height - (value / y_max) * plot_height
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                left - 10,
                y + 4,
                format_number(value),
                fill="#4b5563",
                **{"font-size": 12, "font-family": "sans-serif", "text-anchor": "end"},
            )
        )

    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111111" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#111111" stroke-width="1.2"/>'
    )

    legend_x = width - right - 350
    legend_y = 36
    for index, method in enumerate(METHODS):
        x = legend_x + index * 84
        parts.append(
            f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{METHOD_COLORS[method]}" rx="2"/>'
        )
        parts.append(
            svg_text(
                x + 20,
                legend_y,
                METHOD_LABELS[method],
                fill="#111111",
                **{"font-size": 12, "font-family": "sans-serif"},
            )
        )

    for group_index, row in enumerate(rows):
        group_left = left + group_index * group_width + group_offset
        for method_index, method in enumerate(METHODS):
            value = float(row[f"{method}_makespan"])
            bar_height = (value / y_max) * plot_height
            x = group_left + method_index * bar_width
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2:.2f}" height="{bar_height:.2f}" '
                f'fill="{METHOD_COLORS[method]}"><title>{html.escape(str(row["Config"]))} | '
                f'{METHOD_LABELS[method]} | makespan={format_number(value)}</title></rect>'
            )
        label_x = left + group_index * group_width + group_width / 2.0
        parts.append(
            svg_text(
                label_x,
                top + plot_height + 18,
                f"C{row['config_index']}",
                fill="#111111",
                **{"font-size": 11, "font-family": "sans-serif", "text-anchor": "middle"},
            )
        )

    parts.append(
        svg_text(
            width / 2.0,
            height - 110,
            "Config index",
            fill="#111111",
            **{"font-size": 13, "font-family": "sans-serif", "text-anchor": "middle", "font-weight": 700},
        )
    )
    parts.append(
        f'<g transform="translate(22,{top + plot_height / 2.0}) rotate(-90)">' +
        svg_text(
            0,
            0,
            "Makespan",
            fill="#111111",
            **{"font-size": 13, "font-family": "sans-serif", "text-anchor": "middle", "font-weight": 700},
        ) +
        "</g>"
    )

    parts.append(
        svg_text(
            left,
            height - 78,
            "Config mapping is in the merged CSV: config_index -> Config.",
            fill="#4b5563",
            **{"font-size": 12, "font-family": "sans-serif"},
        )
    )
    parts.append(
        svg_text(
            left,
            height - 56,
            "MIP is omitted here because the matching 20260226_070718 MIP run timed out on all 15 configs.",
            fill="#4b5563",
            **{"font-size": 12, "font-family": "sans-serif"},
        )
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def write_mean_svg(path: Path, summary_rows: list[dict[str, object]], title: str, subtitle: str) -> None:
    width = 900
    height = 540
    left = 85
    right = 35
    top = 90
    bottom = 95
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = nice_axis_max(max(float(row["mean_makespan"]) for row in summary_rows) * 1.10)
    tick_count = 5
    group_width = plot_width / max(len(summary_rows), 1)
    bar_width = min(90.0, group_width * 0.55)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}">'
    )
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(svg_text(left, 36, title, fill="#111111", **{"font-size": 24, "font-family": "sans-serif", "font-weight": 700}))
    parts.append(svg_text(left, 62, subtitle, fill="#4b5563", **{"font-size": 13, "font-family": "sans-serif"}))

    for idx in range(tick_count + 1):
        value = y_max * idx / tick_count
        y = top + plot_height - (value / y_max) * plot_height
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                left - 10,
                y + 4,
                format_number(value),
                fill="#4b5563",
                **{"font-size": 12, "font-family": "sans-serif", "text-anchor": "end"},
            )
        )

    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111111" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#111111" stroke-width="1.2"/>'
    )

    for index, row in enumerate(summary_rows):
        method = str(row["method"])
        label = str(row["label"])
        value = float(row["mean_makespan"])
        bar_height = (value / y_max) * plot_height
        x = left + index * group_width + (group_width - bar_width) / 2.0
        y = top + plot_height - bar_height
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            f'fill="{METHOD_COLORS[method]}" rx="3"><title>{label} | mean makespan={format_number(value)}</title></rect>'
        )
        parts.append(
            svg_text(
                x + bar_width / 2.0,
                y - 8,
                format_number(value),
                fill="#111111",
                **{"font-size": 11, "font-family": "sans-serif", "text-anchor": "middle"},
            )
        )
        parts.append(
            svg_text(
                x + bar_width / 2.0,
                top + plot_height + 18,
                label,
                fill="#111111",
                **{"font-size": 12, "font-family": "sans-serif", "text-anchor": "middle"},
            )
        )

    parts.append(
        f'<g transform="translate(22,{top + plot_height / 2.0}) rotate(-90)">' +
        svg_text(
            0,
            0,
            "Mean makespan",
            fill="#111111",
            **{"font-size": 13, "font-family": "sans-serif", "text-anchor": "middle", "font-weight": 700},
        ) +
        "</g>"
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def main() -> None:
    args = parse_args()
    config_list = Path(args.config_list)
    diff_csv = Path(args.diff_csv)
    order_csv = Path(args.order_csv)
    compare_csv = Path(args.compare_csv)
    out_dir = Path(args.out_dir)
    tag = args.tag

    config_order = read_config_order(config_list)
    merged_rows = merge_results(
        config_order,
        [
            read_makespan_rows(diff_csv),
            read_makespan_rows(order_csv),
            read_makespan_rows(compare_csv),
        ],
    )
    summary_rows = compute_summary(merged_rows)

    merged_path = out_dir / f"{tag}_merged.csv"
    summary_path = out_dir / f"{tag}_summary.csv"
    grouped_svg = out_dir / f"{tag}_grouped.svg"
    mean_svg = out_dir / f"{tag}_mean.svg"

    write_csv(
        merged_path,
        merged_rows,
        ["config_index", "Config"] + [f"{method}_makespan" for method in METHODS],
    )
    write_csv(
        summary_path,
        summary_rows,
        ["method", "label", "mean_makespan", "median_makespan", "best_count"],
    )
    write_grouped_svg(
        grouped_svg,
        merged_rows,
        title="Makespan Comparison on Shared 15-Config Run",
        subtitle="Methods: Diff-GNN, Diff-GNN+Order, GCPS, GL25",
    )
    write_mean_svg(
        mean_svg,
        summary_rows,
        title="Mean Makespan Across Shared 15-Config Run",
        subtitle="Lower is better",
    )

    print(f"Wrote merged data: {merged_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote grouped SVG: {grouped_svg}")
    print(f"Wrote mean SVG: {mean_svg}")


if __name__ == "__main__":
    main()
