#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


DATASET_AREA05 = "dataset_area05"
SQUEEZENET_AREA_SWEEP = "squeezenet_area_sweep"
LARGE_SCALE_AREA05 = "large_scale_area05"

DATASET_FAMILIES = (DATASET_AREA05, SQUEEZENET_AREA_SWEEP, LARGE_SCALE_AREA05)
ALLOWED_METHOD_DIRS = {"mip", "diff_gnn_order", "random", "greedy"}
ALL_METHOD_PREFIXES = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed BatchExperiments2 with existing mip/diff_gnn_order outputs.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dest-root", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=list(DATASET_FAMILIES))
    parser.add_argument("--force", action="store_true", help="Overwrite existing destination files.")
    return parser.parse_args()


def _copy_file(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        return
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        if not force:
            return
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _filtered_gnn_csv(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        return
    if dst.exists() and not force:
        return

    frame = pd.read_csv(src)
    if frame.empty:
        dst.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dst, index=False)
        return

    allowed_prefixes = ("diff_gnn_order_", "random_", "greedy_")
    keep_cols: list[str] = []
    for col in frame.columns:
        if any(col.startswith(f"{prefix}_") for prefix in ALL_METHOD_PREFIXES):
            if col.startswith(allowed_prefixes):
                keep_cols.append(col)
            continue
        keep_cols.append(col)

    filtered = frame.loc[:, keep_cols]
    dst.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(dst, index=False)


def _bootstrap_dataset_area05(source_root: Path, dest_root: Path, force: bool) -> None:
    src = source_root / DATASET_AREA05
    dst = dest_root / DATASET_AREA05
    if not src.exists():
        return

    _copy_file(
        src / "dataset_area05_10seed_selected_manifest.csv",
        dst / "dataset_area05_10seed_selected_manifest.csv",
        force=force,
    )

    for dataset_dir in sorted(path for path in src.iterdir() if path.is_dir()):
        dataset_name = dataset_dir.name
        if dataset_name in {"partitions"}:
            continue
        dataset_dst = dst / dataset_name
        for manifest in dataset_dir.glob("*selected_manifest.csv"):
            _copy_file(manifest, dataset_dst / manifest.name, force=force)
        for method in ALLOWED_METHOD_DIRS:
            _copy_tree(dataset_dir / method, dataset_dst / method, force=force)


def _bootstrap_large_scale_area05(source_root: Path, dest_root: Path, force: bool) -> None:
    src = source_root / LARGE_SCALE_AREA05
    dst = dest_root / LARGE_SCALE_AREA05
    if not src.exists():
        return

    _copy_file(
        src / "large_scale_area05_10seed_selected_manifest.csv",
        dst / "large_scale_area05_10seed_selected_manifest.csv",
        force=force,
    )

    for dataset_dir in sorted(path for path in src.iterdir() if path.is_dir()):
        dataset_name = dataset_dir.name
        if dataset_name in {"config_cache", "selected_configs"}:
            continue
        dataset_dst = dst / dataset_name
        for manifest in dataset_dir.glob("*selected_manifest.csv"):
            _copy_file(manifest, dataset_dst / manifest.name, force=force)
        for method in ALLOWED_METHOD_DIRS:
            _copy_tree(dataset_dir / method, dataset_dst / method, force=force)


def _bootstrap_squeezenet_area_sweep(source_root: Path, dest_root: Path, force: bool) -> None:
    src = source_root / SQUEEZENET_AREA_SWEEP
    dst = dest_root / SQUEEZENET_AREA_SWEEP
    if not src.exists():
        return

    _copy_file(
        src / "squeezenet_area_sweep_10seed_selected_manifest.csv",
        dst / "squeezenet_area_sweep_10seed_selected_manifest.csv",
        force=force,
    )
    _copy_file(
        src / "mip_squeezenet_area_sweep_10seed-result-summary-soda-graphs-config.csv",
        dst / "mip_squeezenet_area_sweep_10seed-result-summary-soda-graphs-config.csv",
        force=force,
    )
    _filtered_gnn_csv(
        src / "squeezenet_area_sweep_10seed-result-summary-soda-graphs-config.csv",
        dst / "squeezenet_area_sweep_10seed-result-summary-soda-graphs-config.csv",
        force=force,
    )


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()

    handlers = {
        DATASET_AREA05: _bootstrap_dataset_area05,
        SQUEEZENET_AREA_SWEEP: _bootstrap_squeezenet_area_sweep,
        LARGE_SCALE_AREA05: _bootstrap_large_scale_area05,
    }

    for family in args.families:
        handler = handlers.get(str(family).strip())
        if handler is None:
            continue
        handler(source_root, dest_root, force=bool(args.force))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
