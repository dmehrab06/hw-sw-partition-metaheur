#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf


def _resolve_output_csv(config_path: str, root: Path, outdir: Path, env: dict[str, str]) -> Path:
    csv_override = env.get("HWSW_RESULT_CSV") or env.get("RESULT_CSV") or ""
    if csv_override:
        out_path = Path(csv_override)
        if not out_path.is_absolute():
            out_path = outdir / out_path
        return out_path.resolve()

    cfg = OmegaConf.load(config_path)
    result_prefix = env.get("HWSW_RESULT_PREFIX") or env.get("RESULT_PREFIX") or cfg.get("result-file-prefix", "mip_solver")
    return (outdir / f"mip_{result_prefix}-result-summary-soda-graphs-config.csv").resolve()


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        return [], []
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return list(rows[0]), rows[1:]


def _append_csv_with_alignment(final_csv: Path, temp_csv: Path) -> None:
    if not temp_csv.exists():
        return

    src_cols, src_rows = _read_csv(temp_csv)
    if not src_cols:
        return

    dst_cols, dst_rows = _read_csv(final_csv)
    merged_cols = list(dst_cols) if dst_cols else list(src_cols)
    for col in src_cols:
        if col not in merged_cols:
            merged_cols.append(col)

    def _row_dicts(cols: list[str], rows: list[list[str]]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for raw in rows:
            padded = list(raw) + [""] * max(0, len(cols) - len(raw))
            out.append(dict(zip(cols, padded[: len(cols)])))
        return out

    merged_rows = []
    if dst_cols:
        merged_rows.extend(_row_dicts(dst_cols, dst_rows))
    merged_rows.extend(_row_dicts(src_cols, src_rows))

    final_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(final_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(merged_cols)
        for row in merged_rows:
            writer.writerow([row.get(col, "") for col in merged_cols])


def _maybe_copy_final_csv(final_csv: Path, outdir: Path) -> None:
    if not final_csv.exists():
        return
    out_copy = outdir / final_csv.name
    try:
        if final_csv.resolve() != out_copy.resolve():
            shutil.copyfile(final_csv, out_copy)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run run_all_mip_configs.sh for multiple configs with subprocess-level parallelism.")
    parser.add_argument("--root", required=True, help="Repository root containing run_all_mip_configs.sh")
    parser.add_argument("--outdir", required=True, help="Directory for wrapper logs and CSV fragments")
    parser.add_argument("--script", required=True, help="Path to run_all_mip_configs.sh")
    parser.add_argument("--jobs", type=int, required=True, help="Maximum parallel child processes")
    parser.add_argument("configs", nargs="+", help="Config files to run")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    script = Path(args.script).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_csv_dir = outdir / ".tmp_parallel_mip_csv"
    tmp_csv_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    run_tag_env = base_env.get("HWSW_RUN_TAG", "") or base_env.get("RUN_TAG", "")
    final_csvs_by_config = {
        config: _resolve_output_csv(config, root, outdir, base_env) for config in args.configs
    }

    print(f"Running MIP solver on {len(args.configs)} configs")
    print(f"Parallel jobs: {args.jobs}")
    if run_tag_env:
        print(f"Run tag: {run_tag_env}")

    batch_start = time.monotonic()
    active: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    failures = 0

    def _poll_active(block: bool = False) -> None:
        nonlocal failures
        while active:
            changed = False
            remaining = []
            for job in active:
                proc = job["proc"]
                ret = proc.poll()
                if ret is None:
                    remaining.append(job)
                    continue
                changed = True
                job["log_handle"].close()
                elapsed = int(round(time.monotonic() - job["start"]))
                success = ret == 0
                if success:
                    print(f"Completed {job['config_base']} in {elapsed}s")
                else:
                    failures += 1
                    print(f"Failed {job['config_base']} in {elapsed}s (exit={ret}, see {job['log_file']})")
                job["success"] = success
                results.append(job)
            active[:] = remaining
            if changed or not block:
                return
            time.sleep(1.0)

    for config in args.configs:
        config_path = Path(config).resolve()
        config_base = config_path.stem
        if run_tag_env:
            log_file = outdir / f"mip_runner_{config_base}__run-{run_tag_env}.log"
        else:
            log_file = outdir / f"mip_runner_{config_base}.log"
        temp_csv = tmp_csv_dir / f"{config_base}.csv"
        if temp_csv.exists():
            temp_csv.unlink()

        while len(active) >= max(1, int(args.jobs)):
            _poll_active(block=True)

        child_env = base_env.copy()
        child_env["PYTHONNOUSERSITE"] = "1"
        child_env["CONFIG_GLOB"] = str(config_path)
        child_env["OUTDIR"] = str(outdir)
        child_env["HWSW_RESULT_CSV"] = str(temp_csv)
        child_env["HWSW_MAX_PARALLEL_CONFIGS"] = "1"
        child_env["MAX_PARALLEL_CONFIGS"] = "1"

        log_handle = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [str(script)],
            cwd=str(root),
            env=child_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"Launched {config_base} pid={proc.pid} log={log_file}")
        active.append(
            {
                "proc": proc,
                "log_handle": log_handle,
                "config": str(config_path),
                "config_base": config_base,
                "log_file": log_file,
                "temp_csv": temp_csv,
                "final_csv": final_csvs_by_config[config],
                "start": time.monotonic(),
            }
        )

    while active:
        _poll_active(block=True)

    grouped: defaultdict[Path, list[Path]] = defaultdict(list)
    for job in results:
        if job.get("success"):
            grouped[job["final_csv"]].append(job["temp_csv"])

    for final_csv, temp_csvs in grouped.items():
        for temp_csv in temp_csvs:
            _append_csv_with_alignment(final_csv, temp_csv)
        _maybe_copy_final_csv(final_csv, outdir)

    total_elapsed = int(round(time.monotonic() - batch_start))
    print(f"MIP batch complete in {total_elapsed}s. CSV copies are in {outdir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
