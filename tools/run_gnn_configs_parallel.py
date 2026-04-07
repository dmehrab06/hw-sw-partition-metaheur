#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf


def _resolve_output_csv(config_path: str, root: Path, env: dict[str, str]) -> Path:
    cfg = OmegaConf.load(config_path)
    out_dir = env.get("HWSW_CSV_DIR") or env.get("CSV_DIR") or cfg.get("output-dir", "outputs")
    csv_override = env.get("HWSW_RESULT_CSV") or env.get("RESULT_CSV") or cfg.get("result-csv") or cfg.get("result-csv-name")
    result_prefix = env.get("HWSW_RESULT_PREFIX") or env.get("RESULT_PREFIX") or cfg.get("result-file-prefix", "results")
    if csv_override:
        out_path = Path(csv_override)
        if not out_path.is_absolute():
            out_path = Path(out_dir) / out_path
    else:
        out_path = Path(out_dir) / f"{result_prefix}-result-summary-soda-graphs-config.csv"
    if not out_path.is_absolute():
        out_path = root / out_path
    return out_path


def _read_csv(path: Path):
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
    if dst_cols:
        merged_cols = list(dst_cols)
    else:
        merged_cols = list(src_cols)

    for col in src_cols:
        if col not in merged_cols:
            merged_cols.append(col)

    def _row_dicts(cols, rows):
        out = []
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
    parser = argparse.ArgumentParser(description="Run gnn_main.py for multiple configs with subprocess-level parallelism.")
    parser.add_argument("--root", required=True, help="Repository root containing gnn_main.py")
    parser.add_argument("--outdir", required=True, help="Directory for per-config log files")
    parser.add_argument("--python", required=True, help="Python interpreter to launch")
    parser.add_argument("--jobs", type=int, required=True, help="Maximum parallel child processes")
    parser.add_argument("configs", nargs="+", help="Config files to run")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_csv_dir = outdir / ".tmp_parallel_csv"
    tmp_csv_dir.mkdir(parents=True, exist_ok=True)

    methods_env = os.getenv("HWSW_METHODS") or os.getenv("METHODS") or ""
    result_csv_env = os.getenv("HWSW_RESULT_CSV") or os.getenv("RESULT_CSV") or ""
    result_prefix_env = os.getenv("HWSW_RESULT_PREFIX") or os.getenv("RESULT_PREFIX") or ""
    csv_dir_env = os.getenv("HWSW_CSV_DIR") or os.getenv("CSV_DIR") or ""
    run_tag_env = os.getenv("HWSW_RUN_TAG") or os.getenv("RUN_TAG") or ""

    if methods_env:
        print(f"Running gnn_main.py on {len(args.configs)} configs (selected methods={methods_env})")
    else:
        print(f"Running gnn_main.py on {len(args.configs)} configs (selected methods=default)")
    print(f"Parallel jobs: {args.jobs}")
    if result_csv_env:
        print(f"CSV output override: {result_csv_env}")
    if result_prefix_env:
        print(f"CSV prefix override: {result_prefix_env}")
    if csv_dir_env:
        print(f"CSV directory override: {csv_dir_env}")
    if run_tag_env:
        print(f"Run tag: {run_tag_env}")

    base_env = os.environ.copy()
    final_csv_env = dict(base_env)
    final_csvs_by_config = {}
    for config in args.configs:
        final_csvs_by_config[config] = _resolve_output_csv(config, root, final_csv_env)

    batch_start = time.monotonic()
    active = []
    results = []
    failures = 0

    def _poll_active(block: bool = False):
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
                if not success:
                    failures += 1
                    print(f"Failed {job['config_base']} in {elapsed}s (see {job['log_file']})")
                else:
                    print(f"Completed {job['config_base']} in {elapsed}s")
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
            log_file = outdir / f"gnn_main_{config_base}__run-{run_tag_env}.log"
        else:
            log_file = outdir / f"gnn_main_{config_base}.log"
        temp_csv = tmp_csv_dir / f"{config_base}.csv"
        if temp_csv.exists():
            temp_csv.unlink()

        while len(active) >= max(1, int(args.jobs)):
            _poll_active(block=True)

        child_env = base_env.copy()
        child_env["PYTHONNOUSERSITE"] = "1"
        child_env["HWSW_RESULT_CSV"] = str(temp_csv)
        if methods_env:
            child_env["HWSW_METHODS"] = methods_env
        if result_prefix_env:
            child_env["HWSW_RESULT_PREFIX"] = result_prefix_env
        if csv_dir_env:
            child_env["HWSW_CSV_DIR"] = csv_dir_env
        if run_tag_env:
            child_env["HWSW_RUN_TAG"] = run_tag_env

        log_handle = open(log_file, "w", encoding="utf-8")
        cmd = [args.python, "gnn_main.py", "-c", str(config_path)]
        proc = subprocess.Popen(
            cmd,
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

    grouped = defaultdict(list)
    for job in results:
        if job.get("success"):
            grouped[job["final_csv"]].append(job["temp_csv"])

    for final_csv, temp_csvs in grouped.items():
        for temp_csv in temp_csvs:
            _append_csv_with_alignment(final_csv, temp_csv)
        _maybe_copy_final_csv(final_csv, outdir)

    total_elapsed = int(round(time.monotonic() - batch_start))
    print(f"Method batch complete in {total_elapsed}s. CSV copies are in {outdir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
