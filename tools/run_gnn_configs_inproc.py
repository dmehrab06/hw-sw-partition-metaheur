#!/usr/bin/env python3
import argparse
import gc
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from omegaconf import OmegaConf


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                if getattr(stream, "closed", False):
                    continue
                stream.write(data)
            except Exception:
                continue
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                if getattr(stream, "closed", False):
                    continue
                stream.flush()
            except Exception:
                continue


def _resolve_output_csv(config_path: str, root: Path) -> Path:
    cfg = OmegaConf.load(config_path)
    out_dir = os.getenv("HWSW_CSV_DIR") or cfg.get("output-dir", "outputs")
    csv_override = os.getenv("HWSW_RESULT_CSV") or cfg.get("result-csv") or cfg.get("result-csv-name")
    result_prefix = os.getenv("HWSW_RESULT_PREFIX") or cfg.get("result-file-prefix", "results")
    if csv_override:
        out_path = Path(csv_override)
        if not out_path.is_absolute():
            out_path = Path(out_dir) / out_path
    else:
        out_path = Path(out_dir) / f"{result_prefix}-result-summary-soda-graphs-config.csv"
    if not out_path.is_absolute():
        out_path = root / out_path
    return out_path


def _maybe_cleanup_accelerator():
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _import_gnn_main_silently(root: Path):
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            sys.path.insert(0, str(root))
            import gnn_main  # type: ignore

    return gnn_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Run gnn_main.py for multiple configs in one Python process.")
    parser.add_argument("--root", required=True, help="Repository root containing gnn_main.py")
    parser.add_argument("--outdir", required=True, help="Directory for per-config log files")
    parser.add_argument("configs", nargs="+", help="Config files to run")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    os.chdir(root)

    gnn_main = _import_gnn_main_silently(root)

    methods_env = os.getenv("HWSW_METHODS") or os.getenv("METHODS") or ""
    result_csv_env = os.getenv("HWSW_RESULT_CSV") or os.getenv("RESULT_CSV") or ""
    result_prefix_env = os.getenv("HWSW_RESULT_PREFIX") or os.getenv("RESULT_PREFIX") or ""
    csv_dir_env = os.getenv("HWSW_CSV_DIR") or os.getenv("CSV_DIR") or ""
    run_tag_env = os.getenv("HWSW_RUN_TAG") or os.getenv("RUN_TAG") or ""
    live_log = str(os.getenv("HWSW_LIVE_LOG", "1")).strip().lower() not in {"0", "false", "no", "off"}

    if methods_env:
        print(f"Running gnn_main.py on {len(args.configs)} configs (selected methods={methods_env})")
    else:
        print(f"Running gnn_main.py on {len(args.configs)} configs (selected methods=default)")
    if result_csv_env:
        print(f"CSV output override: {result_csv_env}")
    if result_prefix_env:
        print(f"CSV prefix override: {result_prefix_env}")
    if csv_dir_env:
        print(f"CSV directory override: {csv_dir_env}")
    if run_tag_env:
        print(f"Run tag: {run_tag_env}")

    batch_start = time.monotonic()
    original_argv = sys.argv[:]
    failures = 0

    for config in args.configs:
        config_path = Path(config).resolve()
        config_base = config_path.stem
        if run_tag_env:
            log_file = outdir / f"gnn_main_{config_base}__run-{run_tag_env}.log"
        else:
            log_file = outdir / f"gnn_main_{config_base}.log"

        print(f"---- [METHOD] {config_base} ----")
        config_start = time.monotonic()
        success = True

        with open(log_file, "w", encoding="utf-8") as handle:
            try:
                sys.argv = ["gnn_main.py", "-c", str(config_path)]
                out_stream = _Tee(handle, sys.__stdout__) if live_log else handle
                err_stream = _Tee(handle, sys.__stderr__) if live_log else handle
                with redirect_stdout(out_stream), redirect_stderr(err_stream):
                    gnn_main.main()
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                if code not in (0, None):
                    success = False
            except Exception:
                success = False
                raise
            finally:
                sys.argv = original_argv[:]

        if not success:
            failures += 1
            print(f"gnn_main.py failed for {config_path} (see {log_file})")
            continue

        out_src = _resolve_output_csv(str(config_path), root)
        if out_src.exists():
            out_copy = outdir / out_src.name
            try:
                if out_src.resolve() != out_copy.resolve():
                    out_copy.write_bytes(out_src.read_bytes())
            except Exception:
                pass

        gc.collect()
        _maybe_cleanup_accelerator()

        elapsed = int(round(time.monotonic() - config_start))
        print(f"Completed {config_base} in {elapsed}s")

    total_elapsed = int(round(time.monotonic() - batch_start))
    print(f"Method batch complete in {total_elapsed}s. CSV copies are in {outdir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
