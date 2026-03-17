"""
Parse slurm .err and .out files from the 'slurm_outputs' folder,
extract solver stats, and update corresponding MIP output pickle files.
"""

import glob
import os
import re
import pickle
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/parse_slurm_logs.log")

logger = LogManager.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_job_pairs(slurm_dir: str) -> list[tuple[str, str]]:
    """Return sorted list of (err_path, out_path) pairs found in slurm_dir."""
    err_files = glob.glob(os.path.join(slurm_dir, "*.err"))
    pairs = []
    for err_path in sorted(err_files):
        stem = os.path.splitext(err_path)[0]          # e.g. slurm_outputs/cp_abc
        out_path = stem + ".out"
        if os.path.isfile(out_path):
            pairs.append((err_path, out_path))
        else:
            logger.warning("No matching .out file for %s — skipping.", err_path)
    return pairs


def extract_pkl_name_from_err(err_path: str) -> str | None:
    """
    Read line 4 (1-indexed) of the .err file and extract the pickle filename.

    Expected line format (example):
        2026-03-14 14:18:55,532 - utils.partition_utils - INFO - Data loaded successfully from
        inputs/task_graph_complete/taskgraph-squeeze_net_tosa-instance-config-config_arato_area_0.3_hw_0.1_seed_0.pkl

    Returns the bare filename, e.g.:
        taskgraph-squeeze_net_tosa-instance-config-config_arato_area_0.3_hw_0.1_seed_0.pkl
    """
    with open(err_path, "r") as fh:
        lines = fh.readlines()

    if len(lines) < 4:
        logger.error("%s has fewer than 4 lines.", err_path)
        return None

    line = lines[3].strip()          # 0-indexed → line 4

    # Grab the last path-like token on the line
    match = re.search(r"(\S+\.pkl)\s*$", line)
    if not match:
        logger.error("Cannot find .pkl path in line 4 of %s:\n  %s", err_path, line)
        return None

    return Path(match.group(1)).name  # just the filename, no directory


def build_output_path(pkl_name: str, output_dir: str = "mip-opt-partitions") -> str | None:
    """
    Convert an input pkl filename to its corresponding output JSON path.

    Input  : taskgraph-squeeze_net_tosa-instance-config-config_arato_area_0.3_hw_0.1_seed_0.pkl
    Output : mip-opt-partitions/taskgraph-squeeze_net_tosa_area-0.30_hwscale-0.1_hwvar-0.50_seed-0_assignment-mip.json

    Transformation rules applied:
      • Strip the trailing  -instance-config-config_arato  segment.
      • area_X   → area-{X:.2f}
      • hw_X     → hwscale-X
      • seed_X   → seed-X
      • Append   _hwvar-0.50_assignment-mip  before .pkl
    """
    # Remove the .pkl extension for manipulation
    name = pkl_name.replace(".pkl", "")

    # Pattern:
    #   taskgraph-<model>-instance-config-config_arato_area_<A>_hw_<H>_seed_<S>
    pattern = re.compile(
        r"^(taskgraph-[^-]+(?:-[^-]+)*?)"    # group 1: taskgraph-<model>
        r"-instance-config-config_arato"      # literal middle section
        r"_area_([\d.]+)"                     # group 2: area value
        r"_hw_([\d.]+)"                       # group 3: hw value
        r"_seed_(\d+)$"                       # group 4: seed value
    )

    m = pattern.match(name)
    if not m:
        logger.error("Filename does not match expected pattern: %s", pkl_name)
        return None

    model_part = m.group(1)          # e.g. taskgraph-squeeze_net_tosa
    area_val   = float(m.group(2))   # e.g. 0.3
    hw_val     = m.group(3)          # e.g. 0.1  (keep as string to preserve precision)
    seed_val   = m.group(4)          # e.g. 0

    out_name = (
        f"{model_part}"
        f"_area-{area_val:.2f}"
        f"_hwscale-{hw_val}"
        f"_hwvar-0.50"
        f"_seed-{seed_val}"
        f"_assignment-mip.json"
    )

    return os.path.join(output_dir, out_name)


def extract_solver_stats(out_path: str) -> dict | None:
    """
    Extract SCIP solver statistics from the tail of a .out file.

    Looks for a block that starts with 'SCIP Status' and ends with the
    'Summary' separator line.

    Returns a dict with keys:
        status, solving_time_sec, solving_nodes,
        primal_bound, primal_solutions, dual_bound, gap_pct
    """
    with open(out_path, "r") as fh:
        content = fh.read()

    # Find the SCIP stats block
    block_match = re.search(
        r"(SCIP Status\s*:.*?)(?=-{10,}\s*Summary)",
        content,
        re.DOTALL,
    )
    if not block_match:
        logger.error("Cannot find SCIP stats block in %s", out_path)
        return None

    block = block_match.group(1)

    def _get(pattern, text, cast=str):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    # Status string
    status = _get(r"SCIP Status\s*:\s*(.+)", block)

    # Solving time
    solving_time = _get(r"Solving Time \(sec\)\s*:\s*([\d.]+)", block, float)

    # Solving nodes
    solving_nodes = _get(r"Solving Nodes\s*:\s*(\d+)", block, int)

    # Primal bound and number of solutions
    primal_match = re.search(
        r"Primal Bound\s*:\s*([+\-\d.e]+)\s*\((\d+) solutions?\)", block
    )
    primal_bound     = float(primal_match.group(1)) if primal_match else None
    primal_solutions = int(primal_match.group(2))   if primal_match else None

    # Dual bound
    dual_bound = _get(r"Dual Bound\s*:\s*([+\-\d.e]+)", block, float)

    # Gap
    gap_pct = _get(r"Gap\s*:\s*([\d.]+)\s*%", block, float)

    stats = {
        "status":            status.strip() if status else None,
        "solving_time_sec":  solving_time,
        "solving_nodes":     solving_nodes,
        "primal_bound":      primal_bound,
        "primal_solutions":  primal_solutions,
        "dual_bound":        dual_bound,
        "gap_pct":           gap_pct,
    }

    return stats


def update_pickle(pkl_path: str, solver_stats: dict) -> bool:
    """
    Load the pickle at pkl_path, add / update the 'solver_stats' key, and save.

    Returns True on success, False otherwise.
    """
    if not os.path.isfile(pkl_path):
        logger.error("Output pickle not found: %s", pkl_path)
        return False

    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)

    if isinstance(data, dict):
        data["solver_stats"] = solver_stats
    else:
        # Wrap in a dict if the stored object isn't already one
        logger.warning(
            "%s does not contain a dict (type=%s). Wrapping in a dict.",
            pkl_path, type(data).__name__,
        )
        data = {"original_data": data, "solver_stats": solver_stats}

    with open(pkl_path, "wb") as fh:
        pickle.dump(data, fh)

    return True

def update_json(json_path: str, solver_stats: dict) -> bool:
    """
    Load the JSON at json_path, add / update the 'solver_stats' key, and save.

    Returns True on success, False otherwise.
    """
    if not os.path.isfile(json_path):
        logger.error(f"Output JSON not found: {json_path}")
        return False

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data["solver_stats"] = solver_stats
    else:
        # Wrap in a dict if the stored object isn't already one
        logger.warning(
            "%s does not contain a dict (type=%s). Wrapping in a dict.",
            json_path, type(data).__name__,
        )
        data = {"original_data": data, "solver_stats": solver_stats}

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(slurm_dir: str = "slurm_outputs", output_dir: str = "mip-opt-partitions"):
    pairs = find_job_pairs(slurm_dir)
    if not pairs:
        logger.warning("No .err/.out pairs found in '%s'.", slurm_dir)
        return

    logger.info("Found %d job pair(s) in '%s'.", len(pairs), slurm_dir)

    success_count = 0
    for err_path, out_path in pairs:
        job_name = Path(err_path).stem
        logger.info("─── Processing job: %s", job_name)

        # 1. Extract pkl name from .err
        pkl_name = extract_pkl_name_from_err(err_path)
        if pkl_name is None:
            continue

        # 2. Build output path
        output_json = build_output_path(pkl_name, output_dir)
        if output_json is None:
            continue
        logger.info("  Input pkl  : %s", pkl_name)
        logger.info("  Output json : %s", output_json)

        # 3. Extract solver stats from .out
        stats = extract_solver_stats(out_path)
        if stats is None:
            continue
        logger.info("  Solver stats: %s", stats)

        # 4. Update the pickle file
        ok = update_json(output_json, stats)
        if ok:
            logger.info("  ✓ Pickle updated successfully.")
            success_count += 1
        else:
            logger.error("  ✗ Failed to update pickle.")

    logger.info("Done. %d/%d job(s) updated successfully.", success_count, len(pairs))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse slurm outputs and update MIP result pickles with solver stats."
    )
    parser.add_argument(
        "--slurm-dir",
        default="slurm_outputs",
        help="Directory containing .err/.out slurm files (default: slurm_outputs)",
    )
    parser.add_argument(
        "--output-dir",
        default="mip-opt-partitions",
        help="Directory containing MIP output pickle files (default: mip-opt-partitions)",
    )
    args = parser.parse_args()

    main(slurm_dir=args.slurm_dir, output_dir=args.output_dir)