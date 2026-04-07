#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _status_is_time_limited(status: Any) -> bool:
    text = str(status).strip().lower()
    return any(token in text for token in ("time limit", "user_limit", "time_limit", "timed_out", "timeout"))


def _extract_solver_stats(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    block_match = re.search(
        r"(SCIP Status\s*:.*?)(?=-{10,}\s*Summary|$)",
        text,
        re.DOTALL,
    )
    if not block_match:
        return None

    block = block_match.group(1)

    def _get(pattern: str, cast=str):
        match = re.search(pattern, block)
        if not match:
            return None
        value = match.group(1)
        try:
            return cast(value)
        except Exception:
            return value

    status = _get(r"SCIP Status\s*:\s*(.+)")
    solving_time = _get(r"Solving Time \(sec\)\s*:\s*([\d.]+)", float)
    solving_nodes = _get(r"Solving Nodes\s*:\s*(\d+)", int)

    primal_match = re.search(
        r"Primal Bound\s*:\s*([+\-\d.eE]+)\s*\((\d+) solutions?\)",
        block,
    )
    primal_bound = float(primal_match.group(1)) if primal_match else None
    primal_solutions = int(primal_match.group(2)) if primal_match else None

    dual_bound = _get(r"Dual Bound\s*:\s*([+\-\d.eE]+)", float)
    gap_pct = _get(r"Gap\s*:\s*([\d.]+)\s*%", float)

    stats = {
        "status": status.strip() if isinstance(status, str) else status,
        "solving_time_sec": solving_time,
        "solving_nodes": solving_nodes,
        "primal_bound": primal_bound,
        "primal_solutions": primal_solutions,
        "dual_bound": dual_bound,
        "gap_pct": gap_pct,
    }
    return {k: v for k, v in stats.items() if v is not None}


def _merge_solver_stats(path: Path, parsed_stats: dict[str, Any]) -> bool:
    if not path or not path.exists():
        return False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False

    existing = payload.get("solver_stats", {})
    if not isinstance(existing, dict):
        existing = {}

    merged = dict(existing)
    merged.update(parsed_stats)
    payload["solver_stats"] = merged
    if _status_is_time_limited(merged.get("status")):
        payload["time_limit_exceeded"] = True
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach SCIP solver stats parsed from a log to MIP JSON/meta artifacts.")
    parser.add_argument("--log-file", required=True, help="Path to the MIP run log containing SCIP verbose output")
    parser.add_argument("--json-path", help="Path to *_assignment-mip.json")
    parser.add_argument("--meta-path", help="Path to *_assignment-mip.meta.json")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    parsed_stats = _extract_solver_stats(log_path)
    if not parsed_stats:
        return 0

    updated = False
    if args.json_path:
        updated = _merge_solver_stats(Path(args.json_path), parsed_stats) or updated
    if args.meta_path:
        updated = _merge_solver_stats(Path(args.meta_path), parsed_stats) or updated

    if updated:
        print(json.dumps(parsed_stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
