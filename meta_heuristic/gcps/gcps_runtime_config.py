from __future__ import annotations

from copy import deepcopy
import os
from typing import Any


# Keep the existing GCPS behavior for genuinely small graphs.
GCPS_SMALL_GRAPH_MAX_NODES = 64

GCPS_SMALL_CONFIG: dict[str, Any] = {
    "lr": 1e-3,
    "dropout": 0.2,
    "pretrain_iter": 100,
    "posttrain_iter": 800,
    "iter": 800,
    "schedule_skip": 5,
    "sigma": 0.3,
    "hidden_dim_1": 10,
    "hidden_dim_2": 5,
    "weight_decay": 0.0,
    "quick_search": True,
    "area_penalty_coeff": 0.0,
    "verbose": 0,
    "schedule_eval": "lssp",
    "schedule_auto_repair": True,
    "alpha": 5.0,
    "device": "auto",
}

GCPS_LARGE_CONFIG: dict[str, Any] = {
    "lr": 1e-3,
    "dropout": 0.2,
    "pretrain_iter": 500,
    "posttrain_iter": 1500,
    "iter": 1500,
    "schedule_skip": 2,
    "sigma": 0.3,
    "hidden_dim_1": 256,
    "hidden_dim_2": 256,
    "weight_decay": 0.0,
    "quick_search": True,
    "area_penalty_coeff": 0.0,
    "verbose": 50,
    "schedule_eval": "lssp",
    "schedule_auto_repair": True,
    "alpha": 5.0,
    "device": "auto",
}


def _normalize_preset_name(value: str | None) -> str:
    name = str(value or "auto").strip().lower()
    if name in {"small", "small_graph", "small-graph"}:
        return "small"
    if name in {"large", "large_graph", "large-graph"}:
        return "large"
    return "auto"


def resolve_gcps_runtime_config(num_nodes: int) -> tuple[dict[str, Any], str]:
    """
    Resolve the GCPS preset without consulting the YAML `gcps:` block.

    Override path, if needed later:
      - HWSW_GCPS_PRESET=small|large
    Default path:
      - small preset when num_nodes <= GCPS_SMALL_GRAPH_MAX_NODES
      - large preset otherwise
    """
    forced = _normalize_preset_name(os.getenv("HWSW_GCPS_PRESET"))
    if forced == "small":
        return deepcopy(GCPS_SMALL_CONFIG), "small"
    if forced == "large":
        return deepcopy(GCPS_LARGE_CONFIG), "large"
    if int(num_nodes) <= GCPS_SMALL_GRAPH_MAX_NODES:
        return deepcopy(GCPS_SMALL_CONFIG), "small"
    return deepcopy(GCPS_LARGE_CONFIG), "large"


__all__ = [
    "GCPS_LARGE_CONFIG",
    "GCPS_SMALL_CONFIG",
    "GCPS_SMALL_GRAPH_MAX_NODES",
    "resolve_gcps_runtime_config",
]
