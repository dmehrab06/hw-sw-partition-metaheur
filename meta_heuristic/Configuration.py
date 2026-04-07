from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


# Central default settings for the classical black-box methods.
# YAML config entries still override these values when present.
_CLASSICAL_METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "greedy": {},
    "random": {
        "num_samples": 500,
        "p": 0.5,
    },
    "pso": {
        "c1": 0.575,
        "c2": 0.1,
        "w": 1.05,
        "n_particles": 500,
        "iterations": 10,
        "verbose": True,
    },
    "dbpso": {
        "c1": 0.575,
        "c2": 0.1,
        "w": 1.05,
        "k": 4,
        "p": 2,
        "n_particles": 500,
        "iterations": 10,
        "verbose": True,
    },
    "clpso": {
        "c": 1,
        "n_individuals": 500,
        "iterations": 200,
        "verbose": 500,
        "seed_rng": None,
    },
    "ccpso": {
        "c": 1,
        "n_individuals": 500,
        "iterations": 200,
        "verbose": 500,
        "group_sizes": [5, 10, 20],
        "seed_rng": None,
    },
    "esa": {
        "iter": 1000,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "shade": {
        "iter": 2000,
        "n_individuals": 500,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "jade": {
        "iter": 2000,
        "n_individuals": 500,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "gl25": {
        "iter": 2000,
        "n_pop": 500,
        "verbose": 500,
        "seed_rng": 2022,
    },
}

_CLASSICAL_SEARCH_OBJECTIVE_METHODS: tuple[str, ...] = (
    "greedy",
    "random",
    "pso",
    "dbpso",
    "clpso",
    "ccpso",
    "esa",
    "shade",
    "jade",
    "gl25",
    "non_diffgnn",
)

# Global defaults shared by the classical search-time objective path.
#
# search_objective canonical options:
# - "lssp": use evaluate_partition_lssp() during search
# - "mip": use the fast DAG surrogate during search
# - "makespan": use the queue/taskgraph makespan evaluator during search
# - "partition": use partition cost only
#
# Accepted override aliases in gnn_main.py:
# - "dag", "fast-dag", "fast_dag" -> "mip"
# - "queue", "taskgraph" -> "makespan"
# - "requested", "same" -> respect YAML opt-cost-type instead of forcing a
#   classical default
#
# Current repo default for classical methods is "lssp".
# GCPS is handled separately and uses its own "schedule_eval" setting
# (lssp|taskgraph|queue).
_CLASSICAL_GLOBAL_DEFAULTS: dict[str, Any] = {
    "search_objective": "lssp",
}


def get_classical_method_defaults(method_name: str) -> dict[str, Any]:
    return deepcopy(_CLASSICAL_METHOD_DEFAULTS.get(str(method_name).lower(), {}))


def get_classical_method_names() -> tuple[str, ...]:
    return tuple(_CLASSICAL_SEARCH_OBJECTIVE_METHODS)


def is_classical_search_method(method_name: str) -> bool:
    return str(method_name).lower() in _CLASSICAL_SEARCH_OBJECTIVE_METHODS


def get_classical_search_objective_default() -> str:
    return str(_CLASSICAL_GLOBAL_DEFAULTS["search_objective"])


def resolve_classical_method_config(
    config: Mapping[str, Any] | None,
    method_name: str,
) -> dict[str, Any]:
    method_key = str(method_name).lower()
    resolved = get_classical_method_defaults(method_key)
    if not isinstance(config, Mapping):
        return resolved

    user_cfg = config.get(method_key, {})
    if isinstance(user_cfg, Mapping):
        resolved.update(user_cfg)
    return resolved


__all__ = [
    "get_classical_method_defaults",
    "get_classical_method_names",
    "get_classical_search_objective_default",
    "is_classical_search_method",
    "resolve_classical_method_config",
]
