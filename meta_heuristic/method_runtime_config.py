from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Mapping


# Legacy GCPS defaults kept here for compatibility with generic config queries.
# The active GCPS runtime selection now lives in
# meta_heuristic/gcps/gcps_runtime_config.py and is applied directly inside
# simulate_gcps, intentionally ignoring the per-YAML `gcps:` block.
_GCPS_RUNTIME_DEFAULTS: dict[str, Any] = {
    "lr": 1e-3,
    "dropout": 0.2,
    "pretrain_iter": 100,
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

# Arato profile.
# Small classical-method budget:
#   - 2,000 function evaluations
#   - population / sample size 500 when applicable
_METHOD_RUNTIME_DEFAULTS_ARATO: dict[str, dict[str, Any]] = {
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
        "iterations": 4,
        "verbose": True,
    },
    "dbpso": {
        "c1": 0.575,
        "c2": 0.1,
        "w": 1.05,
        "k": 4,
        "p": 2,
        "n_particles": 500,
        "iterations": 4,
        "verbose": True,
    },
    "clpso": {
        "c": 1,
        "n_individuals": 500,
        "iterations": 2000,
        "verbose": 500,
        "seed_rng": None,
    },
    "ccpso": {
        "c": 1,
        "n_individuals": 500,
        "iterations": 2000,
        "verbose": 500,
        "group_sizes": [5, 10, 20],
        "seed_rng": None,
    },
    "esa": {
        "iter": 2000,
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
    "gcps": deepcopy(_GCPS_RUNTIME_DEFAULTS),
}

# Makespan profile.
# Large classical-method budget:
#   - 10,000 function evaluations
#   - population / sample size 1,500 when applicable
_METHOD_RUNTIME_DEFAULTS_MAKESPAN: dict[str, dict[str, Any]] = {
    "greedy": {},
    "random": {
        "num_samples": 1500,
        "p": 0.5,
    },
    "pso": {
        "c1": 0.575,
        "c2": 0.1,
        "w": 1.05,
        "n_particles": 1500,
        "iterations": 7,
        "verbose": True,
    },
    "dbpso": {
        "c1": 0.575,
        "c2": 0.1,
        "w": 1.05,
        "k": 4,
        "p": 2,
        "n_particles": 1500,
        "iterations": 7,
        "verbose": True,
    },
    "clpso": {
        "c": 1,
        "n_individuals": 1500,
        "iterations": 10000,
        "verbose": 500,
        "seed_rng": None,
    },
    "ccpso": {
        "c": 1,
        "n_individuals": 1500,
        "iterations": 10000,
        "verbose": 500,
        "group_sizes": [5, 10, 20],
        "seed_rng": None,
    },
    "esa": {
        "iter": 10000,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "shade": {
        "iter": 10000,
        "n_individuals": 1500,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "jade": {
        "iter": 10000,
        "n_individuals": 1500,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "gl25": {
        "iter": 10000,
        "n_pop": 1500,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "gcps": deepcopy(_GCPS_RUNTIME_DEFAULTS),
}

# Balanced profile.
# Mid classical-method budget:
#   - 5,000 function evaluations
#   - population / sample size 1,000 when applicable
_METHOD_RUNTIME_DEFAULTS_BALANCED: dict[str, dict[str, Any]] = {
    "greedy": {},
    "random": {
        "num_samples": 1000,
        "p": 0.5,
    },
    "pso": {
        "c1": 0.575,
        "c2": 0.1,
        "w": 1.05,
        "n_particles": 1000,
        "iterations": 5,
        "verbose": True,
    },
    "dbpso": {
        "c1": 0.575,
        "c2": 0.1,
        "w": 1.05,
        "k": 4,
        "p": 2,
        "n_particles": 1000,
        "iterations": 5,
        "verbose": True,
    },
    "clpso": {
        "c": 1,
        "n_individuals": 1000,
        "iterations": 5000,
        "verbose": 500,
        "seed_rng": None,
    },
    "ccpso": {
        "c": 1,
        "n_individuals": 1000,
        "iterations": 5000,
        "verbose": 500,
        "group_sizes": [5, 10, 20],
        "seed_rng": None,
    },
    "esa": {
        "iter": 5000,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "shade": {
        "iter": 5000,
        "n_individuals": 1000,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "jade": {
        "iter": 5000,
        "n_individuals": 1000,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "gl25": {
        "iter": 5000,
        "n_pop": 1000,
        "verbose": 500,
        "seed_rng": 2022,
    },
    "gcps": deepcopy(_GCPS_RUNTIME_DEFAULTS),
}

_METHOD_RUNTIME_DEFAULTS_BY_PROFILE: dict[str, dict[str, dict[str, Any]]] = {
    "arato": _METHOD_RUNTIME_DEFAULTS_ARATO,
    "makespan": _METHOD_RUNTIME_DEFAULTS_MAKESPAN,
    "balanced": _METHOD_RUNTIME_DEFAULTS_BALANCED,
}

# Search-objective forcing still only applies to the classical black-box search
# methods. GCPS keeps its own schedule evaluator knob and diff_gnn* keep their
# existing Python-side configuration paths.
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

_CLASSICAL_GLOBAL_DEFAULTS_BY_PROFILE: dict[str, dict[str, Any]] = {
    "arato": {
        "search_objective": "lssp",
    },
    # Follow the config-requested objective for the makespan profile. With the
    # current makespan YAMLs that resolves to queue-makespan rather than LSSP.
    "makespan": {
        "search_objective": "requested",
    },
    # Keep the same objective resolution as the makespan profile; this profile
    # only changes the classical-method budgets.
    "balanced": {
        "search_objective": "requested",
    },
}


def _resolve_active_runtime_profile() -> str:
    profile = str(os.getenv("HWSW_METHOD_RUNTIME_PROFILE", "balanced")).strip().lower()
    if profile in _METHOD_RUNTIME_DEFAULTS_BY_PROFILE:
        return profile
    return "balanced"


_ACTIVE_METHOD_RUNTIME_PROFILE = _resolve_active_runtime_profile()
_METHOD_RUNTIME_DEFAULTS = _METHOD_RUNTIME_DEFAULTS_BY_PROFILE[_ACTIVE_METHOD_RUNTIME_PROFILE]
_CLASSICAL_GLOBAL_DEFAULTS = _CLASSICAL_GLOBAL_DEFAULTS_BY_PROFILE[_ACTIVE_METHOD_RUNTIME_PROFILE]

# Dataset-specific runtime overrides. These apply before the global defaults,
# which means they can specialize the Python defaults for a given graph while
# still allowing explicit YAML values to win if the user manually adds them.
#
# Shape:
#   {graph_name: {method_name: {key: value, ...}}}
_METHOD_DATASET_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "paper_fig3_11node": {},
    "mobile_net_tosa": {},
    "rez_net_tosa": {},
    "squeeze_net_tosa": {},
    "anomaly_detection_tosa": {},
    "image_classification_tosa": {},
    "keyword_spotting_tosa": {},
    "visual_wake_words_tosa": {},
    "squeezenet_like_10": {},
    "squeezenet_like_12": {},
    "squeezenet_like_15": {},
    "squeezenet_like_1000": {},
    "squeezenet_like_10000": {},
}


def _apply_recursive_defaults(target: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        if isinstance(value, Mapping):
            current = target.get(key, None)
            if isinstance(current, Mapping):
                merged = dict(current)
                _apply_recursive_defaults(merged, value)
                target[key] = merged
            elif key not in target:
                nested: dict[str, Any] = {}
                _apply_recursive_defaults(nested, value)
                target[key] = nested
            continue
        target.setdefault(key, value)


def _resolve_dataset_name(config: Mapping[str, Any] | None) -> str | None:
    if not isinstance(config, Mapping):
        return None

    graph_file = str(config.get("graph-file", "") or "").strip()
    if graph_file:
        stem = Path(graph_file).stem
        return stem or None

    taskgraph_pickle = str(config.get("taskgraph-pickle", "") or "").strip()
    if not taskgraph_pickle:
        return None

    stem = Path(taskgraph_pickle).stem
    if stem.startswith("taskgraph-"):
        stem = stem[len("taskgraph-") :]
    if "_area-" in stem:
        stem = stem.split("_area-", 1)[0]
    return stem or None


def get_method_runtime_defaults(method_name: str) -> dict[str, Any]:
    return deepcopy(_METHOD_RUNTIME_DEFAULTS.get(str(method_name).lower(), {}))


def get_classical_method_defaults(method_name: str) -> dict[str, Any]:
    return get_method_runtime_defaults(method_name)


def get_method_runtime_profile_name() -> str:
    return _ACTIVE_METHOD_RUNTIME_PROFILE


def get_method_dataset_overrides(dataset_name: str | None, method_name: str) -> dict[str, Any]:
    if not dataset_name:
        return {}
    per_dataset = _METHOD_DATASET_OVERRIDES.get(str(dataset_name), {})
    if not isinstance(per_dataset, Mapping):
        return {}
    overrides = per_dataset.get(str(method_name).lower(), {})
    if not isinstance(overrides, Mapping):
        return {}
    return deepcopy(dict(overrides))


def resolve_method_runtime_config(
    config: Mapping[str, Any] | None,
    method_name: str,
) -> dict[str, Any]:
    method_key = str(method_name).lower()
    resolved: dict[str, Any] = {}

    if isinstance(config, Mapping):
        user_cfg = config.get(method_key, {})
        if isinstance(user_cfg, Mapping):
            resolved.update(deepcopy(dict(user_cfg)))

    dataset_name = _resolve_dataset_name(config)
    dataset_overrides = get_method_dataset_overrides(dataset_name, method_key)
    if dataset_overrides:
        _apply_recursive_defaults(resolved, dataset_overrides)

    global_defaults = get_method_runtime_defaults(method_key)
    if global_defaults:
        _apply_recursive_defaults(resolved, global_defaults)

    return resolved


def resolve_classical_method_config(
    config: Mapping[str, Any] | None,
    method_name: str,
) -> dict[str, Any]:
    return resolve_method_runtime_config(config, method_name)


def get_classical_method_names() -> tuple[str, ...]:
    return tuple(_CLASSICAL_SEARCH_OBJECTIVE_METHODS)


def is_classical_search_method(method_name: str) -> bool:
    return str(method_name).lower() in _CLASSICAL_SEARCH_OBJECTIVE_METHODS


def get_classical_search_objective_default() -> str:
    return str(_CLASSICAL_GLOBAL_DEFAULTS["search_objective"])


__all__ = [
    "get_classical_method_defaults",
    "get_classical_method_names",
    "get_classical_search_objective_default",
    "get_method_dataset_overrides",
    "get_method_runtime_profile_name",
    "get_method_runtime_defaults",
    "is_classical_search_method",
    "resolve_classical_method_config",
    "resolve_method_runtime_config",
]
