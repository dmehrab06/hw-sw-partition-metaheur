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


def get_classical_method_defaults(method_name: str) -> dict[str, Any]:
    return deepcopy(_CLASSICAL_METHOD_DEFAULTS.get(str(method_name).lower(), {}))


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
    "resolve_classical_method_config",
]
