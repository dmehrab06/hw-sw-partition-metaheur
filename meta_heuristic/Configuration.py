from __future__ import annotations

try:
    from .method_runtime_config import (
        get_classical_method_defaults,
        get_classical_method_names,
        get_classical_search_objective_default,
        get_method_runtime_profile_name,
        get_method_runtime_defaults,
        is_classical_search_method,
        resolve_classical_method_config,
        resolve_method_runtime_config,
    )
except ImportError:
    from method_runtime_config import (  # type: ignore
        get_classical_method_defaults,
        get_classical_method_names,
        get_classical_search_objective_default,
        get_method_runtime_profile_name,
        get_method_runtime_defaults,
        is_classical_search_method,
        resolve_classical_method_config,
        resolve_method_runtime_config,
    )

__all__ = [
    "get_classical_method_defaults",
    "get_classical_method_names",
    "get_classical_search_objective_default",
    "get_method_runtime_profile_name",
    "get_method_runtime_defaults",
    "is_classical_search_method",
    "resolve_classical_method_config",
    "resolve_method_runtime_config",
]
