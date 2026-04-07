from __future__ import annotations

from importlib import import_module

__all__ = [
    "TaskGraph",
    "parse_arguments",
    "simulate_PSO",
    "simulate_DBPSO",
    "simulate_CLPSO",
    "simulate_CCPSO",
    "random_assignment",
    "simulate_GL25",
    "simulate_ESA",
    "simulate_SHADE",
    "simulate_JADE",
    "simulate_nondiff_GNN",
    "simulate_diff_GNN",
    "simulate_diff_GNN_order",
    "simulate_gcps",
]


_LAZY_EXPORTS = {
    "TaskGraph": (".task_graph", "TaskGraph"),
    "parse_arguments": (".parser_utils", "parse_arguments"),
    "simulate_PSO": (".pso_utils", "simulate_PSO"),
    "simulate_DBPSO": (".pso_utils", "simulate_DBPSO"),
    "simulate_CLPSO": (".pso_utils", "simulate_CLPSO"),
    "simulate_CCPSO": (".pso_utils", "simulate_CCPSO"),
    "simulate_GL25": (".ga_utils", "simulate_GL25"),
    "simulate_ESA": (".sa_utils", "simulate_ESA"),
    "simulate_SHADE": (".de_utils", "simulate_SHADE"),
    "simulate_JADE": (".de_utils", "simulate_JADE"),
    "simulate_nondiff_GNN": (".nondiff_gnn_utils", "simulate_nondiff_GNN"),
    "simulate_diff_GNN": (".diff_gnn_utils_schedule", "simulate_diff_GNN"),
    "simulate_diff_GNN_order": (".diff_gnn_ordering", "simulate_diff_GNN_order"),
    "simulate_gcps": (".gcps", "simulate_gcps"),
}


def __getattr__(name: str):
    if name == "random_assignment":
        return random_assignment

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def random_assignment(dim, func_to_optimize, config):
    import numpy as np
    from .Configuration import resolve_classical_method_config

    random_cfg = resolve_classical_method_config(config, "random")
    all_samples = []
    for _ in range(int(random_cfg["num_samples"])):
        bernoulli_samples = np.random.binomial(n=1, p=float(random_cfg["p"]), size=dim)
        all_samples.append(bernoulli_samples)

    sample_array = np.array(all_samples)
    all_costs = func_to_optimize(sample_array)

    best_cost = np.min(all_costs)
    min_index = np.argmin(all_costs)
    best_solution = all_samples[min_index]

    return best_cost, best_solution
