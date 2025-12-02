"""
Synthetic task-graph generator compatible with the project's TaskGraph attributes.
Provides:
 - generate_synthetic_taskgraph(N, E, ...)
 - generate_taskgraph_from_spec(nodes, edges, ...)
 - attach_to_TaskGraph(attrs)  # optional helper to create TaskGraph instance
 - visualize_taskgraph(attrs_or_TG, out_path=None)
 - save_config_yaml(path, params)
"""
from typing import List, Tuple, Dict, Any
import networkx as nx
import numpy as np
import random
import yaml
import os

def _ensure_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def generate_synthetic_taskgraph(
    N: int,
    E: int,
    seed: int = 42,
    k: float = 0.1,                # hw-scale-factor (keeps same naming as config)
    l: float = 0.5,                # hw-scale-variance
    mu: float = 1.0,               # comm-scale-factor
    A_max: float = 100.0,
    directed: bool = True
) -> Dict[str, Any]:
    """
    Generate a synthetic DAG with N nodes and E edges (at most N*(N-1)/2).
    Returns a dict with keys:
      - graph: networkx.DiGraph
      - software_costs, hardware_costs, hardware_area, communication_costs (dicts)
      - node_to_num, num_to_node, total_area
    The assignment rules mirror the TaskGraph.load_graph_from_pydot approach.
    """
    _ensure_seed(seed)

    # create DAG by ordering nodes and only allowing edges from lower->higher index
    nodes = list(range(N))
    order = nodes.copy()
    random.shuffle(order)
    max_edges = N * (N - 1) // 2
    E = min(E, max_edges)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(order)

    # sample E distinct directed edges respecting the ordering
    possible_edges = [(u, v) for i, u in enumerate(order) for v in order[i+1:]]
    chosen = random.sample(possible_edges, E)
    G.add_edges_from(chosen)

    # software costs ~ Uniform[1,100]
    software_costs = {n: float(np.random.uniform(1.0, 100.0)) for n in order}
    s_max = max(software_costs.values())

    # hardware area ~ Uniform[1, A_max]
    hardware_area = {n: float(np.random.uniform(1.0, A_max)) for n in order}
    total_area = float(sum(hardware_area.values()))

    # hardware costs: Normal(mean = k * s_i, sd = l * k * s_i). Clip to small positive.
    hardware_costs = {}
    for n, s in software_costs.items():
        mean = k * s
        sd = abs(l * k * s) if (l is not None) else 0.0
        # if sd == 0 then it's deterministic
        val = float(np.random.normal(loc=mean, scale=sd)) if sd > 0 else float(mean)
        hardware_costs[n] = max(1e-6, val)

    # communication costs for each edge ~ Uniform[0, 2 * mu * s_max]
    communication_costs = {}
    for u, v in G.edges():
        communication_costs[(u, v)] = float(np.random.uniform(0.0, 2.0 * mu * s_max))

    # maps
    node_to_num = {n: i for i, n in enumerate(order)}
    num_to_node = {i: n for n, i in node_to_num.items()}

    # attach node/edge attributes on the graph to make it easy to visualize/use
    nx.set_node_attributes(G, {n: software_costs[n] for n in G.nodes()}, 'software_time')
    nx.set_node_attributes(G, {n: hardware_costs[n] for n in G.nodes()}, 'hardware_time')
    nx.set_node_attributes(G, {n: hardware_area[n] for n in G.nodes()}, 'area_cost')
    nx.set_edge_attributes(G, { (u,v): communication_costs[(u,v)] for u,v in G.edges() }, 'communication_cost')

    return {
        'graph': G,
        'software_costs': software_costs,
        'hardware_costs': hardware_costs,
        'hardware_area': hardware_area,
        'communication_costs': communication_costs,
        'node_to_num': node_to_num,
        'num_to_node': num_to_node,
        'total_area': total_area,
        'seed': seed,
        'params': {'k': k, 'l': l, 'mu': mu, 'A_max': A_max}
    }

def generate_taskgraph_from_spec(
    nodes: List[Any],
    edges: List[Tuple[Any, Any]],
    seed: int = 42,
    k: float = 0.1,
    l: float = 0.5,
    mu: float = 1.0,
    A_max: float = 100.0,
    directed: bool = True
) -> Dict[str, Any]:
    """
    Build a graph from explicit node list and edge list.
    Node labels can be any hashable object. Edges are list of (u,v) tuples.
    All other costs/areas/comm times are assigned similarly to the synthetic generator.
    """
    _ensure_seed(seed)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # software costs
    software_costs = {n: float(np.random.uniform(1.0, 100.0)) for n in nodes}
    s_max = max(software_costs.values())
    hardware_area = {n: float(np.random.uniform(1.0, A_max)) for n in nodes}
    total_area = float(sum(hardware_area.values()))

    hardware_costs = {}
    for n, s in software_costs.items():
        mean = k * s
        sd = abs(l * k * s)
        val = float(np.random.normal(loc=mean, scale=sd)) if sd > 0 else float(mean)
        hardware_costs[n] = max(1e-6, val)

    communication_costs = {}
    for u, v in G.edges():
        communication_costs[(u, v)] = float(np.random.uniform(0.0, 2.0 * mu * s_max))

    node_to_num = {n: i for i, n in enumerate(nodes)}
    num_to_node = {i: n for n, i in node_to_num.items()}

    nx.set_node_attributes(G, {n: software_costs[n] for n in G.nodes()}, 'software_time')
    nx.set_node_attributes(G, {n: hardware_costs[n] for n in G.nodes()}, 'hardware_time')
    nx.set_node_attributes(G, {n: hardware_area[n] for n in G.nodes()}, 'area_cost')
    nx.set_edge_attributes(G, { (u,v): communication_costs[(u,v)] for u,v in G.edges() }, 'communication_cost')

    return {
        'graph': G,
        'software_costs': software_costs,
        'hardware_costs': hardware_costs,
        'hardware_area': hardware_area,
        'communication_costs': communication_costs,
        'node_to_num': node_to_num,
        'num_to_node': num_to_node,
        'total_area': total_area,
        'seed': seed,
        'params': {'k': k, 'l': l, 'mu': mu, 'A_max': A_max}
    }

def attach_to_TaskGraph(attrs: Dict[str, Any]):
    """
    Try to create/attach attributes to your project's TaskGraph object.
    If the import fails it returns the attrs dict unchanged.
    Usage: TG = attach_to_TaskGraph(attrs)
    """
    try:
        from meta_heuristic.task_graph import TaskGraph
    except Exception:
        # TaskGraph not importable or incomplete; return raw attrs
        return attrs

    TG = TaskGraph(area_constraint=0.5)  # default area_constraint can be adjusted later
    TG.graph = attrs['graph']
    TG.software_costs = attrs['software_costs']
    TG.hardware_costs = attrs['hardware_costs']
    TG.hardware_area = attrs['hardware_area']
    TG.communication_costs = attrs['communication_costs']
    TG.node_to_num = attrs['node_to_num']
    TG.num_to_node = attrs['num_to_node']
    TG.total_area = attrs['total_area']
    # create rounak_graph as used in code
    TG.rounak_graph = TG.graph.copy()
    nx.set_node_attributes(TG.rounak_graph, TG.hardware_area, 'area_cost')
    nx.set_node_attributes(TG.rounak_graph, TG.hardware_costs, 'hardware_time')
    nx.set_node_attributes(TG.rounak_graph, TG.software_costs, 'software_time')
    nx.set_edge_attributes(TG.rounak_graph, TG.communication_costs, 'communication_cost')
    return TG

# def visualize_taskgraph(attrs_or_TG, out_path: str = None, figsize=(10, 8), save_pdf: bool = True):
#     """
#     Simple visualizer using networkx + matplotlib. Colors nodes by software_time (or hardware_time if present).
#     Accepts either the attrs dict returned by generator functions or an instance of TaskGraph.
#     If save_pdf is True the figure will be saved to the Figs/ directory with a filename derived from graph
#     and generator parameters (N, E, k, l, mu, seed, A_max).
#     """
#     try:
#         import matplotlib.pyplot as plt
#     except Exception:
#         raise RuntimeError("matplotlib is required for visualization (pip install matplotlib)")

#     if hasattr(attrs_or_TG, 'graph'):
#         G = attrs_or_TG.graph
#         sw = getattr(attrs_or_TG, 'software_costs', None)
#         params = getattr(attrs_or_TG, 'params', None)
#         seed = getattr(attrs_or_TG, 'seed', None)
#     elif isinstance(attrs_or_TG, dict):
#         G = attrs_or_TG['graph']
#         sw = attrs_or_TG.get('software_costs')
#         params = attrs_or_TG.get('params', {})
#         seed = attrs_or_TG.get('seed', None)
#     else:
#         raise ValueError("Input must be attributes dict or TaskGraph instance")

#     pos = nx.spring_layout(G, seed=seed if seed is not None else 42)
#     node_sizes = [ (sw[n] if sw and n in sw else 1.0) * 10.0 for n in G.nodes() ]
#     node_labels = {n: str(n) for n in G.nodes()}
#     plt.figure(figsize=figsize)
#     nx.draw_networkx_edges(G, pos, alpha=0.6)
#     nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='C0')
#     nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
#     title = f"Task Graph | |V|={G.number_of_nodes()} |E|={G.number_of_edges()}"
#     plt.title(title)
#     plt.axis('off')

#     # construct filename from params if requested
#     if save_pdf:
#         def _fmt(x, fmt="{:.2f}"):
#             if x is None:
#                 return "NA"
#             try:
#                 # format floats to two decimals, replace dot with 'p' for filename safety
#                 if isinstance(x, float):
#                     return fmt.format(x).replace('.', 'p')
#                 return str(x)
#             except Exception:
#                 return str(x)

#         N = G.number_of_nodes()
#         E = G.number_of_edges()
#         k = params.get('k') if params else None
#         l = params.get('l') if params else None
#         mu = params.get('mu') if params else None
#         A_max = params.get('A_max') if params else None

#         seed_str = _fmt(seed, "{:.0f}") if seed is not None else "NA"
#         fname = (f"synthetic_nodes-{N}_edges-{E}_k-{_fmt(k)}_l-{_fmt(l)}_mu-{_fmt(mu)}"
#                  f"_seed-{seed_str}_Amax-{_fmt(A_max)}.pdf")
#         figs_dir = os.path.join("Figs")
#         os.makedirs(figs_dir, exist_ok=True)
#         pdf_path = os.path.join(figs_dir, fname)
#         plt.savefig(pdf_path, bbox_inches='tight', format='pdf', dpi=200)

#         # if an explicit out_path was provided, also save there (keeps previous behavior)
#         if out_path:
#             os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
#             plt.savefig(out_path, bbox_inches='tight', dpi=200)

#         plt.close()
#         return pdf_path

#     # fallback: show or save to explicit out_path
#     if out_path:
#         os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
#         plt.savefig(out_path, bbox_inches='tight', dpi=200)
#         plt.close()
#         return out_path
#     else:
#         plt.show()
#         return None

def visualize_taskgraph(attrs_or_TG, out_path: str = None, figsize=(10, 8), save_pdf: bool = True, fig_name: str = None, show: bool = False):
    """
    Wrapper that delegates visualization to figures.utils.visualize_task_graph.
    Accepts either the attrs dict returned by generator functions or an instance of TaskGraph.

    Parameters:
      - attrs_or_TG: dict or TaskGraph instance
      - out_path: (ignored) kept for compatibility
      - figsize: forwarded to visualize_task_graph
      - save_pdf: honored by underlying visualize_task_graph via out_dir and fig_name
      - fig_name: optional filename prefix
      - show: if True the figure will be displayed (forwarded)

    Returns:
      path to the saved PDF (as returned by visualize_task_graph)
    """
    # convert attrs dict to TaskGraph instance if needed
    TG = None
    if hasattr(attrs_or_TG, 'graph'):
        TG = attrs_or_TG
    elif isinstance(attrs_or_TG, dict):
        TG = attach_to_TaskGraph(attrs_or_TG)
        # attach_to_TaskGraph may return the original dict if TaskGraph import failed.
        # Wrap a dict into a lightweight object with required attributes so downstream code
        # (figures.utils.visualize_task_graph) can access .graph, .software_costs, etc.
        if isinstance(TG, dict):
            class _DummyTG:
                pass
            dummy = _DummyTG()
            dummy.graph = TG.get('graph')
            dummy.software_costs = TG.get('software_costs', {})
            dummy.hardware_costs = TG.get('hardware_costs', {})
            dummy.hardware_area = TG.get('hardware_area', {})
            dummy.communication_costs = TG.get('communication_costs', {})
            dummy.node_to_num = TG.get('node_to_num', {})
            dummy.num_to_node = TG.get('num_to_node', {})
            dummy.total_area = TG.get('total_area', None)
            # preserve params/seed/area_constraint for filename composition
            dummy.params = TG.get('params', {})
            dummy.seed = TG.get('seed', None)
            dummy.area_constraint = TG.get('area_constraint', None)
            TG = dummy
    else:
        raise ValueError("Input must be attributes dict or TaskGraph instance")

    # build a config dict compatible with utils._config_basename
    params = {}
    if isinstance(attrs_or_TG, dict):
        params = attrs_or_TG.get('params', {}) or {}
        seed = attrs_or_TG.get('seed', None)
    else:
        params = getattr(attrs_or_TG, 'params', {}) or {}
        seed = getattr(attrs_or_TG, 'seed', None)

    config = {
        # keep names utils expects to compose filenames
        'graph-file': params.get('graph-file') or f"synthetic_nodes_{TG.graph.number_of_nodes()}",
        'config': params.get('config', None),
        'area-constraint': getattr(TG, 'area_constraint', None),
        'hw-scale-factor': params.get('k', params.get('hw-scale-factor', None)),
        'hw-scale-variance': params.get('l', params.get('hw-scale-variance', None)),
        'comm-scale-factor': params.get('mu', params.get('comm-scale-factor', None)),
        'seed': seed
    }

    # delegate to figures.utils.visualize_task_graph
    try:
        from utils import visualize_task_graph
    except Exception as exc:
        raise RuntimeError("Failed to import figures.utils.visualize_task_graph: " + str(exc))

    # call visualize_task_graph which saves PDF into the specified out_dir ('Figs' here)
    out_dir = "Figs"

    saved_path = visualize_task_graph(config=config, task_graph=TG, out_dir=out_dir, fig_name=fig_name, figsize=figsize, show=show)

    return saved_path

def save_config_yaml(path: str, params: Dict[str, Any]):
    """
    Write a small YAML config file. Simple helper.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(params, f, sort_keys=False)
    return path


# Example quick-run (not executed on import)
if __name__ == "__main__":
    attrs = generate_synthetic_taskgraph(4, 5, seed=123, k=0.1, l=0.5, mu=1.0, A_max=50)
    print("Generated graph nodes:", attrs['graph'].nodes())
    visualize_taskgraph(attrs)