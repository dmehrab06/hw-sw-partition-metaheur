"""
Hardware-Software Partitioning Optimization Solver
Implements the incidence matrix formulation for DAG partitioning
"""

import sys, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cvxpy as cp
import pickle
import random
import pydot
from typing import Any, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from utils.scheduler_utils import compute_dag_makespan
from meta_heuristic.task_graph import TaskGraph

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/schedule_partitioner.log")

logger = LogManager.get_logger(__name__)

class ScheduleConstPartitionSolver:
    """
    Schedule Constrained Hardware-Software Partitioning Solver using MILP formulation
    """
    
    def __init__(self):
        self.graph = None
        self.n_nodes = 0
        self.n_edges = 0
        self.edge_list = []
        
        # Problem matrices and vectors
        self.h = None  # hardware execution times
        self.s = None  # software execution times  
        self.a = None  # hardware area costs
        self.c = None  # communication costs
        self.S_source = None  # source selection matrix
        self.S_target = None  # target selection matrix
        
        # Solution
        self.x_sol = None  # partition assignment
        self.t_sol = None  # start times
        self.f_sol = None  # finish times
        self.z_sol = None  # communication variables
        self.T_sol = None  # makespan
        self.Y_sol = None  # software ordering

    def load_pickle_graph(self, graph_file):
        try:
            with open(graph_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded successfully from {graph_file}")
            # You can now work with the loaded graph 'G'
        except FileNotFoundError:
            logger.error(f"Error: The pickle file '{graph_file}' was not found.")
            raise FileNotFoundError(f"Error: The pickle file '{graph_file}' was not found.")
        except Exception as e:
            logger.error(f"An error occurred while loading the graph data: {e}")
            raise
        
        self.populate_graph_with_attributes(data)
        self.n_nodes = len(self.graph.nodes())
        self.n_edges = self.graph.number_of_edges()
        self.edge_list = list(self.graph.edges())

        # Create problem vectors
        self._create_problem_matrices()
        logger.info(f"Created DAG with {self.n_nodes} nodes and {self.n_edges} edges")
        return self.graph
    
    def populate_graph_with_attributes(self, data:TaskGraph):
        self.graph = data.graph
        # node/edge attributes
        nx.set_node_attributes(self.graph, data.hardware_area, 'area_cost')
        nx.set_node_attributes(self.graph, data.hardware_costs, 'hardware_time')
        nx.set_node_attributes(self.graph, data.software_costs, 'software_time')
        nx.set_edge_attributes(self.graph, data.communication_costs, 'communication_cost')
        return
        
    def load_pydot_graph(self, pydot_file, k=1.5, l=0.2, mu=0.5, A_max=100) -> nx.DiGraph:
        """
        Create a random directed acyclic graph with required attributes
        
        Args:
            
            
        Returns:
            NetworkX directed graph
        """
        # Robust DOT loader (avoids NetworkX/pydot get_strict() bug)
        pydot_graphs = pydot.graph_from_dot_file(pydot_file)
        if not pydot_graphs:
            raise ValueError(f"Could not load DOT file: {pydot_file}")
        pgraph = pydot_graphs[0]

        g = nx.DiGraph()

        def _collect_nodes_edges(pg):
            # Add nodes, skip pydot defaults ('node', 'graph', 'edge')
            for node in pg.get_nodes():
                name = node.get_name().strip('"')
                if name in {"node", "graph", "edge"}:
                    continue
                g.add_node(name)
            # Add edges
            for edge in pg.get_edges():
                src = edge.get_source().strip('"')
                dst = edge.get_destination().strip('"')
                g.add_edge(src, dst)
            # Recurse into subgraphs (DOT clusters)
            for sg in pg.get_subgraphs():
                _collect_nodes_edges(sg)

        _collect_nodes_edges(pgraph)
        self.graph = g
        logger.info(f"Loaded graph from {pydot_file} with {len(self.graph.nodes())} nodes")

        # Assign node attributes
        for node in self.graph.nodes():
            
            # Assign software execution time and hardware areas
            software_time = random.uniform(1, 100)
            hardware_area = random.uniform(1, A_max)
            self.graph.nodes[node]['software_time'] = software_time
            self.graph.nodes[node]['area_cost'] = hardware_area

            # Assign random hardware execution time
            mean = k * software_time
            std_dev = l * k * software_time
            # Ensure hardware costs are positive
            hardware_time = max(0.1, np.random.normal(mean, std_dev))
            self.graph.nodes[node]['hardware_time'] = hardware_time

        # Calculate total area and maximum software cost
        s_max = max([self.graph.nodes[n]['software_time'] for n in self.graph.nodes])
        total_area = sum([self.graph.nodes[n]['area_cost'] for n in self.graph.nodes])

        # # Ensure graph connectivity
        # assert nx.is_connected(self.graph), "Graph must be connected"
        
        # Assign communication costs
        for u, v in self.graph.edges():
            comm_cost = random.uniform(0, 2 * mu * s_max)
            self.graph[u][v]['communication_cost'] = comm_cost
            
        logger.info(f"Graph initialized with total hardware area requirement: {total_area}")
        
        self.n_nodes = len(self.graph.nodes())
        self.n_edges = self.graph.number_of_edges()
        self.edge_list = list(self.graph.edges())

        # Create problem vectors
        self._create_problem_matrices()
        logger.info(f"Created DAG with {self.n_nodes} nodes and {self.n_edges} edges")
        return self.graph
    
    
    def _create_problem_matrices(self):
        """Create matrices and vectors for optimization problem"""
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        # Get sorted list of node IDs for consistent ordering
        self.node_list = sorted(list(self.graph.nodes()))
        self.node_to_index = {node: i for i, node in enumerate(self.node_list)}
        self.index_to_node = {i: node for i, node in enumerate(self.node_list)}
        
        # Create node attribute vectors (indexed by position in node_list)
        self.h = np.array([self.graph.nodes[node]['hardware_time'] for node in self.node_list])
        self.s = np.array([self.graph.nodes[node]['software_time'] for node in self.node_list])
        self.a = np.array([self.graph.nodes[node]['area_cost'] for node in self.node_list])
        
        # Create communication cost vector
        self.c = np.array([self.graph.edges[edge]['communication_cost'] for edge in self.edge_list])
        
        # Create selection matrices
        self.S_source = np.zeros((self.n_edges, self.n_nodes))
        self.S_target = np.zeros((self.n_edges, self.n_nodes))
        
        for k, (source_node, target_node) in enumerate(self.edge_list):
            source_idx = self.node_to_index[source_node]
            target_idx = self.node_to_index[target_node]
            self.S_source[k, source_idx] = 1  # edge k starts from source_node
            self.S_target[k, target_idx] = 1  # edge k ends at target_node
        
        logger.info("Problem matrices and vectors created successfully")
    
    def solve_optimization(
        self,
        A_max: float,
        big_M: float = 1000,
        use_reduced_sw_constraints: bool = True,
        solver_cfg: Dict[str, Any] = None,
    ) -> Dict:
        """
        Solve the HW/SW partitioning problem.

        Modes (configured via solver_cfg["solve-mode"]):
        - exact: MILP only
        - hybrid: MILP first, fallback to relaxed+rounding if no incumbent
        - relaxed: LP relaxation + rounding/repair (fast approximation)
        """
        if self.S_source is None:
            raise ValueError("Problem matrices not created. Load a graph first.")

        cfg = self._normalize_solver_cfg(solver_cfg, use_reduced_sw_constraints)
        solve_mode = str(cfg.get("solve-mode", "exact")).lower()
        sw_constraint_mode = str(cfg.get("sw-constraint-mode", "pairwise_topo")).lower()
        round_threshold = float(cfg.get("round-threshold", 0.5))

        if solve_mode not in {"exact", "hybrid", "relaxed"}:
            logger.warning(f"Unknown solve mode '{solve_mode}', falling back to hybrid")
            solve_mode = "hybrid"

        logger.info(
            f"Solving partition optimization: mode={solve_mode}, A_max={A_max:.4f}, "
            f"sw_constraints={sw_constraint_mode}, reduced_sw={cfg['use-reduced-sw-constraints']}"
        )

        if solve_mode in {"exact", "hybrid"}:
            raw = self._solve_cvxpy_model(
                A_max=A_max,
                big_M=big_M,
                use_reduced_sw_constraints=cfg["use-reduced-sw-constraints"],
                sw_constraint_mode=sw_constraint_mode,
                solver_cfg=cfg,
                relax_integrality=False,
            )
            if raw is not None and raw.get("x") is not None and raw.get("T") is not None:
                status = str(raw.get("status", "unknown"))
                if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or cfg.get("accept-nonoptimal", True):
                    return self._finalize_solution(
                        x_values=raw["x"],
                        t_values=raw["t"],
                        f_values=raw["f"],
                        z_values=raw["z"],
                        makespan_value=raw["T"],
                        A_max=A_max,
                        status=status,
                    )

            if solve_mode == "exact":
                logger.error("Exact MILP did not return an acceptable solution")
                return None

            logger.warning("Hybrid mode fallback: MILP had no usable incumbent, switching to relaxed rounding")

        raw_relaxed = self._solve_cvxpy_model(
            A_max=A_max,
            big_M=big_M,
            use_reduced_sw_constraints=cfg["use-reduced-sw-constraints"],
            sw_constraint_mode=sw_constraint_mode,
            solver_cfg=cfg,
            relax_integrality=True,
        )

        if raw_relaxed is not None and raw_relaxed.get("x") is not None:
            x_candidate = self._round_and_repair_assignment(
                x_relaxed=raw_relaxed["x"],
                A_max=A_max,
                threshold=round_threshold,
            )
            status = f"{raw_relaxed.get('status', 'unknown')}_relaxed_rounded"
            return self._solution_from_assignment(x_candidate, A_max, status=status)

        logger.warning("Relaxed solve failed, using greedy area-constrained fallback assignment")
        x_candidate = self._greedy_assignment(A_max)
        return self._solution_from_assignment(x_candidate, A_max, status="greedy_fallback")

    def _normalize_solver_cfg(self, solver_cfg: Dict[str, Any], use_reduced_sw_constraints: bool) -> Dict[str, Any]:
        defaults = {
            "solve-mode": "exact",
            "verbose": False,
            "preferred-solvers": ["GUROBI", "HIGHS", "SCIP", "GLPK_MI", "CBC", "ECOS_BB"],
            "time-limit-sec": None,
            "mip-gap": None,
            "node-limit": None,
            "accept-nonoptimal": True,
            "sw-constraint-mode": "pairwise_topo",
            "round-threshold": 0.5,
            "use-reduced-sw-constraints": use_reduced_sw_constraints,
        }

        user_cfg: Dict[str, Any] = {}
        if solver_cfg is not None:
            if isinstance(solver_cfg, dict):
                user_cfg = dict(solver_cfg)
            else:
                try:
                    user_cfg = {str(k): solver_cfg[k] for k in solver_cfg.keys()}
                except Exception:
                    try:
                        user_cfg = dict(solver_cfg)
                    except Exception:
                        user_cfg = {}

        cfg = dict(defaults)
        cfg.update(user_cfg)

        preferred = cfg.get("preferred-solvers", defaults["preferred-solvers"])
        try:
            cfg["preferred-solvers"] = [str(s) for s in preferred]
        except Exception:
            cfg["preferred-solvers"] = defaults["preferred-solvers"]

        cfg["verbose"] = bool(cfg.get("verbose", False))
        cfg["accept-nonoptimal"] = bool(cfg.get("accept-nonoptimal", True))
        cfg["use-reduced-sw-constraints"] = bool(cfg.get("use-reduced-sw-constraints", use_reduced_sw_constraints))
        return cfg

    def _build_solver_kwargs(self, solver_name: str, solver_cfg: Dict[str, Any], relax_integrality: bool) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"verbose": bool(solver_cfg.get("verbose", False))}
        if relax_integrality:
            return kwargs

        time_limit = solver_cfg.get("time-limit-sec", None)
        mip_gap = solver_cfg.get("mip-gap", None)
        node_limit = solver_cfg.get("node-limit", None)

        if solver_name == "GUROBI":
            if time_limit is not None:
                kwargs["TimeLimit"] = float(time_limit)
            if mip_gap is not None:
                kwargs["MIPGap"] = float(mip_gap)
            if node_limit is not None:
                kwargs["NodeLimit"] = int(node_limit)
        elif solver_name == "SCIP":
            scip_params = {}
            if time_limit is not None:
                scip_params["limits/time"] = float(time_limit)
            if mip_gap is not None:
                scip_params["limits/gap"] = float(mip_gap)
            if node_limit is not None:
                scip_params["limits/nodes"] = int(node_limit)
            if scip_params:
                kwargs["scip_params"] = scip_params
        elif solver_name == "HIGHS":
            highs_options = {}
            if time_limit is not None:
                highs_options["time_limit"] = float(time_limit)
            if mip_gap is not None:
                highs_options["mip_rel_gap"] = float(mip_gap)
            if node_limit is not None:
                highs_options["mip_max_nodes"] = int(node_limit)
            if highs_options:
                kwargs["highs_options"] = highs_options
        elif solver_name == "CBC":
            if time_limit is not None:
                kwargs["maximumSeconds"] = float(time_limit)

        return kwargs

    def _solve_problem_with_preferred_solvers(
        self,
        problem: cp.Problem,
        solver_cfg: Dict[str, Any],
        relax_integrality: bool,
    ) -> str:
        preferred = solver_cfg.get("preferred-solvers", [])
        installed = set(cp.installed_solvers())
        last_err = None

        for solver_name in preferred:
            if solver_name not in installed:
                continue

            solver_attr = getattr(cp, solver_name, None)
            if solver_attr is None:
                continue

            kwargs = self._build_solver_kwargs(solver_name, solver_cfg, relax_integrality)
            try:
                logger.info(f"[cvxpy] Trying solver: {solver_name} with options={kwargs}")
                problem.solve(solver=solver_attr, **kwargs)
                return solver_name
            except Exception as err_with_opts:
                last_err = err_with_opts
                # Retry once without any solver-specific options.
                try:
                    logger.info(f"[cvxpy] Retrying solver {solver_name} without solver-specific options")
                    problem.solve(solver=solver_attr, verbose=bool(solver_cfg.get("verbose", False)))
                    return solver_name
                except Exception as err_no_opts:
                    last_err = err_no_opts
                    logger.warning(f"[cvxpy] Solver {solver_name} failed: {err_no_opts}")
                    continue

        raise RuntimeError(
            f"No solver succeeded. Installed={sorted(installed)}; preferred={preferred}; last_err={last_err}"
        )

    def _solve_cvxpy_model(
        self,
        A_max: float,
        big_M: float,
        use_reduced_sw_constraints: bool,
        sw_constraint_mode: str,
        solver_cfg: Dict[str, Any],
        relax_integrality: bool,
    ) -> Dict[str, Any]:
        if relax_integrality:
            x = cp.Variable(self.n_nodes)
            z = cp.Variable(self.n_edges)
        else:
            x = cp.Variable(self.n_nodes, boolean=True)
            z = cp.Variable(self.n_edges, boolean=True)

        t = cp.Variable(self.n_nodes, nonneg=True)
        f = cp.Variable(self.n_nodes, nonneg=True)
        T = cp.Variable(nonneg=True)
        objective = cp.Minimize(T)
        constraints = []

        if relax_integrality:
            constraints.append(x >= 0)
            constraints.append(x <= 1)
            constraints.append(z >= 0)
            constraints.append(z <= 1)

        constraints.append(self.a.T @ (1 - x) <= A_max)
        constraints.append(f == t + cp.multiply(self.h, 1 - x) + cp.multiply(self.s, x))
        constraints.append(self.S_source @ f + cp.multiply(self.c, z) <= self.S_target @ t)
        constraints.append(z >= self.S_source @ x - self.S_target @ x)
        constraints.append(z >= self.S_target @ x - self.S_source @ x)
        constraints.append(z <= self.S_source @ x + self.S_target @ x)
        constraints.append(z <= 2 - self.S_source @ x - self.S_target @ x)

        Y = None
        use_reduced_sw_constraints = False
        if use_reduced_sw_constraints:
            topo_order = list(nx.topological_sort(self.graph))
            topo_indices = [self.node_to_index[node] for node in topo_order]

            if sw_constraint_mode in {"adjacent", "adjacent_topo", "consecutive"}:
                for pos in range(len(topo_indices) - 1):
                    i = topo_indices[pos]
                    j = topo_indices[pos + 1]
                    constraints.append(f[i] <= t[j] + big_M * (1 - x[i]) + big_M * (1 - x[j]))
            else:
                for a in range(len(topo_indices) - 1):
                    i = topo_indices[a]
                    for b in range(a + 1, len(topo_indices)):
                        j = topo_indices[b]
                        constraints.append(f[i] <= t[j] + big_M * (1 - x[i]) + big_M * (1 - x[j]))
        else:
            if relax_integrality:
                Y = cp.Variable((self.n_nodes, self.n_nodes))
                constraints.append(Y >= 0)
                constraints.append(Y <= 1)
            else:
                Y = cp.Variable((self.n_nodes, self.n_nodes), boolean=True)

            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    # if i == j: #rounak
                        # continue 
                    if i<j:
                        constraints.append(f[i] <= t[j] + big_M * (1 - Y[i, j]) + big_M * (2 - x[i] - x[j]))
                        constraints.append(f[j] <= t[i] + big_M * Y[i, j] + big_M * (2 - x[i] - x[j]))
                        constraints.append(Y[i, j] <= x[i])
                        constraints.append(Y[i, j] <= x[j])

        for i in range(self.n_nodes):
            constraints.append(T >= f[i])

        problem = cp.Problem(objective, constraints)

        try:
            solver_used = self._solve_problem_with_preferred_solvers(problem, solver_cfg, relax_integrality)
            logger.info(f"[cvxpy] Solver used: {solver_used}; status={problem.status}")
        except Exception as err:
            logger.warning(f"CVXPY solve failed: {err}")
            return None

        if x.value is None:
            return {
                "status": problem.status,
                "x": None,
                "z": None,
                "t": None,
                "f": None,
                "T": None,
            }

        result = {
            "status": problem.status,
            "x": np.array(x.value).reshape(-1),
            "z": np.array(z.value).reshape(-1) if z.value is not None else None,
            "t": np.array(t.value).reshape(-1) if t.value is not None else None,
            "f": np.array(f.value).reshape(-1) if f.value is not None else None,
            "T": float(T.value) if T.value is not None else None,
        }
        if Y is not None and Y.value is not None:
            result["Y"] = np.array(Y.value)
        return result

    def _round_and_repair_assignment(self, x_relaxed: np.ndarray, A_max: float, threshold: float = 0.5) -> np.ndarray:
        x = (np.array(x_relaxed).reshape(-1) >= threshold).astype(int)
        area_used = float(np.sum(self.a * (1 - x)))

        if area_used > A_max:
            hw_idx = [i for i in range(self.n_nodes) if x[i] == 0]
            hw_idx.sort(key=lambda i: ((self.s[i] - self.h[i]) / max(self.a[i], 1e-9), self.s[i] - self.h[i]))
            for i in hw_idx:
                x[i] = 1
                area_used -= float(self.a[i])
                if area_used <= A_max + 1e-9:
                    break

        if area_used < A_max:
            sw_idx = [i for i in range(self.n_nodes) if x[i] == 1]
            sw_idx.sort(
                key=lambda i: ((self.s[i] - self.h[i]) / max(self.a[i], 1e-9), self.s[i] - self.h[i]),
                reverse=True,
            )
            for i in sw_idx:
                benefit = float(self.s[i] - self.h[i])
                if benefit <= 0:
                    continue
                if area_used + float(self.a[i]) <= A_max + 1e-9:
                    x[i] = 0
                    area_used += float(self.a[i])

        return x

    def _greedy_assignment(self, A_max: float) -> np.ndarray:
        x = np.ones(self.n_nodes, dtype=int)  # start with all software
        area_used = 0.0
        candidates = list(range(self.n_nodes))
        candidates.sort(
            key=lambda i: ((self.s[i] - self.h[i]) / max(self.a[i], 1e-9), self.s[i] - self.h[i]),
            reverse=True,
        )
        for i in candidates:
            if self.s[i] <= self.h[i]:
                continue
            area_i = float(self.a[i])
            if area_used + area_i <= A_max + 1e-9:
                x[i] = 0
                area_used += area_i
        return x

    def _finalize_solution(
        self,
        x_values: np.ndarray,
        t_values: np.ndarray,
        f_values: np.ndarray,
        z_values: np.ndarray,
        makespan_value: float,
        A_max: float,
        status: str,
    ) -> Dict[str, Any]:
        x_arr = np.array(x_values, dtype=float).reshape(-1)
        self.x_sol = np.rint(np.clip(x_arr, 0, 1)).astype(int)

        if t_values is None:
            self.t_sol = np.zeros(self.n_nodes, dtype=float)
        else:
            self.t_sol = np.array(t_values, dtype=float).reshape(-1)

        if f_values is None:
            exec_times = np.where(self.x_sol == 0, self.h, self.s)
            self.f_sol = self.t_sol + exec_times
        else:
            self.f_sol = np.array(f_values, dtype=float).reshape(-1)

        if z_values is None:
            self.z_sol = np.array(
                [int(self.x_sol[self.node_to_index[u]] != self.x_sol[self.node_to_index[v]]) for (u, v) in self.edge_list],
                dtype=int,
            )
        else:
            z_arr = np.array(z_values, dtype=float).reshape(-1)
            self.z_sol = np.rint(np.clip(z_arr, 0, 1)).astype(int)

        self.T_sol = float(makespan_value) if makespan_value is not None else float(np.max(self.f_sol))
        self.Y_sol = None

        hw_nodes = [self.node_list[i] for i in range(self.n_nodes) if self.x_sol[i] == 0]
        sw_nodes = [self.node_list[i] for i in range(self.n_nodes) if self.x_sol[i] == 1]
        hw_sequence = sorted(hw_nodes, key=lambda node: self.t_sol[self.node_to_index[node]])
        sw_sequence = sorted(sw_nodes, key=lambda node: self.t_sol[self.node_to_index[node]])

        start_times_dict = {self.node_list[i]: float(self.t_sol[i]) for i in range(self.n_nodes)}
        finish_times_dict = {self.node_list[i]: float(self.f_sol[i]) for i in range(self.n_nodes)}
        total_hw_area = float(np.sum(self.a * (1 - self.x_sol)))

        solution = {
            "status": status,
            "makespan": self.T_sol,
            "hardware_nodes": hw_nodes,
            "software_nodes": sw_nodes,
            "hardware_sequence": hw_sequence,
            "software_sequence": sw_sequence,
            "start_times": start_times_dict,
            "finish_times": finish_times_dict,
            "total_hardware_area": total_hw_area,
            "area_constraint": A_max,
        }

        logger.info(f"Optimization finished (status={status}) makespan={self.T_sol:.4f}")
        logger.info(f"Hardware nodes: {len(hw_nodes)} | Software nodes: {len(sw_nodes)}")
        logger.info(f"Total hardware area used: {total_hw_area:.4f} / {A_max:.4f}")
        return solution

    def _solution_from_assignment(self, x_assignment: np.ndarray, A_max: float, status: str) -> Dict[str, Any]:
        x_assignment = np.array(x_assignment, dtype=int).reshape(-1)
        graph_nodes = list(self.graph.nodes())
        graph_order_assignment = [int(x_assignment[self.node_to_index[node]]) for node in graph_nodes]

        try:
            makespan, start_times_graph = compute_dag_makespan(self.graph, graph_order_assignment)
            t_values = np.array([start_times_graph[node] for node in self.node_list], dtype=float)
        except Exception as err:
            logger.warning(f"Failed to compute LP schedule for rounded assignment: {err}")
            t_values = np.zeros(self.n_nodes, dtype=float)
            exec_times = np.where(x_assignment == 0, self.h, self.s)
            makespan = float(np.max(exec_times))

        exec_times = np.where(x_assignment == 0, self.h, self.s)
        f_values = t_values + exec_times
        z_values = np.array(
            [int(x_assignment[self.node_to_index[u]] != x_assignment[self.node_to_index[v]]) for (u, v) in self.edge_list],
            dtype=int,
        )
        return self._finalize_solution(
            x_values=x_assignment,
            t_values=t_values,
            f_values=f_values,
            z_values=z_values,
            makespan_value=makespan,
            A_max=A_max,
            status=status,
        )
    
    def _compute_hierarchical_layout(self):
        """Compute hierarchical layout for DAG visualization"""
        if self.graph is None:
            raise ValueError("No graph available")
        
        # Find source nodes (no incoming edges)
        source_nodes = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
        
        # If no source nodes, handle as a special case
        if not source_nodes:
            source_nodes = [min(self.graph.nodes())]  # fallback to first node
        
        # Compute levels using longest path from any source
        levels = {}
        
        # Initialize source nodes at level 0
        for node in source_nodes:
            levels[node] = 0
        
        # Use topological sort to process nodes in dependency order
        try:
            topo_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # Fallback if graph has cycles
            topo_order = list(self.graph.nodes())
        
        # Assign levels based on maximum level of predecessors + 1
        for node in topo_order:
            if node not in levels:  # Not a source node
                predecessors = list(self.graph.predecessors(node))
                if predecessors:
                    # Level is max level of all predecessors + 1
                    pred_levels = [levels.get(pred, 0) for pred in predecessors]
                    levels[node] = max(pred_levels) + 1
                else:
                    levels[node] = 0  # Isolated node
        
        # Group nodes by level (vertical lines)
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Create positions
        pos = {}
        max_nodes_in_level = max(len(nodes) for nodes in level_groups.values()) if level_groups else 1
        
        for level, nodes in level_groups.items():
            x_coord = level * 3  # Horizontal spacing between levels
            n_nodes_in_level = len(nodes)
            
            # Center nodes vertically in each level
            if n_nodes_in_level == 1:
                y_coords = [0]
            else:
                # Spread nodes vertically, centered around y=0
                y_range = max(2, n_nodes_in_level - 1)
                y_coords = np.linspace(-y_range, y_range, n_nodes_in_level)
            
            # Sort nodes by ID for consistent positioning
            sorted_nodes = sorted(nodes)
            for i, node in enumerate(sorted_nodes):
                pos[node] = (x_coord, y_coords[i])
        
        return pos
    
    def display_graph(self, title: str = "Task Graph with Attributes"):
        """Display the directed graph with all attributes and edge weights"""
        if self.graph is None:
            raise ValueError("No graph to display")
        
        fig = plt.figure(figsize=(20, 15))
        
        # Create hierarchical layout
        if not hasattr(self, '_fixed_pos') or self._fixed_pos is None:
            self._fixed_pos = self._compute_hierarchical_layout()
        pos = self._fixed_pos
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                              arrows=True, arrowsize=25, arrowstyle='->', 
                              width=2, alpha=0.7, connectionstyle="arc3,rad=0.1")
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                              node_size=2000, alpha=0.8)
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=20, font_weight='bold')
        
        # Add node attribute labels
        node_labels = {}
        for node in self.graph.nodes():
            hw_time = self.graph.nodes[node]['hardware_time']
            sw_time = self.graph.nodes[node]['software_time'] 
            area = self.graph.nodes[node]['area_cost']
            node_labels[node] = f"HW:{hw_time:.1f}\nSW:{sw_time:.1f}\nArea:{area:.1f}"
        
        # Position labels below nodes
        label_pos = {node: (pos[node][0], pos[node][1] - 0.4) for node in pos}
        
        for node, label in node_labels.items():
            plt.text(label_pos[node][0], label_pos[node][1], label, 
                    horizontalalignment='center', fontsize=20,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add edge weight labels
        edge_labels = {}
        for edge in self.graph.edges():
            comm_cost = self.graph.edges[edge]['communication_cost']
            edge_labels[edge] = f"{comm_cost:.1f}"
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=16)
        
        plt.title(title, fontsize=25, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        fig.savefig('data/test_example.png', bbox_inches='tight')
    
    def display_solution(self, solution: Dict = None):
        """
        Display the result partitions with execution sequence
        Hardware nodes in red, software nodes in blue
        """
        if self.graph is None:
            raise ValueError("No graph to display")
        
        if self.x_sol is None:
            raise ValueError("No solution available. Solve optimization first.")
        
        if solution is None:
            # Create basic solution info
            hw_nodes = [i for i in range(self.n_nodes) if self.x_sol[i] == 0]
            sw_nodes = [i for i in range(self.n_nodes) if self.x_sol[i] == 1]
            solution = {
                'hardware_nodes': hw_nodes,
                'software_nodes': sw_nodes,
                'makespan': self.T_sol
            }
        
        fig = plt.figure(figsize=(20, 15))
        
        # Use the same fixed layout as the original graph
        if not hasattr(self, '_fixed_pos') or self._fixed_pos is None:
            self._fixed_pos = self._compute_hierarchical_layout()
        pos = self._fixed_pos
        
        # Draw edges with communication costs
        edge_colors = []
        edge_widths = []
        for edge in self.graph.edges():
            edge_idx = self.edge_list.index(edge)
            if self.z_sol[edge_idx] == 1:  # Inter-partition communication
                edge_colors.append('red')
                edge_widths.append(3)
            else:
                edge_colors.append('gray')
                edge_widths.append(2)
        
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, 
                              arrows=True, arrowsize=25, arrowstyle='->', 
                              width=edge_widths, alpha=0.8, connectionstyle="arc3,rad=0.1")
        
        # Draw nodes with partition colors
        node_colors = []
        for node in self.graph.nodes():
            node_idx = self.node_to_index[node]
            if self.x_sol[node_idx] == 0:  # Hardware
                node_colors.append('red')
            else:  # Software
                node_colors.append('blue')
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=2500, alpha=0.8)
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=20, 
                               font_weight='bold', font_color='white')
        
        # Add execution time information
        for node in self.graph.nodes():
            node_idx = self.node_to_index[node]
            x_pos, y_pos = pos[node]
            start_time = self.t_sol[node_idx]
            finish_time = self.f_sol[node_idx]
            partition = "HW" if self.x_sol[node_idx] == 0 else "SW"
            
            label = f"{partition}\nStart: {start_time:.1f}\nEnd: {finish_time:.1f}"
            
            plt.text(x_pos, y_pos - 0.4, label,
                    horizontalalignment='center', fontsize=20,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Add edge communication cost labels
        edge_labels = {}
        for edge in self.graph.edges():
            edge_idx = self.edge_list.index(edge)
            comm_cost = self.c[edge_idx]
            active = "✓" if self.z_sol[edge_idx] == 1 else ""
            edge_labels[edge] = f"{comm_cost:.1f}{active}"
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=20)
        
        # Create legend
        hw_patch = mpatches.Patch(color='red', label='Hardware Nodes')
        sw_patch = mpatches.Patch(color='blue', label='Software Nodes')
        comm_patch = mpatches.Patch(color='red', label='Inter-partition Communication')
        no_comm_patch = mpatches.Patch(color='gray', label='Intra-partition Communication')
        
        plt.legend(handles=[hw_patch, sw_patch, comm_patch, no_comm_patch], 
                  loc='upper right', bbox_to_anchor=(1.15, 1), markerscale=3, fontsize=20)
        
        # Add execution sequence information
        hw_seq = [node for node in self.graph.nodes() if self.x_sol[self.node_to_index[node]] == 0]
        sw_seq = [node for node in self.graph.nodes() if self.x_sol[self.node_to_index[node]] == 1]
        hw_seq.sort(key=lambda node: self.t_sol[self.node_to_index[node]])
        sw_seq.sort(key=lambda node: self.t_sol[self.node_to_index[node]])
        
        info_text = f"Makespan: {solution['makespan']:.2f}\n"
        info_text += f"Hardware Execution (Parallel): {hw_seq}\n"
        info_text += f"Software Execution (Sequential): {sw_seq}\n"
        info_text += f"Total Hardware Area: {np.sum(self.a * (1 - self.x_sol)):.2f}"
        
        plt.figtext(0.7, 0.2, info_text, fontsize=20, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        plt.title("Hardware-Software Partitioning Solution", fontsize=25, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        fig.savefig('data/test_result.png', bbox_inches='tight')
