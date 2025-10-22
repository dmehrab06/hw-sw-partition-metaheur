from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
import cvxpy as cp
import os, sys
import time

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/execution_time_evaluator.log")

logger = LogManager.get_logger(__name__)

def compute_dag_execution_time(graph: nx.DiGraph, partition_assignment: Dict[int, str], 
                              verbose: bool = False) -> Dict:
    """
    Compute overall execution time for a DAG with given hardware/software partitions
    
    Args:
        graph: NetworkX directed graph with node attributes (hardware_time, software_time, area_cost)
               and edge attributes (communication_cost)
        partition_assignment: Dict mapping node_id -> 'hardware' or 'software'
        verbose: If True, print detailed execution schedule
        
    Returns:
        Dict containing execution details including makespan, start times, finish times
    """
    if not isinstance(graph, nx.DiGraph):
        raise ValueError("Graph must be a NetworkX DiGraph")
    
    # Validate that graph is a DAG
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be a directed acyclic graph (DAG)")
    
    # Get all node IDs from the graph
    graph_nodes = set(graph.nodes())
    
    # Validate partition assignment
    for node in graph_nodes:
        if node not in partition_assignment:
            raise ValueError(f"Partition not specified for node {node}")
        if partition_assignment[node] not in ['hardware', 'software']:
            raise ValueError(f"Invalid partition '{partition_assignment[node]}' for node {node}")
    
    # Check for extra nodes in partition assignment
    partition_nodes = set(partition_assignment.keys())
    if partition_nodes != graph_nodes:
        extra_nodes = partition_nodes - graph_nodes
        missing_nodes = graph_nodes - partition_nodes
        if extra_nodes:
            print(f"Warning: Partition assignment contains nodes not in graph: {extra_nodes}")
        if missing_nodes:
            raise ValueError(f"Missing partition assignments for nodes: {missing_nodes}")
    
    # Validate required node attributes
    required_node_attrs = ['hardware_time', 'software_time', 'area_cost']
    for node in graph.nodes():
        for attr in required_node_attrs:
            if attr not in graph.nodes[node]:
                raise ValueError(f"Node {node} missing required attribute '{attr}'")
    
    # Validate required edge attributes
    for u, v in graph.edges():
        if 'communication_cost' not in graph.edges[u, v]:
            raise ValueError(f"Edge ({u}, {v}) missing required attribute 'communication_cost'")
    
    # Initialize data structures
    start_times = {}
    finish_times = {}
    hw_nodes = [node for node in graph_nodes if partition_assignment[node] == 'hardware']
    sw_nodes = [node for node in graph_nodes if partition_assignment[node] == 'software']
    
    # Track which nodes have completed execution
    completed_nodes = set()
    
    # Get node and edge attributes
    node_hw_times = {node: graph.nodes[node]['hardware_time'] for node in graph.nodes()}
    node_sw_times = {node: graph.nodes[node]['software_time'] for node in graph.nodes()}
    edge_comm_costs = {(u, v): graph.edges[u, v]['communication_cost'] for u, v in graph.edges()}
    
    def get_execution_time(node):
        """Get execution time for a node based on its partition"""
        if partition_assignment[node] == 'hardware':
            return node_hw_times[node]
        else:
            return node_sw_times[node]
    
    def get_earliest_start_time(node):
        """Calculate earliest start time considering dependencies and communication"""
        earliest_start = 0
        
        for predecessor in graph.predecessors(node):
            if predecessor not in finish_times:
                return None  # Predecessor not yet completed
            
            pred_finish = finish_times[predecessor]
            
            # Add communication delay if nodes are in different partitions
            if partition_assignment[predecessor] != partition_assignment[node]:
                comm_delay = edge_comm_costs.get((predecessor, node), 0)
                pred_finish += comm_delay
            
            earliest_start = max(earliest_start, pred_finish)
        
        return earliest_start
    
    # Get topological order for processing
    topo_order = list(nx.topological_sort(graph))
    
    # Initialize with source nodes (nodes with no predecessors)
    ready_nodes = [node for node in topo_order if graph.in_degree(node) == 0]
    
    # Track software execution (sequential)
    sw_execution_queue = []
    sw_current_time = 0
    
    # Track hardware nodes ready to execute
    hw_ready_nodes = []
    
    current_time = 0
    iteration = 0
    max_iterations = len(graph.nodes()) * 10  # Prevent infinite loops
    
    if verbose:
        logger.info(f"Starting execution simulation...")
        logger.info(f"Hardware nodes: {hw_nodes}")
        logger.info(f"Software nodes: {sw_nodes}")
        logger.info(f"Initial ready nodes: {ready_nodes}")
    
    while (ready_nodes or hw_ready_nodes or sw_execution_queue) and iteration < max_iterations:
        iteration += 1
        
        if verbose:
            logger.info(f"\\n--- Iteration {iteration}, Time {current_time:.2f} ---")
            logger.info(f"Ready nodes: {ready_nodes}")
            logger.info(f"HW ready queue: {[n for n, _ in hw_ready_nodes]}")
            logger.info(f"SW execution queue: {[n for n, _ in sw_execution_queue]}")
            logger.info(f"Completed nodes: {sorted(completed_nodes)}")
        
        # Process ready nodes and categorize them
        new_hw_ready = []
        new_sw_ready = []
        
        for node in ready_nodes:
            earliest_start = get_earliest_start_time(node)
            if earliest_start is not None:
                if partition_assignment[node] == 'hardware':
                    new_hw_ready.append((node, earliest_start))
                else:
                    new_sw_ready.append((node, earliest_start))
        
        ready_nodes = []
        
        # Add new hardware nodes to ready queue
        hw_ready_nodes.extend(new_hw_ready)
        
        # Start hardware nodes that can execute now
        hw_to_start = [(node, ready_time) for node, ready_time in hw_ready_nodes 
                       if ready_time <= current_time]
        
        for node, ready_time in hw_to_start:
            actual_start = max(current_time, ready_time)
            start_times[node] = actual_start
            finish_times[node] = actual_start + get_execution_time(node)
            completed_nodes.add(node)
            hw_ready_nodes.remove((node, ready_time))
            
            if verbose:
                logger.info(f"  Started HW node {node}: start={start_times[node]:.2f}, "
                     f"finish={finish_times[node]:.2f}, duration={get_execution_time(node):.2f}")
        
        # Add software nodes to execution queue
        for node, earliest_start in new_sw_ready:
            sw_execution_queue.append((node, earliest_start))
        
        # Execute next software node if possible
        if sw_execution_queue:
            node, earliest_start = sw_execution_queue[0]
            if earliest_start <= max(sw_current_time, current_time):
                sw_execution_queue.pop(0)
                actual_start = max(sw_current_time, earliest_start, current_time)
                start_times[node] = actual_start
                finish_times[node] = actual_start + get_execution_time(node)
                sw_current_time = finish_times[node]
                completed_nodes.add(node)
                
                if verbose:
                    logger.info(f"  Started SW node {node}: start={start_times[node]:.2f}, "
                         f"finish={finish_times[node]:.2f}, duration={get_execution_time(node):.2f}")
        
        # Find newly ready nodes
        for node in topo_order:
            if (node not in completed_nodes and 
                node not in ready_nodes and
                node not in [n for n, _ in hw_ready_nodes] and
                node not in [n for n, _ in sw_execution_queue]):
                
                # Check if all predecessors are completed
                predecessors = list(graph.predecessors(node))
                if all(pred in completed_nodes for pred in predecessors):
                    ready_nodes.append(node)
                    if verbose:
                        logger.info(f"  Node {node} became ready (all predecessors completed)")
        
        # Advance time to next event
        next_events = []
        
        # Hardware nodes that will complete
        hw_completions = [finish_times[node] for node in completed_nodes 
                         if node in finish_times and finish_times[node] > current_time]
        next_events.extend(hw_completions)
        
        # Hardware nodes that will become ready
        hw_ready_times = [ready_time for node, ready_time in hw_ready_nodes 
                         if ready_time > current_time]
        next_events.extend(hw_ready_times)
        
        # Next software node ready time
        if sw_execution_queue:
            node, ready_time = sw_execution_queue[0]
            next_sw_time = max(ready_time, sw_current_time)
            if next_sw_time > current_time:
                next_events.append(next_sw_time)
        
        # Advance to next event or make small increment
        if next_events:
            current_time = min(next_events)
        elif ready_nodes or hw_ready_nodes or sw_execution_queue:
            current_time += 0.1  # Small increment to avoid infinite loop
        else:
            break  # No more events
    
    if iteration >= max_iterations:
        logger.warning(f"Maximum iterations ({max_iterations}) reached. Possible infinite loop.")
    
    # Calculate results
    if not finish_times:
        return {'makespan': 0, 'start_times': {}, 'finish_times': {}, 'error': 'No nodes completed'}
    
    makespan = max(finish_times.values())
    
    # Calculate partition-specific metrics
    hw_finish_times = [finish_times[node] for node in hw_nodes if node in finish_times]
    sw_finish_times = [finish_times[node] for node in sw_nodes if node in finish_times]
    
    hw_makespan = max(hw_finish_times) if hw_finish_times else 0
    sw_makespan = max(sw_finish_times) if sw_finish_times else 0
    
    # Calculate communication overhead
    total_comm_delay = 0
    active_comm_edges = []
    
    for u, v in graph.edges():
        if partition_assignment[u] != partition_assignment[v]:
            comm_delay = edge_comm_costs[(u, v)]
            total_comm_delay += comm_delay
            active_comm_edges.append((u, v, comm_delay))
    
    result = {
        'makespan': makespan,
        'start_times': start_times,
        'finish_times': finish_times,
        'hardware_nodes': hw_nodes,
        'software_nodes': sw_nodes,
        'hardware_makespan': hw_makespan,
        'software_makespan': sw_makespan,
        'total_communication_delay': total_comm_delay,
        'active_communication_edges': active_comm_edges,
        'execution_summary': {
            'total_nodes': len(graph.nodes()),
            'completed_nodes': len(completed_nodes),
            'hardware_execution_time': hw_makespan,
            'software_execution_time': sw_makespan,
            'communication_overhead': total_comm_delay,
            'total_makespan': makespan
        }
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total Makespan: {makespan:.2f}")
        print(f"Hardware Makespan: {hw_makespan:.2f} (nodes: {hw_nodes})")
        print(f"Software Makespan: {sw_makespan:.2f} (nodes: {sw_nodes})")
        print(f"Communication Overhead: {total_comm_delay:.2f}")
        print(f"Active Communication Edges: {len(active_comm_edges)}")
        print(f"Nodes Completed: {len(completed_nodes)}/{len(graph.nodes())}")
        
        if active_comm_edges:
            print(f"\nInter-partition communications:")
            for u, v, delay in active_comm_edges:
                print(f"  ({u} -> {v}): {delay:.2f}")
    
    return result

def compute_dag_makespan(graph: nx.DiGraph, partition_assignment: List[int]) -> Tuple[float, Dict[int, float]]:
    """
    Compute optimal makespan for DAG execution with hardware/software partitioning using linear programming.
    
    This function formulates the scheduling problem as an LP optimization to find optimal start times
    for each node while respecting:
    - Dependency constraints (precedence)
    - Hardware parallel execution
    - Software sequential execution  
    - Communication delays between partitions
    
    Variables in LP formulation:
    - start_time[i]: Start time for node i
    - makespan: Total execution time (objective to minimize)
    
    Constraints:
    1. Precedence: start_time[j] >= finish_time[i] + comm_delay[i][j] for edge (i,j)
    2. Software sequencing: Software nodes execute sequentially in topological order
    3. Non-negativity: start_time[i] >= 0
    4. Makespan definition: makespan >= finish_time[i] for all i
    
    Args:
        graph (nx.DiGraph): Directed acyclic graph with node attributes:
                           'hardware_time', 'software_time' and edge attribute 'communication_cost'
        partition_assignment (List[int]): Binary assignment (0=hardware, 1=software)
        
    Returns:
        Tuple[float, Dict[int, float]]: (optimal_makespan, start_times_dict)
        
    Raises:
        ValueError: If graph attributes are missing or partition assignment is invalid
        RuntimeError: If LP optimization fails
    """
    
    # Extract timing information from graph
    hw_times, sw_times, comm_delays, node_to_index = _extract_timing_from_graph(graph)
    n_nodes = len(hw_times)
    assignment = np.array(partition_assignment)
    
    # Validate inputs
    if len(assignment) != n_nodes:
        raise ValueError(f"Partition assignment length ({len(assignment)}) doesn't match number of nodes ({n_nodes})")
    
    logger.debug(f"Computing optimal makespan for DAG with {n_nodes} nodes")
    logger.debug(f"Hardware nodes: {np.sum(assignment == 0)}, Software nodes: {np.sum(assignment == 1)}")
    
    # Step 1: Determine execution times and communication costs based on partition
    exec_times, comm_costs = _compute_timing_parameters(assignment, hw_times, sw_times, comm_delays)
    
    # Step 2: Formulate CVXPY problem
    start_times, makespan, problem = _formulate_cvxpy_problem(graph, assignment, exec_times, comm_costs, node_to_index)
    
    # Step 3: Solve the optimization problem
    logger.debug("Solving linear programming problem with CVXPY...")
    problem.solve(solver=cp.CLARABEL, verbose=False)
    
    if problem.status not in ["infeasible", "unbounded"]:
        if problem.status != "optimal":
            logger.warning(f"Solver status: {problem.status}")
    else:
        logger.error(f"Optimization failed with status: {problem.status}")
        raise RuntimeError(f"LP optimization failed with status: {problem.status}")
    
    # Step 4: Extract solution
    optimal_makespan = makespan.value
    start_times_dict = {node: start_times[node_to_index[node]].value for node in graph.nodes()}
    
    logger.debug(f"Optimal makespan computed: {optimal_makespan:.4f}")
    
    # Validate solution
    # _validate_solution(graph, start_times_dict, assignment, exec_times, comm_costs, optimal_makespan, node_to_index)
    
    return optimal_makespan, start_times_dict


def _extract_timing_from_graph(graph: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Extract timing matrices and vectors from graph node and edge attributes.
    
    Args:
        graph (nx.DiGraph): Graph with timing attributes
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]: (hw_times, sw_times, comm_delays, node_to_index)
    """
    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    
    # Initialize arrays
    hw_times = np.zeros(n_nodes)
    sw_times = np.zeros(n_nodes)
    comm_delays = np.zeros((n_nodes, n_nodes))
    
    # Extract node timing attributes
    for idx, node in enumerate(nodes):
        node_data = graph.nodes[node]
        
        if 'hardware_time' in node_data:
            hw_times[idx] = node_data['hardware_time']
        else:
            logger.warning(f"Node {node} missing 'hardware_time' attribute, using 0")
            hw_times[idx] = 0.0
        
        if 'software_time' in node_data:
            sw_times[idx] = node_data['software_time']
        else:
            logger.warning(f"Node {node} missing 'software_time' attribute, using 0")
            sw_times[idx] = 0.0
    
    # Extract edge communication costs
    for u, v, edge_data in graph.edges(data=True):
        u_idx = node_to_index[u]
        v_idx = node_to_index[v]
        
        if 'communication_cost' in edge_data:
            comm_delays[u_idx][v_idx] = edge_data['communication_cost']
        else:
            logger.warning(f"Edge ({u}, {v}) missing 'communication_cost' attribute, using 0")
            comm_delays[u_idx][v_idx] = 0.0
    
    logger.debug(f"Extracted {n_nodes} node timings and {graph.number_of_edges()} edge costs")
    logger.debug(f"Hardware times range: [{hw_times.min():.3f}, {hw_times.max():.3f}]")
    logger.debug(f"Software times range: [{sw_times.min():.3f}, {sw_times.max():.3f}]")
    
    return hw_times, sw_times, comm_delays, node_to_index


def _compute_timing_parameters(assignment: np.ndarray, hw_times: np.ndarray, 
                              sw_times: np.ndarray, comm_delays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute execution times and communication costs based on partition assignment.
    
    Args:
        assignment (np.ndarray): Binary partition assignment
        hw_times (np.ndarray): Hardware execution times
        sw_times (np.ndarray): Software execution times
        comm_delays (np.ndarray): Base communication delay matrix
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (execution_times, communication_costs)
    """
    n_nodes = len(assignment)
    
    # Determine execution times based on partition assignment
    exec_times = np.where(assignment == 0, hw_times, sw_times)
    
    # Compute communication costs - only between different partitions
    comm_costs = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if assignment[i] != assignment[j]:  # Different partitions
                comm_costs[i][j] = comm_delays[i][j]
            # Same partition: communication cost remains 0
    
    logger.debug(f"Execution times range: [{exec_times.min():.3f}, {exec_times.max():.3f}]")
    logger.debug(f"Non-zero communication costs: {np.count_nonzero(comm_costs)}")
    
    return exec_times, comm_costs


def _formulate_cvxpy_problem(graph: nx.DiGraph, assignment: np.ndarray, exec_times: np.ndarray,
                           comm_costs: np.ndarray, node_to_index: Dict) -> Tuple[cp.Variable, cp.Variable, cp.Problem]:
    """
    Formulate the linear programming problem using CVXPY for optimal scheduling.
    
    Mathematical Formulation:
    
    Variables:
    - s[i]: Start time for node i, i ∈ {0, 1, ..., n-1}
    - M: Makespan
    
    Objective:
    minimize M
    
    Constraints:
    1. Precedence: s[j] >= s[i] + e[i] + c[i,j] for each edge (i,j) ∈ E
    2. Software sequencing: s[j] >= s[i] + e[i] for consecutive software nodes
    3. Makespan: M >= s[i] + e[i] for all nodes i
    4. Non-negativity: s[i] >= 0, M >= 0
    
    Args:
        graph (nx.DiGraph): Task dependency graph
        assignment (np.ndarray): Partition assignment
        exec_times (np.ndarray): Execution times for each node
        comm_costs (np.ndarray): Communication cost matrix
        node_to_index (Dict): Mapping from node IDs to array indices
        
    Returns:
        Tuple[cp.Variable, cp.Variable, cp.Problem]: (start_times_var, makespan_var, problem)
    """
    n_nodes = len(exec_times)
    
    # Define optimization variables
    start_times = cp.Variable(n_nodes, nonneg=True, name="start_times")
    makespan = cp.Variable(nonneg=True, name="makespan")
    
    # Define objective: minimize makespan
    objective = cp.Minimize(makespan)
    
    # Initialize constraints list
    constraints = []
    
    # Constraint 1: Precedence constraints for DAG edges
    # For each edge (i,j): s[j] >= s[i] + e[i] + c[i,j]
    precedence_count = 0
    for u, v in graph.edges():
        u_idx = node_to_index[u]
        v_idx = node_to_index[v]
        
        constraint = start_times[v_idx] >= start_times[u_idx] + exec_times[u_idx] + comm_costs[u_idx][v_idx]
        constraints.append(constraint)
        precedence_count += 1
    
    # Constraint 2: Software sequential execution
    # For consecutive software nodes: s[j] >= s[i] + e[i]
    software_nodes = [node for node in graph.nodes() if assignment[node_to_index[node]] == 1]
    software_count = 0
    
    if len(software_nodes) > 1:
        # Create subgraph of software nodes to determine execution order
        software_subgraph = graph.subgraph(software_nodes)
        
        if software_subgraph.number_of_edges() > 0:
            # Use topological ordering for dependent software nodes
            try:
                sw_order = list(nx.topological_sort(software_subgraph))
            except nx.NetworkXError:
                # Fallback to node order if subgraph has cycles (shouldn't happen in DAG)
                logger.error("Software subgraph has cycles")
                sw_order = sorted(software_nodes)
        else:
            # No dependencies among software nodes, use arbitrary order
            sw_order = sorted(software_nodes)
        
        # Add sequential constraints for consecutive software nodes
        for i in range(len(sw_order) - 1):
            curr_node = sw_order[i]
            next_node = sw_order[i + 1]
            curr_idx = node_to_index[curr_node]
            next_idx = node_to_index[next_node]
            
            constraint = start_times[next_idx] >= start_times[curr_idx] + exec_times[curr_idx]
            # constraints.append(constraint)
            software_count += 1
    
    # Constraint 3: Makespan definition constraints
    # For each node i: M >= s[i] + e[i]
    makespan_count = 0
    for node in graph.nodes():
        i = node_to_index[node]
        constraint = makespan >= start_times[i] + exec_times[i]
        constraints.append(constraint)
        makespan_count += 1
    
    # Create the optimization problem
    problem = cp.Problem(objective, constraints)
    
    logger.debug(f"CVXPY formulation: {n_nodes + 1} variables")
    logger.debug(f"Constraints: {precedence_count} precedence, {software_count} software sequencing, {makespan_count} makespan")
    logger.debug(f"Total constraints: {len(constraints)}")
    
    return start_times, makespan, problem


def _validate_solution(graph: nx.DiGraph, start_times: Dict[int, float], assignment: np.ndarray,
                      exec_times: np.ndarray, comm_costs: np.ndarray, makespan: float, node_to_index: Dict):
    """
    Validate the computed solution against all constraints.
    
    Args:
        graph (nx.DiGraph): Task dependency graph
        start_times (Dict[int, float]): Computed start times
        assignment (np.ndarray): Partition assignment
        exec_times (np.ndarray): Execution times
        comm_costs (np.ndarray): Communication costs
        makespan (float): Computed makespan
        node_to_index (Dict): Node to index mapping
    """
    tolerance = 1e-6
    
    # Check precedence constraints
    for u, v in graph.edges():
        u_idx = node_to_index[u]
        v_idx = node_to_index[v]
        finish_u = start_times[u] + exec_times[u_idx]
        required_start_v = finish_u + comm_costs[u_idx][v_idx]
        if start_times[v] < required_start_v - tolerance:
            logger.warning(f"Precedence violation: edge ({u},{v})")
    
    # Check software sequencing
    software_nodes = [node for node in graph.nodes() if assignment[node_to_index[node]] == 1]
    software_times = [(start_times[node], node) for node in software_nodes]
    software_times.sort()
    
    for i in range(len(software_times) - 1):
        curr_node = software_times[i][1]
        next_node = software_times[i + 1][1]
        curr_idx = node_to_index[curr_node]
        curr_finish = software_times[i][0] + exec_times[curr_idx]
        next_start = software_times[i + 1][0]
        if next_start < curr_finish - tolerance:
            logger.warning(f"Software sequencing violation between nodes {curr_node} and {next_node}")
    
    # Check makespan constraint
    max_finish = max(start_times[node] + exec_times[node_to_index[node]] for node in graph.nodes())
    if makespan < max_finish - tolerance:
        logger.warning(f"Makespan constraint violation: computed={makespan:.4f}, required={max_finish:.4f}")
    
    logger.info("Solution validation completed")


def get_execution_schedule(graph: nx.DiGraph, start_times: Dict[int, float], 
                          assignment: List[int]) -> Dict[str, List[Tuple[int, float, float]]]:
    """
    Generate a detailed execution schedule from the optimal solution.
    
    Args:
        graph (nx.DiGraph): Task dependency graph
        start_times (Dict[int, float]): Optimal start times
        assignment (List[int]): Partition assignment
        
    Returns:
        Dict with 'hardware' and 'software' keys containing lists of 
        (node_id, start_time, finish_time) tuples
    """
    _, _, _, node_to_index = _extract_timing_from_graph(graph)
    hw_times, sw_times, _, _ = _extract_timing_from_graph(graph)
    exec_times = np.where(np.array(assignment) == 0, hw_times, sw_times)
    
    schedule = {'hardware': [], 'software': []}
    
    for node in graph.nodes():
        idx = node_to_index[node]
        start_time = start_times[node]
        finish_time = start_time + exec_times[idx]
        
        if assignment[idx] == 0:  # Hardware
            schedule['hardware'].append((node, start_time, finish_time))
        else:  # Software
            schedule['software'].append((node, start_time, finish_time))
    
    # Sort by start time
    schedule['hardware'].sort(key=lambda x: x[1])
    schedule['software'].sort(key=lambda x: x[1])
    
    return schedule


if __name__ == '__main__':
    import pickle
    data_file = "makespan-opt-partitions/taskgraph-squeeze_net_tosa_area-0.50_hwscale-0.1_hwvar-0.50_comm-1.00_seed-42_assignment-gl25.pkl"
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    graph_file = "inputs/task_graph_complete/taskgraph-squeeze_net_tosa-instance-config-config_mkspan_default.pkl"
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    graph = graph_data.graph
    nx.set_node_attributes(graph, graph_data.hardware_area, 'area_cost')
    nx.set_node_attributes(graph, graph_data.hardware_costs, 'hardware_time')
    nx.set_node_attributes(graph, graph_data.software_costs, 'software_time')
    nx.set_edge_attributes(graph, graph_data.communication_costs, 'communication_cost')

    partition_assignment = {k: 'hardware' if data[k]==1 else 'software' for k in graph.nodes}
    t_start = time.time()
    sol = compute_dag_execution_time(graph, partition_assignment, verbose=False)
    print(f"Solution: {sol['makespan']}")
    t_end = time.time()
    print(f"Execution time computed in {t_end-t_start} seconds")

    t_start = time.time()
    # the method has been written with software assignment denoted with 1
    assignment = [1-data[k] for k in graph.nodes]
    makespan,_ = compute_dag_makespan(graph, assignment)
    print(f"Solution: {makespan.value}")
    t_end = time.time()
    print(f"Execution time computed in {t_end-t_start} seconds")