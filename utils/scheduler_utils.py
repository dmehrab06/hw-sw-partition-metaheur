from typing import Dict
import networkx as nx
import os, sys

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
    
    # Validate partition assignment
    for node in graph.nodes():
        if node not in partition_assignment:
            raise ValueError(f"Partition not specified for node {node}")
        if partition_assignment[node] not in ['hardware', 'software']:
            raise ValueError(f"Invalid partition '{partition_assignment[node]}' for node {node}")
    
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
    hw_nodes = [node for node, partition in partition_assignment.items() if partition == 'hardware']
    sw_nodes = [node for node, partition in partition_assignment.items() if partition == 'software']
    
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