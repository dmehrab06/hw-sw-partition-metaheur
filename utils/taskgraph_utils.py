import pickle
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_taskgraph(filepath):
    """Load a saved TaskGraph instance from pickle file"""
    try:
        with open(filepath, 'rb') as file:
            task_graph = pickle.load(file)
        logger.info(f"TaskGraph instance loaded from: {filepath}")
        return task_graph
    except Exception as e:
        logger.error(f"Failed to load TaskGraph instance from {filepath}: {e}")
        return None

def find_taskgraph_instance(graph_name, area_constraint, hw_scale_factor, 
                           hw_scale_variance, comm_scale_factor, seed, 
                           taskgraph_dir="taskgraph_instances"):
    """Find TaskGraph instance file based on parameters"""
    
    graph_name = Path(graph_name).stem  # Remove .dot extension if present
    
    filename = (f"taskgraph-{graph_name}_"
               f"area-{area_constraint:.2f}_"
               f"hwscale-{hw_scale_factor:.1f}_"
               f"hwvar-{hw_scale_variance:.2f}_"
               f"comm-{comm_scale_factor:.2f}_"
               f"seed-{seed}_"
               f"instance.pkl")
    
    filepath = os.path.join(taskgraph_dir, filename)
    
    if os.path.exists(filepath):
        return filepath
    else:
        logger.warning(f"TaskGraph instance file not found: {filepath}")
        return None

def get_taskgraph_metadata(task_graph):
    """Extract metadata from TaskGraph instance"""
    metadata = {
        'num_nodes': len(task_graph.graph.nodes()),
        'num_edges': len(task_graph.graph.edges()),
        'total_area': task_graph.total_area,
        'area_constraint': task_graph.area_constraint,
        'software_costs_sum': sum(task_graph.software_costs.values()),
        'hardware_costs_sum': sum(task_graph.hardware_costs.values()),
        'communication_costs_sum': sum(task_graph.communication_costs.values()),
        'nodes': list(task_graph.graph.nodes()),
        'edges': list(task_graph.graph.edges())
    }
    return metadata

def compare_taskgraphs(tg1, tg2):
    """Compare two TaskGraph instances for consistency"""
    checks = {
        'same_nodes': set(tg1.graph.nodes()) == set(tg2.graph.nodes()),
        'same_edges': set(tg1.graph.edges()) == set(tg2.graph.edges()),
        'same_area_constraint': tg1.area_constraint == tg2.area_constraint,
        'same_total_area': abs(tg1.total_area - tg2.total_area) < 1e-6,
        'same_software_costs': tg1.software_costs == tg2.software_costs,
        'same_hardware_costs': tg1.hardware_costs == tg2.hardware_costs,
        'same_communication_costs': tg1.communication_costs == tg2.communication_costs
    }
    
    all_same = all(checks.values())
    
    return all_same, checks