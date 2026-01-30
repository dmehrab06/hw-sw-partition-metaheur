"""
Hardware-Software Partitioning Optimization Solver
Implements the incidence matrix formulation for DAG partitioning
"""

import os
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils.logging_utils import LogManager
from utils.partition_utils import ScheduleConstPartitionSolver    # toggle utils.partition_utils / utils.partition_utils_new
from utils.cuopt_utils import CuOptScheduleConstPartitionSolver
from utils.scheduler_utils import compute_dag_execution_time
from utils.parser_utils import parse_arguments

from task_graph_generation import TaskGraphDataset, generate_random_dag, networkx_to_pytorch, create_data_lists
import networkx as nx

import torch

from csched_env import CSchedEnv, EnvState, NOT_READY, READY, IN_PROGRESS, COMPLETE
import pickle


def solve_dataset(dataset):
    solver = ScheduleConstPartitionSolver()
    partition_t_list   = []
    start_times_t_list = []
    end_times_t_list   = []
    makespan_t_list    = []
    for k in range(len(dataset)):
        print("Presolving testset instance {}".format(k))
        graph, adj_matrices, node_features, edge_features, hw_area_limit = dataset[k]

        software_costs = node_features[0,:]
        hardware_areas = node_features[1,:]                  #  node_features = torch.cat([sw_computation_cost, hw_area_cost], dim=0)
        hardware_costs = software_costs*0.5
        communication_costs = edge_features[0,:]             # only one edge feature

        node_list = list(graph.nodes())

        solver.load_networkx_graph_with_torch_feats(graph, hardware_areas, hardware_costs, software_costs, communication_costs)
        solution = solver.solve_optimization(A_max=hw_area_limit)

        partition = {}
        for n in solution['hardware_nodes']:
            partition[n] = 1
        for n in solution['software_nodes']:
            partition[n] = 0

        # Compute execution time
        result = compute_dag_execution_time(graph, partition, verbose=False, full_dict = True)
        start_times  = result['start_times']
        end_times = result['finish_times']
        hw_nodes = result['hardware_nodes']
        sw_nodes = result['software_nodes']
        makespan = result['makespan']

        start_times_list = [start_times[node] for node in node_list]
        end_times_list   = [end_times[node] for node in node_list]
        partition_list   = [partition[node] for node in node_list]

        partition_t   = torch.Tensor(partition_list)
        start_times_t = torch.Tensor(start_times_list)
        end_times_t   = torch.Tensor(end_times_list)
        makespan_t   = torch.Tensor([makespan])

        proc_times_t = partition_t*hardware_costs + (1-partition_t)*software_costs

        partition_t_list.append(partition_t)
        start_times_t_list.append(start_times_t)
        end_times_t_list.append(end_times_t)
        makespan_t_list.append(makespan_t)



    partition_batch = torch.stack(partition_t_list)
    start_times_batch = torch.stack(start_times_t_list)
    end_times_batch = torch.stack(end_times_t_list)
    makespan_batch = torch.stack(makespan_t_list)

    opt_sols_batch = {"partitions":  partition_batch,
                      "start_times": start_times_batch,
                      "end_times":   end_times_batch,
                      "makespans":   makespan_batch
    }

    return opt_sols_batch



# Returns an opt_sols_batch of size 1
def solve_dataset_instance(dataset, idx):
    solver = ScheduleConstPartitionSolver()

    k = idx

    print("Presolving testset instance {}".format(k))
    graph, adj_matrices, node_features, edge_features, hw_area_limit = dataset[k]

    software_costs = node_features[0,:]
    hardware_areas = node_features[1,:]
    hardware_costs = node_features[2,:]
    communication_costs = edge_features[0,:]

    node_list = list(graph.nodes())


    solver.load_networkx_graph_with_torch_feats(graph, hardware_areas, hardware_costs, software_costs, communication_costs)
    solution = solver.solve_optimization(A_max=hw_area_limit)

    partition = {}
    for n in solution['hardware_nodes']:
        partition[n] = 1
    for n in solution['software_nodes']:
        partition[n] = 0

    # Compute execution time
    result = compute_dag_execution_time(graph, partition, verbose=False, full_dict = True)
    start_times  = result['start_times']
    end_times = result['finish_times']
    hw_nodes = result['hardware_nodes']
    sw_nodes = result['software_nodes']
    makespan = result['makespan']


    start_times_list = [start_times[node] for node in node_list]
    end_times_list   = [end_times[node] for node in node_list]
    partition_list   = [partition[node] for node in node_list]
    #start_times_list = [start_times[i] for i in range(len(start_times))]
    #end_times_list   = [end_times[i] for i in range(len(end_times))]
    #partition_list   = [partition[i] for i in range(len(partition))]

    partition_t   = torch.Tensor(partition_list)
    start_times_t = torch.Tensor(start_times_list)
    end_times_t   = torch.Tensor(end_times_list)
    makespan_t   = torch.Tensor([makespan])

    proc_times_t = partition_t*hardware_costs + (1-partition_t)*software_costs


    partition_t_list = [partition_t]
    start_times_t_list = [start_times_t]
    end_times_t_list = [end_times_t]
    makespan_t_list = [makespan_t]

    partition_batch = torch.stack(partition_t_list)
    start_times_batch = torch.stack(start_times_t_list)
    end_times_batch = torch.stack(end_times_t_list)
    makespan_batch = torch.stack(makespan_t_list)


    opt_sols_batch = {"partitions":  partition_batch,
                      "start_times": start_times_batch,
                      "end_times":   end_times_batch,
                      "makespans":   makespan_batch
    }

    return opt_sols_batch





def update_env_presolve(env, opt_sols_batch):

    partition_batch   = opt_sols_batch["partitions"]
    start_times_batch = opt_sols_batch["start_times"]
    end_times_batch   = opt_sols_batch["end_times"]
    makespan_batch    = opt_sols_batch["makespans"]

    batch_size = len(partition_batch)
    n = partition_batch.shape[1]   # TODO: this maybe should vary per the bsatch sample

    env.op_status_batch   = COMPLETE*torch.ones((batch_size, n),dtype=torch.int32) # COMPLETE
    env.op_resource_batch = partition_batch.type(torch.int32)
    #env.hw_usage_batch =
    env.current_time_batch = makespan_batch
    env.makespan_batch     = makespan_batch
    #env.hw_area_remaining_batch
    #env.n_pred_remaining_batch = torch.zeros((batch_size, n),dtype=torch.int32)
    env.op_start_time_batch = start_times_batch
    env.op_end_time_batch   = end_times_batch





def main():

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)


    # Generate a dataset of instances
    graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits = create_data_lists(
        num_samples=10,
        min_nodes=10,
        max_nodes=10,
        edge_probability=0.3
    )
    dataset = TaskGraphDataset(graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits)

    # Get MIP solutions for each instance
    opt_sols_batch = solve_dataset(dataset)

    env_paras = {
        "batch_size": 3,
        "device": "cpu",
        "timestep_mode": "next_complete",
        "timestep_trigger": "every",
        "prevent_all_HW": False
    }
    env = CSchedEnv(dataset, env_paras)

    # Update solutions in the environment's batch of instances
    update_env_presolve(env, opt_sols_batch)

    # Argument to render is the instance's index
    env.render(0)
    env.render(1)
    env.render(2)
    # etc





if __name__ == "__main__":
    main()
