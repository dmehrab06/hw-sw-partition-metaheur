import os
import random
import numpy as np
import warnings
import torch
import pickle
warnings.filterwarnings('ignore')

from task_graph_generation import TaskGraphDataset, generate_random_dag, networkx_to_pytorch, create_data_lists
import networkx as nx


from utils.scheduler_utils import compute_dag_execution_time
from utils.parser_utils import parse_arguments

from utils.partition_utils     import ScheduleConstPartitionSolver
from utils.partition_utils_new import ScheduleConstPartitionSolver as ScheduleConstPartitionSolver_new



def solve_dataset_with_solver(dataset, SolverClass):
    solver = SolverClass  #ScheduleConstPartitionSolver()
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





def main():

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    num_samples = 100

    graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits = create_data_lists(
        num_samples=num_samples,
        min_nodes=10,
        max_nodes=10,
        edge_probability=0.3
    )
    dataset = TaskGraphDataset(graphs, adj_matrices, node_features_list, edge_features_list, hw_area_limits)


    opt_sols_batch     = solve_dataset_with_solver(dataset, ScheduleConstPartitionSolver())
    opt_sols_batch_new = solve_dataset_with_solver(dataset, ScheduleConstPartitionSolver_new())


    stimes     = opt_sols_batch['start_times']
    stimes_new = opt_sols_batch_new['start_times']


    print("stimes (schedule from original solver)")
    print( stimes )

    print("stimes_new (schedule from revised solver)")
    print( stimes_new )

    print("Maximum absolute difference over {} sample problems:".format(num_samples))
    print( (stimes_new - stimes).abs().max() )





if __name__ == "__main__":
    main()
