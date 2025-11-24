import pandas as pd
import pickle
import os

from utils.scheduler_utils import compute_dag_execution_time

dirname = "mip-opt-partitions"
opt_files = [f for f in os.listdir(dirname) if f.endswith('.pkl')]

data_to_save = {
    "area-constraint": [],
    "acceleration-factor": [],
    "seed":[],
    "makespan": []
}

for pkl_filename in opt_files:
    params = pkl_filename.split('_')
    area = float(params[3].split('-')[-1])
    hw = float(params[4].split('-')[-1])
    seed = int(params[6].split('-')[-1])

    with open(f"{dirname}/{pkl_filename}", 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    
    graph_pkl_filename = f"inputs/task_graph_complete/taskgraph-squeeze_net_tosa-instance-config-config_mkspan_area_{area:0.1f}_hw_{hw:0.1f}_seed_{seed}.pkl"
    with open(graph_pkl_filename, 'rb') as graph_pkl_file:
        graph_data = pickle.load(graph_pkl_file)
    graph = graph_data.graph

    makespan, _ = compute_dag_execution_time(graph, data)

    data_to_save['acceleration-factor'].append(1.0/hw)
    data_to_save['area-constraint'].append(area)
    data_to_save['seed'].append(seed)
    data_to_save['makespan'].append(makespan)


df = pd.DataFrame(data_to_save)
df.to_csv("outputs/squeeze-net-mip-partitions.csv", index=False)