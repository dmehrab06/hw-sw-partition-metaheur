import pickle
import pandas as pd
import os
from utils.scheduler_utils import compute_dag_execution_time
from utils.partition_utils import ScheduleConstPartitionSolver

solution_dir = "makespan-opt-partitions"
solutions_files = os.listdir(solution_dir)
solution_dict = {}

solver = ScheduleConstPartitionSolver()
graph = solver.load_pydot_graph(
        pydot_file="inputs/task_graph_topology/soda-benchmark-graphs/pytorch-graphs/squeeze_net_tosa.dot", 
        k=0.1,l=0.5,mu=1.0,A_max=100)

for f in solutions_files:
    if f.endswith(".pkl"):
        file_base = f[:-4]  # Remove .pkl extension
        algorithm_key = file_base.split('-')[-1]  # Get the algorithm part
        with open(os.path.join(solution_dir, f), 'rb') as file:
            assignment = pickle.load(file)

        partition_assignment = {k: 'hardware' if v == 1 else 'software' for k, v in assignment.items()}
        makespan = compute_dag_execution_time(graph, partition_assignment)

        solution_dict[algorithm_key] = makespan['makespan']

solution_df = pd.DataFrame(list(solution_dict.items()), columns=['Algorithm', 'Makespan'])
solution_df = solution_df.sort_values(by='Makespan')

import matplotlib.pyplot as plt
import seaborn as sns


fig,ax = plt.subplots(1,1,figsize=(12,6))
sns.barplot(data=solution_df,x='Algorithm',y='Makespan',ax=ax, palette=sns.color_palette("Set2"))
ax.set_ylabel("Makespan (lower is better)",fontsize=18)
ax.set_xlabel("Algorithm",fontsize=18)
ax.set_title("Makespan Comparison of Different Algorithms on squeeze-net-tosa graph",fontsize=16)
ax.hlines(y=solution_df['Makespan'].min(), color='r', linestyle='--', label='MIP', xmin=-0.5, xmax=len(solution_df)-0.5)
ax.tick_params(axis='x', labelsize=14, rotation=45)
fig.savefig("outputs/makespan_comparison.png", bbox_inches='tight',dpi=300)
