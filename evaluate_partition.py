from utils.taskgraph_utils import load_taskgraph
from utils.partition_utils import ScheduleConstPartitionSolver
import pandas as pd
import sys

#all_strategies = ['arato','comm','cpl','ed','spt']
all_strategies = ['arato','comm','cpl','ed']
area = float(sys.argv[1])
hw = float(sys.argv[2])
mh_technique = sys.argv[3]

graph_name = 'taskgraph-squeeze_net_tosa'

config_dict = {'area': area, 'hw': hw, 'mh': mh_technique}

all_results = []

for seed in range(0,4):

    print('working on seed', seed, flush=True)
    
    pickle_graph = f"inputs/task_graph_complete/{graph_name}-instance-config-config_arato_area_{area}_hw_{hw}_seed_{seed}.pkl"
    soln_partition_file = f'mip-opt-partitions/{graph_name}_area-{area:.2f}_hwscale-{hw}_hwvar-0.50_seed-{seed}_assignment-mip.pkl'

    data = load_taskgraph(pickle_graph)
    tot_area = sum(list(data.hardware_area.values()))
    opt_soln_parition = load_taskgraph(soln_partition_file)

    solver = ScheduleConstPartitionSolver()
    solver.load_pickle_graph(pickle_graph)
    
    opt_sol = solver.solve_optimization(A_max=tot_area, partition_assignment=opt_soln_parition,verbose=False)

    print(f"optimal solution has a makespan of {opt_sol['makespan']}, uses {opt_sol['total_hardware_area']/tot_area} of total area")
    
    for strategy in all_strategies:

        print('metaheuristic strategy', strategy, flush=True)
        
        mh_partition_file = f'{strategy}-opt-partitions/{graph_name}_area-{area:.2f}_hwscale-{hw}_hwvar-0.50_comm-1.00_seed-{seed}_assignment-{mh_technique}.pkl'

        meta_soln_parition = load_taskgraph(mh_partition_file)
        meta_sol = solver.solve_optimization(A_max=tot_area, partition_assignment=meta_soln_parition,verbose=False)
        
        print(f"metaheuristic solution has a makespan of {meta_sol['makespan']}, uses {meta_sol['total_hardware_area']/tot_area} of total area")
        
        result_dict = {'seed': seed, 'mip_cost': opt_sol['makespan'], 'mh_cost': meta_sol['makespan'], 
                       'mip_area_used': opt_sol['total_hardware_area']/tot_area, 'mh_area_used': meta_sol['total_hardware_area']/tot_area}
    
        all_results.append(result_dict | config_dict)

df = pd.DataFrame.from_dict(all_results)
df.to_csv(f'eval_logs/mh-vs-mip-area-{area:.2f}-hw-{hw:.2f}-{mh_technique}.csv',index=False)

