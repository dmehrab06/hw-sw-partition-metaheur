import argparse
import logging
import numpy as np
import pandas as pd
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
import os
import pickle
from meta_heuristic.TaskGraph import TaskGraph

warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments with proper validation."""
    parser = argparse.ArgumentParser(description='Task Graph Partitioning Evaluation')
    
    parser.add_argument('--graph-file', type=str, required=True,
                       help='Graph file name to load')
    parser.add_argument('--area-constraint', type=float, required=True,
                       help='Area constraint (should be between 0 and 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()

def validate_arguments(args):
    """Validate input arguments with assertions."""
    try:
        assert 0 < args.area_constraint <= 1, f"Area constraint must be between 0 and 1, got {args.area_constraint}"
        
        print("All input arguments validated successfully")
        
    except AssertionError as e:
        print(f"Argument validation failed: {e}")
        sys.exit(1)



def main():
    # Parse arguments first
    args = parse_arguments()

    validate_arguments(args)
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    makespan_dicts = []

    
    for l in [0.5,1,5,10]: ## hw scale variance
        for mu in [0.5,1,5,10]: ## communication scale
            #print(f'sbatch pso_eval.sbatch {gdir}{graph_name} {area_constraint} 0.3 {l} {mu} 42')
            TG = TaskGraph(area_constraint=args.area_constraint)
            TG.load_graph_from_pydot(
                args.graph_file,
                k=0.3,
                l=l,
                mu=mu,
                A_max=100,seed=args.seed
            )
            graph_name = (args.graph_file.split('/')[1]).split('.')[0]
            partition_name  = (f"taskgraph-{graph_name}_"
                               f"area-{args.area_constraint:.2f}_"
                               f"hwscale-{0.3}_"
                               f"hwvar-{l:.2f}_"
                               f"comm-{mu:.2f}_"
                               f"seed-{args.seed}")
            greedy_partition_file = f'partitions/{partition_name}_assignment-greedy.pkl'
            pso_partition_file = f'partitions/{partition_name}_assignment-pso.pkl'
            random_partition_file = f'partitions/{partition_name}_assignment-random.pkl'

            greedy_assignment = pd.read_pickle(greedy_partition_file)
            pso_assignment = pd.read_pickle(pso_partition_file)
            random_assignment = pd.read_pickle(random_partition_file)
            
            greedy_perf = TG.get_makespan(greedy_assignment,verbose=False)
            pso_perf = TG.get_makespan(pso_assignment,verbose=False)
            random_perf = TG.get_makespan(random_assignment,verbose=False)

            print('Greedy',greedy_perf['makespan'],'PSO',pso_perf['makespan'],'Random',random_perf['makespan'])

            makespan_data = {
            'GraphName': graph_name,
            'HW_Scale_Var': l,
            'Comm_Scale_Var': mu,
            'Area_Percentage': args.area_constraint,
            'Seed': 42,
            'Random_makespan': random_perf['makespan'],
            'PSO_makespan': pso_perf['makespan'],
            'Greedy_makespan': greedy_perf['makespan']
            }

            makespan_dicts.append(makespan_data)

    set_results = pd.DataFrame.from_dict(makespan_dicts)
    os.makedirs('outputs', exist_ok=True)
    file_path = f'outputs/result_makespan_{graph_name}_{args.area_constraint}_areaconstraint.csv'
    set_results.to_csv(file_path, mode='a', index=False, header=write_header)

if __name__ == "__main__":
    main()