from utils.logging_utils import LogManager
from utils.graph_generator import GraphGenerator
from utils.optimizer import MISDPSolver

LogManager.initialize("logs/test_sdp_optimizer.log")
logger = LogManager.get_logger(__name__)


    


def main():
    """Main function to run the optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Solve MISDP for node partitioning')
    parser.add_argument('-g','--graph', type=str, default="data/example_graph.pkl", help='Path to load graph from')
    parser.add_argument('-R','--constraint', type=float, default=0, help='Constraint bound')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples for rounding')
    parser.add_argument('--output', type=str, default='results/solution.png', help='Output file for visualization')
    
    args = parser.parse_args()
    
    try:
        # Load the graph
        logger.info(f"Creating graph")
        generator = GraphGenerator()
        graph = generator.load_graph(args.graph)

        sum_software = sum([graph.nodes[n]['software_cost'] for n in graph.nodes])
        logger.info(f"All software cost: {sum_software}")
        
        # Create solver
        solver = MISDPSolver()
        
        # Solve the problem
        result = solver.solve(graph, R=args.constraint, num_samples=args.samples)
        
        if result:
            # Print results
            print("\nOptimization Results:")
            print("-" * 50)
            print(f"Hardware Cost: {result['hw_cost']:.4f}")
            print(f"Software Cost: {result['sw_cost']:.4f}")
            print(f"Communication Cost: {result['comm_cost']:.4f}")
            print(f"Total SW/Comm Cost: {result['sw_comm_cost']:.4f}")
            print(f"Constraint Satisfied: {result['constraint_satisfied']}")
            if not result['constraint_satisfied']:
                print(f"Constraint Violation: {result['constraint_violation']:.4f}")
            print(f"Solution Time: {result['solve_time']:.2f} seconds")
            print("-" * 50)
            
            # Node assignments
            print("\nNode Assignments:")
            for i, xi in enumerate(result['x']):
                node_type = "Software" if xi > 0.5 else "Hardware"
                print(f"Node {i}: {node_type}")
            
            # Visualize solution
            solver.visualize_solution(args.constraint, args.output)
            print(f"\nVisualization saved to {args.output}")
        else:
            print("Failed to find a solution")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    

if __name__ == "__main__":
    import os
    # Ensure necessary directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    main()