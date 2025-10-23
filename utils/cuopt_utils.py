import os, sys
import networkx as nx
import numpy as np
from typing import Dict
from cuopt.linear_programming.problem import Problem, INTEGER, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from utils.partition_utils import ScheduleConstPartitionSolver

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/schedule_partitioner.log")

logger = LogManager.get_logger(__name__)

class CuOptScheduleConstPartitionSolver(ScheduleConstPartitionSolver):
    def solve_optimization(self, A_max: float, big_M: float = 1000, use_reduced_sw_constraints=True, time_limit=300) -> Dict:
        """
        Solve the hardware-software partitioning optimization problem
        
        Args:
            A_max: Maximum hardware area constraint
            big_M: Big-M constant for logical constraints
            use_reduced_sw_constraints: flag to reduce SW sequence constraints through topological ordering
            
        Returns:
            Dictionary containing solution details
        """
        if self.S_source is None:
            raise ValueError("Problem matrices not created. Load a graph first.")
        
        logger.info(f"Solving optimization with A_max = {A_max}")
        
        # Create problem
        problem = Problem("Hardware_Software_Partitioning")
        
        # ===== Decision Variables =====
        
        # x[i]: Binary variable for partition assignment (1 = software, 0 = hardware)
        x = [problem.addVariable(vtype=INTEGER, lb=0, ub=1, name=f"x_{i}") for i in range(self.n_nodes)]
        
        # z[e]: Binary variable for communication (1 = nodes on different partitions)
        z = [problem.addVariable(vtype=INTEGER, lb=0, ub=1, name=f"z_{e}") for e in range(self.n_edges)]
        
        # t[i]: Start time of node i (continuous, non-negative)
        t = [problem.addVariable(vtype=CONTINUOUS, lb=0, name=f"t_{i}") for i in range(self.n_nodes)]
        
        # f[i]: Finish time of node i (continuous, non-negative)
        f = [problem.addVariable(vtype=CONTINUOUS, lb=0, name=f"f_{i}") for i in range(self.n_nodes)]
        
        # T: Makespan (continuous, non-negative)
        T = problem.addVariable(vtype=CONTINUOUS, lb=0, name="Makespan")

        # ===== Constraints =====
    
        # 1. Hardware area constraint
        # sum(a[i] * (1 - x[i])) <= A_max
        hardware_area_expr = sum(self.a[i] * (1 - x[i]) for i in range(self.n_nodes))
        problem.addConstraint(hardware_area_expr <= A_max, name="Hardware_Area_Constraint")
        
        # 2. Execution time definition
        # f[i] = t[i] + h[i] * (1 - x[i]) + s[i] * x[i]
        for i in range(self.n_nodes):
            execution_time = t[i] + self.h[i] * (1 - x[i]) + self.s[i] * x[i]
            problem.addConstraint(f[i] == execution_time, name=f"Exec_Time_{i}")
        
        # 3. Precedence constraints with communication
        # For each edge e: S_source[e] @ f + c[e] * z[e] <= S_target[e] @ t
        for e in range(self.n_edges):
            # S_source[e] is a row vector; compute dot product with f
            source_finish = sum(self.S_source[e, i] * f[i] for i in range(self.n_nodes))
            target_start = sum(self.S_target[e, i] * t[i] for i in range(self.n_nodes))
            
            problem.addConstraint(
                source_finish + self.c[e] * z[e] <= target_start,
                name=f"Precedence_{e}"
            )
        
        # 4. Communication cost variables (detect different partition assignments)
        # z[e] = 1 if source and target of edge e are on different partitions
        for e in range(self.n_edges):
            # Get the source and target nodes for this edge
            source_x = sum(self.S_source[e, i] * x[i] for i in range(self.n_nodes))
            target_x = sum(self.S_target[e, i] * x[i] for i in range(self.n_nodes))
            
            # z[e] >= |source_x - target_x|
            problem.addConstraint(z[e] >= source_x - target_x, name=f"Comm_Lower_1_{e}")
            problem.addConstraint(z[e] >= target_x - source_x, name=f"Comm_Lower_2_{e}")
            
            # z[e] <= source_x + target_x (both on software)
            problem.addConstraint(z[e] <= source_x + target_x, name=f"Comm_Upper_1_{e}")
            
            # z[e] <= 2 - source_x - target_x (both on hardware)
            problem.addConstraint(z[e] <= 2 - source_x - target_x, name=f"Comm_Upper_2_{e}")
        
        # 5. Sequential execution of software nodes
        if use_reduced_sw_constraints:
            # Use topological ordering to reduce constraints from O(n²) to O(n)
            topo_order = list(nx.topological_sort(self.graph))
            topo_indices = [self.node_to_index[node] for node in topo_order]
            
            logger.info(f"Using reduced SW constraints with topological order: {topo_order}")
            
            # Only constrain consecutive nodes in topological order
            for k in range(len(topo_indices) - 1):
                i, j = topo_indices[k], topo_indices[k + 1]
                
                # If both nodes are software (x[i] = 1 and x[j] = 1),
                # then node i must complete before node j starts: f[i] <= t[j]
                # Using big-M: f[i] <= t[j] + M * (1 - x[i]) + M * (1 - x[j])
                problem.addConstraint(
                    f[i] <= t[j] + big_M * (1 - x[i]) + big_M * (1 - x[j]),
                    name=f"SW_Sequence_{i}_{j}"
                )
        else:
            # Original O(n²) approach with binary ordering variables
            # Y[i,j] = 1 if software node i executes before software node j
            Y = [[problem.addVariable(vtype=INTEGER, name=f"Y_{i}_{j}") 
                for j in range(self.n_nodes)] for i in range(self.n_nodes)]
            
            logger.info(f"Adding {self.n_nodes*(self.n_nodes-1)} software sequencing constraints (full pairwise)")
            
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        # If Y[i,j] = 1 and both are software, then f[i] <= t[j]
                        problem.addConstraint(
                            f[i] <= t[j] + big_M * (1 - Y[i][j]) + big_M * (2 - x[i] - x[j]),
                            name=f"SW_Order_1_{i}_{j}"
                        )
                        
                        # If Y[i,j] = 0 and both are software, then f[j] <= t[i]
                        problem.addConstraint(
                            f[j] <= t[i] + big_M * Y[i][j] + big_M * (2 - x[i] - x[j]),
                            name=f"SW_Order_2_{i}_{j}"
                        )
                        
                        # Y[i,j] can only be 1 if both i and j are software
                        problem.addConstraint(Y[i][j] <= x[i], name=f"Y_Valid_1_{i}_{j}")
                        problem.addConstraint(Y[i][j] <= x[j], name=f"Y_Valid_2_{i}_{j}")
        
        # 6. Makespan definition: T >= f[i] for all i
        for i in range(self.n_nodes):
            problem.addConstraint(T >= f[i], name=f"Makespan_{i}")
        
        # ===== Objective =====
        # Minimize makespan
        problem.setObjective(T, sense=MINIMIZE)
        
        # ===== Solve =====
        settings = SolverSettings()
        settings.set_parameter("time_limit", time_limit)
        
        logger.info("Starting cuOpt solver...")
        
        problem.solve(settings)
        
        # ===== Extract Solution =====
        if problem.Status.name != "Optimal" and problem.Status.name != "Feasible":
            logger.error(f"Optimization failed with status: {problem.Status.name}")
            return {
                'status': problem.Status.name,
                'makespan': None,
                'error': 'No feasible solution found'
            }
        
        # Store solution
        self.x_sol = np.array([x[i].getValue() for i in range(self.n_nodes)]).round().astype(int)
        self.t_sol = np.array([t[i].getValue() for i in range(self.n_nodes)])
        self.f_sol = np.array([f[i].getValue() for i in range(self.n_nodes)])
        self.z_sol = np.array([z[e].getValue() for e in range(self.n_edges)]).round().astype(int)
        self.T_sol = T.getValue()
        
        #  Convert solution back to node IDs
        hw_nodes = [self.node_list[i] for i in range(self.n_nodes) if self.x_sol[i] == 0]
        sw_nodes = [self.node_list[i] for i in range(self.n_nodes) if self.x_sol[i] == 1]
        
        # Sort by start times
        hw_sequence = sorted(hw_nodes, key=lambda node: self.t_sol[self.node_to_index[node]])
        sw_sequence = sorted(sw_nodes, key=lambda node: self.t_sol[self.node_to_index[node]])
        
        # Create start/finish time dictionaries with node IDs
        start_times_dict = {self.node_list[i]: self.t_sol[i] for i in range(self.n_nodes)}
        finish_times_dict = {self.node_list[i]: self.f_sol[i] for i in range(self.n_nodes)}
        
        # Calculate total hardware area used
        total_hw_area = np.sum(self.a * (1 - self.x_sol))
        
        solution = {
            'status': problem.Status.name,
            'makespan': self.T_sol,
            'hardware_nodes': hw_nodes,
            'software_nodes': sw_nodes,
            'hardware_sequence': hw_sequence,
            'software_sequence': sw_sequence,
            'start_times': start_times_dict,
            'finish_times': finish_times_dict,
            'total_hardware_area': total_hw_area,
            'area_constraint': A_max,
            'solve_time': problem.SolveTime
        }
        
        logger.info(f"Optimization successful! Status: {problem.Status.name}")
        logger.info(f"Makespan = {self.T_sol:.2f}")
        logger.info(f"Solve time = {problem.SolveTime:.2f} seconds")
        logger.info(f"Hardware nodes: {hw_nodes}")
        logger.info(f"Software nodes: {sw_nodes}")
        logger.info(f"Total hardware area used: {total_hw_area:.2f} / {A_max}")
        
        return solution