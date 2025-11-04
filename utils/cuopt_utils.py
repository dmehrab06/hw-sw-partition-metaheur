import os, sys
import networkx as nx
import numpy as np
from typing import Dict
import gc

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
        try:
            from cuopt.linear_programming.problem import Problem, INTEGER, CONTINUOUS, MINIMIZE
            from cuopt.linear_programming.solver_settings import SolverSettings
        
        except ImportError as e:
            logger.error(f'cuOpt import failed: {e}')
            return {
                'status': 'Error',
                'makespan': None
            }
        
        if self.S_source is None:
            logger.error("Problem matrices not created. Load a graph first.")
            raise ValueError("Problem matrices not created. Load a graph first.")
        
        logger.info(f"Solving optimization with A_max = {A_max}")

        # Force garbage collection before starting
        gc.collect()

        # Convert all inputs to basic Python types to avoid numpy issues
        s_list = [float(x) for x in self.s]
        h_list = [float(x) for x in self.h]
        a_list = [float(x) for x in self.a]
        c_list = [float(x) for x in self.c]
        A_max_val = float(A_max)
        big_M_val = float(big_M)

        logger.info("Converted inputs to Python types")

        # Pre-compute reasonable bounds
        max_time = sum(s_list) * 2.0  # Conservative upper bound

        # Pre-compute edge structure once
        edge_sources = []
        edge_targets = []
        for e in range(self.n_edges):
            src = [i for i in range(self.n_nodes) if self.S_source[e, i] > 0]
            tgt = [i for i in range(self.n_nodes) if self.S_target[e, i] > 0]
            edge_sources.append(src)
            edge_targets.append(tgt)
        
        logger.info("Pre-computed edge structure")
    
        
        
        # Create problem
        try:
            problem = Problem("Hardware_Software_Partitioning")
        except Exception as e:
            return {
                'status': 'Error', 
                'makespan': None, 
                'error': f'Problem creation failed: {e}'
                }
        
        # ===== Decision Variables =====
        # Store all variables in lists to maintain references
        x_vars = []
        z_vars = []
        t_vars = []
        f_vars = []
        
        try:
            # Create partition assignment variables (1 = software, 0 = hardware)
            for i in range(self.n_nodes):
                x_i = problem.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"x_{i}")
                x_vars.append(x_i)
            
            gc.collect()
            
            # Create communication variables
            for e in range(self.n_edges):
                z_e = problem.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"z_{e}")
                z_vars.append(z_e)
            
            gc.collect()
            
            # Create timing variables with reasonable bounds
            max_time = float(np.sum(self.s))  # Upper bound: all tasks in software sequentially
            
            for i in range(self.n_nodes):
                t_i = problem.addVariable(lb=0, ub=max_time, name=f"t_{i}")
                f_i = problem.addVariable(lb=0, ub=max_time, name=f"f_{i}")
                t_vars.append(t_i)
                f_vars.append(f_i)

            gc.collect()
            
            # Makespan variable
            T_var = problem.addVariable(lb=0, ub=max_time, name="T")

            gc.collect()
        
        except Exception as e:
            logger.error(f'Variable creation failed: {e}')
            return {
                'status': 'Error', 
                'makespan': None
                }

        # ===== Add Constraints - One at a time with garbage collection =====
        constraint_count = 0

        # ===== Constraint 1: Hardware area constraint =====
        try:
            # Build sum term by term
            hw_sum = None
            for i in range(self.n_nodes):
                term = a_list[i] * (1.0 - x_vars[i])
                if hw_sum is None:
                    hw_sum = term
                else:
                    hw_sum = hw_sum + term
            
            problem.addConstraint(hw_sum <= A_max_val, name="HW_Area")
            constraint_count += 1
            gc.collect()
        
            logger.info("Added hardware area constraint")

        except Exception as e:
            logger.error(f'Hardware area constraint addition failed: {e}')
            return {
                'status': 'Error', 
                'makespan': None
                }
        
        # ===== Constraint 2: Execution time definition =====
        try:
            for i in range(self.n_nodes):
                # f[i] = t[i] + h[i]*(1-x[i]) + s[i]*x[i]
                hw_term = h_list[i] * (1.0 - x_vars[i])
                sw_term = s_list[i] * x_vars[i]
                rhs = t_vars[i] + hw_term + sw_term
                
                problem.addConstraint(f_vars[i] == rhs, name=f"E{i}")
                constraint_count += 1
                
                if (i + 1) % 10 == 0:
                    gc.collect()
        
            logger.info(f"Added {self.n_nodes} execution time constraints")
        
        except Exception as e:
            logger.error(f'Execution time constraint addition failed: {e}')
            return {
                'status': 'Error', 
                'makespan': None
                }
        
        # ===== Constraint 3: Precedence constraints =====
        # Pre-compute edge source and target nodes to reduce memory operations
        try:
            for e in range(self.n_edges):
                src_nodes = edge_sources[e]
                tgt_nodes = edge_targets[e]
                
                # Most edges have single source/target
                if len(src_nodes) == 1:
                    src_finish = f_vars[src_nodes[0]]
                else:
                    src_finish = sum(f_vars[i] for i in src_nodes)
                
                if len(tgt_nodes) == 1:
                    tgt_start = t_vars[tgt_nodes[0]]
                else:
                    tgt_start = sum(t_vars[i] for i in tgt_nodes)
                
                comm = c_list[e] * z_vars[e]
                lhs = src_finish + comm
                
                problem.addConstraint(lhs <= tgt_start, name=f"P{e}")
                constraint_count += 1
                
                if (e + 1) % 10 == 0:
                    gc.collect()
            
            logger.info(f"Added {self.n_edges} precedence constraints")
        
        except Exception as e:
            logger.error(f'Precedence constraint addition failed: {e}')
            return {
                'status': 'Error', 
                'makespan': None
                }
        
        # ===== Constraint 4: Communication cost detection =====
        # z[e] = 1 if source and target are on different partitions
        try:
            for e in range(self.n_edges):
                src_nodes = edge_sources[e]
                tgt_nodes = edge_targets[e]
                
                # Get x values
                if len(src_nodes) == 1:
                    src_x = x_vars[src_nodes[0]]
                else:
                    src_x = sum(x_vars[i] for i in src_nodes)
                
                if len(tgt_nodes) == 1:
                    tgt_x = x_vars[tgt_nodes[0]]
                else:
                    tgt_x = sum(x_vars[i] for i in tgt_nodes)
                
                # Four constraints per edge
                problem.addConstraint(z_vars[e] >= src_x - tgt_x, name=f"C1_{e}")
                problem.addConstraint(z_vars[e] >= tgt_x - src_x, name=f"C2_{e}")
                problem.addConstraint(z_vars[e] <= src_x + tgt_x, name=f"C3_{e}")
                problem.addConstraint(z_vars[e] <= 2.0 - src_x - tgt_x, name=f"C4_{e}")
                constraint_count += 4
                
                if (e + 1) % 10 == 0:
                    gc.collect()
            
            logger.info(f"Added {4*self.n_edges} communication detection constraints")
        
        except Exception as e:
            logger.error(f'Communication cost detection constraint addition failed: {e}')
            return {
                'status': 'Error', 
                'makespan': None
                }
        
        # ===== Constraint 5: Software sequencing =====
        if use_reduced_sw_constraints:
            try:
                # Use topological ordering for O(n) constraints
                topo_order = list(nx.topological_sort(self.graph))
                topo_indices = [self.node_to_index[node] for node in topo_order]
                
                logger.info(f"Using reduced SW constraints with {len(topo_indices)-1} ordering constraints")
                
                for k in range(len(topo_indices) - 1):
                    i = topo_indices[k]
                    j = topo_indices[k + 1]
                    
                    # f[i] <= t[j] + M*(2 - x[i] - x[j])
                    # Active only when both are software (x[i]=1, x[j]=1)
                    penalty = big_M_val * (2.0 - x_vars[i] - x_vars[j])
                    problem.addConstraint(f_vars[i] <= t_vars[j] + penalty, name=f"S{i}_{j}")
                    constraint_count += 1

                    if (k + 1) % 10 == 0:
                        gc.collect()
                
                logger.info(f"Added {len(topo_indices)-1} reduced SW constraints")

            except Exception as e:
                logger.error(f'Software sequencing constraints (reduced) addition failed: {e}')
                return {
                    'status': 'Error', 
                    'makespan': None
                    }
        
        else:
            try:
                # Full O(n²) approach with ordering variables
                Y_vars = [[None] * self.n_nodes for _ in range(self.n_nodes)]
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        if i != j:
                            Y_vars[i][j] = problem.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"Y_{i}_{j}")
                            
                gc.collect()
                
                logger.info(f"Using full SW constraints with {self.n_nodes*(self.n_nodes-1)} ordering constraints")
                
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        if i != j:
                            Y_ij = Y_vars[i][j]
                            
                            # f[i] <= t[j] + M*(1 - Y[i,j]) + M*(2 - x[i] - x[j])
                            penalty1 = big_M_val * (1.0 - Y_ij) + big_M_val * (2.0 - x_vars[i] - x_vars[j])
                            problem.addConstraint(f_vars[i] <= t_vars[j] + penalty1, name=f"SW_O1_{i}_{j}")
                            
                            # f[j] <= t[i] + M*Y[i,j] + M*(2 - x[i] - x[j])
                            penalty2 = big_M_val * Y_ij + big_M_val * (2.0 - x_vars[i] - x_vars[j])
                            problem.addConstraint(f_vars[j] <= t_vars[i] + penalty2, name=f"SW_O2_{i}_{j}")
                            
                            # Y[i,j] <= x[i] and Y[i,j] <= x[j]
                            problem.addConstraint(Y_ij <= x_vars[i], name=f"Y_V1_{i}_{j}")
                            problem.addConstraint(Y_ij <= x_vars[j], name=f"Y_V2_{i}_{j}")
                            
                            constraint_count += 4

                            if constraint_count % 50 == 0:
                                gc.collect()
                
                logger.info(f"Added full sequencing constraints")
            
            except Exception as e:
                logger.error(f'Software sequencing constraints (full) addition failed: {e}')
                return {
                    'status': 'Error', 
                    'makespan': None
                    }
        
        # ===== Constraint 6: Makespan definition =====
        try:
            for i in range(self.n_nodes):
                problem.addConstraint(T_var >= f_vars[i], name=f"Makespan_{i}")
                constraint_count += 1
            
            gc.collect()
            
            logger.info(f"Added {self.n_nodes} makespan constraints")

        except Exception as e:
            logger.error(f'Makespan definition addition failed: {e}')
            return {
                'status': 'Error', 
                'makespan': None
                }
        
        logger.info(f"Total constraints added: {constraint_count}")
        
        # ===== Objective: Minimize makespan =====
        try:
            problem.setObjective(T_var, sense=MINIMIZE)
        
        except Exception as e:
            logger.error(f'Objective setting failed: {e}')
            return {
                'status': 'Error', 
                'makespan': None
                }
        
        # ===== Solver Settings =====
        try:
            settings = SolverSettings()
            settings.set_parameter("time_limit", time_limit)
        
        except Exception as e:
            logger.error(f'Settings creation failed: {e}')
            return {
                'status': 'Error', 
                'makespan': None
                }
        
        logger.info("Starting cuOpt solver...")

        gc.collect()  # Final cleanup before solve
        
        # ===== Solve =====
        try:
            problem.solve(settings)
            logger.info(f"Solve completed with status: {problem.Status.name}")

        except Exception as e:
            logger.error(f"Solver failed with exception: {e}")
            return {
                'status': 'Error',
                'makespan': None
            }
        
        # ===== Extract Solution =====
        try:
            if problem.Status.name not in ["Optimal", "Feasible"]:
                logger.error(f"Optimization failed with status: {problem.Status.name}")
                return {
                    'status': problem.Status.name,
                    'makespan': None
                }
            
            # Carefully extract values to avoid memory issues
            self.x_sol = np.zeros(self.n_nodes, dtype=int)
            self.t_sol = np.zeros(self.n_nodes)
            self.f_sol = np.zeros(self.n_nodes)
            
            for i in range(self.n_nodes):
                self.x_sol[i] = int(round(x_vars[i].getValue()))
                self.t_sol[i] = t_vars[i].getValue()
                self.f_sol[i] = f_vars[i].getValue()
            
            self.z_sol = np.zeros(self.n_edges, dtype=int)
            for e in range(self.n_edges):
                self.z_sol[e] = int(round(z_vars[e].getValue()))
            
            self.T_sol = T_var.getValue()
            
            # Convert solution back to node IDs
            hw_nodes = [self.node_list[i] for i in range(self.n_nodes) if self.x_sol[i] == 0]
            sw_nodes = [self.node_list[i] for i in range(self.n_nodes) if self.x_sol[i] == 1]
            
            # Sort by start times
            hw_sequence = sorted(hw_nodes, key=lambda node: self.t_sol[self.node_to_index[node]])
            sw_sequence = sorted(sw_nodes, key=lambda node: self.t_sol[self.node_to_index[node]])
            
            # Create dictionaries
            start_times_dict = {self.node_list[i]: self.t_sol[i] for i in range(self.n_nodes)}
            finish_times_dict = {self.node_list[i]: self.f_sol[i] for i in range(self.n_nodes)}
            
            # Calculate total hardware area
            total_hw_area = float(np.sum(self.a * (1 - self.x_sol)))
            
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
        
        except Exception as e:
            logger.error(f"Solution extraction failed: {e}")
            return {
                'status': 'Error', 
                'makespan': None
            }
        
        finally:
            # Final cleanup
            gc.collect()