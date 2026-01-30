import sys
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Union

from task_graph_generation import TaskGraphDataset, generate_random_dag, networkx_to_pytorch


# Task status constants
NOT_READY = 0    # Task has predecessors that are not complete
READY = 1        # All predecessors are complete, task can be scheduled
IN_PROGRESS = 2  # Task is currently being processed
COMPLETE = 3     # Task has finished processing


# Custom exception for action errors
class ActionError(Exception):
    """Exception raised for errors in the action execution."""
    pass


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # Static features
    N_op: torch.Tensor = None            # Scalar value per sample - number of operations in the task graph
    op_adj_mat: torch.Tensor = None      # Adjacency matrix of operations in the task graph as a DAG - represents task precedence
    comm_cost_mat: torch.Tensor = None   # Matrix of communication costs, size [N_op, N_op]
    sw_cost_vec: torch.Tensor = None     # Vector of SW costs (CPU runtimes) per operation
    hw_cost_vec: torch.Tensor = None     # new
    hw_area_vec: torch.Tensor = None     # Vector of HW costs (area for HW implementation) per operation
    hw_area_limit: torch.Tensor = None   # Scalar value per sample - HW area limit for the task graph

    # Dynamic features
    batch_idxes: torch.Tensor = None     # Indices of active batch samples
    op_status: torch.Tensor = None       # Status of each operation (NOT_READY, READY, IN_PROGRESS, COMPLETE)
    op_start_time: torch.Tensor = None   # Start time of each operation (estimated until scheduled)
    op_end_time: torch.Tensor = None     # End time of each operation (estimated until scheduled)
    n_pred_remaining: torch.Tensor = None  # Number of predecessors remaining for each op
    running_op_adj_mat: torch.Tensor = None  # Current adjacency matrix (updated as tasks complete)
    current_time: torch.Tensor = None    # Current time in the environment
    op_resource: torch.Tensor = None     # Resource assignment for each operation (-1: not scheduled, 0: SW, 1: HW)

    # TODO: add the finish time of each predecessor, or even precompute their max (would be the READY time)


    # Tracking variables
    hw_usage: torch.Tensor = None        # Current HW area usage
    hw_area_remaining: torch.Tensor = None  # Remaining HW area (hw_area_limit - hw_usage)
    sw_busy: torch.Tensor = None         # Whether SW is busy for each instance
    sw_end_time: torch.Tensor = None     # Time when SW will become available again
    makespan: torch.Tensor = None        # Current makespan (max end time of any operation)
    done: torch.Tensor = None            # Whether each instance is done

    def update(self, batch_idxes, op_status, op_start_time, op_end_time, n_pred_remaining,
               running_op_adj_mat, current_time, op_resource, hw_usage, hw_area_remaining, sw_busy, sw_end_time, makespan, done):
        '''
        Update the state with new values
        '''
        self.batch_idxes = batch_idxes
        self.op_status = op_status
        self.op_start_time = op_start_time
        self.op_end_time = op_end_time
        self.n_pred_remaining = n_pred_remaining
        self.running_op_adj_mat = running_op_adj_mat
        self.current_time = current_time
        self.op_resource = op_resource
        self.hw_usage = hw_usage
        self.hw_area_remaining = hw_area_remaining
        self.sw_busy = sw_busy
        self.sw_end_time = sw_end_time
        self.makespan = makespan
        self.done = done
        return self


class CSchedEnv(gym.Env):
    '''
    Computational Scheduling Environment
    '''
    def __init__(self, dataset, env_paras, mode=0):
        '''
        Initialize the environment

        Args:
            dataset: TaskGraphDataset containing the task graphs
            env_paras: Dictionary of environment parameters
        '''
        self.mode = mode
        #print("Inside CSchedEnv initializer")

        self.sparse_reward = False

        # Store parameters
        self.batch_size       = env_paras.get("batch_size", len(dataset))
        self.device           = env_paras.get("device", "cpu")
        self.timestep_mode    = env_paras.get("timestep_mode", "next_complete")
        self.timestep_trigger = env_paras.get("timestep_trigger", "every")

        #print("self.batch_size")
        #print( self.batch_size )
        #input("waiting")

        # Set up action and observation spaces (placeholder for now)
        self.action_space = gym.spaces.Discrete(2)  # Placeholder - will be updated based on actual action space
        self.observation_space = gym.spaces.Discrete(2)  # Placeholder - will be updated based on actual observation space

        # Load task graphs from dataset
        self.graphs = []
        self.adj_matrices = []
        self.node_features = []
        self.edge_features = []
        self.hw_area_limits = []
        self.true_num_nodes = []

        # Sample batch_size graphs from the dataset
        indices = random.sample(range(len(dataset)), min(self.batch_size, len(dataset)))


        #input("waiting")
        for idx in indices:
            graph, adj_matrix, node_feat, edge_feat, hw_limit = dataset[idx]
            self.graphs.append(graph)
            self.adj_matrices.append(adj_matrix)
            self.node_features.append(node_feat)
            self.edge_features.append(edge_feat)
            self.hw_area_limits.append(hw_limit)
            self.true_num_nodes.append(adj_matrix.shape[0])

        # Find the maximum number of nodes across all graphs in the batch
        max_nodes = max(adj.shape[0] for adj in self.adj_matrices)

        # Pad adjacency matrices and node features to the same size
        padded_adj_matrices = []
        padded_node_features = []

        for i, (adj, node_feat) in enumerate(zip(self.adj_matrices, self.node_features)):
            num_nodes = adj.shape[0]

            # Pad adjacency matrix
            padded_adj = torch.zeros((max_nodes, max_nodes), dtype=adj.dtype)
            padded_adj[:num_nodes, :num_nodes] = adj
            padded_adj_matrices.append(padded_adj)

            # Pad node features
            padded_node_feat = torch.zeros((node_feat.shape[0], max_nodes), dtype=node_feat.dtype)
            padded_node_feat[:, :num_nodes] = node_feat
            padded_node_features.append(padded_node_feat)

        # Stack tensors for batch processing
        self.adj_matrices_batch = torch.stack(padded_adj_matrices, dim=0).to(self.device)
        self.node_features_batch = torch.stack(padded_node_features, dim=0).to(self.device)
        self.hw_area_limits_batch = torch.stack(self.hw_area_limits, dim=0).to(self.device)

        # Store the true number of nodes for each graph
        self.N_op_batch = torch.tensor(self.true_num_nodes, dtype=torch.int32, device=self.device)


        # Handle edge features (which might be None for some graphs)
        valid_edge_features = [ef for ef in self.edge_features if ef is not None]
        if valid_edge_features:
            # For simplicity, we'll create a communication cost matrix from edge features
            # This assumes edge_features contains communication costs
            self.comm_cost_mat_batch = torch.zeros_like(self.adj_matrices_batch, dtype=torch.float32)
            for i, (adj, edge_feat) in enumerate(zip(self.adj_matrices, self.edge_features)):
                if edge_feat is not None:
                    # Create a matrix where each edge has its communication cost
                    # Use the original (unpadded) adjacency matrix to get edge indices
                    edge_indices = torch.nonzero(adj)
                    for j, (src, dst) in enumerate(edge_indices):
                        if j < edge_feat.shape[1]:  # Ensure we don't go out of bounds
                            self.comm_cost_mat_batch[i, src, dst] = edge_feat[0, j]
        else:
            # If no edge features, use adjacency matrix with unit costs
            self.comm_cost_mat_batch = self.adj_matrices_batch.float()



        # Extract SW and HW costs from node features
        # Assuming node_features[0] is SW cost and node_features[1] is HW cost
        self.sw_cost_vec_batch = self.node_features_batch[:, 0, :]  # Shape: [batch_size, max_nodes]
        self.hw_area_vec_batch = self.node_features_batch[:, 1, :]  # Shape: [batch_size, max_nodes]
        self.hw_cost_vec_batch = self.node_features_batch[:, 2, :]  # Shape: [batch_size, max_nodes]


        # Note: N_op_batch was already set above with the true number of nodes for each graph

        # Initialize dynamic features
        self.batch_idxes = torch.arange(self.batch_size, device=self.device)
        self.current_time_batch = torch.zeros(self.batch_size, device=self.device)

        # Initialize operation status
        self.op_status_batch = torch.zeros(
            (self.batch_size, self.N_op_batch.max().item()),
            dtype=torch.int32, device=self.device
        )

        # Set operations with no predecessors to READY
        for i in range(self.batch_size):
            num_ops = self.N_op_batch[i].item()
            # Find operations with no predecessors (column sum of adjacency matrix is 0)
            no_pred_mask = (self.adj_matrices_batch[i, :num_ops, :num_ops].sum(dim=0) == 0)
            self.op_status_batch[i, :num_ops][no_pred_mask] = READY

        # Initialize number of predecessors remaining
        self.n_pred_remaining_batch = torch.zeros_like(self.op_status_batch, dtype=torch.int32)
        for i in range(self.batch_size):
            num_ops = self.N_op_batch[i].item()
            # Count predecessors for each operation
            self.n_pred_remaining_batch[i, :num_ops] = self.adj_matrices_batch[i, :num_ops, :num_ops].sum(dim=0).int()

        # Initialize running adjacency matrix (copy of original)
        self.running_op_adj_mat_batch = self.adj_matrices_batch.clone()

        # Initialize operation start and end times
        self.op_start_time_batch = torch.zeros(
            (self.batch_size, self.N_op_batch.max().item()),
            dtype=torch.float32, device=self.device
        )

        self.op_end_time_batch = torch.zeros_like(self.op_start_time_batch)

        # Estimate initial end times based on SW costs
        for i in range(self.batch_size):
            num_ops = self.N_op_batch[i].item()
            self.op_end_time_batch[i, :num_ops] = self.sw_cost_vec_batch[i, :num_ops]

        # Get a matrix of indicators whether the node exists (for heterogeneous graph sizes)
        # TODO: there's no corresponding state variable for this yet
        self.op_exists_batch = torch.zeros_like(self.op_start_time_batch).bool()
        for i in range(self.batch_size):
            num_ops = self.N_op_batch[i].item()
            self.op_exists_batch[i, :num_ops] = True


        # Initialize HW usage and remaining area
        self.hw_usage_batch = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.hw_area_remaining_batch = self.hw_area_limits_batch.clone()  # Initially, all HW area is available

        # Initialize resource assignment for operations
        self.op_resource_batch = torch.full(
            (self.batch_size, self.N_op_batch.max().item()),
            -1,  # -1: not scheduled, 0: SW, 1: HW
            dtype=torch.int32,  #dtype=torch.int8,
            device=self.device
        )

        # Initialize SW status
        self.sw_busy_batch = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.sw_end_time_batch = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        # Initialize makespan (max end time of any operation)
        self.makespan_batch = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        # Initialize done flag
        self.done_batch = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # Create initial state
        self.state = EnvState(
            N_op=self.N_op_batch,
            op_adj_mat=self.adj_matrices_batch,
            comm_cost_mat=self.comm_cost_mat_batch,
            sw_cost_vec=self.sw_cost_vec_batch,
            hw_area_vec=self.hw_area_vec_batch,
            hw_area_limit=self.hw_area_limits_batch,
            batch_idxes=self.batch_idxes,
            op_status=self.op_status_batch,
            op_start_time=self.op_start_time_batch,
            op_end_time=self.op_end_time_batch,
            n_pred_remaining=self.n_pred_remaining_batch,
            running_op_adj_mat=self.running_op_adj_mat_batch,
            current_time=self.current_time_batch,
            op_resource=self.op_resource_batch,
            hw_usage=self.hw_usage_batch,
            hw_area_remaining=self.hw_area_remaining_batch,
            sw_busy=self.sw_busy_batch,
            sw_end_time=self.sw_end_time_batch,
            makespan=self.makespan_batch,
            done=self.done_batch
        )

        # Save initial state for reset
        self.initial_state = copy.deepcopy(self.state)
        self.initial_op_status_batch = self.op_status_batch.clone()
        self.initial_n_pred_remaining_batch = self.n_pred_remaining_batch.clone()
        self.initial_running_op_adj_mat_batch = self.running_op_adj_mat_batch.clone()

        # Define action space
        # Action space: (operation_idx, machine_type)
        # machine_type: 0 for SW, 1 for HW
        max_ops = self.N_op_batch.max().item()
        self.action_space = gym.spaces.MultiDiscrete([max_ops, 2])

        # Define observation space (placeholder - will be defined properly based on state)
        self.observation_space = gym.spaces.Dict({
            "N_op": gym.spaces.Box(low=0, high=100, shape=(self.batch_size,), dtype=np.int32),
            "op_status": gym.spaces.Box(low=0, high=3, shape=(self.batch_size, max_ops), dtype=np.int32),
            "current_time": gym.spaces.Box(low=0, high=np.inf, shape=(self.batch_size,), dtype=np.float32),
            "hw_usage": gym.spaces.Box(low=0, high=np.inf, shape=(self.batch_size,), dtype=np.float32),
            "sw_busy": gym.spaces.Box(low=0, high=1, shape=(self.batch_size,), dtype=np.bool_),
        })


    def step(self, actions, notdone_idxes, ready_idxes):
        '''
        Environment transition function

        Args:
            actions: Actions to take, shape [batch_size, 2]
                    actions[:, 0] = operation index to schedule
                    actions[:, 1] = machine type (0 for SW, 1 for HW)
        '''


        self._schedule_op_vec(actions, ready_idxes)
        #self._schedule_op_vec(actions, ready_idxes)

        #if self.mode==0:
        #    self._schedule_op(actions, ready_idxes)
        #else:
        #    self._schedule_op_vec(actions, ready_idxes)


        self._advance_time_vec(notdone_idxes)
        #if self.mode==0:
        #    self._advance_time(notdone_idxes)
        #else:
        #    self._advance_time_vec(notdone_idxes)



        #if self.mode==0:
        #    reward_batch = self._calc_reward_and_makespan()
        #else:
        reward_batch = self._calc_reward_and_makespan_vec(notdone_idxes)



        #print("finished one action over batch")

        # Update state
        self.state.update(
            self.batch_idxes,
            self.op_status_batch,
            self.op_start_time_batch,
            self.op_end_time_batch,
            self.n_pred_remaining_batch,
            self.running_op_adj_mat_batch,
            self.current_time_batch,
            self.op_resource_batch,
            self.hw_usage_batch,
            self.hw_area_remaining_batch,
            self.sw_busy_batch,
            self.sw_end_time_batch,
            self.makespan_batch,
            self.done_batch
        )

        # Return state, reward, done, info
        info = {}
        return self.state, reward_batch, self.done_batch, None, info





    def _schedule_op_vec(self, actions, batch_idxes):

        max_ops = self.op_end_time_batch.shape[1]
        active_bat = len(batch_idxes)

        op_idx = actions[batch_idxes, 0]
        machine_type = actions[batch_idxes, 1]
        num_ops = self.N_op_batch[batch_idxes]#.item()
        complete_pred_idx = torch.logical_and( (self.adj_matrices_batch[batch_idxes, :, op_idx] > 0), (self.op_status_batch[batch_idxes, :] == COMPLETE) )
        complete_opposite_pred_idx = torch.logical_and(complete_pred_idx, self.op_resource_batch[batch_idxes, :] != machine_type.unsqueeze(1))    # machine_type for 1 sample is a 1d tensor, vertical 2d reshape before comparing to matrix  is correct and avoids shape mismatch

        pred_relay = torch.zeros(active_bat, max_ops)
        pred_relay[complete_pred_idx] += self.op_end_time_batch[batch_idxes][complete_pred_idx]
        pred_relay[complete_opposite_pred_idx] += torch.gather(self.comm_cost_mat_batch[batch_idxes],2,op_idx.unsqueeze(1).unsqueeze(2).repeat(1,max_ops,1)).squeeze(-1)[complete_opposite_pred_idx]
        max_comm_finish = pred_relay.max(1).values
        start_time = torch.max(self.current_time_batch[batch_idxes], max_comm_finish)

        self.op_status_batch[batch_idxes, op_idx] = IN_PROGRESS
        self.op_resource_batch[batch_idxes,op_idx] = machine_type.int()
        self.op_start_time_batch[batch_idxes, op_idx] = start_time
        #self.op_end_time_batch[batch_idxes, op_idx] = start_time + torch.where(machine_type==0, self.sw_cost_vec_batch[batch_idxes, op_idx], 0.5*self.sw_cost_vec_batch[batch_idxes, op_idx])
        self.op_end_time_batch[batch_idxes, op_idx] = start_time + torch.where(machine_type==0, self.sw_cost_vec_batch[batch_idxes, op_idx], self.hw_cost_vec_batch[batch_idxes, op_idx])
        #start_time + (self.sw_cost_vec_batch[batch_idxes, op_idx] if machine_type == 0 else 0.5*self.sw_cost_vec_batch[b, op_idx])

        self.sw_busy_batch[batch_idxes] = torch.where(machine_type==0, True, self.sw_busy_batch[batch_idxes])
        self.sw_end_time_batch[batch_idxes]  = torch.where(machine_type==0, self.op_end_time_batch[batch_idxes, op_idx], self.sw_end_time_batch[batch_idxes])
        self.hw_usage_batch[batch_idxes] = torch.where(machine_type!=0, self.hw_usage_batch[batch_idxes] + self.hw_area_vec_batch[batch_idxes, op_idx], self.hw_usage_batch[batch_idxes]  )
        self.hw_area_remaining_batch[batch_idxes] = torch.where(machine_type==1, self.hw_area_remaining_batch[batch_idxes] - self.hw_area_vec_batch[batch_idxes, op_idx], self.hw_area_remaining_batch[batch_idxes]  )




    def _schedule_op(self, actions, batch_idxes):

        # Process each instance in the batch
        for b in batch_idxes: #range(self.batch_size):


            # Extract action components
            op_idx = actions[b, 0]
            machine_type = actions[b, 1]  # 0 for SW, 1 for HW
            num_ops = self.N_op_batch[b].item()
            # This block vectorizes the green block above
            complete_pred_idx = torch.logical_and( (self.adj_matrices_batch[b, :, op_idx] > 0), (self.op_status_batch[b, :] == COMPLETE) )
            complete_opposite_pred_idx = torch.logical_and(complete_pred_idx, self.op_resource_batch[b, :] != machine_type)
            # Time when each predecessor is complete and has finished communicating
            pred_relay = torch.zeros(len(self.op_end_time_batch[b]))
            pred_relay[complete_pred_idx] = self.op_end_time_batch[b, complete_pred_idx]
            pred_relay[complete_opposite_pred_idx] += self.comm_cost_mat_batch[b, complete_opposite_pred_idx, op_idx]
            max_comm_finish = pred_relay.max()

            # Add communication cost to start time
            #start_time += max_comm_cost        # Max start time of the predecessors, plus their max comm cost across partition
            start_time = torch.max(self.current_time_batch[b], max_comm_finish)

            self.op_status_batch[b, op_idx] = IN_PROGRESS         # NOTE: Important - operation is considered IN_PROGRESS while waiting for comm costs
            self.op_resource_batch[b, op_idx] = machine_type  # 0 for SW
            self.op_start_time_batch[b, op_idx] = start_time
            self.op_end_time_batch[b, op_idx] = start_time + (self.sw_cost_vec_batch[b, op_idx] if machine_type == 0 else  self.hw_cost_vec_batch[b, op_idx])
            #self.op_end_time_batch[b, op_idx] = start_time + (self.sw_cost_vec_batch[b, op_idx] if machine_type == 0 else  0.5*self.sw_cost_vec_batch[b, op_idx])


        for b in batch_idxes:
            op_idx = actions[b, 0]
            machine_type = actions[b, 1]  # 0 for SW, 1 for HW

            hw_area = self.hw_area_vec_batch[b, op_idx].item()

            # Update SW/HW resource variables
            if machine_type == 0:  # SW
                # Mark SW as busy
                self.sw_busy_batch[b] = True          # NOTE: Important - CPU/SW is considered BUSY while waiting for comm costs
                self.sw_end_time_batch[b] = self.op_end_time_batch[b, op_idx]
            else:
                # Update HW usage and remaining area
                self.hw_usage_batch[b] += hw_area
                self.hw_area_remaining_batch[b] -= hw_area




    def _advance_time_vec(self, batch_idxes):
        '''
        Advance time to the next event for each instance in the batch
        '''

        #print("in vec")

        num_ops = self.N_op_batch[batch_idxes]
        op_exists = self.op_exists_batch[batch_idxes, :]

        # purpose is to determine if there are any ready ops
        not_ready_bool  = torch.logical_and( self.op_status_batch[batch_idxes, :] != READY, op_exists)
        some_ready_bool = torch.logical_not( not_ready_bool.sum(1)==num_ops )

        #if self.timestep_trigger == "SW":
        #    if (not self.sw_busy_batch[b]) and some_ready:
        #        print("Sample {}: SW not yet scheduled, not advancing time".format(b))
        #        continue

        in_progress_bool = (self.op_status_batch[batch_idxes, :] == IN_PROGRESS)
        in_progress_bool = torch.logical_and(in_progress_bool, op_exists)

        # indicate batch samples that will not advance time because SW is not yet scheduled
        sw_idle_bool = torch.logical_not(self.sw_busy_batch[batch_idxes])
        sw_idle_other_ready_bool = torch.logical_and(sw_idle_bool, some_ready_bool)
        # indicate batch samples that will not advance time because no ops are in progress
        none_in_progress_bool = torch.logical_not(in_progress_bool.any(1))
        no_advance_bool = torch.logical_or(sw_idle_other_ready_bool, none_in_progress_bool)
        time_advance_bool = torch.logical_not(no_advance_bool)

        #print("no_advance_bool")
        #print( no_advance_bool )
        #print("time_advance_bool")
        #print( time_advance_bool )
        #print("batch_idxes[time_advance_bool]")
        #print( batch_idxes[time_advance_bool] )
        #input("waiting")

        # Filter batch_idxes that access self vars, and necessary temp vars
        batch_idxes = batch_idxes[time_advance_bool]
        in_progress_bool = in_progress_bool[time_advance_bool]
        op_exists = op_exists[time_advance_bool]


        #print("in_progress_bool")  # verified
        #print( in_progress_bool )

        sw_ready_time = self.sw_end_time_batch[batch_idxes]
        next_op_ready_time = ( self.op_end_time_batch[batch_idxes, :]+99999.0*torch.logical_not(in_progress_bool).float() ).min(1).values    # zero-out the values for nodes that aren't in progress or don't exist - consistent with taking min()
        next_event_time = torch.max(sw_ready_time, next_op_ready_time)
        #print("next_op_ready_time")
        #print( next_op_ready_time )
        self.current_time_batch[batch_idxes] = next_event_time



        # Update status of completed operations              #:num_ops
        passed_end_time_bool = (self.op_end_time_batch[batch_idxes, :] <= self.current_time_batch[batch_idxes].unsqueeze(1))
        passed_end_time_bool = torch.logical_and(passed_end_time_bool, op_exists)

        # indicates ops that were completed this step
        completed_bool = torch.logical_and(passed_end_time_bool, in_progress_bool)


        #print("passed_end_time_bool")
        #print( passed_end_time_bool )
        #print("in_progress_bool")
        #print( in_progress_bool )


        #print("self.op_end_time_batch[batch_idxes]")
        #print( self.op_end_time_batch[batch_idxes] )
        #print("self.current_time_batch[batch_idxes]")
        #print( self.current_time_batch[batch_idxes] )

        #print("in_progress_bool")   # verified
        #print( in_progress_bool )
        #print("passed_end_time_bool")
        #print( passed_end_time_bool )
        #print("completed_bool")
        #print( completed_bool )



        # Update operation status
        #print("completed_bool")
        #print( completed_bool )
        #print("self.op_status_batch")
        #print( self.op_status_batch )
        #print("completed_bool.shape")
        #print( completed_bool.shape )
        #print("self.op_status_batch.shape")
        #print( self.op_status_batch.shape )
        #self.op_status_batch[batch_idxes,completed_bool] = COMPLETE
        self.op_status_batch[batch_idxes] = torch.where(completed_bool, COMPLETE, self.op_status_batch[batch_idxes])

        #print("self.op_status_batch[batch_idxes]")   # verified
        #print( self.op_status_batch[batch_idxes] )


        sw_end_passed_bool = self.sw_end_time_batch[batch_idxes] <= self.current_time_batch[batch_idxes]
        set_sw_free_bool   = torch.logical_and(sw_end_passed_bool, self.sw_busy_batch[batch_idxes])
        #self.sw_busy_batch[batch_idxes] = False
        self.sw_busy_batch[batch_idxes] = torch.where(set_sw_free_bool, False, self.sw_busy_batch[batch_idxes])



        # Update the running adjacency matrix by removing edges from completed operations
        # should be setting the whole row to zero for each op in completed_bool
        self.running_op_adj_mat_batch[batch_idxes][completed_bool, :] = 0   #TODO: likely wrong, fix. maybe need another where statement


        #for b in batch_idxes:
        #    for op_idx in torch.nonzero(completed_mask).squeeze(-1):
        #        # Remove outgoing edges from completed operation
        #        self.running_op_adj_mat_batch[b, op_idx, :] = 0



        #first get the successor indices of each completed op
        # row ind is completed op idx
        # column holds 0-1 indicating successor
        # sum across column is how many predecessors have just been completed
        # subtract this from n_pred_remaining of each succ
        # note - all potential succs are represented in the matrix, but only completed predecessors. This might clarify the point of summing over column
        completed_succs = self.adj_matrices_batch[batch_idxes]*completed_bool.unsqueeze(2).float()   # unsqueeze(2) vertically orients the rows of completed_bool to broadcast their elements to the rows of adj_mat
        #completed_succs = self.adj_matrices_batch[batch_idxes, completed_bool, :]   #right idea but cannot return a matrix this way, accessing by indicators, think hetergen size, use multiplicative masking before sum instead to sum over right indices
        n_pred_decrement = completed_succs.sum(1)
        self.n_pred_remaining_batch[batch_idxes, :] -= n_pred_decrement.int()

        no_pred_remain_bool = self.n_pred_remaining_batch[batch_idxes]==0
        no_pred_remain_bool = torch.logical_and(no_pred_remain_bool, op_exists)

        promote_to_ready_bool = torch.logical_and(no_pred_remain_bool, self.op_status_batch[batch_idxes, :] == NOT_READY)

        #print("self.n_pred_remaining_batch[batch_idxes]")
        #print( self.n_pred_remaining_batch[batch_idxes] )

        #print("no_pred_remain_bool")
        #print( no_pred_remain_bool )

        #print("\n\n\n\nvectorized")
        #print("no_pred_remain_bool")
        #print( no_pred_remain_bool )
        #print("self.op_status_batch (before)")
        #print( self.op_status_batch )
        #print("promote_to_ready_bool")
        #print( promote_to_ready_bool )



        self.op_status_batch[batch_idxes] = torch.where(promote_to_ready_bool, READY, self.op_status_batch[batch_idxes])









        """
        for op_idx in torch.nonzero(completed_mask).squeeze(-1):
            # Remove outgoing edges from completed operation
            self.running_op_adj_mat_batch[b, op_idx, :] = 0

            # Update predecessors remaining count for successor operations
            for succ_idx in range(num_ops):
                if self.adj_matrices_batch[b, op_idx, succ_idx] > 0:
                    self.n_pred_remaining_batch[b, succ_idx] -= 1

                    # If all predecessors are complete, mark as READY
                    if self.n_pred_remaining_batch[b, succ_idx] == 0 and \
                       self.op_status_batch[b, succ_idx] == NOT_READY:
                        self.op_status_batch[b, succ_idx] = READY
        """




    def _advance_time(self, batch_idxes):
        '''
        Advance time to the next event for each instance in the batch
        '''

        # TODO: sanity check - make sure that CPU is BUSY at this time, since we'll be advancing.  NOTE SW is considered busy when waiting for comms from HW
        #    Q: what if the CPU has to wait for a communication delay from HW
        #    A: it's fine, as long as the HW op is finished, because the SW op start_time will simply add the comm cost
        #print("in original")
        for b in batch_idxes:   #range(self.batch_size):
            # Skip if this instance is already done
            #if self.done_batch[b]:
            #    continue

            # If SW is not busy, don't advance time - UNLESS no ops are ready
            num_ops = self.N_op_batch[b].item()
            not_ready_mask = torch.nonzero(self.op_status_batch[b, :num_ops] != READY).squeeze(-1)
            some_ready = not (len(not_ready_mask) == num_ops)
            if self.timestep_trigger == "SW":
                if (not self.sw_busy_batch[b]) and some_ready:
                    print("Sample {}: SW not yet scheduled, not advancing time".format(b))    # TODO: remove comment and let print
                    continue

            num_ops = self.N_op_batch[b].item()
            in_progress_mask = (self.op_status_batch[b, :num_ops] == IN_PROGRESS)

            # If no operations are in progress, time doesn't advance
            if not in_progress_mask.any():
                print("Sample {}: WARNING no operations in progress in _advance_time".format(b))
                continue

            # If any ops in this sample are still READY, don't advance time
            # TODO: NEW, check this
            #if (self.op_status_batch[b, :num_ops] == READY).any():
            #    print("Sample {}: Operations still READY, not ## time".format(b))
            #    print("self.op_status_batch[b, :num_ops]")
            #    print( self.op_status_batch[b, :num_ops] )
            #    continue

            #print("in_progress_mask")
            #print( in_progress_mask )

            # Get the minimum end time of in-progress operations
            #next_event_time = self.op_end_time_batch[b, :num_ops][in_progress_mask].min()


            # Minimum time advance: go to end time of the SW resource
            sw_ready_time = self.sw_end_time_batch[b]
            next_op_ready_time = self.op_end_time_batch[b, :num_ops][in_progress_mask].min()
            #print("next_op_ready_time")
            #print( next_op_ready_time )
            if self.timestep_mode   == "next_complete":
                next_event_time = next_op_ready_time
                #print("Advancing sample {} to next ready: {}".format(b,next_event_time))   # TODO: rm comment and let print
            elif self.timestep_mode == "next_complete_plus_SW":
                next_event_time = torch.max(sw_ready_time, next_op_ready_time)
                #print("Advancing sample {} to next ready plus SW: {}.    SW: {}   next:{}".format(b,next_event_time,next_op_ready_time,sw_ready_time))   # TODO: rm comment and let print
            else:
                input("Error: Invalid timestep mode specified in env_paras {}".format(self.timestep_mode))   # TODO: rm comment and let print


            #next_event_time = self.op_end_time_batch[b, :num_ops][in_progress_mask].min()

            # Advance until SW is ready and some op is ready
            self.current_time_batch[b] = next_event_time


            #print("in_progress_mask")  # NOTE we're after the continue
            #print( in_progress_mask )


            # Update status of completed operations
            completed_mask = (self.op_end_time_batch[b, :num_ops] <= self.current_time_batch[b]) & \
                            (self.op_status_batch[b, :num_ops] == IN_PROGRESS)




            # Update operation status
            self.op_status_batch[b, :num_ops][completed_mask] = COMPLETE



            # Free resources for completed operations
            for op_idx in torch.nonzero(completed_mask).squeeze(-1):
                # If operation was on SW, mark SW as available
                if self.sw_end_time_batch[b] <= self.current_time_batch[b] and self.sw_busy_batch[b]:     # New: both conditions should be ensured at this point
                    self.sw_busy_batch[b] = False
                # HW usage is permanent and never decremented


            # Update the running adjacency matrix by removing edges from completed operations
            for op_idx in torch.nonzero(completed_mask).squeeze(-1):
                # Remove outgoing edges from completed operation
                self.running_op_adj_mat_batch[b, op_idx, :] = 0

                # Update predecessors remaining count for successor operations
                for succ_idx in range(num_ops):
                    if self.adj_matrices_batch[b, op_idx, succ_idx] > 0:
                        self.n_pred_remaining_batch[b, succ_idx] -= 1

                        # If all predecessors are complete, mark as READY
                        if self.n_pred_remaining_batch[b, succ_idx] == 0 and \
                           self.op_status_batch[b, succ_idx] == NOT_READY:
                            self.op_status_batch[b, succ_idx] = READY



    # TODO: sparse_reward option has been removed from this
    def _calc_reward_and_makespan_vec(self, batch_idxes):

        num_ops = self.N_op_batch[batch_idxes]
        op_exists = self.op_exists_batch[batch_idxes, :]
        reward_batch = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        not_ready_bool  = torch.logical_and( self.op_status_batch[batch_idxes, :] != READY, op_exists)
        some_ready_bool = torch.logical_not( not_ready_bool.sum(1)==num_ops )

        num_ops = self.N_op_batch[batch_idxes]
        op_exists = self.op_exists_batch[batch_idxes, :]
        completed_ops = torch.logical_and(self.op_status_batch[batch_idxes, :] == COMPLETE, op_exists)
        some_complete = completed_ops.any(1)
        new_makespan = (self.op_end_time_batch[batch_idxes, :]*completed_ops.float()).max(1).values       # Nonexisting nodes should have 0, max over them should be fine
        reward_batch[batch_idxes]  = torch.where(some_complete, (-new_makespan) - (-self.makespan_batch[batch_idxes]), reward_batch[batch_idxes])
        self.makespan_batch[batch_idxes] = torch.where(some_complete, new_makespan, self.makespan_batch[batch_idxes])
        completed_if_exists = torch.logical_or(self.op_status_batch[batch_idxes, :] == COMPLETE, torch.logical_not(op_exists))
        self.done_batch[batch_idxes] = completed_if_exists.all(1)

        return reward_batch


    def _calc_reward_and_makespan(self):

        # Update makespan and check if done
        # Initialize reward and info dictionaries
        reward_batch = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        for b in range(self.batch_size):
            if self.done_batch[b]:
                continue

            # Update makespan (max end time of any operation)
            num_ops = self.N_op_batch[b].item()
            completed_ops = (self.op_status_batch[b, :num_ops] == COMPLETE)     # TODO: does self.op_status_batch==COMPLETE get set in _advance_time?
            if completed_ops.any():
                new_makespan = self.op_end_time_batch[b, :num_ops][completed_ops].max()
                if not self.sparse_reward:
                    reward_batch[b] += (-new_makespan) - (-self.makespan_batch[b])   # negative makespan increment: these sum to the negative makespan when gamma=1
                self.makespan_batch[b] = new_makespan



            # Check if all operations are complete
            self.done_batch[b] = (self.op_status_batch[b, :num_ops] == COMPLETE).all()

            # Calculate reward (negative makespan)
            if self.done_batch[b]:
                #print("element {} is done after!!".format(b))
                #input("waiting")
                if self.sparse_reward:
                    reward_batch[b] = -self.makespan_batch[b]

        return reward_batch


    def reset(self):
        '''
        Reset the environment to its initial state

        Returns:
            state: Initial state
        '''
        # Reset all dynamic features to their initial values
        self.batch_idxes = torch.arange(self.batch_size, device=self.device)
        self.current_time_batch = torch.zeros(self.batch_size, device=self.device)
        self.op_status_batch = self.initial_op_status_batch.clone()
        self.n_pred_remaining_batch = self.initial_n_pred_remaining_batch.clone()
        self.running_op_adj_mat_batch = self.initial_running_op_adj_mat_batch.clone()

        self.op_start_time_batch = torch.zeros(
            (self.batch_size, self.N_op_batch.max().item()),
            dtype=torch.float32, device=self.device
        )

        self.op_end_time_batch = torch.zeros_like(self.op_start_time_batch)

        # Estimate initial end times based on SW costs
        for i in range(self.batch_size):
            num_ops = self.N_op_batch[i].item()
            self.op_end_time_batch[i, :num_ops] = self.sw_cost_vec_batch[i, :num_ops]

        self.hw_usage_batch = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        self.hw_area_remaining_batch = self.hw_area_limits_batch.clone()  # Reset remaining area to full capacity
        self.op_resource_batch = torch.full(
            (self.batch_size, self.N_op_batch.max().item()),
            -1,  # -1: not scheduled, 0: SW, 1: HW
            dtype=torch.int32, #dtype=torch.int8,
            device=self.device
        )
        self.sw_busy_batch = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.sw_end_time_batch = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.makespan_batch = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.done_batch = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # Update state
        self.state = EnvState(
            N_op=self.N_op_batch,
            op_adj_mat=self.adj_matrices_batch,
            comm_cost_mat=self.comm_cost_mat_batch,
            sw_cost_vec=self.sw_cost_vec_batch,
            hw_cost_vec=self.hw_cost_vec_batch,
            hw_area_vec=self.hw_area_vec_batch,
            hw_area_limit=self.hw_area_limits_batch,
            batch_idxes=self.batch_idxes,
            op_status=self.op_status_batch,
            op_start_time=self.op_start_time_batch,
            op_end_time=self.op_end_time_batch,
            n_pred_remaining=self.n_pred_remaining_batch,
            running_op_adj_mat=self.running_op_adj_mat_batch,
            current_time=self.current_time_batch,
            op_resource=self.op_resource_batch,
            hw_usage=self.hw_usage_batch,
            hw_area_remaining=self.hw_area_remaining_batch,
            sw_busy=self.sw_busy_batch,
            sw_end_time=self.sw_end_time_batch,
            makespan=self.makespan_batch,
            done=self.done_batch
        )
        return self.state


    def render(self, batch_idx=0, show=True):
        '''
        Render the environment as a DAG with node and edge features and a Gantt chart

        Args:
            batch_idx: Index of the batch to render
            show: Whether to display the plot
        '''
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import to_rgba
        import numpy as np

        # Check if batch_idx is valid
        if batch_idx >= self.batch_size:
            print(f"Error: batch_idx {batch_idx} is out of range (batch_size: {self.batch_size})")
            return

        # Get the number of operations for this graph
        num_ops = self.N_op_batch[batch_idx].item()

        # Create a NetworkX DiGraph from the adjacency matrix
        G = nx.DiGraph()
        G.add_nodes_from(range(num_ops))

        # Add edges from the adjacency matrix
        adj_matrix = self.adj_matrices_batch[batch_idx, :num_ops, :num_ops].cpu().numpy()
        for i in range(num_ops):
            for j in range(num_ops):
                if adj_matrix[i, j] > 0:
                    G.add_edge(i, j)

        # Create a figure with three subplots: DAG, state info, and Gantt chart
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # DAG
        ax2 = fig.add_subplot(gs[0, 1])  # State info
        ax3 = fig.add_subplot(gs[1, :])  # Gantt chart (spans both columns)

        # First subplot: Graph topology with node status
        ax1.set_title(f"Task Graph (DAG) - Batch {batch_idx}")

        # Position nodes using hierarchical layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            try:
                pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
            except:
                pos = nx.spring_layout(G)

        # Define colors for different operation statuses and resources
        # Resource-based color scheme:
        # - Not assigned: grayscale colors
        # - SW: blue-based colors
        # - HW: red-based colors
        resource_status_colors = {
            # Not assigned (-1)
            (-1, NOT_READY): 'lightgray',
            (-1, READY): 'darkgray',
            (-1, IN_PROGRESS): 'gray',
            (-1, COMPLETE): 'dimgray',

            # SW (0) - blue-based colors
            (0, NOT_READY): 'lightblue',
            (0, READY): 'skyblue',
            (0, IN_PROGRESS): 'dodgerblue',
            (0, COMPLETE): 'royalblue',

            # HW (1) - red-based colors
            (1, NOT_READY): 'mistyrose',
            (1, READY): 'lightcoral',
            (1, IN_PROGRESS): 'indianred',
            (1, COMPLETE): 'firebrick'
        }

        # Get node colors based on resource assignment and operation status
        node_colors = []
        for i in range(num_ops):
            status = self.op_status_batch[batch_idx, i].item()
            resource = self.op_resource_batch[batch_idx, i].item()
            node_colors.append(resource_status_colors.get((resource, status), 'white'))

        # Draw nodes with colors based on status
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=500, node_color=node_colors)

        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax1, arrowsize=20, width=1.5)

        # Create node labels with features and status
        labels = {}
        for i in range(num_ops):
            sw_cost = self.sw_cost_vec_batch[batch_idx, i].item()
            hw_area = self.hw_area_vec_batch[batch_idx, i].item()
            status = self.op_status_batch[batch_idx, i].item()
            status_str = ["NOT_READY", "READY", "IN_PROGRESS", "COMPLETE"][status]
            start_time = self.op_start_time_batch[batch_idx, i].item()

            # Add resource assignment to label
            resource = self.op_resource_batch[batch_idx, i].item()
            resource_str = "Not Assigned" if resource == -1 else "SW" if resource == 0 else "HW"

            labels[i] = f"{i}\nSW: {sw_cost:.2f}\nHW: {hw_area:.2f}\n{status_str}\nStart: {start_time:.2f}\nResource: {resource_str}"

        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax1, font_size=8)

        # Draw edge labels with communication costs
        edge_labels = {}
        for i, j in G.edges():
            comm_cost = self.comm_cost_mat_batch[batch_idx, i, j].item()
            edge_labels[(i, j)] = f"{comm_cost:.2f}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1, font_size=8)

        # Add legend for node colors
        legend_elements = []

        # Not assigned nodes
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=resource_status_colors.get((-1, READY), 'darkgray'),
                                         label="Not Assigned - READY", markersize=10))

        # SW nodes
        for status, label in [(READY, "READY"), (IN_PROGRESS, "IN_PROGRESS"), (COMPLETE, "COMPLETE")]:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=resource_status_colors.get((0, status), 'white'),
                                             label=f"SW - {label}", markersize=10))

        # HW nodes
        for status, label in [(READY, "READY"), (IN_PROGRESS, "IN_PROGRESS"), (COMPLETE, "COMPLETE")]:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=resource_status_colors.get((1, status), 'white'),
                                             label=f"HW - {label}", markersize=10))

        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

        ax1.axis('off')

        # Second subplot: State information
        ax2.axis('off')
        ax2.set_title(f"Environment State - Batch {batch_idx}")

        # Create text for state information
        hw_area_limit = self.hw_area_limits_batch[batch_idx].item()
        hw_usage = self.hw_usage_batch[batch_idx].item()
        current_time = self.current_time_batch[batch_idx].item()
        makespan = self.makespan_batch[batch_idx].item()
        sw_busy = self.sw_busy_batch[batch_idx].item()
        sw_end_time = self.sw_end_time_batch[batch_idx].item()

        ready_ops = (self.op_status_batch[batch_idx, :num_ops] == READY).sum().item()
        in_progress_ops = (self.op_status_batch[batch_idx, :num_ops] == IN_PROGRESS).sum().item()
        complete_ops = (self.op_status_batch[batch_idx, :num_ops] == COMPLETE).sum().item()

        # Count operations by resource
        sw_ops = (self.op_resource_batch[batch_idx, :num_ops] == 0).sum().item()
        hw_ops = (self.op_resource_batch[batch_idx, :num_ops] == 1).sum().item()
        not_assigned = (self.op_resource_batch[batch_idx, :num_ops] == -1).sum().item()

        state_text = (
            f"Number of Operations: {num_ops}\n\n"
            f"HW Area Limit: {hw_area_limit:.2f}\n"
            f"Current HW Usage: {hw_usage:.2f} ({hw_usage/hw_area_limit*100:.1f}%)\n"
            f"HW Area Remaining: {self.hw_area_remaining_batch[batch_idx].item():.2f} ({self.hw_area_remaining_batch[batch_idx].item()/hw_area_limit*100:.1f}%)\n\n"
            f"SW Status: {'Busy' if sw_busy else 'Available'}\n"
            f"SW End Time: {sw_end_time:.2f}\n\n"
            f"Current Time: {current_time:.2f}\n"
            f"Current Makespan: {makespan:.2f}\n\n"
            f"Resource Assignment:\n"
            f"  - SW: {sw_ops}\n"
            f"  - HW: {hw_ops}\n"
            f"  - Not Assigned: {not_assigned}\n\n"
            f"Operation Status:\n"
            f"  - Ready: {ready_ops}\n"
            f"  - In Progress: {in_progress_ops}\n"
            f"  - Complete: {complete_ops}\n"
            f"  - Not Ready: {num_ops - ready_ops - in_progress_ops - complete_ops}\n\n"
            f"Predecessors Remaining:\n"
        )

        # Add information about predecessors remaining for each operation
        for i in range(min(num_ops, 10)):  # Limit to first 10 operations to avoid clutter
            pred_remaining = self.n_pred_remaining_batch[batch_idx, i].item()
            state_text += f"  - Op {i}: {pred_remaining}\n"

        if num_ops > 10:
            state_text += f"  - ... ({num_ops - 10} more operations)"

        ax2.text(0.05, 0.95, state_text, verticalalignment='top', fontsize=10)

        # Third subplot: Gantt chart
        ax3.set_title(f"Gantt Chart - Batch {batch_idx}")

        # Set up the Gantt chart
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Resource Lanes')

        # Find operations that have been scheduled (status IN_PROGRESS or COMPLETE)
        scheduled_ops = []
        for i in range(num_ops):
            status = self.op_status_batch[batch_idx, i].item()
            resource = self.op_resource_batch[batch_idx, i].item()

            # Only include operations that have been assigned to a resource
            if resource != -1:
                start_time = self.op_start_time_batch[batch_idx, i].item()
                end_time = self.op_end_time_batch[batch_idx, i].item()
                scheduled_ops.append({
                    'op_idx': i,
                    'resource': resource,  # 0 for SW, 1 for HW
                    'status': status,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })

        # Sort operations by start time
        scheduled_ops.sort(key=lambda x: x['start_time'])

        # Separate SW and HW operations
        sw_ops = [op for op in scheduled_ops if op['resource'] == 0]
        hw_ops = [op for op in scheduled_ops if op['resource'] == 1]

        # For SW operations, we have a single lane (sequential execution)
        sw_lane = 0

        # For HW operations, we need to assign lanes to handle parallel execution
        hw_lanes = []  # List of lists, each inner list represents operations in one lane

        # Assign HW operations to lanes
        for op in hw_ops:
            # Try to find an existing lane where this operation doesn't overlap with any other
            assigned = False
            for lane_idx, lane in enumerate(hw_lanes):
                # Check if this operation overlaps with any operation in this lane
                overlap = False
                for lane_op in lane:
                    # Check for overlap: op1 starts before op2 ends AND op1 ends after op2 starts
                    if (op['start_time'] < lane_op['end_time'] and
                        op['end_time'] > lane_op['start_time']):
                        overlap = True
                        break

                if not overlap:
                    # No overlap, can assign to this lane
                    lane.append(op)
                    op['lane'] = lane_idx + 2  # +2 because SW is lane 0 and COMM is lane 1
                    assigned = True
                    break

            if not assigned:
                # Create a new lane
                hw_lanes.append([op])
                op['lane'] = len(hw_lanes) + 1  # +1 because COMM is lane 1

        # Calculate total number of lanes
        total_lanes = 2 + len(hw_lanes)  # 1 SW lane + 1 COMM lane + HW lanes

        # Set y-axis limits and labels
        ax3.set_ylim(-0.5, total_lanes - 0.5)

        # Create y-tick labels
        y_labels = ['SW', 'COMM']  # SW lane is 0, COMM lane is 1
        for i in range(len(hw_lanes)):
            y_labels.append(f'HW {i+1}')

        ax3.set_yticks(range(total_lanes))
        ax3.set_yticklabels(y_labels)

        # Add grid lines
        ax3.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Draw operations as horizontal bars
        for op in scheduled_ops:
            if op['resource'] == 0:  # SW
                lane = 0
            else:  # HW
                lane = op['lane']

            # Get color based on status
            status = op['status']
            resource = op['resource']
            color = resource_status_colors.get((resource, status), 'white')

            # Create rectangle for this operation
            rect = patches.Rectangle(
                (op['start_time'], lane - 0.4),  # (x, y)
                op['duration'],  # width
                0.8,  # height
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.8
            )

            # Add the rectangle to the plot
            ax3.add_patch(rect)

            # Add operation index as text
            ax3.text(
                op['start_time'] + op['duration'] / 2,  # x position (center of bar)
                lane,  # y position
                f"{op['op_idx']}",  # text (operation index)
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold'
            )

        # Draw communication time bars
        for op in scheduled_ops:
            # Find predecessors of this operation
            for pred_op in scheduled_ops:
                # Check if pred_op is a predecessor of op
                if self.adj_matrices_batch[batch_idx, pred_op['op_idx'], op['op_idx']] > 0:
                    # If resources differ, there's communication cost
                    if pred_op['resource'] != op['resource']:
                        comm_cost = self.comm_cost_mat_batch[batch_idx, pred_op['op_idx'], op['op_idx']].item()

                        if comm_cost > 0:
                            # Get base color for the predecessor operation
                            pred_status = pred_op['status']
                            pred_resource = pred_op['resource']
                            pred_color = resource_status_colors.get((pred_resource, pred_status), 'white')

                            # All communication times go in the COMM lane (lane 1)
                            comm_lane = 1  # COMM lane

                            # Determine start time based on direction
                            if pred_op['resource'] == 1 and op['resource'] == 0:
                                # HW to SW communication
                                start_time = pred_op['end_time']
                            elif pred_op['resource'] == 0 and op['resource'] == 1:
                                # SW to HW communication
                                start_time = pred_op['end_time']
                            else:
                                # Should not happen, but skip if it does
                                continue

                            # Create and add communication rectangle (slightly skinnier than operation bars)
                            comm_rect = patches.Rectangle(
                                (start_time, comm_lane - 0.35),  # (x, y) - adjusted to center the bar
                                comm_cost,  # width
                                0.7,  # height - between original 0.8 and skinnier 0.6
                                linewidth=1,
                                edgecolor='black',
                                facecolor=pred_color,
                                alpha=0.3,  # Transparent
                                hatch='///'  # Hatched pattern
                            )
                            ax3.add_patch(comm_rect)

                            # Add sending and receiving nodes as text if there's enough space
                            if comm_cost > 0.5:  # Only add text if bar is wide enough
                                ax3.text(
                                    start_time + comm_cost / 2,  # x position (center of bar)
                                    comm_lane,  # y position
                                    f"{pred_op['op_idx']}→{op['op_idx']}",  # text (sending→receiving nodes)
                                    ha='center',
                                    va='center',
                                    fontsize=8,
                                    alpha=0.7
                                )

        # Add a vertical line for current time
        if current_time > 0:
            ax3.axvline(x=current_time, color='green', linestyle='-', linewidth=2, label='Current Time')
            ax3.text(current_time, total_lanes - 0.5, f"Current: {current_time:.2f}",
                    ha='right', va='top', color='green', fontsize=10, rotation=90)

        # Set x-axis limits with some padding
        if scheduled_ops:
            max_end_time = max(op['end_time'] for op in scheduled_ops)
            ax3.set_xlim(-0.1, max(current_time, max_end_time) * 1.1)
        else:
            ax3.set_xlim(-0.1, max(1, current_time) * 1.1)

        # Add legend for the Gantt chart
        gantt_legend_elements = []

        # Add legend elements for operation status (COMPLETE only)
        # SW - COMPLETE
        gantt_legend_elements.append(patches.Patch(
            facecolor=resource_status_colors.get((0, COMPLETE), 'white'),
            edgecolor='black',
            label=f"SW - COMPLETE"
        ))
        # HW - COMPLETE
        gantt_legend_elements.append(patches.Patch(
            facecolor=resource_status_colors.get((1, COMPLETE), 'white'),
            edgecolor='black',
            label=f"HW - COMPLETE"
        ))

        # Add communication bars to legend
        gantt_legend_elements.append(patches.Patch(
            facecolor=resource_status_colors.get((1, COMPLETE), 'white'),
            edgecolor='black',
            alpha=0.3,
            hatch='///',
            label="HW → SW Communication"
        ))
        gantt_legend_elements.append(patches.Patch(
            facecolor=resource_status_colors.get((0, COMPLETE), 'white'),
            edgecolor='black',
            alpha=0.3,
            hatch='///',
            label="SW → HW Communication"
        ))

        # Add current time to legend
        gantt_legend_elements.append(plt.Line2D([0], [0], color='green', lw=2, label='Current Time'))

        ax3.legend(handles=gantt_legend_elements, loc='upper right', fontsize=8)

        if show==True:
            plt.tight_layout()
            plt.show()

        return fig


    def close(self):
        '''
        Clean up resources
        '''
        pass


# Example usage
if __name__ == "__main__":
    print("Creating Task Graph Dataset...")
    # Create a dataset with more varied graphs
    dataset = TaskGraphDataset(
        num_samples=5,
        min_nodes=5,
        max_nodes=15,
        edge_probability=0.3
    )

    print("Initializing Computational Scheduling Environment...")
    # Create environment parameters
    env_paras = {
        "batch_size": 2,
        "device": "cpu",
        "timestep_mode":"next_complete",  # Skip to the time the nearest op completion time
        "timestep_trigger":"every"
    }
    # "timestep_mode"    : "next_complete"          Skip to the nearest op completion time
    #                      "next_complete_plus_SW"  Skip to the nearest op completion time after SW is cleared  (skipping to SW clear leads to deadlock)

    # "timestep_trigger" : "every"               Trigger a timestep after any operation is scheduled
    #                    : "SW"                  Trigger a timestep after an operation is scheduled to software


    #                      "all_ready"         Schedule all ready ops while they remain - Is this a mode or a trigger?



    # Create environment
    env = CSchedEnv(dataset, env_paras)

    # Test the step function with random actions
    print("\nTesting step function with random actions...")
    state = env.reset()

    # Run a few steps
    for i in range(5):
        print(f"\nStep {i+1}")

        # Generate random actions
        actions = torch.zeros((env.batch_size, 2), dtype=torch.int64)
        for b in range(env.batch_size):
            # Find ready operations
            num_ops = env.N_op_batch[b].item()
            ready_ops = torch.nonzero(env.op_status_batch[b, :num_ops] == READY).squeeze(-1)

            if len(ready_ops) > 0:
                # Randomly select a ready operation
                op_idx = ready_ops[torch.randint(0, len(ready_ops), (1,))].item()

                # Randomly select machine type (0 for SW, 1 for HW)
                machine_type = torch.randint(0, 2, (1,)).item()

                actions[b, 0] = op_idx
                actions[b, 1] = machine_type

                print(f"  Batch {b}: Scheduling operation {op_idx} on {'SW' if machine_type == 0 else 'HW'}")

        # Take a step
        state, reward, done, _, info = env.step(actions)

    # Print state information
    print(f"\nEnvironment Information:")
    print(f"Batch size: {env.batch_size}")
    print(f"Number of operations per graph: {env.N_op_batch}")
    print(f"HW area limits: {env.hw_area_limits_batch}")

    # Print detailed information about each graph in the batch
    for i in range(env.batch_size):
        num_ops = env.N_op_batch[i].item()
        print(f"\n{'='*50}")
        print(f"Graph {i} Details:")
        print(f"{'='*50}")
        print(f"Operations: {num_ops}")
        print(f"Edges: {env.adj_matrices_batch[i, :num_ops, :num_ops].sum().item()}")
        print(f"HW Area Limit: {env.hw_area_limits_batch[i].item():.2f}")

        # Operation status counts
        ready_ops = (env.op_status_batch[i, :num_ops] == READY).sum().item()
        in_progress_ops = (env.op_status_batch[i, :num_ops] == IN_PROGRESS).sum().item()
        complete_ops = (env.op_status_batch[i, :num_ops] == COMPLETE).sum().item()
        not_ready_ops = num_ops - ready_ops - in_progress_ops - complete_ops

        print(f"\nOperation Status:")
        print(f"  - Ready: {ready_ops}")
        print(f"  - In Progress: {in_progress_ops}")
        print(f"  - Complete: {complete_ops}")
        print(f"  - Not Ready: {not_ready_ops}")

        print(f"\nNode Features:")
        print(f"  - SW costs (min/max/avg): {env.sw_cost_vec_batch[i, :num_ops].min().item():.2f} / "
              f"{env.sw_cost_vec_batch[i, :num_ops].max().item():.2f} / "
              f"{env.sw_cost_vec_batch[i, :num_ops].mean().item():.2f}")
        print(f"  - HW costs (min/max/avg): {env.hw_area_vec_batch[i, :num_ops].min().item():.2f} / "
              f"{env.hw_area_vec_batch[i, :num_ops].max().item():.2f} / "
              f"{env.hw_area_vec_batch[i, :num_ops].mean().item():.2f}")

        print(f"\nPredecessors Remaining:")
        for j in range(min(num_ops, 5)):  # Show first 5 operations
            print(f"  - Op {j}: {env.n_pred_remaining_batch[i, j].item()}")
        if num_ops > 5:
            print(f"  - ... ({num_ops - 5} more operations)")

    # Render each graph in the batch
    print("\nRendering graphs...")
    for i in range(env.batch_size):
        print(f"\nRendering Graph {i}...")
        env.render(batch_idx=i)












    # Advance time to next event for each instance
    # Current issue: Scheduled ops to IN_PROGRESS trigger time advance before next action, preventing parallelism
    # Propose: only advance if there are no operations READY
    # Issue with that: Forces scheduling to HW whenever possible, taking away agent choice (schedules that save HW for later are impossible)
    # New proposal: advance time after each action BUT only if SW is busy
    #     Pros: - Allows both choices. If clearing all READY ops is needed, just do all HW assignments before scheduling one to SW   -   BUT what if you put all READY ops in HW by chance - impossible to advance time and complete the episode
    #                                  If scheduling to SW and moving on is needed, just schedule to SW first. BUT scheduling to HW and moving on is not possible.
    #           - Aligns with rule of thumb that SW idle time should be minimized (can't advance time without scheduling SW )
    #     Cons: -
    # Addressing the BUTS:
    #           - Q: What if no READY ops remain but SW is idle - deadlock
    #           - A: Revised advance_time condition: advance if SW is BUSY OR if no READY ops remain
    #           - Q: Scheduling to HW and moving
    #

    # Issue: possible to put everything in HW, causing deadlock

    #
    # Second Issue: how to revise the use of advance_time when
    # Observe: Now we MUST advance time until SW is not busy, since otherwise next decision time will have SW busy and all READY ops will be expended to HW (a likely case since HW is faster)
    # Downside: Lost opportunity to schedule more ops in HW before SW op is finished (those ops would need all their predecessors to be on HW for this to be the case)
    # Effectively, the resulting policy will schedule to SW at each step, after optionally scheduling other READY ops to HW, before skipping to next SW READY time.


    # Two options:
    #     - Prevent scheduling the last READY operation to HW (HW is infeasible if only one op ready and SW not busy)
    #     -
