## imports
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from collections import defaultdict
from utils.scheduler_utils import compute_dag_makespan

def analyze_method_across_parameter(method_name, param_name, param_values, 
                                    fixed_params, seeds, config_type, 
                                    solution_dir, task_graph_dir='inputs/task_graph_complete'):
    """
    Analyze a specific method's performance across different parameter values.
    
    Parameters
    ----------
    method_name : str
        Name of the method to analyze (e.g., 'pso', 'jade', 'greedy')
    param_name : str
        Name of the parameter to vary ('hw' or 'area')
    param_values : list
        List of parameter values to test
    fixed_params : dict
        Dictionary of fixed parameters. Must contain 'hw' and/or 'area'
        depending on which is being varied
    seeds : list of int
        List of seed values to process
    config_type : str
        Configuration type ('arato' or 'mkspan')
    solution_dir : str
        Directory containing solution pickle files
    task_graph_dir : str, optional
        Directory containing task graph pickle files
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'param_values': list of parameter values
        - 'mean_makespans': list of mean makespans per parameter value
        - 'std_makespans': list of std makespans per parameter value
        - 'all_makespans': dict mapping param_value to list of makespans
        - 'naive_means': list of naive baseline means per parameter value
        - 'naive_stds': list of naive baseline stds per parameter value
    """
    
    assert param_name in ['hw', 'area'], "param_name must be 'hw' or 'area'"
    
    # Storage for results at each parameter value
    makespans_by_param = defaultdict(list)
    naive_makespans_by_param = defaultdict(list)
    
    # Process each parameter value
    for param_value in param_values:
        print(f"\n{'='*60}")
        print(f"Processing {param_name} = {param_value}")
        print('='*60)
        
        # Set up parameters for this iteration
        if param_name == 'hw':
            hw = param_value
            area = fixed_params['area']
        else:  # param_name == 'area'
            hw = fixed_params['hw']
            area = param_value
        
        # Process each seed for this parameter value
        for seed in seeds:
            print(f"  Seed {seed}...")
            
            # Load task graph
            task_graph_location = (f'{task_graph_dir}/taskgraph-squeeze_net_tosa-instance-'
                                  f'config-config_{config_type}_area_{area}_hw_{hw}_seed_{seed}.pkl')
            
            try:
                with open(task_graph_location, 'rb') as file:
                    task_graph = pickle.load(file)
            except FileNotFoundError:
                print(f"    Task graph not found, skipping...")
                continue
            
            # Set up graph attributes
            graph = task_graph.graph
            nx.set_node_attributes(graph, task_graph.hardware_area, 'area_cost')
            nx.set_node_attributes(graph, task_graph.hardware_costs, 'hardware_time')
            nx.set_node_attributes(graph, task_graph.software_costs, 'software_time')
            nx.set_edge_attributes(graph, task_graph.communication_costs, 'communication_cost')
            
            # Calculate naive baseline
            naive_partition = {v: 0 for v in graph.nodes()}
            naive_makespan = task_graph.evaluate_makespan(naive_partition)['makespan']
            naive_makespans_by_param[param_value].append(naive_makespan)
            
            # Find solution file for this method
            solution_prefix = (f'taskgraph-squeeze_net_tosa_area-{area:.2f}_hwscale-{hw:.1f}_'
                              f'hwvar-0.50_comm-1.00_seed-{seed}_assignment-{method_name}.pkl')
            
            solution_path = os.path.join(solution_dir, solution_prefix)
            
            if not os.path.exists(solution_path):
                print(f"    Solution file not found: {solution_prefix}")
                continue
            
            # Load and evaluate solution
            with open(solution_path, 'rb') as file:
                partition = pickle.load(file)
                assignment = [1 - partition[k] for k in graph.nodes]
                #makespan, _ = compute_dag_makespan(graph, assignment)
                makespan = task_graph.evaluate_makespan(partition)['makespan']
                makespans_by_param[param_value].append(makespan)
                print(f"    Makespan: {makespan:.2f}")
    
    # Compute statistics for each parameter value
    results = {
        'param_values': [],
        'mean_makespans': [],
        'std_makespans': [],
        'all_makespans': {},
        'naive_means': [],
        'naive_stds': [],
        'all_naive_makespans': {}
    }
    
    for param_value in param_values:
        if param_value in makespans_by_param and len(makespans_by_param[param_value]) > 0:
            makespans = makespans_by_param[param_value]
            naive_makespans = naive_makespans_by_param[param_value]
            
            results['param_values'].append(param_value)
            results['mean_makespans'].append(np.mean(makespans))
            results['std_makespans'].append(np.std(makespans))
            results['all_makespans'][param_value] = makespans
            results['naive_means'].append(np.mean(naive_makespans))
            results['naive_stds'].append(np.std(naive_makespans))
            results['all_naive_makespans'][param_value] = naive_makespans
            
            print(f"\n{param_name}={param_value}:")
            print(f"  {method_name}: {np.mean(makespans):.2f} ± {np.std(makespans):.2f}")
            print(f"  Naive: {np.mean(naive_makespans):.2f} ± {np.std(naive_makespans):.2f}")
    
    return results


def plot_methods_across_parameter(method_names, param_name, param_values,
                                  fixed_params, seeds, config_type,
                                  solution_dir, task_graph_dir='inputs/task_graph_complete',
                                  title=None, figsize=(12, 6), ylim=None, 
                                  show_baseline=True, show_individual_points=True,
                                  save_path=None):
    """
    Plot one or more methods' performance across parameter values.
    
    Parameters
    ----------
    method_names : str or list of str
        Single method name or list of method names to plot
    param_name : str
        Name of the parameter to vary ('hw' or 'area')
    param_values : list
        List of parameter values to test
    fixed_params : dict
        Dictionary of fixed parameters
    seeds : list of int
        List of seed values to process
    config_type : str
        Configuration type ('arato' or 'mkspan')
    solution_dir : str
        Directory containing solution pickle files
    task_graph_dir : str, optional
        Directory containing task graph pickle files
    title : str, optional
        Plot title. If None, auto-generated
    figsize : tuple, optional
        Figure size (width, height)
    ylim : list, optional
        Y-axis limits [ymin, ymax]
    show_baseline : bool, optional
        Whether to show naive baseline
    show_individual_points : bool, optional
        Whether to show individual seed results as scatter points
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects
    results_dict : dict
        Dictionary mapping method names to their results
    """
    
    # Convert single method to list
    if isinstance(method_names, str):
        method_names = [method_names]
    
    # Analyze each method
    results_dict = {}
    for method_name in method_names:
        print(f"\n{'#'*80}")
        print(f"ANALYZING METHOD: {method_name.upper()}")
        print('#'*80)
        
        results = analyze_method_across_parameter(
            method_name=method_name,
            param_name=param_name,
            param_values=param_values,
            fixed_params=fixed_params,
            seeds=seeds,
            config_type=config_type,
            solution_dir=solution_dir,
            task_graph_dir=task_graph_dir
        )
        results_dict[method_name] = results
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette for methods
    colors = sns.color_palette("husl", len(method_names))
    
    # Plot each method
    for idx, method_name in enumerate(method_names):
        results = results_dict[method_name]
        
        if len(results['param_values']) == 0:
            print(f"Warning: No data for method {method_name}")
            continue
        
        param_vals = results['param_values']
        means = results['mean_makespans']
        stds = results['std_makespans']
        
        # Plot line with error bars
        ax.errorbar(param_vals, means, yerr=stds,
                    marker='o', linewidth=2.5, markersize=8, capsize=5,
                    label=method_name.upper(), color=colors[idx],
                    ecolor=colors[idx], alpha=0.8, zorder=3)
        
        # Plot individual points if requested
        if show_individual_points:
            for param_val in param_vals:
                if param_val in results['all_makespans']:
                    y_vals = results['all_makespans'][param_val]
                    x_vals = [param_val] * len(y_vals)
                    ax.scatter(x_vals, y_vals, color=colors[idx], 
                             alpha=0.3, s=40, zorder=2, edgecolors='none')
    
    # Plot baseline if requested (use first method's baseline data)
    if show_baseline and method_names:
        first_results = results_dict[method_names[0]]
        if len(first_results['param_values']) > 0:
            ax.errorbar(first_results['param_values'], 
                       first_results['naive_means'],
                       yerr=first_results['naive_stds'],
                       marker='s', linewidth=2, markersize=8, capsize=5,
                       label='All SW Assignment', color='gray',
                       ecolor='gray', alpha=0.7, linestyle='--', zorder=3)
            
            # Plot individual baseline points if requested
            if show_individual_points:
                for param_val in first_results['param_values']:
                    if param_val in first_results['all_naive_makespans']:
                        y_vals = first_results['all_naive_makespans'][param_val]
                        x_vals = [param_val] * len(y_vals)
                        ax.scatter(x_vals, y_vals, color='gray', 
                                 alpha=0.2, s=40, zorder=1, edgecolors='none')
    
    # Customize plot
    param_label = 'Hardware Scale Factor (hw)' if param_name == 'hw' else 'Area Constraint'
    ax.set_xlabel(param_label, fontsize=13, fontweight='bold')
    ax.set_ylabel('Makespan', fontsize=13, fontweight='bold')
    
    if title is None:
        fixed_str = ', '.join([f"{k}={v}" for k, v in fixed_params.items() if k != param_name])
        if len(method_names) == 1:
            title = f'{method_names[0].upper()} Performance vs {param_name.upper()}'
        else:
            title = f'Method Comparison vs {param_name.upper()}'
        title += f'\n({fixed_str}, Seeds={seeds})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Improve legend
    legend_cols = 2 if len(method_names) > 4 else 1
    ax.legend(loc='best', fontsize=10, ncol=legend_cols, framealpha=0.9)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    fig.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=350, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    return fig, ax, results_dict


def print_comparison_summary(results_dict, param_name):
    """
    Print a summary comparison table of all methods.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method names to their results
    param_name : str
        Name of the parameter being varied
    """
    
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    # Get all unique parameter values across all methods
    all_param_values = set()
    for results in results_dict.values():
        all_param_values.update(results['param_values'])
    all_param_values = sorted(list(all_param_values))
    
    # Print header
    print(f"\n{param_name.upper():>10s} | ", end="")
    for method_name in results_dict.keys():
        print(f"{method_name.upper():>20s} | ", end="")
    print()
    print("-" * 80)
    
    # Print data for each parameter value
    for param_val in all_param_values:
        print(f"{param_val:>10.2f} | ", end="")
        
        for method_name, results in results_dict.items():
            if param_val in results['all_makespans']:
                makespans = results['all_makespans'][param_val]
                mean = np.mean(makespans)
                std = np.std(makespans)
                print(f"{mean:>8.1f} ± {std:>6.1f} | ", end="")
            else:
                print(f"{'N/A':>20s} | ", end="")
        print()
    
    print("="*80)


def load_and_process_results(hw, area, seeds, config_type, solution_dir, 
                             task_graph_dir='inputs/task_graph_complete'):
    """
    Load and process optimization results across multiple seeds.
    
    Parameters
    ----------
    hw : float
        Hardware scale factor
    area : float
        Area constraint
    seeds : list of int
        List of seed values to process
    config_type : str
        Configuration type ('arato' or 'mkspan')
    solution_dir : str
        Directory containing solution pickle files
    task_graph_dir : str, optional
        Directory containing task graph pickle files
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'methods': list of method names
        - 'mean_makespans': list of mean makespans per method
        - 'std_makespans': list of std makespans per method
        - 'naive_mean': mean of naive baseline
        - 'naive_std': std of naive baseline
        - 'results_by_method': full results dict
    """
    
    # Dictionary to store results: {method_name: [makespan1, makespan2, ...]}
    results_by_method = defaultdict(list)
    naive_makespans = []
    
    # Process each seed
    for seed in seeds:
        print(f"\n=== Processing seed {seed} ===")
        
        # Construct task graph location based on config type
        task_graph_location = (f'{task_graph_dir}/taskgraph-squeeze_net_tosa-instance-'
                              f'config-config_{config_type}_area_{area}_hw_{hw}_seed_{seed}.pkl')
        
        try:
            with open(task_graph_location, 'rb') as file:
                task_graph = pickle.load(file)
        except FileNotFoundError:
            print(f"Task graph not found for seed {seed}, skipping...")
            continue
        
        # Set up graph attributes
        graph = task_graph.graph
        nx.set_node_attributes(graph, task_graph.hardware_area, 'area_cost')
        nx.set_node_attributes(graph, task_graph.hardware_costs, 'hardware_time')
        nx.set_node_attributes(graph, task_graph.software_costs, 'software_time')
        nx.set_edge_attributes(graph, task_graph.communication_costs, 'communication_cost')
        
        # Calculate naive (all software) makespan
        naive_partition = {v: 0 for v in graph.nodes()}
        naive_makespan = task_graph.evaluate_makespan(naive_partition)['makespan']
        naive_makespans.append(naive_makespan)
        
        # Process solution files for this seed
        solution_prefix = (f'taskgraph-squeeze_net_tosa_area-{area:.2f}_hwscale-{hw:.1f}_'
                          f'hwvar-0.50_comm-1.00_seed-{seed}_assignment')
        
        for sol in os.listdir(solution_dir):
            if sol.startswith(solution_prefix):
                print(f"  Processing: {sol}")
                
                with open(f'{solution_dir}/{sol}', 'rb') as file:
                    # Extract method name
                    method_name = (sol.split('assignment-')[1]).split('.')[0]
                    
                    # Load partition and compute makespan
                    partition = pickle.load(file)
                    assignment = [1 - partition[k] for k in graph.nodes]
                    #makespan, _ = compute_dag_makespan(graph, assignment)
                    makespan = task_graph.evaluate_makespan(partition)['makespan']
                    # Store result
                    results_by_method[method_name].append(makespan)
    
    # Compute statistics
    methods = []
    mean_makespans = []
    std_makespans = []
    
    for method, makespans in sorted(results_by_method.items()):
        methods.append(method)
        mean_makespans.append(np.mean(makespans))
        std_makespans.append(np.std(makespans))
        print(f"\n{method}:")
        print(f"  Makespans: {makespans}")
        print(f"  Mean: {np.mean(makespans):.2f}, Std: {np.std(makespans):.2f}")
    
    # Calculate statistics for naive baseline
    naive_mean = np.mean(naive_makespans)
    naive_std = np.std(naive_makespans)
    print(f"\nAll SW Assignment:")
    print(f"  Makespans: {naive_makespans}")
    print(f"  Mean: {naive_mean:.2f}, Std: {naive_std:.2f}")
    
    # Find best method
    if mean_makespans:
        best_mean_makespan = min(mean_makespans)
        best_method = methods[mean_makespans.index(best_mean_makespan)]
        print(f"\nBest method: {best_method} with mean makespan: {best_mean_makespan:.2f}")
    
    return {
        'methods': methods,
        'mean_makespans': mean_makespans,
        'std_makespans': std_makespans,
        'naive_mean': naive_mean,
        'naive_std': naive_std,
        'results_by_method': dict(results_by_method),
        'naive_makespans': naive_makespans
    }


def plot_makespan_comparison(results, hw, area, seeds, title, 
                             figsize=(10, 6), ylim=None, save_path=None,show_error=False):
    """
    Create bar plot comparing makespan across methods with error bars.
    
    Parameters
    ----------
    results : dict
        Dictionary returned by load_and_process_results()
    hw : float
        Hardware scale factor
    area : float
        Area constraint
    seeds : list of int
        List of seed values processed
    title : str
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    ylim : list, optional
        Y-axis limits [ymin, ymax]
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects
    """
    
    methods = results['methods']
    mean_makespans = results['mean_makespans']
    std_makespans = results['std_makespans'] if show_error else None
    naive_mean = results['naive_mean']
    naive_std = results['naive_std'] if show_error else 0
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add SW Assignment baseline as shaded rectangle
    ax.axhspan(naive_mean - naive_std, naive_mean + naive_std, 
               color='gray', alpha=0.3, zorder=1,
               label='All SW Assignment')
    
    # Add mean line for SW assignment
    ax.axhline(y=naive_mean, color='gray', linestyle='--', linewidth=2, zorder=2)
    
    # Create bar plot with error bars
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, mean_makespans, yerr=std_makespans, 
                  capsize=5, alpha=0.8, ecolor='black', 
                  color=sns.color_palette("Spectral", len(methods)),
                  zorder=3)
    
    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Makespan', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{title}\n(HW={hw}, Area={area}, Seeds={seeds})', fontsize=13)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    
    fig.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=350, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    return fig, ax


def print_summary_statistics(results, seeds):
    """
    Print summary statistics for the results.
    
    Parameters
    ----------
    results : dict
        Dictionary returned by load_and_process_results()
    seeds : list of int
        List o# Construct task graph location based on config type
        task_graph_location = (f'{task_graph_dir}/taskgraph-squeeze_net_tosa-instance-'
                              f'config-config_{config_type}_area_{area}_hw_{hw}_seed_{seed}.pkl')
        
        try:
            with open(task_graph_location, 'rb') as file:
                task_graph = pickle.load(file)
        except FileNotFoundError:
            print(f"Task graph not found for seed {seed}, skipping...")
            continue
        f seed values processed
    """
    methods = results['methods']
    mean_makespans = results['mean_makespans']
    std_makespans = results['std_makespans']
    
    print("\n" + "="*60)
    print("=== SUMMARY STATISTICS ===")
    print("="*60)
    print(f"Number of seeds processed: {len(seeds)}")
    print(f"Methods found: {len(methods)}")
    print("\nMethod Results (Mean ± Std):")
    print("-" * 60)
    for i, method in enumerate(methods):
        print(f"  {method:20s}: {mean_makespans[i]:8.2f} ± {std_makespans[i]:6.2f}")
    print("="*60)


