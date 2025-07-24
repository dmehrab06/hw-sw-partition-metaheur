import os
import yaml
from pathlib import Path

def load_base_config(config_file="../configs/config_default.yaml"):
    """Load base configuration from YAML file"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Default config file '{config_file}' not found!")
    
    with open(config_file, 'r') as f:
        base_config = yaml.safe_load(f)
    
    print(f"Loaded base configuration from: {config_file}")
    return base_config

def generate_config_filename(graph_dir, graph_name, area_constraint, hw_scale_variance, comm_scale_factor):
    """Generate a descriptive filename for the config"""
    # Extract directory name for filename (remove trailing slash and path)
    dir_name = Path(graph_dir.rstrip('/')).name
    
    # Remove .dot extension from graph name
    graph_base = Path(graph_name).stem
    
    filename = (f"config_default_{dir_name}_{graph_base}_"
               f"area-{area_constraint:.1f}_"
               f"hwvar-{hw_scale_variance:.1f}_"
               f"comm-{comm_scale_factor:.1f}.yaml")
    
    return filename

def generate_all_configs(default_config_file="../configs/config_default.yaml"):
    """Generate all configuration files based on parameter combinations"""
    
    # Load base configuration from YAML file
    try:
        base_config = load_base_config(default_config_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create a config_default.yaml file with the base configuration.")
        return 0
    
    # Parameter combinations from your original script
    graph_dirs = [
        'soda-benchmark-graphs/pytorch-graphs/',
        'soda-benchmark-graphs/tflite-graphs/',
        'test-data/'
    ]
    
    graph_files = [
        ['mobile_net_tosa.dot', 'rez_net_tosa.dot', 'squeeze_net_tosa.dot'],
        ['anomaly_detection_tosa.dot', 'image_classification_tosa.dot',
         'visual_wake_words_tosa.dot', 'keyword_spotting_tosa.dot'],
        ['01_tosa.dot']
    ]
    
    area_constraints = [0.1, 0.5, 0.9]
    hw_scale_variances = [0.5, 1, 5]  # l parameter
    comm_scale_factors = [0.5, 1, 5]  # mu parameter
    
    # Create output directory for config files
    config_dir = "../configs/"
    os.makedirs(config_dir, exist_ok=True)

    graph_parent_dir = "/people/mehr668/encode_scripts/hw_sw_partition_min_cut/"
    
    total_configs = 0
    
    # Generate configs for all combinations
    for gdiridx, gdir in enumerate(graph_dirs):
        for graph_name in graph_files[gdiridx]:
            for area_constraint in area_constraints:
                for hw_scale_variance in hw_scale_variances:
                    for comm_scale_factor in comm_scale_factors:
                        
                        # Create a copy of the base config for this combination
                        config = base_config.copy()
                        
                        # Set the variable parameters
                        config['graph-file'] = f"{graph_parent_dir}{gdir}{graph_name}"
                        config['area-constraint'] = area_constraint
                        config['hw-scale-variance'] = hw_scale_variance
                        config['comm-scale-factor'] = comm_scale_factor
                        
                        # Generate filename
                        filename = generate_config_filename(
                            gdir, graph_name, area_constraint, 
                            hw_scale_variance, comm_scale_factor
                        )
                        
                        # Full path for the config file
                        config_path = os.path.join(config_dir, filename)
                        
                        # Write config to YAML file
                        with open(config_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        
                        total_configs += 1
                        
                        # Print progress (optional)
                        if total_configs % 50 == 0:
                            print(f"Generated {total_configs} config files...")
    
    print(f"\nCompleted! Generated {total_configs} configuration files in '{config_dir}' directory")
    
    # Print some example filenames
    print("\nExample generated files:")
    example_files = os.listdir(config_dir)[:5]  # Show first 5 files
    for example in example_files:
        print(f"  - {example}")
    
    return total_configs

if __name__ == "__main__":
    print("Generating configuration files...")
    print("=" * 50)
    
    # Validate default config first
    # if not validate_default_config():
    #     print("Please fix the config_default.yaml file before proceeding.")
    #     exit(1)
    
    total = generate_all_configs()

    config_files = os.listdir('../configs/')  # Show first 5 files
    for config in config_files:
        print(f"sbatch meta_heuristic_eval.sbatch  configs/{config}")