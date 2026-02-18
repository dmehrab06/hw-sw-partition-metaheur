import os, sys
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import argparse

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/test_argument_parser.log")

logger = LogManager.get_logger(__name__)

def parse_arguments() -> DictConfig:
    """
    Parse command line arguments with proper validation.
    Supports both command-line arguments and YAML configuration files.
    Command-line arguments take precedence over YAML configuration.
    
    Returns:
        DictConfig: Configuration object with all parameters
    """
    parser = argparse.ArgumentParser(description='Task Graph Partitioning Evaluation')

    # Add config file argument
    parser.add_argument('-c', '--config', type=str,
                       help='Path to YAML configuration file')
    
    # Original arguments
    parser.add_argument('--graph-file', type=str,
                       help='Graph file name to load')
    parser.add_argument('--area-constraint', type=float,
                       help='Area constraint (should be between 0 and 1)')
    parser.add_argument('--hw-scale-factor', type=float,
                       help='Hardware scale factor (should be positive)')
    parser.add_argument('--hw-scale-variance', type=float,
                       help='Hardware scale variance (should be positive)')
    parser.add_argument('--comm-scale-factor', type=float,
                       help='Communication scale factor (should be positive)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()

    # Load configuration from YAML file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            yaml_config = OmegaConf.load(config_path)
        except Exception as e:
            logger.error(f"Error loading YAML configuration: {e}")
            raise ValueError(f"Error loading YAML configuration: {e}")
    else:
        yaml_config = OmegaConf.create({})

    # Convert command-line arguments to OmegaConf format
    cli_config = OmegaConf.create({
        k.replace('_', '-'): v for k, v in vars(args).items() 
        if v is not None #and k != 'config'
    })
    
    # Merge configurations (CLI arguments override YAML)
    config = OmegaConf.merge(yaml_config, cli_config)

    # Fill default seed if not provided anywhere
    if config.get('seed', None) is None:
        config['seed'] = 42

    # Validate required parameters
    required_params = [
        'graph-file', 'area-constraint', 'hw-scale-factor', 
        'hw-scale-variance', 'comm-scale-factor'
    ]
    
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        logger.error(f"Missing required parameters: {missing_params}")
        raise ValueError(f"Missing required parameters: {missing_params}")
    
    # Validate parameter ranges
    validate_config(config)

    # Log the input parameters
    log_input_parameters(config)
    
    return config

def validate_config(config):
    """Validate input arguments with assertions."""
    try:
        assert 0 < config['area-constraint'] <= 1, f"Area constraint must be between 0 and 1, got {config['area-constraint']}"
        assert config['hw-scale-factor'] > 0, f"Hardware scale factor must be positive, got {config['hw-scale-factor']}"
        assert config['hw-scale-variance'] > 0, f"Hardware scale variance must be positive, got {config['hw-scale-variance']}"
        assert config['comm-scale-factor'] > 0, f"Communication scale factor must be positive, got {config['comm-scale-factor']}"
        
        if config['hw-scale-factor'] > 1:
            logger.warning(f"Hardware scale factor is greater than 1 ({config['hw-scale-factor']}). This might lead to unexpected behavior.")
        
        # Validate graph file exists
        graph_file = Path(config['graph-file'])
        if not graph_file.exists():
            logger.error(f"Graph file not found: {graph_file}")
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        
        # Validate seed is non-negative
        if config.get('seed', 42) < 0:
            logger.error(f"Seed must be non-negative, got: {config['seed']}")
            raise ValueError(f"Seed must be non-negative, got: {config['seed']}")
            
        logger.info("All input arguments validated successfully")
        
    except AssertionError as e:
        logger.error(f"Argument validation failed: {e}")
        sys.exit(1)

def log_input_parameters(config:DictConfig):
    """
    Log all input parameters for reference.
    
    Args:
        config: Parsed command line arguments
        logger: Logger instance
    """
    logger.info("INPUT PARAMETERS:")
    logger.info("-" * 40)
    logger.info(f"Graph file: {config['graph-file']}")
    logger.info(f"Area constraint: {config['area-constraint']}")
    logger.info(f"Hardware scale factor: {config['hw-scale-factor']}")
    logger.info(f"Hardware scale variance: {config['hw-scale-variance']}")
    logger.info(f"Communication scale factor: {config['comm-scale-factor']}")
    logger.info(f"Random seed: {config['seed']}")
    logger.info("-" * 40)

    return

if __name__ == "__main__":
    config = parse_arguments()
