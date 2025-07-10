# Task Graph Partitioning Evaluation

A Python tool for evaluating task graph partitioning using Particle Swarm Optimization (PSO) algorithms.

## Description

This tool analyzes and evaluates task graph partitioning solutions by applying hardware and communication scaling factors while respecting area constraints. 

## Usage

Please change the directory structure in `TaskGraph.py` according to your local before running anything.

```bash
python pso_eval.py --graph-file <GRAPH_FILE> --area-constraint <AREA_CONSTRAINT> --hw-scale-factor <HW_SCALE_FACTOR> --hw-scale-variance <HW_SCALE_VARIANCE> --comm-scale-factor <COMM_SCALE_FACTOR> [OPTIONS]
```

## Required Arguments

- `--graph-file GRAPH_FILE`: Path to the graph file to be loaded and analyzed, currently supports .dot file only. Files can be found in the min_cut repository
- `--area-constraint AREA_CONSTRAINT`: Area constraint value (must be between 0 and 1)
- `--hw-scale-factor HW_SCALE_FACTOR`: Hardware scaling factor (must be positive, ideally less than 1)
- `--hw-scale-variance HW_SCALE_VARIANCE`: Hardware scaling variance (must be positive)
- `--comm-scale-factor COMM_SCALE_FACTOR`: Communication scaling factor (must be positive)

## Optional Arguments

- `--verbose, -v`: Enable verbose output (can be used multiple times: -v, -vv, -vvv for increasing verbosity levels)
- `--seed SEED`: Set random seed for reproducible results
- `--log-dir LOG_DIR`: Specify directory for log file storage (default: `logs`)
- `--help, -h`: Display help message and exit

## Examples

### Basic Usage
```bash
python pso_eval.py --graph-file test-data/01_tosa.dot --area-constraint 0.8 --hw-scale-factor 0.5 --hw-scale-variance 0.1 --comm-scale-factor 1.2
```

### With Verbose Output and Custom Seed
```bash
python pso_eval.py --graph-file test-data/01_tosa.dot --area-constraint 0.8 --hw-scale-factor 0.5 --hw-scale-variance 0.1 --comm-scale-factor 1.2 --verbose --seed 42
```

## Requirements

- Python 3.x
- Required dependencies (see requirements.txt)

## Output

The tool generates evaluation results and stores log files in the specified log directory (default: `logs/`). Use the `--verbose` flag to see detailed progress information during execution.

## Notes

- Ensure that the area constraint is a value between 0 and 1
- All scaling factors and variance values must be positive numbers
- Hardware scaling factors should be ideally less than 1, assuming Hardware tasks are faster in general
- For reproducible results across runs, use the `--seed` parameter with a fixed value

## Acknowledgments

This README was generated with assistance from Claude AI.
