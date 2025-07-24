# Task Graph Partitioning Evaluation

Evaluation of task graph partitioning 

## Mixed Integer Programming (MIP)

WIP

## Vanilla Particle Swarm Optimization (PSO) algorithm

### Description

This pipeline solves task graph hw/sw partitioning while respecting area constraints through various metaheuristics

### Usage

```bash
python meta_heuristic_main.py --config <CONFIG_FILE> 
```

### Currently Implemented MetaHeuristics
- Particle Swarm Optimization (PSO) with Sigmoid Activation
- Discrete Binary Particle Swarm Optimization (DBPSO)
- Comprehensive learning PSO (CLPSO)
- Cooperative Coevolving PSO (CCPSO)
- Global and Local genetic algorithm (GL25)
- Enhanced Simulated Annealing (ESA)
- Success-History based Adaptive DE (SHADE)
- Adaptive DE (JADE)


### Examples

#### Basic Usage
```bash
python pso_eval.py --config configs/config_default.yaml
```

### Requirements

- Python 3.x
- Required dependencies (see requirements-pso.txt)


### Acknowledgments

This README was generated with assistance from Claude AI.
