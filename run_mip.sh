#!/bin/bash
set -euo pipefail

# Local runner based on the working run_mip_local.sh (no Slurm)
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate combopt
export PYTHONNOUSERSITE=1
PYTHON="/people/dass304/.conda/envs/combopt/bin/python"

echo "Running MIP sweeps locally with combopt env"

# for area in 0.1 0.3 0.5 0.7 0.9; do
#   for hw in 0.1 0.3 0.5 0.7 0.9; do
#     for seed in 0 1 2 3; do

for area in 0.1; do
  for hw in 0.1; do
    for seed in 1; do
      config="configs/config_mkspan_area_${area}_hw_${hw}_seed_${seed}.yaml"
      if [[ ! -f "$config" ]]; then
        echo "Config not found: $config (skipping)"
        continue
      fi
      echo "---- area=${area} hw=${hw} seed=${seed} ----"
      echo "Config: $config"
      $PYTHON milp_eval.py -c "$config" -t cvxpy
    done
  done
done

echo "Finished. Outputs:"
echo " - Logs: logs/run_milp_optimizer_area-<area>_hw-<hw>_seed-<seed>.log"
echo " - Partitions: saved in the config 'solution-dir' (e.g., makespan-opt-partitions/ or makespan-mip-opt-partitions/) with filenames like taskgraph-...-assignment-mip.pkl"
