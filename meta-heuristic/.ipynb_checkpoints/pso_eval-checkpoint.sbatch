#!/bin/bash
#SBATCH --job-name=hw-sw
#SBATCH --output=slurm_logs/hw-sw-%j.out
#SBATCH --error=slurm_logs/hw-sw-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Load required modules

#module purge
#module load pnnl_env

module load python/miniconda4.12
source /share/apps/python/miniconda4.12/etc/profile.d/conda.sh

conda activate hw_sw_partition

python pso_eval.py --graph-file $1 --area-constraint $2 --hw-scale-factor $3 --hw-scale-variance $4 --comm-scale-factor $5 --verbose --seed $6
