#!/bin/bash
for area in 0.1 0.3 0.5 0.7 0.9; do
    for hw in 0.1 0.3 0.5 0.7 0.9; do
        for seed in 0 1 2 3; do
            config=configs/mip_config/config_mip_area_${area}_hw_${hw}_seed_${seed}.yaml

            echo "#!/bin/bash">>run_cvxpy.sbatch
            echo "#SBATCH --output='slurm_outputs/cp_%j.out'">>run_cvxpy.sbatch
            echo "#SBATCH --error='slurm_outputs/cp_%j.err'">>run_cvxpy.sbatch
            echo "#SBATCH -A encode_optimize">>run_cvxpy.sbatch
            echo "#SBATCH -p a100_shared">>run_cvxpy.sbatch
            echo "#SBATCH --ntasks-per-node=4">>run_cvxpy.sbatch
            echo "#SBATCH --time=00-00:20:00">>run_cvxpy.sbatch
            echo "#SBATCH --job-name=run_cvxpy_mip">>run_cvxpy.sbatch

            echo "module load python/miniforge25.3.0">>run_cvxpy.sbatch
            echo "source /share/apps/python/miniforge25.3.0/etc/profile.d/conda.sh">>run_cvxpy.sbatch

            echo "conda run -n encode --no-capture-output python milp_eval.py -c ${config} -t cvxpy">>run_cvxpy.sbatch
            sbatch run_cvxpy.sbatch
            rm run_cvxpy.sbatch
        done
    done
done