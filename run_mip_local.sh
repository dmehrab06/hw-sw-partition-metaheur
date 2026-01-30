#!/bin/bash
set -euo pipefail

# ---- fixed user + environment ----
USERNAME="dass304"

# ---- one case to run ----
area="0.1"
hw="0.1"
seed="1"
# config="configs/mip_config/config_mip_area_${area}_hw_${hw}_seed_${seed}.yaml"
# config="/people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/configs/config_mkspan_mip_default.yaml"
config="/people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/configs/config_mkspan_default_gnn.yaml"

echo "Running locally as ${USERNAME}"
echo "Config: ${config}"
echo

# # Avoid picking up user-site packages
# export PYTHONNOUSERSITE=1

# # ---- environment setup ----
# module load python/miniconda25.5.1
# #source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
# conda activate combopt
# source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh


# # Ensure at least one MILP solver is present (install HiGHS/GLPK/CBC if missing)
# python - <<'PY'
# import subprocess, sys
# import cvxpy as cp
# installed = set(cp.installed_solvers())
# needed = []
# if "HIGHS" not in installed:
#     needed.append("highspy")
# if "GLPK_MI" not in installed:
#     needed.append("glpk")
# if "CBC" not in installed:
#     needed.append("coincbc")
# if needed:
#     cmd = ["conda", "install", "-n", "combopt", "-c", "conda-forge", "-y"] + needed
#     print("[setup] Installing solvers:", " ".join(needed))
#     subprocess.check_call(cmd)
# else:
#     print("[setup] MILP solvers already present:", sorted(installed))
# PY

# ---- move to project directory ----
cd /people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/

# ---- run directly in current terminal (no Slurm) ----
python milp_eval.py -c "${config}" -t cvxpy


# python milp_eval.py -c /people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/configs/config_mkspan_default_gnn.yaml -t cvxpy