# usage in deception

Deception reference:

```
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate combopt
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
cd /people/dass304/dass304/HWSWpartition/hw-sw-partitioning/GNN/copt-main
```

if conda is not pointing to correct python do the following:

```
conda deactivate
conda deactivate
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate /people/dass304/.conda/envs/combopt
which python
```

Dir: `/people/dass304/dass304/HWSWpartition/hw-sw-partitioning/GNN/copt-main`

Commands to install if requires:

```
conda env create -f sid_environment.yml
conda activate combopt
```

# Running copt

`python main.py --cfg configs/benchmarks/maxclique/maxclique_rb_small.yaml`

`python main.py --cfg configs/maxcut.yaml`

`python main.py --cfg configs/benchmarks/maxcut/maxcut_ba_small.yaml`


# Packages

`conda install -y pytorch=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia`
`pip install torchvision torchaudio`

```python -c "import torch, torchvision, torchaudio; print(torch.__version__, torch.version.cuda, torchvision.__version__, torchaudio.__version__, torch.cuda.is_available())"```

Installed: ```2.6.0+cu124 12.4 0.21.0+cu124 2.6.0+cu124 True```

`pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html`


```
# First conda updates (safe, no PyTorch reinstall)
conda install -y \
  gurobi=11.0.3 \
  numpy=1.26.4 \
  scipy=1.14.1 \
  scikit-learn=1.5.2 \
  pandas=2.2.3 \
  mkl \
  matplotlib=3.9.2 \
  ipykernel \
  ipython \
  notebook \
  lightning=2.4.0 \
  lightning-utilities=0.11.6 \
  packaging=24.1 \
  typing_extensions=4.12.2 \
  tqdm=4.66.5 \
  -c gurobi -c conda-forge -c defaults

# Then pip installs/updates (skip torch itself)
pip install --upgrade \
  ogb==1.3.6 \
  hydra-core==1.1.0 \
  hydra-submitit-launcher==1.2.0 \
  performer-pytorch==1.1.4 \
  local-attention==1.9.14 \
  axial-positional-embedding==0.2.1 \
  einops==0.8.0 \
  yacs==0.1.8 \
  dimod==0.12.16 \
  dwave-networkx==0.8.15 \
  wandb==0.17.7 \
  visdom==0.2.4 \
  loguru==0.7.2 \
  gputil==1.4.0 \
  submitit==1.5.2 \
  psutil==5.9.0 \
  requests==2.32.3 \
  protobuf==5.27.3 \
  PyYAML==6.0.2 \
  setuptools==72.1.0 \
  wheel==0.44.0

```

Upgrade Git: `conda install -c conda-forge git`


@author Siddhartha
