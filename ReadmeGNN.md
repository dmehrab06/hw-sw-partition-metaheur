module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate combopt

cd /people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/
cd /people/dass304/dass304/HWSWpartition/hw-sw-partitioning/

Current environments:
combopt              * /people/dass304/.conda/envs/combopt
py312                  /people/dass304/.conda/envs/py312
pygeo                  /people/dass304/.conda/envs/pygeo
base                   /share/apps/python/miniconda25.5.1

in HPC system:



Running: 


```bash
python meta_heuristic_main.py --config /people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/configs/config_mkspan_default.yaml
```

```bash
python meta_heuristic_main.py --config /people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/configs/config_mkspan_area_0.1_hw_0.1_seed_1.yaml

python gnn_main.py --config /people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/configs/config_mkspan_area_0.1_hw_0.1_seed_1.yaml
```

```bash
python gnn_main.py --config /people/dass304/dass304/HWSWpartition/hw-sw-partition-metaheur/configs/config_mkspan_default.yaml

python gnn_main.py --config configs/config_mkspan_default.yaml

python gnn_main.py --config configs/config_mkspan_default_gnn.yaml
/people/dass304/.conda/envs/combopt/bin/python gnn_main.py --config configs/config_mkspan_default_gnn.yaml

```
`conda install -y pytorch=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia`
`pip install torchvision torchaudio`

```python -c "import torch, torchvision, torchaudio; print(torch.__version__, torch.version.cuda, torchvision.__version__, torchaudio.__version__, torch.cuda.is_available())"```

Installed: ```2.6.0+cu124 12.4 0.21.0+cu124 2.6.0+cu124 True```

`pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html`


## Run diff_gnn + diff_gnn_order and print visualization paths

```bash
HWSW_METHODS="diff_gnn,diff_gnn_order" /people/dass304/.conda/envs/combopt/bin/python gnn_main.py -c configs/config_mkspan_default_gnn.yaml

echo "[viz] saved under: outputs/final_visualizations/mkspan_default"
find outputs/final_visualizations/mkspan_default -type f -name "*.png" | sort

HWSW_METHODS="diff_gnn,diff_gnn_order" /people/dass304/.conda/envs/combopt/bin/python gnn_main.py -c configs/config_fig3_taskgraph_gnn.yaml

echo "[viz] saved under: outputs/final_visualizations/fig3"
find outputs/final_visualizations/fig3 -type f -name "*.png" | sort
```



# Mermaid diagram

## without ordering

```mermaid
flowchart TD
    A[simulate_diff_GNN] --> B[Read TG + diffgnn config defaults]
    B --> C[Select device]
    C --> D[optimize_diff_gnn]
    D --> E[Build graph data + placement model]

    E --> F[Training loop]
    F --> F1[Forward pass -> logits]
    F1 --> F2[Relaxed binary assignment]
    F2 --> F3[Optional: paper_sigma blend + DLS refine]
    F3 --> F4[Differentiable makespan loss]
    F4 --> F5[Backprop + Adam]

    F5 --> G{Hard eval step?}
    G -- Yes --> H[Hard decode + repair/fill]
    H --> I[Optional LSSP local search]
    I --> J[Evaluate train metric]
    J --> K[Track best assignment]
    G -- No --> F
    K --> F

    F --> L[Final hard decode]
    L --> M[Repair/fill + optional final LSSP]
    M --> N[Evaluate final metric]
    N --> O[Return best_assign + best_cost]
    O --> P[Wrapper: to sol_arr + final repair]
    P --> Q[Output: best_cost, sol_arr]

```
## with ordering 

```mermaid
flowchart TD
    A[simulate_diff_GNN_order] --> B[Read TG + diffgnn_order defaults]
    B --> C[Select device]
    C --> D[optimize_diff_gnn_order]
    D --> E[Build graph data + order model]

    E --> F[Training loop]
    F --> F1[Forward -> logits + prio_hw + prio_sw]
    F1 --> F2[Relaxed binary assignment]
    F2 --> F3[Optional: paper_sigma blend + DLS refine]
    F3 --> F4[Order-aware differentiable loss]
    F4 --> F5[Includes sinkhorn/gumbel/pairwise/perm terms]
    F5 --> F6[Backprop + Adam]

    F6 --> G{Hard eval step?}
    G -- Yes --> H[Order-aware decode score]
    H --> I[Repair/fill using decode score]
    I --> J[Optional LSSP local search]
    J --> K[Evaluate train metric]
    K --> L[Track best assignment]
    G -- No --> F
    L --> F

    F --> M[Final hard decode]
    M --> N[Order-aware repair/fill + optional final LSSP]
    N --> O[Evaluate final metric]
    O --> P[Return best_assign + best_cost]
    P --> Q[Wrapper: to sol_arr + final repair]
    Q --> R[Output: best_cost, sol_arr]

```