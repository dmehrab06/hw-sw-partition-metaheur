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

## GCPS method ([https://link.springer.com/article/10.1007/s10617-021-09255-9])

```mermaid
flowchart LR
  A[Input DAG and task attributes]
  A --> B[Preprocess: normalized adjacency and HG HGP features]
  B --> C[Two layer GCN model]
  C --> D[Pretraining with auxiliary Tcost]
  D --> E[Training loop]
  E --> F[Predict HW probability per task]
  F --> G[Greedy decode under area constraint]
  G --> H[LSSP scheduling]
  H --> I{Schedule improved}
  I -- Yes --> J[Update best partition]
  J --> E
  I -- No --> K{Stop condition met}
  K -- No --> E
  K -- Yes --> L[Final partition and final schedule]

```

## DIFF-GNN (without ordering)-ours

```mermaid
flowchart LR
  subgraph T["diff_gnn training"]
    IG[Input DAG graph] --> FEAT[Features: sw_time, hw_time, area, optional HG and HGP]
    FEAT --> GNN[Diff GNN encoder with optional edge_mlp]
    GNN --> HEAD[Placement head: HW probability per node]
    HEAD --> LOSS[Differentiable loss: makespan surrogate, area, regularizers]
    LOSS -.->|backprop| GNN
  end

  subgraph I["diff_gnn postprocess and inference"]
    IG --> FEATI[Same feature build]
    FEATI --> GNNI[Trained Diff GNN]
    GNN -->|trained weights| GNNI
    GNNI --> DEC[Decode to binary HW SW partition]
    DEC --> DLS[DLS refine optional]
    DLS --> LSSP[LSSP local search optional]
    LSSP --> OUT[Final partition and final metric]
  end

  linkStyle 4 stroke:#d62728,stroke-width:2px,stroke-dasharray:6 4,color:#d62728

```

## DIFF-GNN-Order (with ordering)


```mermaid
flowchart LR
  subgraph T["diff_gnn_order training"]
    IG[Input DAG graph] --> FEAT[Features: sw_time, hw_time, area, optional HG and HGP]
    FEAT --> ENC[Diff GNN Order encoder with optional edge_mlp]
    ENC --> PH[Placement head: HW probability]
    ENC --> OH[Ordering heads: prio_hw and prio_sw]
    PH --> OLOSS[Order aware differentiable loss: makespan, area, ordering terms]
    OH --> OLOSS
    OLOSS -.->|backprop| ENC
  end

  subgraph I["diff_gnn_order postprocess and inference"]
    IG --> FEATI[Same feature build]
    FEATI --> ENCI[Trained Diff GNN Order]
    ENC -->|trained weights| ENCI
    ENCI --> ODEC[Order aware decode combines placement plus prio_hw and prio_sw]
    ODEC --> DLS[DLS refine optional]
    DLS --> LSSP[LSSP local search optional]
    LSSP --> OUT[Final ordered partition and final metric]
  end

  linkStyle 6 stroke:#d62728,stroke-width:2px,stroke-dasharray:6 4,color:#d62728

```

## DIFF-GNN-Order (with ordering details)

```mermaid
flowchart LR
  subgraph T["diff_gnn_order training detail"]
    IG[Input DAG graph] --> FEAT[Feature build]
    FEAT --> ENC[Graph encoder with optional edge_mlp]

    ENC --> PHEAD[Placement head]
    ENC --> OHW[Order head hw priority]
    ENC --> OSW[Order head sw priority]

    PHEAD --> ASSIGN[Relaxed HW probability assignment]
    OHW --> SINK[Differentiable ordering via sinkhorn]
    OSW --> SINK
    SINK --> PAIR[Pairwise ranking consistency]
    ASSIGN --> PAIR

    ASSIGN --> LOSS[Schedule surrogate and area loss]
    PAIR --> LOSS
    LOSS -.->|backprop| ENC
  end

  subgraph I["diff_gnn_order inference and postprocess detail"]
    IG2[Input DAG graph] --> FEAT2[Feature build]
    FEAT2 --> ENCI[Trained diff_gnn_order]
    ENC -->|trained weights| ENCI

    ENCI --> PHEADI[Placement output]
    ENCI --> OHWI[HW priority output]
    ENCI --> OSWI[SW priority output]

    PHEADI --> HARD[Hard decode HW SW]
    OHWI --> ODEC[Order aware decode score]
    OSWI --> ODEC
    HARD --> ODEC

    ODEC --> AREA[Area repair and fill]
    AREA --> DLS[DLS refine optional]
    DLS --> LSSP[LSSP local search optional]
    LSSP --> OUT[Final ordered partition and metric]
  end

  linkStyle 12 stroke:#d62728,stroke-width:2px,stroke-dasharray:6 4,color:#d62728

```

### DIFF-GNN-Order (with details)

```mermaid
flowchart TD
  G["Input DAG and node features"]

  subgraph N["diff_gnn_order network"]
    H["Graph encoder with optional edge_mlp"]
    P["Placement logits"]
    OH["HW order scores"]
    OS["SW order scores"]
    G --> H
    H --> P
    H --> OH
    H --> OS
  end

  P --> X["HW probability x_i via relaxed binary"]
  OH --> SINK["Sinkhorn soft permutation"]
  OS --> SINK
  SINK --> B["Soft precedence weights B(j to i)"]

  subgraph D["DAG dependency clock"]
    P1["Pred arrival 1"]
    P2["Pred arrival 2"]
    Pk["..."]
    DMAX["dep_ready_i = soft max of predecessor arrivals"]
    P1 --> DMAX
    P2 --> DMAX
    Pk --> DMAX
  end

  subgraph Q["Order aware queue clock"]
    J1["Same lane finish 1"]
    J2["Same lane finish 2"]
    BQ["B(j to i) from soft order"]
    QMAX["queue_ready_i = weighted soft max with B"]
    J1 --> QMAX
    J2 --> QMAX
    BQ --> QMAX
  end

  B --> BQ
  DMAX --> S["start_i = soft max(dep_ready_i, queue_ready_i)"]
  QMAX --> S

  X --> DUR["duration_i from HW SW probability"]
  S --> F["finish_i = start_i + duration_i"]
  DUR --> F

  F --> M["soft makespan = soft max over all finishes"]
  X --> L["Total loss = makespan + area + partition + ordering regularizers"]
  SINK --> L
  M --> L

  L -.->|backprop| H

  M --> DEC["Hard decode to HW SW partition"]
  DEC --> POST["Optional DLS and optional LSSP"]
  POST --> OUT["Final ordered partition and final metric"]

```