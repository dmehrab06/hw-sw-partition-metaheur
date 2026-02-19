import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
except Exception as e:
    raise ImportError("This module requires torch_geometric. Install with 'pip install torch-geometric'.") from e


def _apply_learned_edge_scaling(edge_weight, edge_logits, min_scale: float, max_scale: float):
    """
    Compute effective edge weights from optional base weights and learned per-edge gates.
    Gate range is bounded to [min_scale, max_scale] for stability.
    """
    if edge_logits is None:
        return edge_weight
    scales = min_scale + (max_scale - min_scale) * torch.sigmoid(edge_logits)
    if edge_weight is None:
        return scales
    return edge_weight * scales.to(edge_weight.device, edge_weight.dtype)


def _resolve_edge_weight_learner(edge_weight_learner, learn_edge_weight: bool) -> str:
    learner = edge_weight_learner
    if learner is None:
        learner = "per_edge" if bool(learn_edge_weight) else "none"
    learner = str(learner).lower()
    aliases = {
        "true": "per_edge",
        "false": "none",
        "param": "per_edge",
        "params": "per_edge",
        "parameter": "per_edge",
        "parameterized": "per_edge",
    }
    learner = aliases.get(learner, learner)
    if learner not in {"none", "per_edge", "mlp"}:
        raise ValueError("edge_weight_learner must be one of: none|per_edge|mlp")
    return learner


class DiffGNNPlacement(nn.Module):
    """
    Default placement model used by diff_gnn (backward-compatible behavior).
    Returns two logits per node: [software, hardware].
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        edge_weight_learner=None,
        learn_edge_weight: bool = False,
        num_edges: int = 0,
        edge_attr_dim: int = 0,
        edge_mlp_hidden_dim: int = 16,
        edge_weight_min_scale: float = 0.5,
        edge_weight_max_scale: float = 1.5,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if float(edge_weight_min_scale) > float(edge_weight_max_scale):
            raise ValueError("edge_weight_min_scale must be <= edge_weight_max_scale")

        self.dropout = float(dropout)
        self.edge_weight_learner = _resolve_edge_weight_learner(edge_weight_learner, learn_edge_weight)
        self.learn_edge_weight = self.edge_weight_learner != "none"
        self.edge_weight_min_scale = float(edge_weight_min_scale)
        self.edge_weight_max_scale = float(edge_weight_max_scale)
        if self.edge_weight_learner == "per_edge":
            if int(num_edges) <= 0:
                raise ValueError("num_edges must be > 0 when learn_edge_weight=True")
            self.edge_weight_logits = nn.Parameter(torch.zeros(int(num_edges), dtype=torch.float32))
            self.edge_mlp = None
        elif self.edge_weight_learner == "mlp":
            if int(edge_attr_dim) <= 0:
                raise ValueError("edge_attr_dim must be > 0 when edge_weight_learner='mlp'")
            hidden = max(1, int(edge_mlp_hidden_dim))
            self.edge_mlp = nn.Sequential(
                nn.Linear(int(edge_attr_dim), hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            self.register_parameter("edge_weight_logits", None)
        else:
            self.register_parameter("edge_weight_logits", None)
            self.edge_mlp = None

        convs = []
        if num_layers == 1:
            convs.append(GCNConv(in_channels, hidden_dim // 2))
            last_dim = hidden_dim // 2
        else:
            convs.append(GCNConv(in_channels, hidden_dim))
            for _ in range(num_layers - 2):
                convs.append(GCNConv(hidden_dim, hidden_dim))
            convs.append(GCNConv(hidden_dim, hidden_dim // 2))
            last_dim = hidden_dim // 2

        self.convs = nn.ModuleList(convs)
        self.lin = nn.Linear(last_dim, 1)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        if self.edge_weight_learner == "per_edge":
            edge_logits = self.edge_weight_logits
        elif self.edge_weight_learner == "mlp":
            if edge_attr is None:
                raise ValueError("edge_attr is required when edge_weight_learner='mlp'")
            edge_logits = self.edge_mlp(edge_attr).squeeze(-1)
        else:
            edge_logits = None
        eff_edge_weight = _apply_learned_edge_scaling(
            edge_weight=edge_weight,
            edge_logits=edge_logits,
            min_scale=self.edge_weight_min_scale,
            max_scale=self.edge_weight_max_scale,
        )
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_weight=eff_edge_weight)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.lin(h).squeeze(-1)
        logits2 = torch.stack([-logits, logits], dim=1)
        return logits2


class DiffGNNOrder(nn.Module):
    """
    Default ordered model used by diff_gnn_order (backward-compatible behavior).
    Returns placement logits + HW/SW priorities.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        edge_weight_learner=None,
        learn_edge_weight: bool = False,
        num_edges: int = 0,
        edge_attr_dim: int = 0,
        edge_mlp_hidden_dim: int = 16,
        edge_weight_min_scale: float = 0.5,
        edge_weight_max_scale: float = 1.5,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if float(edge_weight_min_scale) > float(edge_weight_max_scale):
            raise ValueError("edge_weight_min_scale must be <= edge_weight_max_scale")

        self.dropout = float(dropout)
        self.edge_weight_learner = _resolve_edge_weight_learner(edge_weight_learner, learn_edge_weight)
        self.learn_edge_weight = self.edge_weight_learner != "none"
        self.edge_weight_min_scale = float(edge_weight_min_scale)
        self.edge_weight_max_scale = float(edge_weight_max_scale)
        if self.edge_weight_learner == "per_edge":
            if int(num_edges) <= 0:
                raise ValueError("num_edges must be > 0 when learn_edge_weight=True")
            self.edge_weight_logits = nn.Parameter(torch.zeros(int(num_edges), dtype=torch.float32))
            self.edge_mlp = None
        elif self.edge_weight_learner == "mlp":
            if int(edge_attr_dim) <= 0:
                raise ValueError("edge_attr_dim must be > 0 when edge_weight_learner='mlp'")
            hidden = max(1, int(edge_mlp_hidden_dim))
            self.edge_mlp = nn.Sequential(
                nn.Linear(int(edge_attr_dim), hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            self.register_parameter("edge_weight_logits", None)
        else:
            self.register_parameter("edge_weight_logits", None)
            self.edge_mlp = None

        convs = []
        if num_layers == 1:
            convs.append(GCNConv(in_channels, hidden_dim // 2))
            last_dim = hidden_dim // 2
        else:
            convs.append(GCNConv(in_channels, hidden_dim))
            for _ in range(num_layers - 2):
                convs.append(GCNConv(hidden_dim, hidden_dim))
            convs.append(GCNConv(hidden_dim, hidden_dim // 2))
            last_dim = hidden_dim // 2

        self.convs = nn.ModuleList(convs)
        self.partition_head = nn.Linear(last_dim, 1)
        self.order_hw_head = nn.Linear(last_dim, 1)
        self.order_sw_head = nn.Linear(last_dim, 1)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        if self.edge_weight_learner == "per_edge":
            edge_logits = self.edge_weight_logits
        elif self.edge_weight_learner == "mlp":
            if edge_attr is None:
                raise ValueError("edge_attr is required when edge_weight_learner='mlp'")
            edge_logits = self.edge_mlp(edge_attr).squeeze(-1)
        else:
            edge_logits = None
        eff_edge_weight = _apply_learned_edge_scaling(
            edge_weight=edge_weight,
            edge_logits=edge_logits,
            min_scale=self.edge_weight_min_scale,
            max_scale=self.edge_weight_max_scale,
        )
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_weight=eff_edge_weight)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        assign_logit = self.partition_head(h).squeeze(-1)
        logits2 = torch.stack([-assign_logit, assign_logit], dim=1)
        prio_hw = self.order_hw_head(h).squeeze(-1)
        prio_sw = self.order_sw_head(h).squeeze(-1)
        return logits2, prio_hw, prio_sw


class MPNNs(nn.Module):
    """
    Generic message-passing network backbone (supports gcn/sage/gat).
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        local_layers=3,
        dropout=0.5,
        heads=1,
        pre_ln=False,
        pre_linear=False,
        res=False,
        ln=False,
        bn=False,
        jk=False,
        gnn='gcn',
        edge_aware=False,
        edge_attr_dim=0,
    ):
        super().__init__()
        if local_layers < 1:
            raise ValueError("local_layers must be >= 1")

        self.dropout = float(dropout)
        self.pre_ln = bool(pre_ln)
        self.pre_linear = bool(pre_linear)
        self.res = bool(res)
        self.ln = bool(ln)
        self.bn = bool(bn)
        self.jk = bool(jk)
        self.gnn = str(gnn).lower()
        self.heads = int(heads)
        self.edge_aware = bool(edge_aware)
        self.edge_attr_dim = int(edge_attr_dim)

        self.local_convs = nn.ModuleList()
        self.res_lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pre_lns = nn.ModuleList() if self.pre_ln else None

        self.lin_in = nn.Linear(in_channels, hidden_channels)

        current_dim = in_channels
        if self.pre_linear:
            current_dim = hidden_channels

        for _ in range(local_layers):
            if self.gnn == 'gat':
                use_edge_attr = self.edge_aware and self.edge_attr_dim > 0
                conv = GATConv(
                    current_dim,
                    hidden_channels,
                    heads=self.heads,
                    concat=True,
                    add_self_loops=False,
                    bias=False,
                    edge_dim=self.edge_attr_dim if use_edge_attr else None,
                )
                conv_out_dim = hidden_channels * self.heads
            elif self.gnn == 'sage':
                conv = SAGEConv(current_dim, hidden_channels)
                conv_out_dim = hidden_channels
            else:
                conv = GCNConv(current_dim, hidden_channels, cached=False, normalize=True)
                conv_out_dim = hidden_channels

            self.local_convs.append(conv)
            self.res_lins.append(nn.Linear(current_dim, conv_out_dim))
            self.lns.append(nn.LayerNorm(conv_out_dim))
            self.bns.append(nn.BatchNorm1d(conv_out_dim))
            if self.pre_ln:
                self.pre_lns.append(nn.LayerNorm(current_dim))

            current_dim = conv_out_dim

        self.pred_local = nn.Linear(current_dim, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.res_lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        use_gat_edge_attr = self.gnn == "gat" and self.edge_aware and self.edge_attr_dim > 0
        if use_gat_edge_attr and edge_attr is None:
            raise ValueError("edge_attr must be provided when using gnn='gat' with edge_aware=True")

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = None
        for i, local_conv in enumerate(self.local_convs):
            x_in = self.pre_lns[i](x) if self.pre_ln else x

            if self.res:
                if self.gnn == "gcn":
                    x = local_conv(x_in, edge_index, edge_weight=edge_weight) + self.res_lins[i](x)
                elif use_gat_edge_attr:
                    x = local_conv(x_in, edge_index, edge_attr=edge_attr) + self.res_lins[i](x)
                else:
                    x = local_conv(x_in, edge_index) + self.res_lins[i](x)
            else:
                if self.gnn == "gcn":
                    x = local_conv(x_in, edge_index, edge_weight=edge_weight)
                elif use_gat_edge_attr:
                    x = local_conv(x_in, edge_index, edge_attr=edge_attr)
                else:
                    x = local_conv(x_in, edge_index)

            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.jk:
                x_final = x if x_final is None else (x_final + x)
            else:
                x_final = x

        x = self.pred_local(x_final)
        return x


class MPNNsPlacement(nn.Module):
    """Placement wrapper around MPNNs. Returns [software, hardware] logits."""

    def __init__(self, in_channels, hidden_dim, num_layers, dropout, **mpnns_kwargs):
        super().__init__()
        self.net = MPNNs(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=1,
            local_layers=num_layers,
            dropout=dropout,
            **mpnns_kwargs,
        )

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        logits = self.net(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr).squeeze(-1)
        logits2 = torch.stack([-logits, logits], dim=1)
        return logits2


class MPNNsOrder(nn.Module):
    """Ordering wrapper around MPNNs. Returns placement logits + HW/SW priorities."""

    def __init__(self, in_channels, hidden_dim, num_layers, dropout, **mpnns_kwargs):
        super().__init__()
        self.net = MPNNs(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=3,
            local_layers=num_layers,
            dropout=dropout,
            **mpnns_kwargs,
        )

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        out = self.net(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
        assign_logit = out[:, 0]
        prio_hw = out[:, 1]
        prio_sw = out[:, 2]
        logits2 = torch.stack([-assign_logit, assign_logit], dim=1)
        return logits2, prio_hw, prio_sw


def _extract_mpnns_kwargs(model_kwargs: dict) -> dict:
    # Accept either nested dict `mpnns` or flat `mpnns_*` keys.
    nested = dict(model_kwargs.get("mpnns", {})) if isinstance(model_kwargs.get("mpnns", {}), dict) else {}

    flat_map = {
        "gnn": model_kwargs.get("mpnns_gnn", None),
        "heads": model_kwargs.get("mpnns_heads", None),
        "pre_ln": model_kwargs.get("mpnns_pre_ln", None),
        "pre_linear": model_kwargs.get("mpnns_pre_linear", None),
        "res": model_kwargs.get("mpnns_res", None),
        "ln": model_kwargs.get("mpnns_ln", None),
        "bn": model_kwargs.get("mpnns_bn", None),
        "jk": model_kwargs.get("mpnns_jk", None),
        "edge_aware": model_kwargs.get("mpnns_edge_aware", None),
        "edge_attr_dim": model_kwargs.get("mpnns_edge_attr_dim", None),
    }
    for k, v in flat_map.items():
        if v is not None and k not in nested:
            nested[k] = v

    nested.setdefault("gnn", "gcn")
    nested.setdefault("heads", 1)
    nested.setdefault("pre_ln", False)
    nested.setdefault("pre_linear", False)
    nested.setdefault("res", False)
    nested.setdefault("ln", False)
    nested.setdefault("bn", False)
    nested.setdefault("jk", False)
    nested.setdefault("edge_aware", False)
    nested.setdefault("edge_attr_dim", 0)
    return nested


def build_placement_model(model_name: str, in_channels: int, hidden_dim: int, num_layers: int, dropout: float, **model_kwargs):
    name = str(model_name or "default").lower()
    if name in ("default", "diffgnn", "gcn"):
        return DiffGNNPlacement(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_weight_learner=model_kwargs.get("edge_weight_learner", None),
            learn_edge_weight=bool(model_kwargs.get("learn_edge_weight", False)),
            num_edges=int(model_kwargs.get("num_edges", 0)),
            edge_attr_dim=int(model_kwargs.get("edge_attr_dim", 0)),
            edge_mlp_hidden_dim=int(model_kwargs.get("edge_mlp_hidden_dim", 16)),
            edge_weight_min_scale=float(model_kwargs.get("edge_weight_min_scale", 0.5)),
            edge_weight_max_scale=float(model_kwargs.get("edge_weight_max_scale", 1.5)),
        )
    if name in ("mpnns", "mpnn"):
        mpnns_kwargs = _extract_mpnns_kwargs(model_kwargs)
        return MPNNsPlacement(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **mpnns_kwargs,
        )
    raise ValueError(f"Unsupported model '{model_name}'. Use 'default' or 'mpnns'.")


def build_order_model(model_name: str, in_channels: int, hidden_dim: int, num_layers: int, dropout: float, **model_kwargs):
    name = str(model_name or "default").lower()
    if name in ("default", "diffgnn", "gcn"):
        return DiffGNNOrder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_weight_learner=model_kwargs.get("edge_weight_learner", None),
            learn_edge_weight=bool(model_kwargs.get("learn_edge_weight", False)),
            num_edges=int(model_kwargs.get("num_edges", 0)),
            edge_attr_dim=int(model_kwargs.get("edge_attr_dim", 0)),
            edge_mlp_hidden_dim=int(model_kwargs.get("edge_mlp_hidden_dim", 16)),
            edge_weight_min_scale=float(model_kwargs.get("edge_weight_min_scale", 0.5)),
            edge_weight_max_scale=float(model_kwargs.get("edge_weight_max_scale", 1.5)),
        )
    if name in ("mpnns", "mpnn"):
        mpnns_kwargs = _extract_mpnns_kwargs(model_kwargs)
        return MPNNsOrder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **mpnns_kwargs,
        )
    raise ValueError(f"Unsupported model '{model_name}'. Use 'default' or 'mpnns'.")
