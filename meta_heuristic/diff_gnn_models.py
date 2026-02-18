import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
except Exception as e:
    raise ImportError("This module requires torch_geometric. Install with 'pip install torch-geometric'.") from e


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
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.dropout = float(dropout)

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

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
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
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.dropout = float(dropout)

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

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
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
                conv = GATConv(
                    current_dim,
                    hidden_channels,
                    heads=self.heads,
                    concat=True,
                    add_self_loops=False,
                    bias=False,
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

    def forward(self, x, edge_index):
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = None
        for i, local_conv in enumerate(self.local_convs):
            x_in = self.pre_lns[i](x) if self.pre_ln else x

            if self.res:
                x = local_conv(x_in, edge_index) + self.res_lins[i](x)
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

    def forward(self, x, edge_index):
        logits = self.net(x, edge_index).squeeze(-1)
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

    def forward(self, x, edge_index):
        out = self.net(x, edge_index)
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
    return nested


def build_placement_model(model_name: str, in_channels: int, hidden_dim: int, num_layers: int, dropout: float, **model_kwargs):
    name = str(model_name or "default").lower()
    if name in ("default", "diffgnn", "gcn"):
        return DiffGNNPlacement(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
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
