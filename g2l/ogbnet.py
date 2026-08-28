"""A faithful reproduction of OGB's reference GIN / GCN for ogbg-mol*, run inside OUR pipeline.

Quoting leaderboard numbers is not enough for a calibration: if our harness is what is broken, a
published number cannot show it. So the baseline is rebuilt here and trained through the same
loader, batcher, optimiser loop and evaluator as the LGM. If this reproduction lands near OGB's
published 0.7606 (GCN) / 0.7558 (GIN) test ROC-AUC, the harness is sound and any remaining gap is
the LGM's. If it does not, the harness is the problem and no LGM tuning would have been meaningful.

Architecture follows OGB's `examples/graphproppred/mol/{conv,gnn}.py`: AtomEncoder on the nodes, a
per-layer BondEncoder on the edges, five layers of emb_dim 300, BatchNorm then ReLU then dropout
after each (no ReLU on the last), JK = "last", no residual, mean pooling, one linear head.
"""
import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn


class OGBGINConv(nn.Module):
    """h_i <- MLP( (1+eps) h_i + sum_j relu(h_j + e_ij) )."""

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.BatchNorm1d(2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.eps = nn.Parameter(torch.zeros(1))
        self.bond_encoder = BondEncoder(emb_dim=d)

    def forward(self, x, edge_index, edge_attr):
        e = self.bond_encoder(edge_attr)
        src, dst = edge_index
        agg = torch.zeros_like(x).index_add_(0, dst, F.relu(x[src] + e))
        return self.mlp((1 + self.eps) * x + agg)


class OGBGCNConv(nn.Module):
    """h_i <- sum_j norm_ij relu(W h_j + e_ij) + relu(W h_i + root)/deg_i, norm = (deg_i deg_j)^-1/2."""

    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.root_emb = nn.Embedding(1, d)
        self.bond_encoder = BondEncoder(emb_dim=d)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        e = self.bond_encoder(edge_attr)
        src, dst = edge_index
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype).index_add_(
            0, src, torch.ones(src.numel(), device=x.device, dtype=x.dtype)) + 1
        dis = deg.pow(-0.5)
        msg = (dis[src] * dis[dst]).unsqueeze(-1) * F.relu(x[src] + e)
        return torch.zeros_like(x).index_add_(0, dst, msg) + \
            F.relu(x + self.root_emb.weight) / deg.unsqueeze(-1)


class OGBGNN(nn.Module):
    """OGB's GNN_node + mean pooling + linear head. `kind` in {gin, gcn}."""

    def __init__(self, kind: str = "gin", d: int = 300, layers: int = 5, dropout: float = 0.5,
                 seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        conv = {"gin": OGBGINConv, "gcn": OGBGCNConv}[kind]
        self.atom_encoder = AtomEncoder(emb_dim=d)
        self.convs = nn.ModuleList([conv(d) for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d) for _ in range(layers)])
        self.head = nn.Linear(d, 1)
        self.dropout, self.layers = dropout, layers

    def forward(self, g, n_graphs: int) -> torch.Tensor:
        src, dst = g.edge_index
        ei = g.edge_index
        h = self.atom_encoder(g.x)
        for k, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = bn(conv(h, ei, g.edge_attr))
            h = F.dropout(h if k == self.layers - 1 else F.relu(h), self.dropout, self.training)
        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, g.batch, h)
        cnt = torch.zeros(n_graphs, 1, device=h.device, dtype=h.dtype).index_add_(
            0, g.batch, torch.ones_like(h[:, :1]))
        return self.head(pooled / cnt.clamp(min=1)).squeeze(-1)
