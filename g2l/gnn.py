"""Graph-native backbone (Phase 4, LGM v0): message passing over the model's own input graph.
The edge set is rebuilt from the observed adjacency on every forward (the masked input at train
time, the test input at eval), so held-out cells never reach the backbone (leak rule as for the
bias). Two input routes:
  `gnn`         tokens [N, d] from the E1 row tokenizer -> L pre-norm residual blocks
  `gnn_direct`  the raw adjacency rows [N, N] go straight into the first conv (input width N,
                its weight is the row projection) -> L-1 pre-norm residual blocks
The lab GFM (a GNN) will take this slot."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


def make_conv(kind: str, d_in: int, d_out: int, heads: int):
    if kind == "gcn":
        return GCNConv(d_in, d_out)
    if kind == "sage":
        return SAGEConv(d_in, d_out)
    if kind == "gin":
        return GINConv(nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out)))
    if kind == "gat":
        assert d_out % heads == 0
        return GATConv(d_in, d_out // heads, heads=heads)
    raise ValueError(kind)


class GNNBody(nn.Module):
    needs_graph = True

    def __init__(self, d_model: int, layers: int = 2, kind: str = "gcn", dropout: float = 0.0, heads: int = 4,
                 in_dim: int | None = None):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        self.first = make_conv(kind, in_dim, d_model, heads) if in_dim else None  # direct route
        n_blocks = layers - 1 if in_dim else layers
        assert n_blocks >= 0
        self.convs = nn.ModuleList([make_conv(kind, d_model, d_model, heads) for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])

    def forward(self, tokens, A):
        edge_index = A.nonzero().T  # A is symmetric 0/1: both directions present
        h = tokens
        if self.first is not None:
            h = F.dropout(F.gelu(self.first(h, edge_index)), self.dropout, self.training)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.gelu(conv(norm(h), edge_index)), self.dropout, self.training)
        return h
