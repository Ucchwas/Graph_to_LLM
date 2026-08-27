"""Graph-native backbone (Phase 4, LGM v0): message passing over the model's own input graph.
Tokens [N, d] from the row tokenizer + the edges of the observed adjacency -> H [N, d].
Pre-norm residual blocks so depth is safe; the graph is the masked input at train time and the
test input at eval, so held-out cells never reach the backbone (same leak rule as the bias).
The lab GFM (a GNN) will take this slot."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


def make_conv(kind: str, d: int, heads: int):
    if kind == "gcn":
        return GCNConv(d, d)
    if kind == "sage":
        return SAGEConv(d, d)
    if kind == "gin":
        return GINConv(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)))
    if kind == "gat":
        assert d % heads == 0
        return GATConv(d, d // heads, heads=heads)
    raise ValueError(kind)


class GNNBody(nn.Module):
    needs_graph = True

    def __init__(self, d_model: int, layers: int = 2, kind: str = "gcn", dropout: float = 0.0, heads: int = 4):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        self.convs = nn.ModuleList([make_conv(kind, d_model, heads) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])

    def forward(self, tokens, A):
        edge_index = A.nonzero().T  # A is symmetric 0/1: both directions present
        h = tokens
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.gelu(conv(norm(h), edge_index)), self.dropout, self.training)
        return h
