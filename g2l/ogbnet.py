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
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch import nn

try:
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
except ImportError:
    # The dev laptop's Application Control policy intermittently blocks scipy DLLs, and importing
    # ogb.graphproppred runs its __init__ -> sklearn -> scipy. mol_encoder.py itself needs only
    # torch and ogb.utils.features, so load that one file directly. Marlowe takes the normal path.
    import importlib.util
    import pathlib

    import ogb

    _spec = importlib.util.spec_from_file_location(
        "ogb_mol_encoder", pathlib.Path(ogb.__file__).parent / "graphproppred" / "mol_encoder.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    AtomEncoder, BondEncoder = _mod.AtomEncoder, _mod.BondEncoder


def _widened(dims, emb_dim):
    """OGB's encoder construction verbatim (one xavier-init Embedding per categorical column,
    summed) with ONE extra row per column: the reserved MASK index. Index `dims[i]` means "this
    attribute is hidden", so masked-attribute pretraining can mark an attribute absent instead of
    silently assigning it a real category. Everything else -- init, summation, column order -- is
    identical to ogb.graphproppred.mol_encoder."""
    ms = nn.ModuleList()
    for dim in dims:
        emb = nn.Embedding(dim + 1, emb_dim)
        nn.init.xavier_uniform_(emb.weight.data)
        ms.append(emb)
    return ms


class MaskAtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.atom_embedding_list = _widened(get_atom_feature_dims(), emb_dim)

    def forward(self, x):
        return sum(emb(x[:, i]) for i, emb in enumerate(self.atom_embedding_list))


class MaskBondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.bond_embedding_list = _widened(get_bond_feature_dims(), emb_dim)

    def forward(self, edge_attr):
        return sum(emb(edge_attr[:, i]) for i, emb in enumerate(self.bond_embedding_list))


class OGBGINConv(nn.Module):
    """h_i <- MLP( (1+eps) h_i + sum_j relu(h_j + e_ij) )."""

    def __init__(self, d: int, mask_tokens: bool = False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.BatchNorm1d(2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.eps = nn.Parameter(torch.zeros(1))
        self.bond_encoder = (MaskBondEncoder if mask_tokens else BondEncoder)(emb_dim=d)

    def forward(self, x, edge_index, edge_attr):
        e = self.bond_encoder(edge_attr)
        src, dst = edge_index
        agg = torch.zeros_like(x).index_add_(0, dst, F.relu(x[src] + e))
        return self.mlp((1 + self.eps) * x + agg)


class OGBGCNConv(nn.Module):
    """h_i <- sum_j norm_ij relu(W h_j + e_ij) + relu(W h_i + root)/deg_i, norm = (deg_i deg_j)^-1/2."""

    def __init__(self, d: int, mask_tokens: bool = False):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.root_emb = nn.Embedding(1, d)
        self.bond_encoder = (MaskBondEncoder if mask_tokens else BondEncoder)(emb_dim=d)

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
                 seed: int | None = None, mask_tokens: bool = False):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        conv = {"gin": OGBGINConv, "gcn": OGBGCNConv}[kind]
        self.atom_encoder = (MaskAtomEncoder if mask_tokens else AtomEncoder)(emb_dim=d)
        self.convs = nn.ModuleList([conv(d, mask_tokens) for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d) for _ in range(layers)])
        self.head = nn.Linear(d, 1)
        self.dropout, self.layers = dropout, layers

    def node_states(self, g) -> torch.Tensor:
        """The conv stack, pre-pool: [N, d]. Split out so masked-attribute pretraining can read
        per-node states -- forward() is exactly this plus mean pooling and the head."""
        h = self.atom_encoder(g.x)
        for k, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = bn(conv(h, g.edge_index, g.edge_attr))
            h = F.dropout(h if k == self.layers - 1 else F.relu(h), self.dropout, self.training)
        return h

    def forward(self, g, n_graphs: int) -> torch.Tensor:
        h = self.node_states(g)
        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, g.batch, h)
        cnt = torch.zeros(n_graphs, 1, device=h.device, dtype=h.dtype).index_add_(
            0, g.batch, torch.ones_like(h[:, :1]))
        return self.head(pooled / cnt.clamp(min=1)).squeeze(-1)
