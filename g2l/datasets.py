"""Phase 5: several graphs behind the one split interface the harness consumes.

load_graph(name, seed) -> (data, (train, val, test)). Every split object carries `edge_index`
(the message-passing graph at that stage), `pos_edge_label_index` and `neg_edge_label_index`
(seeded 1:1 uniform non-edges: the AP@1:1 protocol of Phases 1-4). ogbl-ddi's val/test also carry
`hits_neg_edge_label_index`, the official negative sets scored by Hits@20.

  cora    Planetoid Cora, RandomLinkSplit 85/5/10 (Phases 1-4); test input = train + val edges
  photo   Amazon Photo (Shchur et al. 2018), same split protocol
  ddi     ogbl-ddi, the official protein-target split; input graph = the train edges at val and
          test time (the OGB convention), the split is fixed and only the 1:1 negatives are seeded
  ppi     the 24 GraphSAGE tissue graphs: inspection only (variable N, no split defined yet)
"""
import functools
import os

import torch
from torch_geometric.data import Data

from g2l.data import dense_adjacency, edge_split, load_cora


def root() -> str:
    return os.environ.get("PYG_DATA_ROOT", "data/pyg")


def load_photo():
    from torch_geometric.datasets import Amazon

    return Amazon(root(), "Photo")[0]


def ogb_dataset(name: str):
    """OGB's processed files pickle PyG classes; torch >= 2.6 refuses them under weights_only."""
    from ogb.linkproppred import PygLinkPropPredDataset

    orig = torch.load
    torch.load = functools.partial(orig, weights_only=False)
    try:
        ds = PygLinkPropPredDataset(name, root=root())
        return ds, ds.get_edge_split()
    finally:
        torch.load = orig


def sample_non_edges(A_all: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """n distinct seeded uniform pairs (i, j), i != j, with A_all[i, j] == 0: RandomLinkSplit's
    negative protocol on a dense graph. Returns [2, n]."""
    N = A_all.shape[0]
    g = torch.Generator().manual_seed(seed)
    taken = A_all.bool() | A_all.bool().T | torch.eye(N, dtype=torch.bool)
    out, got = [], 0
    while got < n:
        i = torch.randint(N, (2 * (n - got) + 16,), generator=g)
        j = torch.randint(N, (i.numel(),), generator=g)
        ok = ~taken[i, j]
        key = torch.unique(torch.minimum(i[ok], j[ok]) * N + torch.maximum(i[ok], j[ok]))
        i, j = key // N, key % N
        taken[i, j] = taken[j, i] = True
        out.append(torch.stack([i, j]))
        got += i.numel()
    return torch.cat(out, 1)[:, :n]


def load_ddi(seed: int = 0):
    ds, se = ogb_dataset("ogbl-ddi")
    g = ds[0]
    N, ei = g.num_nodes, g.edge_index  # the train edges, both directions
    tr, va, te = (se[k]["edge"].T.contiguous() for k in ("train", "valid", "test"))
    assert ei.size(1) == 2 * tr.size(1)
    A_all = dense_adjacency(torch.cat([tr, va, te], 1), N)

    def part(pos, hits_neg, s):
        d = Data(edge_index=ei, num_nodes=N)
        d.pos_edge_label_index = pos
        d.neg_edge_label_index = sample_non_edges(A_all, pos.size(1), seed=s)
        d.hits_neg_edge_label_index = hits_neg.T.contiguous()
        return d

    train = Data(edge_index=ei, num_nodes=N)
    train.pos_edge_label_index = tr
    val = part(va, se["valid"]["edge_neg"], 2 * seed + 1)
    test = part(te, se["test"]["edge_neg"], 2 * seed + 2)
    return Data(edge_index=ei, num_nodes=N), (train, val, test)


def load_graph(name: str, seed: int = 0):
    if name == "cora":
        data = load_cora()
        return data, edge_split(data, seed=seed)
    if name == "photo":
        data = load_photo()
        return data, edge_split(data, seed=seed)
    if name == "ddi":
        return load_ddi(seed)
    raise ValueError(name)


def ppi_graphs() -> list[dict]:
    """Sizes of the 24 PPI graphs (20 train / 2 val / 2 test); downloaded on first call."""
    from torch_geometric.datasets import PPI
    from torch_geometric.utils import to_undirected

    rows = []
    for split in ("train", "val", "test"):
        for k, g in enumerate(PPI(os.path.join(root(), "PPI"), split=split)):
            rows.append({"split": split, "index": k, "n_nodes": g.num_nodes,
                         "n_edges": to_undirected(g.edge_index, num_nodes=g.num_nodes).size(1) // 2})
    return rows
