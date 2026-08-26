"""Data loading, splits, and masking. Everything downstream consumes DenseBatch.

Two masking views (both required by the eval protocol):
  1. edge-split  -- RandomLinkSplit, comparable to published GAE/VGAE numbers.
  2. matrix-mask -- hide a fraction of upper-triangle cells, mirrored into the input.
"""
import random
from dataclasses import dataclass

import torch


@dataclass
class DenseBatch:
    A_obs: torch.Tensor       # [B, N, N] float, observed adjacency (hidden cells zeroed)
    A_true: torch.Tensor      # [B, N, N] float, ground truth
    sup_mask: torch.Tensor    # [B, N, N] bool, upper-triangle cells to predict and score
    node_mask: torch.Tensor   # [B, N] bool, real nodes vs padding
    X: torch.Tensor | None    # [B, N, F] float or None


def dense_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = 1.0
    return A


def mask_matrix(A: torch.Tensor, frac: float = 0.15, seed: int = 0):
    """Hide `frac` of the strict upper triangle. Returns (A_obs, sup_mask).

    sup_mask holds upper-triangle cells only (scored once per pair); the input
    A_obs has BOTH directions of every hidden pair zeroed. Diagonal excluded.
    """
    N = A.shape[-1]
    g = torch.Generator().manual_seed(seed)
    iu, ju = torch.triu_indices(N, N, offset=1)
    pick = torch.rand(iu.numel(), generator=g) < frac
    sup_mask = torch.zeros(N, N, dtype=torch.bool)
    sup_mask[iu[pick], ju[pick]] = True
    hidden = sup_mask | sup_mask.T
    return A * (~hidden).float(), sup_mask


def rewire_degree_preserving(A: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Random graph with the same degree sequence (double-edge swaps): the shuffled-adjacency
    control. Only degree information survives."""
    import networkx as nx
    import numpy as np

    G = nx.from_numpy_array(A.cpu().numpy())
    m = G.number_of_edges()
    nx.double_edge_swap(G, nswap=10 * m, max_tries=1000 * m, seed=seed)
    return torch.from_numpy(nx.to_numpy_array(G, nodelist=range(A.shape[0]), dtype=np.float32))


def collate(graphs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]]) -> DenseBatch:
    """Pad variable-size graphs to a common N. Padded cells never enter loss or metrics."""
    B = len(graphs)
    N = max(a.shape[-1] for a, _, _, _ in graphs)
    F = next((x.shape[-1] for _, _, _, x in graphs if x is not None), None)
    A_obs = torch.zeros(B, N, N)
    A_true = torch.zeros(B, N, N)
    sup = torch.zeros(B, N, N, dtype=torch.bool)
    nmask = torch.zeros(B, N, dtype=torch.bool)
    X = torch.zeros(B, N, F) if F is not None else None
    for b, (a_obs, a_true, s, x) in enumerate(graphs):
        n = a_obs.shape[-1]
        A_obs[b, :n, :n] = a_obs
        A_true[b, :n, :n] = a_true
        sup[b, :n, :n] = s
        nmask[b, :n] = True
        if x is not None:
            X[b, :n] = x
    return DenseBatch(A_obs, A_true, sup, nmask, X)


def load_cora(root: str | None = None):
    import os

    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    root = root or os.environ.get("PYG_DATA_ROOT", "data/pyg")
    return Planetoid(root, "Cora", transform=NormalizeFeatures())[0]


def edge_split(data, num_val: float = 0.05, num_test: float = 0.10, seed: int = 0):
    """The published-comparable protocol. Applied ONCE (never as a dataset transform:
    transforms rerun on every access and would yield a different split each time)."""
    from torch_geometric.transforms import RandomLinkSplit

    torch.manual_seed(seed)
    random.seed(seed)  # PyG's negative_sampling draws with random.sample, not torch
    split = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False,
    )
    return split(data)  # (train, val, test)


def sparse_eval_mask(train, val, test):
    """Scorable cells at test time under the edge-split protocol: the strict upper
    triangle minus train/val edge cells. Exactly the test held-out edges are positive.
    Returns (mask [N,N] bool, target [N,N] float)."""
    N = train.num_nodes
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    for ei in (train.pos_edge_label_index, val.pos_edge_label_index):
        i, j = ei
        mask[torch.minimum(i, j), torch.maximum(i, j)] = False
    target = torch.zeros(N, N)
    i, j = test.pos_edge_label_index
    target[torch.minimum(i, j), torch.maximum(i, j)] = 1.0
    assert not target[~mask].any(), "a test edge collided with a train/val edge"
    return mask, target
