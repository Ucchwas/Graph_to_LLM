"""Shared plumbing for all Phase-1 baselines.

Every baseline sees train edges during training and train+val edges at test time
(what the RandomLinkSplit objects encode), and is scored by the one
evaluate_edge_split() on the one split. seed varies split AND init together.
"""
import time

import torch

from g2l.data import dense_adjacency
from g2l.metrics import evaluate_edge_split


def get_split(seed: int, dataset: str = "cora"):
    from g2l.datasets import load_graph

    return load_graph(dataset, seed)


def recon_bce(logits_full: torch.Tensor, pos_edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """GAE-style reconstruction loss: BCE on positive edges + an equal number of
    freshly sampled negatives (excluding only known positives and self-loops)."""
    from torch_geometric.utils import negative_sampling

    neg_edge_index = negative_sampling(pos_edge_index, num_nodes)
    pos = logits_full[pos_edge_index[0], pos_edge_index[1]]
    neg = logits_full[neg_edge_index[0], neg_edge_index[1]]
    logits = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def run_baseline(name: str, seed: int, score_fn, dataset: str = "cora") -> dict:
    """score_fn(data, split, device) -> full [N, N] logits. Returns one table row."""
    torch.manual_seed(seed)
    data, split = get_split(seed, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    logits = score_fn(data, split, device)
    row = evaluate_edge_split(logits.cpu(), split, seed=seed)
    row.update(model=name, seed=seed, dataset=dataset, wallclock_s=round(time.time() - t0, 1))
    return row


def observed_dense(split_data, num_nodes: int) -> torch.Tensor:
    return dense_adjacency(split_data.edge_index, num_nodes)
