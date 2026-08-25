"""Edge-split protocol invariants on a tiny synthetic PyG graph."""
import pytest
import torch

pytest.importorskip("torch_geometric")

from g2l.data import edge_split, sparse_eval_mask
from g2l.metrics import evaluate_edge_split


@pytest.fixture(scope="module")
def tiny_split():
    from torch_geometric.data import Data

    torch.manual_seed(0)
    n = 40
    up = torch.triu((torch.rand(n, n) < 0.2).float(), diagonal=1)
    ei = up.nonzero().t()
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    return edge_split(Data(x=torch.randn(n, 3), edge_index=ei, num_nodes=n), seed=0)


def test_sparse_mask_excludes_seen_edges(tiny_split):
    train, val, test = tiny_split
    mask, target = sparse_eval_mask(train, val, test)
    for ei in (train.pos_edge_label_index, val.pos_edge_label_index):
        i, j = ei
        assert not mask[torch.minimum(i, j), torch.maximum(i, j)].any()
    i, j = test.pos_edge_label_index
    assert mask[torch.minimum(i, j), torch.maximum(i, j)].all()
    assert int(target.sum()) == test.pos_edge_label_index.shape[1]


def test_perfect_scorer_maxes_all_columns(tiny_split):
    train, val, test = tiny_split
    _, target = sparse_eval_mask(train, val, test)
    m = evaluate_edge_split(torch.where((target + target.T) > 0, 10.0, -10.0), tiny_split)
    assert m["auc"] == 1.0 and m["ap"] == 1.0 and m["auroc_sparse"] == 1.0


def test_heuristic_triangle_closure():
    from baselines.heuristics import adamic_adar, common_neighbors

    A = torch.zeros(5, 5)
    for i, j in [(0, 1), (1, 2), (0, 3)]:
        A[i, j] = A[j, i] = 1.0
    for fn in (common_neighbors, adamic_adar):
        s = fn(A)
        assert s[0, 2] > s[0, 4]  # shares neighbour 1 vs shares nothing
