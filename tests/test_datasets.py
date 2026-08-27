"""Phase 5: the dataset layer (ogbl-ddi official split, Amazon Photo), Hits@K, the negative
sampler, and the featureless autoencoder baselines. Dataset-dependent tests skip when the download
is absent."""
import os

import pytest
import torch

pytest.importorskip("torch_geometric")

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from baselines import gae, maskgae
from g2l.data import edge_split, sparse_eval_mask
from g2l.datasets import load_graph, root, sample_non_edges
from g2l.metrics import evaluate_edge_split, hits_at_k
from g2l.model import build_model


def synthetic(n=40, p=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(n, n, generator=g) < p).float(), diagonal=1)
    ei = to_undirected(up.nonzero().T, num_nodes=n)
    data = Data(x=torch.randn(n, 8, generator=g), edge_index=ei, num_nodes=n)
    return data, edge_split(data, seed=seed)


def test_hits_at_k_matches_the_ogb_evaluator():
    pytest.importorskip("ogb")
    from ogb.linkproppred import Evaluator

    ev = Evaluator(name="ogbl-ddi")
    ev.K = 20
    pos, neg = torch.randn(300), torch.randn(500)
    ref = ev.eval({"y_pred_pos": pos.numpy(), "y_pred_neg": neg.numpy()})["hits@20"]
    assert abs(hits_at_k(pos, neg, 20) - ref) < 1e-6


def test_sample_non_edges_are_distinct_seeded_non_edges():
    A = synthetic(60, 0.3)[0]
    A = torch.zeros(60, 60).index_put_((A.edge_index[0], A.edge_index[1]), torch.ones(A.edge_index.size(1)))
    e = sample_non_edges(A, 300, seed=1)
    assert e.shape == (2, 300)
    assert not A[e[0], e[1]].any() and not A[e[1], e[0]].any() and (e[0] != e[1]).all()
    keys = torch.minimum(e[0], e[1]) * 60 + torch.maximum(e[0], e[1])
    assert keys.unique().numel() == 300
    assert torch.equal(e, sample_non_edges(A, 300, seed=1)) and not torch.equal(e, sample_non_edges(A, 300, seed=2))


def test_direct_gcn_takes_any_n():
    data, _ = synthetic(50)
    A = torch.zeros(50, 50).index_put_((data.edge_index[0], data.edge_index[1]), torch.ones(data.edge_index.size(1)))
    m = build_model("gnn_direct", 50, 16, 1.0, scratch_layers=2, seed=0, kind="gcn")
    assert m.body.first.in_channels == 50 and m(A).shape == (50, 50)


@pytest.mark.parametrize("name", ["gae", "vgae"])
def test_featureless_autoencoders_run(name):
    data, split = synthetic()
    L = gae.make_score_fn(name, max_epochs=3, featureless=True)(data, split, "cpu")
    assert L.shape == (40, 40) and torch.isfinite(L).all()


@pytest.mark.parametrize("featureless", [True, False])
def test_maskgae_runs_and_scores_every_pair(featureless):
    data, split = synthetic()
    L = maskgae.make_score_fn(featureless=featureless, epochs=2, eval_period=1)(data, split, "cpu")
    assert L.shape == (40, 40) and torch.isfinite(L).all()
    assert "auc" in evaluate_edge_split(L, split)


@pytest.fixture(scope="module")
def ddi():
    if not os.path.isdir(os.path.join(root(), "ogbl_ddi", "processed")):
        pytest.skip("ogbl-ddi not downloaded")
    return load_graph("ddi", seed=0)


def test_ddi_official_split(ddi):
    data, (train, val, test) = ddi
    N = data.num_nodes
    assert N == 4267 and train.pos_edge_label_index.shape == (2, 1067911)
    assert val.pos_edge_label_index.shape == (2, 133489) and test.pos_edge_label_index.shape == (2, 133489)
    assert val.hits_neg_edge_label_index.shape == (2, 101882) and test.hits_neg_edge_label_index.shape == (2, 95599)
    assert val.neg_edge_label_index.shape == (2, 133489) and test.neg_edge_label_index.shape == (2, 133489)
    assert torch.equal(test.edge_index, train.edge_index) and test.edge_index.shape == (2, 2 * 1067911)
    A_all = torch.zeros(N, N, dtype=torch.bool)
    for s in (train, val, test):
        A_all[s.pos_edge_label_index[0], s.pos_edge_label_index[1]] = True
    A_all = A_all | A_all.T
    for s in (val, test):
        e = s.neg_edge_label_index
        assert not A_all[e[0], e[1]].any() and (e[0] != e[1]).all()
    mask, target = sparse_eval_mask(train, val, test)
    assert mask.sum() == N * (N - 1) // 2 - 1067911 - 133489 and target.sum() == 133489
    row = evaluate_edge_split(torch.randn(N, N), (train, val, test))
    assert 0.0 <= row["hits20"] <= 1.0 and abs(row["auc"] - 0.5) < 0.01


def test_photo_split():
    if not os.path.isdir(os.path.join(root(), "Photo", "processed")):
        pytest.skip("Amazon Photo not downloaded")
    data, (train, val, test) = load_graph("photo", seed=0)
    assert data.num_nodes == 7650 and data.edge_index.shape == (2, 2 * 119081) and data.x.shape == (7650, 745)
    assert test.pos_edge_label_index.shape[1] == int(0.10 * 119081) and val.pos_edge_label_index.shape[1] == int(0.05 * 119081)
    mp = set(map(tuple, test.edge_index.t().tolist()))
    held = set(map(tuple, test.pos_edge_label_index.t().tolist()))
    assert not (held & mp) and not ({(j, i) for i, j in held} & mp)
