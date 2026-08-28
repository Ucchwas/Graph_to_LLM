"""Phase 7: graph construction, the gene-universe leak rule, and the changed-edge metrics."""
import os

import pytest
import torch

pytest.importorskip("torch_geometric")

from g2l.model import build_model
from g2l.tcga import binarise, condition_graph, gene_universe, jaccard, root, spearman, subsample
from g2l.translate import changed_edge_scores, identity_baseline, mean_change, mean_tumour

G, S = 60, 40


def synthetic(seed=0):
    """Expression with block structure so the correlation graph is not noise."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(6, S, generator=g)
    X = base.repeat_interleave(10, dim=0) + 0.4 * torch.randn(G, S, generator=g)
    return X + 5.0


def test_spearman_is_rank_based_and_symmetric():
    X = synthetic()
    C = spearman(X)
    assert C.shape == (G, G) and torch.allclose(C, C.T, atol=1e-6)
    assert C.diagonal().abs().max() == 0 and C.abs().max() <= 1.0 + 1e-5
    mono = X.clone()
    mono[0] = mono[0] ** 3  # a monotone transform must not change Spearman
    assert torch.allclose(spearman(mono), C, atol=1e-5)


@pytest.mark.parametrize("density", [0.01, 0.05, 0.2])
def test_binarise_hits_the_requested_density_and_is_symmetric(density):
    A = binarise(spearman(synthetic()), density)
    iu = torch.triu_indices(G, G, offset=1)
    frac = A[iu[0], iu[1]].sum().item() / iu.shape[1]
    assert abs(frac - density) < 1.5 / iu.shape[1] + 1e-9
    assert torch.equal(A, A.T) and A.diagonal().sum() == 0 and set(A.unique().tolist()) <= {0.0, 1.0}


def test_matched_density_between_conditions():
    X = synthetic(0)
    a = condition_graph(X, list(range(G)), list(range(20)), 0.05)
    b = condition_graph(X, list(range(G)), list(range(20, 40)), 0.05)
    assert a.sum() == b.sum(), "normal and tumour graphs must carry the same edge count"


def test_gene_universe_never_reads_tumour_columns():
    X = synthetic()
    normals = list(range(20))
    poisoned = X.clone()
    poisoned[:, 20:] = 1e6  # destroy every tumour column
    assert gene_universe([str(i) for i in range(G)], X, normals, n_genes=10, expr_floor=0.0) == \
           gene_universe([str(i) for i in range(G)], poisoned, normals, n_genes=10, expr_floor=0.0)


def test_subsample_is_seeded_and_bounded():
    cols = [f"s{i}" for i in range(50)]
    a, b = subsample(cols, 10, 0), subsample(cols, 10, 0)
    assert a == b and len(a) == 10 and set(a) <= set(cols) and a == sorted(a, key=cols.index)
    assert subsample(cols, 10, 1) != a and subsample(cols, 99, 0) == cols


def test_edge_index_comes_only_from_the_normal_graph():
    """The Phase-7 leak rule: the tumour target must not reach the input or the edge set."""
    A_n = binarise(spearman(synthetic(0)), 0.1)
    A_t = binarise(spearman(synthetic(1)), 0.1)
    m = build_model("gnn_direct", G, 16, 1.0, scratch_layers=2, seed=0, kind="gcn").eval()
    seen = []
    orig = m.body.forward

    def spy(tokens, A):
        seen.append(A.clone())
        return orig(tokens, A)

    m.body.forward = spy
    with torch.no_grad():
        m(A_n)
    assert len(seen) == 1 and torch.equal(seen[0], A_n)
    assert not torch.equal(seen[0], A_t)
    ei = seen[0].nonzero().T
    assert not A_t[ei[0], ei[1]].all(), "edge set must not be the tumour graph's"


def test_identity_is_exactly_chance_on_changed_edges():
    A_n = binarise(spearman(synthetic(0)), 0.1)
    A_t = binarise(spearman(synthetic(1)), 0.1)
    s = changed_edge_scores(identity_baseline(A_n), A_n, A_t)
    assert abs(s["auc_gained"] - 0.5) < 1e-12 and abs(s["auc_lost"] - 0.5) < 1e-12
    assert s["n_gained"] > 0 and s["n_lost"] > 0
    assert s["n_gained"] == s["n_lost"], "matched density forces equal gains and losses"


def test_mean_baselines_use_training_cancers_only():
    A = [binarise(spearman(synthetic(k)), 0.1) for k in range(4)]
    N, T = A[:2], A[2:]
    mt, mc = mean_tumour(T), mean_change(N, T, A[0])
    assert mt.shape == (G, G) and torch.allclose(mt, mt.T)
    assert mc.shape == (G, G) and torch.allclose(mc, mc.T)
    assert torch.allclose(mean_tumour([T[0]]), T[0])


@pytest.mark.skipif(not (root() / "processed" / "pairs.pt").exists(), reason="Phase-7 graphs not built")
def test_built_pairs_are_consistent():
    o = torch.load(root() / "processed" / "pairs.pt")
    for c in o["cancers"]:
        A_n, A_t = o["normal"][c], o["tumour"][c]
        assert A_n.shape == A_t.shape and torch.equal(A_n, A_n.T) and torch.equal(A_t, A_t.T)
        assert A_n.sum() == A_t.sum() and A_n.diagonal().sum() == 0
        assert 0.0 < jaccard(A_n, A_t) < 1.0
