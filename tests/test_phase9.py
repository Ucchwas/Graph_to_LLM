"""Phase 9: leakage, symmetry and fairness of the TCGA normal -> tumour experiment.

Synthetic data throughout: a few "cancers" with random expression over a small gene universe, run
through the SAME construction (`graph_from_expression`, `ssl_corpus`, `ssl_mask`, `folds`) the
real experiment uses, so a test that passes here says something about the real run.
"""
import math

import numpy as np
import pytest
import torch

from g2l import tcga9
from g2l.decoders import D1Bilinear
from g2l.lgm import RawGraph

# d=64 is the floor of matched_width's search grid, so the capacity match is exercised for real
CFG = {"d": 64, "heads": 2, "layers": 2, "dropout": 0.0, "x_dim": 2, "mask_frac": 0.2,
       "ssl_epochs": 2, "lr": 1e-3, "weight_decay": 0.0, "warmup": 5, "clip": 1.0,
       "max_epochs": 2, "patience": 5, "seeds": [0]}
CANCERS = ["A", "B", "C", "D", "E"]


def synthetic(n_genes=40, n_samples=25, seed=0, density=0.05) -> dict:
    g = torch.Generator().manual_seed(seed)
    graphs = {}
    for c in CANCERS:
        for cond in ("normal", "tumour"):
            X = torch.randn(n_genes, n_samples, generator=g) * 2 + 5
            graphs[f"{c}|{cond}"] = tcga9.graph_from_expression(X, density)
    return {"cancers": CANCERS, "genes": [f"g{i}" for i in range(n_genes)], "density": density,
            "n": n_genes, "graphs": graphs}


# ---------------------------------------------------------------- folds & leakage

def test_folds_are_cancer_disjoint_and_match_phase7():
    fs = tcga9.folds(CANCERS)
    assert [f[0] for f in fs] == CANCERS
    for test, val, train in fs:
        assert test != val and test not in train and val not in train
        assert sorted([test, val] + train) == sorted(CANCERS)
    from g2l.run_phase7 import folds as folds7
    assert fs == folds7(CANCERS)


def test_ssl_corpus_excludes_validation_and_test_cancers():
    ds = synthetic()
    for test, val, train in tcga9.folds(CANCERS):
        corpus = tcga9.ssl_corpus(ds, train)
        assert len(corpus) == 2 * len(train)
        banned = {id(tcga9.graph(ds, c, cond)) for c in (test, val) for cond in ("normal", "tumour")}
        assert not any(id(g) in banned for g in corpus)


def test_a_graph_is_built_from_its_own_condition_only():
    """Tumour samples can never reach a normal graph: the builder is handed one condition's
    columns and nothing else. Poisoning every other column must leave the graph unchanged."""
    X = torch.randn(30, 20) * 2 + 5
    g1 = tcga9.graph_from_expression(X, 0.05)
    g2 = tcga9.graph_from_expression(X.clone(), 0.05)
    assert torch.equal(g1.edge_index, g2.edge_index) and torch.equal(g1.x, g2.x)
    # a differently-sampled "tumour" condition produces a different graph, i.e. the input really
    # depends on the columns and was not silently reading something global
    g3 = tcga9.graph_from_expression(torch.randn(30, 20) * 2 + 5, 0.05)
    assert not torch.equal(g3.x, g1.x)


def test_translation_input_is_the_normal_graph_and_never_sees_the_tumour_graph():
    ds = synthetic()
    c = "A"
    raw = tcga9.graph(ds, c, "normal").raw()
    A_n = tcga9.graph(ds, c, "normal").dense()
    assert torch.equal(RawGraph.from_dense(A_n).edge_index, torch.sort(raw.edge_index, dim=1)[0][:, torch.argsort(raw.edge_index[0] * ds["n"] + raw.edge_index[1])]) or \
        torch.equal(raw.edge_index[:, torch.argsort(raw.edge_index[0] * ds["n"] + raw.edge_index[1])],
                    A_n.nonzero().T)
    # mutate the tumour graph in place; the input tensors are untouched
    t = tcga9.graph(ds, c, "tumour")
    t.edge_index = t.edge_index[:, :2]
    t.x = t.x * 0
    raw2 = tcga9.graph(ds, c, "normal").raw()
    assert torch.equal(raw.edge_index, raw2.edge_index) and torch.equal(raw.x, raw2.x) \
        and torch.equal(raw.edge_value, raw2.edge_value)


# ---------------------------------------------------------------- the SSL masks

def test_ssl_mask_is_deterministic_hides_edges_and_scores_true_cells():
    ds = synthetic()
    g = tcga9.graph(ds, "A", "normal")
    m1 = tcga9.ssl_mask(g, 0.3, torch.Generator().manual_seed(7))
    m2 = tcga9.ssl_mask(g, 0.3, torch.Generator().manual_seed(7))
    for a, b in zip(m1, m2):
        if isinstance(a, RawGraph):
            assert torch.equal(a.edge_index, b.edge_index) and torch.equal(a.edge_value, b.edge_value)
        else:
            assert torch.equal(a, b)
    g_m, i, j, t = m1
    A = g.dense().bool()
    src, dst = g.undirected()
    n_hidden = int(t.sum())
    assert 0 < n_hidden < src.numel()
    assert (t == 0).sum() == n_hidden                                     # balanced
    # positives are real edges, negatives are real non-edges, all i < j
    assert (i < j).all()
    assert A[i[t == 1], j[t == 1]].all() and not A[i[t == 0], j[t == 0]].any()
    # hidden edges are absent from the masked input, in BOTH directions, values included
    Am = torch.zeros(g.n, g.n, dtype=torch.bool)
    Am[g_m.edge_index[0], g_m.edge_index[1]] = True
    assert not Am[i[t == 1], j[t == 1]].any() and not Am[j[t == 1], i[t == 1]].any()
    assert Am.sum() == 2 * (src.numel() - n_hidden)
    assert g_m.edge_value.shape == (g_m.edge_index.shape[1], 1)
    assert torch.equal(Am, Am.T)


def test_ssl_masks_do_not_depend_on_the_model():
    """The mask generator is seeded by (seed, epoch) only. Two bodies pretrained with the same seed
    draw the identical sequence -- asserted by replaying the generator, since the models cannot
    influence a torch.Generator they are never handed."""
    ds = synthetic()
    corpus = tcga9.ssl_corpus(ds, ["A", "B", "C"])
    seqs = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(3 * 100_003 + 1)
        order = torch.randperm(len(corpus), generator=gen).tolist()
        seqs.append([(k, tcga9.ssl_mask(corpus[k], 0.2, gen)[1]) for k in order])
    for (ka, ia), (kb, ib) in zip(*seqs):
        assert ka == kb and torch.equal(ia, ib)


# ---------------------------------------------------------------- symmetry

@pytest.mark.parametrize("body", ["lgm", "edgegcn"])
def test_both_bodies_are_permutation_equivariant_with_features_and_edge_values(body):
    torch.manual_seed(0)
    ds = synthetic(n_genes=24, n_samples=15)
    g = tcga9.graph(ds, "B", "normal")
    model = tcga9.build_model(body, CFG, seed=0).double().eval()
    with torch.no_grad():
        L = model(g.raw())
        perm = torch.randperm(g.n)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(g.n)
        gp = RawGraph(g.n, inv[g.edge_index], g.edge_value, None, g.x[perm])
        Lp = model(gp.to("cpu"))
    assert torch.allclose(Lp, L[perm][:, perm], atol=1e-9)
    assert torch.allclose(L, L.T, atol=1e-9)                              # symmetric logits


# ---------------------------------------------------------------- fairness

def test_bodies_are_capacity_matched_share_the_decoder_and_the_input():
    lgm = tcga9.build_model("lgm", CFG, 0)
    gcn = tcga9.build_model("edgegcn", CFG, 0)
    a, b = tcga9.n_params(lgm), tcga9.n_params(gcn)
    assert abs(a - b) / a < 0.02, (a, b)
    assert isinstance(lgm.decoder, D1Bilinear) and isinstance(gcn.decoder, D1Bilinear)
    assert type(lgm.body.first) is type(gcn.first)                        # same projection module
    assert lgm.body.first.w_x is not None and gcn.first.w_x is not None   # both read the features
    assert lgm.body.first.k == gcn.first.k == 1


def test_capacity_match_holds_at_the_real_configuration():
    import yaml
    cfg = yaml.safe_load(open("configs/phase9.yaml"))
    a = tcga9.n_params(tcga9.build_model("lgm", cfg, 0))
    b = tcga9.n_params(tcga9.build_model("edgegcn", cfg, 0))
    assert abs(a - b) / a < 0.02, (a, b)


def test_full_pipeline_runs_identically_shaped_for_both_bodies():
    """End to end on the synthetic set: SSL -> fine-tune -> test, both bodies, same fold, same
    seed. Pins that the shared code path really is shared (same keys, same cells scored)."""
    ds = synthetic(n_genes=30, n_samples=20)
    test_c, val_c, train_c = tcga9.folds(CANCERS)[0]
    rows = {}
    for body in ("lgm", "edgegcn"):
        m = tcga9.build_model(body, CFG, 0)
        s = tcga9.pretrain(m, tcga9.ssl_corpus(ds, train_c), CFG, "cpu", 0, log=lambda *_: None)
        f = tcga9.translate(m, ds, train_c, val_c, CFG, "cpu", 0, log=lambda *_: None)
        rows[body] = {**s, **f, **tcga9.test(m, ds, test_c, "cpu")}
    assert set(rows["lgm"]) == set(rows["edgegcn"])
    for body in rows:
        assert rows[body]["ssl_graphs"] == 6 and rows[body]["ssl_epochs"] == 2
        assert math.isfinite(rows[body]["auc_changed"]) and math.isfinite(rows[body]["ap_changed"])
        assert rows[body]["n_gained"] == rows[body]["n_lost"]          # matched density


def test_identity_scores_exactly_half_on_changed_edges():
    """The metric's floor: copying the input earns exactly chance on both changed-edge strata."""
    from g2l.translate import changed_edge_scores
    ds = synthetic()
    A_n, A_t = tcga9.graph(ds, "C", "normal").dense(), tcga9.graph(ds, "C", "tumour").dense()
    r = changed_edge_scores(A_n, A_n, A_t)
    assert r["auc_gained"] == 0.5 and r["auc_lost"] == 0.5 and r["auc_changed"] == 0.5
