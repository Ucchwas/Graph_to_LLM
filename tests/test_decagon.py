"""Phase 6: supervision exclusion, the aligned layer split, the shared training loop, the priors,
and (when downloaded) the Decagon split facts."""
import os

import pytest
import torch

pytest.importorskip("torch_geometric")

from g2l.data import mask_matrix
from g2l.datasets import root
from g2l.layers import dense, split_layers
from g2l.metrics import evaluate_edge_split
from g2l.model import build_model
from g2l.priors import Priors
from g2l.shared import layer_rows, train_shared

N = 40


def synthetic_layers(n_layers=6, seed=0):
    """Random layers over one node set, sharing a common core of pairs (so layers overlap)."""
    g = torch.Generator().manual_seed(seed)
    core = torch.triu(torch.rand(N, N, generator=g) < 0.25, 1).nonzero().T
    out = {}
    for k in range(n_layers):
        keep = torch.rand(core.size(1), generator=g) < 0.6
        extra = torch.triu(torch.rand(N, N, generator=g) < 0.05, 1).nonzero().T
        out[f"L{k}"] = torch.cat([core[:, keep], extra], 1)
    return out


def test_mask_exclusion_only_removes_supervision():
    A = dense(synthetic_layers(1)["L0"], N)
    ex = torch.zeros(N, N, dtype=torch.bool)
    ex[:10, :] = ex[:, :10] = True
    A0, s0 = mask_matrix(A, 0.3, seed=1)
    A1, s1 = mask_matrix(A, 0.3, seed=1, exclude=ex)
    assert torch.equal(A0, A1) and not (s1 & ex).any() and torch.equal(s1, s0 & ~ex) and s1.sum() < s0.sum()


def test_split_layers_properties():
    layers, hidden, info = split_layers(N, synthetic_layers(), seed=0, tau=0.0, hidden_frac=0.2,
                                        groups={f"L{k}": ("train", "train", "train", "train", "val", "test")[k] for k in range(6)})
    assert info["groups"] == {"train": 4, "val": 1, "test": 1} and torch.equal(hidden, hidden.T)
    for L in layers:
        A = dense(torch.cat([L.train, L.val_pos, L.test_pos, L.hidden_pos], 1), N).bool()
        assert A.sum() // 2 == L.n_pairs  # the four parts partition the layer
        for e in (L.train, L.val_pos, L.test_pos):  # visible parts avoid the hidden pairs
            assert not hidden[e[0], e[1]].any()
        assert hidden[L.hidden_pos[0], L.hidden_pos[1]].all() and hidden[L.hidden_neg[0], L.hidden_neg[1]].all()
        for e in (L.val_neg, L.test_neg, L.hidden_neg):  # negatives are non-edges of the layer
            assert not A[e[0], e[1]].any() and (e[0] < e[1]).all()
        assert not hidden[L.val_neg[0], L.val_neg[1]].any()
        ex = L.exclude(N, hidden)
        assert ex[L.test_pos[0], L.test_pos[1]].all() and ex[L.hidden_pos[0], L.hidden_pos[1]].all() and not ex[L.train[0], L.train[1]].any()
        tr, va, te = L.split(N, hidden)
        assert te.edge_index.size(1) == 2 * (L.train.size(1) + L.val_pos.size(1)) and te.hidden_pos_edge_label_index is L.hidden_pos


def test_shared_loop_and_priors_on_synthetic_layers():
    layers, hidden, _ = split_layers(N, synthetic_layers(), seed=0, tau=0.0, hidden_frac=0.2,
                                     groups={f"L{k}": ("train", "train", "train", "train", "val", "test")[k] for k in range(6)})
    model = build_model("gnn_direct", N, 16, 1.0, scratch_layers=2, seed=0, kind="gcn")
    cfg = {"mask_frac": 0.15, "lr": 3e-3, "lr_norm": 1e-4, "weight_decay": 0.01, "warmup": 2, "max_epochs": 2, "patience": 5}
    s = train_shared(model, layers, hidden, N, cfg, "cpu", 0, log=lambda *a: None)
    assert s["epochs"] == 2 and 0 <= s["val_auc"] <= 1
    rows = layer_rows(model, layers, hidden, N, "cpu", fracs=(0.5,))
    assert len(rows) == 6 and all("auc_hidden" in r and "auc" in r for r in rows)
    assert all("auc_in0.5" in r for r in rows if r["group"] == "test")
    P = Priors(layers, N)
    L = next(l for l in layers if l.group == "test")
    inp = torch.cat([L.train, L.val_pos], 1)
    pf = P.pair_frequency(L, inp)
    assert not pf[hidden].any() and pf.max() <= 4 and torch.equal(pf, pf.T)
    row = evaluate_edge_split(P.identity(L, inp), L.split(N, hidden))
    assert abs(row["auc"] - 0.5) < 1e-9 and abs(row["auc_hidden"] - 0.5) < 1e-9
    assert P.knn(L, inp).shape == (N, N)
    Lt = next(l for l in layers if l.group == "train")  # leave-one-out: own pairs do not count
    assert P.pair_frequency(Lt, torch.cat([Lt.train, Lt.val_pos], 1)).max() <= 3


def test_decagon_split_facts():
    if not os.path.exists(os.path.join(root(), "Decagon", "processed", "decagon.pt")):
        pytest.skip("Decagon not downloaded")
    from g2l.decagon import load_decagon

    N_, layers, hidden, info, names = load_decagon()
    assert N_ == 645 and info["n_layers"] == 963 and info["groups"] == {"train": 770, "val": 96, "test": 97}
    assert info["max_cross_jaccard"] < 0.2 and info["n_union_pairs"] == 63472
    assert abs(info["n_hidden_pairs"] - 6347) < 200
    assert all(l.n_pairs >= 500 for l in layers) and all(l.hidden_pos.size(1) > 0 for l in layers)
    assert names[layers[0].name]
