"""Encoder scale, decoders, the masked-cell loss and the shuffled-adjacency control."""
import math

import pytest
import torch
import torch.nn.functional as F

from g2l.data import mask_matrix, rewire_degree_preserving
from g2l.decoders import D1Bilinear, D3PairMLP
from g2l.encoders import E1Linear
from g2l.train import masked_loss, pos_weight_of


def test_encoder_output_norm_matches_T_on_cora():
    try:
        from g2l.data import load_cora
        data = load_cora()
    except Exception as e:
        pytest.skip(f"Cora unavailable: {e}")
    from baselines.common import observed_dense
    from g2l.data import edge_split
    A = observed_dense(edge_split(data, seed=0)[0], data.num_nodes)
    T = 1.0
    torch.manual_seed(0)
    tok = E1Linear(data.num_nodes, 2048, gain=T / math.sqrt(2048))(A)
    assert torch.isfinite(tok).all()
    assert abs(tok.norm(dim=-1).mean().item() - T) / T < 0.25


def test_masked_loss_matches_hand_computation():
    A = torch.zeros(6, 6)
    for i in range(5):
        A[i, i + 1] = A[i + 1, i] = 1.0
    A_obs, sup = mask_matrix(A, frac=0.4, seed=4)
    assert torch.equal(A_obs * (~(sup | sup.T)).float(), A_obs)
    pw = torch.tensor(pos_weight_of(A))
    assert pw.item() == (15 - 5) / 5
    logits = torch.randn(6, 6)
    i, j = sup.nonzero(as_tuple=True)
    by_hand = -(pw * A[i, j] * F.logsigmoid(logits[i, j]) + (1 - A[i, j]) * F.logsigmoid(-logits[i, j])).mean()
    assert torch.allclose(masked_loss(logits[i, j], A[i, j], pw), by_hand)


def test_decoders_pairs_equal_full():
    torch.manual_seed(0)
    Z = torch.randn(10, 16)
    i, j = torch.triu_indices(10, 10, 1)
    d1 = D1Bilinear(16)
    assert torch.allclose(d1.pairs(Z, i, j), d1(Z)[i, j])
    d3 = D3PairMLP(16, hidden=8, chunk=7)
    assert torch.allclose(d3.pairs(Z, i, j), d3(Z)[i, j], atol=1e-6)
    Z.requires_grad_(True)
    d3.pairs(Z, i, j).sum().backward()
    assert Z.grad is not None


def test_rewiring_preserves_degrees():
    g = torch.Generator().manual_seed(0)
    up = torch.triu((torch.rand(40, 40, generator=g) < 0.2).float(), diagonal=1)
    A = up + up.T
    R = rewire_degree_preserving(A, seed=0)
    assert torch.equal(R, R.T) and torch.equal(R.sum(1), A.sum(1)) and R.diagonal().sum() == 0
    assert not torch.equal(R, A)
