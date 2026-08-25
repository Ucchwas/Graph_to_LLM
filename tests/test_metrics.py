import pytest
import torch

from g2l.metrics import evaluate, evaluate_pairs


def test_perfect_scorer():
    target = torch.zeros(20, 20)
    target[0, 1] = target[1, 0] = target[2, 3] = target[3, 2] = 1.0
    mask = torch.triu(torch.ones(20, 20, dtype=torch.bool), diagonal=1)
    logits = torch.where(target > 0, 10.0, -10.0)
    m = evaluate(logits, target, mask, seed=0)
    assert m["auroc"] == 1.0
    assert m["ap_sparse"] == 1.0
    assert m["ap_balanced"] == 1.0


def test_constant_scorer_is_chance():
    torch.manual_seed(0)
    target = (torch.rand(30, 30) < 0.1).float()
    mask = torch.triu(torch.ones(30, 30, dtype=torch.bool), diagonal=1)
    logits = torch.zeros(30, 30)
    m = evaluate(logits, target, mask, seed=0)
    assert abs(m["auroc"] - 0.5) < 1e-9
    assert abs(m["ap_sparse"] - m["base_rate"]) < 1e-9
    assert abs(m["lift"] - 1.0) < 1e-9


def test_balanced_draw_seeded():
    torch.manual_seed(1)
    target = (torch.rand(40, 40) < 0.05).float()
    mask = torch.triu(torch.ones(40, 40, dtype=torch.bool), diagonal=1)
    logits = torch.randn(40, 40)
    a = evaluate(logits, target, mask, seed=3)
    b = evaluate(logits, target, mask, seed=3)
    assert a == b


def test_single_class_raises():
    target = torch.zeros(10, 10)
    mask = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    with pytest.raises(ValueError):
        evaluate(torch.randn(10, 10), target, mask)


def test_evaluate_pairs_perfect():
    m = evaluate_pairs(torch.full((5,), 3.0), torch.full((5,), -3.0))
    assert m["auroc"] == 1.0 and m["ap"] == 1.0
