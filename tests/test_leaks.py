"""The canaries. Identity must score at chance; structural features must be a
function of the observed graph only."""
import torch

from g2l.bias import spd_matrix
from g2l.data import mask_matrix
from g2l.metrics import evaluate


def random_graph(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(n, n, generator=g) < p).float(), diagonal=1)
    return up + up.T


def test_identity_scores_at_chance():
    A = random_graph(120, 0.05, seed=0)
    A_obs, sup = mask_matrix(A, frac=0.15, seed=0)
    m = evaluate(A_obs * 100.0, A, sup, seed=0)
    assert abs(m["auroc"] - 0.5) < 1e-9, "identity beat chance: masking leaks"
    assert abs(m["lift"] - 1.0) < 1e-9


def test_spd_depends_only_on_observed_graph():
    A1 = random_graph(60, 0.08, seed=1)
    A_obs, sup = mask_matrix(A1, frac=0.2, seed=0)
    hidden = sup | sup.T
    A2 = A1.clone()
    A2[hidden] = 1.0 - A2[hidden]  # flip every held-out cell
    A2_obs = A2 * (~hidden).float()
    assert torch.equal(A_obs, A2_obs)
    assert torch.equal(spd_matrix(A_obs), spd_matrix(A2_obs))


def test_spd_reacts_to_observed_edges():
    A = torch.zeros(6, 6)
    A[0, 1] = A[1, 0] = A[1, 2] = A[2, 1] = 1.0
    d = spd_matrix(A, max_dist=8)
    assert d[0, 2] == 2 and d[0, 0] == 0
    assert d[0, 5] == 9  # unreachable bucket = max_dist + 1
    A[2, 5] = A[5, 2] = 1.0
    assert spd_matrix(A, max_dist=8)[0, 5] == 3
