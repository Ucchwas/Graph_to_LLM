import torch

from g2l.data import mask_matrix


def ring(n):
    A = torch.zeros(n, n)
    idx = torch.arange(n)
    A[idx, (idx + 1) % n] = 1.0
    A[(idx + 1) % n, idx] = 1.0
    return A


def test_sup_mask_is_upper_triangle():
    _, sup = mask_matrix(ring(50), frac=0.3, seed=1)
    assert not sup.tril().any()


def test_input_hides_both_directions():
    A = ring(50)
    A_obs, sup = mask_matrix(A, frac=0.3, seed=1)
    hidden = sup | sup.T
    assert (A_obs[hidden] == 0).all()
    assert torch.equal(A_obs[~hidden], A[~hidden])
    assert torch.equal(A_obs, A_obs.T)


def test_diagonal_never_masked():
    _, sup = mask_matrix(ring(50), frac=1.0, seed=0)
    assert not sup.diagonal().any()


def test_seed_reproducible():
    A = ring(50)
    _, s1 = mask_matrix(A, seed=7)
    _, s2 = mask_matrix(A, seed=7)
    _, s3 = mask_matrix(A, seed=8)
    assert torch.equal(s1, s2)
    assert not torch.equal(s1, s3)


def test_fraction_approximate():
    n = 200
    _, sup = mask_matrix(ring(n), frac=0.15, seed=0)
    got = sup.sum().item() / (n * (n - 1) / 2)
    assert 0.10 < got < 0.20
