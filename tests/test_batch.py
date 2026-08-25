import torch

from g2l.data import collate, mask_matrix


def graph(n, seed):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(n, n, generator=g) < 0.2).float(), diagonal=1)
    A = up + up.T
    A_obs, sup = mask_matrix(A, frac=0.2, seed=seed)
    X = torch.randn(n, 7, generator=g)
    return A_obs, A, sup, X


def test_padding_and_masks():
    b = collate([graph(10, 0), graph(6, 1)])
    assert b.A_obs.shape == (2, 10, 10) and b.X.shape == (2, 10, 7)
    assert b.node_mask[0].all()
    assert b.node_mask[1, :6].all() and not b.node_mask[1, 6:].any()
    assert not b.sup_mask[1, 6:, :].any() and not b.sup_mask[1, :, 6:].any()
    assert (b.A_obs[1, 6:, :] == 0).all() and (b.A_true[1, 6:, :] == 0).all()


def test_single_graph_roundtrip():
    A_obs, A, sup, X = graph(8, 2)
    b = collate([(A_obs, A, sup, X)])
    assert torch.equal(b.A_obs[0], A_obs) and torch.equal(b.sup_mask[0], sup)
