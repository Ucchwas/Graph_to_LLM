"""Phase 4: the GNN backbone (`gnn` arm) and the `gt` body flags."""
import pytest
import torch

pytest.importorskip("torch_geometric")

from g2l.model import build_model, param_groups
from g2l.train import make_optimizer

N, D = 30, 32


def graph(n=N, p=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(n, n, generator=g) < p).float(), diagonal=1)
    return up + up.T


@pytest.mark.parametrize("kind", ["gcn", "sage", "gin", "gat"])
def test_gnn_shapes_groups_and_gradients(kind):
    A = graph()
    m = build_model("gnn", N, D, 1.0, seed=0, scratch_layers=2, kind=kind)
    assert m(A).shape == (N, N)
    g = param_groups(m)
    assert g["scratch"] and g["main"] and not g["bias_table"] and not g["rmsnorm"]
    opt = make_optimizer(m, 1e-3, 1e-4, 0.01)
    assert len(opt.param_groups) == 2
    m(A).sum().backward()
    body = list(m.body.parameters())
    assert all(p.grad is not None for p in body) and sum(p.grad.abs().sum() for p in body) > 0
    assert m.encoder.proj.weight.grad.abs().sum() > 0


def test_gnn_reads_the_input_graph_only():
    A = graph(40, 0.1, 3)
    m = build_model("gnn", 40, D, 1.0, seed=0, scratch_layers=2, kind="gcn").eval()
    B = A.clone()
    i, j = A.nonzero()[0]
    B[i, j] = B[j, i] = 0.0  # drop one edge: the body's message passing must notice
    with torch.no_grad():
        assert torch.equal(m(A), m(A.clone()))
        assert not torch.allclose(m(A), m(B))
    with pytest.raises(AssertionError):
        build_model("gnn", N, D, 1.0, seed=0, kind="gcn", bias=True)


def test_gnn_body_is_permutation_equivariant():
    A = graph()
    m = build_model("gnn", N, D, 1.0, seed=0, scratch_layers=2, kind="gcn").eval()
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        tokens = m.encoder(A)
        H, Hp = m.body(tokens, A), m.body(tokens[perm], A[perm][:, perm])
    assert torch.allclose(H[perm], Hp, atol=1e-5)


def test_gt_frozen_flag_heads_and_independent_layers():
    m = build_model("gt", N, 128, 1.0, seed=0, scratch_layers=2, frozen=True, bias=True)
    assert m.body.heads == 2 and m.bias.bias_table.shape[0] == 2
    g = param_groups(m)
    assert g["rmsnorm"] and not g["scratch"] and g["bias_table"]
    assert all(p.requires_grad == (".norm" in n) for n, p in m.body.enc.named_parameters())
    m2 = build_model("gt", N, 128, 1.0, seed=0, scratch_layers=2)
    l0, l1 = m2.body.enc.layers
    assert not torch.equal(l0.linear1.weight, l1.linear1.weight)
    assert all(p.requires_grad for p in m2.body.parameters()) and param_groups(m2)["scratch"]
