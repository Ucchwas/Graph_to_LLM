"""The sequential edge-then-node update, and an honest account of what it does and does not change.

In the default (parallel) layer both updates read the same input states, so information travels
node -> edge in one layer and edge -> node in the next: TWO layers per hop of *structural*
propagation. Sequential mode updates the edge states first and lets the node update read them, so
h_i absorbs h_j within a single layer.

What this does NOT change, and what a naive test would get wrong: the dense node<-node block already
gives every node global REACH after two layers, because node i attends to all node states directly.
So a test that perturbs something far away and watches node 0 move will fire in both modes and prove
nothing. The rate-limited quantity is *topological* information -- which nodes are adjacent to which
-- and that travels only through the incidence channel. These tests therefore pin the mechanism
(the node update consumes updated edge states, and the difference lives entirely in the edge
pathway) and leave the question of whether it helps to the experiment.
"""
import pytest
import torch

from g2l.lgm import LGM, RawGraph

D, TOL = 32, 1e-10


def path(n: int, k: int = 1) -> RawGraph:
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    ei = A.nonzero().T
    return RawGraph(n, ei, torch.ones(ei.shape[1], k, dtype=torch.float64))


def build(sequential: bool, seed: int = 0, edges: bool = True, layers: int = 2) -> LGM:
    return LGM(d=D, layers=layers, heads=4, seed=seed, sequential=sequential, edges=edges).double().eval()


def test_the_two_modes_share_a_parameter_count_and_shapes():
    a, b = build(False), build(True)
    assert {k: tuple(v.shape) for k, v in a.state_dict().items()} == \
           {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_the_modes_differ_only_through_the_edge_pathway():
    """With the edge channel removed the two modes must be bit-identical -- that is what shows the
    difference is the edge update ordering and nothing else."""
    g = path(12)
    with torch.no_grad():
        off = [build(s, edges=False)(g) for s in (False, True)]
        on = [build(s, edges=True)(g) for s in (False, True)]
    assert (off[0] - off[1]).abs().max() < TOL, "modes diverge with no edges: the change is not confined to the edge path"
    assert (on[0] - on[1]).abs().max() > 1e-6, "modes are identical with edges: sequential is a no-op"


def test_sequential_node_update_consumes_the_updated_edge_states():
    """Directly: run one layer's pieces by hand and confirm the node update in sequential mode is
    fed the post-update edge states, not the pre-update ones."""
    m = build(True, layers=1)
    g = path(10)
    layer = m.body.enc.layers[0]
    with torch.no_grad():
        hn, he, src, dst = m.body.first(g)
        he_new = he + layer.attn.edge_from_nodes(layer.ln1(hn), layer.ln1(he), src, dst)
        assert (he_new - he).abs().max() > 1e-6, "the edge update did nothing; the test is vacuous"
        from_old = layer.attn.node_from_all(layer.ln1(hn), layer.ln1(he), src, dst, None)
        from_new = layer.attn.node_from_all(layer.ln1(hn), layer.ln1(he_new), src, dst, None)
        assert (from_old - from_new).abs().max() > 1e-6
        actual = layer(hn, he, src, dst, None)[0]
    expected = hn + from_new
    expected = expected + layer.ff(layer.ln2(expected))
    assert (actual - expected).abs().max() < TOL


# ---- sequential mode must inherit every guarantee the default mode has ----------------------

@pytest.mark.parametrize("sequential", [False, True])
def test_still_permutation_equivariant(sequential):
    g = path(9)
    m = build(sequential)
    perm = torch.randperm(9, generator=torch.Generator().manual_seed(4))
    A = torch.zeros(9, 9, dtype=torch.float64)
    A[g.edge_index[0], g.edge_index[1]] = 1.0
    Ap = A[perm][:, perm]
    with torch.no_grad():
        s = m(g)
        sp = m(RawGraph(9, Ap.nonzero().T, torch.ones(Ap.nonzero().shape[0], 1, dtype=torch.float64)))
    assert (sp - s[perm][:, perm]).abs().max() < TOL


@pytest.mark.parametrize("sequential", [False, True])
def test_still_symmetric_and_size_independent(sequential):
    a, b = build(sequential, seed=0), build(sequential, seed=999)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True)
    assert not missing and not unexpected
    for n in (5, 17, D):
        with torch.no_grad():
            out = b(path(n))
        assert out.shape == (n, n) and torch.isfinite(out).all()
        assert (out - out.T).abs().max() < TOL
