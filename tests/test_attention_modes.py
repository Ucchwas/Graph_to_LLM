"""The three attention-balance modes, and the dilution they exist to fix.

Under `joint` the node<-node and node<-edge blocks share one softmax, so they compete for attention
mass on KEY COUNT. On ogbg-molhiv a node has ~26 node keys carrying no structure and ~2.1 edge keys
carrying all of it, which leaves the structural channel ~7% of the mass at init. `count` offsets
each block's logits by -log(its key count) so a block contributes its mean rather than its sum --
still ONE softmax, so a node update is still never a pure neighbourhood aggregate. `split`
normalises the two pools separately and mixes them with a learned per-head gate, which controls the
balance most directly but makes the local half an explicitly normalised neighbourhood pool.

These tests measure the mass rather than asserting the fix works, and require every mode to keep the
architecture's guarantees: permutation equivariance, output symmetry, size independence.
"""
import math

import pytest
import torch

from g2l.lgm import LGM, RawGraph

D, TOL = 32, 1e-10
MODES = ["joint", "count", "split"]


def sparse_graph(n=26, dtype=torch.float64) -> RawGraph:
    """A cycle: degree 2 everywhere, like a molecule's mean degree of 2.14."""
    A = torch.zeros(n, n, dtype=dtype)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
    ei = A.nonzero().T
    return RawGraph(n, ei, torch.ones(ei.shape[1], 1, dtype=dtype))


def build(mode, seed=0, layers=2, edges=True):
    return LGM(d=D, layers=layers, heads=4, seed=seed, attn_mode=mode, sequential=True,
               edges=edges).double().eval()


def measured_edge_mass(mode, n=26) -> float:
    m, g = build(mode), sparse_graph(n)
    layer = m.body.enc.layers[0]
    with torch.no_grad():
        hn, he, src, dst = m.body.first(g)
        return float(layer.attn.edge_mass(layer.ln1(hn), layer.ln1(he), src, dst))


def test_joint_mode_starves_the_structural_channel():
    """The measured problem. On a degree-2 graph of 26 nodes the edge block -- which carries every
    bit of structural information -- gets under a tenth of the attention mass at initialisation."""
    mass = measured_edge_mass("joint")
    assert mass < 0.12, f"expected the joint mode to be diluted, measured {mass:.3f}"
    # and it gets worse as the graph grows, because the dense block grows and the degree does not
    assert measured_edge_mass("joint", n=100) < measured_edge_mass("joint", n=20)


@pytest.mark.parametrize("mode", ["count", "split"])
def test_the_fixes_rebalance_the_channels(mode):
    mass = measured_edge_mass(mode)
    assert 0.3 < mass < 0.7, f"{mode} should start near balance, measured {mass:.3f}"


def test_count_mode_is_size_stable_where_joint_is_not():
    """The point of the -log(count) offset: the structural channel's share should not collapse
    simply because a graph has more nodes."""
    j20, j200 = measured_edge_mass("joint", 20), measured_edge_mass("joint", 200)
    c20, c200 = measured_edge_mass("count", 20), measured_edge_mass("count", 200)
    assert j20 - j200 > 0.03, f"joint did not decay with N ({j20:.3f} -> {j200:.3f})"
    assert abs(c20 - c200) < abs(j20 - j200), \
        f"count decayed as much as joint ({c20:.3f} -> {c200:.3f} vs {j20:.3f} -> {j200:.3f})"


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_stays_permutation_equivariant(mode):
    g, m = sparse_graph(11), build(mode)
    perm = torch.randperm(11, generator=torch.Generator().manual_seed(2))
    A = torch.zeros(11, 11, dtype=torch.float64)
    A[g.edge_index[0], g.edge_index[1]] = 1.0
    Ap = A[perm][:, perm]
    with torch.no_grad():
        s = m(g)
        sp = m(RawGraph(11, Ap.nonzero().T, torch.ones(Ap.nonzero().shape[0], 1, dtype=torch.float64)))
    assert (sp - s[perm][:, perm]).abs().max() < TOL


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_stays_symmetric_and_size_independent(mode):
    a, b = build(mode, seed=0), build(mode, seed=777)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True)
    assert not missing and not unexpected
    for n in (6, 19, D):
        with torch.no_grad():
            out = b(sparse_graph(n))
        assert out.shape == (n, n) and torch.isfinite(out).all()
        assert (out - out.T).abs().max() < TOL


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_handles_a_graph_with_no_edges(mode):
    """Isolated nodes have no incident edge keys; no mode may produce NaN there."""
    g = RawGraph(6, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 1, dtype=torch.float64))
    with torch.no_grad():
        out = build(mode)(g)
    assert torch.isfinite(out).all()


def test_only_split_adds_parameters_and_only_the_gate():
    base = sum(p.numel() for p in build("joint").parameters())
    assert sum(p.numel() for p in build("count").parameters()) == base
    extra = sum(p.numel() for p in build("split").parameters()) - base
    assert extra == 2 * 4, f"split should add one gate per head per layer, got {extra}"
