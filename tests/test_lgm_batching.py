"""Gate 8.1 -- variable-size batching without padding, and no leakage between batched graphs.

Graphs of different sizes are batched as a block-diagonal disjoint union: node indices are offset,
segment ids recorded, and the dense node<-node block masked off-block. No dummy node is ever
created, so this is masking rather than padding -- and the claim is testable rather than asserted:
a batch of three differently sized graphs must reproduce three single-graph runs exactly, and
perturbing one graph in a batch must leave the others bit-identical.
"""
import torch

from g2l.lgm import LGM, RawGraph, concat_graphs

D, LAYERS, HEADS, TOL = 32, 2, 4, 1e-10
SIZES = [12, 25, 40]


def er(n: int, p: float = 0.2, seed: int = 0) -> RawGraph:
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=g) < p).double()
    A = torch.triu(A, 1)
    A = A + A.T
    A[torch.arange(n), (torch.arange(n) + 1) % n] = 1.0
    A[(torch.arange(n) + 1) % n, torch.arange(n)] = 1.0
    return RawGraph.from_dense(A)


def model():
    return LGM(d=D, layers=LAYERS, heads=HEADS, seed=0).double().eval()


def blocks(z: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    return list(torch.split(z, sizes))


def test_a_batch_reproduces_the_single_graph_runs():
    m, graphs = model(), [er(n, seed=n) for n in SIZES]
    with torch.no_grad():
        single = [m.node_states(g) for g in graphs]
        batched = blocks(m.node_states(concat_graphs(graphs, dtype=torch.float64)), SIZES)
    for n, a, b in zip(SIZES, single, batched):
        assert (a - b).abs().max() < TOL, f"N={n} differs when batched"


def test_perturbing_one_graph_leaves_the_others_untouched():
    """The leakage test: if any information crossed the block-diagonal mask, graph 0 and graph 2
    would move when graph 1's edges change.

    Asserted at float-noise scale rather than bit-identity. Changing graph 1 changes the LENGTH of
    the reductions the batch runs (the dense block's axis and the incidence index_add_), so float
    non-associativity perturbs the other graphs by ~1 ulp -- measured 8.3e-17 and 1.7e-16 against a
    state scale of 0.4, versus 6.6e-01 for the graph that actually changed. The ratio, not the
    absolute, is what separates rounding from leakage."""
    m, graphs = model(), [er(n, seed=n) for n in SIZES]
    other = list(graphs)
    other[1] = er(SIZES[1], p=0.6, seed=99)
    with torch.no_grad():
        a = blocks(m.node_states(concat_graphs(graphs, dtype=torch.float64)), SIZES)
        b = blocks(m.node_states(concat_graphs(other, dtype=torch.float64)), SIZES)
    moved = (a[1] - b[1]).abs().max().item()
    assert moved > 1e-3, "graph 1 did not change; the perturbation was a no-op"
    for k in (0, 2):
        d = (a[k] - b[k]).abs().max().item()
        assert d < TOL and d < moved / 1e6, \
            f"graph {k} moved by {d:.2e} when graph 1 was perturbed (graph 1 moved {moved:.2e})"


def test_batch_order_does_not_matter():
    m, graphs = model(), [er(n, seed=n) for n in SIZES]
    order = [2, 0, 1]
    with torch.no_grad():
        a = blocks(m.node_states(concat_graphs(graphs, dtype=torch.float64)), SIZES)
        b = blocks(m.node_states(concat_graphs([graphs[k] for k in order], dtype=torch.float64)),
                   [SIZES[k] for k in order])
    for slot, k in enumerate(order):
        assert (a[k] - b[slot]).abs().max() < TOL


def test_chunked_pair_decoding_matches_the_dense_decode():
    """D1Bilinear.pairs must never materialise [N, N]; it must still agree with the one that does."""
    m, g = model(), er(60, seed=1)
    iu = torch.triu_indices(60, 60, 1)
    m.decoder.chunk = 97                      # force many chunks
    with torch.no_grad():
        dense = m(g)[iu[0], iu[1]]
        chunked = m.pairs(g, iu[0], iu[1])
    assert (dense - chunked).abs().max() < TOL


def test_the_no_edge_control_removes_the_edge_channel():
    """The control the first-class-edge claim rests on: identical model, edge states deleted."""
    g = er(30, seed=4)
    with torch.no_grad():
        full = LGM(d=D, layers=LAYERS, heads=HEADS, seed=0).double().eval().node_states(g)
        none = LGM(d=D, layers=LAYERS, heads=HEADS, seed=0, edges=False).double().eval().node_states(g)
    assert full.shape == none.shape
    assert (full - none).abs().max() > 1e-6, "the edge channel had no effect on node states"
