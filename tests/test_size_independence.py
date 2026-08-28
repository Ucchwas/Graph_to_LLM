"""Gate 8.0b -- one checkpoint, many graph sizes.

The claim is behavioural, so the test is behavioural: take a state_dict from one model, load it
STRICTLY into a differently-seeded second model, and run that at several N.

Deliberately NOT tested by scanning parameter shapes for a dimension equal to some graph's N. That
rule is unsound in both directions -- it flags every [d, d] tensor the moment a probe graph has
N = d_model, and it misses a genuinely N-tied model whose dimension happens to match no probe. One
probe below is chosen with N exactly equal to d_model to pin that the sound criterion passes where
the shape scan would have failed.

Probe graphs are Erdos-Renyi, not cycles or other vertex-transitive graphs: those collapse to a
constant output (see tests/test_equivariance.py) and every assertion here would pass vacuously.
"""
import pytest
import torch

from g2l.lgm import LGM, RawGraph

D, LAYERS, HEADS = 32, 2, 4
SIZES = [7, 23, 101, D]          # D deliberately included: N == d_model


def er(n: int, p: float = 0.25, seed: int = 0) -> RawGraph:
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=g) < p).double()
    A = torch.triu(A, 1)
    A = A + A.T
    A[torch.arange(n), (torch.arange(n) + 1) % n] = 1.0   # no isolated nodes
    A[(torch.arange(n) + 1) % n, torch.arange(n)] = 1.0
    return RawGraph.from_dense(A)


def build(seed: int) -> LGM:
    return LGM(d=D, layers=LAYERS, heads=HEADS, seed=seed).double().eval()


def shapes(m: LGM) -> dict:
    return {k: tuple(v.shape) for k, v in m.state_dict().items()}


@pytest.fixture(scope="module")
def loaded() -> LGM:
    """A differently-seeded model carrying the first model's weights, so a pass cannot come from
    the two models being identical anyway."""
    a, b = build(0), build(12345)
    before = {k: v.clone() for k, v in b.state_dict().items()}
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True)
    assert not missing and not unexpected
    assert any(not torch.equal(before[k], v) for k, v in b.state_dict().items()), \
        "the two seeds produced identical weights; the test would be vacuous"
    return b


def test_state_dict_transfers_strictly_between_independently_built_models(loaded):
    assert shapes(loaded) == shapes(build(999))


@pytest.mark.parametrize("n", SIZES)
def test_one_checkpoint_runs_at_every_size(loaded, n):
    with torch.no_grad():
        out = loaded(er(n, seed=n))
    assert out.shape == (n, n)
    assert torch.isfinite(out).all()
    assert out.std() > 1e-6, "output is constant; the probe graph is degenerate"


def test_the_parameter_set_is_unchanged_by_running_at_many_sizes(loaded):
    """Catches lazily materialised per-N parameters or buffers, which a one-shot shape scan of a
    freshly built model cannot see."""
    before = shapes(loaded)
    for n in SIZES:
        with torch.no_grad():
            loaded(er(n, seed=n))
        assert shapes(loaded) == before, f"the state_dict changed after a forward at N={n}"


def test_the_same_parameters_receive_gradient_at_every_size():
    """Catches a dead N-conditional branch: a parameter that only trains at some sizes."""
    m = build(0)
    got = {}
    for n in SIZES:
        m.zero_grad(set_to_none=True)
        m(er(n, seed=n)).pow(2).mean().backward()
        got[n] = {k for k, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    ref = got[SIZES[0]]
    assert ref, "no parameter received gradient"
    for n in SIZES[1:]:
        assert got[n] == ref, f"N={n} trains a different parameter set: {ref ^ got[n]}"


def test_no_parameter_is_indexed_by_node_identity():
    """The structural counterpart of the behavioural tests: the constructor never receives N, so no
    parameter can be a per-node table. Phases 4-7's gnn_direct fails this by construction --
    GCNConv(N -> d)'s first weight is one column per node index."""
    import inspect
    assert "n_nodes" not in inspect.signature(LGM.__init__).parameters
    assert "num_nodes" not in inspect.signature(LGM.__init__).parameters
