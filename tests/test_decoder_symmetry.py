"""The decoder must score an undirected pair identically as (i, j) and as (j, i).

The trap this file is written around: `Z W Z^T` is symmetric whenever W is, and D1Bilinear's W is
initialised to 0.1 * I. So the naive assertion `allclose(logits, logits.T)` passes on a freshly
built model no matter what the code does, and only starts failing after training moves W off the
diagonal. Every test here therefore checks symmetry BOTH at init and after gradient steps have
demonstrably made the raw parameter asymmetric.
"""
import pytest
import torch

from g2l.decoders import D1Bilinear
from g2l.lgm import LGM, RawGraph

D, TOL = 24, 1e-10


def er(n, p=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=g) < p).double()
    A = torch.triu(A, 1)
    A = A + A.T
    A[torch.arange(n), (torch.arange(n) + 1) % n] = 1.0
    A[(torch.arange(n) + 1) % n, torch.arange(n)] = 1.0
    return RawGraph.from_dense(A)


def trained_decoder(steps: int = 25) -> D1Bilinear:
    """A decoder whose W is asymmetric, which is the only case where symmetrisation does any work.

    Training alone will not produce one: the gradient of f((W + W^T)/2) with respect to W is
    itself symmetric, so from the symmetric 0.1*I init the antisymmetric half receives exactly zero
    gradient and never moves. An asymmetric W therefore arrives only from outside -- above all from
    a Phase 1-7 checkpoint, every one of which was trained before this fix and whose W is genuinely
    asymmetric. That load is the case these tests must cover, so it is what is simulated here: train
    on the strict upper triangle (as the real loss does), then inject an antisymmetric component the
    way an old state_dict would carry one."""
    torch.manual_seed(0)
    dec = D1Bilinear(D).double()
    Z = torch.randn(30, D, dtype=torch.float64)
    iu = torch.triu_indices(30, 30, 1)
    target = (torch.rand(iu.shape[1], dtype=torch.float64) < 0.2).double()
    opt = torch.optim.Adam(dec.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            dec.pairs(Z, iu[0], iu[1]), target)
        loss.backward()
        opt.step()
    if steps:
        skew = torch.randn(D, D, dtype=torch.float64,
                           generator=torch.Generator().manual_seed(11))
        with torch.no_grad():
            dec.W += skew - skew.T                      # pure antisymmetric, as an old W carries
    return dec


def test_training_alone_cannot_make_W_asymmetric():
    """The property that makes the fix stable: symmetry is preserved by the gradient, not just by
    the init."""
    torch.manual_seed(0)
    dec = D1Bilinear(D).double()
    Z = torch.randn(30, D, dtype=torch.float64)
    iu = torch.triu_indices(30, 30, 1)
    target = (torch.rand(iu.shape[1], dtype=torch.float64) < 0.2).double()
    opt = torch.optim.Adam(dec.parameters(), lr=0.05)
    for _ in range(25):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            dec.pairs(Z, iu[0], iu[1]), target).backward()
        opt.step()
    assert (dec.W - dec.W.T).abs().max() < TOL
    assert not torch.allclose(dec.W, torch.eye(D, dtype=torch.float64) * 0.1), "W never moved"


def test_the_perturbed_parameter_really_is_asymmetric():
    """Guards every test below from passing vacuously: without symmetrisation this W would give a
    visibly asymmetric score matrix."""
    dec = trained_decoder()
    assert (dec.W - dec.W.T).abs().max() > 1e-3
    Z = torch.randn(30, D, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    raw = Z @ dec.W @ Z.T                                # the pre-fix computation
    assert (raw - raw.T).abs().max() > 1e-3, "the perturbation does not reach the output"


@pytest.mark.parametrize("steps", [0, 25])
def test_full_matrix_decode_is_symmetric(steps):
    dec = trained_decoder(steps) if steps else D1Bilinear(D).double()
    Z = torch.randn(30, D, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    L = dec(Z)
    assert (L - L.T).abs().max() < TOL


@pytest.mark.parametrize("steps", [0, 25])
def test_pair_decode_gives_the_same_score_both_ways(steps):
    dec = trained_decoder(steps) if steps else D1Bilinear(D).double()
    Z = torch.randn(30, D, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    iu = torch.triu_indices(30, 30, 1)
    assert (dec.pairs(Z, iu[0], iu[1]) - dec.pairs(Z, iu[1], iu[0])).abs().max() < TOL


def test_pair_decode_matches_the_full_matrix():
    dec = trained_decoder()
    Z = torch.randn(30, D, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    iu = torch.triu_indices(30, 30, 1)
    assert (dec.pairs(Z, iu[0], iu[1]) - dec(Z)[iu[0], iu[1]]).abs().max() < TOL


def test_symmetrising_W_equals_symmetrising_the_output():
    """The identity that makes this change free for Phases 1-7: their metrics already applied
    (L + L.T)/2 post hoc, so their reported numbers are unchanged by moving it into the decoder."""
    dec = trained_decoder()
    Z = torch.randn(30, D, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    raw = Z @ dec.W @ Z.T
    assert ((raw + raw.T) / 2 - dec(Z)).abs().max() < TOL


def test_the_whole_model_is_symmetric_end_to_end():
    m = LGM(d=D, layers=2, heads=4, seed=0).double().eval()
    with torch.no_grad():
        L = m(er(20, seed=1))
    assert (L - L.T).abs().max() < TOL
