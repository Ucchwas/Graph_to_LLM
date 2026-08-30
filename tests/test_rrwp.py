"""Phase 10 -- the RRWP / RWSE study: definition, leakage, symmetry, size independence, capacity.

The claim under test is narrow: the ONLY thing added to the LGM is a learned per-head bias on the
dense node<-node attention logits, derived from the K-step relative random-walk probabilities of the
edge set the model is handed; the baselines get that object's node-level diagonal (RWSE) on their
node input. So the tests are correspondingly narrow:

  * the encodings are what they claim to be (equal to explicit matrix powers), and equivariant;
  * they see only the edges the MODEL sees -- the leakage rule, asserted on both benchmarks;
  * every "+encoding" arm is bit-identical to its base at initialisation, so each contrast is a
    strict superset and the phase measures the encoding rather than a reinitialisation;
  * permutation equivariance, output symmetry, size independence and batching survive the addition;
  * all four TCGA arms stay capacity-matched within 2 %.

Nothing here asserts that RRWP helps. Whether it does is what the GPU runs measure.
"""
import pytest
import torch

from g2l.lgm import LGM, RawGraph, concat_graphs
from g2l.multigraph import EdgeGCNBaseline
from g2l.rrwp import rrwp, rwse, transition

K = 16
TOL = 1e-10


def er(n: int, p: float = 0.3, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """Erdos-Renyi with a Hamiltonian cycle forced in, so no node is isolated and the graph is not
    vertex-transitive (which would collapse every output to a constant -- see test_equivariance)."""
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=g) < p).to(dtype)
    A = torch.triu(A, 1)
    A = A + A.T
    idx, nxt = torch.arange(n), (torch.arange(n) + 1) % n
    A[idx, nxt] = A[nxt, idx] = 1.0
    return A


def graph(n, A=None, x_dim=0, seed=0):
    A = er(n, seed=seed) if A is None else A
    x = None if not x_dim else torch.randn(n, x_dim, dtype=torch.float64,
                                          generator=torch.Generator().manual_seed(seed + 77))
    return RawGraph(n, A.nonzero().T, None, None, x)


# ---------------------------------------------------------------- what the encodings are

def test_rrwp_equals_explicit_powers_of_the_transition_matrix():
    n = 11
    A = er(n, seed=1)
    ei = A.nonzero().T
    M = transition(ei, n, dtype=torch.float64)
    R = rrwp(ei, n, 6, dtype=torch.float64, normalise=False)
    assert R.shape == (n, n, 6)
    P = torch.eye(n, dtype=torch.float64)
    for k in range(6):
        assert torch.allclose(R[..., k], P, atol=1e-12), f"channel {k} is not M^{k}"
        P = P @ M
    assert torch.equal(R[..., 0], torch.eye(n, dtype=torch.float64))     # channel 0 is the identity
    assert torch.allclose(M.sum(1), torch.ones(n, dtype=torch.float64))  # row-stochastic


def test_rwse_is_the_diagonal_of_the_same_powers():
    n = 11
    ei = er(n, seed=2).nonzero().T
    M = transition(ei, n, dtype=torch.float64)
    S = rwse(ei, n, 6, dtype=torch.float64, normalise=False)
    assert S.shape == (n, 6)
    for k in range(1, 7):
        assert torch.allclose(S[:, k - 1], torch.matrix_power(M, k).diagonal(), atol=1e-12)


def test_normalisation_rescales_each_channel_without_changing_its_direction():
    """What normalisation is and is not: a positive per-channel, per-graph scalar. It cannot change
    a channel's sign pattern, its zeros, or the ratio between two entries of the same channel."""
    n = 13
    ei = er(n, seed=11).nonzero().T
    raw = rrwp(ei, n, 5, dtype=torch.float64, normalise=False)
    nrm = rrwp(ei, n, 5, dtype=torch.float64, normalise=True)
    assert torch.equal(raw[..., 0], nrm[..., 0]), "channel 0 is already O(1) and must be untouched"
    for k in range(1, 5):
        nz = raw[..., k] > 0
        ratio = (nrm[..., k][nz] / raw[..., k][nz])
        assert ratio.std() < 1e-9 and ratio.mean() > 0, f"channel {k} was not a pure rescale"
        assert torch.equal(nrm[..., k] == 0, raw[..., k] == 0)
        rms = nrm[..., k].pow(2).mean().sqrt()
        assert abs(rms.item() - 1.0) < 1e-9, f"channel {k} RMS is {rms:.3e}, not 1"


def test_normalisation_makes_the_scale_independent_of_graph_size_and_degree():
    """The reason it exists: raw channels shrink like 1/N, so a bias built from them is inert on a
    large graph. Normalised channels are O(1) at every size."""
    raws, nrms = [], []
    for n in (20, 200, 1000):
        ei = er(n, p=min(0.9, 20 / n), seed=n).nonzero().T
        raws.append(rrwp(ei, n, 5, normalise=False)[..., 1:].pow(2).mean().sqrt().item())
        nrms.append(rrwp(ei, n, 5, normalise=True)[..., 1:].pow(2).mean().sqrt().item())
    assert raws[0] / raws[-1] > 5, f"raw scale did not fall with N: {raws}"
    assert max(nrms) / min(nrms) < 1.05, f"normalised scale still depends on N: {nrms}"


def test_normalisation_is_per_graph_not_per_batch():
    """A batch-wide statistic would couple graphs. Each block must be normalised by its own scale,
    so a batched RRWP still equals the per-graph ones exactly."""
    gs = [graph(n, seed=n) for n in (9, 30)]
    b = concat_graphs(gs, dtype=torch.float64)
    R = rrwp(b.edge_index, b.n, K, dtype=torch.float64, batch=b.batch)
    off = 0
    for g in gs:
        sl = slice(off, off + g.n)
        assert torch.allclose(R[sl, sl], rrwp(g.edge_index, g.n, K, dtype=torch.float64), atol=1e-12)
        off += g.n


def test_an_isolated_node_gives_a_zero_row_rather_than_a_nan():
    A = torch.zeros(4, 4, dtype=torch.float64)
    A[0, 1] = A[1, 0] = A[2, 3] = A[3, 2] = 1.0        # node-pair components, none isolated
    A = torch.cat([A, torch.zeros(4, 1, dtype=torch.float64)], 1)
    A = torch.cat([A, torch.zeros(1, 5, dtype=torch.float64)], 0)        # node 4 is isolated
    R = rrwp(A.nonzero().T, 5, 4, dtype=torch.float64)
    assert torch.isfinite(R).all()
    assert R[4, :, 1:].abs().max() == 0.0


def test_rrwp_and_rwse_are_permutation_equivariant():
    n = 12
    A = er(n, seed=3)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(4))
    R = rrwp(A.nonzero().T, n, K, dtype=torch.float64)
    Rp = rrwp(A[perm][:, perm].nonzero().T, n, K, dtype=torch.float64)
    assert torch.allclose(Rp, R[perm][:, perm], atol=1e-12)
    S = rwse(A.nonzero().T, n, K, dtype=torch.float64)
    Sp = rwse(A[perm][:, perm].nonzero().T, n, K, dtype=torch.float64)
    assert torch.allclose(Sp, S[perm], atol=1e-12)


def test_rrwp_is_block_diagonal_across_a_batch():
    """Why batching needs no per-graph loop: edges never cross graphs, so M and every M^k are
    block-diagonal and a batched RRWP already equals the per-graph ones.

    `batch` must be passed -- it is what keeps the normaliser per graph. Both model call sites do
    (g2l/lgm.py), and the model-level batch-equivalence and cross-graph-perturbation tests below
    pin that end to end; here the point is the off-block structure."""
    gs = [graph(n, seed=n) for n in (7, 11, 5)]
    b = concat_graphs(gs, dtype=torch.float64)
    R = rrwp(b.edge_index, b.n, K, dtype=torch.float64, batch=b.batch)
    off = 0
    for g in gs:
        sl = slice(off, off + g.n)
        assert torch.allclose(R[sl, sl], rrwp(g.edge_index, g.n, K, dtype=torch.float64), atol=1e-12)
        cross = R[sl].clone()
        cross[:, sl] = 0.0
        assert cross.abs().max() == 0.0, "RRWP leaked across the block diagonal"
        off += g.n


# ---------------------------------------------------------------- leakage

def test_rrwp_sees_only_the_edges_the_model_sees_tcga():
    """The TCGA leak rule. `ssl_mask` removes hidden edges from edge_index in BOTH directions, and
    RRWP is computed from that edge_index inside the forward pass -- so a hidden pair's one-step
    probability is exactly 0 and no power of M can reintroduce the edge itself."""
    from g2l import tcga9

    X = torch.randn(40, 25, generator=torch.Generator().manual_seed(5)) * 2 + 5
    g = tcga9.graph_from_expression(X, 0.05)
    g_m, i, j, t = tcga9.ssl_mask(g, 0.3, torch.Generator().manual_seed(9))
    hid_i, hid_j = i[t == 1], j[t == 1]
    assert hid_i.numel() > 0

    M_full = transition(g.edge_index, g.n)
    M_seen = transition(g_m.edge_index, g_m.n)
    assert M_full[hid_i, hid_j].min() > 0, "vacuous: the hidden pairs were not edges to begin with"
    assert M_seen[hid_i, hid_j].abs().max() == 0.0
    assert M_seen[hid_j, hid_i].abs().max() == 0.0        # both directions
    assert not torch.allclose(M_seen, M_full)

    # and what the model actually computes is the masked one
    R_seen = rrwp(g_m.edge_index, g_m.n, K)
    assert R_seen[hid_i, hid_j, 1].abs().max() == 0.0
    assert not torch.allclose(R_seen, rrwp(g.edge_index, g.n, K))


def test_rrwp_is_unchanged_by_molhiv_attribute_masking():
    """The molhiv counterpart: SSL there hides atom and bond ATTRIBUTES, never topology. So RRWP is
    identical between the masked and unmasked input and there is nothing to leak. Asserted rather
    than assumed, because a future change to the masking would silently break the guarantee."""
    b = _mol_batch()
    from g2l.molpretrain import draw

    nmask, emask, umask, tgt = draw(b, 0.4, torch.Generator().manual_seed(6))
    assert bool(emask.any()) and bool(nmask.any())
    masked, plain = b.raw(nmask, emask, mask_tokens=True), b.raw()
    assert torch.equal(masked.edge_index, plain.edge_index)
    assert torch.equal(rrwp(masked.edge_index, masked.n, K), rrwp(plain.edge_index, plain.n, K))
    # non-vacuity: the attributes really were hidden
    assert not torch.equal(masked.edge_value[:, :3], plain.edge_value[:, :3])


# ---------------------------------------------------------------- zero init: every arm is a superset

def _mol_batch(n_mols=6, seed=0):
    """A synthetic molhiv-shaped batch: 9 categorical atom columns, 3 categorical bond columns.
    Built by hand rather than loaded, because `ogb.graphproppred` pulls in scipy, which the dev
    laptop's Application Control policy blocks (see g2l/ogbnet.py)."""
    from g2l.molclass import Batch, atom_feature_dims
    from g2l.molpretrain import bond_feature_dims

    adims, bdims = torch.tensor(atom_feature_dims()), torch.tensor(bond_feature_dims())
    g = torch.Generator().manual_seed(seed)
    off, ei, ea, xs, bt = 0, [], [], [], []
    for m in range(n_mols):
        n = int(torch.randint(4, 14, (1,), generator=g))
        A = er(n, p=0.25, seed=seed * 100 + m, dtype=torch.float32)
        e = A.nonzero().T
        ei.append(e + off)
        # each categorical column must stay inside ITS OWN dimension -- OGB's atom columns run
        # [119, 5, 12, 12, 10, 6, 6, 2, 2] and bonds [5, 6, 2], so a uniform draw overflows the
        # narrow ones and the embedding lookup raises
        ea.append((torch.rand(e.shape[1], 3, generator=g) * bdims).long())
        xs.append((torch.rand(n, 9, generator=g) * adims).long())
        bt.append(torch.full((n,), m, dtype=torch.long))
        off += n
    return Batch(off, torch.cat(ei, 1), torch.cat(ea, 0), torch.cat(xs, 0), torch.cat(bt),
                 torch.rand(n_mols, generator=g))


def test_the_lgm_with_rrwp_is_bit_identical_to_the_lgm_at_initialisation():
    g = graph(14, x_dim=2, seed=7)
    a = LGM(d=64, layers=2, heads=4, k=1, x_dim=2, seed=3).double().eval()
    b = LGM(d=64, layers=2, heads=4, k=1, x_dim=2, seed=3, rrwp_k=K).double().eval()
    with torch.no_grad():
        assert torch.equal(a(g), b(g)), "the RRWP arm does not start from its base"
    assert b.body.enc.layers[0].attn.w_rrwp.abs().max() == 0.0
    assert sum(p.numel() for p in b.parameters()) - sum(p.numel() for p in a.parameters()) \
        == 2 * 4 * K                                             # layers x heads x K


def test_the_edge_gcn_with_rwse_is_bit_identical_to_the_edge_gcn_at_initialisation():
    g = graph(14, x_dim=2, seed=7)
    a = EdgeGCNBaseline(64, layers=2, k=1, x_dim=2, seed=3).double().eval()
    b = EdgeGCNBaseline(64, layers=2, k=1, x_dim=2, seed=3, rwse_k=K).double().eval()
    with torch.no_grad():
        assert torch.equal(a(g), b(g))
    assert b.first.w_rwse.abs().max() == 0.0


@pytest.mark.parametrize("kind", ["gin", "gcn"])
def test_the_ogb_gnns_with_rwse_are_bit_identical_at_initialisation(kind):
    from g2l.ogbnet import OGBGNN

    b = _mol_batch()
    a = OGBGNN(kind=kind, d=64, layers=2, dropout=0.0, seed=3, mask_tokens=True).eval()
    c = OGBGNN(kind=kind, d=64, layers=2, dropout=0.0, seed=3, mask_tokens=True, rwse_k=K).eval()
    with torch.no_grad():
        assert torch.equal(a(b, 6), c(b, 6))
    assert c.w_rwse.abs().max() == 0.0
    assert sum(p.numel() for p in c.parameters()) - sum(p.numel() for p in a.parameters()) == 64 * K


@pytest.mark.parametrize("base,plus,kw", [
    (lambda **k: LGM(d=64, layers=2, heads=4, k=1, x_dim=2, seed=3, **k), "rrwp_k", {}),
    (lambda **k: EdgeGCNBaseline(64, layers=2, k=1, x_dim=2, seed=3, **k), "rwse_k", {}),
])
def test_the_plain_state_dict_loads_into_the_encoding_arm_as_a_strict_superset(base, plus, kw):
    a, b = base(**kw), base(**{plus: K}, **kw)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert not unexpected, f"the arm dropped keys: {unexpected}"
    assert missing and all("rrwp" in m or "rwse" in m for m in missing), missing
    for m in missing:                                   # the new keys carry no N-dependent shape
        assert 14 not in tuple(dict(b.named_parameters())[m].shape)


# ---------------------------------------------------------------- symmetry and equivariance

def test_the_lgm_with_rrwp_stays_permutation_equivariant_and_symmetric():
    n = 16
    A = er(n, seed=8)
    x = torch.randn(n, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    ev = torch.rand(int(A.sum()), 1, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(2))
    ei = A.nonzero().T
    # a symmetric edge value, so the permuted graph really is the same weighted graph
    V = torch.zeros(n, n, dtype=torch.float64)
    V[ei[0], ei[1]] = ev[:, 0]
    V = (V + V.T) / 2
    m = LGM(d=64, layers=2, heads=4, k=1, x_dim=2, seed=0, rrwp_k=K).double().eval()

    def run(P):
        e = P.nonzero().T
        return m(RawGraph(n, e, V[e[0], e[1]].unsqueeze(-1), None, x if P is A else None))

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(3))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    with torch.no_grad():
        L = m(RawGraph(n, ei, ev, None, x))
        eip = inv[ei]
        Lp = m(RawGraph(n, eip, V[ei[0], ei[1]].unsqueeze(-1), None, x[perm]))
    assert torch.allclose(Lp, L[perm][:, perm], atol=1e-9)
    assert torch.allclose(L, L.T, atol=1e-9)


def test_edge_index_column_order_still_does_not_matter_with_rrwp():
    g = graph(13, x_dim=2, seed=9)
    m = LGM(d=64, layers=2, heads=4, k=1, x_dim=2, seed=0, rrwp_k=K).double().eval()
    cols = torch.randperm(g.edge_index.shape[1], generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        a = m(g)
        b = m(RawGraph(g.n, g.edge_index[:, cols], None, None, g.x))
    assert (a - b).abs().max() < TOL


def test_the_rrwp_bias_actually_changes_the_output_once_trained():
    """Non-vacuity for every equivalence above: with non-zero bias weights the arm is a different
    function, so the zero-init equalities are a property of the initialisation, not of a bias that
    can never do anything."""
    g = graph(14, x_dim=2, seed=7)
    a = LGM(d=64, layers=2, heads=4, k=1, x_dim=2, seed=3).double().eval()
    b = LGM(d=64, layers=2, heads=4, k=1, x_dim=2, seed=3, rrwp_k=K).double().eval()
    with torch.no_grad():
        for layer in b.body.enc.layers:
            layer.attn.w_rrwp.normal_(0.0, 5.0, generator=torch.Generator().manual_seed(4))
        # measured 1.0e-2 at this scale against a decoder-logit range of ~0.1 (4.7e-4 at std 0.5)
        assert (a(g) - b(g)).abs().max() > 1e-3


def test_the_rrwp_weights_receive_gradient():
    g = graph(14, x_dim=2, seed=7)
    m = LGM(d=64, layers=2, heads=4, k=1, x_dim=2, seed=3, rrwp_k=K).double()
    m(g).pow(2).mean().backward()
    for layer in m.body.enc.layers:
        w = layer.attn.w_rrwp
        assert w.grad is not None and w.grad.abs().sum() > 0, "the RRWP bias is not trained"


# ---------------------------------------------------------------- size independence and batching

def test_one_rrwp_checkpoint_runs_at_every_size():
    a = LGM(d=32, layers=2, heads=4, k=1, seed=0, rrwp_k=K).double().eval()
    b = LGM(d=32, layers=2, heads=4, k=1, seed=12345).double().eval()
    b = LGM(d=32, layers=2, heads=4, k=1, seed=12345, rrwp_k=K).double().eval()
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True)
    assert not missing and not unexpected
    before = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    for n in (7, 23, 32, 101):                    # 32 == d_model, deliberately
        with torch.no_grad():
            out = b(graph(n, seed=n))
        assert out.shape == (n, n) and torch.isfinite(out).all() and out.std() > 1e-9
        assert {k: tuple(v.shape) for k, v in b.state_dict().items()} == before


def test_a_batch_reproduces_the_single_graph_runs_with_rrwp():
    sizes = [12, 25, 40]
    gs = [graph(n, seed=n) for n in sizes]
    m = LGM(d=32, layers=2, heads=4, k=1, seed=0, rrwp_k=K).double().eval()
    with torch.no_grad():
        single = [m.node_states(g) for g in gs]
        batched = torch.split(m.node_states(concat_graphs(gs, dtype=torch.float64)), sizes)
    for n, a, b in zip(sizes, single, batched):
        assert (a - b).abs().max() < TOL, f"N={n} differs when batched with RRWP on"


def test_perturbing_one_graph_in_a_batch_leaves_the_others_untouched_with_rrwp():
    sizes = [12, 25, 40]
    gs = [graph(n, seed=n) for n in sizes]
    other = list(gs)
    other[1] = graph(sizes[1], A=er(sizes[1], p=0.7, seed=99), seed=99)
    m = LGM(d=32, layers=2, heads=4, k=1, seed=0, rrwp_k=K).double().eval()
    with torch.no_grad():
        for layer in m.body.enc.layers:                     # a bias that is actually doing work
            layer.attn.w_rrwp.normal_(0.0, 5.0, generator=torch.Generator().manual_seed(5))
        a = torch.split(m.node_states(concat_graphs(gs, dtype=torch.float64)), sizes)
        b = torch.split(m.node_states(concat_graphs(other, dtype=torch.float64)), sizes)
    moved = (a[1] - b[1]).abs().max().item()
    assert moved > 1e-3, "the perturbation was a no-op"
    for k in (0, 2):
        d = (a[k] - b[k]).abs().max().item()
        assert d < TOL and d < moved / 1e6, f"graph {k} moved by {d:.2e} (graph 1 moved {moved:.2e})"


# ---------------------------------------------------------------- fairness

def test_all_four_tcga_arms_are_capacity_matched_within_two_percent():
    import yaml

    from g2l import tcga9

    cfg = yaml.safe_load(open("configs/phase10_tcga.yaml"))
    cnt = {k: tcga9.n_params(tcga9.build_model(k, cfg, 0)) for k in tcga9.KINDS}
    assert set(cnt) == set(cfg["bodies"])
    ks = list(cnt)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            gap = abs(cnt[a] - cnt[b]) / cnt[a]
            assert gap < 0.02, f"{a} {cnt[a]:,} vs {b} {cnt[b]:,} = {100 * gap:.2f}%"
    # each encoding arm is its base plus only the encoding
    assert cnt["lgm_rrwp"] - cnt["lgm"] == cfg["layers"] * cfg["heads"] * cfg["rrwp_k"]
    assert cnt["edgegcn_rwse"] > cnt["edgegcn"]


def test_the_two_gcn_arms_share_a_width_so_only_the_encoding_differs():
    import yaml

    from g2l import tcga9

    cfg = yaml.safe_load(open("configs/phase10_tcga.yaml"))
    a, b = tcga9.build_model("edgegcn", cfg, 0), tcga9.build_model("edgegcn_rwse", cfg, 0)
    assert a.first.w_deg.out_features == b.first.w_deg.out_features
    assert a.first.w_rwse is None and b.first.w_rwse is not None
    lgm, lgm_r = tcga9.build_model("lgm", cfg, 0), tcga9.build_model("lgm_rrwp", cfg, 0)
    assert lgm.body.enc.layers[0].attn.w_rrwp is None
    assert lgm_r.body.enc.layers[0].attn.w_rrwp.shape == (cfg["heads"], cfg["rrwp_k"])
    assert lgm_r.body.first.w_rwse is None, "the LGM's only structural addition is the bias"


# ---------------------------------------------------------------- memory at the real TCGA size

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_memory_at_the_real_tcga_size():
    """Forward + backward at N = 2000 with 19,990 edges, the Phase-9/10 configuration, with and
    without RRWP. The added tensor is [N, N, K] float32 = 256 MB plus one [N, N, H] bias per layer;
    the assertion is that the delta stays inside a gigabyte, so an 80 GB H100 is never at risk."""
    import yaml

    from g2l import tcga9

    cfg = yaml.safe_load(open("configs/phase10_tcga.yaml"))
    n = 2000
    g = torch.Generator().manual_seed(0)
    iu = torch.triu_indices(n, n, 1)
    pick = torch.randperm(iu.shape[1], generator=g)[:9995]
    i, j = iu[0][pick], iu[1][pick]
    ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
    ev = torch.randn(ei.shape[1], 1, generator=g)
    x = torch.randn(n, 2, generator=g)
    raw = RawGraph(n, ei, ev, None, x).to("cuda")

    peaks = {}
    for kind in ("lgm", "lgm_rrwp"):
        m = tcga9.build_model(kind, cfg, 0).to("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m(raw).pow(2).mean().backward()
        peaks[kind] = torch.cuda.max_memory_allocated() / 2 ** 20
        del m
        torch.cuda.empty_cache()
    delta = peaks["lgm_rrwp"] - peaks["lgm"]
    print(f"\nN=2000 peak MiB: lgm {peaks['lgm']:.0f}, lgm_rrwp {peaks['lgm_rrwp']:.0f}, "
          f"delta {delta:.0f}")
    assert delta < 1024, f"RRWP added {delta:.0f} MiB at N=2000"
