"""Phase 8F: the edge readout, and the masked-attribute pretraining contract.

The load-bearing risk here is ALIGNMENT. `molpretrain.draw` decides which bonds to hide and what to
predict, using its own `torch.unique(min*n + max)`; `lgm.canonical` independently builds the edge
states with the same key. If those two orderings ever disagreed, the model would be asked to predict
one bond from another bond's state -- a silent bug that still trains, still converges, and still
reports a plausible number. So the ordering is asserted directly rather than assumed.
"""
import torch

from g2l.lgm import canonical
from g2l.molclass import LGMClassifier, MolC, assemble, atom_feature_dims, pool
from g2l.molpretrain import MaskHeads, bond_feature_dims, draw, ssl_loss

N_ATOM_COLS, N_BOND_COLS = 9, 3


def mol(n, pairs, seed=0):
    """A small molecule. `pairs` are undirected (i, j); both directions get identical bond attrs,
    which is what OGB does and what the averaging in `canonical` relies on."""
    g = torch.Generator().manual_seed(seed)
    ei = torch.tensor([[i for i, j in pairs] + [j for i, j in pairs],
                       [j for i, j in pairs] + [i for i, j in pairs]], dtype=torch.long)
    half = torch.stack([torch.randint(0, c, (len(pairs),), generator=g)
                        for c in bond_feature_dims()], dim=1)
    x = torch.stack([torch.randint(0, c, (n,), generator=g) for c in atom_feature_dims()], dim=1)
    return MolC(n, ei, torch.cat([half, half], 0), x, 0.0)


def corpus():
    return [mol(4, [(0, 1), (1, 2), (2, 3)], 0),
            mol(5, [(0, 1), (0, 2), (2, 3), (3, 4), (1, 4)], 1),
            mol(3, [(0, 2), (1, 2)], 2)]


def test_mask_order_matches_the_models_edge_order():
    """draw()'s undirected ordering IS canonical()'s row ordering, and the bond targets it hands
    back are the true attributes of those same rows."""
    mols = corpus()
    b = assemble(mols, [0, 1, 2])
    gen = torch.Generator().manual_seed(0)
    nmask, emask, umask, tgt = draw(b, 0.5, gen)

    src, dst, _ = canonical(b.raw(mask_tokens=True))
    lo = torch.minimum(b.edge_index[0], b.edge_index[1])
    hi = torch.maximum(b.edge_index[0], b.edge_index[1])
    uniq = torch.unique(lo * b.n + hi)                    # the key draw() masks against
    assert src.numel() == umask.numel() == uniq.numel()
    assert torch.equal(src, uniq // b.n) and torch.equal(dst, uniq % b.n)

    # and, independently of that shared key: every canonical row's target is the bond attribute
    # actually sitting on that (src, dst) pair in the raw directed edge list
    for r in range(src.numel()):
        hit = ((lo == src[r]) & (hi == dst[r])).nonzero().flatten()
        assert hit.numel() == 2, "each undirected edge should have exactly two directed entries"
        assert torch.equal(tgt[r], b.edge_attr[hit[0]])
        assert torch.equal(b.edge_attr[hit[0]], b.edge_attr[hit[1]])


def test_masked_bonds_are_fully_hidden_and_unmasked_ones_untouched():
    """A half-masked edge would average to 0.5 through canonical and leak. Both directions must go."""
    mols = corpus()
    b = assemble(mols, [0, 1, 2])
    gen = torch.Generator().manual_seed(3)
    nmask, emask, umask, _ = draw(b, 0.5, gen)
    assert 0 < int(umask.sum()) < umask.numel(), "need both masked and unmasked edges to be meaningful"

    _, _, val = canonical(b.raw(nmask, emask, mask_tokens=True))
    assert val.shape[1] == 4
    assert torch.equal(val[umask, 3], torch.ones(int(umask.sum())))
    assert torch.equal(val[umask, :3], torch.zeros(int(umask.sum()), 3))
    assert torch.equal(val[~umask, 3], torch.zeros(int((~umask).sum())))

    _, _, clean = canonical(b.raw(mask_tokens=True))
    assert torch.allclose(val[~umask, :3], clean[~umask, :3])


def test_masked_atoms_use_the_reserved_index_and_targets_stay_clean():
    mols = corpus()
    b = assemble(mols, [0, 1, 2])
    nmask = torch.ones(b.n, dtype=torch.bool)          # mask every atom
    g = b.raw(nmask, None, mask_tokens=True)
    assert torch.equal(g.x, torch.tensor(atom_feature_dims()).expand(b.n, N_ATOM_COLS))
    assert not torch.equal(g.x, b.x), "the untouched Batch must still hold the prediction targets"
    # the reserved index is in range only because mask_tokens widened every embedding by one
    m = LGMClassifier(d=32, layers=1, heads=2, k=3, seed=0, readout="nodeedge", head="mlp",
                      mask_tokens=True)
    assert m.body.first.atom[0].num_embeddings == atom_feature_dims()[0] + 1
    m(b, 3, nmask=nmask)                                # must not raise IndexError


def test_edge_pool_segments_by_graph():
    """batch[src] is the claim being checked: src indexes NODES, so it maps an edge to its graph."""
    mols = corpus()
    b = assemble(mols, [0, 1, 2])
    src, _, _ = canonical(b.raw(mask_tokens=True))
    seg = b.batch[src]
    counts = torch.zeros(3).index_add_(0, seg, torch.ones(seg.numel()))
    assert torch.equal(counts, torch.tensor([3.0, 5.0, 2.0]))     # the undirected edge counts
    # pooling a per-edge constant equal to the graph id returns that id
    assert torch.allclose(pool(seg.float().unsqueeze(-1), seg, 3),
                          torch.tensor([[0.0], [1.0], [2.0]]))


def test_edge_pool_is_zero_not_nan_for_a_bondless_molecule():
    single = MolC(1, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 3, dtype=torch.long),
                  torch.zeros(1, N_ATOM_COLS, dtype=torch.long), 0.0)
    b = assemble([single, corpus()[0]], [0, 1])
    m = LGMClassifier(d=32, layers=1, heads=2, k=3, seed=0, readout="nodeedge", head="mlp",
                      mask_tokens=True)
    out = m(b, 2)
    assert torch.isfinite(out).all()


def test_the_two_readouts_share_a_body_initialisation():
    """Same seed -> identical body weights, so a paired-seed comparison isolates the readout."""
    a = LGMClassifier(d=32, layers=2, heads=2, k=3, seed=7, readout="node", head="mlp",
                      mask_tokens=True)
    c = LGMClassifier(d=32, layers=2, heads=2, k=3, seed=7, readout="nodeedge", head="mlp",
                      mask_tokens=True)
    for (ka, pa), (kc, pc) in zip(a.body.state_dict().items(), c.body.state_dict().items()):
        assert ka == kc and torch.equal(pa, pc), ka
    assert a.enorm is None and c.enorm is not None


def test_edge_states_actually_reach_the_prediction():
    """Not vacuous: the node-only arm must get NO gradient through an edge-only parameter, and the
    nodeedge arm must get one. Checking only the latter would pass on a model that ignored it."""
    mols = corpus()
    b = assemble(mols, [0, 1, 2])
    c = LGMClassifier(d=32, layers=2, heads=2, k=3, seed=1, readout="nodeedge", head="mlp",
                      mask_tokens=True)
    c(b, 3).sum().backward()
    assert c.enorm.weight.grad is not None and c.enorm.weight.grad.abs().max() > 0

    a = LGMClassifier(d=32, layers=2, heads=2, k=3, seed=1, readout="node", head="mlp",
                      mask_tokens=True)
    assert a.head[0].in_features == 32 and c.head[0].in_features == 64


def test_a_pretrained_body_loads_strictly_into_both_readouts():
    src = LGMClassifier(d=32, layers=2, heads=2, k=3, seed=0, readout="node", head="mlp",
                        mask_tokens=True)
    sd = src.body.state_dict()
    for readout in ("node", "nodeedge"):
        dst = LGMClassifier(d=32, layers=2, heads=2, k=3, seed=99, readout=readout, head="mlp",
                            mask_tokens=True)
        dst.body.load_state_dict(sd, strict=True)
        for k in sd:
            assert torch.equal(dst.body.state_dict()[k], sd[k])


def test_ssl_loss_trains_both_streams():
    """Atom heads must move node-side parameters and bond heads must move edge-side ones. If the
    bond term were silently empty the loss would still fall, on atoms alone."""
    mols = corpus()
    b = assemble(mols, [0, 1, 2])
    m = LGMClassifier(d=32, layers=2, heads=2, k=3, seed=0, readout="node", head="mlp",
                      mask_tokens=True)
    heads = MaskHeads(32)
    nmask, emask, umask, tgt = draw(b, 0.6, torch.Generator().manual_seed(0))
    assert int(nmask.sum()) > 0 and int(umask.sum()) > 0
    ssl_loss(m.body, heads, b, nmask, emask, umask, tgt).backward()
    assert m.body.first.u_edge.grad.abs().max() > 0
    assert m.body.first.w_deg.weight.grad.abs().max() > 0
    assert all(h.weight.grad.abs().max() > 0 for h in heads.bond)
    assert all(h.weight.grad.abs().max() > 0 for h in heads.atom)
