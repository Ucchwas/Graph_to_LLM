"""Phase 8G: equal-treatment pretraining for the OGB GNNs, and the widened mask-token encoders.

Same load-bearing risks as the LGM side (tests/test_readout_pretrain.py): the bond a head predicts
must be the bond whose endpoints it reads, a masked bond must vanish from BOTH directed entries,
and a "pretrained" model must provably carry the checkpoint rather than silently fine-tuning from
scratch through a lenient load.
"""
import torch

from g2l.molclass import assemble, atom_feature_dims
from g2l.molpretrain import (MaskHeads, bond_feature_dims, canonical_endpoints, draw,
                             gnn_masked_batch, gnn_ssl_loss)
from g2l.ogbnet import OGBGNN
from tests.test_readout_pretrain import corpus


def test_masked_batch_uses_reserved_indices_in_both_directions():
    b = assemble(corpus(), [0, 1, 2])
    nmask, emask, umask, _ = draw(b, 0.5, torch.Generator().manual_seed(3))
    assert 0 < int(umask.sum()) < umask.numel()
    m = gnn_masked_batch(b, nmask, emask)

    adims = torch.tensor(atom_feature_dims())
    bdims = torch.tensor(bond_feature_dims())
    assert torch.equal(m.x[nmask], adims.expand(int(nmask.sum()), -1))
    assert torch.equal(m.x[~nmask], b.x[~nmask])
    assert torch.equal(m.edge_attr[emask], bdims.expand(int(emask.sum()), -1))
    assert torch.equal(m.edge_attr[~emask], b.edge_attr[~emask])
    # both directions of each undirected edge agree on being masked -- emask came from umask[inv]
    lo = torch.minimum(b.edge_index[0], b.edge_index[1])
    hi = torch.maximum(b.edge_index[0], b.edge_index[1])
    for key in torch.unique(lo * b.n + hi):
        hit = ((lo * b.n + hi) == key).nonzero().flatten()
        assert emask[hit[0]] == emask[hit[1]]
    # the original batch is untouched: it still holds the prediction targets
    assert not torch.equal(m.x, b.x) and not torch.equal(m.edge_attr, b.edge_attr)


def test_endpoint_order_matches_the_bond_targets():
    """canonical_endpoints row r and draw's tgt row r must be the same physical bond."""
    b = assemble(corpus(), [0, 1, 2])
    _, _, umask, tgt = draw(b, 0.5, torch.Generator().manual_seed(0))
    usrc, udst = canonical_endpoints(b)
    assert usrc.numel() == umask.numel() == tgt.shape[0]
    lo = torch.minimum(b.edge_index[0], b.edge_index[1])
    hi = torch.maximum(b.edge_index[0], b.edge_index[1])
    for r in range(usrc.numel()):
        hit = ((lo == usrc[r]) & (hi == udst[r])).nonzero().flatten()
        assert hit.numel() == 2
        assert torch.equal(tgt[r], b.edge_attr[hit[0]])


def test_widened_encoders_accept_the_mask_index_and_count_as_expected():
    base = OGBGNN(kind="gin", d=32, layers=2, seed=0, mask_tokens=False)
    wide = OGBGNN(kind="gin", d=32, layers=2, seed=0, mask_tokens=True)
    extra = (len(atom_feature_dims()) + 2 * len(bond_feature_dims())) * 32   # 2 layers of bonds
    assert sum(p.numel() for p in wide.parameters()) - \
        sum(p.numel() for p in base.parameters()) == extra

    b = assemble(corpus(), [0, 1, 2])
    nmask = torch.ones(b.n, dtype=torch.bool)
    emask = torch.ones(b.edge_index.shape[1], dtype=torch.bool)
    wide.eval()
    out = wide(gnn_masked_batch(b, nmask, emask), 3)      # every attribute hidden: must not crash
    assert torch.isfinite(out).all()


def test_gnn_ssl_loss_trains_body_not_head():
    b = assemble(corpus(), [0, 1, 2])
    m = OGBGNN(kind="gin", d=32, layers=2, seed=0, mask_tokens=True)
    heads = MaskHeads(32)
    nmask, emask, umask, tgt = draw(b, 0.6, torch.Generator().manual_seed(0))
    assert int(nmask.sum()) > 0 and int(umask.sum()) > 0
    gnn_ssl_loss(m, heads, b, nmask, emask, umask, tgt).backward()
    assert m.atom_encoder.atom_embedding_list[0].weight.grad.abs().max() > 0
    assert m.convs[0].bond_encoder.bond_embedding_list[0].weight.grad.abs().max() > 0
    assert all(h.weight.grad.abs().max() > 0 for h in heads.atom + heads.bond)
    assert m.head.weight.grad is None, "the classification head must stay out of the SSL loss"


def test_pretrained_load_carries_the_body_and_keeps_the_head_fresh():
    import types

    from g2l.run_phase8g import load_pretrained

    src = OGBGNN(kind="gcn", d=32, layers=2, seed=0, mask_tokens=True)
    sd = {k: v for k, v in src.state_dict().items() if not k.startswith("head.")}

    dst = OGBGNN(kind="gcn", d=32, layers=2, seed=99, mask_tokens=True)
    head_before = dst.head.weight.clone()
    args = types.SimpleNamespace(model="gcn")
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "ck.pt"
        torch.save(sd, p)
        load_pretrained(dst, args, p)
        for k in sd:
            assert torch.equal(dst.state_dict()[k], sd[k])
        assert torch.equal(dst.head.weight, head_before)

        # a checkpoint with a missing key must be refused, not silently part-loaded
        bad = dict(sd)
        bad.pop(next(iter(bad)))
        torch.save(bad, p)
        try:
            load_pretrained(OGBGNN(kind="gcn", d=32, layers=2, seed=5, mask_tokens=True), args, p)
        except AssertionError:
            pass
        else:
            raise AssertionError("a key-mismatched checkpoint loaded silently")


def test_scratch_final_gnn_is_the_exact_ogb_architecture():
    """A scratch winner's leaderboard row must carry no unused mask rows in its parameter count."""
    import types

    from g2l.run_phase8g import build

    cfg = {"ogb_d": 300, "ogb_layers": 5, "ogb_dropout": 0.5}
    n = lambda m: sum(p.numel() for p in m.parameters())
    scratch = build(types.SimpleNamespace(model="gin", stage="final", init="scratch"), cfg, 0)
    pre = build(types.SimpleNamespace(model="gin", stage="final", init="pretrained"), cfg, 0)
    assert n(scratch) == 1_885_506          # the Phase-8D reproduction, to the parameter
    assert n(pre) == 1_885_506 + (9 + 5 * 3) * 300
