"""Phase 8F -- self-supervised pretraining on the official MolHIV TRAINING molecules only.

No external data and no extra molecules: the pretraining corpus is exactly `split["train"]`, the
same graphs the supervised run already sees, so nothing from valid or test reaches the model in any
form and the comparison against a scratch model stays clean.

The task hides the ATTRIBUTES of a random fraction of atoms and bonds -- never the topology -- and
predicts them back. Atom attributes are read off the node states and bond attributes off the EDGE
states, so the objective trains the edge channel directly rather than only through its downstream
effect on nodes. That pairs with the `nodeedge` readout: one supplies a gradient to the edge
representation, the other lets it reach the prediction.

Leak control. A masked bond has its value zeroed and its mask flag raised in BOTH directed entries,
because `lgm.canonical` averages the two directions and a half-masked edge would land at 0.5 and
leak. The masks are drawn against the SAME `torch.unique(min*n + max)` key `canonical` uses, so row
i of the edge states is the same edge as row i of the mask. Prediction targets are read from the
untouched `Batch`, never from the masked copy handed to the model.
"""
import functools
import time

import torch
import torch.nn.functional as F

from g2l.molclass import Batch, assemble, atom_feature_dims, batches


@functools.lru_cache(maxsize=1)
def bond_feature_dims() -> tuple:
    from ogb.utils.features import get_bond_feature_dims

    return tuple(get_bond_feature_dims())


class MaskHeads(torch.nn.Module):
    """One linear classifier per attribute column. Outputs cover the REAL categories only -- the
    extra MASK index the input contract adds is never a prediction target."""

    def __init__(self, d: int):
        super().__init__()
        self.atom = torch.nn.ModuleList([torch.nn.Linear(d, c) for c in atom_feature_dims()])
        self.bond = torch.nn.ModuleList([torch.nn.Linear(d, c) for c in bond_feature_dims()])


def draw(b: Batch, frac: float, gen: torch.Generator):
    """(node mask [n], directed edge mask [M], undirected edge mask [M_u], bond targets [M_u, 3]).

    `uniq`/`inv` reproduce `lgm.canonical` exactly, which is what aligns `umask` with the rows of
    the edge states. `umask[inv]` then masks both directions of an edge together."""
    lo = torch.minimum(b.edge_index[0], b.edge_index[1])
    hi = torch.maximum(b.edge_index[0], b.edge_index[1])
    uniq, inv = torch.unique(lo * b.n + hi, return_inverse=True)
    umask = torch.rand(uniq.numel(), generator=gen) < frac
    tgt = torch.zeros(uniq.numel(), b.edge_attr.shape[1], dtype=torch.long)
    tgt[inv] = b.edge_attr          # both directions of an OGB bond carry identical attributes
    nmask = torch.rand(b.n, generator=gen) < frac
    return nmask, umask[inv], umask, tgt


def canonical_endpoints(b: Batch):
    """(usrc, udst) for the undirected edges in the SAME order `draw` masks them -- the
    torch.unique(min*n + max) key, which is also the order `lgm.canonical` builds edge states.
    The GNN has no edge states, so its bond predictions read the two ENDPOINT node states instead,
    and this is the row alignment that makes those predictions point at the right bonds."""
    lo = torch.minimum(b.edge_index[0], b.edge_index[1])
    hi = torch.maximum(b.edge_index[0], b.edge_index[1])
    uniq = torch.unique(lo * b.n + hi)
    return uniq // b.n, uniq % b.n


def gnn_masked_batch(b: Batch, nmask, emask) -> Batch:
    """The GNN's input contract for hidden attributes: categorical, via the reserved MASK index the
    widened encoders add (index `dims[i]` per column). Same information as the LGM's zero+flag
    channel -- "this attribute is absent" -- expressed in each family's native input type. The
    original Batch is untouched; it still holds the prediction targets."""
    x = b.x.clone()
    x[nmask] = torch.tensor(atom_feature_dims(), device=x.device, dtype=x.dtype)
    ea = b.edge_attr.clone()
    ea[emask] = torch.tensor(bond_feature_dims(), device=ea.device, dtype=ea.dtype)
    return Batch(b.n, b.edge_index, ea, x, b.batch, b.y)


def gnn_ssl_loss(model, heads: MaskHeads, b: Batch, nmask, emask, umask, tgt) -> torch.Tensor:
    """The GNN twin of `ssl_loss`: identical objective, same masks, same targets. Atom attributes
    are read off node states; bond attributes off h_src + h_dst -- the sum, not a concatenation,
    so the prediction is symmetric in the endpoints exactly as the LGM's edge states are."""
    hn = model.node_states(gnn_masked_batch(b, nmask, emask))
    loss = hn.sum() * 0.0
    if bool(nmask.any()):
        for i, h in enumerate(heads.atom):
            loss = loss + F.cross_entropy(h(hn[nmask]), b.x[nmask, i])
    if bool(umask.any()):
        usrc, udst = canonical_endpoints(b)
        hp = hn[usrc[umask]] + hn[udst[umask]]
        for i, h in enumerate(heads.bond):
            loss = loss + F.cross_entropy(h(hp), tgt[umask, i])
    return loss


def ssl_loss(body, heads: MaskHeads, b: Batch, nmask, emask, umask, tgt) -> torch.Tensor:
    """Summed cross-entropy over every masked atom column and every masked bond column."""
    hn, he, _, _ = body.encode(b.raw(nmask, emask, mask_tokens=True))
    loss = hn.sum() * 0.0
    if bool(nmask.any()):
        for i, h in enumerate(heads.atom):
            loss = loss + F.cross_entropy(h(hn[nmask]), b.x[nmask, i])
    if bool(umask.any()):
        for i, h in enumerate(heads.bond):
            loss = loss + F.cross_entropy(h(he[umask]), tgt[umask, i])
    return loss


def _pretrain_loop(step, params, cfg, split, mols, device, seed, log) -> dict:
    """The shared optimiser loop: `step(b, nmask, emask, umask, tgt)` returns the loss. One loop
    for both model families, so mask schedule, epochs, lr, batch order and corpus are equal by
    construction rather than by parallel maintenance."""
    opt = torch.optim.Adam(params, lr=cfg["pretrain_lr"])
    gen = torch.Generator().manual_seed(seed + 10_000)
    t0, last = time.time(), 0.0
    for epoch in range(cfg["pretrain_epochs"]):
        tot, nb = 0.0, 0
        for idx in batches(split["train"], cfg["batch_size"], seed=seed * 1000 + epoch):
            b = assemble(mols, idx)
            nmask, emask, umask, tgt = draw(b, cfg["mask_frac"], gen)
            opt.zero_grad(set_to_none=True)
            loss = step(b.to(device), nmask.to(device), emask.to(device),
                        umask.to(device), tgt.to(device))
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        last = tot / max(nb, 1)
        if epoch % 5 == 0 or epoch == cfg["pretrain_epochs"] - 1:
            log(f"  pretrain epoch {epoch:3d} loss {last:.4f}")
    return {"pretrain_epochs": cfg["pretrain_epochs"], "final_loss": last,
            "mask_frac": cfg["mask_frac"], "wallclock_s": round(time.time() - t0, 1),
            "n_pretrain_graphs": len(split["train"])}


def pretrain(model, mols, split, cfg, device, seed: int = 0, log=print) -> dict:
    """Trains `model.body` (an LGMClassifier's) in place on the training split only. Returns a
    summary; the caller saves the body state dict. Fixed epoch count, no early stopping -- there is
    no held-out set here that would not either be the supervised validation split (which would leak
    selection into it) or a slice carved out of training (which would shrink the corpus for no
    measurement we use)."""
    heads = MaskHeads(model.norm.normalized_shape[0]).to(device)
    model.to(device).train()
    heads.train()
    return _pretrain_loop(
        lambda b, nm, em, um, tg: ssl_loss(model.body, heads, b, nm, em, um, tg),
        list(model.body.parameters()) + list(heads.parameters()),
        cfg, split, mols, device, seed, log)


def pretrain_gnn(model, mols, split, cfg, device, seed: int = 0, log=print) -> dict:
    """The GNN twin of `pretrain`: same loop, same masks, same schedule, same corpus. Trains
    everything except the classification head (atom encoder, convs with their bond encoders,
    batch norms); the head is untouched and stays fresh for fine-tuning, exactly as the LGM's."""
    heads = MaskHeads(model.head.in_features).to(device)
    model.to(device).train()
    heads.train()
    body_params = [p for k, p in model.named_parameters() if not k.startswith("head.")]
    return _pretrain_loop(
        lambda b, nm, em, um, tg: gnn_ssl_loss(model, heads, b, nm, em, um, tg),
        body_params + list(heads.parameters()),
        cfg, split, mols, device, seed, log)
