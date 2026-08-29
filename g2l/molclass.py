"""Phase 8D -- bounded ogbg-molhiv calibration.

The question is NOT "can we win the leaderboard". It is "is our training harness sound?". We scored
0.6817 against a last-place 0.7549, and GIN reaches 0.7908 with 1/99th our parameters. A gap that
size is more consistent with a broken setup than with an architecture difference, and if the setup
is broken then every other Phase-8 number is suspect too.

So the baseline is REPRODUCED inside our pipeline (g2l/ogbnet.py) rather than quoted: same loader,
batcher, optimiser loop and Evaluator as the LGM. Then only LGM capacity and regularisation are
tuned, on VALIDATION, over three fixed seeds.

The test split is untouched during tuning -- `train()` cannot see it, and `final_test` exists as a
separate call that is made once, on a frozen configuration, only if the validation gate is passed.
"""
import functools
import time

import numpy as np
import torch
import torch.nn.functional as F

from g2l.datasets import root
from g2l.lgm import ISETBody, RawGraph

BOND_SCALE = torch.tensor([3.0, 5.0, 1.0])


@functools.lru_cache(maxsize=1)
def atom_feature_dims() -> tuple:
    from ogb.utils.features import get_atom_feature_dims

    return tuple(get_atom_feature_dims())


def pool(h: torch.Tensor, seg: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Mean of `h` over each segment. Segments with no rows (a single-atom molecule has no bonds)
    come out as zeros rather than NaN."""
    tot = torch.zeros(n_graphs, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, seg, h)
    cnt = torch.zeros(n_graphs, 1, device=h.device, dtype=h.dtype).index_add_(
        0, seg, torch.ones_like(h[:, :1]))
    return tot / cnt.clamp(min=1)


class Batch:
    """One block-diagonal batch, carrying what BOTH model families need: raw categorical atom and
    bond features for the OGB reproduction, and the scaled float edge values the LGM's encoder
    expects. Identical topology and identical graphs either way."""

    def __init__(self, n, edge_index, edge_attr, x, batch, y):
        self.n, self.edge_index, self.edge_attr = n, edge_index, edge_attr
        self.x, self.batch, self.y = x, batch, y

    def to(self, device):
        return Batch(self.n, self.edge_index.to(device), self.edge_attr.to(device),
                     self.x.to(device), self.batch.to(device), self.y.to(device))

    def raw(self, nmask=None, emask=None, mask_tokens: bool = False) -> RawGraph:
        """`mask_tokens` widens the input contract so an attribute can be marked ABSENT instead of
        being silently set to some real category: every atom column gains one extra index meaning
        MASK, and edge values gain a fourth channel that is 1 exactly when the bond attributes are
        hidden. `emask` is per DIRECTED entry and must agree across both directions of an edge --
        `canonical` averages them, so a half-masked edge would land at 0.5 and leak.

        Both arms of the pretraining comparison carry the wider contract, so scratch and pretrained
        are the same architecture (+640 parameters against the k=3 Phase-8D model at d=64)."""
        dev = self.x.device
        x, val = self.x, self.edge_attr.float() / BOND_SCALE.to(dev)
        if not mask_tokens:
            return RawGraph(self.n, self.edge_index, val, self.batch, x)
        flag = torch.zeros(val.shape[0], 1, device=dev, dtype=val.dtype)
        if emask is not None:
            val = val * (~emask).unsqueeze(-1).to(val.dtype)
            flag = emask.to(val.dtype).unsqueeze(-1)
        if nmask is not None:
            x = x.clone()
            x[nmask] = torch.tensor(atom_feature_dims(), device=dev, dtype=x.dtype)
        return RawGraph(self.n, self.edge_index, torch.cat([val, flag], 1), self.batch, x)


class MolC:
    def __init__(self, n, edge_index, edge_attr, x, y):
        self.n, self.edge_index = n, edge_index
        self.edge_attr = edge_attr if edge_attr is not None and edge_attr.numel() \
            else torch.zeros(edge_index.shape[1], 3, dtype=torch.long)
        self.x, self.y = x.long(), float(y)


def load_molhiv():
    from ogb.graphproppred import PygGraphPropPredDataset

    orig = torch.load
    torch.load = functools.partial(orig, weights_only=False)
    try:
        ds = PygGraphPropPredDataset("ogbg-molhiv", root=root())
    finally:
        torch.load = orig
    mols = [MolC(int(d.num_nodes), d.edge_index, d.edge_attr, d.x, d.y.item()) for d in ds]
    return mols, {k: v.tolist() for k, v in ds.get_idx_split().items()}


def batches(idx, size: int, seed: int | None = None):
    """Fixed graph-count batches, matching OGB's batch_size=32 so the reproduction is faithful."""
    order = list(idx)
    if seed is not None:
        perm = torch.randperm(len(order), generator=torch.Generator().manual_seed(seed)).tolist()
        order = [order[p] for p in perm]
    return [order[s:s + size] for s in range(0, len(order), size)]


def assemble(mols, idx) -> Batch:
    off, ei, ea, xs, bt, ys = 0, [], [], [], [], []
    for b, i in enumerate(idx):
        m = mols[i]
        ei.append(m.edge_index + off)
        ea.append(m.edge_attr)
        xs.append(m.x)
        bt.append(torch.full((m.n,), b, dtype=torch.long))
        ys.append(m.y)
        off += m.n
    return Batch(off, torch.cat(ei, 1), torch.cat(ea, 0), torch.cat(xs, 0),
                 torch.cat(bt), torch.tensor(ys))


class LGMClassifier(torch.nn.Module):
    """LGM body -> pooled graph vector -> head.

    `readout="node"` is the Phase-8C/8D behaviour: it pools node states and DISCARDS the final edge
    states, even though every layer computes them. That throws away the one representation this
    architecture has that a node-centric MPNN does not, which is why `readout="nodeedge"` --
    [mean(H_nodes) || mean(H_edges)] -> MLP -> score -- exists to be measured against it.

    The body is constructed FIRST so that, at a given seed, both readouts start from an identical
    body initialisation and the comparison is genuinely paired."""

    def __init__(self, d=256, layers=4, heads=8, k=3, dropout=0.0, atom_dims=None, seed=None,
                 readout="node", head="linear", mask_tokens=False):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        assert readout in ("node", "nodeedge") and head in ("linear", "mlp")
        self.readout, self.mask_tokens = readout, mask_tokens
        if mask_tokens:
            atom_dims = [c + 1 for c in (atom_dims or atom_feature_dims())]
            k = k + 1
        self.body = ISETBody(d, layers, heads, k, dropout, atom_dims=atom_dims)
        self.norm = torch.nn.LayerNorm(d)
        self.enorm = torch.nn.LayerNorm(d) if readout == "nodeedge" else None
        self.drop = torch.nn.Dropout(dropout)
        w = d * (2 if readout == "nodeedge" else 1)
        self.head = torch.nn.Linear(w, 1) if head == "linear" else torch.nn.Sequential(
            torch.nn.Linear(w, d), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(d, 1))

    def forward(self, b: Batch, n_graphs: int, nmask=None, emask=None) -> torch.Tensor:
        hn, he, src, _ = self.body.encode(b.raw(nmask, emask, self.mask_tokens))
        z = pool(self.norm(hn), b.batch, n_graphs)
        if self.readout == "nodeedge":
            # src indexes NODES, so batch[src] is the graph each undirected edge belongs to
            z = torch.cat([z, pool(self.enorm(he), b.batch[src], n_graphs)], dim=-1)
        return self.head(self.drop(z)).squeeze(-1)


@torch.no_grad()
def rocauc(model, mols, idx, size, device, evaluator) -> float:
    model.eval()
    s, y = [], []
    for b in batches(idx, size):
        bat = assemble(mols, b).to(device)
        s.append(model(bat, len(b)).float().cpu())
        y.append(bat.y.cpu())
    s, y = torch.cat(s).unsqueeze(-1).numpy(), torch.cat(y).unsqueeze(-1).numpy()
    return float(evaluator.eval({"y_true": y, "y_pred": s})["rocauc"])


def train(model, mols, split, cfg, device, seed=0, log=print) -> dict:
    """Trains and selects on VALIDATION only. This function never touches split['test'] -- the test
    set is reached solely through `final_test`, once, on a frozen configuration."""
    from ogb.graphproppred import Evaluator

    ev = Evaluator("ogbg-molhiv")
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # Optional, off unless the config asks (Phase 8D/8F/8G configs do not, so their rows still
    # reproduce): linear LR warm-up over `warmup_steps` and gradient-norm clipping at `clip`.
    # Measured on the LGM in Phase 8E at +0.0166 with the seed spread nearly halved. Lives in this
    # ONE shared loop so every family gets it identically or not at all.
    warm, clip = cfg.get("warmup_steps", 0), cfg.get("clip", None)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm)) if warm else None
    best, best_state, best_epoch, bad, t0 = -1.0, None, 0, 0, time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for b in batches(split["train"], cfg["batch_size"], seed=seed * 1000 + epoch):
            bat = assemble(mols, b).to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(bat, len(b)), bat.y)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if sched:
                sched.step()
            tot, nb = tot + loss.item(), nb + 1
        v = rocauc(model, mols, split["valid"], cfg["batch_size"], device, ev)
        if v > best:
            best, best_epoch, bad = v, epoch, 0
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        if epoch % 5 == 0:
            log(f"  epoch {epoch:3d} loss {tot / max(nb, 1):.4f} val {v:.4f} best {best:.4f}@{best_epoch}")
    model.load_state_dict(best_state)
    return {"val_rocauc": best, "best_epoch": best_epoch, "epochs": epoch + 1,
            "wallclock_s": round(time.time() - t0, 1),
            "n_params": sum(p.numel() for p in model.parameters())}


def final_test(model, mols, split, cfg, device) -> float:
    """The single permitted look at the test split, on an already-frozen model."""
    from ogb.graphproppred import Evaluator

    return rocauc(model, mols, split["test"], cfg["batch_size"], device, Evaluator("ogbg-molhiv"))
