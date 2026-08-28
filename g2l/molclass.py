"""Phase 8C -- ogbg-molhiv graph classification, the one task in this project with an external
reference point.

Every Phase-8 number so far is on a link-prediction task we defined ourselves, so "is 0.93 good?"
had no answer. This runs the official OGB benchmark instead: the published scaffold split, the
official Evaluator, and the same ROC-AUC the leaderboard reports.

Two arms, identical in everything but one input channel:
  `atom`     the nine OGB atom features plus the three bond features -- comparable to the leaderboard
  `noatom`   atom features disabled ONLY; structural node initialisation, topology and bond features
             all retained -- an ablation, NOT comparable to published numbers

The atom encoder is one embedding per categorical feature column, summed (OGB's AtomEncoder). It is
indexed by feature VALUE, never by node index, so it carries no node identity and the checkpoint
stays size-independent.
"""
import functools
import time

import numpy as np
import torch
import torch.nn.functional as F

from g2l.datasets import root
from g2l.lgm import ISETBody, RawGraph, concat_graphs

BOND_SCALE = torch.tensor([3.0, 5.0, 1.0])


class MolC:
    """One molecule: raw graph, categorical atom features, and the binary activity label."""

    def __init__(self, n, edge_index, edge_attr, x, y):
        ev = (edge_attr.float() / BOND_SCALE) if edge_attr is not None and edge_attr.numel() \
            else torch.zeros(edge_index.shape[1], 3)
        self.g = RawGraph(n, edge_index, ev, None, x.long())
        self.y = float(y)


def load_molhiv():
    """The official dataset and its official scaffold split -- no filtering, no re-splitting."""
    from ogb.graphproppred import PygGraphPropPredDataset

    orig = torch.load
    torch.load = functools.partial(orig, weights_only=False)
    try:
        ds = PygGraphPropPredDataset("ogbg-molhiv", root=root())
    finally:
        torch.load = orig
    split = ds.get_idx_split()
    mols = [MolC(int(d.num_nodes), d.edge_index, d.edge_attr, d.x, d.y.item()) for d in ds]
    return mols, {k: v.tolist() for k, v in split.items()}


class GraphClassifier(torch.nn.Module):
    """LGM body, mean pooling over node states, linear head. The body is byte-identical between the
    two arms except for the atom encoder."""

    def __init__(self, d=256, layers=4, heads=8, k=3, dropout=0.0, atom_dims=None, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.body = ISETBody(d, layers, heads, k, dropout, atom_dims=atom_dims)
        self.norm = torch.nn.LayerNorm(d)
        self.head = torch.nn.Linear(d, 1)

    def forward(self, g: RawGraph, n_graphs: int) -> torch.Tensor:
        h = self.norm(self.body(g))
        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, g.batch, h)
        cnt = torch.zeros(n_graphs, 1, device=h.device, dtype=h.dtype).index_add_(
            0, g.batch, torch.ones_like(h[:, :1]))
        return self.head(pooled / cnt.clamp(min=1)).squeeze(-1)


def batches(mols, idx, budget, seed=None):
    """Node-budget block-diagonal batches -- no padding, same batcher as the link-prediction gate."""
    order = list(idx)
    if seed is not None:
        perm = torch.randperm(len(order), generator=torch.Generator().manual_seed(seed)).tolist()
        order = [order[p] for p in perm]
    out, cur, tot = [], [], 0
    for i in order:
        if cur and tot + mols[i].g.n > budget:
            out.append(cur)
            cur, tot = [], 0
        cur.append(i)
        tot += mols[i].g.n
    return out + ([cur] if cur else [])


def assemble(mols, idx, device):
    g = concat_graphs([mols[i].g for i in idx]).to(device)
    y = torch.tensor([mols[i].y for i in idx], device=device)
    return g, y, len(idx)


@torch.no_grad()
def evaluate(model, mols, idx, budget, device, evaluator) -> float:
    model.eval()
    s, y = [], []
    for b in batches(mols, idx, budget):
        g, t, n = assemble(mols, b, device)
        s.append(model(g, n).float().cpu())
        y.append(t.cpu())
    s, y = torch.cat(s).unsqueeze(-1).numpy(), torch.cat(y).unsqueeze(-1).numpy()
    return float(evaluator.eval({"y_true": y, "y_pred": s})["rocauc"])


def train(model, mols, split, cfg, device, seed=0, log=print) -> dict:
    """Selection on validation ROC-AUC; the test split is scored once, from the selected state."""
    from ogb.graphproppred import Evaluator

    ev = Evaluator("ogbg-molhiv")
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]))
    best, best_state, best_epoch, bad, t0 = -1.0, None, 0, 0, time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for b in batches(mols, split["train"], cfg["budget"], seed=seed * 1000 + epoch):
            g, y, n = assemble(mols, b, device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(g, n), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        v = evaluate(model, mols, split["valid"], cfg["budget"], device, ev)
        if v > best:
            best, best_epoch, bad = v, epoch, 0
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        log(f"epoch {epoch:3d} loss {tot / max(nb, 1):.4f} val_rocauc {v:.4f} best {best:.4f}@{best_epoch}")
    model.load_state_dict(best_state)
    test = evaluate(model, mols, split["test"], cfg["budget"], device, ev)
    return {"val_rocauc": best, "test_rocauc": test, "best_epoch": best_epoch, "epochs": epoch + 1,
            "wallclock_s": round(time.time() - t0, 1),
            "n_params": sum(p.numel() for p in model.parameters())}
