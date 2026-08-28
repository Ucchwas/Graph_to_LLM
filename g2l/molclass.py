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

    def raw(self) -> RawGraph:
        return RawGraph(self.n, self.edge_index, self.edge_attr.float() / BOND_SCALE.to(self.x.device),
                        self.batch, self.x)


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
    """LGM body, mean pooling, linear head. `d` and `dropout` are the only knobs this phase tunes."""

    def __init__(self, d=256, layers=4, heads=8, k=3, dropout=0.0, atom_dims=None, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.body = ISETBody(d, layers, heads, k, dropout, atom_dims=atom_dims)
        self.norm = torch.nn.LayerNorm(d)
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(d, 1)

    def forward(self, b: Batch, n_graphs: int) -> torch.Tensor:
        h = self.norm(self.body(b.raw()))
        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, b.batch, h)
        cnt = torch.zeros(n_graphs, 1, device=h.device, dtype=h.dtype).index_add_(
            0, b.batch, torch.ones_like(h[:, :1]))
        return self.head(self.drop(pooled / cnt.clamp(min=1))).squeeze(-1)


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
    best, best_state, best_epoch, bad, t0 = -1.0, None, 0, 0, time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for b in batches(split["train"], cfg["batch_size"], seed=seed * 1000 + epoch):
            bat = assemble(mols, b).to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(bat, len(b)), bat.y)
            loss.backward()
            opt.step()
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
