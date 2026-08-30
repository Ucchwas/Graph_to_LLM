"""Phase 8 -- the multi-graph corpus, node-budget batching, and Gate 8.2.

Every graph benchmark this repo has used so far (Cora, Amazon Photo, ogbl-ddi, PPT-Ohmnet, the
Phase-7 TCGA graphs) is a single fixed-N graph with no edge values, so neither claim Phase 8 makes
can be tested on them: "one checkpoint, many sizes" needs graphs of different N, and "edge states
are first-class" needs edges that carry something. The OGB molecular sets supply both -- variable N
and genuine 3-dim bond features -- which is why they come first rather than last.

Gate 8.2 asks the two questions that can kill the phase:
  1. does ONE checkpoint transfer across sizes at all, or is it at chance on unseen graphs?
  2. does the edge channel carry anything, measured against the no-edge control?
The control shares the LGM's parameter count exactly -- `edges=False` empties the edge states at
run time and changes no shape -- so a difference cannot be a capacity artefact.
"""
import functools
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from g2l.data import mask_matrix
from g2l.datasets import root
from g2l.lgm import RawGraph

BOND_SCALE = torch.tensor([3.0, 5.0, 1.0])   # ogb bond features: type, stereo, is_conjugated
MIN_NODES, MIN_EDGES = 4, 3


class Mol:
    """One graph: dense A for masking and metrics, plus the raw directed edge list and its values."""

    def __init__(self, n, edge_index, edge_value):
        self.n, self.edge_index, self.edge_value = n, edge_index, edge_value
        self.A = torch.zeros(n, n)
        self.A[edge_index[0], edge_index[1]] = 1.0

    def observed(self, hidden: torch.Tensor) -> RawGraph:
        """The model's input: the graph minus the hidden pairs. edge_index is rebuilt from the
        observed edges only, so held-out cells never reach the model (the leak rule, by
        construction, exactly as in every earlier phase)."""
        keep = ~hidden[self.edge_index[0], self.edge_index[1]]
        return RawGraph(self.n, self.edge_index[:, keep], self.edge_value[keep])


def load_mol(name: str = "ogbg-molhiv", limit: int | None = None) -> list[Mol]:
    """OGB graph-property datasets. Their processed files pickle PyG classes, which torch >= 2.6
    refuses under weights_only -- same workaround as datasets.ogb_dataset."""
    from ogb.graphproppred import PygGraphPropPredDataset

    orig = torch.load
    torch.load = functools.partial(orig, weights_only=False)
    try:
        ds = PygGraphPropPredDataset(name, root=root())
    finally:
        torch.load = orig
    out = []
    for k, d in enumerate(ds):
        if limit and len(out) >= limit:
            break
        n, ei = int(d.num_nodes), d.edge_index
        if n < MIN_NODES or ei.shape[1] < 2 * MIN_EDGES:
            continue
        ev = (d.edge_attr.float() / BOND_SCALE) if d.edge_attr is not None else torch.ones(ei.shape[1], 1)
        out.append(Mol(n, ei, ev))
    return out


def split(mols: list[Mol], seed: int = 0, fracs=(0.8, 0.1)) -> tuple[list, list, list]:
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(len(mols), generator=g).tolist()
    a, b = int(fracs[0] * len(p)), int((fracs[0] + fracs[1]) * len(p))
    return [mols[i] for i in p[:a]], [mols[i] for i in p[a:b]], [mols[i] for i in p[b:]]


def batches(mols: list[Mol], budget: int, seed: int | None = None) -> list[list[int]]:
    """Group graph indices so the total node count of a batch stays under `budget`. No padding:
    a batch is a block-diagonal disjoint union whose length is whatever its members sum to."""
    idx = list(range(len(mols)))
    if seed is not None:
        idx = torch.randperm(len(mols), generator=torch.Generator().manual_seed(seed)).tolist()
    out, cur, tot = [], [], 0
    for i in idx:
        if cur and tot + mols[i].n > budget:
            out.append(cur)
            cur, tot = [], 0
        cur.append(i)
        tot += mols[i].n
    return out + ([cur] if cur else [])


def assemble(mols: list[Mol], idx: list[int], frac: float, seed: int):
    """Mask each graph, then concatenate. Returns (batched RawGraph, supervised cell indices in
    batch coordinates, targets). The mask is per graph and seeded per graph, so a graph gets the
    same held-out cells however it is batched."""
    off, ei, ev, batch, si, sj, y, sz = 0, [], [], [], [], [], [], []
    for b, i in enumerate(idx):
        m = mols[i]
        _, sup = mask_matrix(m.A, frac, seed=seed + i)
        g = m.observed(sup | sup.T)
        ei.append(g.edge_index + off)
        ev.append(g.edge_value)
        batch.append(torch.full((m.n,), b, dtype=torch.long))
        a, c = sup.nonzero(as_tuple=True)
        si.append(a + off)
        sj.append(c + off)
        y.append(m.A[a, c])
        sz.append(torch.full((a.numel(),), m.n))
        off += m.n
    return (RawGraph(off, torch.cat(ei, 1), torch.cat(ev, 0), torch.cat(batch)),
            torch.cat(si), torch.cat(sj), torch.cat(y), torch.cat(sz))


# ---------------------------------------------------------------- non-learned references

class GCNBaseline(torch.nn.Module):
    """The external baseline: size-independent message passing, everything else held fixed.

    This is NOT part of the LGM (Phase 8 forbids a convolution before or inside it) -- it is the
    competitor. It reuses the LGM's node featuriser, its decoder-input LayerNorm and its decoder, so
    the only difference from the `noedge` arm is the body: GCN message passing instead of the
    incidence transformer's dense node<-node attention.

    Two deliberate choices, both stated because both could otherwise flatter the LGM.
    (1) The node input is the LGM's full structural featuriser, not a bare scalar. A literal
        GCN(1 -> d) would be handicapped by receiving strictly less input than the arm it is
        compared against, and the comparison would measure the featuriser rather than the body.
        Size independence -- the property that matters -- is identical either way.
    (2) The width is chosen to MATCH the LGM's parameter count (see `matched_width`), because the
        LGM's per-layer cost is far higher at equal d and an unmatched baseline would lose on
        capacity rather than on architecture.

    GCNConv cannot consume edge values, which is inherent to it. So GCN vs `noedge` isolates the
    body at equal information, and GCN vs `lgm` additionally includes the edge channel."""

    needs_graph = True

    def __init__(self, d: int, layers: int = 4, k: int = 1, dropout: float = 0.0, seed: int | None = None,
                 x_dim: int = 0):
        from torch_geometric.nn import GCNConv

        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        from g2l.decoders import D1Bilinear
        from g2l.lgm import NodeEdgeProjection

        self.first = NodeEdgeProjection(d, k=k, x_dim=x_dim)
        self.convs = torch.nn.ModuleList([GCNConv(d, d) for _ in range(layers)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(d) for _ in range(layers)])
        self.dec_norm = torch.nn.LayerNorm(d)
        torch.nn.init.constant_(self.dec_norm.weight, 1 / d ** 0.5)
        torch.nn.init.zeros_(self.dec_norm.bias)
        self.decoder = D1Bilinear(d)
        self.dropout = dropout

    def node_states(self, g: RawGraph, z=None) -> torch.Tensor:
        h, _, src, dst = self.first(g, z)
        ei = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.gelu(conv(norm(h), ei)), self.dropout, self.training)
        return self.dec_norm(h)

    def forward(self, g: RawGraph, z=None) -> torch.Tensor:
        return self.decoder(self.node_states(g, z))

    def pairs(self, g: RawGraph, i, j, z=None) -> torch.Tensor:
        return self.decoder.pairs(self.node_states(g, z), i, j)


class EdgeGCNLayer(torch.nn.Module):
    """One OGB-style edge-aware GCN layer: the bond attributes enter every neighbour message.

    Reproduces the convolution from OGB's `examples/graphproppred/mol/conv.py` (that file ships in
    the examples repo, not the pip package, so it could not be diffed against source here):

        x        = W h
        m_(j->i) = norm_ij * relu(x_j + e_ij)          norm_ij = (deg_i deg_j)^-1/2, deg = indeg + 1
        h'_i     = sum_j m_(j->i) + relu(x_i + root) / deg_i

    Written with plain scatter ops rather than MessagePassing so the arithmetic is inspectable.
    `e_ij` is the shared edge state re-projected per layer, which is the counterpart of OGB giving
    each layer its own BondEncoder."""

    def __init__(self, d: int):
        super().__init__()
        self.linear = torch.nn.Linear(d, d)
        self.edge_proj = torch.nn.Linear(d, d)
        self.root = torch.nn.Parameter(torch.zeros(d))

    def forward(self, h, e_dir, src, dst, n):
        x = self.linear(h)
        e = self.edge_proj(e_dir)
        deg = torch.zeros(n, device=h.device, dtype=h.dtype).index_add_(
            0, dst, torch.ones(dst.numel(), device=h.device, dtype=h.dtype)) + 1.0
        dis = deg.pow(-0.5)
        msg = (dis[src] * dis[dst]).unsqueeze(-1) * F.relu(x[src] + e)
        out = torch.zeros_like(x).index_add_(0, dst, msg)
        return out + F.relu(x + self.root) / deg.unsqueeze(-1)


class EdgeGCNBaseline(torch.nn.Module):
    """The fair baseline: message passing that CAN see the bond attributes.

    The previous `GCNBaseline` was fairly configured but structurally unable to consume edge values,
    so its gap to the LGM confounded architecture with edge access. This arm removes that single
    unfairness and changes nothing else: identical node featuriser, identical edge encoder
    (`NodeEdgeProjection`), identical decoder-input LayerNorm and decoder, identical residual /
    pre-norm wrapper. Only the body differs -- edge-augmented message passing instead of incidence
    attention."""

    needs_graph = True

    def __init__(self, d: int, layers: int = 4, k: int = 1, dropout: float = 0.0, seed: int | None = None,
                 x_dim: int = 0, rwse_k: int = 0):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        from g2l.decoders import D1Bilinear
        from g2l.lgm import NodeEdgeProjection

        # rwse_k > 0 (Phase 10): the shared projection also adds the random-walk structural
        # encoding of each node -- the message-passing counterpart of the LGM's RRWP bias
        self.first = NodeEdgeProjection(d, k=k, x_dim=x_dim, rwse_k=rwse_k)
        self.convs = torch.nn.ModuleList([EdgeGCNLayer(d) for _ in range(layers)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(d) for _ in range(layers)])
        self.dec_norm = torch.nn.LayerNorm(d)
        torch.nn.init.constant_(self.dec_norm.weight, 1 / d ** 0.5)
        torch.nn.init.zeros_(self.dec_norm.bias)
        self.decoder = D1Bilinear(d)
        self.dropout = dropout

    def node_states(self, g: RawGraph, z=None) -> torch.Tensor:
        h, e, src, dst = self.first(g, z)
        # both directions of every undirected edge carry the same edge state
        s = torch.cat([src, dst])
        t = torch.cat([dst, src])
        e_dir = torch.cat([e, e])
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.gelu(conv(norm(h), e_dir, s, t, g.n)), self.dropout, self.training)
        return self.dec_norm(h)

    def forward(self, g: RawGraph, z=None) -> torch.Tensor:
        return self.decoder(self.node_states(g, z))

    def pairs(self, g: RawGraph, i, j, z=None) -> torch.Tensor:
        return self.decoder.pairs(self.node_states(g, z), i, j)


BODIES = {"gcn": GCNBaseline, "edgegcn": EdgeGCNBaseline}


def matched_width(target_params: int, layers: int, k: int, kind: str = "gcn",
                  lo: int = 64, hi: int = 2048, refine: bool = False, **kw) -> int:
    """Width whose parameter count is nearest `target_params`, scanned at step 8. `kw` is forwarded
    to the body (e.g. x_dim), so the match counts the same input path the run will use.

    `refine` adds a step-1 pass around the step-8 winner. Parameters grow ~d^2, so at small widths
    one step of 8 moves the count by several percent, and Phase 9 needs the tighter match (0.17 %
    rather than 0.83 %). It is OPT-IN because turning it on unconditionally silently changes the
    Phase-8 baseline widths -- gcn 800 -> 799, edgegcn 592 -> 596 -- and those widths are part of
    committed, published rows. Default off keeps every Phase-8 row reproducible from this code."""
    cls = BODIES[kind]

    def gap(d):
        return abs(sum(p.numel() for p in cls(d, layers, k, **kw).parameters()) - target_params)

    coarse = min(range(lo, hi + 1, 8), key=gap)
    if not refine:
        return coarse
    return min(range(max(lo, coarse - 8), min(hi, coarse + 8) + 1), key=gap)


PRIORS = ("common_neighbours", "degree", "neg_degree", "random")


def prior_scores(mols, frac, seed, kind="common_neighbours"):
    """Non-learned references on exactly the same held-out cells the model is scored on.

    Two of these run backwards on molecules, which is why `neg_degree` is declared as its own arm
    rather than obtained by flipping a sign after seeing the result. Common neighbours scores 0.462
    (below chance): atoms sharing a neighbour sit at ring or bond-angle distance and are precisely
    the ones NOT bonded. Preferential attachment scores 0.229, because a high-degree atom is
    valence-saturated and less available to bond -- so its negation, at 0.771, is the strongest
    non-learned bar on this corpus and the one the gate must clear."""
    g = torch.Generator().manual_seed(seed)
    s, y = [], []
    for i, m in enumerate(mols):
        A_obs, sup = mask_matrix(m.A, frac, seed=seed + i)
        a, c = sup.nonzero(as_tuple=True)
        d = A_obs.sum(1)
        v = {"common_neighbours": lambda: (A_obs @ A_obs)[a, c],
             "degree": lambda: d[a] * d[c],
             "neg_degree": lambda: -(d[a] * d[c]),
             "random": lambda: torch.rand(a.numel(), generator=g)}[kind]()
        s.append(v)
        y.append(m.A[a, c])
    return torch.cat(s), torch.cat(y)


def score(y: torch.Tensor, s: torch.Tensor) -> dict:
    y, s = y.cpu().numpy().astype(int), s.detach().float().cpu().numpy()
    if y.min() == y.max():
        return {"auc": float("nan"), "ap": float("nan"), "n": len(y), "pos": int(y.sum())}
    return {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
            "n": len(y), "pos": int(y.sum())}


# ---------------------------------------------------------------- train / evaluate

@torch.no_grad()
def evaluate(model, mols, budget, frac, seed, device, by_size=False) -> dict:
    model.eval()
    s, y, ns = [], [], []
    for idx in batches(mols, budget):
        g, i, j, t, sz = assemble(mols, idx, frac, seed)
        s.append(model.pairs(g.to(device), i.to(device), j.to(device)).cpu())
        y.append(t)
        ns.append(sz)
    s, y, ns = torch.cat(s), torch.cat(y), torch.cat(ns)
    out = score(y, s)
    if by_size:
        qs = torch.quantile(ns.float(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])).tolist()
        out["by_size"] = [{"n_lo": int(lo), "n_hi": int(hi), **score(y[m], s[m])}
                          for lo, hi in zip(qs[:-1], qs[1:])
                          if (m := (ns >= lo) & (ns <= hi)).sum() > 20]
    return out


def train_gate(model, train, val, cfg, device, seed=0, log=print) -> dict:
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]))
    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for idx in batches(train, cfg["budget"], seed=seed * 1000 + epoch):
            g, i, j, t, _ = assemble(train, idx, cfg["mask_frac"], seed=epoch * 7919)
            g, i, j, t = g.to(device), i.to(device), j.to(device), t.to(device)
            pw = ((t.numel() - t.sum()) / t.sum().clamp(min=1)).detach()
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model.pairs(g, i, j), t, pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        v = evaluate(model, val, cfg["budget"], cfg["mask_frac"], cfg["eval_seed"], device)["auc"]
        if v > best:
            best, bad = v, 0
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        log(f"epoch {epoch:3d} loss {tot / max(nb, 1):.4f} val_auc {v:.4f} best {best:.4f}")
    model.load_state_dict(best_state)
    return {"val_auc": best, "epochs": epoch + 1, "wallclock_s": round(time.time() - t0, 1),
            "n_params": sum(p.numel() for p in model.parameters())}
