"""Phase 9 -- TCGA normal-graph -> tumour-graph translation, rebuilt from scratch for two
size-independent bodies under identical treatment: the LGM and a capacity-matched edge-aware GCN.

The raw graph is the whole input. Per (cancer, condition) it is
    edge_index [2, M]   the binarised Spearman co-expression graph (Phase 7's, rebuilt bit-exactly)
    edge_value [M, 1]   the signed Spearman rho on each edge, from that condition's samples only
    x          [N, 2]   each gene's mean and std of log2-TPM over that condition's samples, z-scored
N = 2,000 genes, one fixed universe, 11 cancers. Nothing is indexed by gene identity: a node is
described by what it is (degree, expression statistics), never by which node it is.

Protocol, per leave-one-cancer-out fold (test cancer, validation cancer, 9 training cancers):
  SSL        masked-cell reconstruction on the 18 training graphs (9 normal + 9 tumour): hide 15 %
             of edges plus an equal draw of non-edges, predict them. Graph-in, graph-out -- the
             architecture's own objective. The validation and test cancers never enter it.
  translate  input = a training cancer's NORMAL graph, target = its TUMOUR graph, stratified
             pos-weighted BCE (Phase 7B's objective, which removes the incentive to copy the
             input). Select on the validation cancer's changed-edge AUROC; score the test cancer
             once.
Both bodies share this file entirely: the same graphs, the same SSL masks (drawn from a generator
seeded by fold, seed and epoch, never by model), the same decoder (D1Bilinear), the same loss, the
same optimiser, schedule and budget. tests/test_phase9.py pins each of those.
"""
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from g2l.lgm import LGM, RawGraph
from g2l.multigraph import EdgeGCNBaseline, matched_width
from g2l.translate import changed_edge_scores, overall_scores, translation_loss, upper

DENSITY = 0.01


def root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("G2L_PHASE9_ROOT", "data/phase9"))


def data_path() -> pathlib.Path:
    return root() / "tcga9.pt"


# ---------------------------------------------------------------- graphs

class CGraph:
    """One (cancer, condition) graph. `edge_index` holds both directions of every undirected edge."""

    def __init__(self, n: int, edge_index: torch.Tensor, edge_value: torch.Tensor, x: torch.Tensor):
        self.n, self.edge_index, self.edge_value, self.x = n, edge_index.long(), edge_value.float(), x.float()

    def raw(self, device="cpu") -> RawGraph:
        return RawGraph(self.n, self.edge_index, self.edge_value, None, self.x).to(device)

    def dense(self, device="cpu") -> torch.Tensor:
        A = torch.zeros(self.n, self.n, device=device)
        A[self.edge_index[0].to(device), self.edge_index[1].to(device)] = 1.0
        return A

    def undirected(self):
        """(src, dst) with src < dst, one row per undirected edge, in the order `lgm.canonical` uses."""
        lo = torch.minimum(self.edge_index[0], self.edge_index[1])
        hi = torch.maximum(self.edge_index[0], self.edge_index[1])
        uniq = torch.unique(lo * self.n + hi)
        return uniq // self.n, uniq % self.n

    def to_dict(self) -> dict:
        return {"n": self.n, "edge_index": self.edge_index.int(), "edge_value": self.edge_value, "x": self.x}

    @staticmethod
    def from_dict(d: dict) -> "CGraph":
        return CGraph(int(d["n"]), d["edge_index"], d["edge_value"], d["x"])


def graph_from_expression(X: torch.Tensor, density: float = DENSITY) -> CGraph:
    """[G, S] log-expression over ONE condition's samples -> that condition's graph. The topology
    is Phase 7's `binarise(spearman(.))`; the edge value is the signed rho on each kept edge and
    the node features are per-gene (mean, std), z-scored over genes. Everything comes from the
    columns passed in, so a normal graph cannot see a tumour sample by construction."""
    from g2l.tcga import binarise, spearman

    C = spearman(X)
    A = binarise(C, density)
    ei = A.nonzero().T
    ev = C[ei[0], ei[1]].unsqueeze(-1)
    x = torch.stack([X.mean(1), X.std(1)], dim=1)
    x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp(min=1e-6)
    return CGraph(X.shape[0], ei, ev, x)


def build(pairs_path: pathlib.Path, expr_path: pathlib.Path, out: pathlib.Path) -> dict:
    """Rebuild every (cancer, condition) graph from the expression matrix and Phase 7's sample
    lists, asserting bit-exact agreement with Phase 7's stored topology, and attach edge values and
    node features. Writes one small file; the 300 MB expression matrix is not needed afterwards."""
    o = torch.load(pairs_path, weights_only=False)
    e = torch.load(expr_path, weights_only=False)
    at = {g: i for i, g in enumerate(e["genes"])}
    pos = {s: i for i, s in enumerate(e["samples"])}
    keep = [at[g] for g in o["genes"]]
    graphs = {}
    for c in o["cancers"]:
        for cond in ("normal", "tumour"):
            cols = [pos[s] for s in o["samples"][c][cond]]
            g = graph_from_expression(e["expr"][keep][:, cols], o["density"])
            assert torch.equal(g.dense(), o[cond][c]), f"rebuilt {cond} graph differs for {c}"
            graphs[f"{c}|{cond}"] = g.to_dict()
            print(f"  {c[:36]:36s} {cond:6s} samples {len(cols):3d} edges {g.edge_index.shape[1] // 2:6d} "
                  f"rho[{g.edge_value.min():+.2f},{g.edge_value.max():+.2f}]", flush=True)
    ds = {"cancers": list(o["cancers"]), "genes": list(o["genes"]), "density": o["density"],
          "n": len(o["genes"]), "graphs": graphs}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ds, out)
    return ds


def load(path: pathlib.Path | None = None) -> dict:
    ds = torch.load(path or data_path(), weights_only=False)
    ds["graphs"] = {k: CGraph.from_dict(v) for k, v in ds["graphs"].items()}
    return ds


def graph(ds: dict, cancer: str, cond: str) -> CGraph:
    return ds["graphs"][f"{cancer}|{cond}"]


def folds(cancers: list[str]) -> list[tuple[str, str, list[str]]]:
    """Leave-one-cancer-out, Phase 7's rotation: (test, val = next cancer, train = the rest)."""
    return [(c, cancers[(k + 1) % len(cancers)],
             [x for x in cancers if x != c and x != cancers[(k + 1) % len(cancers)]])
            for k, c in enumerate(cancers)]


def ssl_corpus(ds: dict, train: list[str]) -> list[CGraph]:
    """The self-supervision graphs: both conditions of the TRAINING cancers, nothing else."""
    return [graph(ds, c, cond) for c in train for cond in ("normal", "tumour")]


# ---------------------------------------------------------------- models

def build_model(kind: str, cfg: dict, seed: int):
    common = dict(layers=cfg["layers"], k=1, dropout=cfg["dropout"], seed=seed, x_dim=cfg["x_dim"])
    if kind == "lgm":
        return LGM(d=cfg["d"], heads=cfg["heads"], **common)
    if kind == "edgegcn":
        ref = sum(p.numel() for p in LGM(d=cfg["d"], heads=cfg["heads"], **common).parameters())
        w = matched_width(ref, cfg["layers"], 1, kind="edgegcn", x_dim=cfg["x_dim"])
        return EdgeGCNBaseline(w, **common)
    raise ValueError(kind)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- self-supervision

def ssl_mask(g: CGraph, frac: float, gen: torch.Generator):
    """Hide `frac` of the undirected edges and draw as many non-edges. Returns the masked RawGraph
    (hidden edges removed from edge_index AND edge_value, so the model cannot see them) plus the
    scored cells (i, j, target). Depends only on `gen`, never on the model."""
    src, dst = g.undirected()
    m_u = src.numel()
    hide = torch.rand(m_u, generator=gen) < frac
    keep = ~hide
    ks, kd = src[keep], dst[keep]
    # the kept edges' values, looked up on the dense value matrix so direction never matters
    V = torch.zeros(g.n, g.n)
    V[g.edge_index[0], g.edge_index[1]] = g.edge_value[:, 0]
    ei = torch.cat([torch.stack([ks, kd]), torch.stack([kd, ks])], dim=1)
    ev = V[ei[0], ei[1]].unsqueeze(-1)
    A = g.dense().bool()
    n_neg, ni, nj = int(hide.sum()), [], []
    while sum(t.numel() for t in ni) < n_neg:
        a = torch.randint(g.n, (2 * n_neg + 16,), generator=gen)
        b = torch.randint(g.n, (2 * n_neg + 16,), generator=gen)
        lo, hi = torch.minimum(a, b), torch.maximum(a, b)
        ok = (lo != hi) & ~A[lo, hi]
        ni.append(lo[ok])
        nj.append(hi[ok])
    ni, nj = torch.cat(ni)[:n_neg], torch.cat(nj)[:n_neg]
    i = torch.cat([src[hide], ni])
    j = torch.cat([dst[hide], nj])
    t = torch.cat([torch.ones(int(hide.sum())), torch.zeros(n_neg)])
    return RawGraph(g.n, ei, ev, None, g.x), i, j, t


def _optimiser(model, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]))
    return opt, sched


def pretrain(model, corpus: list[CGraph], cfg: dict, device: str, seed: int, log=print) -> dict:
    """Fixed-epoch masked-cell SSL. Masks come from a generator seeded by (seed, epoch), and the
    graphs are visited in a (seed, epoch)-seeded order, so every model sees the identical sequence
    of masked inputs and scored cells."""
    torch.manual_seed(seed)
    model.to(device).train()
    opt, sched = _optimiser(model, cfg)
    t0, last = time.time(), 0.0
    for epoch in range(cfg["ssl_epochs"]):
        gen = torch.Generator().manual_seed(seed * 100_003 + epoch)
        order = torch.randperm(len(corpus), generator=gen).tolist()
        tot = 0.0
        for k in order:
            g_m, i, j, t = ssl_mask(corpus[k], cfg["mask_frac"], gen)
            g_m, i, j, t = g_m.to(device), i.to(device), j.to(device), t.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model.pairs(g_m, i, j), t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            sched.step()
            tot += loss.item()
        last = tot / len(corpus)
        if epoch % 10 == 0 or epoch == cfg["ssl_epochs"] - 1:
            log(f"  ssl epoch {epoch:3d} loss {last:.4f}")
    return {"ssl_epochs": cfg["ssl_epochs"], "ssl_final_loss": last, "ssl_graphs": len(corpus),
            "ssl_wallclock_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- translation

def pos_weight_of(A_t: torch.Tensor, iu: torch.Tensor) -> torch.Tensor:
    y = A_t[iu[0], iu[1]]
    pos = y.sum()
    return ((y.numel() - pos) / pos.clamp(min=1)).detach()


@torch.no_grad()
def changed_auc(model, ds, cancer, device) -> float:
    model.eval()
    g_n = graph(ds, cancer, "normal")
    A_n, A_t = g_n.dense(device), graph(ds, cancer, "tumour").dense(device)
    return changed_edge_scores(model(g_n.raw(device)), A_n, A_t)["auc_changed"]


def translate(model, ds, train: list[str], val: str, cfg: dict, device: str, seed: int, log=print) -> dict:
    """Fine-tune normal -> tumour on the training cancers; select on the validation cancer."""
    torch.manual_seed(seed)
    model.to(device)
    N = ds["n"]
    iu = upper(N, device)
    opt, sched = _optimiser(model, cfg)
    data = {c: (graph(ds, c, "normal").raw(device), graph(ds, c, "normal").dense(device),
                graph(ds, c, "tumour").dense(device)) for c in train}
    pw = {c: pos_weight_of(data[c][2], iu) for c in train}
    g = torch.Generator().manual_seed(seed)
    best, best_epoch, bad, best_state, t0 = -1.0, 0, 0, None, time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot = 0.0
        for k in torch.randperm(len(train), generator=g).tolist():
            c = train[k]
            raw, A_n, A_t = data[c]
            opt.zero_grad(set_to_none=True)
            loss = translation_loss(model(raw), A_t, iu, pw[c], A_n=A_n, mode="stratified")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            sched.step()
            tot += loss.item()
        v = changed_auc(model, ds, val, device)
        if v > best:
            best, best_epoch, bad = v, epoch, 0
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        if epoch % 20 == 0:
            log(f"  ft epoch {epoch:3d} loss {tot / len(train):.4f} val_changed {v:.4f} best {best:.4f}@{best_epoch}")
    model.load_state_dict(best_state)
    return {"val_changed_auc": best, "best_epoch": best_epoch, "ft_epochs": epoch + 1,
            "ft_wallclock_s": round(time.time() - t0, 1)}


@torch.no_grad()
def test(model, ds, cancer: str, device: str, seed: int = 0) -> dict:
    """The single look at the held-out cancer."""
    model.eval()
    g_n = graph(ds, cancer, "normal")
    A_n, A_t = g_n.dense(), graph(ds, cancer, "tumour").dense()
    S = model(g_n.raw(device)).cpu()
    row = {**overall_scores(S, A_t, seed), **changed_edge_scores(S, A_n, A_t)}
    row["ap_changed"] = float(np.nanmean([row["ap_gained"], row["ap_lost"]]))
    return row
