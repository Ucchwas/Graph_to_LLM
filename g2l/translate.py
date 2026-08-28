"""Phase 7: normal graph -> tumour graph translation, leave-one-cancer-out.

The model is the Phase-5/6 one, unchanged: the raw normal adjacency rows and the edge set of the
**normal** graph go in, a predicted tumour adjacency comes out. The tumour matrix is only ever a
supervision target and a metric input -- `g2l.model.GraphLLM.embed` derives `edge_index` from the
matrix it is called with, so passing `A_normal` is what enforces the leak rule
(`tests/test_tcga.py::test_edge_index_comes_only_from_the_normal_graph`).

Scoring separates two regimes. *Overall* asks whether the predicted matrix looks like the tumour
graph at all -- but the normal and tumour graphs overlap heavily, so copying the input already
scores well there. *Changed-edge* restricts to the cells that actually flip, where copying is
worth exactly chance, and is the metric the phase is judged on.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from g2l.train import make_optimizer


def upper(n: int, device="cpu"):
    return torch.triu_indices(n, n, offset=1, device=device)


def _auc_ap(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    if y.min() == y.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


@torch.no_grad()
def changed_edge_scores(scores: torch.Tensor, A_n: torch.Tensor, A_t: torch.Tensor) -> dict:
    """Metrics on the cells that change between conditions -- the headline of Phase 7.

    `gained`: among cells absent in the normal graph, rank the ones present in the tumour graph
    above the ones still absent. `lost`: among cells present in the normal graph, rank the ones
    still present above the ones removed. In both strata the identity map is constant, so it
    scores exactly 0.5 -- copying the input earns nothing here.
    `direction`: over the changed cells only, separate gained (label 1) from lost (label 0);
    identity scores exactly 0.0 there, chance is 0.5.
    """
    iu = upper(A_n.shape[0], A_n.device)
    s = ((scores + scores.T) / 2)[iu[0], iu[1]].float().cpu().numpy()
    n = A_n[iu[0], iu[1]].cpu().numpy().astype(bool)
    t = A_t[iu[0], iu[1]].cpu().numpy().astype(bool)
    out = {"n_gained": int((~n & t).sum()), "n_lost": int((n & ~t).sum()),
           "n_edges_normal": int(n.sum()), "n_edges_tumour": int(t.sum())}
    for name, stratum in (("gained", ~n), ("lost", n)):
        auc, ap = _auc_ap(t[stratum].astype(int), s[stratum])
        out[f"auc_{name}"], out[f"ap_{name}"] = auc, ap
    changed = n != t
    auc, ap = _auc_ap(t[changed].astype(int), s[changed])
    out["auc_direction"], out["ap_direction"] = auc, ap
    out["auc_changed"] = float(np.nanmean([out["auc_gained"], out["auc_lost"]]))
    return out


@torch.no_grad()
def overall_scores(scores: torch.Tensor, A_t: torch.Tensor, seed: int = 0, n_draws: int = 5) -> dict:
    """AUROC / AP over every upper-triangle cell, plus AP on an equal seeded draw of non-edges
    (the AP@1:1 column carried by every earlier phase)."""
    iu = upper(A_t.shape[0], A_t.device)
    s = ((scores + scores.T) / 2)[iu[0], iu[1]].float().cpu().numpy()
    y = A_t[iu[0], iu[1]].cpu().numpy().astype(int)
    auc, ap = _auc_ap(y, s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    bal = [_auc_ap(y[sel], s[sel]) for sel in
           (np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)]) for _ in range(n_draws))]
    return {"auc": auc, "ap_sparse": ap, "auc_balanced": float(np.mean([b[0] for b in bal])),
            "ap_balanced": float(np.mean([b[1] for b in bal])), "base_rate": float(y.mean())}


def score_all(scores: torch.Tensor, A_n: torch.Tensor, A_t: torch.Tensor, seed: int = 0) -> dict:
    return {**overall_scores(scores, A_t, seed), **changed_edge_scores(scores, A_n, A_t)}


# ---------------------------------------------------------------- baselines (training cancers only)

def identity_baseline(A_n: torch.Tensor) -> torch.Tensor:
    """Predict the tumour graph to be the normal graph. Strong overall, chance on changed edges."""
    return A_n.clone()


def mean_tumour(train_tumour: list[torch.Tensor]) -> torch.Tensor:
    """The average tumour graph of the training cancers -- ignores the input entirely."""
    return torch.stack(train_tumour).mean(0)


def mean_change(train_normal: list[torch.Tensor], train_tumour: list[torch.Tensor], A_n: torch.Tensor) -> torch.Tensor:
    """Apply the training cancers' average normal->tumour edge change to this cancer's normal
    graph. The baseline to beat: it is what "the rewiring is the same in every cancer" predicts."""
    delta = torch.stack([t - n for n, t in zip(train_normal, train_tumour)]).mean(0)
    return A_n + delta


def common_neighbours(A_n: torch.Tensor) -> torch.Tensor:
    return A_n @ A_n


LAMBDAS = (0.0, -0.1, -0.2, -0.3, -0.4, -0.6, -1.0)


def linear_prior(train_normal, train_tumour, A_n: torch.Tensor, lam: float) -> torch.Tensor:
    """`mean_tumour + lam * (A_n - mean_normal)`. The diagnostic that motivated Phase 7B: a
    *negative* lam beats the mean-tumour prior at every density, i.e. edges specific to this
    cancer's normal graph are preferentially absent from its tumour graph. Two parameters."""
    mT = torch.stack(train_tumour).mean(0)
    mN = torch.stack(train_normal).mean(0)
    return mT + lam * (A_n - mN)


def _changed_auc_subsampled(scores, A_n, A_t, cells) -> float:
    """Mean of the gained / lost AUROCs on a fixed subsample of upper-triangle cells. Used only to
    pick lam, where the objective is a flat plateau -- the reported metrics always use every cell."""
    i, j = cells
    s = scores[i, j].float().cpu().numpy()
    n = A_n[i, j].cpu().numpy().astype(bool)
    t = A_t[i, j].cpu().numpy().astype(bool)
    return float(np.nanmean([_auc_ap(t[m].astype(int), s[m])[0] for m in (~n, n)]))


def fit_lambda(train_normal, train_tumour, n_cells: int = 200_000, seed: int = 0) -> float:
    """Choose lam on the TRAINING cancers only (leave-one-out inside the training set), so the
    held-out cancer never informs the baseline it is scored against."""
    N = train_normal[0].shape[0]
    iu = upper(N)
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(iu.shape[1], generator=g)[:min(n_cells, iu.shape[1])]
    cells = (iu[0][pick], iu[1][pick])
    sumN = torch.stack(train_normal).sum(0)
    sumT = torch.stack(train_tumour).sum(0)
    k = len(train_normal)
    best, best_lam = -np.inf, 0.0
    for lam in LAMBDAS:
        v = []
        for x in range(k):
            mN = (sumN - train_normal[x]) / (k - 1)
            mT = (sumT - train_tumour[x]) / (k - 1)
            v.append(_changed_auc_subsampled(mT + lam * (train_normal[x] - mN),
                                             train_normal[x], train_tumour[x], cells))
        m = float(np.nanmean(v))
        if m > best:
            best, best_lam = m, lam
    return best_lam


def baseline_scores(name: str, A_n, train_normal, train_tumour) -> torch.Tensor:
    return {"identity": lambda: identity_baseline(A_n),
            "mean_tumour": lambda: mean_tumour(train_tumour),
            "mean_change": lambda: mean_change(train_normal, train_tumour, A_n),
            "common_neighbors": lambda: common_neighbours(A_n),
            "linear_prior": lambda: linear_prior(train_normal, train_tumour, A_n,
                                                 fit_lambda(train_normal, train_tumour))}[name]()


# ---------------------------------------------------------------- training

def pos_weight_of(A_t: torch.Tensor) -> float:
    iu = upper(A_t.shape[0])
    y = A_t[iu[0], iu[1]]
    pos = y.sum().item()
    return (y.numel() - pos) / max(pos, 1)


def translation_loss(logits: torch.Tensor, A_t: torch.Tensor, iu: torch.Tensor, pw: torch.Tensor,
                     A_n: torch.Tensor | None = None, mode: str = "full", changed_weight: float = 10.0) -> torch.Tensor:
    """Pos-weighted BCE over the strict upper triangle of the target.

    `full` (Phase 7 MVP): every cell weighted equally. ~99 % of cells do not change between
    conditions and copying the input is right for those, so this objective rewards copying -- the
    measured failure (direction 0.393, below chance).
    `weighted`: cells that flip are upweighted by `changed_weight`, so training pays attention to
    what the headline metric scores.
    `stratified`: one pos-weighted BCE per input stratum (A_n = 0 and A_n = 1), the two weighted
    equally. Within a stratum the input is constant, so copying carries no gradient at all -- this
    is the objective that matches the gained / lost metrics exactly.
    """
    y, s = A_t[iu[0], iu[1]], logits[iu[0], iu[1]]
    if mode == "full":
        return F.binary_cross_entropy_with_logits(s, y, pos_weight=pw)
    n = A_n[iu[0], iu[1]]
    if mode == "weighted":
        w = torch.where(n != y, changed_weight, 1.0)
        return F.binary_cross_entropy_with_logits(s, y, weight=w, pos_weight=pw)
    if mode == "stratified":
        loss = 0.0
        for v in (0.0, 1.0):
            m = n == v
            if not m.any():
                continue
            yv = y[m]
            pos = yv.sum()
            w = ((yv.numel() - pos) / pos.clamp(min=1)).detach()
            loss = loss + F.binary_cross_entropy_with_logits(s[m], yv, pos_weight=w)
        return loss / 2
    raise ValueError(mode)


@torch.no_grad()
def val_metric(model, pairs, cancers, device) -> float:
    model.eval()
    v = []
    for c in cancers:
        A_n, A_t = pairs["normal"][c].to(device), pairs["tumour"][c].to(device)
        v.append(changed_edge_scores(model(A_n), A_n, A_t)["auc_changed"])
    return float(np.nanmean(v))


def train_translation(model, pairs, train_cancers, val_cancers, cfg, device, seed, log=print) -> dict:
    """One shared model over the training cancers; a step is one cancer's (normal -> tumour) pair.
    Selection on the validation cancers' changed-edge AUROC."""
    torch.manual_seed(seed)
    model.to(device)
    N = pairs["normal"][train_cancers[0]].shape[0]
    iu = upper(N, device)
    opt = make_optimizer(model, cfg["lr"], cfg["lr_norm"], cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    data = {c: (pairs["normal"][c].to(device), pairs["tumour"][c].to(device)) for c in train_cancers}
    pw = {c: torch.tensor(pos_weight_of(pairs["tumour"][c]), device=device) for c in train_cancers}
    g = torch.Generator().manual_seed(seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_epoch, bad, best_state, t0 = -1.0, 0, 0, None, time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot = 0.0
        for k in torch.randperm(len(train_cancers), generator=g).tolist():
            c = train_cancers[k]
            A_n, A_t = data[c]
            opt.zero_grad(set_to_none=True)
            loss = translation_loss(model(A_n), A_t, iu, pw[c], A_n=A_n, mode=cfg.get("loss_mode", "full"),
                                    changed_weight=cfg.get("changed_weight", 10.0))
            loss.backward()
            if epoch == 0 and not tot:
                missing = [n for n, p in trainable.items() if p.grad is None]
                assert not missing, f"no gradient reached: {missing[:5]}"
            torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        v = val_metric(model, pairs, val_cancers, device)
        if v > best:
            best, best_epoch, bad = v, epoch, 0
            best_state = {n: p.detach().clone() for n, p in trainable.items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        if epoch % 20 == 0:
            log(f"epoch {epoch:3d} loss {tot / len(train_cancers):.4f} val_changed_auc {v:.4f} best {best:.4f}@{best_epoch}")
    epochs, wall = epoch + 1, time.time() - t0
    model.load_state_dict(best_state, strict=False)
    return {"val_changed_auc": best, "best_epoch": best_epoch, "epochs": epochs,
            "sec_per_epoch": wall / epochs, "wallclock_s": round(wall, 1),
            "n_trainable": sum(p.numel() for p in trainable.values()),
            "n_train_cancers": len(train_cancers),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None}
