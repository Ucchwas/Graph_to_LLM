"""Three-column evaluation: AUROC, AP@1:1, AP@true-prevalence (+ lift).

Feed LOGITS, never probabilities (float32 sigmoid tail-collapse creates ties that
change AP). sklearn average_precision_score only -- never interpolated PR-AUC.
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@torch.no_grad()
def evaluate(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
             seed: int = 0, n_draws: int = 5) -> dict:
    """Metrics on mask==True cells only. Balanced column = all positives + an
    equal seeded draw of negatives from the same masked set, averaged over n_draws."""
    y = target[mask].to(torch.int8).cpu().numpy()
    s = logits[mask].float().cpu().numpy()
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        raise ValueError("mask selects a single class; metrics undefined")
    base = n_pos / len(y)

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    ap_bal, auroc_bal = [], []
    for _ in range(n_draws):
        sel = np.concatenate([pos, rng.choice(neg, size=n_pos, replace=False)])
        ap_bal.append(average_precision_score(y[sel], s[sel]))
        auroc_bal.append(roc_auc_score(y[sel], s[sel]))

    ap_sparse = float(average_precision_score(y, s))
    return {
        "auroc": float(roc_auc_score(y, s)),
        "ap_balanced": float(np.mean(ap_bal)),
        "ap_balanced_std": float(np.std(ap_bal)),
        "ap_sparse": ap_sparse,
        "base_rate": float(base),
        "lift": ap_sparse / base,
        "n_pos": n_pos,
        "n_scored": int(len(y)),
    }


@torch.no_grad()
def evaluate_pairs(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> dict:
    """The GAE-comparable edge-split view (e.g. 527 pos + 527 neg on Cora)."""
    y = np.concatenate([np.ones(len(pos_logits)), np.zeros(len(neg_logits))])
    s = torch.cat([pos_logits, neg_logits]).float().cpu().numpy()
    return {
        "auroc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
    }


@torch.no_grad()
def evaluate_edge_split(logits_full: torch.Tensor, split, seed: int = 0) -> dict:
    """Score a full [N, N] logit matrix under the edge-split protocol.

    auc/ap: the published-comparable column (test pos + the split's sampled negs).
    *_sparse/lift: the honest column over every scorable cell (~3.66M on Cora).
    Scores are symmetrized so direction conventions cannot matter.
    """
    from g2l.data import sparse_eval_mask

    train, val, test = split
    L = (logits_full + logits_full.T) / 2
    pairs = evaluate_pairs(
        L[test.pos_edge_label_index[0], test.pos_edge_label_index[1]],
        L[test.neg_edge_label_index[0], test.neg_edge_label_index[1]],
    )
    mask, target = sparse_eval_mask(train, val, test)
    sp = evaluate(L, target, mask, seed=seed)
    return {
        "auc": pairs["auroc"],
        "ap": pairs["ap"],
        "auroc_sparse": sp["auroc"],
        "ap_sparse": sp["ap_sparse"],
        "lift": sp["lift"],
        "base_rate": sp["base_rate"],
    }
