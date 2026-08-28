"""Phase 7C: re-derive the model + prior headline under the strict selection protocol, then freeze.

Phase 7B reported 0.8701 changed-edge AUROC for `model + linear_prior` against the prior's 0.8412.
That combination used equal weights and no fitted mixing parameter, and the prior's lambda was
fitted leave-one-out inside the training cancers -- leak-free, but not the protocol a reviewer will
ask for. Here **every** free parameter (lambda AND the mixing weight alpha) is chosen on the fold's
validation cancer, the priors are built from the 9 training cancers only, and the test cancer is
scored exactly once. The saved Phase-7B checkpoints are reused unchanged, so this re-scores the
published models rather than retraining them.

One caveat stated rather than hidden: the validation cancer already drove that fold's early
stopping, so selecting alpha on it reuses validation. That is what validation is for and the test
cancer stays untouched, but it is not a fresh selection set.

  python -m g2l.verify7 [--density 0.01] [--rows results/phase7]
"""
import argparse
import json
import pathlib

import numpy as np
import torch
import yaml

from g2l.aggregate6 import mean_se, ttest_1samp
from g2l.model import build_model
from g2l.run_phase7 import RUNS, folds, pairs_path
from g2l.translate import (LAMBDAS, _changed_auc_subsampled, fit_lambda, linear_prior, score_all,
                           upper)

ALPHAS = tuple(round(0.1 * k, 1) for k in range(11))
ARM = "shared_stratified"  # the Phase-7B checkpoints that back the published 0.7869 / 0.8701


def sym(S: torch.Tensor) -> torch.Tensor:
    return (S + S.T) / 2


def znorm(S: torch.Tensor, iu: torch.Tensor) -> torch.Tensor:
    """Symmetrise, then standardise using the upper-triangle cells of this cancer. Uses no labels;
    the rank normaliser below is the scale-free check that this choice changes nothing."""
    S = sym(S)
    v = S[iu[0], iu[1]]
    return (S - v.mean()) / v.std().clamp(min=1e-12)


def ranknorm(S: torch.Tensor, iu: torch.Tensor) -> torch.Tensor:
    """Within-cancer percentile ranks over the upper triangle, mirrored back to a full matrix.

    Ties take their AVERAGE rank. This is not a detail: the prior is `mT + lam*(A_n - mN)` with mT,
    mN means of 9 binary matrices, so it takes ~200 distinct values over 2M cells and is massively
    tied. Breaking those ties by index order (what a bare argsort does) would inject index-order
    noise at the same magnitude as the real signal and make this check measure tie-breaking rather
    than scale-invariance."""
    S = sym(S)
    v = S[iu[0], iu[1]]
    order = v.argsort()
    _, inv, counts = torch.unique(v[order], return_inverse=True, return_counts=True)
    ends = counts.cumsum(0)
    avg = ((ends - counts) + ends - 1).to(v.dtype) / 2          # mean position within each tie run
    r = torch.empty_like(v)
    r[order] = avg[inv]
    R = torch.zeros_like(S)
    R[iu[0], iu[1]] = r / max(v.numel() - 1, 1)
    return R + R.T


def load_fold_model(cancer: str, density: float, cfg: dict, n_genes: int):
    key = f"{cancer.replace(' ', '_')}_{ARM}_rho{density}"
    path = RUNS / f"{key}.pt"
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path} -- Phase 7C re-scores saved models, it does not retrain")
    model = build_model("gnn_direct", n_genes, cfg["width"], 1.0, scratch_layers=cfg["layers"],
                        seed=cfg["seed"], kind=cfg["kind"])
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


@torch.no_grad()
def fold_row(o: dict, fold, density: float, cfg: dict, norm) -> dict:
    test, val, train = fold
    N = len(o["genes"])
    iu = upper(N)
    tn = [o["normal"][c] for c in train]
    tt = [o["tumour"][c] for c in train]

    model = load_fold_model(test, density, cfg, N)
    S = {c: norm(model(o["normal"][c]), iu) for c in (val, test)}
    # translate.linear_prior's formula with the two training-cancer means hoisted out of the grid
    mN, mT = torch.stack(tn).mean(0), torch.stack(tt).mean(0)
    priors = {c: {lam: norm(mT + lam * (o["normal"][c] - mN), iu) for lam in LAMBDAS}
              for c in (val, test)}

    # ---- select (lambda, alpha) on the validation cancer only -------------------------------
    g = torch.Generator().manual_seed(cfg["seed"])
    pick = torch.randperm(iu.shape[1], generator=g)[:200_000]
    cells = (iu[0][pick], iu[1][pick])
    A_nv, A_tv = o["normal"][val], o["tumour"][val]
    best = (-np.inf, LAMBDAS[0], 0.5)
    for lam in LAMBDAS:
        for a in ALPHAS:
            v = _changed_auc_subsampled(a * S[val] + (1 - a) * priors[val][lam], A_nv, A_tv, cells)
            if v > best[0]:
                best = (v, lam, a)
    val_auc, lam_star, a_star = best

    # ---- score the test cancer once ---------------------------------------------------------
    A_n, A_t = o["normal"][test], o["tumour"][test]
    lam_loo = fit_lambda(tn, tt, seed=cfg["seed"])          # the published prior's lambda
    p_loo = norm(linear_prior(tn, tt, A_n, lam_loo), iu)
    arms = {"model": S[test],
            "prior_val": priors[test][lam_star],
            "prior_loo": p_loo,
            "combo_published": 0.5 * S[test] + 0.5 * p_loo,     # Phase 7B.3's equal-weight number
            "combo_strict": a_star * S[test] + (1 - a_star) * priors[test][lam_star]}
    row = {"cancer": test, "val_cancer": val, "density": density, "n_train_cancers": len(train),
           "lam_star": lam_star, "alpha_star": a_star, "lam_loo": lam_loo,
           "val_changed_auc": float(val_auc), "normaliser": norm.__name__}
    for name, sc in arms.items():
        s = score_all(sc, A_n, A_t, seed=cfg["seed"])
        row[name] = {k: s[k] for k in ("auc_changed", "auc_gained", "auc_lost", "auc_direction",
                                       "auc", "ap_balanced")}
    return row


def report(rows: list[dict], norm_name: str) -> list[str]:
    arms = ["combo_strict", "combo_published", "prior_loo", "prior_val", "model"]
    out = [f"### normaliser: {norm_name} ({len(rows)} folds)", "",
           "| arm | changed-edge AUROC | direction | overall AUROC |", "|---|---|---|---|"]
    for a in arms:
        m, se, n = mean_se([r[a]["auc_changed"] for r in rows])
        d, _, _ = mean_se([r[a]["auc_direction"] for r in rows])
        v, _, _ = mean_se([r[a]["auc"] for r in rows])
        out.append(f"| {a} | {m:.4f} +- {se:.4f} | {d:.4f} | {v:.4f} |")
    out.append("")
    for a, b in (("combo_strict", "prior_loo"), ("combo_strict", "model"),
                 ("combo_strict", "combo_published"), ("combo_published", "prior_loo")):
        delta = np.array([r[a]["auc_changed"] - r[b]["auc_changed"] for r in rows])
        t, p = ttest_1samp(delta)
        out.append(f"- paired {a} - {b}: {delta.mean():+.4f} (p {p:.2g}, {int((delta > 0).sum())}/{len(delta)})")
    sel = sorted({(r["lam_star"], r["alpha_star"]) for r in rows})
    out += ["", f"selected (lambda, alpha) per fold: {sel}",
            f"lambda by train-LOO: {sorted({r['lam_loo'] for r in rows})}", ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase7.yaml")
    ap.add_argument("--density", type=float, default=0.01)
    ap.add_argument("--out", default="results/phase7")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    o = torch.load(pairs_path(args.density), weights_only=False)
    fs = folds(o["cancers"])
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    text, everything = [f"# Phase 7C -- strict-protocol verification (rho = {args.density})", ""], {}
    for norm in (znorm, ranknorm):
        rows = []
        for k, fold in enumerate(fs):
            r = fold_row(o, fold, args.density, cfg, norm)
            rows.append(r)
            print(f"[{norm.__name__}] {k + 1:2d}/{len(fs)} {fold[0][:34]:34s} "
                  f"lam {r['lam_star']:+.1f} alpha {r['alpha_star']:.1f} | "
                  f"strict {r['combo_strict']['auc_changed']:.4f} "
                  f"published {r['combo_published']['auc_changed']:.4f} "
                  f"prior {r['prior_loo']['auc_changed']:.4f} "
                  f"model {r['model']['auc_changed']:.4f}", flush=True)
        everything[norm.__name__] = rows
        text += report(rows, norm.__name__)
    (out_dir / "verify.json").write_text(json.dumps(everything))
    body = "\n".join(text)
    print("\n" + body)
    (out_dir / "verify.md").write_text(body + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
