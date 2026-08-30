"""Phase 10 aggregation -- the two tables, and only the columns that were asked for.

  python -m g2l.aggregate10 --benchmark molhiv [--rows results/phase10/molhiv/rows]
  python -m g2l.aggregate10 --benchmark tcga   [--rows results/phase10/tcga/rows]

MolHIV: Test ROC-AUC, Validation ROC-AUC, inference parameter count -- mean +- unbiased std over the
10 final seeds (OGB's submission rule), plus the pre-registered paired per-seed contrasts on test.

TCGA: changed-edge AUROC/AP, overall AUROC and direction -- mean +- SE over the 11 held-out cancers
(seeds averaged within a cancer first), plus the paired per-cancer contrasts.

Contrasts are fixed here rather than chosen after seeing the numbers.
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np

from g2l.aggregate6 import ttest_1samp

MOLHIV_ARMS = ("lgm", "lgm_rrwp", "gin", "gin_rwse", "gcn", "gcn_rwse")
MOLHIV_LABEL = {"lgm": "LGM", "lgm_rrwp": "**LGM + RRWP**", "gin": "GIN", "gin_rwse": "GIN + RWSE",
                "gcn": "GCN", "gcn_rwse": "GCN + RWSE"}
MOLHIV_CONTRASTS = [("lgm_rrwp", "lgm", "does RRWP help the LGM"),
                    ("gin_rwse", "gin", "does RWSE help GIN"),
                    ("gcn_rwse", "gcn", "does RWSE help GCN"),
                    ("lgm_rrwp", "gin_rwse", "best-equipped vs best-equipped"),
                    ("lgm", "gin", "Phase 8H replication check")]

TCGA_ARMS = ("lgm", "lgm_rrwp", "edgegcn", "edgegcn_rwse")
TCGA_LABEL = {"lgm": "LGM", "lgm_rrwp": "**LGM + RRWP**", "edgegcn": "edge-aware GCN",
              "edgegcn_rwse": "edge-aware GCN + RWSE"}
TCGA_CONTRASTS = [("lgm_rrwp", "lgm", "does RRWP help the LGM"),
                  ("edgegcn_rwse", "edgegcn", "does RWSE help the GCN"),
                  ("lgm_rrwp", "edgegcn_rwse", "best-equipped vs best-equipped"),
                  ("lgm", "edgegcn", "Phase 9 replication check")]
TCGA_METRICS = (("auc_changed", "changed-edge AUROC"), ("ap_changed", "changed-edge AP"),
                ("auc", "overall AUROC"), ("auc_direction", "direction"))


def _stat(d: np.ndarray) -> dict:
    t, p = ttest_1samp(d)
    return {"n": len(d), "mean": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "t": float(t), "p": float(p), "wins": int((d > 0).sum())}


# ---------------------------------------------------------------- molhiv

def load_molhiv(rows: pathlib.Path) -> dict:
    """{arm: {seed: (val, test)}}, pooled over the final-stage seed slices."""
    by = {}
    for f in glob.glob(str(rows / "final_*.json")):
        r = json.loads(pathlib.Path(f).read_text())
        if r.get("smoke"):
            continue
        a = by.setdefault(r["arm"], {"seeds": {}, "n_params": r["n_params"], "d": r["d"]})
        assert a["n_params"] == r["n_params"], f"{r['arm']}: mixed parameter counts across slices"
        for s, v, t in zip(r["seeds"], r["val_rocauc"], r["test_rocauc"]):
            assert s not in a["seeds"], f"{r['arm']}: seed {s} appears twice"
            a["seeds"][s] = (v, t)
    return by


def molhiv_markdown(by: dict) -> tuple[str, dict]:
    L = ["| model | Test ROC-AUC | Validation ROC-AUC | #Params |", "|---|---|---|---|"]
    summary = {"arms": {}, "paired": {}}
    for arm in MOLHIV_ARMS:
        if arm not in by:
            continue
        a = by[arm]
        v = np.array([a["seeds"][s][0] for s in sorted(a["seeds"])])
        t = np.array([a["seeds"][s][1] for s in sorted(a["seeds"])])
        L.append(f"| {MOLHIV_LABEL[arm]} | {t.mean():.4f} ± {t.std(ddof=1):.4f} "
                 f"| {v.mean():.4f} ± {v.std(ddof=1):.4f} | {a['n_params']:,} |")
        summary["arms"][arm] = {"d": a["d"], "n_params": a["n_params"], "n_seeds": len(t),
                                "test_mean": float(t.mean()), "test_std": float(t.std(ddof=1)),
                                "val_mean": float(v.mean()), "val_std": float(v.std(ddof=1))}
    L += ["", "Paired per seed, on test:", "", "| contrast | Δ | seeds won | t | p |",
          "|---|---|---|---|---|"]
    for x, y, why in MOLHIV_CONTRASTS:
        if x not in by or y not in by:
            continue
        common = sorted(set(by[x]["seeds"]) & set(by[y]["seeds"]))
        d = np.array([by[x]["seeds"][s][1] - by[y]["seeds"][s][1] for s in common])
        st = _stat(d)
        summary["paired"][f"{x}-{y}"] = {**st, "why": why}
        L.append(f"| {x} − {y} | {st['mean']:+.4f} ± {d.std(ddof=1):.4f} | "
                 f"{st['wins']}/{st['n']} | {st['t']:+.2f} | {st['p']:.2e} |")
    return "\n".join(L) + "\n", summary


# ---------------------------------------------------------------- tcga

def load_tcga(rows: pathlib.Path) -> dict:
    """{body: {cancer: {seed: row}}}, non-smoke rows only."""
    by = {}
    for f in glob.glob(str(rows / "*.json")):
        r = json.loads(pathlib.Path(f).read_text())
        if r.get("smoke"):
            continue
        by.setdefault(r["body"], {}).setdefault(r["cancer"], {})[r["seed"]] = r
    return by


def per_cancer(by: dict, body: str, metric: str) -> dict:
    return {c: float(np.mean([r[metric] for r in seeds.values()])) for c, seeds in by[body].items()}


def tcga_markdown(by: dict) -> tuple[str, dict]:
    L = ["| body | changed-edge AUROC | changed-edge AP | overall AUROC | direction | #Params | seeds |",
         "|---|---|---|---|---|---|---|"]
    summary = {"bodies": {}, "paired": {}}
    for body in TCGA_ARMS:
        if body not in by:
            continue
        seeds = sorted({s for c in by[body].values() for s in c})
        params = sorted({r["n_params"] for c in by[body].values() for r in c.values()})
        e = {"n_cancers": len(by[body]), "seeds": seeds, "n_params": params}
        cells = []
        for m, _ in TCGA_METRICS:
            v = np.array(list(per_cancer(by, body, m).values()))
            e[m] = {"mean": float(v.mean()),
                    "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")}
            cells.append(f"{e[m]['mean']:.4f} ± {e[m]['se']:.4f}")
        summary["bodies"][body] = e
        L.append(f"| {TCGA_LABEL[body]} | " + " | ".join(cells) +
                 f" | {params[0]:,} | {len(seeds)} |")

    L += ["", "Paired over the held-out cancers (seeds averaged within a cancer first):", "",
          "| contrast | metric | Δ ± SE | t | p | wins |", "|---|---|---|---|---|---|"]
    for x, y, why in TCGA_CONTRASTS:
        if x not in by or y not in by:
            continue
        common = sorted(set(by[x]) & set(by[y]))
        for m, mname in TCGA_METRICS:
            a, b = per_cancer(by, x, m), per_cancer(by, y, m)
            d = np.array([a[c] - b[c] for c in common])
            st = _stat(d)
            summary["paired"].setdefault(f"{x}-{y}", {"why": why})[m] = {
                **st, "per_cancer": {c: float(a[c] - b[c]) for c in common}}
            L.append(f"| {x} − {y} | {mname} | {st['mean']:+.4f} ± {st['se']:.4f} | "
                     f"{st['t']:+.2f} | {st['p']:.2e} | {st['wins']}/{st['n']} |")

    x, y = TCGA_CONTRASTS[2][0], TCGA_CONTRASTS[2][1]
    if x in by and y in by:
        pairs = [(c, s) for c in sorted(set(by[x]) & set(by[y])) for s in by[x][c] if s in by[y][c]]
        d = np.array([by[x][c][s]["auc_changed"] - by[y][c][s]["auc_changed"] for c, s in pairs])
        st = _stat(d)
        summary["paired_by_seed"] = st
        L += ["", f"Pairing every (cancer, seed) for {x} − {y}, n = {st['n']}: changed-edge AUROC "
              f"Δ {st['mean']:+.4f}, t = {st['t']:+.2f}, p = {st['p']:.2e}, "
              f"wins {st['wins']}/{st['n']}."]
        L += ["", "Per cancer (changed-edge AUROC, mean over seeds):", "",
              "| held-out cancer | " + " | ".join(TCGA_LABEL[b] for b in TCGA_ARMS if b in by) + " |",
              "|---" * (1 + len([b for b in TCGA_ARMS if b in by])) + "|"]
        cols = {b: per_cancer(by, b, "auc_changed") for b in TCGA_ARMS if b in by}
        ref = cols[TCGA_ARMS[0]]
        for c in sorted(ref, key=lambda c: -ref[c]):
            L.append(f"| {c} | " + " | ".join(f"{cols[b][c]:.4f}" for b in cols) + " |")
    return "\n".join(L) + "\n", summary


def main():
    # the tables carry U+0394 and U+00B1; a Windows console defaults to cp1252 and would raise
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["molhiv", "tcga"])
    ap.add_argument("--rows", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = pathlib.Path(args.rows or f"results/phase10/{args.benchmark}/rows")
    out = pathlib.Path(args.out or f"results/phase10/{args.benchmark}/aggregate.md")
    if args.benchmark == "molhiv":
        md, summary = molhiv_markdown(load_molhiv(rows))
    else:
        md, summary = tcga_markdown(load_tcga(rows))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(md)


if __name__ == "__main__":
    main()
