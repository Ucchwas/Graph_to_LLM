"""Phase 9 aggregation: per-body mean +- SE over the 11 held-out cancers (seeds averaged within a
cancer first), and the paired LGM - edge-GCN difference with a paired t-test and wins / 11.

  python -m g2l.aggregate9 [--rows results/phase9/rows] [--out results/phase9/aggregate.md]
"""
import argparse
import glob
import json
import pathlib

import numpy as np

from g2l.aggregate6 import ttest_1samp

METRICS = (("auc_changed", "changed-edge AUROC"), ("ap_changed", "changed-edge AP"),
           ("auc", "overall AUROC"), ("auc_direction", "direction"))


def load(rows_dir: pathlib.Path) -> dict:
    """{body: {cancer: {seed: row}}}, non-smoke rows only."""
    by = {}
    for f in glob.glob(str(rows_dir / "*.json")):
        r = json.loads(pathlib.Path(f).read_text())
        if r.get("smoke"):
            continue
        by.setdefault(r["body"], {}).setdefault(r["cancer"], {})[r["seed"]] = r
    return by


def per_cancer(by: dict, body: str, metric: str) -> dict:
    """cancer -> mean over that cancer's seeds."""
    return {c: float(np.mean([r[metric] for r in seeds.values()])) for c, seeds in by[body].items()}


def summarise(by: dict) -> dict:
    out = {"bodies": {}, "paired": {}}
    for body in by:
        n_seeds = sorted({s for c in by[body].values() for s in c})
        params = {r["n_params"] for c in by[body].values() for r in c.values()}
        out["bodies"][body] = {"n_cancers": len(by[body]), "seeds": n_seeds, "n_params": sorted(params)}
        for m, _ in METRICS:
            v = np.array(list(per_cancer(by, body, m).values()))
            out["bodies"][body][m] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v)))
                                      if len(v) > 1 else float("nan")}
    if "lgm" in by and "edgegcn" in by:
        common = sorted(set(by["lgm"]) & set(by["edgegcn"]))
        for m, _ in METRICS:
            a, b = per_cancer(by, "lgm", m), per_cancer(by, "edgegcn", m)
            d = np.array([a[c] - b[c] for c in common])
            t, p = ttest_1samp(d)
            out["paired"][m] = {"n": len(d), "mean": float(d.mean()),
                                "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                                "t": float(t), "p": float(p), "wins": int((d > 0).sum()),
                                "per_cancer": {c: float(a[c] - b[c]) for c in common}}
        # secondary: pair by (cancer, seed) too, 33 pairs
        pairs = [(c, s) for c in common for s in by["lgm"][c] if s in by["edgegcn"][c]]
        d = np.array([by["lgm"][c][s]["auc_changed"] - by["edgegcn"][c][s]["auc_changed"] for c, s in pairs])
        t, p = ttest_1samp(d)
        out["paired_by_seed"] = {"n": len(d), "mean": float(d.mean()), "t": float(t), "p": float(p),
                                 "wins": int((d > 0).sum())}
    return out


def markdown(by: dict, s: dict) -> str:
    L = ["| body | changed-edge AUROC | changed-edge AP | overall AUROC | direction | #Params | seeds |",
         "|---|---|---|---|---|---|---|"]
    for body, name in (("lgm", "**LGM**"), ("edgegcn", "edge-aware GCN")):
        if body not in s["bodies"]:
            continue
        b = s["bodies"][body]
        f = lambda m: f"{b[m]['mean']:.4f} ± {b[m]['se']:.4f}"
        L.append(f"| {name} | {f('auc_changed')} | {f('ap_changed')} | {f('auc')} | {f('auc_direction')} "
                 f"| {b['n_params'][0]:,} | {len(b['seeds'])} |")
    if s["paired"]:
        L += ["", "Paired LGM − edge-GCN over the held-out cancers (seeds averaged within a cancer first):", "",
              "| metric | Δ mean ± SE | t | p | wins |", "|---|---|---|---|---|"]
        for m, name in METRICS:
            p = s["paired"][m]
            L.append(f"| {name} | {p['mean']:+.4f} ± {p['se']:.4f} | {p['t']:+.2f} | {p['p']:.2e} | {p['wins']}/{p['n']} |")
        q = s["paired_by_seed"]
        L += ["", f"Pairing every (cancer, seed) instead, n = {q['n']}: changed-edge AUROC Δ {q['mean']:+.4f}, "
              f"t = {q['t']:+.2f}, p = {q['p']:.2e}, wins {q['wins']}/{q['n']}.", "",
              "Per cancer (changed-edge AUROC, mean over seeds):", "",
              "| held-out cancer | LGM | edge-GCN | Δ |", "|---|---|---|---|"]
        a, b = per_cancer(by, "lgm", "auc_changed"), per_cancer(by, "edgegcn", "auc_changed")
        for c in sorted(a, key=lambda c: -a[c]):
            L.append(f"| {c} | {a[c]:.4f} | {b[c]:.4f} | {a[c] - b[c]:+.4f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/phase9/rows")
    ap.add_argument("--out", default="results/phase9/aggregate.md")
    args = ap.parse_args()
    by = load(pathlib.Path(args.rows))
    s = summarise(by)
    md = markdown(by, s)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(s, indent=1))
    print(md)


if __name__ == "__main__":
    main()
