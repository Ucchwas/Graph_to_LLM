"""Phase-6 table: per layer group (train / val / test) and arm, the mean +- SE over layers of
AUROC / AP on the layer's test pairs and on the globally hidden pairs; paired shared - independent
per layer; the shared model's reduced-input columns on the test layers.

  python -m g2l.aggregate6 [--rows results/phase6/rows]
"""
import argparse
import glob
import json
import pathlib
from collections import defaultdict

import numpy as np
from scipy import stats

ARMS = ["shared", "independent", "gae0", "knn", "pair_frequency", "common_neighbors", "identity"]
METRICS = [("auc", "AUROC"), ("ap", "AP@1:1"), ("auc_hidden", "AUROC hidden pairs"), ("ap_hidden", "AP hidden pairs"),
           ("auroc_sparse", "AUROC sparse"), ("ap_sparse", "AP sparse")]


def mean_se(v):
    v = np.array([x for x in v if x is not None], dtype=float)
    return (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan"), len(v)) if len(v) else (np.nan, np.nan, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/phase6/rows")
    args = ap.parse_args()
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{args.rows}/decagon_*.json"))]
    summ = next((json.load(open(f)) for f in glob.glob(f"{args.rows}/summary_shared_*.json")), None)
    by = defaultdict(dict)  # (group, arm) -> layer -> row
    for r in rows:
        by[(r["group"], r["arm"])][r["layer"]] = r
    commits = sorted({r["commit"][:8] for r in rows})
    out = [f"# Phase 6 -- Decagon aligned multi-layer completion ({len(rows)} rows @ {', '.join(commits)})", ""]
    if summ:
        out += [f"shared model: val-layer AUROC {summ['val_auc']:.4f} at epoch {summ['best_epoch']} of {summ['epochs']}, "
                f"{summ['sec_per_epoch']:.1f} s/epoch, {summ['n_trainable']:,} params, peak {summ['peak_mem_gb']} GB; "
                f"split {summ['info']}", ""]
    for group, title in (("test", "E2 held-out layers (completion from the layer's own partial input; shared never trained on them)"),
                         ("val", "val layers (selection set of the shared model)"),
                         ("train", "E1 train layers (in-distribution)")):
        out += [f"## {title}", "", "| arm | n layers | " + " | ".join(m for _, m in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
        for arm in ARMS:
            d = by.get((group, arm))
            if not d:
                continue
            cells = []
            for k, _ in METRICS:
                m, se, n = mean_se([r.get(k) for r in d.values()])
                cells.append(f"{m:.4f} +- {se:.4f}" if n > 1 else (f"{m:.4f}" if n else "-"))
            out.append(f"| {arm} | {len(d)} | " + " | ".join(cells) + " |")
        out.append("")
        s, i = by.get((group, "shared"), {}), by.get((group, "independent"), {})
        common = sorted(set(s) & set(i))
        if len(common) > 1:
            out += [f"paired shared - independent over {len(common)} layers:", ""]
            for k, name in METRICS[:4]:
                d = np.array([s[l][k] - i[l][k] for l in common if s[l].get(k) is not None and i[l].get(k) is not None])
                if len(d) > 1:
                    t, p = stats.ttest_1samp(d, 0.0)
                    out.append(f"- {name}: delta = {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f} (t = {t:.2f}, p = {p:.2g}, n = {len(d)}; "
                               f"shared better on {int((d > 0).sum())} / {len(d)})")
            out.append("")
    s = by.get(("test", "shared"), {})
    fr = sorted({k for r in s.values() for k in r if k.startswith("auc_in")})
    if fr:
        out += ["## E4 shared model on the test layers from reduced input (fraction of the layer's train + val pairs kept)", "",
                "| input | AUROC | AP@1:1 |", "|---|---|---|"]
        m, se, _ = mean_se([r["auc"] for r in s.values()])
        ma, sa, _ = mean_se([r["ap"] for r in s.values()])
        out.append(f"| 1.0 (all) | {m:.4f} +- {se:.4f} | {ma:.4f} +- {sa:.4f} |")
        for k in sorted(fr, key=lambda x: -float(x[6:])):
            m, se, _ = mean_se([r.get(k) for r in s.values()])
            ma, sa, _ = mean_se([r.get("ap_in" + k[6:]) for r in s.values()])
            out.append(f"| {k[6:]} | {m:.4f} +- {se:.4f} | {ma:.4f} +- {sa:.4f} |")
        out.append("")
    text = "\n".join(out)
    print(text)
    pathlib.Path(args.rows).parent.joinpath("aggregate.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
