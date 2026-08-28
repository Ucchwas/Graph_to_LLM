"""Phase-7 table: per arm, the mean +- SE over the held-out cancers of the overall and
changed-edge metrics, plus paired shared - baseline deltas (n = folds).

  python -m g2l.aggregate7 [--rows results/phase7/rows]
"""
import argparse
import glob
import json
import pathlib
from collections import defaultdict

import numpy as np

from g2l.aggregate6 import mean_se, ttest_1samp

ARMS = ["shared", "mean_change", "mean_tumour", "common_neighbors", "identity"]
METRICS = [("auc_changed", "changed-edge AUROC"), ("auc_gained", "gained"), ("auc_lost", "lost"),
           ("auc_direction", "direction"), ("auc", "overall AUROC"), ("ap_balanced", "overall AP@1:1")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/phase7/rows")
    args = ap.parse_args()
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{args.rows}/*.json")) if "gate" not in f]
    if not rows:
        raise SystemExit(f"no rows under {args.rows}")
    gate = pathlib.Path(args.rows).parent / "gate.json"
    by = defaultdict(dict)  # (density, arm) -> cancer -> row
    for r in rows:
        by[(r["density"], r["arm"])][r["cancer"]] = r
    commits = sorted({r["commit"][:8] for r in rows})
    n_genes = sorted({r["n_genes"] for r in rows})
    out = [f"# Phase 7 -- normal graph -> tumour graph translation ({len(rows)} rows @ {', '.join(commits)})", "",
           f"{len(by[(rows[0]['density'], rows[0]['arm'])])} held-out cancers, {n_genes} genes, "
           f"leave-one-cancer-out. Identity is exactly 0.5 on the changed-edge strata and 0.0 on "
           f"direction by construction.", ""]
    if gate.exists():
        g = json.loads(gate.read_text())
        out += [f"Gate 7.0: **{'PASS' if g['pass'] else 'FAIL'}** ({sum(r['pass'] for r in g['rows'])}/"
                f"{len(g['rows'])} probe cancers, {g['bootstraps']} bootstraps)", "",
                "| cancer | bootstrap Jaccard normal (p5) | tumour (p5) | normal vs tumour | verdict |",
                "|---|---|---|---|---|"]
        for r in g["rows"]:
            out.append(f"| {r['cancer']} | {r['normal_boot_jaccard_mean']:.4f} ({r['normal_boot_jaccard_p5']:.4f}) | "
                       f"{r['tumour_boot_jaccard_mean']:.4f} ({r['tumour_boot_jaccard_p5']:.4f}) | "
                       f"{r['normal_vs_tumour_jaccard']:.4f} | {'pass' if r['pass'] else 'FAIL'} |")
        out.append("")
    for density in sorted({r["density"] for r in rows}):
        main_tag = " (main protocol)" if abs(density - 0.01) < 1e-9 else " (sensitivity)"
        out += [f"## density {density}{main_tag}", "",
                "| arm | n | " + " | ".join(m for _, m in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
        for arm in ARMS:
            d = by.get((density, arm))
            if not d:
                continue
            cells = []
            for k, _ in METRICS:
                m, se, n = mean_se([r.get(k) for r in d.values()])
                cells.append(f"{m:.4f} +- {se:.4f}" if n > 1 else (f"{m:.4f}" if n else "-"))
            out.append(f"| {arm} | {len(d)} | " + " | ".join(cells) + " |")
        out.append("")
        s = by.get((density, "shared"), {})
        for arm in ARMS[1:]:
            b = by.get((density, arm), {})
            common = sorted(set(s) & set(b))
            if len(common) < 3:
                continue
            line = []
            for k, name in METRICS[:4]:
                d = np.array([s[c][k] - b[c][k] for c in common
                              if s[c].get(k) is not None and b[c].get(k) is not None])
                if len(d) > 1 and np.isfinite(d).all():
                    t, p = ttest_1samp(d)
                    line.append(f"{name} {d.mean():+.4f} (p {p:.2g}, {int((d > 0).sum())}/{len(d)})")
            if line:
                out.append(f"- paired shared - {arm} over {len(common)} cancers: " + "; ".join(line))
        out.append("")
    per = by.get((0.01, "shared"), {})
    if per:
        out += ["## Per-cancer detail, main protocol (shared model)", "",
                "| held-out cancer | changed AUROC | gained | lost | direction | overall AUROC | edges | gained/lost cells |",
                "|---|---|---|---|---|---|---|---|"]
        for c, r in sorted(per.items(), key=lambda kv: -kv[1]["auc_changed"]):
            out.append(f"| {c} | {r['auc_changed']:.4f} | {r['auc_gained']:.4f} | {r['auc_lost']:.4f} | "
                       f"{r['auc_direction']:.4f} | {r['auc']:.4f} | {r['n_edges_normal']} | "
                       f"{r['n_gained']}/{r['n_lost']} |")
        out.append("")
    text = "\n".join(out)
    print(text)
    pathlib.Path(args.rows).parent.joinpath("aggregate.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
