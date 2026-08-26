"""Phase-2 aggregation: rows -> per-arm LR selection (edge rule) -> table -> paired deltas.

  python -m g2l.aggregate                 # markdown to stdout
All rows must come from one commit; mixed commits abort (stale rows from an older code
version would otherwise enter the table silently).
"""
import glob
import json
import math
import pathlib
from collections import defaultdict

import numpy as np
from scipy import stats

ROWS = pathlib.Path("results/phase2/rows")
PHASE1 = pathlib.Path("results/phase1/table.jsonl")
COLS = ["auc", "ap", "auroc_sparse", "ap_sparse", "lift"]
REFERENCE = ["identity", "random", "ppr", "gae_600ep", "gat"]


def load_rows() -> list[dict]:
    rows = [json.load(open(f)) for f in sorted(glob.glob(str(ROWS / "*.json"))) if not f.endswith("probe.json")]
    commits = {r["commit"] for r in rows}
    if len(commits) > 1:
        raise SystemExit(f"rows span {len(commits)} commits: {sorted(commits)} -- delete the stale ones")
    return rows


def arm_label(r: dict) -> str:
    arm = f"scratch d{r['width']}" if r["arm"] == "scratch" else r["arm"]
    tags = [t for t in (r["decoder"] if r["decoder"] != "d1" else "", r["loss"] if r["loss"] != "masked" else "",
                        "shuffled-A" if r["input"] == "shuffled" else "", "gain 1" if r["encoder"] == "e1gain1" else "") if t]
    return arm + (f" ({', '.join(tags)})" if tags else "")


def select_lr(rows: list[dict], grid: list[float]) -> dict[str, dict]:
    """Per configuration: the LR with the best mean validation AUROC; flags grid edges."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[arm_label(r)][r["lr"]].append(r)
    out = {}
    for label, lrs in by.items():
        curve = {lr: float(np.mean([r["val_auc"] for r in rs])) for lr, rs in lrs.items()}
        best = max(curve, key=curve.get)
        edge = len(curve) > 1 and best in (min(curve), max(curve)) and best in (min(grid), max(grid))
        out[label] = {"lr": best, "rows": lrs[best], "curve": curve, "edge": edge}
    return out


def fmt(v, digits=3):
    return f"{np.mean(v):.{digits}f}±{np.std(v):.{digits}f}"


def table(selected: dict) -> str:
    lines = ["| model | lr | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift | params | best epoch | s/epoch |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    if PHASE1.exists():
        ref = defaultdict(list)
        for r in map(json.loads, PHASE1.open()):
            if r["model"] in REFERENCE:
                ref[r["model"]].append(r)
        for m in REFERENCE:
            rs = ref.get(m, [])
            if rs:
                v = {c: [r[c] for r in rs] for c in COLS}
                lines.append(f"| {m} (Phase 1, recon) | – | {len(rs)} | {fmt(v['auc'])} | {fmt(v['ap'])} | {fmt(v['auroc_sparse'])} "
                             f"| {fmt(v['ap_sparse'], 4)} | {np.mean(v['lift']):.0f} | – | – | – |")
    for label, s in selected.items():
        rs = s["rows"]
        v = {c: [r[c] for r in rs] for c in COLS}
        lines.append(f"| {label} | {s['lr']:.0e}{' EDGE' if s['edge'] else ''} | {len(rs)} | {fmt(v['auc'])} | {fmt(v['ap'])} "
                     f"| {fmt(v['auroc_sparse'])} | {fmt(v['ap_sparse'], 4)} | {np.mean(v['lift']):.0f} "
                     f"| {rs[0]['n_trainable'] / 1e6:.2f}M | {np.mean([r['best_epoch'] for r in rs]):.0f} "
                     f"| {np.mean([r['sec_per_epoch'] for r in rs]):.2f} |")
    return "\n".join(lines)


def paired_delta(a: list[dict], b: list[dict], col: str) -> dict:
    """Per-seed difference a - b on the same split; mean, SE, paired t, and the minimum
    detectable effect at alpha 0.05 / power 0.8 for this n."""
    sa, sb = {r["seed"]: r[col] for r in a}, {r["seed"]: r[col] for r in b}
    seeds = sorted(set(sa) & set(sb))
    d = np.array([sa[s] - sb[s] for s in seeds])
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    t, p = stats.ttest_rel([sa[s] for s in seeds], [sb[s] for s in seeds]) if n > 1 else (float("nan"), float("nan"))
    mde = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.8, n - 1)) * se if n > 1 else float("nan")
    return {"n": n, "mean": float(d.mean()), "se": float(se), "t": float(t), "p": float(p), "mde": float(mde)}


def deltas(selected: dict) -> str:
    pairs = [("pretrained", "none"), ("pretrained", "random"), ("random", "none")]
    for w in (256, 512):
        pairs.append(("pretrained", f"scratch d{w}"))
    lines = ["| Δ (A − B) | column | n | mean | SE | paired t | p | MDE |", "|---|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        if a in selected and b in selected:
            for col in ("auc", "ap", "ap_sparse"):
                d = paired_delta(selected[a]["rows"], selected[b]["rows"], col)
                lines.append(f"| {a} − {b} | {col} | {d['n']} | {d['mean']:+.4f} | {d['se']:.4f} | {d['t']:+.2f} | {d['p']:.3f} | {d['mde']:.4f} |")
    return "\n".join(lines)


def lr_curves(selected: dict) -> str:
    lines = ["| model | " + " | ".join(f"val AUROC @ {lr:.0e}" for lr in sorted(next(iter(selected.values()))["curve"], reverse=True)) + " |"]
    lines.append("|---|" + "---|" * (len(lines[0].split("|")) - 3))
    for label, s in selected.items():
        lines.append(f"| {label} | " + " | ".join(f"{s['curve'][lr]:.4f}" if lr in s["curve"] else "–" for lr in sorted(s["curve"], reverse=True)) + " |")
    return "\n".join(lines)


def main():
    import yaml

    grid = yaml.safe_load(open("configs/phase2.yaml"))["lr_grid"]
    rows = load_rows()
    if not rows:
        raise SystemExit("no rows")
    selected = select_lr(rows, grid)
    print(f"{len(rows)} rows @ {rows[0]['commit'][:8]}\n")
    print(table(selected), "\n")
    print("Validation-AUROC curves (mean over seeds):\n")
    print(lr_curves(selected), "\n")
    print("Paired per-seed deltas at the selected LRs:\n")
    print(deltas(selected))


if __name__ == "__main__":
    main()
