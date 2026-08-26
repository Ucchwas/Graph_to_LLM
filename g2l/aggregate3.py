"""Phase-3 aggregation: rows -> per-arm (encoder LR, bias LR) selection -> table -> paired deltas
against the Phase-2 bias-off rows (same seeds and splits).

  python -m g2l.aggregate3
"""
import glob
import json
import pathlib
from collections import defaultdict

import numpy as np
import yaml

from g2l.aggregate import COLS, fmt, paired_delta

ROWS = pathlib.Path("results/phase3/rows")
P2 = pathlib.Path("results/phase2/rows")


def load(path: pathlib.Path) -> list[dict]:
    rows = [json.load(open(f)) for f in sorted(glob.glob(str(path / "*.json"))) if not f.endswith("probe.json")]
    commits = {r["commit"] for r in rows}
    if len(commits) > 1:
        raise SystemExit(f"{path}: rows span {len(commits)} commits: {sorted(commits)} -- delete the stale ones")
    return rows


def label(r: dict) -> str:
    arm = r["arm"] + (f" d{r['width']} L{r['layers']}" if r["arm"] == "scratch" else "")
    tags = [t for t in ("+ SPD" if r["bias"] else "", "shuffled-A" if r["input"] == "shuffled" else "",
                        f"train {r['frac']:.0%}" if r["frac"] < 1 else "") if t]
    return arm + (" " + " ".join(tags) if tags else "")


def phase2_refs(p2: list[dict], cfg2: dict) -> dict[str, list[dict]]:
    """Bias-off references at their Phase-2 selected LR: pretrained, random, none, scratch d256 L4."""
    sel = cfg2["selected_lr"]
    out = {}
    for arm in ("pretrained", "random", "none"):
        out[arm] = [r for r in p2 if r["arm"] == arm and r["decoder"] == "d1" and r["input"] == "real"
                    and r["loss"] == "masked" and r["encoder"] == "e1" and abs(r["lr"] - sel[arm]) < 1e-12]
    out["scratch d256 L4"] = [r for r in p2 if r["arm"] == "scratch" and r["width"] == 256 and abs(r["lr"] - 1e-3) < 1e-12]
    return out


def select(rows: list[dict], cfg: dict) -> dict[str, dict]:
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[label(r)][(r["lr"], r.get("lr_bias") if r["bias"] else None)].append(r)
    out = {}
    for lab, cells in by.items():
        curve = {k: float(np.mean([r["val_auc"] for r in rs])) for k, rs in cells.items()}
        best = max(curve, key=curve.get)
        lrs = sorted({k[0] for k in curve})
        lbs = sorted({k[1] for k in curve if k[1] is not None})
        edge = (len(lrs) > 1 and best[0] in (lrs[0], lrs[-1])) or (len(lbs) > 1 and best[1] in (lbs[0], lbs[-1]))
        out[lab] = {"key": best, "rows": cells[best], "curve": curve, "edge": edge}
    return out


def table(selected: dict, refs: dict) -> str:
    lines = ["| model | enc lr | bias lr | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift | best epoch | s/epoch |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, lr, lb, rs, mark=""):
        v = {c: [r[c] for r in rs] for c in COLS}
        lines.append(f"| {name} | {lr:.0e} | {lb} | {len(rs)} | {fmt(v['auc'])} | {fmt(v['ap'])} | {fmt(v['auroc_sparse'])} "
                     f"| {fmt(v['ap_sparse'], 4)} | {np.mean(v['lift']):.0f} | {np.mean([r['best_epoch'] for r in rs]):.0f} "
                     f"| {np.mean([r['sec_per_epoch'] for r in rs]):.2f} |{mark}")

    for name, rs in refs.items():
        if rs:
            row(f"{name} (Phase 2, no bias)", rs[0]["lr"], "–", rs)
    for lab, s in selected.items():
        lr, lb = s["key"]
        row(lab, lr, f"{lb:.0e}" if lb is not None else "–", s["rows"], " EDGE" if s["edge"] else "")
    return "\n".join(lines)


def merged(selected: dict, refs: dict, name: str) -> list[dict]:
    """Bias-off arm at 10 seeds: Phase-2 rows plus the Phase-3 seeds it lacks."""
    rows = list(refs.get(name, []))
    have = {r["seed"] for r in rows}
    rows += [r for r in selected.get(name, {"rows": []})["rows"] if r["seed"] not in have]
    return rows


def deltas(selected: dict, refs: dict) -> str:
    def rows_of(name):
        return merged(selected, refs, name) if name in refs else selected[name]["rows"] if name in selected else None

    pairs = [("pretrained + SPD", "pretrained"), ("random + SPD", "random"), ("pretrained + SPD", "none"),
             ("pretrained + SPD", "random + SPD"), ("pretrained + SPD", "scratch d256 L4 + SPD"),
             ("pretrained + SPD", "scratch d256 L1 + SPD"), ("scratch d256 L4 + SPD", "scratch d256 L4"),
             ("scratch d256 L1 + SPD", "scratch d256 L1"), ("random + SPD", "none"),
             ("pretrained + SPD", "pretrained + SPD shuffled-A")]
    lines = ["| Δ (A − B) | column | n | mean | SE | paired t | p | MDE |", "|---|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        ra, rb = rows_of(a), rows_of(b)
        if ra and rb:
            for col in ("auc", "ap", "ap_sparse"):
                d = paired_delta(ra, rb, col)
                lines.append(f"| {a} − {b} | {col} | {d['n']} | {d['mean']:+.4f} | {d['se']:.4f} | {d['t']:+.2f} | {d['p']:.3f} | {d['mde']:.4f} |")
    return "\n".join(lines)


def surfaces(selected: dict) -> str:
    out = []
    for lab, s in selected.items():
        lbs = sorted({k[1] for k in s["curve"] if k[1] is not None}, reverse=True)
        lrs = sorted({k[0] for k in s["curve"]}, reverse=True)
        if not lbs:
            continue
        out.append(f"\n{lab} — mean val AUROC (rows: encoder LR; columns: bias LR; * selected)\n")
        out.append("| enc lr | " + " | ".join(f"{lb:.0e}" for lb in lbs) + " |")
        out.append("|---|" + "---|" * len(lbs))
        for lr in lrs:
            cells = [f"{s['curve'][(lr, lb)]:.4f}" + (" *" if (lr, lb) == s["key"] else "") if (lr, lb) in s["curve"] else "–" for lb in lbs]
            out.append(f"| {lr:.0e} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def bias_tables(selected: dict) -> str:
    out = []
    for lab, s in selected.items():
        rs = [r for r in s["rows"] if "bias_table" in r]
        if not rs:
            continue
        t = np.mean([r["bias_table"] for r in rs], axis=0)  # [heads, buckets], mean over seeds
        out.append(f"\n{lab} — learned bias, mean over seeds and heads per distance bucket (0 self … 9 far/unreachable):")
        out.append("  " + "  ".join(f"d{k}:{v:+.3f}" for k, v in enumerate(t.mean(0))))
        out.append(f"  per-head spread (std over heads) at d1 {t[:, 1].std():.3f}, d9 {t[:, 9].std():.3f}; max |bias| {np.abs(t).max():.3f}")
    return "\n".join(out)


def main():
    cfg = yaml.safe_load(open("configs/phase3.yaml"))
    cfg2 = yaml.safe_load(open("configs/phase2.yaml"))
    rows = load(ROWS)
    if not rows:
        raise SystemExit("no rows")
    refs = phase2_refs(load(P2), cfg2) if P2.exists() else {}
    selected = select(rows, cfg)
    print(f"{len(rows)} Phase-3 rows @ {rows[0]['commit'][:8]}; Phase-2 references @ {next(iter(refs.values()))[0]['commit'][:8] if refs else '–'}\n")
    print(table(selected, refs), "\n")
    print("Paired per-seed deltas (bias-off references from Phase 2 where needed):\n")
    print(deltas(selected, refs), "\n")
    print("LR surfaces (mean val AUROC):")
    print(surfaces(selected))
    print(bias_tables(selected))


if __name__ == "__main__":
    main()
