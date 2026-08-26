"""Phase-3 aggregation: rows -> per-arm (encoder LR, bias LR) selection -> table -> paired deltas.
Every reference (bias-off arm) is a Phase-3 row at the same commit; the Phase-2 rows are a
same-seed reproduction check, not a reference.

  python -m g2l.aggregate3
"""
import glob
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import yaml

from g2l.aggregate import COLS, fmt, paired_delta

ROWS = pathlib.Path("results/phase3/rows")
P2 = pathlib.Path("results/phase2/rows")
CONTROLS = ("shuffled-A",)  # never selected or extended: the edge rule does not apply
INIT_KEYS = ("z_norm", "logit_diag", "logit_offdiag_std", "logit_train_edge", "layer_rms")


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
    """Phase-2 bias-off rows at their selected LR, keyed by the Phase-3 label they reproduce."""
    sel = cfg2["selected_lr"]
    out = {}
    for arm in ("pretrained", "random", "none"):
        out[arm] = [r for r in p2 if r["arm"] == arm and r["decoder"] == "d1" and r["input"] == "real"
                    and r["loss"] == "masked" and r["encoder"] == "e1" and abs(r["lr"] - sel[arm]) < 1e-12]
    out["scratch d256 L4"] = [r for r in p2 if r["arm"] == "scratch" and r["width"] == 256 and abs(r["lr"] - 1e-3) < 1e-12]
    return out


def select(rows: list[dict], cfg: dict) -> dict[str, dict]:
    """Per label, the (encoder LR, bias LR) cell with the best mean val AUROC; `edge` if it sits on
    the boundary of either grid (controls exempt); `short` lists cells with fewer rows than seeds."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[label(r)][(r["lr"], r.get("lr_bias") if r["bias"] else None)].append(r)
    out = {}
    for lab, cells in by.items():
        curve = {k: float(np.mean([r["val_auc"] for r in rs])) for k, rs in cells.items()}
        best = max(curve, key=curve.get)
        lrs = sorted({k[0] for k in curve})
        lbs = sorted({k[1] for k in curve if k[1] is not None})
        control = any(c in lab for c in CONTROLS)
        edge = not control and ((len(lrs) > 1 and best[0] in (lrs[0], lrs[-1])) or (len(lbs) > 1 and best[1] in (lbs[0], lbs[-1])))
        short = {k: len(rs) for k, rs in cells.items() if len(rs) < len(cfg["seeds"])}
        out[lab] = {"key": best, "rows": cells[best], "curve": curve, "edge": edge, "short": short}
    return out


def cell_name(k) -> str:
    return f"{k[0]:.0e}" + (f"/{k[1]:.0e}" if k[1] is not None else "")


def table(selected: dict, refs: dict) -> str:
    lines = ["| model | enc lr | bias lr | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift | best epoch | s/epoch |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, lr, lb, rs, mark=""):
        v = {c: [r[c] for r in rs] for c in COLS}
        lines.append(f"| {name} | {lr:.0e}{mark} | {lb} | {len(rs)} | {fmt(v['auc'])} | {fmt(v['ap'])} | {fmt(v['auroc_sparse'])} "
                     f"| {fmt(v['ap_sparse'], 4)} | {np.mean(v['lift']):.0f} | {np.mean([r['best_epoch'] for r in rs]):.0f} "
                     f"| {np.mean([r['sec_per_epoch'] for r in rs]):.2f} |")

    for name, rs in refs.items():
        if rs:
            row(f"{name} (Phase 2)", rs[0]["lr"], "–", rs)
    for lab, s in selected.items():
        lr, lb = s["key"]
        row(lab, lr, f"{lb:.0e}" if lb is not None else "–", s["rows"], " EDGE" if s["edge"] else "")
    return "\n".join(lines)


def reproduction(selected: dict, refs: dict) -> str:
    """Same recipe, same seeds, two commits: the Phase-3 bias-off rows against the Phase-2 rows."""
    lines = ["| arm | seeds | Phase-2 AUROC | Phase-3 AUROC | mean abs Δ | max abs Δ |", "|---|---|---|---|---|---|"]
    for name, p2 in refs.items():
        if name in selected and p2:
            a = {r["seed"]: r["auc"] for r in p2}
            b = {r["seed"]: r["auc"] for r in selected[name]["rows"]}
            seeds = sorted(set(a) & set(b))
            if seeds:
                d = [abs(a[s] - b[s]) for s in seeds]
                lines.append(f"| {name} | {len(seeds)} | {fmt([a[s] for s in seeds])} | {fmt([b[s] for s in seeds])} "
                             f"| {np.mean(d):.4f} | {np.max(d):.4f} |")
    return "\n".join(lines)


def inertness(rows: list[dict]) -> str:
    """Zero-init check on the real bodies: before any step a biased model equals its bias-off
    counterpart, so a biased row's first-batch stats must match the bias-off row at the same
    arm / width / layers / input / fraction / seed (the learning rates play no role at init)."""
    def key(r):
        return (r["arm"], r["width"], r["layers"], r["input"], r["frac"], r["seed"])

    off = {key(r): r["init"] for r in rows if not r["bias"]}
    checked, bad = 0, []
    for r in rows:
        if not r["bias"] or key(r) not in off:
            continue
        checked += 1
        a, b = r["init"], off[key(r)]
        va = np.concatenate([np.atleast_1d(np.asarray(a[k], dtype=float)) for k in INIT_KEYS if k in a])
        vb = np.concatenate([np.atleast_1d(np.asarray(b[k], dtype=float)) for k in INIT_KEYS if k in b])
        rel = float(np.max(np.abs(va - vb) / (np.abs(vb) + 1e-8))) if va.shape == vb.shape else float("inf")
        if rel > 1e-3:
            bad.append((r["key"], rel))
    line = (f"inertness at init: {checked} biased rows compared with their bias-off counterpart "
            f"({', '.join(INIT_KEYS)}); {len(bad)} mismatch(es) above 1e-3 relative")
    return line + "".join(f"\n  {k}: {rel:.2e}" for k, rel in bad[:10])


def deltas(selected: dict) -> str:
    pairs = [("pretrained + SPD", "pretrained"), ("random + SPD", "random"), ("pretrained + SPD", "none"),
             ("pretrained + SPD", "random + SPD"), ("pretrained + SPD", "scratch d256 L4 + SPD"),
             ("pretrained + SPD", "scratch d256 L1 + SPD"), ("scratch d256 L4 + SPD", "scratch d256 L4"),
             ("scratch d256 L1 + SPD", "scratch d256 L1"), ("random + SPD", "none"),
             ("pretrained", "none"), ("pretrained", "random"),
             ("pretrained + SPD", "pretrained + SPD shuffled-A")]
    for f in sorted({lab.split("train ")[1] for lab in selected if "train " in lab}):  # data-fraction stage
        pairs += [(f"pretrained + SPD train {f}", f"random + SPD train {f}"),
                  (f"pretrained + SPD train {f}", f"none train {f}"), (f"random + SPD train {f}", f"none train {f}")]
    lines = ["| Δ (A − B) | column | n | mean | SE | paired t | p | MDE |", "|---|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        if a in selected and b in selected:
            for col in ("auc", "ap", "ap_sparse"):
                d = paired_delta(selected[a]["rows"], selected[b]["rows"], col)
                lines.append(f"| {a} − {b} | {col} | {d['n']} | {d['mean']:+.4f} | {d['se']:.4f} | {d['t']:+.2f} | {d['p']:.3f} | {d['mde']:.4f} |")
    return "\n".join(lines)


def surfaces(selected: dict) -> str:
    out = []
    for lab, s in selected.items():
        lbs = sorted({k[1] for k in s["curve"] if k[1] is not None}, reverse=True)
        lrs = sorted({k[0] for k in s["curve"]}, reverse=True)
        if not lbs:
            if len(lrs) > 1:
                out.append(f"\n{lab} — mean val AUROC by encoder LR (* selected): " + ", ".join(
                    f"{lr:.0e} {s['curve'][(lr, None)]:.4f}" + (" *" if (lr, None) == s["key"] else "") for lr in lrs))
            continue
        out.append(f"\n{lab} — mean val AUROC (rows: encoder LR; columns: bias LR; * selected; (n) if fewer rows than seeds)\n")
        out.append("| enc lr | " + " | ".join(f"{lb:.0e}" for lb in lbs) + " |")
        out.append("|---|" + "---|" * len(lbs))
        for lr in lrs:
            cells = []
            for lb in lbs:
                k = (lr, lb)
                if k not in s["curve"]:
                    cells.append("–")
                    continue
                cells.append(f"{s['curve'][k]:.4f}" + (" *" if k == s["key"] else "") + (f" ({s['short'][k]})" if k in s["short"] else ""))
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = yaml.safe_load(open("configs/phase3.yaml"))
    cfg2 = yaml.safe_load(open("configs/phase2.yaml"))
    rows = load(ROWS)
    if not rows:
        raise SystemExit("no rows")
    refs = phase2_refs(load(P2), cfg2) if P2.exists() else {}
    selected = select(rows, cfg)
    p2c = next((rs[0]["commit"][:8] for rs in refs.values() if rs), "–")
    print(f"{len(rows)} Phase-3 rows @ {rows[0]['commit'][:8]}; Phase-2 rows @ {p2c} (reproduction check only)\n")
    short = {lab: {cell_name(k): n for k, n in s["short"].items()} for lab, s in selected.items() if s["short"]}
    if short:
        print(f"INCOMPLETE — cells with fewer rows than the {len(cfg['seeds'])} seeds: {short}\n")
    print(table(selected, refs), "\n")
    print("Same-seed reproduction of the Phase-2 bias-off rows at this commit:\n")
    print(reproduction(selected, refs), "\n")
    print(inertness(rows), "\n")
    print("Paired per-seed deltas (all rows at this commit):\n")
    print(deltas(selected), "\n")
    print("LR surfaces (mean val AUROC):")
    print(surfaces(selected))
    print(bias_tables(selected))


if __name__ == "__main__":
    main()
