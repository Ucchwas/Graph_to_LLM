"""Phase-3 / 3B aggregation: rows -> per-arm cell selection (encoder LR, bias LR, LoRA LR) ->
table -> paired deltas. Every reference (bias-off / LoRA-off arm) is a row of the same phase at
the same commit; the previous phase's rows are a same-seed reproduction check, not a reference.

  python -m g2l.aggregate3                       (Phase 3: results/phase3/rows vs results/phase2/rows)
  python -m g2l.aggregate3 --phase 3b            (Phase 3B: results/phase3b/rows vs results/phase3/rows)
"""
import argparse
import glob
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import yaml

from g2l.aggregate import COLS, fmt, paired_delta

PHASES = {"3": ("results/phase3/rows", "results/phase2/rows", "configs/phase3.yaml"),
          "3b": ("results/phase3b/rows", "results/phase3/rows", "configs/phase3b.yaml")}
CONTROLS = ("shuffled-A",)  # never selected or extended: the edge rule does not apply
INIT_KEYS = ("z_norm", "logit_diag", "logit_offdiag_std", "logit_train_edge", "layer_rms")
PAIRS = [("pretrained + SPD", "pretrained"), ("random + SPD", "random"), ("pretrained + SPD", "none"),
         ("pretrained + SPD", "random + SPD"), ("pretrained + SPD", "scratch d256 L4 + SPD"),
         ("pretrained + SPD", "scratch d256 L1 + SPD"), ("scratch d256 L4 + SPD", "scratch d256 L4"),
         ("scratch d256 L1 + SPD", "scratch d256 L1"), ("random + SPD", "none"),
         ("pretrained", "none"), ("pretrained", "random"),
         ("pretrained + SPD", "pretrained + SPD shuffled-A"),
         # Phase 3B
         ("pretrained + SPD + LoRA", "pretrained + SPD"), ("pretrained + SPD + LoRA", "none"),
         ("pretrained + SPD + LoRA", "random + SPD + LoRA"), ("random + SPD + LoRA", "random + SPD"),
         ("pretrained + LoRA", "pretrained + SPD + LoRA"), ("pretrained + LoRA", "none"),
         ("random + SPD + LoRA", "none"), ("pretrained + SPD + LoRA", "pretrained + SPD + LoRA shuffled-A")]


def load(path: pathlib.Path) -> list[dict]:
    rows = [json.load(open(f)) for f in sorted(glob.glob(str(path / "*.json"))) if not f.endswith("probe.json")]
    commits = {r["commit"] for r in rows}
    if len(commits) > 1:
        raise SystemExit(f"{path}: rows span {len(commits)} commits: {sorted(commits)} -- delete the stale ones")
    return rows


def label(r: dict) -> str:
    """Arm name + tags. Phase-2 rows (no bias / frac / layers keys) label as their bias-off arm."""
    arm = r["arm"] + (f" d{r['width']} L{r.get('layers', 4)}" if r["arm"] == "scratch" else "")
    tags = [t for t in ("+ SPD" if r.get("bias") else "", "+ LoRA" if r.get("lora") else "",
                        "shuffled-A" if r.get("input", "real") == "shuffled" else "",
                        f"train {r['frac']:.0%}" if r.get("frac", 1.0) < 1 else "") if t]
    return arm + (" " + " ".join(tags) if tags else "")


def cell(r: dict) -> tuple:
    return (r["lr"], r.get("lr_bias") if r.get("bias") else None, r.get("lr_lora") if r.get("lora") else None)


def cell_name(k) -> str:
    return "/".join(f"{v:.0e}" for v in k if v is not None)


def comparable(r: dict) -> bool:
    """Rows on the shared recipe (E1, D1, masked loss); Phase-2 rows carry the extra keys."""
    return r.get("decoder", "d1") == "d1" and r.get("loss", "masked") == "masked" and r.get("encoder", "e1") == "e1"


def select(rows: list[dict], cfg: dict) -> dict[str, dict]:
    """Per label, the cell with the best mean val AUROC over the base seeds (`cfg['seeds']`;
    extension seeds never take part in selection, they only enter the deltas); `edge` if the best
    sits on the boundary of any axis with more than one point (controls exempt); `short` lists
    cells with fewer base-seed rows than seeds."""
    base = set(cfg["seeds"])
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[label(r)][cell(r)].append(r)
    out = {}
    for lab, cells in by.items():
        n_base = {k: sum(r["seed"] in base for r in rs) for k, rs in cells.items()}
        curve = {k: float(np.mean([r["val_auc"] for r in rs if r["seed"] in base])) for k, rs in cells.items() if n_base[k]}
        best = max(curve, key=curve.get)
        control = any(c in lab for c in CONTROLS)
        edge = False
        for axis in range(3):
            pts = sorted({k[axis] for k in curve if k[axis] is not None})
            edge |= len(pts) > 1 and best[axis] in (pts[0], pts[-1])
        short = {k: n for k, n in n_base.items() if n < len(base)}
        out[lab] = {"key": best, "rows": cells[best], "curve": curve, "edge": edge and not control, "short": short}
    return out


def table(selected: dict, prior: list[dict], prior_name: str) -> str:
    lines = ["| model | enc lr | bias lr | LoRA lr | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift | best epoch | s/epoch |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, k, rs, mark=""):
        v = {c: [r[c] for r in rs] for c in COLS}
        cells_ = [f"{k[0]:.0e}{mark}"] + [f"{x:.0e}" if x is not None else "–" for x in k[1:]]
        lines.append(f"| {name} | " + " | ".join(cells_) + f" | {len(rs)} | {fmt(v['auc'])} | {fmt(v['ap'])} | {fmt(v['auroc_sparse'])} "
                     f"| {fmt(v['ap_sparse'], 4)} | {np.mean(v['lift']):.0f} | {np.mean([r['best_epoch'] for r in rs]):.0f} "
                     f"| {np.mean([r['sec_per_epoch'] for r in rs]):.2f} |")

    for lab, s in selected.items():  # the previous phase's rows at this phase's selected cell, if any
        rs = [r for r in prior if label(r) == lab and cell(r) == s["key"]]
        if rs:
            row(f"{lab} ({prior_name})", s["key"], rs)
    for lab, s in selected.items():
        row(lab, s["key"], s["rows"], " EDGE" if s["edge"] else "")
    return "\n".join(lines)


def reproduction(selected: dict, prior: list[dict]) -> str:
    """Same recipe, same seeds, two commits: this phase's rows against the previous phase's rows
    at the same label and cell."""
    lines = ["| arm | seeds | previous AUROC | this phase AUROC | mean abs Δ | max abs Δ |", "|---|---|---|---|---|---|"]
    for lab, s in selected.items():
        a = {r["seed"]: r["auc"] for r in prior if label(r) == lab and cell(r) == s["key"]}
        b = {r["seed"]: r["auc"] for r in s["rows"]}
        seeds = sorted(set(a) & set(b))
        if seeds:
            d = [abs(a[x] - b[x]) for x in seeds]
            lines.append(f"| {lab} | {len(seeds)} | {fmt([a[x] for x in seeds])} | {fmt([b[x] for x in seeds])} "
                         f"| {np.mean(d):.4f} | {np.max(d):.4f} |")
    return "\n".join(lines)


def inertness(rows: list[dict]) -> str:
    """Zero-init check on the real bodies: before any step a biased / LoRA model equals its plain
    counterpart, so a row's first-batch stats must match the plain row at the same arm / width /
    layers / input / fraction / seed (learning rates play no role at init). Rows from different
    jobs can differ by ~1 bf16 ulp (kernel reduction order), so the tolerance is 1e-2 relative."""
    def key(r):
        return (r["arm"], r["width"], r["layers"], r["input"], r["frac"], r["seed"])

    plain = {key(r): r["init"] for r in rows if not r["bias"] and not r.get("lora")}
    checked, bad = 0, []
    for r in rows:
        if not (r["bias"] or r.get("lora")) or key(r) not in plain:
            continue
        checked += 1
        a, b = r["init"], plain[key(r)]
        va = np.concatenate([np.atleast_1d(np.asarray(a[k], dtype=float)) for k in INIT_KEYS if k in a])
        vb = np.concatenate([np.atleast_1d(np.asarray(b[k], dtype=float)) for k in INIT_KEYS if k in b])
        rel = float(np.max(np.abs(va - vb) / (np.abs(vb) + 1e-8))) if va.shape == vb.shape else float("inf")
        if rel > 1e-2:
            bad.append((r["key"], rel))
    line = (f"inertness at init: {checked} biased / LoRA rows compared with their plain counterpart "
            f"({', '.join(INIT_KEYS)}); {len(bad)} mismatch(es) above 1e-2 relative")
    return line + "".join(f"\n  {k}: {rel:.2e}" for k, rel in bad[:10])


def deltas(selected: dict) -> str:
    pairs = list(PAIRS)
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
        curve, key, short = s["curve"], s["key"], s["short"]
        lls = sorted({k[2] for k in curve if k[2] is not None}, reverse=True)
        lbs = sorted({k[1] for k in curve if k[1] is not None}, reverse=True)
        lrs = sorted({k[0] for k in curve}, reverse=True)

        def val(k):
            return f"{curve[k]:.4f}" + (" *" if k == key else "") + (f" ({short[k]})" if k in short else "")

        if lls:  # LoRA axis (Phase 3B): encoder / bias LR fixed per arm
            for lr in lrs:
                for lb in lbs or [None]:
                    ks = [(lr, lb, ll) for ll in lls if (lr, lb, ll) in curve]
                    if ks:
                        out.append(f"\n{lab} — mean val AUROC by LoRA LR at enc {lr:.0e}" + (f", bias {lb:.0e}" if lb else "")
                                   + " (* selected): " + ", ".join(f"{k[2]:.0e} {val(k)}" for k in ks))
            continue
        if not lbs:
            if len(lrs) > 1:
                out.append(f"\n{lab} — mean val AUROC by encoder LR (* selected): "
                           + ", ".join(f"{lr:.0e} {val((lr, None, None))}" for lr in lrs))
            continue
        out.append(f"\n{lab} — mean val AUROC (rows: encoder LR; columns: bias LR; * selected; (n) if fewer rows than seeds)\n")
        out.append("| enc lr | " + " | ".join(f"{lb:.0e}" for lb in lbs) + " |")
        out.append("|---|" + "---|" * len(lbs))
        for lr in lrs:
            out.append(f"| {lr:.0e} | " + " | ".join(val((lr, lb, None)) if (lr, lb, None) in curve else "–" for lb in lbs) + " |")
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


def main(phase: str = "3"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows_dir, prior_dir, cfg_path = (pathlib.Path(p) for p in PHASES[phase])
    cfg = yaml.safe_load(open(cfg_path))
    rows = load(rows_dir)
    if not rows:
        raise SystemExit("no rows")
    prior = [r for r in load(prior_dir) if comparable(r)] if prior_dir.exists() else []
    prior_name = f"Phase {'2' if phase == '3' else '3'}"
    selected = select(rows, cfg)
    print(f"{len(rows)} Phase-{phase.upper()} rows @ {rows[0]['commit'][:8]}; {prior_name} rows @ "
          f"{prior[0]['commit'][:8] if prior else '–'} (reproduction check only)\n")
    short = {lab: {cell_name(k): n for k, n in s["short"].items()} for lab, s in selected.items() if s["short"]}
    if short:
        print(f"INCOMPLETE — cells with fewer rows than the {len(cfg['seeds'])} seeds: {short}\n")
    print(table(selected, prior, prior_name), "\n")
    print(f"Same-seed reproduction of the {prior_name} rows at this commit:\n")
    print(reproduction(selected, prior), "\n")
    print(inertness(rows), "\n")
    print("Paired per-seed deltas (all rows at this commit):\n")
    print(deltas(selected), "\n")
    print("LR surfaces (mean val AUROC):")
    print(surfaces(selected))
    print(bias_tables(selected))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="3", choices=sorted(PHASES))
    main(ap.parse_args().phase)
