"""Aggregate Phase-1 JSONL rows into the mean+-std markdown table."""
import json
import pathlib
from collections import defaultdict

import numpy as np

FILES = [pathlib.Path("results/phase1/table.jsonl"),
         pathlib.Path("results/phase1/scratch_sweep.jsonl")]
COLS = ["auc", "ap", "auroc_sparse", "ap_sparse", "lift"]


def rows():
    for f in FILES:
        if f.exists():
            for r in map(json.loads, f.open()):
                if r.get("stage", "final") == "final":
                    yield r


def table() -> str:
    by = defaultdict(list)
    for r in rows():
        by[r["model"]].append(r)
    lines = ["| model | n | AUC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift |",
             "|---|---|---|---|---|---|---|"]
    for m, rs in by.items():
        v = {c: np.array([r[c] for r in rs]) for c in COLS}
        extra = f" ({rs[0]['n_params']/1e6:.1f}M)" if "n_params" in rs[0] else ""
        lines.append(
            f"| {m}{extra} | {len(rs)} "
            f"| {v['auc'].mean():.3f}±{v['auc'].std():.3f} "
            f"| {v['ap'].mean():.3f}±{v['ap'].std():.3f} "
            f"| {v['auroc_sparse'].mean():.3f}±{v['auroc_sparse'].std():.3f} "
            f"| {v['ap_sparse'].mean():.4f}±{v['ap_sparse'].std():.4f} "
            f"| {v['lift'].mean():.0f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
