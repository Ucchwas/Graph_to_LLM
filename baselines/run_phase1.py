"""Phase-1 runner: every baseline x 5 seeds -> results/phase1/table.jsonl.
Idempotent: (model, seed) rows already present are skipped."""
import json
import pathlib
import sys

import torch

from baselines import feature_only, gae, heuristics
from baselines.common import run_baseline

OUT = pathlib.Path("results/phase1/table.jsonl")
SEEDS = [0, 1, 2, 3, 4]

ROWS = {
    **{k: heuristics.make_score_fn(k) for k in (*heuristics.CONTROLS, *heuristics.HEURISTICS)},
    "feature_only_logreg": feature_only.score_fn,
    "gae": gae.make_score_fn("gae"),
    "gae_600ep": gae.make_score_fn("gae", fixed_epochs=600),
    "vgae": gae.make_score_fn("vgae"),
    "gat": gae.make_score_fn("gat"),
}


def main(only=None):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {(r["model"], r["seed"]) for r in map(json.loads, OUT.open())}
    for name, fn in ROWS.items():
        if only and name != only:
            continue
        for seed in SEEDS:
            if (name, seed) in done:
                continue
            row = run_baseline(name, seed, fn)
            with OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{name:22s} seed={seed}  auc={row['auc']:.4f}  ap={row['ap']:.4f}"
                  f"  ap_sparse={row['ap_sparse']:.4f}  lift={row['lift']:.0f}"
                  f"  [{row['wallclock_s']}s]", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main(only=sys.argv[1] if len(sys.argv) > 1 else None)
