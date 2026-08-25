"""Scratch-transformer capacity sweep: per-d_model LR selection on seed-0 val AUC,
then 5 seeds at the chosen LR. The curve is Phase 2's matched-params reference."""
import json
import pathlib

import torch

from baselines.common import get_split
from baselines.scratch_transformer import train_one
from g2l.metrics import evaluate_edge_split

OUT = pathlib.Path("results/phase1/scratch_sweep.jsonl")
D_MODELS = [128, 256, 512, 1024, 2048]
LRS = [1e-3, 3e-4, 1e-4]
SEEDS = [0, 1, 2, 3, 4]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if OUT.exists():
        for r in map(json.loads, OUT.open()):
            done[(r["d_model"], r["lr"], r["seed"], r["stage"])] = r
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for d in D_MODELS:
        # stage 1: LR selection on seed 0 by val AUC
        val_auc = {}
        for lr in LRS:
            key = (d, lr, 0, "lr_select")
            if key in done:
                val_auc[lr] = done[key]["val_auc"]
                continue
            data, split = get_split(0)
            _, auc, n_params, best_epoch = train_one(data, split, device, d, lr, seed=0)
            row = {"stage": "lr_select", "d_model": d, "lr": lr, "seed": 0,
                   "val_auc": auc, "best_epoch": best_epoch, "n_params": n_params}
            val_auc[lr] = auc
            with OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"d={d:5d} lr={lr:.0e}  val_auc={auc:.4f}  best_epoch={best_epoch}", flush=True)
        best_lr = max(val_auc, key=val_auc.get)

        # stage 2: 5 seeds at the selected LR
        for seed in SEEDS:
            key = (d, best_lr, seed, "final")
            if key in done:
                continue
            data, split = get_split(seed)
            logits, auc, n_params, best_epoch = train_one(data, split, device, d, best_lr, seed=seed)
            row = evaluate_edge_split(logits, split, seed=seed)
            row.update(stage="final", model=f"scratch_d{d}", d_model=d, lr=best_lr,
                       seed=seed, val_auc=auc, best_epoch=best_epoch, n_params=n_params)
            with OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"d={d:5d} seed={seed}  auc={row['auc']:.4f}  ap={row['ap']:.4f}"
                  f"  best_epoch={best_epoch}  params={n_params/1e6:.1f}M", flush=True)


if __name__ == "__main__":
    main()
