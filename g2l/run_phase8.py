"""Phase 8 Gate 8.2 -- does one checkpoint transfer across graph sizes, and do edges carry anything?

  python -m g2l.run_phase8 --arm lgm                 (the model)
  python -m g2l.run_phase8 --arm noedge              (the control: identical parameters, no edge states)
  python -m g2l.run_phase8 --arm lgm --limit 4000 --cpu --epochs 6      (laptop smoke)

Training uses only graphs with N <= `train_max_n`. Evaluation reports two held-out splits: graphs at
sizes the checkpoint saw, and graphs LARGER than anything it was trained on. The second is the claim
Phases 4-7 could not even state -- `gnn_direct`'s GCNConv(N -> d) has one weight column per node
index, so it cannot be evaluated at a different N at all.

Rows land in results/phase8/rows/, one JSON per arm.
"""
import argparse
import json
import os
import pathlib
import subprocess

import torch
import yaml

from g2l.lgm import LGM
from g2l.multigraph import (BODIES, PRIORS, evaluate, load_mol, matched_width, prior_scores,
                            score, split, train_gate)

ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase8/rows"))
RUNS = pathlib.Path(os.environ.get("G2L_RUNS", "results/runs/phase8"))


def commit_hash() -> str:
    """$G2L_COMMIT wins: the Marlowe checkout stays pinned for a whole phase and Phase-8 code is
    shipped on top by scp, so `git rev-parse HEAD` there names the wrong commit for these rows."""
    if os.environ.get("G2L_COMMIT"):
        return os.environ["G2L_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase8.yaml")
    ap.add_argument("--arm", default="lgm", choices=["lgm", "noedge", "gcn", "edgegcn"])
    ap.add_argument("--limit", type=int, default=None, help="cap the corpus (smoke runs only)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    if args.epochs:
        cfg["max_epochs"] = args.epochs
    seed = args.seed if args.seed is not None else cfg["seed"]
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    mols = load_mol(cfg["dataset"], limit=args.limit)
    train_all, val_all, test_all = split(mols, seed=cfg["seed"])
    cap = cfg["train_max_n"]
    train = [m for m in train_all if m.n <= cap]
    val = [m for m in val_all if m.n <= cap]
    seen = [m for m in test_all if m.n <= cap]
    unseen = [m for m in test_all if m.n > cap]
    print(f"{cfg['dataset']}: {len(mols)} graphs | train {len(train)} (N <= {cap}) val {len(val)} | "
          f"test seen-size {len(seen)}, unseen-size {len(unseen)} "
          f"(N {min((m.n for m in unseen), default=0)}-{max((m.n for m in unseen), default=0)})", flush=True)

    ref = sum(p.numel() for p in LGM(d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"],
                                     k=cfg["k"], seed=seed).parameters())
    if args.arm in BODIES:
        # capacity-matched to the LGM: at equal d a convolutional body is far smaller, and an
        # unmatched baseline would lose on parameters rather than on architecture
        w = matched_width(ref, cfg["layers"], cfg["k"], kind=args.arm)
        model = BODIES[args.arm](w, layers=cfg["layers"], k=cfg["k"], dropout=cfg["dropout"], seed=seed)
        print(f"{args.arm} width {w} matched to the LGM's {ref} params", flush=True)
    else:
        model = LGM(d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"], k=cfg["k"],
                    dropout=cfg["dropout"], edges=(args.arm == "lgm"), seed=seed)
    print(f"arm={args.arm} device={device} params={sum(p.numel() for p in model.parameters())} "
          f"(lgm reference {ref})", flush=True)
    summary = train_gate(model, train, val, cfg, device, seed=seed)

    row = {"arm": args.arm, "seed": seed, "commit": commit_hash(), "device": device,
           "n_train": len(train), "train_max_n": cap, "limit": args.limit, **summary}
    for name, ms in (("seen_size", seen), ("unseen_size", unseen)):
        row[name] = evaluate(model, ms, cfg["budget"], cfg["mask_frac"], cfg["eval_seed"],
                             device, by_size=True)
        print(f"{name:12s} AUROC {row[name]['auc']:.4f} AP {row[name]['ap']:.4f} "
              f"({row[name]['n']} cells, {row[name]['pos']} positive)", flush=True)
        for b in row[name].get("by_size", []):
            print(f"    N {b['n_lo']:3d}-{b['n_hi']:3d}  AUROC {b['auc']:.4f}  AP {b['ap']:.4f}  n {b['n']}", flush=True)

    row["priors"] = {}
    for name, ms in (("seen_size", seen), ("unseen_size", unseen)):
        for p in PRIORS:
            s, y = prior_scores(ms, cfg["mask_frac"], cfg["eval_seed"], p)
            row["priors"][f"{name}/{p}"] = score(y, s)
    for k, v in row["priors"].items():
        print(f"prior {k:34s} AUROC {v['auc']:.4f} AP {v['ap']:.4f}", flush=True)

    ROWS.mkdir(parents=True, exist_ok=True)
    key = f"gate82_{args.arm}_seed{seed}" + ("_smoke" if args.limit else "")
    (ROWS / f"{key}.json").write_text(json.dumps(row))
    if not args.limit:
        RUNS.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), RUNS / f"{key}.pt")
    print(f"\nwrote {ROWS / f'{key}.json'}")


if __name__ == "__main__":
    main()
