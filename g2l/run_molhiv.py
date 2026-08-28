"""Phase 8C runner -- ogbg-molhiv graph classification against the official OGB benchmark.

  python -m g2l.run_molhiv --arm atom       (nine atom features + three bond features)
  python -m g2l.run_molhiv --arm noatom     (ablation: atom features off, everything else identical)

Official scaffold split, official Evaluator, one seed, selection on validation ROC-AUC, test scored
once. Only the `atom` arm is comparable to published numbers; `noatom` is an ablation.
"""
import argparse
import json
import os
import pathlib
import subprocess

import torch
import yaml
from ogb.utils.features import get_atom_feature_dims

from g2l.molclass import GraphClassifier, load_molhiv, train

ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase8/rows"))
RUNS = pathlib.Path(os.environ.get("G2L_RUNS", "results/runs/phase8"))


def commit_hash() -> str:
    if os.environ.get("G2L_COMMIT"):
        return os.environ["G2L_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase8c.yaml")
    ap.add_argument("--arm", default="atom", choices=["atom", "noatom"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap the training split (smoke only)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    if args.epochs:
        cfg["max_epochs"] = args.epochs
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    mols, split = load_molhiv()
    if args.limit:
        split = {k: v[:args.limit] for k, v in split.items()}
    print(f"ogbg-molhiv: {len(mols)} graphs | official scaffold split "
          f"train {len(split['train'])} valid {len(split['valid'])} test {len(split['test'])}", flush=True)

    model = GraphClassifier(d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"], k=cfg["k"],
                            dropout=cfg["dropout"], seed=cfg["seed"],
                            atom_dims=get_atom_feature_dims() if args.arm == "atom" else None)
    print(f"arm={args.arm} device={device} params={sum(p.numel() for p in model.parameters())}", flush=True)
    summary = train(model, mols, split, cfg, device, seed=cfg["seed"])

    row = {"task": "ogbg-molhiv-classification", "arm": args.arm, "seed": cfg["seed"],
           "commit": commit_hash(), "device": device, "limit": args.limit,
           "comparable_to_leaderboard": args.arm == "atom", **summary}
    key = f"molhiv_{args.arm}_seed{cfg['seed']}" + ("_smoke" if args.limit else "")
    ROWS.mkdir(parents=True, exist_ok=True)
    (ROWS / f"{key}.json").write_text(json.dumps(row))
    if not args.limit:
        RUNS.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), RUNS / f"{key}.pt")
    print(f"\n{args.arm}: val ROC-AUC {summary['val_rocauc']:.4f} | test ROC-AUC "
          f"{summary['test_rocauc']:.4f} | {summary['n_params']} params | "
          f"best epoch {summary['best_epoch']}/{summary['epochs']} | {summary['wallclock_s']:.0f}s")
    print(f"wrote {ROWS / f'{key}.json'}")


if __name__ == "__main__":
    main()
