"""Phase 8F runner -- the classification readout fix, and training-only self-supervision.

  python -m g2l.run_phase8f --stage pretrain
  python -m g2l.run_phase8f --stage tune --readout node     --init scratch
  python -m g2l.run_phase8f --stage tune --readout nodeedge --init pretrained

Two changes are measured as a 2x2 over three paired seeds, VALIDATION ONLY. The test split is not
reachable from this file at all.

  readout  node       mean(H_nodes) -> MLP            (the Phase-8C/8D behaviour: edge states are
                                                       computed by every layer and then discarded)
           nodeedge   [mean(H_nodes) || mean(H_edges)] -> MLP

  init     scratch    random initialisation
           pretrained body initialised from masked-attribute self-supervision on the official
                      MolHIV TRAINING molecules only -- no external data, no extra graphs

Both readouts share one pretrained body per seed, because the body is identical across them, and
both arms carry the mask-token input contract so scratch and pretrained are the same architecture.
The head is an MLP in every arm, so the readout comparison isolates the edge states rather than
head depth. The Phase-8D reference (node pooling, LINEAR head, k=3) stays as the separate
`lgm_d64_p0.5` row at val 0.7525 +- 0.0410.
"""
import argparse
import json
import os
import pathlib
import subprocess

import numpy as np
import torch
import yaml

from g2l.molclass import LGMClassifier, load_molhiv, train
from g2l.molpretrain import pretrain

ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase8f/rows"))
PRE = pathlib.Path(os.environ.get("G2L_PRETRAIN", "results/runs/phase8f"))


def commit_hash() -> str:
    if os.environ.get("G2L_COMMIT"):
        return os.environ["G2L_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build(cfg, seed, readout):
    # The body is constructed first inside LGMClassifier, so at a given seed both readouts start
    # from an identical body initialisation -- that is what makes the seeds paired.
    return LGMClassifier(d=cfg["d"], layers=cfg["layers"], heads=min(8, max(1, cfg["d"] // 32)),
                         k=cfg["k"], dropout=cfg["dropout"], seed=seed, readout=readout,
                         head="mlp", mask_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase8f.yaml")
    ap.add_argument("--stage", required=True, choices=["pretrain", "tune"])
    ap.add_argument("--readout", default="node", choices=["node", "nodeedge"])
    ap.add_argument("--init", default="scratch", choices=["scratch", "pretrained"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap the training split (smoke only)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    if args.epochs:
        cfg["max_epochs"] = cfg["pretrain_epochs"] = args.epochs
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    mols, split = load_molhiv()
    if args.limit:
        split = {k: v[:args.limit] for k, v in split.items()}
    name = f"{args.readout}_{args.init}" if args.stage == "tune" else "pretrain"
    print(f"stage={args.stage} arm={name} device={device} seeds={cfg['seeds']} d={cfg['d']} "
          f"dropout={cfg['dropout']} | train {len(split['train'])} valid {len(split['valid'])}",
          flush=True)

    vals, summaries, params = [], [], None
    for seed in cfg["seeds"]:
        model = build(cfg, seed, args.readout)
        params = sum(p.numel() for p in model.parameters())
        ckpt = PRE / f"pretrain_seed{seed}.pt"
        if args.stage == "pretrain":
            print(f"seed {seed}: {params} params, pretraining on {len(split['train'])} "
                  f"TRAINING molecules only", flush=True)
            s = pretrain(model, mols, split, cfg, device, seed=seed)
            PRE.mkdir(parents=True, exist_ok=True)
            torch.save(model.body.state_dict(), ckpt)
            print(f"seed {seed}: final SSL loss {s['final_loss']:.4f} ({s['wallclock_s']:.0f}s) "
                  f"-> {ckpt}", flush=True)
            summaries.append(s)
            continue
        if args.init == "pretrained":
            # strict=True: a silent key mismatch would quietly train a scratch model and label it
            # pretrained, which is exactly the confound this arm exists to avoid
            model.body.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
            print(f"seed {seed}: loaded {ckpt}", flush=True)
        print(f"seed {seed}: {params} params", flush=True)
        s = train(model, mols, split, cfg, device, seed=seed)
        vals.append(s["val_rocauc"])
        summaries.append(s)
        print(f"seed {seed}: val {s['val_rocauc']:.4f} (best epoch {s['best_epoch']}/{s['epochs']}, "
              f"{s['wallclock_s']:.0f}s)", flush=True)

    row = {"stage": args.stage, "arm": name, "readout": args.readout, "init": args.init,
           "commit": commit_hash(), "seeds": cfg["seeds"], "n_params": params,
           "d": cfg["d"], "dropout": cfg["dropout"], "per_seed": summaries}
    if vals:
        row.update(val_rocauc=vals, val_mean=float(np.mean(vals)),
                   val_std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        print(f"\n{name}: val {row['val_mean']:.4f} +- {row['val_std']:.4f} over {len(vals)} seeds "
              f"| {params} params")
    ROWS.mkdir(parents=True, exist_ok=True)
    (ROWS / f"{args.stage}_{name}.json").write_text(json.dumps(row))
    print(f"wrote {ROWS / f'{args.stage}_{name}.json'}")


if __name__ == "__main__":
    main()
