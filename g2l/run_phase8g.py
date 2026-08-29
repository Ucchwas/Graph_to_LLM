"""Phase 8G -- the leaderboard protocol: LGM, GIN and GCN under identical treatment.

  python -m g2l.run_phase8g --stage pretrain --model lgm
  python -m g2l.run_phase8g --stage select  --model lgm --head linear
  python -m g2l.run_phase8g --stage select  --model gin --init pretrained
  python -m g2l.run_phase8g --stage final   --model lgm --head mlp --seeds 0,1,2,3,4 --confirm-final

OGB's submission rules require mean and unbiased std over 10 random seeds, with test, validation
and parameter count reported. The protocol here:

  pretrain  masked-attribute self-supervision on the OFFICIAL TRAINING SPLIT ONLY, identical for
            all three families -- same loop, same masks, same schedule, same corpus
            (g2l/molpretrain.py). One checkpoint per (family, seed), all 10 seeds.
  select    the one open knob per family, chosen on VALIDATION over 3 seeds. LGM: linear vs MLP
            head (readout and init were already fixed on validation in Phase 8F). GIN/GCN:
            scratch vs pretrained. No stage-select run ever touches the test split.
  final     the frozen configuration, one seed per invocation slice, train -> restore the
            best-validation checkpoint -> read the test split ONCE per seed. Guarded by
            --confirm-final. 10 seeds total across array tasks.

Fairness rules enforced here rather than remembered: every family shares molclass.train (loader,
batcher, optimiser, epochs, patience, evaluator) and molpretrain._pretrain_loop; the same seed
lists pair every comparison; a scratch-winning GNN runs the EXACT OGB architecture in the final
(no unused mask rows in its parameter count), while a pretrained winner carries the widened
mask-token encoders its checkpoint needs.
"""
import argparse
import json
import os
import pathlib
import subprocess

import numpy as np
import torch
import yaml

from g2l.molclass import LGMClassifier, final_test, load_molhiv, train
from g2l.molpretrain import pretrain, pretrain_gnn
from g2l.ogbnet import OGBGNN

ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase8g/rows"))
PRE = pathlib.Path(os.environ.get("G2L_PRETRAIN", "results/runs/phase8g"))


def commit_hash() -> str:
    if os.environ.get("G2L_COMMIT"):
        return os.environ["G2L_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build(args, cfg, seed):
    if args.model == "lgm":
        return LGMClassifier(d=cfg["d"], layers=cfg["layers"], heads=min(8, max(1, cfg["d"] // 32)),
                             k=cfg["k"], dropout=cfg["dropout"], seed=seed, readout="nodeedge",
                             head=args.head, mask_tokens=True)
    # scratch-final GNNs use OGB's exact architecture -- no unused mask rows in the reported
    # parameter count; everywhere masking exists (pretrain, both select arms) the contract is on
    widened = not (args.stage == "final" and args.init == "scratch")
    return OGBGNN(kind=args.model, d=cfg["ogb_d"], layers=cfg["ogb_layers"],
                  dropout=cfg["ogb_dropout"], seed=seed, mask_tokens=widened)


def load_pretrained(model, args, ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    if args.model == "lgm":
        model.body.load_state_dict(sd, strict=True)
        return
    # everything except the classification head, which stays fresh -- and the key set is asserted
    # exactly, because strict=False would let a typo silently fine-tune a scratch model
    cur = model.state_dict()
    expect = {k for k in cur if not k.startswith("head.")}
    assert set(sd) == expect, f"checkpoint keys != body keys: {set(sd) ^ expect}"
    cur.update(sd)
    model.load_state_dict(cur, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase8g.yaml")
    ap.add_argument("--stage", required=True, choices=["pretrain", "select", "final"])
    ap.add_argument("--model", required=True, choices=["lgm", "gin", "gcn"])
    ap.add_argument("--head", default="mlp", choices=["linear", "mlp"], help="lgm only")
    ap.add_argument("--init", default="pretrained", choices=["scratch", "pretrained"])
    ap.add_argument("--seeds", default=None, help="comma list; defaults per stage from the config")
    ap.add_argument("--confirm-final", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap the splits (smoke only)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    if args.epochs:
        cfg["max_epochs"] = cfg["pretrain_epochs"] = args.epochs
    if args.stage == "final" and not args.confirm_final:
        raise SystemExit("--stage final reads the test split; pass --confirm-final to authorise.")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else
             cfg["final_seeds"] if args.stage in ("pretrain", "final") else cfg["select_seeds"])

    mols, split = load_molhiv()
    if args.limit:
        split = {k: v[:args.limit] for k, v in split.items()}
    name = {"pretrain": f"pre_{args.model}",
            "select": (f"lgm_h{args.head}" if args.model == "lgm" else f"{args.model}_{args.init}"),
            "final": f"final_{args.model}_s{seeds[0]}-{seeds[-1]}"}[args.stage]
    print(f"stage={args.stage} arm={name} device={device} seeds={seeds} "
          f"| train {len(split['train'])} valid {len(split['valid'])} test {len(split['test'])}",
          flush=True)

    vals, tests, summaries, params = [], [], [], None
    for seed in seeds:
        model = build(args, cfg, seed)
        params = sum(p.numel() for p in model.parameters())
        ckpt = PRE / f"{args.model}_seed{seed}.pt"
        if args.stage == "pretrain":
            print(f"seed {seed}: {params} params, SSL on {len(split['train'])} TRAINING molecules",
                  flush=True)
            fn = pretrain if args.model == "lgm" else pretrain_gnn
            s = fn(model, mols, split, cfg, device, seed=seed)
            PRE.mkdir(parents=True, exist_ok=True)
            sd = (model.body.state_dict() if args.model == "lgm" else
                  {k: v for k, v in model.state_dict().items() if not k.startswith("head.")})
            torch.save(sd, ckpt)
            print(f"seed {seed}: final SSL loss {s['final_loss']:.4f} ({s['wallclock_s']:.0f}s) "
                  f"-> {ckpt}", flush=True)
            summaries.append(s)
            continue
        if args.model == "lgm" or args.init == "pretrained":
            load_pretrained(model, args, ckpt)
            print(f"seed {seed}: loaded {ckpt}", flush=True)
        print(f"seed {seed}: {params} params", flush=True)
        s = train(model, mols, split, cfg, device, seed=seed)
        vals.append(s["val_rocauc"])
        summaries.append(s)
        line = f"seed {seed}: val {s['val_rocauc']:.4f} (best epoch {s['best_epoch']}/{s['epochs']})"
        if args.stage == "final":
            tests.append(final_test(model, mols, split, cfg, device))
            line += f" TEST {tests[-1]:.4f}"
        print(line, flush=True)

    row = {"stage": args.stage, "arm": name, "model": args.model, "head": args.head,
           "init": args.init, "commit": commit_hash(), "seeds": seeds, "n_params": params,
           "per_seed": summaries}
    for key, xs in (("val", vals), ("test", tests)):
        if xs:
            row[f"{key}_rocauc"] = xs
            row[f"{key}_mean"] = float(np.mean(xs))
            row[f"{key}_std"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    if vals:
        print(f"\n{name}: val {row['val_mean']:.4f} +- {row['val_std']:.4f}"
              + (f" | TEST {row['test_mean']:.4f} +- {row['test_std']:.4f}" if tests else "")
              + f" | {params} params over {len(vals)} seeds")
    ROWS.mkdir(parents=True, exist_ok=True)
    (ROWS / f"{name}.json").write_text(json.dumps(row))
    print(f"wrote {ROWS / f'{name}.json'}")


if __name__ == "__main__":
    main()
