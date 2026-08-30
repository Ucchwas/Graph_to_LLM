"""Phase 10, MolHIV -- six arms under Phase 8H's protocol, with equal tuning budgets.

  python -m g2l.run_phase10 --stage pretrain --arm lgm_rrwp --d 128
  python -m g2l.run_phase10 --stage select   --arm gin_rwse --d 300
  python -m g2l.run_phase10 --stage final    --arm lgm      --d best --seeds 0,1,2,3,4 --confirm-final

Arms: {lgm, gin, gcn} x {plain, + its structural encoding}. The LGM's encoding is the pairwise RRWP
bias on the dense node<-node attention logits; the MPNNs' is that object's node-level diagonal
(RWSE) on their node input. K is fixed at 16 and never tuned. Every arm is otherwise the Phase-8H
configuration: official scaffold split, official Evaluator, all atom and bond attributes, 100-epoch
masked-attribute self-supervision on the TRAINING split only, warm-up + clipping, shared loops.

EQUAL BUDGETS. 8H gave the LGM one free knob (width) and the GNNs none. Here every arm tunes width
over three values on three selection seeds, on VALIDATION only; `--d best` then reads the width the
launcher chose. Stages `pretrain` and `select` cannot reach the test split; `final` reads it once
per seed, on a frozen configuration, behind --confirm-final.

Rows: one JSON per (stage, arm, width[, seed slice]).
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

ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase10/molhiv/rows"))
PRE = pathlib.Path(os.environ.get("G2L_PRETRAIN", "results/runs/phase10"))
ARMS = ("lgm", "lgm_rrwp", "gin", "gin_rwse", "gcn", "gcn_rwse")


def family(arm: str) -> str:
    """lgm_rrwp -> lgm, gin_rwse -> gin. The body; the suffix is the structural encoding."""
    return arm.split("_")[0]


def encoded(arm: str) -> bool:
    return "_" in arm


def widths(arm: str, cfg: dict) -> list[int]:
    return cfg["lgm_widths"] if family(arm) == "lgm" else cfg["ogb_widths"]


def commit_hash() -> str:
    if os.environ.get("G2L_COMMIT"):
        return os.environ["G2L_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build(arm: str, cfg: dict, d: int, seed: int):
    """Every arm carries the widened mask-token input contract, because every arm is pretrained --
    so the reported parameter count is the one the fine-tuned model actually has."""
    K = cfg["rrwp_k"] if encoded(arm) else 0
    if family(arm) == "lgm":
        return LGMClassifier(d=d, layers=cfg["layers"], heads=min(8, max(1, d // 32)),
                             k=cfg["k"], dropout=cfg["dropout"], seed=seed, readout="nodeedge",
                             head="linear", mask_tokens=True, rrwp_k=K)
    return OGBGNN(kind=family(arm), d=d, layers=cfg["ogb_layers"], dropout=cfg["ogb_dropout"],
                  seed=seed, mask_tokens=True, rwse_k=K)


def load_pretrained(model, arm: str, ckpt: pathlib.Path):
    sd = torch.load(ckpt, map_location="cpu")
    if family(arm) == "lgm":
        model.body.load_state_dict(sd, strict=True)
        return
    # everything except the classification head, which stays fresh -- key set asserted exactly,
    # because strict=False would let a typo silently fine-tune a scratch model
    cur = model.state_dict()
    expect = {k for k in cur if not k.startswith("head.")}
    assert set(sd) == expect, f"checkpoint keys != body keys: {set(sd) ^ expect}"
    cur.update(sd)
    model.load_state_dict(cur, strict=True)


def body_state(model, arm: str) -> dict:
    if family(arm) == "lgm":
        return model.body.state_dict()
    return {k: v for k, v in model.state_dict().items() if not k.startswith("head.")}


def winning_width(arm: str) -> int:
    f = ROWS / "winners.json"
    if not f.exists():
        raise SystemExit(f"--d best needs {f}; the launcher writes it after the select stage")
    w = json.loads(f.read_text())
    return int(w["width"][arm])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase10_molhiv.yaml")
    ap.add_argument("--stage", required=True, choices=["pretrain", "select", "final"])
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--d", required=True, help="width, or 'best' to read the launcher's choice")
    ap.add_argument("--seeds", default=None, help="comma list; defaults per stage from the config")
    ap.add_argument("--confirm-final", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="smoke: caps SSL and fine-tune epochs")
    ap.add_argument("--limit", type=int, default=None, help="cap the splits (smoke only)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    if args.epochs:
        cfg["max_epochs"] = cfg["pretrain_epochs"] = args.epochs
    if args.stage == "final" and not args.confirm_final:
        raise SystemExit("--stage final reads the test split; pass --confirm-final to authorise.")

    d = winning_width(args.arm) if args.d == "best" else int(args.d)
    assert d in widths(args.arm, cfg) or args.epochs, f"{d} is not in {args.arm}'s grid"
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else
             cfg["final_seeds"] if args.stage in ("pretrain", "final") else cfg["select_seeds"])

    mols, split = load_molhiv()
    if args.limit:
        split = {k: v[:args.limit] for k, v in split.items()}
    tag = f"{args.arm}_d{d}"
    name = {"pretrain": f"pre_{tag}", "select": f"sel_{tag}",
            "final": f"final_{tag}_s{seeds[0]}-{seeds[-1]}"}[args.stage]
    print(f"stage={args.stage} arm={args.arm} d={d} rrwp_k={cfg['rrwp_k'] if encoded(args.arm) else 0}"
          f" device={device} seeds={seeds} | train {len(split['train'])} valid {len(split['valid'])}"
          f" test {len(split['test'])} | commit {commit_hash()[:8]}", flush=True)

    vals, tests, summaries, params = [], [], [], None
    for seed in seeds:
        model = build(args.arm, cfg, d, seed)
        params = sum(p.numel() for p in model.parameters())
        ckpt = PRE / f"{tag}_seed{seed}.pt"
        if args.stage == "pretrain":
            print(f"seed {seed}: {params:,} params, SSL on {len(split['train'])} TRAINING molecules",
                  flush=True)
            fn = pretrain if family(args.arm) == "lgm" else pretrain_gnn
            s = fn(model, mols, split, cfg, device, seed=seed)
            PRE.mkdir(parents=True, exist_ok=True)
            torch.save(body_state(model, args.arm), ckpt)
            print(f"seed {seed}: final SSL loss {s['final_loss']:.4f} ({s['wallclock_s']:.0f}s)"
                  f" -> {ckpt}", flush=True)
            summaries.append(s)
            continue
        load_pretrained(model, args.arm, ckpt)
        print(f"seed {seed}: loaded {ckpt} | {params:,} params", flush=True)
        s = train(model, mols, split, cfg, device, seed=seed)
        vals.append(s["val_rocauc"])
        summaries.append(s)
        line = f"seed {seed}: val {s['val_rocauc']:.4f} (best epoch {s['best_epoch']}/{s['epochs']})"
        if args.stage == "final":
            tests.append(final_test(model, mols, split, cfg, device))
            line += f" TEST {tests[-1]:.4f}"
        print(line, flush=True)

    row = {"stage": args.stage, "arm": args.arm, "family": family(args.arm), "d": d,
           "rrwp_k": cfg["rrwp_k"] if encoded(args.arm) else 0, "name": name,
           "commit": commit_hash(), "seeds": seeds, "n_params": params,
           "pretrain_epochs": cfg["pretrain_epochs"], "warmup_steps": cfg.get("warmup_steps", 0),
           "clip": cfg.get("clip"), "smoke": bool(args.epochs or args.limit), "per_seed": summaries}
    for key, xs in (("val", vals), ("test", tests)):
        if xs:
            row[f"{key}_rocauc"] = xs
            row[f"{key}_mean"] = float(np.mean(xs))
            row[f"{key}_std"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    if vals:
        print(f"\n{name}: val {row['val_mean']:.4f} +- {row['val_std']:.4f}"
              + (f" | TEST {row['test_mean']:.4f} +- {row['test_std']:.4f}" if tests else "")
              + f" | {params:,} params over {len(vals)} seeds")
    ROWS.mkdir(parents=True, exist_ok=True)
    (ROWS / f"{name}.json").write_text(json.dumps(row))
    print(f"wrote {ROWS / f'{name}.json'}")


if __name__ == "__main__":
    main()
