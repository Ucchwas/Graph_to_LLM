"""Phase 8D runner -- bounded ogbg-molhiv calibration.

  python -m g2l.run_molhiv --stage baseline --model gin      (reproduce OGB's GIN in our pipeline)
  python -m g2l.run_molhiv --stage baseline --model gcn
  python -m g2l.run_molhiv --stage tune --d 128 --dropout 0.5    (LGM, validation only)
  python -m g2l.run_molhiv --stage final --d 64 --dropout 0.5    (test evaluation, authorised)
  python -m g2l.run_molhiv --stage final --model gin             (same, for a reproduced baseline)

Every stage runs the three fixed seeds in `configs/phase8d.yaml` and reports the mean.
`--stage tune` and `--stage baseline` never evaluate the test split. `--stage final` refuses to run
unless `--confirm-gate-passed` is given -- the operator explicitly authorising the test read. It was
originally an assertion that the validation gate had passed; the gate was withdrawn by the user on
2026-08-29, so it now records an authorisation and nothing more. Which architecture is built follows
`--model`, NOT the stage, so a reproduced baseline can be carried into the final stage too.
"""
import argparse
import json
import os
import pathlib
import subprocess

import numpy as np
import torch
import yaml
from ogb.utils.features import get_atom_feature_dims

from g2l.molclass import LGMClassifier, final_test, load_molhiv, train
from g2l.ogbnet import OGBGNN

ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase8d/rows"))


def commit_hash() -> str:
    if os.environ.get("G2L_COMMIT"):
        return os.environ["G2L_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build(args, cfg, seed):
    if args.model in ("gin", "gcn"):
        return OGBGNN(kind=args.model, d=cfg["ogb_d"], layers=cfg["ogb_layers"],
                      dropout=cfg["ogb_dropout"], seed=seed)
    return LGMClassifier(d=args.d, layers=cfg["layers"], heads=min(8, max(1, args.d // 32)),
                         k=cfg["k"], dropout=args.dropout, seed=seed,
                         atom_dims=get_atom_feature_dims())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase8d.yaml")
    ap.add_argument("--stage", required=True, choices=["baseline", "tune", "final"])
    ap.add_argument("--model", default=None, choices=["gin", "gcn", "lgm"],
                    help="architecture; defaults to gin for --stage baseline, lgm otherwise")
    ap.add_argument("--d", type=int, default=128, help="LGM width (the tuned knob)")
    ap.add_argument("--dropout", type=float, default=0.5, help="LGM dropout (the tuned knob)")
    ap.add_argument("--confirm-gate-passed", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.model is None:
        args.model = "gin" if args.stage == "baseline" else "lgm"
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    if args.epochs:
        cfg["max_epochs"] = args.epochs
    if args.stage == "final" and not args.confirm_gate_passed:
        raise SystemExit("--stage final requires --confirm-gate-passed: the operator authorising "
                         "a read of the test split for this frozen configuration.")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    mols, split = load_molhiv()
    name = args.model if args.model != "lgm" else f"lgm_d{args.d}_p{args.dropout}"
    print(f"stage={args.stage} arm={name} device={device} seeds={cfg['seeds']} "
          f"| train {len(split['train'])} valid {len(split['valid'])} test {len(split['test'])}", flush=True)

    vals, tests, params = [], [], None
    for seed in cfg["seeds"]:
        model = build(args, cfg, seed)
        params = sum(p.numel() for p in model.parameters())
        print(f"seed {seed}: {params} params", flush=True)
        s = train(model, mols, split, cfg, device, seed=seed)
        vals.append(s["val_rocauc"])
        print(f"seed {seed}: val {s['val_rocauc']:.4f} (best epoch {s['best_epoch']}/{s['epochs']}, "
              f"{s['wallclock_s']:.0f}s)", flush=True)
        if args.stage == "final":
            tests.append(final_test(model, mols, split, cfg, device))
            print(f"seed {seed}: TEST {tests[-1]:.4f}", flush=True)

    row = {"stage": args.stage, "arm": name, "commit": commit_hash(), "seeds": cfg["seeds"],
           "n_params": params, "val_rocauc": vals, "val_mean": float(np.mean(vals)),
           "val_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    if tests:
        row.update(test_rocauc=tests, test_mean=float(np.mean(tests)),
                   test_std=float(np.std(tests, ddof=1)) if len(tests) > 1 else 0.0)
    ROWS.mkdir(parents=True, exist_ok=True)
    (ROWS / f"{args.stage}_{name}.json").write_text(json.dumps(row))
    print(f"\n{name}: val {row['val_mean']:.4f} +- {row['val_std']:.4f} over {len(vals)} seeds"
          + (f" | TEST {row['test_mean']:.4f} +- {row['test_std']:.4f}" if tests else "")
          + f" | {params} params")
    print(f"wrote {ROWS / f'{args.stage}_{name}.json'}")


if __name__ == "__main__":
    main()
