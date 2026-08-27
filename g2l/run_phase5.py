"""Phase-5 runner: dataset feasibility and smoke test. configs/phase5.yaml -> per-dataset run list
(the direct GCN and the baselines) -> one JSON row per run under $G2L_ROWS (default
results/phase5/rows). Skip on same key + commit; the seed-0 direct run keeps its checkpoint and
fp16 logits under $G2L_RUNS.

  python -m g2l.run_phase5 --list
  python -m g2l.run_phase5 --index 1                                  (every run of dataset #1: one array task)
  python -m g2l.run_phase5 --dataset cora --only gnn_direct --epochs 3 --cpu   (laptop CPU sanity check)
"""
import argparse
import json
import os
import pathlib
import subprocess
import traceback

import torch
import yaml

from baselines import gae, heuristics, maskgae
from baselines.common import get_split, run_baseline
from g2l.model import build_model
from g2l.train import train_run

CONFIG = pathlib.Path("configs/phase5.yaml")
ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase5/rows"))
RUNS = pathlib.Path(os.environ.get("G2L_RUNS", "results/runs/phase5"))
NO_FEATURES = {"ddi", "ohmnet"}


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def baseline_fn(name: str, epochs: int | None = None):
    if name in heuristics.CONTROLS or name in heuristics.HEURISTICS:
        return heuristics.make_score_fn(name)
    if name in ("gae0", "vgae0"):
        return gae.make_score_fn(name[:-1], featureless=True, **({"max_epochs": epochs} if epochs else {}))
    if name in ("maskgae0", "maskgae_x"):
        return maskgae.make_score_fn(featureless=name.endswith("0"), **({"epochs": epochs, "eval_period": 1} if epochs else {}))
    raise ValueError(name)


def expand(cfg: dict, dataset: str) -> list[dict]:
    runs = [{"dataset": dataset, "model": "gnn_direct", **cfg["direct"], "seed": s} for s in cfg["seeds"]]
    for b in cfg["baselines"]:
        if b.endswith("_x") and dataset in NO_FEATURES:
            continue
        runs += [{"dataset": dataset, "model": b, "seed": s} for s in cfg["seeds"]]
    return runs


def key_of(run: dict) -> str:
    m = run["model"]
    if m == "gnn_direct":
        m = f"gnnD-{run['kind']}{run['width']}L{run['layers']}_lr{run['lr']:.0e}"
    return f"{run['dataset']}_{m}_s{run['seed']}"


def run_one(cfg: dict, run: dict, device: str, force: bool = False, epochs: int | None = None):
    key, commit = key_of(run), commit_hash()
    out = ROWS / f"{key}.json"
    if out.exists() and not force and json.loads(out.read_text()).get("commit") == commit:
        print(f"skip {key} (row exists at this commit)")
        return
    print(f"run {key} @ {commit[:8]}", flush=True)
    if run["model"] == "gnn_direct":
        data, split = get_split(run["seed"], run["dataset"])
        model = build_model("gnn_direct", data.num_nodes, run["width"], 1.0, scratch_layers=run["layers"],
                            seed=run["seed"], kind=run["kind"])
        tcfg = {**cfg, "lr": run["lr"], "lr_bias": None, "lr_lora": None, **({"max_epochs": epochs} if epochs else {})}
        row, _, logits = train_run(model, data, split, tcfg, device, run["seed"])
        row["n_nodes"] = data.num_nodes
        if run["seed"] == 0 and not epochs:
            RUNS.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), RUNS / f"{key}.pt")
            torch.save(logits.half(), RUNS / f"{key}.logits.pt")
    else:
        row = run_baseline(run["model"], run["seed"], baseline_fn(run["model"], epochs), run["dataset"])
    row.update(run, key=key, commit=commit)
    ROWS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    os.replace(tmp, out)
    print(f"done {key}  auc={row['auc']:.4f} ap={row['ap']:.4f} ap_sparse={row['ap_sparse']:.4f}"
          + (f" hits20={row['hits20']:.4f}" if "hits20" in row else "") + f"  [{row['wallclock_s']}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--index", type=int, default=None, help="dataset index in the config (one array task per dataset)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--only", default=None, help="run a single model name")
    ap.add_argument("--epochs", type=int, default=None, help="cap the epochs (sanity checks only; no checkpoint)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="laptop sanity checks: never touch the GPU")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if args.dataset:
        datasets = [args.dataset]
    elif args.index is not None or not args.list:
        idx = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        datasets = [cfg["datasets"][idx]]
    else:
        datasets = cfg["datasets"]
    runs = [r for d in datasets for r in expand(cfg, d) if not args.only or r["model"] == args.only]
    if args.list:
        print(len(runs))
        for r in runs:
            print(key_of(r))
        return
    failed = []
    for r in runs:
        try:
            run_one(cfg, r, device, force=args.force, epochs=args.epochs)
        except Exception:
            traceback.print_exc()
            failed.append(key_of(r))
        if device == "cuda":
            torch.cuda.empty_cache()
    if failed:
        raise SystemExit(f"{len(failed)} run(s) failed: {failed}")


if __name__ == "__main__":
    main()
