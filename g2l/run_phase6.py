"""Phase-6 runner: aligned multi-layer completion on Decagon (configs/phase6.yaml). One JSON row
per (layer, arm) under $G2L_ROWS (default results/phase6/rows); skip on same key + commit.

  shared        one model on the train layers -> a row per layer (+ reduced-input columns on the
                test layers) and summary_shared.json; checkpoint under $G2L_RUNS
  independent   one model per layer, chunked: --index i --chunk c
  priors        identity / common_neighbors / pair_frequency / knn on every layer
  gae           featureless GAE on the layers of cfg["gae_groups"]

  python -m g2l.run_phase6 --stage shared --epochs 2 --layers 12 --cpu        (laptop smoke)
  python -m g2l.run_phase6 --stage independent --index 0 --chunk 121
"""
import argparse
import json
import os
import pathlib
import subprocess
import traceback

import torch
import yaml
from torch_geometric.data import Data

from baselines import gae
from g2l.decagon import load_decagon
from g2l.metrics import evaluate_edge_split
from g2l.model import build_model
from g2l.priors import Priors
from g2l.shared import layer_rows, train_shared
from g2l.train import train_run

CONFIG = pathlib.Path("configs/phase6.yaml")
ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase6/rows"))
RUNS = pathlib.Path(os.environ.get("G2L_RUNS", "results/runs/phase6"))


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def key_of(layer: str, arm: str, seed: int) -> str:
    return f"decagon_{layer}_{arm}_s{seed}"


def write_row(row: dict, key: str):
    ROWS.mkdir(parents=True, exist_ok=True)
    tmp = (ROWS / f"{key}.json").with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    os.replace(tmp, ROWS / f"{key}.json")


def exists(key: str, commit: str, force: bool) -> bool:
    f = ROWS / f"{key}.json"
    return f.exists() and not force and json.loads(f.read_text()).get("commit") == commit


def load(cfg: dict, cap: int | None):
    N, layers, hidden, info, names = load_decagon(cfg["min_pairs"], cfg["split_seed"], cfg["tau"], cfg["hidden_frac"])
    if cap:  # smoke: a few layers of every group, in name order
        want = {"train": cap - 2 * (cap // 4), "val": cap // 4, "test": cap // 4}
        keep, seen = [], {"train": 0, "val": 0, "test": 0}
        for L in layers:
            if seen[L.group] < want[L.group]:
                keep.append(L)
                seen[L.group] += 1
        layers = keep
    return N, layers, hidden, info, names


def new_model(cfg: dict, N: int):
    return build_model("gnn_direct", N, cfg["width"], 1.0, scratch_layers=cfg["layers"], seed=cfg["seed"], kind=cfg["kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", required=True, choices=["shared", "independent", "priors", "gae"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="cap the epochs (smoke only)")
    ap.add_argument("--layers", type=int, default=None, help="cap the number of layers (smoke only)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    commit, seed = commit_hash(), cfg["seed"]
    N, layers, hidden, info, names = load(cfg, args.layers)
    print(f"decagon: N={N} {info} stage={args.stage} @ {commit[:8]} on {device}", flush=True)
    base = {"dataset": "decagon", "seed": seed, "commit": commit, "n_hidden_pairs": info["n_hidden_pairs"]}
    tcfg = {k: cfg[k] for k in ("mask_frac", "lr", "lr_norm", "weight_decay", "warmup")}

    if args.stage == "shared":
        if all(exists(key_of(L.name, "shared", seed), commit, args.force) for L in layers):
            print("skip shared (rows exist at this commit)")
            return
        scfg = {**tcfg, **cfg["shared"], **({"max_epochs": args.epochs} if args.epochs else {})}
        model = new_model(cfg, N)
        summary = train_shared(model, layers, hidden, N, scfg, device, seed)
        print(f"shared trained: {summary}", flush=True)
        if not args.layers:
            RUNS.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), RUNS / f"decagon_shared_s{seed}.pt")
        for row in layer_rows(model, layers, hidden, N, device, fracs=cfg["shared"]["fracs"], seed=seed):
            row.update(base, arm="shared", name=names[row["layer"]], key=key_of(row["layer"], "shared", seed))
            write_row(row, row["key"])
        ROWS.mkdir(parents=True, exist_ok=True)
        (ROWS / f"summary_shared_s{seed}.json").write_text(json.dumps({**summary, **base, "info": info}))
        return

    if args.stage == "independent":
        chunk = args.chunk or len(layers)
        idx = args.index if args.index is not None else 0
        icfg = {**tcfg, **cfg["independent"], **({"max_epochs": args.epochs} if args.epochs else {}), "lr_bias": None, "lr_lora": None}
        failed = []
        for L in layers[idx * chunk:(idx + 1) * chunk]:
            key = key_of(L.name, "independent", seed)
            if exists(key, commit, args.force):
                continue
            try:
                tr, va, te = L.split(N, hidden)
                data = Data(edge_index=tr.edge_index, num_nodes=N)
                row, _, _ = train_run(new_model(cfg, N), data, (tr, va, te), icfg, device, seed,
                                      exclude=L.exclude(N, hidden), log=lambda *a: None)
                row.update(base, arm="independent", layer=L.name, name=names[L.name], group=L.group, n_pairs=L.n_pairs,
                           n_train=int(L.train.size(1)), n_test=int(L.test_pos.size(1)), key=key)
                write_row(row, key)
                print(f"done {key} ({L.group}) auc={row['auc']:.4f} ap={row['ap']:.4f} hidden={row.get('auc_hidden', float('nan')):.4f} "
                      f"best={row['best_epoch']}/{row['epochs']} [{row['wallclock_s']}s]", flush=True)
            except Exception:
                traceback.print_exc()
                failed.append(key)
            if device == "cuda":
                torch.cuda.empty_cache()
        if failed:
            raise SystemExit(f"{len(failed)} run(s) failed: {failed}")
        return

    if args.stage == "priors":
        P = Priors(layers, N)
        for L in layers:
            split = L.split(N, hidden)
            inp = torch.cat([L.train, L.val_pos], 1)
            for name in cfg["priors"]:
                key = key_of(L.name, name, seed)
                if exists(key, commit, args.force):
                    continue
                row = evaluate_edge_split(P.score(name, L, inp), split, seed=seed)
                row.update(base, arm=name, layer=L.name, name=names[L.name], group=L.group, n_pairs=L.n_pairs,
                           n_train=int(L.train.size(1)), n_test=int(L.test_pos.size(1)), key=key)
                write_row(row, key)
        print(f"priors done for {len(layers)} layers", flush=True)
        return

    if args.stage == "gae":
        fn = gae.make_score_fn("gae", featureless=True, **({"max_epochs": args.epochs} if args.epochs else {}))
        for L in layers:
            if L.group not in cfg["gae_groups"]:
                continue
            key = key_of(L.name, "gae0", seed)
            if exists(key, commit, args.force):
                continue
            tr, va, te = L.split(N, hidden)
            torch.manual_seed(seed)
            logits = fn(Data(edge_index=tr.edge_index, num_nodes=N), (tr, va, te), device)
            row = evaluate_edge_split(logits, (tr, va, te), seed=seed)
            row.update(base, arm="gae0", layer=L.name, name=names[L.name], group=L.group, n_pairs=L.n_pairs,
                       n_train=int(L.train.size(1)), n_test=int(L.test_pos.size(1)), key=key)
            write_row(row, key)
            print(f"done {key} auc={row['auc']:.4f} ap={row['ap']:.4f}", flush=True)


if __name__ == "__main__":
    main()
