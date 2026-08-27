"""Phase-4 runner (LGM v0: raw adjacency -> row tokenizer -> GNN backbone -> graph decoder).
`configs/phase4.yaml` -> a stage's run list -> one JSON row per run under $G2L_ROWS (default
results/phase4/rows). Contract as run_phase3 (skip on same key + commit, tmp -> rename); seed-0
runs keep their checkpoint and fp16 logits under $G2L_RUNS.

  python -m g2l.run_phase4 --stage smoke --list
  python -m g2l.run_phase4 --stage smoke                 (every run, sequentially: the laptop check)
  python -m g2l.run_phase4 --stage main --index 3 --chunk 50
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import traceback

import torch
import yaml

from baselines.common import get_split
from g2l.model import build_model
from g2l.train import train_run

CONFIG = pathlib.Path("configs/phase4.yaml")
ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase4/rows"))
RUNS = pathlib.Path(os.environ.get("G2L_RUNS", "results/runs/phase4"))
HASHED = ("mask_frac", "max_dist", "lr_norm", "weight_decay", "warmup", "max_epochs", "patience")
FROZEN = ("pretrained", "random")


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def expand(cfg: dict, stage: str) -> list[dict]:
    base = {"decoder": "d1", "input": "real", "frac": 1.0, "layers": 0, "width": cfg["d_model"], "bias": False,
            "lora": False, "kind": None, "dropout": 0.0, "frozen": False, "heads": None}
    st = cfg[stage]
    runs = []

    def add(seeds=None, **kw):
        for seed in (st.get("seeds", cfg["seeds"]) if seeds is None else seeds):
            runs.append({**base, **kw, "seed": seed})

    for arm in st.get("arms", ["gnn"]):
        for kind in st.get("kinds", []):
            for w in st["widths"]:
                for L in st["layers"]:
                    for lr in st["lr"]:
                        add(arm=arm, kind=kind, width=w, layers=L, lr=lr, dropout=st.get("dropout", 0.0))
    for ref in st.get("references", []):
        add(**ref)
    for e in st.get("extend", []):
        add(seeds=e.get("seeds"), **{k: v for k, v in e.items() if k != "seeds"})
    return runs


def key_of(run: dict) -> str:
    arm = run["arm"]
    if arm in ("gnn", "gnn_direct"):
        arm = f"{'gnnD' if arm == 'gnn_direct' else 'gnn'}-{run['kind']}{run['width']}L{run['layers']}" + (f"do{run['dropout']}" if run["dropout"] else "")
    elif arm in ("scratch", "gt"):
        arm += f"{run['width']}L{run['layers']}" + ("frz" if run["frozen"] else "") + (f"do{run['dropout']}" if run["dropout"] else "")
    lb = f"_lb{run['lr_bias']:.0e}" if run["bias"] else ""
    return f"{arm}_{'spd' if run['bias'] else 'nobias'}_{run['input']}_f{run['frac']}_lr{run['lr']:.0e}{lb}_s{run['seed']}"


def load_cfg(path=CONFIG) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text())


def run_one(cfg: dict, run: dict, device: str, force: bool = False):
    key = key_of(run)
    out = ROWS / f"{key}.json"
    commit = commit_hash()
    if out.exists() and not force and json.loads(out.read_text()).get("commit") == commit:
        print(f"skip {key} (row exists at this commit)")
        return
    print(f"run {key} @ {commit[:8]}", flush=True)
    data, split = get_split(run["seed"])
    body = None
    if run["arm"] in FROZEN:
        from g2l.run_phase2 import make_body
        body = make_body(cfg, run, device)
    T = body.T if body is not None else cfg["T"]
    model = build_model(run["arm"], data.num_nodes, run["width"], T, body=body, decoder=run["decoder"],
                        scratch_layers=run["layers"], seed=run["seed"], bias=run["bias"], max_dist=cfg["max_dist"],
                        heads=run["heads"], dropout=run["dropout"], frozen=run["frozen"], kind=run["kind"] or "gcn")
    tcfg = {**cfg, "lr": run["lr"], "lr_bias": run.get("lr_bias") if run["bias"] else None, "lr_lora": None}
    row, state, logits = train_run(model, data, split, tcfg, device, run["seed"],
                                   shuffle_input=run["input"] == "shuffled", train_frac=run["frac"])
    row.update(run, key=key, T=T, commit=commit,
               config_hash=hashlib.sha1(json.dumps({**run, **{k: cfg[k] for k in HASHED}}, sort_keys=True).encode()).hexdigest()[:12])
    if run["seed"] == 0:  # one reloadable checkpoint per cell (full state for the trained bodies)
        RUNS.mkdir(parents=True, exist_ok=True)
        torch.save(state if run["arm"] in FROZEN else model.state_dict(), RUNS / f"{key}.pt")
        torch.save(logits.half(), RUNS / f"{key}.logits.pt")
    ROWS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    os.replace(tmp, out)
    print(f"done {key}  auc={row['auc']:.4f} ap={row['ap']:.4f} ap_sparse={row['ap_sparse']:.4f} "
          f"best_epoch={row['best_epoch']} {row['sec_per_epoch']:.2f}s/epoch peak={row['peak_mem_gb']}GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", required=True)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = expand(cfg, args.stage)
    if args.list:
        print(len(runs))
        for r in runs:
            print(key_of(r))
        return
    idx = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    chunk = args.chunk or len(runs)
    failed = []
    for k in range(idx * chunk, min((idx + 1) * chunk, len(runs))):
        try:  # one bad configuration costs one row, not the rest of the chunk
            run_one(cfg, runs[k], device, force=args.force)
        except Exception:
            traceback.print_exc()
            failed.append(key_of(runs[k]))
        if device == "cuda":
            torch.cuda.empty_cache()
    if failed:
        raise SystemExit(f"{len(failed)} run(s) failed: {failed}")


if __name__ == "__main__":
    main()
