"""Phase-3 runner: `configs/phase3.yaml` -> stages of run keys -> one JSON row per key under
$G2L_ROWS (default results/phase3/rows). Same contract as run_phase2 (skip on same key+commit,
tmp -> rename, checkpoints + fp16 logits for the frozen arms).

  python -m g2l.run_phase3 --stage main --list
  python -m g2l.run_phase3 --stage main --index 3 --chunk 15
  python -m g2l.run_phase3 --probe
"""
import argparse
import hashlib
import json
import os
import pathlib
import traceback

import torch
import yaml

from baselines.common import get_split, observed_dense
from g2l.data import mask_matrix
from g2l.model import build_model
from g2l.run_phase2 import FROZEN, commit_hash, make_body
from g2l.train import first_batch_stats, masked_loss, train_run

CONFIG = pathlib.Path("configs/phase3.yaml")
ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase3/rows"))
RUNS = pathlib.Path(os.environ.get("G2L_RUNS", "results/runs/phase3"))
HASHED = ("mask_frac", "max_dist", "lr_norm", "weight_decay", "warmup", "max_epochs", "patience", "revision")


def expand(cfg: dict, stage: str) -> list[dict]:
    base = {"decoder": "d1", "input": "real", "frac": 1.0, "layers": 0, "width": cfg["d_model"], "bias": True}
    runs = []

    def add(seeds=None, **kw):
        for seed in (cfg["seeds"] if seeds is None else seeds):
            runs.append({**base, **kw, "seed": seed})

    def scratch(layers, **kw):
        return {"arm": "scratch", "width": cfg["scratch_width"], "lr": cfg["enc_lr"]["scratch"], "layers": layers, **kw}

    nb, central = cfg["enc_lr_neighbourhood"], cfg["lr_bias_central"]
    if stage == "main":
        for lb in cfg["lr_bias_grid"]:
            add(arm="pretrained", lr=cfg["enc_lr"]["pretrained"], lr_bias=lb)
            add(arm="random", lr=cfg["enc_lr"]["random"], lr_bias=lb)
            for L in cfg["scratch_layers"]:
                add(**scratch(L), lr_bias=lb)
        for arm in ("pretrained", "random"):  # encoder-LR neighbourhood at the central bias LRs
            for lr in nb[arm]:
                for lb in central:
                    add(arm=arm, lr=lr, lr_bias=lb)
        for lr in nb["scratch"]:  # scratch L1 never had an LR search: neighbourhood with and without the bias
            for lb in central:
                add(**scratch(1, lr=lr), lr_bias=lb)
            add(**scratch(1, lr=lr), bias=False)
        for lb in central:  # shuffled-A: the bias sees the rewired graph too
            add(arm="pretrained", lr=cfg["enc_lr"]["pretrained"], lr_bias=lb, input="shuffled")
        # bias-off references at this commit, all seeds (the Phase-2 rows become a reproduction check)
        for arm in ("pretrained", "random", "none"):
            add(arm=arm, lr=cfg["enc_lr"][arm], bias=False)
        for L in cfg["scratch_layers"]:
            add(**scratch(L), bias=False)
    elif stage == "fraction":  # conditional: pre-registered data-fraction sweep at the selected LRs
        for f in cfg["fractions"]:
            for arm in ("pretrained", "random"):
                s = cfg["selected"][arm]
                add(arm=arm, lr=s["lr"], lr_bias=s["lr_bias"], frac=f)
            add(arm="none", lr=cfg["enc_lr"]["none"], bias=False, frac=f)
    elif stage == "extend":
        for e in cfg.get("extend", []):
            kw = {k: v for k, v in e.items() if k != "seeds"}
            if kw["arm"] == "scratch":
                kw = scratch(kw.pop("layers", 4), **kw)
            add(seeds=e.get("seeds"), **kw)
    else:
        raise ValueError(stage)
    return runs


def key_of(run: dict) -> str:
    arm = run["arm"] + (f"{run['width']}L{run['layers']}" if run["arm"] == "scratch" else "")
    lb = f"_lb{run['lr_bias']:.0e}" if run["bias"] else ""
    return f"{arm}_{'spd' if run['bias'] else 'nobias'}_{run['input']}_f{run['frac']}_lr{run['lr']:.0e}{lb}_s{run['seed']}"


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def run_one(cfg: dict, run: dict, device: str, force: bool = False):
    key = key_of(run)
    out = ROWS / f"{key}.json"
    commit = commit_hash()
    if out.exists() and not force and json.loads(out.read_text()).get("commit") == commit:
        print(f"skip {key} (row exists at this commit)")
        return
    print(f"run {key} @ {commit[:8]}", flush=True)
    assert run["arm"] != "scratch" or run["layers"] > 0, f"scratch run without layers: {key}"
    data, split = get_split(run["seed"])
    body = make_body(cfg, run, device) if run["arm"] in FROZEN else None
    T = body.T if body is not None else cfg["T"]
    model = build_model(run["arm"], data.num_nodes, run["width"], T, body=body, decoder=run["decoder"],
                        scratch_layers=run["layers"], seed=run["seed"], bias=run["bias"], max_dist=cfg["max_dist"])
    tcfg = {**cfg, "lr": run["lr"], "lr_bias": run.get("lr_bias") if run["bias"] else None}
    row, state, logits = train_run(model, data, split, tcfg, device, run["seed"],
                                   shuffle_input=run["input"] == "shuffled", train_frac=run["frac"])
    row.update(run, key=key, T=T, commit=commit,
               config_hash=hashlib.sha1(json.dumps({**run, **{k: cfg[k] for k in HASHED}}, sort_keys=True).encode()).hexdigest()[:12])
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save(state, RUNS / f"{key}.pt")
    if run["arm"] in FROZEN:
        torch.save(logits.half(), RUNS / f"{key}.logits.pt")
    ROWS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    os.replace(tmp, out)
    print(f"done {key}  auc={row['auc']:.4f} ap={row['ap']:.4f} ap_sparse={row['ap_sparse']:.4f} "
          f"best_epoch={row['best_epoch']} {row['sec_per_epoch']:.2f}s/epoch peak={row['peak_mem_gb']}GB", flush=True)


def probe(cfg: dict, device: str):
    """Before the array: the biased arm 1 on a 512-node subgraph (loss halves, table moves,
    gradients everywhere), then full Cora s/epoch and peak memory with the bias."""
    import time

    data, split = get_split(0)
    N = data.num_nodes
    A = observed_dense(split[0], N)
    top = A.sum(1).argsort(descending=True)[:512].sort().values
    A_sub = A[top][:, top].to(device)
    body = make_body(cfg, {"arm": "pretrained", "seed": 0}, device)
    out = {"commit": commit_hash(), "T": body.T}
    model = build_model("pretrained", 512, cfg["d_model"], body.T, body=body, seed=0, bias=True, max_dist=cfg["max_dist"]).to(device)
    body.canary()
    pw = torch.tensor(float((512 * 511 / 2 - torch.triu(A_sub, 1).sum()) / torch.triu(A_sub, 1).sum()), device=device)
    from g2l.train import make_optimizer
    opt = make_optimizer(model, cfg["enc_lr"]["pretrained"], cfg["lr_norm"], cfg["weight_decay"], lr_bias=3e-2)
    losses, tables = [], []
    for step in range(300):
        model.train()
        opt.zero_grad(set_to_none=True)
        A_obs, sup = mask_matrix(A_sub.cpu(), cfg["mask_frac"], seed=step)
        i, j = sup.to(device).nonzero(as_tuple=True)
        loss = masked_loss(model.pairs(A_obs.to(device), i, j), A_sub[i, j], pw)
        loss.backward()
        if step == 0:
            assert model.bias.bias_table.grad is not None and model.bias.bias_table.grad.abs().sum() > 0
        if step % 50 == 0 or step == 299:
            losses.append(round(loss.item(), 4))
            tables.append([round(v, 3) for v in model.bias.bias_table[0].tolist()])
        opt.step()
    out["subgraph"] = {"loss": losses, "head0_table": tables}
    assert losses[-1] < 0.5 * losses[0] and abs(max(tables[-1], key=abs)) > 0

    A_full = A.to(device)
    pos = A_full.nonzero().T
    model = build_model("pretrained", N, cfg["d_model"], body.T, body=body, seed=0, bias=True, max_dist=cfg["max_dist"]).to(device)
    out["init"] = first_batch_stats(model, A_full, pos)
    opt = make_optimizer(model, cfg["enc_lr"]["pretrained"], cfg["lr_norm"], cfg["weight_decay"], lr_bias=3e-2)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    pwf = torch.tensor(float((N * (N - 1) / 2 - pos.shape[1] / 2) / (pos.shape[1] / 2)), device=device)
    for step in range(3):
        model.train()
        opt.zero_grad(set_to_none=True)
        A_obs, sup = mask_matrix(A_full.cpu(), cfg["mask_frac"], seed=step)
        i, j = sup.to(device).nonzero(as_tuple=True)
        masked_loss(model.pairs(A_obs.to(device), i, j), A_full[i, j], pwf).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            model(A_full)
    torch.cuda.synchronize()
    out["full"] = {"sec_per_epoch": (time.time() - t0) / 3, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    ROWS.mkdir(parents=True, exist_ok=True)
    (ROWS / "probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.probe:
        probe(cfg, device)
        return
    runs = expand(cfg, args.stage)
    if args.list:
        print(len(runs))
        for r in runs:
            print(key_of(r))
        return
    idx = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    failed = []
    for k in range(idx * args.chunk, min((idx + 1) * args.chunk, len(runs))):
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
