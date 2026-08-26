"""Phase-2 runner. `configs/phase2.yaml` expands into stages of run keys; one array task = one
key = one JSON row under $G2L_ROWS (default results/phase2/rows). A task is skipped only if a
row with the same key and the same commit exists (--force overrides).

  python -m g2l.run_phase2 --stage d1_grid --list
  python -m g2l.run_phase2 --stage d1_grid --index 7
  python -m g2l.run_phase2 --probe
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess

import torch
import yaml

from baselines.common import get_split
from g2l.llm import FrozenBody, build_body, load_config
from g2l.model import build_model
from g2l.train import first_batch_stats, masked_loss, train_run

CONFIG = pathlib.Path("configs/phase2.yaml")
ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase2/rows"))
RUNS = pathlib.Path(os.environ.get("G2L_RUNS", "results/runs/phase2"))
FROZEN = ("pretrained", "random")


def expand(cfg: dict, stage: str) -> list[dict]:
    base = {"encoder": "e1", "decoder": "d1", "loss": "masked", "input": "real", "width": cfg["d_model"]}
    runs = []

    def add(**kw):
        runs.append({**base, **kw})

    if stage == "d1_grid":
        for arm in ("pretrained", "none", "random"):
            for lr in cfg["lr_grid"]:
                for seed in cfg["seeds"]:
                    add(arm=arm, lr=lr, seed=seed)
    elif stage == "arm4_sweep":
        for width in cfg["scratch_widths"]:
            for lr in cfg["lr_grid"]:
                for seed in cfg["seeds"]:
                    add(arm="scratch", width=width, lr=lr, seed=seed)
    elif stage == "d3":
        for arm in ("pretrained", "none", "random"):
            for seed in cfg["seeds"]:
                add(arm=arm, decoder="d3", lr=cfg["selected_lr"][arm], seed=seed)
    elif stage == "controls":
        for arm in ("pretrained", "none"):
            for seed in cfg["seeds"]:
                add(arm=arm, input="shuffled", lr=cfg["selected_lr"][arm], seed=seed)
        for lr in cfg["lr_grid"]:
            for seed in cfg["seeds"]:
                add(arm="none", loss="recon", lr=lr, seed=seed)
                add(arm="none", encoder="e1gain1", lr=lr, seed=seed)
    elif stage == "extend":  # LR-grid edge rule: extra points listed in the config
        for e in cfg.get("extend", []):
            for seed in cfg["seeds"]:
                add(arm=e["arm"], width=e.get("width", cfg["d_model"]), lr=e["lr"], seed=seed)
    else:
        raise ValueError(stage)
    return runs


def key_of(run: dict) -> str:
    arm = f"scratch{run['width']}" if run["arm"] == "scratch" else run["arm"]
    return f"{run['encoder']}_{arm}_{run['decoder']}_{run['loss']}_{run['input']}_{run['lr']:.0e}_s{run['seed']}"


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def make_body(cfg, run, device):
    body_cfg = load_config(cfg["model_id"], cfg["revision"])
    llm = build_body(body_cfg, pretrained=run["arm"] == "pretrained", seed=run["seed"],
                     model_id=cfg["model_id"], revision=cfg["revision"])
    assert (llm.config.hidden_size, llm.config.num_hidden_layers, llm.config.num_attention_heads,
            llm.config.num_key_value_heads) == (2048, 16, 32, 8), "not the Llama-3.2-1B geometry"
    body = FrozenBody(llm).to(device)
    assert all(p.dtype == torch.bfloat16 for n, p in llm.named_parameters()
               if not (n.endswith("layernorm.weight") or n == "norm.weight")), "body is not bf16"
    return body


def run_one(cfg: dict, run: dict, device: str, force: bool = False):
    key = key_of(run)
    out = ROWS / f"{key}.json"
    commit = commit_hash()
    if out.exists() and not force:
        prev = json.loads(out.read_text())
        if prev.get("commit") == commit:
            print(f"skip {key} (row exists at this commit)")
            return
    print(f"run {key} @ {commit[:8]}", flush=True)
    data, split = get_split(run["seed"])
    body = make_body(cfg, run, device) if run["arm"] in FROZEN else None
    T = body.T if body is not None else cfg["T"]
    assert T, "T (pretrained embedding scale) must be measured and set in configs/phase2.yaml"
    model = build_model(run["arm"], data.num_nodes, run["width"], T, body=body, decoder=run["decoder"],
                        gain=1.0 if run["encoder"] == "e1gain1" else None,
                        scratch_layers=cfg["scratch_layers"], d3_hidden=cfg["d3_hidden"], seed=run["seed"])
    tcfg = {**cfg, "lr": run["lr"]}
    row, state, logits = train_run(model, data, split, tcfg, device, run["seed"], loss=run["loss"],
                                   shuffle_input=run["input"] == "shuffled")
    row.update(run, key=key, T=T, commit=commit, config_hash=hashlib.sha1(json.dumps({**run, **{k: cfg[k] for k in (
        "mask_frac", "lr_norm", "weight_decay", "warmup", "max_epochs", "patience", "revision")}}, sort_keys=True).encode()).hexdigest()[:12])
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
    """Day-one measurements before any array: gradient flow on a 512-node subgraph, full-Cora
    memory and s/epoch with and without checkpointing, and the random body's RMS profile."""
    from g2l.data import mask_matrix

    data, split = get_split(0)
    train = split[0]
    N = data.num_nodes
    out = {"commit": commit_hash()}

    from baselines.common import observed_dense
    A = observed_dense(train, N)
    top = A.sum(1).argsort(descending=True)[:512].sort().values
    A_sub = A[top][:, top].to(device)
    body = make_body(cfg, {"arm": "pretrained", "seed": 0}, device)
    out["T"] = body.T
    model = build_model("pretrained", 512, cfg["d_model"], body.T, body=body, seed=0).to(device)
    body.canary()
    pw = torch.tensor(float((512 * 511 / 2 - torch.triu(A_sub, 1).sum()) / torch.triu(A_sub, 1).sum()), device=device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses, gnorms = [], []
    for step in range(300):
        model.train()
        opt.zero_grad(set_to_none=True)
        A_obs, sup = mask_matrix(A_sub.cpu(), cfg["mask_frac"], seed=step)
        i, j = sup.to(device).nonzero(as_tuple=True)
        loss = masked_loss(model.pairs(A_obs.to(device), i, j), A_sub[i, j], pw)
        loss.backward()
        if step % 50 == 0 or step == 299:
            losses.append(round(loss.item(), 4))
            gnorms.append(round(float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.encoder.parameters()))), 6))
        opt.step()
    out["subgraph"] = {"loss": losses, "enc_grad_norm": gnorms}
    assert losses[-1] < 0.5 * losses[0], "loss did not halve on the 512-node subgraph"
    assert min(gnorms) > 0

    A_full = A.to(device)
    pos = train.pos_edge_label_index.to(device)
    model = build_model("pretrained", N, cfg["d_model"], body.T, body=body, seed=0).to(device)
    out["init_pretrained"] = first_batch_stats(model, A_full, pos)
    for ckpt in (False, True):
        body.checkpointing(ckpt)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        import time
        t0 = time.time()
        for step in range(3):
            model.train()
            opt.zero_grad(set_to_none=True)
            A_obs, sup = mask_matrix(A_full.cpu(), cfg["mask_frac"], seed=step)
            i, j = sup.to(device).nonzero(as_tuple=True)
            pwf = torch.tensor(float((N * (N - 1) / 2 - pos.shape[1] / 2) / (pos.shape[1] / 2)), device=device)
            masked_loss(model.pairs(A_obs.to(device), i, j), A_full[i, j], pwf).backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                model(A_full)
        torch.cuda.synchronize()
        out[f"full_ckpt_{ckpt}"] = {"sec_per_epoch": (time.time() - t0) / 3,
                                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                                    "n_checkpointed_layers": sum(getattr(l, "gradient_checkpointing", False) for l in body.llm.layers)}
    body.checkpointing(False)

    rbody = make_body(cfg, {"arm": "random", "seed": 0}, device)
    rmodel = build_model("random", N, cfg["d_model"], rbody.T, body=rbody, seed=0).to(device)
    out["init_random"] = first_batch_stats(rmodel, A_full, pos)
    ROWS.mkdir(parents=True, exist_ok=True)
    (ROWS / "probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=1, help="runs per task: task i runs indices i*chunk .. i*chunk+chunk-1")
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
    for k in range(idx * args.chunk, min((idx + 1) * args.chunk, len(runs))):
        run_one(cfg, runs[k], device, force=args.force)
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
