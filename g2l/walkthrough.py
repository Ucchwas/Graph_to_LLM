"""Per-phase gate walkthrough: `python -m g2l.walkthrough --phase N`.

Recomputes the phase's headline numbers live so a gate never means
"trust the logs". Phase 0: the metric protocol on a hand-checkable graph.
"""
import argparse

import torch

from g2l.bias import spd_matrix
from g2l.data import edge_split, load_cora, mask_matrix
from g2l.metrics import evaluate


def phase0():
    print("=" * 60)
    print("PHASE 0 WALKTHROUGH -- the evaluation protocol, by hand")
    print("=" * 60)

    # A 6-node path graph: 0-1-2-3-4-5. Mask decided by seed; scorer is 'distance-2
    # neighbours are likely edges' -- imperfect on purpose so no column is 1.0.
    A = torch.zeros(6, 6)
    for i in range(5):
        A[i, i + 1] = A[i + 1, i] = 1.0
    A_obs, sup = mask_matrix(A, frac=0.4, seed=4)
    hidden = [(int(i), int(j)) for i, j in zip(*torch.where(sup))]
    print(f"\npath graph 0-1-2-3-4-5; hidden cells (upper-tri): {hidden}")
    print(f"of which true edges: {[(i, j) for i, j in hidden if A[i, j] > 0]}")

    d = spd_matrix(A_obs, max_dist=8)
    logits = -d.float()  # score = negative observed-graph distance
    m = evaluate(logits, A, sup, seed=0)
    print("\nscorer: -shortest_path_distance on the OBSERVED graph")
    for k in ("auroc", "ap_balanced", "ap_sparse", "base_rate", "lift"):
        print(f"  {k:16s} {m[k]:.4f}")
    print(f"  scored cells     {m['n_scored']} ({m['n_pos']} positive)")
    print("\nCheck by hand: rank the hidden cells by observed distance; AUROC is the")
    print("probability a true edge outranks a non-edge; AP@sparse's floor is base_rate.")

    print("\n-- identity canary (leak check) --")
    g = torch.Generator().manual_seed(0)
    up = torch.triu((torch.rand(200, 200, generator=g) < 0.05).float(), diagonal=1)
    R = up + up.T
    R_obs, rsup = mask_matrix(R, frac=0.15, seed=0)
    mi = evaluate(R_obs * 100, R, rsup, seed=0)
    print(f"identity model: auroc={mi['auroc']:.3f} lift={mi['lift']:.3f}  (must be ~0.5 / ~1.0)")

    try:
        data = load_cora()
        print("\n-- Cora (corrected facts) --")
        print(f"nodes={data.num_nodes}  edge_index={list(data.edge_index.shape)}"
              f"  undirected_edges={data.edge_index.shape[1] // 2}  features={data.x.shape[1]}")
        tr, va, te = edge_split(data, seed=0)
        print(f"split: train={tr.pos_edge_label_index.shape[1]}"
              f" val={va.pos_edge_label_index.shape[1]}"
              f" test={te.pos_edge_label_index.shape[1]} (+{te.neg_edge_label_index.shape[1]} neg)")
        n_cells = data.num_nodes * (data.num_nodes - 1) // 2
        pos_rate = (data.edge_index.shape[1] // 2) / n_cells
        print(f"upper-tri positive rate: {pos_rate:.5f}  -> pos_weight ~ {(1 - pos_rate) / pos_rate:.1f}")
    except Exception as e:
        print(f"\n(Cora not downloaded yet: {e})")


def phase1():
    print("=" * 60)
    print("PHASE 1 WALKTHROUGH -- the baseline table, recomputed live")
    print("=" * 60)
    import json
    import pathlib

    from baselines.aggregate import table
    from baselines.common import run_baseline
    from baselines.heuristics import make_score_fn

    print()
    print("Live rerun of one deterministic row (common_neighbors, seed 0) vs the stored Marlowe row:")
    live = run_baseline("common_neighbors", 0, make_score_fn("common_neighbors"))
    stored = next(r for r in map(json.loads, pathlib.Path("results/phase1/table.jsonl").open())
                  if r["model"] == "common_neighbors" and r["seed"] == 0)
    for k in ("auc", "ap", "ap_sparse", "lift"):
        print(f"  {k:10s} live={live[k]:.5f}  marlowe={stored[k]:.5f}")
    print("  -> all four must agree to float noise: the split (positives AND sampled negatives)")
    print("     is fully seeded, so the laptop reproduces the cluster row exactly.")

    print()
    print("Stored table (mean +- std over seeds):")
    print()
    print(table())
    print()
    print("Reading guide: AP@1:1 is the Kipf-comparable column (GAE paper: 92.0;")
    print("PyG reproduction: 91.2+-1.0). AP(sparse) is the honest 0.144%-prevalence")
    print("number -- note it is ~50x smaller at identical AUC. lift = AP/base-rate.")

def phase2(key: str | None = None):
    """Gate assertions live, then arm 1 seed 0 recomputed on the laptop from its Marlowe
    checkpoint and compared with the logits saved by the cluster job."""
    import json
    import os
    import pathlib

    import pytest
    import yaml

    print("=" * 60)
    print("PHASE 2 WALKTHROUGH -- injection route, checkpoint reload, delta table")
    print("=" * 60)
    print("\n-- gate assertions, live (tiny-model mechanics + real-weight parity/canary) --")
    assert pytest.main(["-q", "tests/test_llm.py", "tests/test_llm_real.py"]) == 0

    from baselines.common import get_split, observed_dense
    from g2l.aggregate import main as aggregate
    from g2l.llm import FrozenBody, build_body, load_config
    from g2l.metrics import evaluate_edge_split
    from g2l.model import build_model

    cfg = yaml.safe_load(open("configs/phase2.yaml"))
    key = key or f"e1_pretrained_d1_masked_real_{cfg['selected_lr']['pretrained']:.0e}_s0"
    runs = pathlib.Path(os.environ.get("G2L_RUNS", os.path.join(os.environ.get("LOCALAPPDATA", "results/runs"), "g2l", "phase2")))
    stored = json.load(open(f"results/phase2/rows/{key}.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n-- {key}: rebuild on the laptop from {runs / key}.pt --")
    data, split = get_split(stored["seed"])
    body = FrozenBody(build_body(load_config(cfg["model_id"], cfg["revision"]), pretrained=True,
                                 model_id=cfg["model_id"], revision=cfg["revision"]))
    model = build_model("pretrained", data.num_nodes, cfg["d_model"], body.T, body=body, seed=stored["seed"]).to(device)
    res = model.load_state_dict(torch.load(runs / f"{key}.pt", map_location=device), strict=False)
    assert not res.unexpected_keys
    model.eval()
    with torch.no_grad():
        logits = model(observed_dense(split[2], data.num_nodes).to(device)).cpu()
    saved = torch.load(runs / f"{key}.logits.pt").float()
    live = evaluate_edge_split(logits, split, seed=stored["seed"])
    delta = (logits - saved).abs().max().item()
    print(f"  max |logit delta| vs Marlowe: {delta:.4f}  (saved logits are fp16; measured 0.0078 on 2026-08-26, locked at 0.02)")
    assert delta < 0.02, f"laptop recomputation drifted from the cluster logits: {delta:.4f}"
    for k in ("auc", "ap", "ap_sparse"):
        print(f"  {k:10s} live={live[k]:.5f}  marlowe={stored[k]:.5f}")

    print("\n-- table and paired deltas (results/phase2/rows) --\n")
    aggregate()


def phase3(key: str | None = None, phase: str = "3"):
    """Bias gate assertions live, then pretrained + SPD seed 0 rebuilt on the laptop from its
    Marlowe checkpoint: logits vs the saved ones, and the learned per-head distance table.
    Phase 3B: the same with the LoRA adapters (`selected_lora`) on the body."""
    import json
    import os
    import pathlib

    import numpy as np
    import pytest
    import yaml

    print("=" * 60)
    print(f"PHASE {phase.upper()} WALKTHROUGH -- attention bias{' + LoRA' if phase == '3b' else ''}: inertness, gradient, leak, reload, learned table")
    print("=" * 60)
    print("\n-- gate assertions, live --")
    tests = ["tests/test_bias.py", "tests/test_llm.py", "tests/test_llm_real.py"] + (["tests/test_lora.py"] if phase == "3b" else [])
    assert pytest.main(["-q", *tests]) == 0

    from baselines.common import get_split, observed_dense
    from g2l.aggregate3 import main as aggregate
    from g2l.llm import FrozenBody, build_body, load_config
    from g2l.metrics import evaluate_edge_split
    from g2l.model import build_model

    cfg = yaml.safe_load(open(f"configs/phase{phase}.yaml"))
    sel = cfg["selected"]["pretrained"]
    lora = f"_lora{cfg['selected_lora']['pretrained']:.0e}" if phase == "3b" else ""
    key = key or f"pretrained_spd_real_f1.0_lr{sel['lr']:.0e}_lb{sel['lr_bias']:.0e}{lora}_s0"
    runs = pathlib.Path(os.environ.get("G2L_RUNS", os.path.join(os.environ.get("LOCALAPPDATA", "results/runs"), "g2l", f"phase{phase}")))
    stored = json.load(open(f"results/phase{phase}/rows/{key}.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n-- {key}: rebuild on the laptop from {runs / key}.pt --")
    data, split = get_split(stored["seed"])
    body = FrozenBody(build_body(load_config(cfg["model_id"], cfg["revision"]), pretrained=True,
                                 model_id=cfg["model_id"], revision=cfg["revision"]))
    if stored.get("lora"):
        body.add_lora(**cfg["lora"])
    model = build_model("pretrained", data.num_nodes, cfg["d_model"], body.T, body=body, seed=stored["seed"],
                        bias=True, max_dist=cfg["max_dist"]).to(device)
    res = model.load_state_dict(torch.load(runs / f"{key}.pt", map_location=device), strict=False)
    assert not res.unexpected_keys
    model.eval()
    with torch.no_grad():
        logits = model(observed_dense(split[2], data.num_nodes).to(device)).cpu()
    saved = torch.load(runs / f"{key}.logits.pt").float()
    live = evaluate_edge_split(logits, split, seed=stored["seed"])
    delta = (logits - saved).abs().max().item()
    print(f"  max |logit delta| vs Marlowe: {delta:.4f}  (saved logits are fp16; locked at 0.02)")
    assert delta < 0.02
    for k in ("auc", "ap", "ap_sparse"):
        print(f"  {k:10s} live={live[k]:.5f}  marlowe={stored[k]:.5f}")
    t = model.bias.bias_table.detach().cpu().numpy()
    print("\n  learned bias table (mean over heads) per distance bucket, 0 = self (fixed), 9 = far/unreachable:")
    print("  " + "  ".join(f"d{k}:{v:+.3f}" for k, v in enumerate(t.mean(0))))
    print(f"  head spread at d1 {t[:, 1].std():.3f}, d9 {t[:, 9].std():.3f}; max |bias| {np.abs(t).max():.3f}")

    print(f"\n-- table and paired deltas (results/phase{phase}/rows; previous phase as reproduction check) --\n")
    aggregate(phase)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["0", "1", "2", "3", "3b"])
    ap.add_argument("--key", default=None)
    args = ap.parse_args()
    {"0": phase0, "1": phase1, "2": lambda: phase2(args.key), "3": lambda: phase3(args.key),
     "3b": lambda: phase3(args.key, "3b")}[args.phase]()
