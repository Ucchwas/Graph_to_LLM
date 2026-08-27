# Phase 4 — LGM v0: raw adjacency → GNN backbone (trained end to end) → graph out

Written 2026-08-27 after the laptop smoke; the `main` grid below is pre-registered before any
of its rows exist. Direction: Dr. Islam (LGM, not LLM; `PLAN-LGM.md`) and Sakib ("use the GNNs
as backbone first; our GFM also uses a GNN backbone; with the GFM you cannot tell which part
works"). The lab GFM takes the backbone slot in Phase 5 when its checkpoint arrives.

## 1. The model

```
A_obs [N,N] (15 % of cells hidden)
  → E1 row tokenizer  Linear(N → d) + LayerNorm            (a layer of the network, trained)
  → GNN backbone: L pre-norm residual blocks  h ← h + GELU(conv(LN(h), edges(A_obs)))
      conv ∈ {GCN, GAT (4 heads), GraphSAGE, GIN}; the edge set is the model's own input graph,
      so held-out cells never reach the backbone (leak rule as in Phase 3)
  → decoder-input LayerNorm → D1  Â = Z W Zᵀ                (trained)
loss: pos-weighted BCE on the hidden cells only; AdamW, one LR for tokenizer + backbone + decoder
```

No node features, no text, no external embeddings. Everything trains. Harness, splits, seeds,
masking, metrics, early stopping and controls are the Phase 2–3 ones unchanged.

## 2. Smoke (laptop, 2026-08-27, `results/phase4/aggregate.md`, 27 rows @ 4638a4f)

d 256, 2 blocks, LR {1e-3, 3e-3}, seeds 0–2; `none` re-run as the reference.

| backbone | AUROC | AP@1:1 | Δ AUROC vs none (paired, n = 3) |
|---|---|---|---|
| GCN | 0.933 ± 0.002 | 0.940 | +0.015 (p 0.03) |
| GAT | 0.933 ± 0.003 | 0.941 | +0.015 (p 0.04) |
| SAGE | 0.925 ± 0.004 | 0.930 | +0.006 (p 0.15) |
| GIN | 0.924 ± 0.007 | 0.930 | +0.005 (p 0.43) |
| none (linear) | 0.919 ± 0.003 | 0.929 | reproduces Phase 3 to 0.0000 |

The pipeline works: a graph-native backbone trained end to end is the first configuration
above the linear model, and above every frozen-LLM arm (best 0.922), GAE (0.905) and GAT with
features (0.910). Every smoke selection sits on the top of its two-point LR grid, so `main`
extends upward. 0.05 s/epoch, < 0.3 GB.

## 3. Questions for `main`

- **Q1** Which backbone, and does width or depth help on one graph? (kind × d × L surface)
- **Q2** How far above the linear model and the Phase-3 arms is the LGM v0? (paired vs `none`,
  random + SPD, pretrained + SPD at this commit)
- **Q3** Is it the structure? (shuffled-A control on the two best kinds)

## 4. The array (one submission, Marlowe, 1 GPU per task)

| block | cells | seeds | runs |
|---|---|---|---|
| backbone grid: kind {gcn, gat, sage, gin} × d {256, 1024} × L {2, 4, 8} × LR {1e-3, 3e-3, 1e-2} | 72 | 0–9 | 720 |
| shuffled-A control: gcn and gat at d 256, L 2, LR 3e-3 | 2 | 0–9 | 20 |
| references at this commit: none (3e-4), random + SPD (3e-3 / 3.0), pretrained + SPD (1e-2 / 1.0) | 3 | 0–9 | 30 |

770 runs; estimate 8–12 GPU-h (GNN runs 20 s–2 min, Llama references 3 min). Dropout 0, weight
decay 0.01, patience 200 — the smoke settings. GAE / GAT-with-features rows come from Phase 1
(same protocol). Edge rule: one ×3 step if a selected LR sits on the grid boundary (1e-2 is
expected to be interior). Seed rule: seeds 10–19 at the selected cells if a gate delta falls
below its MDE. Selection per (kind, d, L): best mean val AUROC over seeds 0–9.

## 5. Gate 4

- [ ] `pytest tests/` green on both machines; cluster commit pinned for the whole phase
- [ ] 770 rows; kind × d × L surface; every selected LR interior or extended
- [ ] paired deltas (SE / t / p / MDE) vs none, random + SPD, pretrained + SPD, and vs the
      shuffled control
- [ ] walkthrough: the selected backbone seed 0 rebuilt on the laptop from its checkpoint,
      logits vs the saved ones
- [ ] `results/phase4/RESULTS.md`, VALIDATED line

## 6. Then

Phase 5: the lab GFM in the backbone slot (from checkpoint and from scratch; leakage check on
its pretraining data), same grid discipline, paired against the best Phase-4 backbone.
Phase 6: multi-graph (OGB molecular) — the size-agnostic setting where "large" is testable.
