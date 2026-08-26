# Phase 1 — Baselines

**Claim:** every cheap and standard link-prediction baseline is measured on the locked
protocol; the protocol reproduces the published GAE numbers; and the bar Phase 2 must
clear is known — **~0.85 AUROC / ~0.90 AP@1:1 for topology-only models** (PPR ≈ scratch
adjacency-row transformer), while feature-using GAE/GAT sit at ~0.91.

Produced on Marlowe (H100, jobs 447748 + 447749, commit `9bfb434`; 2m37s + 11m16s
wall-clock; ≤2.4 GB CPU RSS). Recomputed live on the laptop by
`python -m g2l.walkthrough --phase 1`.

## Table — Cora, RandomLinkSplit 85/5/10, 5 seeds, mean ± std

| model | n | AUROC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift |
|---|---|---|---|---|---|---|
| identity | 5 | 0.500±0.000 | 0.500±0.000 | 0.500±0.000 | 0.0001±0.0000 | 1 |
| random | 5 | 0.497±0.023 | 0.503±0.018 | 0.498±0.010 | 0.0002±0.0000 | 1 |
| common_neighbors | 5 | 0.736±0.007 | 0.733±0.007 | 0.736±0.007 | 0.0124±0.0024 | 86 |
| adamic_adar | 5 | 0.737±0.007 | 0.736±0.008 | 0.737±0.007 | 0.0235±0.0030 | 163 |
| resource_allocation | 5 | 0.737±0.008 | 0.736±0.008 | 0.737±0.007 | 0.0227±0.0027 | 157 |
| ppr (α=0.15) | 5 | 0.850±0.012 | 0.900±0.010 | 0.850±0.012 | 0.0308±0.0049 | 214 |
| feature_only_logreg | 5 | 0.610±0.016 | 0.642±0.007 | 0.611±0.013 | 0.0003±0.0000 | 2 |
| gae | 5 | 0.901±0.013 | 0.900±0.017 | 0.901±0.008 | 0.0054±0.0025 | 38 |
| gae_600ep | 5 | 0.905±0.006 | 0.912±0.005 | 0.908±0.006 | 0.0078±0.0016 | 54 |
| vgae | 5 | 0.894±0.014 | 0.895±0.015 | 0.895±0.014 | 0.0053±0.0008 | 37 |
| gat | 5 | 0.910±0.009 | 0.911±0.008 | 0.910±0.007 | 0.0052±0.0007 | 36 |
| scratch_d128 (1.2M) | 5 | 0.847±0.010 | 0.877±0.008 | 0.848±0.005 | 0.0180±0.0027 | 125 |
| scratch_d256 (3.9M) | 5 | 0.836±0.012 | 0.872±0.010 | 0.837±0.009 | 0.0198±0.0031 | 138 |
| scratch_d512 (14.3M) | 5 | 0.825±0.014 | 0.865±0.010 | 0.826±0.015 | 0.0167±0.0039 | 116 |
| scratch_d1024 (54.2M) | 5 | 0.823±0.007 | 0.864±0.005 | 0.823±0.008 | 0.0181±0.0021 | 126 |
| scratch_d2048 (211.2M) | 5 | 0.819±0.009 | 0.860±0.007 | 0.817±0.008 | 0.0157±0.0010 | 109 |

Columns: AUROC and AP@1:1 on the 527 test edges + 527 seeded non-edges (the
published-comparable view); AUROC(sparse), AP(sparse) and lift = AP/base-rate on all
3.66 M unseen upper-triangle cells (base rate 0.000144). Scratch rows: parameter count in
parentheses; LR chosen per width on seed-0 validation AUROC from {1e-3, 3e-4, 1e-4}.

## Gate checklist
- [x] GAE reproduces arXiv 2107.02658 Table 2 — 0.905±0.006 / 0.912±0.005 vs 90.6±0.9 / 91.2±1.0
      (VGAE 0.894 / 0.895 vs 89.8 / 90.3)
- [x] identity canary at exactly chance on every seed (AUROC 0.500, lift 1.0): no held-out
      edge reaches the observed graph; random at chance
- [x] Marlowe reproduces the laptop: common_neighbors seed 0 identical on all four columns;
      trained models within seed noise
- [x] full table: 11 baselines × 5 seeds; scratch capacity curve 5 widths × 5 seeds, every
      seed converged (min AUROC 0.797)
- [x] 22/22 tests green on both machines (`test_split_is_deterministic` added this phase)
- [ ] user ran `python -m g2l.walkthrough --phase 1` and signed below

## What the numbers show
1. **The protocol gap flips rankings.** Balanced view: GAE 0.912 > PPR 0.900. True-prevalence
   view: PPR 0.031 > GAE 0.008 (4×); even Adamic-Adar (0.024) beats every trained model
   (≤0.008). Identical AUROC, ~50× smaller AP. Both views stay on every table.
2. **Features alone predict nothing** (0.61 AUROC, lift 2). **Topology alone reaches 0.85**
   (PPR, scratch). GAE/GAT combine both → 0.91. So the feature-topology interaction is worth
   ~6 AUROC points on Cora.
3. **The scratch adjacency-row transformer is flat in width**: 0.847 → 0.819 AUROC from
   1.2 M to 211 M parameters, saturating at the topology-only ceiling. Phase 2's E1 model is
   topology-only, so its reference line is 0.83–0.85 / 0.86–0.90 — not GAE's 0.91. At the
   parameter count Phase 2 will train (~10 M: E1 + D1), the matched scratch points are
   d256–d512: 0.825–0.836 / 0.865–0.872.

## Does not show
- Anything about a pretrained LLM (Phase 2) or about features inside a transformer (E4/E5,
  Phase 4). PPR's sparse-view number is a reference ceiling, not a trainable model.

## Recorded: the first scratch sweep failed (file deleted, numbers kept here)
Recipe: default-initialised `Linear(N, d)` input, 300 epochs, patience 30, no warmup.
22 of 25 runs stalled at 0.59–0.69 AUROC — the degree-only solution (preferential attachment
on the same splits: 0.654±0.016 / 0.694) — and 3 reached 0.85–0.87 (per-width best AUROC:
d128 0.867, d256 0.849, d512 0.673, d1024 0.691, d2048 0.674). Fix: LayerNorm after the
input projection (as in encoder E1), 100-step linear warmup, 2000 epochs / patience 200.
Best epochs now fall at 27–235 and every seed converges. Also fixed this phase: PyG's
`negative_sampling` draws with Python's `random`, so `edge_split` now seeds it — before
that the 527 balanced test negatives changed on every call (third-decimal AUC wobble).

## Locked going forward
- Split: `RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True, split_labels=True,
  add_negative_train_samples=False)`, torch and Python `random` seeded together; seeds 0–4.
- Baseline configs: GAE/VGAE GCN 1433→32→16, Adam 0.01, selection on (AUROC+AP)/2 with
  patience 100 (max 1000) or fixed 600 epochs; GAT 8 heads × 8 → 16, dropout 0.2, Adam 0.01;
  PPR α = 0.15; feature-only logistic regression on [x_i ‖ x_j ‖ x_i ⊙ x_j], class-balanced;
  scratch: 4 pre-LN layers, 8 heads, GELU, dropout 0.1, AdamW wd 0.01, grad-clip 1.0,
  bilinear decoder W = 0.1·I at init.
- Phase-2 reference lines: topology-only 0.85 / 0.90 (PPR); matched-parameter scratch
  0.825–0.836 / 0.865–0.872; feature-using 0.91 / 0.91 (GAE, GAT).

VALIDATED: ____________
