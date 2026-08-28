# Phase 8 Gates 8.0–8.2 — one checkpoint transfers across graph sizes, and the edge channel is what makes it work

**Claim (gate scope only):** the incidence-aware Graph Transformer is worth building further. One
checkpoint trained on molecular graphs with **N ≤ 30** scores held-out graphs up to **N = 213** at
**0.9280 AUROC**, essentially unchanged from the 0.9338 it gets at sizes it saw. Deleting the edge
states — at a **byte-identical parameter count** — drops that to **0.8033**, barely above a degree
heuristic's 0.7705. This is a gate, not a benchmark result: see *What this does not show*.

Gates 8.0 / 8.1 (CPU, 31 tests) pass locally and on the cluster build. Gate 8.2: Marlowe array
**452926**, 2 tasks, 1 GPU each, **18 min total (0.3 GPU-h)**, code commit `bd358dd`.
Rows: `results/phase8/rows/`.

## Setup

Architecture (`g2l/lgm.py`, full detail in `docs/PLAN-PHASE8.md`): the raw graph
`(n, edge_index [2,M], edge_value [M,k])` is the external input; node states `[N,d]` and edge states
`[M_u,d]` are internal representations from shared projections. One transformer over both types with
a single shared QKV — node←node dense, node←edge and edge←node on the incidence relation, the first
two sharing one softmax against a common max. `d=256, L=4, H=8`, 3,229,728 parameters, no GCN
anywhere, no node-ID table, no supplied node features, no padding.

Data: `ogbg-molhiv`, 41,123 graphs after filtering (N ≥ 4, ≥ 3 edges), N = 4–222, with genuine
3-dim bond features. **Trained only on N ≤ 30** (25,589 graphs), then scored on held-out graphs at
sizes seen (3,193) and never seen (920, N = 31–213). Masked-cell reconstruction, 15 % of upper-
triangle cells hidden per graph, `edge_index` rebuilt from the observed graph only. One seed.

## Gate 8.2 results

| arm (params identical) | seen sizes N ≤ 30 | unseen sizes N 31–213 |
|---|---|---|
| **LGM** | **0.9338** AUROC / 0.6438 AP | **0.9280** AUROC / 0.4738 AP |
| no-edge control | 0.8065 / 0.3069 | 0.8033 / 0.1637 |
| **edge channel** | **+0.1273** AUROC / +0.3369 AP | **+0.1247** AUROC / +0.3101 AP |
| negated degree (best prior) | 0.7697 / 0.2289 | 0.7705 / 0.1133 |
| common neighbours | 0.4426 | 0.4757 |
| preferential attachment | 0.2303 | 0.2295 |
| random | 0.5010 | 0.4862 |

Across unseen-size quartiles the LGM holds 0.9362 / 0.9321 / 0.9090 / 0.9168 (last bucket N 70–213,
7× the training cap). The control is equally flat at 0.810 / 0.807 / 0.793 / 0.810 — **size
transfer is a property of the architecture, not of the edge channel**; the edge channel supplies the
accuracy, and the two effects are independent.

**Two baselines run backwards on molecules**, and were declared as their own arms before the run
rather than sign-flipped afterwards. Common neighbours scores 0.44 because atoms sharing a neighbour
sit at ring or bond-angle distance and are exactly the pairs *not* bonded; preferential attachment
scores 0.23 because a high-degree atom is valence-saturated. Negated degree at 0.77 is therefore the
real bar, and a much harder one than common neighbours would have been.

## Gates 8.0 / 8.1 — the invariants, and three traps avoided

- **Automorphism invariance is asserted over *pair* orbits.** Node-orbit equivalence proves nothing
  about pairs: Petersen is vertex-transitive, yet (0,1) is an edge and (0,2) is not, and its 45
  pairs form exactly two orbits (15/30).
- **Petersen cannot carry the assertion.** On a vertex-transitive graph the theorem forces every
  node state equal, and with a node-state-only decoder the entire score matrix collapses to one
  constant (measured spread 1.8e-15 in float64) — a broken constant-output model passes identically.
  P₆ and K₂,₃ carry it; Petersen is a separate, labelled *collapse* test.
- **No metric is asserted.** `roc_auc_score` returns **0.5533** in fp32 on a provably constant score
  matrix: orbit-equivalent nodes sit at different memory offsets, their reductions run in a different
  order, and float non-associativity leaves a residual that is never exactly zero — and AUROC swings
  across its full range on ulp ties.
- **Size independence is behavioural**: `strict=True` load of one `state_dict` into a
  differently-seeded model, run at `N ∈ {7, 23, 101, d_model}` on Erdős–Rényi probes, `state_dict`
  shape map compared before and after every forward, gradient-receiving parameter set required to
  match. A shape scan for "no dimension equals N" was rejected — it flags every `[d,d]` tensor the
  moment a probe has `N = d_model`.
- **Batching is masking, not padding**: a batch of 12/25/40 reproduces three single-graph runs;
  perturbing one graph moves the others by 8.3e-17 against its own 6.6e-01.

One design consequence worth recording: there is **no src/dst role tag anywhere**. A node
permutation can flip which endpoint is `min` in the canonical edge key, so anything distinguishing
an edge's two endpoints breaks equivariance — edge←node must score both slots with the same
projection.

## What this shows

- **The first architecture in this project that transfers across graphs.** Phases 4–7 ran
  `gnn_direct`, whose `GCNConv(N → d)` first weight is one column per node index (57 % of the Cora
  model's parameters, 79 % of Photo's). It cannot be evaluated at a different N at all. This one
  runs unchanged from N = 4 to N = 213.
- **Edges as first-class `[M,d]` states are load-bearing.** The control is the same model with the
  edge states emptied at run time — same shapes, same parameter count — and it loses 0.125 AUROC and
  two thirds of its AP. An architecture that had compressed edges into a dense `[N,N]` attention
  bias would plausibly have landed near the control.

## What this does not show

- **Molecules are the friendliest possible domain for this.** Bonds are valence-constrained and
  degree alone reaches 0.77. Nothing here predicts the number on a citation or interaction graph.
- **The edge advantage may not survive leaving molecules.** It rests on real 3-dim bond features.
  Cora, Amazon Photo, ogbl-ddi, PPT-Ohmnet and the binarised TCGA graphs all have `k = 1` and
  all-ones edge values, where the edge channel would carry only endpoint degrees. **The result is
  currently not reproducible on any other dataset in this repo.**
- **The bar cleared is low.** "Beats a degree heuristic" is not "beats GCN". There is no GCN, GAE or
  MaskGAE comparison yet — those are in the deferred grid, and a regression against Phase 5's
  fixed-N numbers on Cora/Photo is *expected* (pre-registered in `docs/PLAN-PHASE8.md` §7).
- **One seed, no variance estimate.** And the default arm is 1-WL bounded with a proven automorphism
  ceiling that has never been measured on a real graph.
- The LGM ran the full 40 epochs with validation still rising (0.9330 at the cap), so it is
  **undertrained** — the number is a floor, but the epoch cap was not selected on anything.

## Gate 8

- [x] 8.0 permutation equivariance, pair-orbit invariance, behavioural size independence
- [x] 8.1 no-padding batch equivalence and cross-graph leakage
- [x] 8.2 one checkpoint transfers to unseen sizes (0.9280 at N 31–213) and beats every non-learned
      prior; edge channel worth +0.125 AUROC at identical parameters
- [ ] VALIDATED line — user

VALIDATED ____
