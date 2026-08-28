# Phase 6 — Aligned multi-layer completion on Decagon: one shared model matches or beats 963 specialists, mostly where data is scarce

> **Corrected 2026-08-28.** The first version of this file read the headline delta as evidence
> that the backbone transfers across graphs. It does not support that. The advantage is
> concentrated in data-poor layers and vanishes on data-rich ones (§"Where the gain lives"), so
> the honest reading is **multi-task regularisation, low-data transfer, and parameter
> efficiency**. All measurements below are unchanged; only the interpretation is.

**Claim:** with the architecture unchanged from Phase 5, **one 2.8 M-parameter model trained on
770 aligned graphs completes a graph it has never seen at least as well as a model trained on
that graph alone, and better when that graph is small** — over the 97 held-out side-effect layers
the shared model averages **0.9770 ± 0.0018 AUROC** against the per-layer models' 0.9711 ± 0.0020,
a paired **+0.0059 ± 0.0012** (t = 4.79, p = 6.2e-06, better on 68 / 97). That average is carried
by the small layers: **+0.0129** in the smallest size quartile against **+0.0008** in the largest,
with the per-layer gain correlating **−0.35** with layer size — a regularisation effect, not
uniform transfer. What the shared model does earn cleanly is **parameter efficiency**: one model
replaces 963, at no cost in accuracy. **What advantage exists is structural, not pair
memorisation:** on the 6,308 drug pairs hidden from the input *and* the supervision of every
layer, the shared model scores **0.9321 ± 0.0050** (independent 0.9226, paired +0.0095,
p = 1.3e-07) while pair frequency and the kNN-layer vote sit at exactly 0.5000 by construction —
though common neighbours, with zero parameters, reaches 0.9187 on those same pairs, so the margin
over non-learned structure is only +0.0134.

Produced on Marlowe (H100 80 GB, 1 GPU per task), jobs **451810** (shared → priors → GAE, 15 min)
and **451811** (array 0-7, per-layer models, ~8 min each) — 9 tasks, all COMPLETED, exit 0, no
failure in any log, **≈ 1.3 GPU-h** total. All **5,875 rows at commit `f33eaa4`**.
Aggregation: `results/phase6/aggregate.md`.

## Locked setup

| item | value |
|---|---|
| data | Decagon (BioSNAP ChChSe-Decagon): 645 drugs, 63,472 drug pairs, **963 side-effect layers** (the paper's ≥ 500-pair filter) over the same 645 node indices. A layer holds a median 9.7 % of the global pairs among its drugs — layers are genuine edge sets, not induced subgraphs |
| model | raw masked adjacency rows of a layer + its `edge_index` → **one shared** `GCNConv(645→1024)` → residual GCN block → LayerNorm → bilinear decoder. 2,763,776 parameters, no per-layer parameters, no drug table, no E1, no node features, no embeddings |
| layer split | components of the layer graph linked at Jaccard ≥ 0.2, assigned whole: **770 train / 96 val / 97 test**; max train–test Jaccard **0.198** |
| within-layer split | 85/5/10 of each layer's pairs, 1:1 negatives among that layer's non-edges |
| pair-disjoint control | **6,308** of the union pairs removed from the input and supervision of every layer (median 210 positives per test layer, min 52), with as many hidden non-edges for a 1:1 AUROC |
| supervision | pos-weighted BCE on masked cells only; held-out and globally hidden cells **excluded** from the loss rather than labelled 0 |
| shared arm | a step = one train layer, fresh 15 % cell mask, that layer's pos_weight; epoch = one pass over the 770 train layers; selection on mean val-**layer** AUROC; ≤ 300 epochs, patience 30 → stopped at epoch 97, best **0.9783 at epoch 66**, 3.9 s/epoch |
| independent arm | one model per layer on its own pairs, ≤ 2,000 epochs, patience 200 → median 442 epochs, max 1,457; **no layer hit the cap**, so the comparison is not confounded by a truncated budget |

## Results (per-layer AUROC / AP@1:1, mean ± SE over layers; the layer is the replication unit)

### E2 — held-out layers (n = 97), the headline

| arm | AUROC | AP@1:1 | AUROC hidden pairs | AP hidden pairs |
|---|---|---|---|---|
| **shared** (never trained on these layers) | **0.9770 ± 0.0018** | **0.9727 ± 0.0023** | **0.9321 ± 0.0050** | **0.9273 ± 0.0055** |
| independent (trained on exactly this layer) | 0.9711 ± 0.0020 | 0.9684 ± 0.0023 | 0.9226 ± 0.0046 | 0.9166 ± 0.0051 |
| featureless GAE | 0.9577 ± 0.0022 | 0.9526 ± 0.0028 | 0.8948 ± 0.0058 | 0.8821 ± 0.0064 |
| common neighbours | 0.9520 ± 0.0022 | 0.9470 ± 0.0025 | 0.9187 ± 0.0045 | 0.9102 ± 0.0050 |
| kNN-layer vote | 0.9500 ± 0.0018 | 0.9318 ± 0.0028 | **0.5000** | **0.5000** |
| pair frequency (memorisation) | 0.9475 ± 0.0018 | 0.9278 ± 0.0028 | **0.5000** | **0.5000** |
| identity (leak canary) | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

Paired shared − independent, n = 97: AUROC **+0.0059 ± 0.0012** (t = 4.79, p = 6.2e-06, 68/97);
AP **+0.0043 ± 0.0011** (p = 0.00026); **hidden-pair AUROC +0.0095 ± 0.0017** (t = 5.71,
p = 1.3e-07, 71/97); hidden-pair AP **+0.0107 ± 0.0023** (p = 9.1e-06).

### Where the gain lives (the correction; regenerated by `python -m g2l.aggregate6`)

Splitting the 97 held-out layers by how much training data each layer has itself:

| quartile by layer size | train pairs | shared | independent | delta |
|---|---|---|---|---|
| 1 (smallest) | 391–871 | 0.9912 | 0.9782 | **+0.0129** |
| 2 | 893–1,580 | 0.9864 | 0.9807 | +0.0057 |
| 3 | 1,617–2,902 | 0.9746 | 0.9707 | +0.0039 |
| 4 (largest) | 3,261–6,962 | 0.9551 | 0.9543 | **+0.0008** |

corr(log layer train size, shared − independent AUROC) = **−0.350**. Delta spread across layers:
min −0.0077, median +0.0029, max +0.0818 — and **the shared model is worse on 29 of the 97
layers**. The mean is pulled up by a few large wins, all on small layers.

Against the zero-parameter heuristic: shared − common neighbours is **+0.0250 ± 0.0017** on the
ordinary test pairs (better on 96/97) but only **+0.0134 ± 0.0019** on the hidden pairs
(78/97) — most of the "structural" signal on never-seen pairs is recoverable by two-hop counting.

**Reading:** once a layer has a few thousand training pairs of its own, a model fitted to that
layer alone is as good as the shared one. The shared model is acting as a prior for data-poor
graphs, which is standard multi-task learning, not evidence of a transferable graph backbone.

### E1 — train layers (n = 770, in-distribution)

shared 0.9690 ± 0.0008 vs independent 0.9633 ± 0.0008; paired **+0.0057 ± 0.0004**
(t = 15.88, p = 2.6e-49, better on 587/770). Hidden pairs: 0.9033 vs 0.8962, +0.0070 (p = 2.9e-32).
Val layers (n = 96) behave identically: +0.0079 (p = 2.8e-06).

### E4 — the shared model from a reduced input (test layers)

| input kept | AUROC | AP@1:1 |
|---|---|---|
| 100 % | 0.9770 ± 0.0018 | 0.9727 ± 0.0023 |
| 50 % | 0.9662 ± 0.0020 | 0.9626 ± 0.0025 |
| 25 % | 0.9416 ± 0.0024 | 0.9377 ± 0.0028 |

Halving a held-out layer's visible edges costs 0.011 AUROC; quartering it costs 0.035.

## What this shows / does not show

- **Shows parameter efficiency.** One 2.8 M-parameter model matches or beats 963 separately
  fitted models — a ~963× reduction at no accuracy cost. This is the claim that survives scrutiny.
- **Shows a low-data advantage, not general transfer.** The shared model helps most where the
  target layer has least data (+0.0129 in the smallest quartile, +0.0008 in the largest,
  correlation −0.35, worse on 29/97 layers). That is multi-task regularisation. A genuine
  foundation-model claim would require the advantage to persist on data-rich graphs; it does not.
- **Shows the advantage that exists is structural.** On pairs no layer ever revealed, the shared
  model keeps 0.932 AUROC while both memorisation references are at chance — and cross-layer pair
  memorisation is a genuinely strong shortcut here (pair frequency alone scores 0.9475 on ordinary
  test pairs), which is why the control was built. But common neighbours reaches 0.9187 on those
  same hidden pairs, so the learned model's margin over free structure is only +0.0134.
- **Does not show a discriminating benchmark.** Every arm lands between 0.947 and 0.977, including
  a three-line heuristic. A task where the trivial baseline is already at 0.95 cannot separate
  architectures, and differences of ±0.006 in that band carry little scientific weight regardless
  of their p-values.
- **Shows the harness is clean.** Identity is exactly 0.5000 on all 963 layers, pair frequency is
  exactly 0.5000 on every hidden-pair set, all 5,875 rows carry one commit, no layer hit its epoch
  cap, and no log contains an error.
- **Does not show a large effect.** +0.006 AUROC is highly significant but small; the practical
  claim is "sharing is at least as good as specialising, at 1/963 of the parameters, and helps
  when the target graph is small", not "sharing is dramatically better".
- **Does not show superiority on every column.** Common neighbours beats every learned arm on the
  sparse full-matrix AP (0.2001 vs the shared model's 0.1468 on test layers): for ranking *all*
  645² cells, a simple heuristic is still stronger on this dataset.
- **Does not have error bars over seeds.** One seed; replication is over layers, not over
  initialisations.
- **Does not generalise beyond aligned node sets.** Decagon works because all 963 layers share the
  same 645 drug indices. Unaligned graphs of different sizes remain out of reach of this
  architecture (see `docs/PLAN-PHASE5.md` §5).

## Gate 6

- [x] all 5,875 rows at one commit (`f33eaa4`); identity exactly 0.5 on every layer
- [x] E1–E4 tables with paired deltas (SE / t / p, n = layers)
- [x] pair-disjoint control clean: both memorisation references at chance on hidden pairs
- [x] independent arm not truncated (0 of 963 layers hit the 2,000-epoch cap)
- [ ] VALIDATED line — user

VALIDATED ____
