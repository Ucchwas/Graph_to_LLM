# Phase 6 — Aligned multi-layer completion on Decagon

Written 2026-08-27, approved by the user before any row exists. Question: with the architecture
fixed, does one model trained on many aligned graphs complete a graph it has never seen better
than a model trained on that graph alone — and is the gain structure or pair memorisation?

## 1. Why Decagon, and not the PPT-Ohmnet layers

PPT-Ohmnet's 144 tissue layers are each *exactly* the induced subgraph of the combined PPI on
the tissue's active proteins (144 / 144 verified); layer-level prediction there collapses to
node-activity prediction, and an edge appears in 52 tissues on average, so cross-layer hold-outs
leak. OhmNet is kept for mechanics checks only. Decagon (BioSNAP ChChSe-Decagon, Zitnik et al.
2018) has 645 drugs, 63,473 drug pairs and 1,317 side-effect types; the paper's filter (≥ 500
pairs) keeps **963 layers** over the same 645 drug indices. A layer holds a median 9.7 % of the
global pairs among its drugs — layers are genuine edge sets. Layer overlap is low (mean Jaccard
0.035, two pairs above 0.5) but the global pair graph is dense (30.6 % of all pairs; a pair
carries ~73 side effects; 15,844 pairs appear in ≥ 100 layers), so **pair memorisation across
layers is the obvious shortcut** and gets its own control.

## 2. The model (unchanged)

Raw masked adjacency of a layer [645×645] + its `edge_index` → **one shared** `GCNConv(645→1024)`
→ residual GCN block → LayerNorm → bilinear decoder. The 645 drugs are the same indices in every
layer, so the first weight is legitimately one matrix; no per-layer parameters, no drug table,
no E1, no features, no embeddings. Inactive drugs in a layer are all-zero rows. Hidden width 1024
as in Phase 5.

## 3. Splits (`g2l/layers.py`, `g2l/decagon.py`; verified in `tests/test_decagon.py`)

1. **Similarity-grouped layer split** — connected components of the layer graph linked at
   Jaccard ≥ 0.2 (715 components, one 226-layer cluster of generic effects, 696 singletons);
   components assigned whole, 80/10/10 by layer count, seed 0 → **770 train / 96 val / 97 test
   layers**, max train–test Jaccard **0.198**. Test layers: 510–9,143 pairs, median 2,063.
2. **Within-layer split** — 85/5/10 of each layer's visible pairs, 1:1 negatives among the
   layer's non-hidden non-edges (the Phase 1–5 protocol), for every layer.
3. **Global drug-pair-disjoint control** — 10 % of the union pairs (**6,308**), removed from the
   input and the supervision of every layer. Median 210 such positives per test layer (min 52);
   each layer also carries as many hidden non-edges, so the control is a 1:1 AUROC inside the
   never-seen pair pool.

**Supervision rule (Phase 6 only):** held-out cells — the layer's val / test pairs and the
global hidden pairs — are excluded from the masked-cell loss (`mask_matrix(..., exclude)`), not
labelled 0 when drawn as in Phases 1–5. The draw itself is unchanged (bit-identical without the
argument; `tests/test_decagon.py`). Every Phase-6 arm uses it.

## 4. Arms

| arm | what | rows |
|---|---|---|
| **shared** | one model; a step = one train layer with a fresh 15 % cell mask and that layer's pos_weight; epoch = one pass over the 770 train layers; selection on the mean val AUROC of the 96 val layers' completion; ≤ 300 epochs, patience 30 | 963 (input = the layer's train + val pairs; test layers also from 50 % / 25 % of that input) |
| **independent** | one model per layer on its own train pairs (the Phase-5 recipe: ≤ 2,000 epochs, patience 200), same hidden pairs | 963 |
| identity / common neighbours | the partial input itself (leak canary, AUROC 0.5) / A_in² | 963 each |
| **pair_frequency** | number of train layers containing the pair (leave-one-out for train layers) — the memorisation reference; 0 on hidden pairs by construction | 963 |
| knn | Jaccard(input, train layer)-weighted vote of the train layers' pairs | 963 |
| gae0 | featureless GAE (X = I) trained on the layer | 97 test layers |

## 5. Evaluations (per layer AUROC / AP@1:1 on the layer's test pairs; mean ± SE over layers; the layer is the replication unit, one seed)

- **E1 in-distribution** — train layers: shared vs independent, paired per layer (n = 770).
- **E2 held-out-layer completion** — test layers: the shared model, which never trained on
  them, from their own 85 % partial input, vs the independent model trained on that input, vs
  the priors and gae0; paired per layer (n = 97).
- **E3 pair-disjoint** — inside every test layer, AUROC over the globally hidden pairs only.
  Shared ≈ its E2 level → structure; a drop to the prior level → pair memorisation.
- **E4** — the shared model's completion of the test layers from 50 % / 25 % of the usual input.

## 6. Runs (Marlowe)

`slurm/phase6_shared.sbatch` (one job, 10 h): shared → priors → gae0. `slurm/phase6_decagon.sbatch`
(array 0-7, 8 h): independent, 121 layers each. Split so the long shared stage has its own
walltime and the short chunks stay backfillable; every stage skips rows already written at the
commit, so a timed-out job is completed by resubmitting the same script. ≈ 4–6 GPU-h total. Aggregation: `python -m g2l.aggregate6` → `results/phase6/aggregate.md`.
Before submission: `pytest tests/`, the 4-stage CPU smoke on 12 layers, the OhmNet mechanics
check (shared loop over six aligned tissue layers on the laptop CPU).

**Later supporting experiment, documented but not run (2026-08-28):** a scaling curve — the
shared model trained on 50 / 100 / 200 / 400 / 770 train layers, scored on the same held-out
layers. It is the missing test of whether "more graphs" has any slope once the low-data
regularisation effect (§ the correction in `results/phase6/RESULTS.md`) is accounted for.
≈ 1.5 GPU-h, one config knob, no new code. Deferred in favour of Phase 7.

## 7. Gate 6

- [ ] all rows at one commit; identity 0.5 on every layer
- [ ] E1–E4 tables with paired deltas (SE / t / p, n = layers)
- [ ] `results/phase6/RESULTS.md`, VALIDATED line
