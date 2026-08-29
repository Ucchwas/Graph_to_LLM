# Phase 8G — the leaderboard protocol

**Stages 1–2 complete; stage 3 (10-seed final, test read once per seed) NOT yet run.** No 8G run
has touched the test split. Marlowe **454601** (pretrain, 3 tasks, 3.3 GPU-h) + **454602** (select,
6 tasks, 2.7 GPU-h), commit `d312225`. Three fixed seeds `[0, 1, 2]`, official scaffold split,
official Evaluator.

## Stage 1 — identical self-supervision for all three families

Masked-attribute prediction, 15 % of atoms and bonds, 30 epochs, official **training split only**
(`n_pretrain_graphs = 32,901` in all 30 rows), one shared loop. Ten seeds each.

| family | params (with mask rows) | final SSL loss, 10 seeds | per seed |
|---|---|---|---|
| GIN | 1,892,706 | **0.7247 ± 0.0064** | 5 min |
| LGM | 221,449 | 0.8325 ± 0.0071 | 10 min |
| GCN | 535,201 | 0.9351 ± 0.0107 | 5 min |

All curves smooth and still falling at epoch 29. GIN, with 8.5× the parameters, is the best
masked-attribute predictor; the LGM sits between.

## Stage 2 — the one open knob per family, on validation

| arm | val ROC-AUC | per seed | params | **winner** |
|---|---|---|---|---|
| LGM, linear head | **0.8141 ± 0.0061** | 0.8119 / 0.8094 / 0.8210 | 213,257 | ✓ |
| LGM, MLP head | 0.8088 ± 0.0101 | 0.8162 / 0.8128 / 0.7973 | 221,449 | |
| GIN, pretrained | **0.8352 ± 0.0052** | 0.8334 / 0.8312 / 0.8411 | 1,892,706 | ✓ |
| GIN, scratch | 0.8275 ± 0.0069 | 0.8197 / 0.8326 / 0.8302 | 1,892,706 | |
| GCN, pretrained | **0.8318 ± 0.0041** | 0.8363 / 0.8284 / 0.8307 | 535,201 | ✓ |
| GCN, scratch | 0.8102 ± 0.0049 | 0.8112 / 0.8048 / 0.8145 | 535,201 | |

Paired per seed:

| comparison | Δ | per seed | seeds won |
|---|---|---|---|
| LGM linear − MLP | +0.0053 ± 0.0159 | −0.004 / −0.003 / +0.024 | 1/3 — one seed decides it |
| GIN pretrained − scratch | +0.0077 ± 0.0080 | +0.014 / −0.001 / +0.011 | 2/3 |
| GCN pretrained − scratch | **+0.0216 ± 0.0048** | +0.025 / +0.024 / +0.016 | 3/3 |

**Pretraining helps every family**, so every final configuration is a pretrained one. That makes the
"scratch encoder in select vs final" question moot — and it was also checked empirically: the
widened-encoder scratch arms land within noise of Phase 8D's exact-OGB scratch runs on the same
seeds (GIN −0.004 ± 0.011, GCN −0.003 ± 0.017).

## The equal-treatment comparison — the number that matters

| family | validation (best config) | params |
|---|---|---|
| **GIN + train-only SSL** | **0.8352 ± 0.0052** | 1,892,706 |
| **GCN + train-only SSL** | 0.8318 ± 0.0041 | 535,201 |
| LGM + train-only SSL | 0.8141 ± 0.0061 | 213,257 |

| paired | Δ | per seed | seeds won |
|---|---|---|---|
| LGM − GIN | **−0.0211 ± 0.0009** | −0.022 / −0.022 / −0.020 | **0/3** |
| LGM − GCN | −0.0177 ± 0.0075 | −0.025 / −0.019 / −0.010 | 0/3 |

**Under identical procedure the LGM loses to both baselines on validation.** The GIN gap has a
paired std of 0.0009 — it is the same 0.02 on every seed. This is not noise.

## What Phase 8F's "88 % of the gap closed" actually was

| phase | LGM | GIN | gap |
|---|---|---|---|
| 8D — both scratch | 0.7525 | 0.8315 | −0.079 |
| 8F — LGM pretrained, **GIN not** | 0.8219 | 0.8315 | −0.010 |
| 8G — **both** pretrained | 0.8141 | 0.8352 | **−0.021** |

8F compared a pretrained LGM to an unpretrained GIN. Give GIN the same treatment and about a third
of the apparent closure disappears; the rest is real. Net: pretraining is worth +0.06 to the LGM and
+0.008 / +0.022 to GIN / GCN, so the gap narrowed 3.7× — from −0.079 to −0.021 — but did not close.

Two more things the data says about the LGM. Its run-to-run variance is larger than the GNNs':
the identical MLP-head configuration scored 0.8219 in 8F and 0.8088 in 8G, differing only in the
SSL random draw; pooled over six runs it is **0.8153 ± 0.0167** against GIN's ±0.005. And its
selection margin (linear vs MLP head) rests on one seed.

## Leaderboard context

Our pretrained **GIN's validation (0.8352) matches the current rank-1 entry's validation
(0.8348)**, and the LGM's 0.814 would sit around rank 4–5 *by validation*. But this leaderboard's
validation barely predicts its test: the highest-validation entry (directional GSN, 0.8473) ranks
16th on test, and GIN+virtual node goes 0.8479 → 0.7707. Where any of the three lands on test is
unknowable until the final runs.

## Stage 3 — not started

Frozen winners: LGM linear head, GIN pretrained, GCN pretrained. Held pending a decision on whether
to improve the LGM first — a test read locks a configuration, and reading it again for a later
configuration is exactly the multiple-look the protocol forbids.
