# Phase 8F — the readout fix and training-only self-supervision close 88 % of the gap to GIN

**Validation only. The test split was not read, and `g2l/run_phase8f.py` cannot reach it.**

Marlowe **453610** (pretraining) + **453611** (2×2, 4 tasks), 5 tasks, ~3.2 GPU-h, commit `6cdfec8`.
Official scaffold split, official Evaluator, three fixed seeds `[0, 1, 2]`, capacity and dropout
frozen at the Phase-8D winner (d=64, dropout 0.5).

## The 2×2

| | node pooling | node + edge pooling | **effect of the edge readout** |
|---|---|---|---|
| **scratch** | 0.7044 ± 0.0296 | 0.7359 ± 0.0137 | **+0.0314** (3/3 seeds) |
| **pretrained** | 0.8160 ± 0.0318 | **0.8219 ± 0.0216** | +0.0059 (1/3 seeds) |
| **effect of pretraining** | **+0.1116** (3/3) | **+0.0860** (3/3) | |

Deltas are paired per seed — both readouts share a body initialisation at a given seed, so these are
matched comparisons, not differences of means.

| arm | params | val ROC-AUC | per seed | best epoch |
|---|---|---|---|---|
| nodeedge, pretrained | 221,449 | **0.8219 ± 0.0216** | 0.8436 / 0.8003 / 0.8217 | 76 / 23 / 47 |
| node, pretrained | 217,225 | 0.8160 ± 0.0318 | 0.7913 / 0.8048 / 0.8518 | 10 / 23 / 59 |
| nodeedge, scratch | 221,449 | 0.7359 ± 0.0137 | 0.7265 / 0.7294 / 0.7516 | 10 / 5 / 49 |
| node, scratch | 217,225 | 0.7044 ± 0.0296 | 0.7235 / 0.6704 / 0.7194 | 36 / 15 / 17 |

Pretraining: 30 epochs, 15 % of atom and bond attributes masked, final SSL loss 0.822 / 0.832 /
0.839, ~10 min per seed. **`n_pretrain_graphs` = 32,901 in every row — exactly the official training
split**, which is the recorded evidence that no valid or test molecule entered the pretraining.

## Against the reproduced baselines, paired on the same three seeds

| comparison | mean Δ | per seed | seeds won |
|---|---|---|---|
| best 8F (nodeedge + pretrained) vs **GIN** | **−0.0097** | +0.0077 / −0.0313 / −0.0054 | 1/3 |
| best 8F vs **GCN** | **+0.0084** | +0.0107 / −0.0037 / +0.0180 | 2/3 |
| Phase-8D best vs GIN *(for reference)* | −0.0790 | −0.0994 / −0.0325 / −0.1052 | 0/3 |

| model | val ROC-AUC | params |
|---|---|---|
| GIN, reproduced | 0.8315 ± 0.0044 | 1,885,506 |
| **LGM, nodeedge + pretrained** | **0.8219 ± 0.0216** | **221,449** |
| GCN, reproduced | 0.8135 ± 0.0168 | 528,001 |
| LGM, Phase-8D best | 0.7525 ± 0.0410 | 212,425 |

**The gap to GIN went from −0.0790 to −0.0097 — 88 % of it closed, at 12 % of GIN's parameters.**
It is now smaller than the LGM's own seed spread (±0.0216), so on validation the two are no longer
separated. That is not the same as beating GIN, and 1/3 seeds is not a win.

## What actually did the work

**Pretraining, not the readout.** +0.086 to +0.112, positive on 3/3 seeds in both readouts. The
edge readout is worth +0.031 on its own but only +0.006 once the body is pretrained, and its sign
flips across seeds there. The two are **not additive**: they appear to supply overlapping
information, which is the interaction the 2×2 was run to detect. Pooling edge states and training
them with a bond-prediction objective are two routes to the same thing, and having one reduces the
value of the other.

**The MLP head may have cost us.** `node_scratch` here (0.7044) sits **below** the Phase-8D
linear-head reference (0.7525) at mean −0.0481, 0/3 seeds. But the per-seed deltas are
−0.013 / −0.129 / −0.003: one seed carries almost the whole effect, so this is a hypothesis worth a
cheap check, not an established cost. If it holds, the best configuration has not been run yet —
pretrained + edge readout + **linear** head.

## The fairness problem, stated before anyone else raises it

**Our best arm is pretrained and the baselines are not.** "LGM + self-supervision reaches 0.8219"
against "GIN with no pretraining reaches 0.8315" does not license any claim about the architecture,
because attribute masking is a *training procedure* that applies to GIN equally well — it is
literally what Hu et al. (2020) do for GNNs. The like-for-like architecture comparison is still the
scratch row, and there the LGM loses to GIN by 0.096.

This is the same shape of error as the Phase-8 `GCNConv` baseline, which could not read edge
features and flattered us by +0.065 until it was replaced. The honest statements are:

- **Procedure claim (supported):** masked-attribute pretraining on the training split alone is worth
  ~+0.09 to +0.11 to this model, consistently across seeds.
- **Architecture claim (not supported):** nothing here shows the LGM matching GIN as an
  architecture. To claim that, GIN must be pretrained the same way and re-measured.
