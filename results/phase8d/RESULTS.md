# Phase 8D — molhiv calibration: the harness is sound, and the gap is the architecture

**Gate FAILS. The test split was never touched.** Our reproduced baselines match OGB's published
numbers, capacity was swept across 15×, and the best LGM configuration still sits **0.079 below GIN
on validation** — about 18× GIN's seed spread. With the harness validated and capacity ruled out,
the remaining shortfall is the architecture on this task.

Marlowe array **453087**, 11 tasks × 3 seeds, ~5 GPU-h, commit `7b23b9d`.
Official scaffold split, official Evaluator, validation only.

## The baselines reproduce — this is what makes the rest meaningful

| model | reproduced here | OGB published | params (ours / theirs) |
|---|---|---|---|
| GIN | **0.8315 ± 0.0044** | 0.8232 ± 0.0090 | 1,885,506 / 1,885,206 |
| GCN | **0.8135 ± 0.0168** | 0.8204 ± 0.0141 | 528,001 / 527,701 |

Both within about one standard deviation, at parameter counts matching to 0.02 %. A published number
could not have told us whether our pipeline was broken; a reproduction can, and it isn't.

## The sweep — capacity and regularisation only

| config | params | val ROC-AUC | per seed |
|---|---|---|---|
| **d=64, dropout 0.5** | 212,425 | **0.7525 ± 0.0410** | 0.7365 / 0.7991 / 0.7219 |
| d=128, dropout 0.0 | 818,065 | 0.7483 ± 0.0228 | 0.7401 / 0.7308 / 0.7741 |
| d=128, dropout 0.3 | 818,065 | 0.7481 ± 0.0107 | 0.7605 / 0.7415 / 0.7423 |
| d=64, dropout 0.0 | 212,425 | 0.7327 ± 0.0325 | |
| d=64, dropout 0.3 | 212,425 | 0.7244 ± 0.0230 | |
| d=256, dropout 0.3 | 3,208,993 | 0.7242 ± 0.0028 | |
| d=256, dropout 0.5 | 3,208,993 | 0.7234 ± 0.0112 | |
| d=128, dropout 0.5 | 818,065 | 0.7042 ± 0.0208 | |
| d=256, dropout 0.0 *(the Phase-8C config)* | 3,208,993 | 0.6985 ± 0.0339 | |

## What this establishes

- **The overfitting diagnosis was right, and insufficient.** Moving from the Phase-8C configuration
  to the best one gains **+0.054** (0.6985 → 0.7525), confirming we were badly over-capacity. It
  closes roughly 40 % of the gap and no more.
- **Smaller is monotonically better**: 212k → 0.7525, 818k → 0.7483, 3.2M → 0.7242. This is the
  direct empirical answer to the "maybe a bigger model" hypothesis — on this dataset the LGM wants
  *less* capacity, not more.
- **The LGM is also unstable here**: seed spread up to ±0.041 against GIN's ±0.0044.
- **The residual −0.079 is not a setup artefact.** Harness validated, capacity swept 15×, three
  seeds. On ogbg-molhiv the architecture is genuinely behind standard MPNNs.

## What it does not establish

- Only capacity and regularisation were tuned. Learning rate, depth, heads, encoder and decoder were
  fixed, and no virtual node was tried — all deliberately out of scope.
- molhiv is graph→scalar through a mean-pool bottleneck; our architecture is built for graph→graph.
  This measures the encoder, not the thing the architecture is for.
- molhiv rewards chemistry features over graph structure (a 5,782-parameter random forest on
  fingerprints outranks every GNN below rank 9), so it flatters domain features generally.
