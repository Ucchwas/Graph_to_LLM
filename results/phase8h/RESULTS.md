# Phase 8H — the final leaderboard table: the LGM wins validation and loses test

**ogbg-molhiv, official scaffold split, official Evaluator, 10 random seeds per OGB's submission
rule.** Marlowe **454796 → 454797 → 454798 → 455018 → 455019**, 26 tasks, zero failures, ~14 GPU-h,
commit `272ab99`. Every family: identical loader, batcher, optimiser, 100-epoch masked-attribute
self-supervision on the **training split only**, warm-up + clipping, same 10 seeds, same evaluator.
Test read once per seed per frozen configuration, after selection had closed on validation.

| model | Test ROC-AUC | Validation ROC-AUC | #Params |
|---|---|---|---|
| **GIN** + train-only SSL | **0.7776 ± 0.0123** | 0.8279 ± 0.0097 | 1,892,706 |
| **GCN** + train-only SSL | 0.7625 ± 0.0072 | 0.8274 ± 0.0079 | 535,201 |
| **LGM** d=256 + train-only SSL | **0.7446 ± 0.0180** | **0.8351 ± 0.0111** | 3,212,321 |

Paired per seed, on test:

| comparison | Δ | seeds won | t |
|---|---|---|---|
| LGM − GIN | **−0.0330 ± 0.0226** | **0 / 10** | −4.62 |
| LGM − GCN | −0.0178 ± 0.0187 | 1 / 10 | −3.01 |
| GCN − GIN | −0.0151 ± 0.0177 | 2 / 10 | −2.70 |

## The result

**The LGM has the best validation of the three and the worst test.** It led validation by +0.007
over GIN and trails test by −0.033, losing on 10 seeds out of 10. This is not noise: t = −4.62.

The mechanism is visible in the val→test drop, which scales with model size exactly as the
leaderboard's own GNN entries predicted (params↔test correlation −0.70, params↔validation +0.72):

| model | validation | test | drop | params |
|---|---|---|---|---|
| GIN | 0.8279 | 0.7776 | **−0.0503** | 1.89 M |
| GCN | 0.8274 | 0.7625 | −0.0649 | 0.54 M |
| LGM d=256 | 0.8351 | 0.7446 | **−0.0905** | 3.21 M |

Under molhiv's scaffold split, validation chemistry sits near training chemistry and test chemistry
does not. The LGM fits the former best and transfers worst. **Phase 8H's validation "win" over GIN
was an artefact of the selection metric, and the test split says so unambiguously.**

This also means the width choice was, in hindsight, the wrong one — the launcher's argmax-validation
rule picked d=256, the largest arm, and size is precisely what predicts a worse test score here. A
one-standard-error rule would have selected d=64 (validation 0.8314 ± 0.0181, inside one SE of
d=256's 0.8352). That was flagged before selection ran and left unchanged by instruction; **d=64 has
not been evaluated on test and must not be, retroactively — that would be selecting on the test
split.**

## Leaderboard placement

| test | method |
|---|---|
| 0.8476 | Multi-RF Fusion + Multi-GNN *(rank 1)* |
| 0.8208 | Molecular FP + Random Forest *(rank 9, 5,782 params)* |
| 0.8094 | CIN *(rank 10, best pure-graph)* |
| 0.7921 | GINE *(rank 20)* |
| 0.7858 | DeeperGCN *(rank 26)* |
| **0.7776** | **ours — GIN + train-only SSL** |
| 0.7707 | GIN + virtual node *(rank 35)* |
| **0.7625** | **ours — GCN + train-only SSL** |
| 0.7606 | GCN *(rank 38, OGB reference)* |
| 0.7558 | GIN *(rank 40, OGB reference — the config we reproduce)* |
| 0.7549 | GCN in Julia *(rank 41, last)* |
| **0.7446** | **ours — LGM d=256** |

**The LGM would not rank**: 0.7446 is 0.0103 below the last entry on the board.

**Our GIN and GCN would rank, around 18th and 38th.** Note that our GIN at 0.7776 beats OGB's own
published GIN at the same configuration (0.7558) by **+0.022** — the train-only self-supervision is
a real, transferable improvement. It just improves the baselines at least as much as it improves us.

## What this settles

- Under fully identical treatment, over 10 seeds, on an externally-defined benchmark with a held-out
  test split, **the LGM is behind both standard MPNNs, and behind by more on test than on
  validation**. The architecture question on molhiv is answered, negatively.
- **Masked-attribute pretraining on the training split alone is worth +0.02 test to GIN.** That is a
  procedure result, independent of our architecture, and the most useful thing this phase produced.
- **Validation is an actively misleading model-selection signal on this dataset.** Any future work
  here should select on something else — a scaffold-split inner CV, or a size-penalised rule.
