# Phase 10 — ogbg-molhiv: RRWP helps the LGM directionally, RWSE is null on test, and the LGM still loses

**Marlowe 455869 → 455870 → 456417 → 456418 → 456679/456706 → 456707**, all COMPLETED, ~35 GPU-h,
commit `2be6daf`. Official scaffold split, official `Evaluator`, all 9 atom and 3 bond attribute
columns, 100-epoch masked-attribute self-supervision on the **training split only**, identical loops
for every arm, **10 final seeds** (OGB's submission rule), test read once per seed on a frozen
configuration.

**Equal tuning budgets.** Phase 8H gave the LGM one free knob and the GNNs none. Here every arm tuned
width over three values on three validation seeds, then froze. Selected: LGM 128, LGM+RRWP 128,
GIN 300, GIN+RWSE 300, GCN 200, GCN+RWSE 200.

## Result

| model | Test ROC-AUC | Validation ROC-AUC | #Params |
|---|---|---|---|
| LGM | 0.7360 ± 0.0130 | 0.8342 ± 0.0064 | 819,729 |
| **LGM + RRWP** | 0.7431 ± 0.0196 | 0.8366 ± 0.0127 | 819,985 |
| **GIN** | 0.7720 ± 0.0148 | 0.8298 ± 0.0080 | 1,892,706 |
| **GIN + RWSE** | **0.7722 ± 0.0096** | **0.8389 ± 0.0087** | 1,897,506 |
| GCN | 0.7667 ± 0.0121 | 0.8261 ± 0.0062 | 256,801 |
| GCN + RWSE | 0.7637 ± 0.0200 | 0.8244 ± 0.0089 | 260,001 |

Paired per seed on test (n = 10):

| contrast | Δ | t | p | seeds won |
|---|---|---|---|---|
| LGM+RRWP − LGM | +0.0071 | +1.01 | 0.34 | 5/10 |
| GIN+RWSE − GIN | +0.0002 | +0.04 | 0.97 | 5/10 |
| GCN+RWSE − GCN | −0.0029 | −0.53 | 0.61 | 5/10 |
| **LGM+RRWP − GIN+RWSE** | **−0.0291** | **−3.64** | **0.0054** | 1/10 |
| **LGM − GIN** | **−0.0360** | **−4.83** | **0.0009** | 1/10 |

## What this says

**RRWP helps the LGM here too, but not significantly.** +0.0071 with p = 0.34 and 5/10 seeds. The
sign agrees with TCGA (+0.0280 there, p = 0.097), so across two unrelated benchmarks the encoding
moves the LGM the same way — but on molhiv the effect is inside the noise and must not be called a
result.

**RWSE is a clean null on test for both MPNNs** (+0.0002 and −0.0029). This is the same asymmetry
TCGA showed, now on a second benchmark: the encoding helps the transformer's dense attention block
and does nothing for message passing, which already propagates along the graph.

**It also reverses the sign of what validation predicted.** GIN+RWSE beat GIN by **+0.0091 on
validation** and by **+0.0002 on test**. The validation gain was entirely illusory. Anyone selecting
on validation here would have concluded RWSE helps GIN; it does not.

**The LGM still loses decisively.** −0.029 against GIN+RWSE (p = 0.005, 1 seed of 10) and −0.036
against plain GIN (p = 0.0009, 1/10). RRWP closes about 20 % of the gap and leaves the rest.

## The Phase-8H explanation for the LGM's test collapse is now falsified

8H attributed the LGM's poor test score to **size**: it was the biggest model (3.2 M) and had the
biggest validation→test drop, matching the leaderboard's own params↔test correlation of −0.70.
Phase 10 selected a width **3.9× smaller** and the drop got slightly *worse*:

| | params | validation | test | drop |
|---|---|---|---|---|
| 8H LGM | 3,212,321 | 0.8351 | 0.7446 | −0.0905 |
| **10 LGM** | **819,729** | 0.8342 | 0.7360 | **−0.0982** |
| 10 GIN | 1,892,706 | 0.8298 | 0.7720 | −0.0578 |
| 10 GCN | 256,801 | 0.8261 | 0.7667 | −0.0594 |

The LGM drops ≈ 0.09–0.10 at both 3.2 M and 0.82 M parameters, while GIN drops 0.058 at 1.9 M and
GCN 0.059 at 0.26 M — a 7× parameter range across the MPNNs with no change in the drop. **The
val→test gap is a property of the architecture, not of its capacity.** The LGM fits molhiv's
validation chemistry as well as any MPNN (it has the second-best validation of the six arms) and
transfers to unseen scaffolds substantially worse.

**This is the most important line in the table for the project's goal.** A general LGM must
generalise to graphs it has not seen; molhiv's scaffold split is exactly that test in miniature, and
the architecture is measurably weaker at it than message passing, at matched or smaller capacity.
That is evidence to weigh before scaling this backbone to many datasets.

## Against the published leaderboard

Our **GIN reaches 0.7720** against OGB's published GIN of **0.7558** at the same configuration — the
train-only self-supervision is worth **+0.016** and reproduces 8H's +0.022 independently. That
remains the most transferable finding from the molhiv work, and it is a procedure result, not an
architecture one.

| test | method |
|---|---|
| 0.8476 | Multi-RF Fusion + Multi-GNN *(rank 1)* |
| 0.8094 | CIN *(best pure-graph)* |
| **0.7722** | **ours — GIN + RWSE + train-only SSL** |
| **0.7720** | **ours — GIN + train-only SSL** |
| 0.7707 | GIN + virtual node *(rank 35)* |
| **0.7667** | **ours — GCN + train-only SSL** |
| 0.7606 | GCN *(rank 38, OGB reference)* |
| 0.7558 | GIN *(rank 40, OGB reference — the config we reproduce)* |
| 0.7549 | GCN in Julia *(rank 41, last)* |
| **0.7431** | **ours — LGM + RRWP** |
| **0.7360** | **ours — LGM** |

Our GIN/GCN arms would rank around 18th and 30th. **Neither LGM arm would rank** — 0.7431 is 0.012
below the last entry.

## Scope

- 10 seeds, official split and Evaluator, test read once per frozen configuration. Stages 1–4 cannot
  reach the test split; stage 5 is guarded by `--confirm-final`.
- Arms are **not** parameter-matched here (GIN 1.89 M, LGM 0.82 M, GCN 0.26 M) — that is OGB's
  protocol, each family at its own reference config, as in 8H. Note the LGM is now the *smaller*
  model and still loses, so the gap is not a capacity artefact in its favour or against it.
- RRWP vs RWSE is a systems comparison, not an information-matched one (see `docs/PLAN-PHASE10.md`).
- The k ≥ 1 RRWP/RWSE channels are normalised per graph to unit RMS; without it the encoding is
  numerically inert at TCGA scale, though on molecules (N ≈ 25) raw values would already be O(1).

## Operational note

Slurm held three jobs during this phase with `user env retrieval failed` / `launch failure limit
exceeded`. The latter is applied by Slurm as admin and is **not user-releasable** (`scontrol release`
returns *Access/permission denied*), so stage-4 task 4 had to be cancelled and resubmitted, and
stage 5 rebuilt with an explicit per-task dependency. A held job neither runs nor fails, so it raises
no ordinary alarm — any future long chain on Marlowe should watch for it explicitly.
