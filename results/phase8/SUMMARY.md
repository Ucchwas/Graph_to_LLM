# Phase 8 summary — LGM vs baselines on `ogbg-molhiv`

Both tasks use the same dataset, `ogbg-molhiv` (41,127 molecular graphs, N = 4–222, 9 atom features,
3 bond features), but they are **different tasks with different splits and different status**:

- **Task A — link prediction** is our own masked-cell reconstruction protocol on those graphs. There
  is **no leaderboard** for it; the comparison is against baselines we built and ran ourselves.
- **Task B — graph classification** is the official OGB benchmark (HIV activity) on the official
  scaffold split with the official `Evaluator`, and **does** have a public leaderboard.

---

## Task A — link prediction (masked-cell reconstruction). No leaderboard.

Trained only on graphs with **N ≤ 30**; scored on held-out graphs at sizes seen and at sizes never
seen (N = 31–213). One seed. All learned arms are capacity-matched to ~3.2 M parameters.

| model | params | seen N ≤ 30 AUROC / AP | unseen N 31–213 AUROC / AP |
|---|---|---|---|
| **edge-aware GCN** (baseline) | 3,187,328 | **0.9480** / **0.7123** | **0.9422** / **0.5615** |
| LGM + sequential update | 3,229,728 | 0.9343 / 0.6444 | 0.9303 / 0.4696 |
| LGM | 3,229,728 | 0.9300 / 0.6328 | 0.9246 / 0.4544 |
| plain GCN — cannot read edge features | 3,238,400 | 0.8657 / 0.4302 | 0.8569 / 0.2724 |
| LGM without edge states (control) | 3,229,728 | 0.7948 / 0.2820 | 0.7927 / 0.1424 |
| negated degree (best non-learned) | — | 0.7697 / 0.2289 | 0.7705 / 0.1133 |
| random | — | 0.5010 / 0.1020 | 0.4862 / 0.0436 |
| common neighbours | — | 0.4426 / 0.1012 | 0.4757 / 0.0457 |

**Result: the LGM loses to the edge-aware GCN by −0.012 AUROC and −0.092 AP on unseen sizes**, with
more parameters. Both architectures transfer across size with essentially no decay, so size
independence is a property of the architecture class, not of ours.

The plain-GCN row is included because it was our *first* baseline and it flattered us by +0.065 —
`GCNConv` structurally cannot consume edge attributes. Once the baseline could see the bond features,
the sign of the comparison reversed.

---

## Task B — graph classification (official OGB benchmark, with leaderboard)

Official scaffold split, official Evaluator, ROC-AUC.

### Our runs

| model | seeds | validation | test | params |
|---|---|---|---|---|
| **GIN — reproduced in our pipeline** | 3 | **0.8315 ± 0.0044** | not run | 1,885,506 |
| **GCN — reproduced in our pipeline** | 3 | **0.8135 ± 0.0168** | not run | 528,001 |
| LGM, tuned (d=64, dropout 0.5) | 3 | 0.7525 ± 0.0410 | **not run — gate failed** | 212,425 |
| LGM, untuned (d=256, dropout 0) | 1 | 0.7002 | 0.6817 | 3,208,993 |
| LGM, no atom features (ablation) | 1 | 0.7076 | 0.6332 | 3,164,449 |

### Why that row has no test number

The Phase-8D protocol was: *tune LGM capacity and regularisation on validation only; if it matches
or beats the reproduced baseline, freeze it and read the test split once; otherwise stop.* Best
tuned LGM **0.7525 ± 0.0410** against reproduced GIN **0.8315 ± 0.0044** is **−0.079, about 18×
GIN's seed spread**. The gate failed, so the test split was never touched for that arm. "Gate
failed" is not a crash — it is the stopping rule doing its job.

The sweep itself (11 tasks × 3 seeds, job 453087) rules out the two obvious explanations:

| config | params | val ROC-AUC |
|---|---|---|
| d=64, dropout 0.5 | 212,425 | **0.7525 ± 0.0410** |
| d=128, dropout 0.0 | 818,065 | 0.7483 ± 0.0228 |
| d=256, dropout 0.3 | 3,208,993 | 0.7242 ± 0.0028 |
| d=256, dropout 0.0 *(the untuned config above)* | 3,208,993 | 0.6985 ± 0.0339 |

Regularisation was worth **+0.054**, which closes about 40 % of the gap and no more. Capacity runs
*monotonically the wrong way* for the "bigger model" hypothesis — 212k > 818k > 3.2M. With the
harness reproducing OGB's own baselines to 0.02 % on parameters, the residual −0.079 is the
architecture on this task, not the setup.

### Against the public leaderboard

Only the LGM arm using all nine atom features is comparable. Its test score is **0.6817**.

| rank | method | test ROC-AUC | params |
|---|---|---|---|
| 1 | Multi-RF Fusion + Multi-GNN | 0.8476 ± 0.0002 | 993,331,107 |
| 9 | Molecular FP + Random Forest | 0.8208 ± 0.0037 | **5,782** |
| 20 | GINE | 0.7921 ± 0.0128 | 33,217 |
| 21 | GIN | 0.7908 ± 0.0102 | **32,385** |
| 23 | GINE + HE | 0.7903 ± 0.0079 | 9,393 |
| 35 | GIN + virtual node | 0.7707 ± 0.0149 | 3,336,306 |
| 38 | GCN | 0.7606 ± 0.0097 | 527,701 |
| 41 (last) | GCN (in Julia) | 0.7549 ± 0.0163 | 527,701 |
| — | **ours (LGM, atom features)** | **0.6817** | 3,208,993 |

**We would not rank**: 0.6817 is below all 41 entries, 0.073 under last place. GIN reaches 0.7908
with **32,385 parameters** — 1/99th of ours.

Two caveats on reading this leaderboard. The entries above 0.82 are **cheminformatics, not graph
architectures** — rank 9 is a random forest on molecular fingerprints with 5,782 parameters, and
ranks 1–8 are fingerprint fusions; the best *pure* graph model is ~0.809. And our 0.6817 comes from
the untuned configuration; the tuned one is +0.017 better on validation but was never evaluated on
test because it failed the gate.

---

## Why the losses are not a harness artefact

Our reproductions of OGB's reference models, trained through **our own** loader, batcher, optimiser
loop and Evaluator, land within one standard deviation of the published numbers at parameter counts
matching to 0.02 %:

| model | ours (validation) | OGB published | params ours / theirs |
|---|---|---|---|
| GIN | 0.8315 ± 0.0044 | 0.8232 ± 0.0090 | 1,885,506 / 1,885,206 |
| GCN | 0.8135 ± 0.0168 | 0.8204 ± 0.0141 | 528,001 / 527,701 |

A published number cannot tell you your pipeline is broken; a reproduction can, and it isn't.
