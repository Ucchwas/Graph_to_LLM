# Phase 10 — TCGA normal → tumour: RRWP helps the LGM, RWSE does nothing for the GCN, and the two end level

**Marlowe array 455868**, 11 folds, all COMPLETED, ~5 GPU-h, commit `2be6daf`. Leave-one-cancer-out
over 11 cancers × 5 seeds × 4 bodies = **220 runs**, every row stamped with that one commit. Rows:
`results/phase10/tcga/rows`, aggregate: `results/phase10/tcga/aggregate.md`.

Phase 9's protocol verbatim (`g2l/tcga9.py`): the same graphs, signed-ρ edge values, per-gene
expression features, training-only self-supervision with model-independent masks, cancer-disjoint
folds, `D1Bilinear` decoder, stratified loss, optimiser, schedule and budget. The only thing added is
the structural encoding, and only to the arm that names it. All four bodies are capacity-matched to
within **0.59 %**.

## Result

| body | changed-edge AUROC | changed-edge AP | overall AUROC | direction | #Params |
|---|---|---|---|---|---|
| LGM | 0.7034 ± 0.0208 | 0.2791 ± 0.0287 | 0.7463 ± 0.0191 | **0.4279 ± 0.0653** | 812,176 |
| **LGM + RRWP** | **0.7314 ± 0.0185** | **0.3086 ± 0.0321** | **0.7854 ± 0.0156** | 0.3888 ± 0.0525 | 812,432 |
| edge-aware GCN | 0.7284 ± 0.0153 | 0.3001 ± 0.0250 | 0.7771 ± 0.0225 | 0.3468 ± 0.0517 | 810,810 |
| edge-aware GCN + RWSE | 0.7261 ± 0.0171 | 0.2997 ± 0.0269 | 0.7728 ± 0.0211 | 0.3552 ± 0.0492 | 815,562 |

Paired over the 11 held-out cancers (seeds averaged within a cancer first):

| contrast | changed-edge AUROC | changed-edge AP | overall AUROC | direction |
|---|---|---|---|---|
| **LGM+RRWP − LGM** | **+0.0280 ± 0.0153**, p 0.097, 7/11 | **+0.0296 ± 0.0126, p 0.041**, 9/11 | **+0.0391 ± 0.0178**, p 0.053, 8/11 | −0.0391, p 0.26, 4/11 |
| GCN+RWSE − GCN | −0.0022 ± 0.0055, p 0.69, 6/11 | −0.0005, p 0.97, 6/11 | −0.0043, p 0.48, 6/11 | +0.0084, p 0.76, 5/11 |
| **LGM+RRWP − GCN+RWSE** | **+0.0053 ± 0.0159, p 0.75, 5/11** | +0.0090, p 0.47, 6/11 | +0.0126, p 0.51, 6/11 | +0.0336, p 0.40, 7/11 |
| LGM − GCN *(Phase 9 check)* | −0.0249 ± 0.0122, p 0.068, 3/11 | −0.0211, p 0.23, 5/11 | −0.0308, p 0.013, 2/11 | +0.0811, p 0.15, 8/11 |

Pairing every (cancer, seed), n = 55: LGM+RRWP − GCN+RWSE Δ +0.0053, t +0.51, p 0.61, 28/55.

## What this says

**RRWP is a real intervention on the LGM.** +0.028 changed-edge AUROC, +0.030 AP (p = 0.041, 9/11
cancers) and +0.039 overall AUROC, three of four metrics moving together. This is the first
intervention in the project to improve the LGM on a fair comparison.

**RWSE does nothing for the GCN** — −0.002, p 0.69, a clean null on every metric. The asymmetry is
the mechanistically expected one and is the most informative single fact here: the transformer's
dense node←node block had *no* relative structural information and gains from being given some,
while a message-passing network already propagates along the graph and gains nothing from the
node-level summary of the same walks. It also means the LGM's gain is not "the encoding is free
signal for anybody".

**But the headline is still a tie.** Best-equipped against best-equipped is +0.005 with SE 0.016 and
5 wins out of 11 — indistinguishable from a coin flip. RRWP closes the LGM's deficit against the
GCN; it does not produce a win.

**Phase 9's tie was underpowered, and the correction goes against the LGM.** The plain-arm contrast
moved from Phase 9's +0.0062 (p 0.72, 6/11, 3 seeds) to **−0.0249 (p 0.068, 3/11, 5 seeds)**. Phase
9's headline should be read as "no difference measurable at n = 3 seeds", not as parity.

**`direction` did not improve.** RRWP moved it −0.039 (p 0.26), and all four bodies remain below the
0.5 chance line (0.35–0.43), so every arm is still partially echoing its input — the pathology
Phase 7B identified. An encouraging single-seed value of 0.657 seen mid-run did **not** survive
aggregation; it was one draw of 55.

## The variance floor — measured, and it governs how these numbers must be read

The plain `lgm` and `edgegcn` arms are the same code, config and seeds as Phase 9, so they should
reproduce it. **They do not: 0 of 33 (cancer, seed) pairs match bit-exactly.**

| | mean abs. difference | median | max |
|---|---|---|---|
| `lgm`, Phase 9 vs Phase 10, identical seed | 0.0452 | 0.0306 | 0.2053 |
| `edgegcn`, identical seed | 0.0304 | 0.0163 | 0.1743 |

The cause is GPU non-determinism: `index_add_` / scatter atomics accumulate in arbitrary order, and
100 SSL epochs plus up to 300 fine-tuning epochs with early stopping amplify the difference into a
different stopping epoch. Nothing is wrong with the code — but **a seed does not pin a run on this
hardware.**

The consequence is quantitative: run-to-run spread at *fixed* seed (0.038 pooled) is the same size as
seed-to-seed spread within this phase (0.034–0.038). Each run is effectively an independent draw, so
5 seeds buys a genuine √5 reduction, and the reported SEs already carry this. It also explains why
Phase 9 at n = 3 could not resolve a ±0.006 effect, and why a per-fold difference of ±0.10 between
two arms means nothing on its own. Any future claim on this benchmark needs ≥ 5 seeds, or
`torch.use_deterministic_algorithms(True)` and the throughput cost that carries.

## Per cancer (changed-edge AUROC, mean over 5 seeds)

| held-out cancer | LGM | LGM + RRWP | edge-GCN | edge-GCN + RWSE |
|---|---|---|---|---|
| Breast Invasive Carcinoma | 0.8118 | 0.7916 | 0.8138 | 0.8253 |
| Head & Neck Squamous Cell Carcinoma | 0.7741 | 0.7670 | 0.8164 | 0.8186 |
| Lung Adenocarcinoma | 0.7673 | 0.8067 | 0.7587 | 0.7757 |
| Lung Squamous Cell Carcinoma | 0.7258 | 0.7785 | 0.7539 | 0.7430 |
| Prostate Adenocarcinoma | 0.7252 | 0.7530 | 0.7196 | 0.6987 |
| Kidney Clear Cell Carcinoma | 0.7180 | 0.6953 | 0.6888 | 0.7154 |
| Colon Adenocarcinoma | 0.6825 | 0.6370 | 0.6907 | 0.6967 |
| Liver Hepatocellular Carcinoma | 0.6599 | 0.6694 | 0.6875 | 0.6910 |
| Stomach Adenocarcinoma | 0.6530 | 0.7484 | 0.7168 | 0.6802 |
| Kidney Papillary Cell Carcinoma | 0.6501 | 0.7628 | 0.6778 | 0.6700 |
| Thyroid Carcinoma | 0.5699 | 0.6357 | 0.6879 | 0.6727 |

Per-cancer spread (0.57–0.83) still dwarfs every between-body difference, which is why the paired
per-cancer test, not the marginal means, is the reported statistic.

## Scope

- 11 graph pairs, 5 seeds, one density (ρ = 0.01). A direction, not a strong claim; n = 11 for the
  paired test.
- **RRWP vs RWSE is a systems comparison, not an information-matched one.** RRWP is pairwise and
  enters attention logits; RWSE is node-level and enters the input. That is how GRIT (ICML 2023) and
  the KDD'26 positional-encoding benchmark compare a transformer against MPNN baselines, and there is
  no node-level object that would make the two exactly equivalent. Reported as such.
- The k ≥ 1 RRWP/RWSE channels are normalised per graph to unit RMS. Without it the raw values on
  these graphs (N = 2000, mean degree 20) give a bias 4e-4 the size of the attention logits and the
  arm is inert — see `docs/PLAN-PHASE10.md`.
- No priors, no Phase-7 models, no scratch controls, no external features.
- Binarisation still discards correlation magnitude from the *target*. Phase 7's limitation #7 stands.
- Aggregate normal/tumour graphs per cancer; adjacent normal tissue carries field effects.
