# Phase 9 — TCGA normal → tumour translation: the LGM and an edge-aware GCN are statistically tied

**Marlowe array 455407**, 11 folds, all COMPLETED, ~1.7 GPU-h, commit `0da81ec`. Leave-one-cancer-out
over 11 cancers × 3 seeds × 2 bodies = 66 runs. Rows: `results/phase9/rows`, aggregate:
`results/phase9/aggregate.md`.

Everything is shared between the two bodies by construction (one code path, `g2l/tcga9.py`):
the same graphs, the same signed-ρ edge values, the same per-gene expression features, the same
training-only self-supervision with masks drawn from a model-independent generator, the same
cancer-disjoint folds, the same `D1Bilinear` decoder, the same stratified loss, the same optimiser,
schedule, seeds and budget. Capacity matched to **0.168 %**.

## Result

| body | changed-edge AUROC | changed-edge AP | overall AUROC | direction | #Params |
|---|---|---|---|---|---|
| **LGM** | **0.7196 ± 0.0213** | **0.2998 ± 0.0273** | 0.7543 ± 0.0253 | **0.4185 ± 0.0728** | 812,176 |
| edge-aware GCN | 0.7135 ± 0.0166 | 0.2837 ± 0.0275 | **0.7735 ± 0.0237** | 0.2917 ± 0.0576 | 810,810 |

Paired over the 11 held-out cancers (seeds averaged within a cancer first):

| metric | Δ (LGM − GCN) | t | p | wins |
|---|---|---|---|---|
| changed-edge AUROC | +0.0062 ± 0.0165 | +0.37 | **0.72** | 6/11 |
| changed-edge AP | +0.0161 ± 0.0202 | +0.80 | 0.44 | 6/11 |
| overall AUROC | −0.0193 ± 0.0180 | −1.07 | 0.31 | 4/11 |
| direction | +0.1268 ± 0.0604 | +2.10 | 0.062 | 8/11 |

Pairing every (cancer, seed), n = 33: changed-edge AUROC Δ +0.0062, t = +0.55, p = 0.59, 18/33.

**The headline is a tie.** +0.006 with a standard error of 0.017 and 6 wins out of 11 is
indistinguishable from a coin flip. This is the first fair comparison in the project the LGM has
not *lost* — but it has not won one either, and the honest statement is that on this task, at this
scale, the two architectures are equivalent.

## Per cancer (changed-edge AUROC)

| held-out cancer | LGM | edge-GCN | Δ |
|---|---|---|---|
| Breast Invasive Carcinoma | 0.8176 | 0.8275 | −0.0100 |
| Lung Adenocarcinoma | 0.8095 | 0.7632 | +0.0462 |
| Head & Neck Squamous Cell Carcinoma | 0.7642 | 0.7583 | +0.0060 |
| Lung Squamous Cell Carcinoma | 0.7384 | 0.7488 | −0.0104 |
| Kidney Clear Cell Carcinoma | 0.7368 | 0.6504 | **+0.0864** |
| Liver Hepatocellular Carcinoma | 0.7322 | 0.6888 | +0.0433 |
| Prostate Adenocarcinoma | 0.7231 | 0.6929 | +0.0302 |
| Kidney Papillary Cell Carcinoma | 0.6890 | 0.6423 | +0.0467 |
| Colon Adenocarcinoma | 0.6709 | 0.6921 | −0.0212 |
| Stomach Adenocarcinoma | 0.6682 | 0.6995 | −0.0314 |
| Thyroid Carcinoma | 0.5662 | 0.6844 | **−0.1182** |

Per-cancer spread (0.57–0.82) dwarfs the between-body difference, and one fold (Thyroid, −0.118)
accounts for most of the variance in the paired test. With 11 folds this benchmark cannot resolve
an effect of ±0.006.

## Two observations worth keeping

**`direction` is the one place the bodies separate.** Among cells that flip, `direction` asks
whether a gained edge outranks a lost one; chance is 0.5 and copying the input scores 0.0. The LGM
reaches 0.419 against the GCN's 0.292 (+0.127, p = 0.062, 8/11) — closer to chance but still below
it. **Both bodies are still partially echoing their input**, which is the pathology Phase 7B
identified; the stratified loss reduces it without removing it, and the LGM echoes less.

**The GCN wins the overall metric while losing direction.** Overall AUROC is dominated by the ~99 %
of cells that do not change, where copying the input is the correct answer — so a body that echoes
more scores better there. The two metrics disagreeing in exactly this way is consistent with that
reading, not with one body being better.

## Scope

- 11 graph pairs, 3 seeds. A direction, not a strong claim. The paired test has n = 11.
- No priors, no old models, no scratch controls, no ablations — this phase compares two
  architectures under identical treatment and reports only that.
- Binarisation still discards correlation magnitude from the *target* (the input now carries signed
  ρ as an edge value, the target does not). Phase 7's limitation #7 is unchanged.
- Aggregate normal/tumour graphs per cancer; adjacent normal tissue carries field effects.
