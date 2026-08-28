# Phase 7 MVP — normal graph → tumour graph translation: the model does not beat a mean-tumour prior on the edges that change

**Claim:** the architecture translates a normal co-expression graph into a tumour one far better
than copying the input (changed-edge AUROC **0.789 ± 0.022** vs identity's exact 0.500, +0.289,
p = 1.3e-07, 11/11 cancers) and better than applying the average normal→tumour change
(+0.107, p = 3.3e-04, 11/11) or common neighbours (+0.157, p = 1.1e-03, 11/11). **But it does not
beat the simplest strong prior: the average tumour graph of the training cancers, which ignores
the input entirely** (mean-tumour 0.813 ± 0.020; paired **−0.024, p = 0.13, better on only
3 / 11 cancers**). The sign is the same at every density tested (−0.024 / −0.029 / −0.029). The
model *is* the best predictor of the tumour graph overall (AUROC 0.837 vs mean-tumour's 0.814,
identity's 0.627), so it has learned real structure — but not the cancer-specific rewiring the
phase set out to capture, which is the part that would justify a graph-to-graph model.

Produced on Marlowe (H100 80 GB, 1 GPU per fold), array **452552** (11 folds × 3 densities, all
COMPLETED, 1–3 min each, **≈ 0.3 GPU-h**), plus CPU baselines and Gate 7.0 on the laptop.
165 rows at commits `f75392d` (baselines, gate) and `8297511` (translation code).
Aggregation: `results/phase7/aggregate.md`.

## Setup (full detail in `docs/PLAN-PHASE7.md`)

Architecture unchanged from Phases 5–6: raw normal adjacency rows and the **normal** graph's
`edge_index` → shared `GCNConv(2000→1024)` → residual GCN block → bilinear decoder → predicted
tumour adjacency. The tumour matrix is only ever a supervision target and a metric input.

Data: UCSC Toil `TcgaTargetGtex_rsem_gene_tpm` (log2(TPM+0.001); sha256 `a8c36cb1…`), phenotype
table `ba4d4461…`. **11 cancers** have ≥ 30 `-11` solid-tissue normals (BRCA 113, KIRC 72,
THCA 59, LUAD 59, PRAD 52, LUSC 50, LIHC 50, HNSC 44, COAD 41, STAD 36, KIRP 32). One aggregate
normal graph and one aggregate tumour graph per cancer — **11 graph pairs**; individual patients
are never graph pairs. Fixed 2,000-gene universe from normal samples only; tumour samples
subsampled to each cancer's normal count; Spearman correlation binarised at matched density
(exactly 19,990 edges per graph at ρ = 0.01, so |gained| = |lost| in every cancer).
Leave-one-cancer-out: 9 train / 1 val / 1 test, selection on the validation cancer's changed-edge
AUROC, full-matrix pos-weighted BCE.

**Vial letters are not available.** Every Xena TCGA hub (Toil and per-cohort `HiSeqV2`, both
checked) uses 15-character sample ids with the vial letter dropped, so `11A` cannot be separated
from `11B`; `-11` and `-01` are the operational definitions.

## Gate 7.0 — passed, and worth reading

| cancer | bootstrap Jaccard normal (p5) | tumour (p5) | normal vs tumour |
|---|---|---|---|
| Breast Invasive Carcinoma | 0.466 (0.397) | 0.546 (0.486) | **0.091** |
| Kidney Clear Cell Carcinoma | 0.413 (0.342) | 0.406 (0.347) | **0.130** |
| Thyroid Carcinoma | 0.511 (0.434) | 0.407 (0.337) | **0.323** |

The normal↔tumour difference is far larger than within-condition resampling noise in all three
probes, so the task is not measuring noise. But note the absolute stability: bootstrapping the
same samples reproduces only 41–55 % of edges, so **individual edges are unreliable** and only
systematic, population-level differences are interpretable.

## Results — main protocol (ρ = 0.01, mean ± SE over the 11 held-out cancers)

| arm | changed-edge AUROC | gained | lost | direction | overall AUROC | overall AP@1:1 |
|---|---|---|---|---|---|---|
| **mean-tumour** (ignores the input) | **0.8130 ± 0.0201** | 0.7726 | 0.8534 | **0.6672** | 0.8143 | 0.8130 |
| **shared model** | 0.7888 ± 0.0221 | 0.7868 | 0.7908 | 0.3925 | **0.8367** | **0.8577** |
| mean-change | 0.6815 ± 0.0191 | 0.6618 | 0.7012 | 0.0004 | 0.7502 | 0.7776 |
| common neighbours | 0.6321 ± 0.0244 | 0.6402 | 0.6240 | 0.0655 | 0.7273 | 0.7256 |
| identity | **0.5000 ± 0.0000** | 0.5000 | 0.5000 | 0.0000 | 0.6272 | 0.6237 |

Paired shared − baseline, n = 11 cancers:

| vs | changed-edge | direction | verdict |
|---|---|---|---|
| identity | **+0.2888** (p 1.3e-07, 11/11) | +0.393 | decisive win |
| mean-change | **+0.1073** (p 3.3e-04, 11/11) | +0.392 | decisive win |
| common neighbours | **+0.1567** (p 1.1e-03, 11/11) | +0.327 | decisive win |
| **mean-tumour** | **−0.0242** (p 0.13, 3/11) | **−0.2747** (p 1.1e-04, 0/11) | **loss** |

Sensitivity: the shared − mean-tumour gap is −0.028 (p 0.17) at ρ = 0.005 and −0.029 (p 0.072)
at ρ = 0.02. The model never wins on changed edges at any density.

### The diagnostic: `direction`

Among cells that flip, `direction` asks whether a gained edge outranks a lost one. Identity
scores exactly **0.000** (it always ranks lost above gained); chance is 0.5; mean-tumour reaches
**0.667**. The shared model scores **0.393 — below chance**, and below chance in 8 of 11 cancers.
It is partially echoing its input: the pull toward copying helps the overall metric (the two
graphs share ~13 % of edges and the vast majority of *non*-edges) and hurts precisely where the
biology lives. That is the mechanism behind the loss to mean-tumour, and it is a property of the
objective, not of the data: a full-matrix loss is dominated by the ~99 % of cells that do not
change, where copying is the right answer.

### Per-cancer (shared model, ρ = 0.01)

| held-out cancer | changed AUROC | direction | overall AUROC |
|---|---|---|---|
| Head & Neck Squamous | 0.886 | 0.687 | 0.900 |
| Breast Invasive | 0.876 | 0.624 | 0.841 |
| Kidney Clear Cell | 0.848 | 0.471 | 0.788 |
| Lung Adenocarcinoma | 0.833 | 0.536 | 0.852 |
| Lung Squamous | 0.814 | 0.448 | 0.798 |
| Kidney Papillary | 0.788 | 0.366 | 0.721 |
| Stomach Adenocarcinoma | 0.786 | 0.323 | 0.925 |
| Prostate Adenocarcinoma | 0.757 | 0.288 | 0.888 |
| Liver Hepatocellular | 0.746 | 0.361 | 0.836 |
| Thyroid Carcinoma | 0.686 | 0.105 | 0.819 |
| Colon Adenocarcinoma | 0.657 | 0.109 | 0.836 |

Spread is wide (0.657–0.886) and `direction` tracks it closely: the cancers where the model does
well are the ones where it echoes least.

## What this shows / does not show

- **Shows the task is real and the harness is sound.** Gate 7.0 separates signal from resampling
  noise; identity is exactly 0.5 on both changed-edge strata and exactly 0.0 on direction in all
  11 folds; edge counts are identical across conditions by construction.
- **Shows the model learns transferable tumour-graph structure.** It is the best arm on the
  overall metric (0.837 AUROC) and beats identity, mean-change and common neighbours decisively
  on changed edges. Trained on 9 cancers, it generalises to a cancer it has never seen.
- **Does not show cancer-specific rewiring.** A prior that discards the input entirely predicts
  the changed edges better. Whatever the model extracts from the specific normal graph does not,
  on net, beat knowing what tumour graphs look like in general.
- **Does not support a graph-to-graph claim yet.** The point of translation is that the *input*
  determines the output; here it mostly does not, and where it does it pulls toward copying.
- **Is a small benchmark.** 11 graph pairs, one seed, paired tests over 11 folds. A direction, not
  a strong claim.
- **Coarsens the biology.** Binary symmetric graphs discard correlation magnitude, which is where
  much of the normal→tumour signal is; density matching removes any real densification.
- **Adjacent normal is not normal** (field effects), and graphs from 32–113 samples reproduce only
  ~half their edges under bootstrap.

## The obvious next experiment (not run, not approved)

The failure is localised and testable: the objective is dominated by unchanged cells. Two cheap
variants, same architecture, same data, ~0.5 GPU-h:

1. **Changed-cell-weighted loss** — upweight cells that differ between conditions, so training
   optimises what the headline metric measures.
2. **Residual/delta head** — predict `A_tumour − A_normal` rather than `A_tumour`, making copying
   the zero solution instead of the easy optimum.

Either would test whether the model *can* capture cancer-specific rewiring, or whether the
information simply is not there in 11 aggregate graph pairs.

## Gate 7

- [x] Gate 7.0 passed on all three probe cancers
- [x] `pytest tests/test_tcga.py` green (10 passed); all rows at one pinned code commit
- [x] identity exactly 0.5 on both changed-edge strata in every fold
- [x] changed-edge table with paired model − baseline deltas (n = 11)
- [ ] VALIDATED line — user

VALIDATED ____
