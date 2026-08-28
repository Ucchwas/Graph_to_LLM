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

---

# Phase 7B — fixing the objective, and what it revealed

Run after the MVP (job **452601**, 11 folds × 3 densities × 2 objectives = 66 runs, all
COMPLETED, ≈ 0.4 GPU-h, commit `9e55816`). Three additions, no architecture change.

## 7B.1 A stronger baseline the MVP missed

Diagnostics on the built graphs showed the input carries genuine cancer-specific signal (a
cancer's own normal graph predicts its own tumour graph better than another cancer's normal, by
**+0.036 overall AUROC at every density**), and that the useful way to use it is **anti-copying**:

`linear_prior = mean_tumour + λ·(A_normal − mean_normal)` with **negative** λ (flat plateau over
λ ∈ [−0.1, −0.6]; λ fitted leave-one-out on training cancers only) reaches **0.8348 ± 0.0185**
changed-edge AUROC at ρ = 0.01 — above mean-tumour's 0.8130 and above the MVP model's 0.7888.
Edges specific to a cancer's normal graph are preferentially *absent* from its tumour graph.
**The bar was therefore 0.835, not 0.813; the MVP write-up above understated it.**

## 7B.2 The objective fix worked on the mechanism, not the outcome

| arm (ρ = 0.01) | changed-edge AUROC | direction |
|---|---|---|
| linear_prior | **0.8348 ± 0.0185** | **0.9424** |
| mean_tumour | 0.8130 ± 0.0201 | 0.6672 |
| shared_weighted | 0.7939 ± 0.0209 | 0.5284 |
| shared (MVP, `full` loss) | 0.7888 ± 0.0221 | 0.3925 |
| shared_stratified | 0.7869 ± 0.0239 | 0.6004 |

The copying was cured exactly as predicted — direction rose from **0.393 (below chance)** to
0.600 (stratified) and 0.528 (weighted), +0.208 paired, p = 8.7e-04, 11/11 cancers. **But the
headline metric did not move**: stratified − MVP = −0.002 (p 0.84), weighted − MVP = +0.005
(p 0.53). Both still lose to the linear prior at every density (−0.041 to −0.068, p ≤ 0.012,
better on 0–2 of 11 cancers). Curing the symptom did not close the gap, so the objective was not
the binding constraint.

## 7B.3 The decisive check: the model *does* carry complementary signal

Combining the trained model's scores with the linear prior (both z-scored, **equal weights, no
fitted parameters**, prior's λ still leave-one-out — so this is leak-free):

| scorer (ρ = 0.01) | changed-edge AUROC |
|---|---|
| model alone | 0.7869 ± 0.0239 |
| linear prior alone | 0.8412 ± 0.0179 |
| **model + prior** | **0.8701 ± 0.0156** |

**+0.0290 over the prior alone (p = 5.2e-04, better on 11 / 11 cancers).** (A tuned mixing weight
reaches 0.874, but that weight was chosen on the test cancer, so only the equal-weight number is
trustworthy.)

So the GNN is learning real cancer-specific structure that a two-parameter rule cannot express —
it simply cannot express the *other* part. The most likely reason is architectural: the bilinear
decoder computes `Z W Zᵀ` with `Z = GCN(A_normal)` and has **no additive per-pair term**, so it
cannot represent a fixed input-independent matrix like the average tumour graph — which is the
single largest component of the target. The model is forced to choose between using the input and
reproducing the prior, and it cannot do both.

## What Phase 7B shows

- **The task has learnable, cancer-specific structure**, and our model finds some of it: adding it
  to the best simple baseline improves on that baseline for **every one of the 11 cancers**.
- **The MVP's copying was an objective artefact** and is fixable by a loss change alone (direction
  0.393 → 0.600), but that alone buys nothing on the headline metric.
- **Alone, the model still loses to a two-parameter rule** — an honest negative that no amount of
  loss engineering removed.
- **The binding constraint is most likely the decoder's lack of an additive bias term.** That is a
  concrete, minimal, testable architecture change (a learned per-pair bias, or training the model
  as a residual on an explicit prior) — deliberately *not* attempted here, since this phase was
  scoped to leave the architecture untouched.

---

# Phase 7C — the strict-protocol verification, and the freeze

The 7B.3 combination above used equal weights with no fitted mixing parameter, which is leak-free
but is not the protocol a reviewer will ask for. Phase 7C re-derives it with **every** free
parameter — λ *and* the mixing weight α — chosen on the fold's **validation cancer**, priors built
from the 9 training cancers only, and the test cancer scored exactly **once**. The saved Phase-7B
checkpoints are re-scored, not retrained (`g2l/verify7.py`, laptop CPU, ~8 min, no GPU).

## Result: the effect verifies, and is slightly larger under strict selection

| arm (ρ = 0.01, n = 11) | z-scored | rank-normalised |
|---|---|---|
| **model + prior, strict (λ, α) on validation** | **0.8686 ± 0.0154** | **0.8645 ± 0.0164** |
| model + prior, equal weights (the 7B.3 route) | 0.8670 ± 0.0158 | 0.8485 ± 0.0194 |
| linear prior alone (λ leave-one-out on train) | 0.8348 ± 0.0185 | 0.8348 ± 0.0185 |
| model alone | 0.7869 ± 0.0239 | 0.7869 ± 0.0239 |

| paired delta | z-scored | rank-normalised |
|---|---|---|
| **strict combo − prior** | **+0.0337 (p 1.0e-04, 11/11)** | **+0.0297 (p 1.3e-04, 11/11)** |
| strict combo − model | +0.0816 (p 1.3e-05, 11/11) | +0.0776 (p 9.7e-06, 11/11) |
| strict combo − equal-weight combo | +0.0015 (p 0.24, 5/11) | +0.0160 (p 0.0049, 11/11) |

**The Phase-7B claim survives.** Selecting the mixing weight honestly costs nothing — the strict
combination is marginally *better* than the equal-weight one, and beats the prior on **11 of 11**
cancers under both normalisers. Selected α ranges 0.1–0.6 across folds, so no fold wanted the model
alone or the prior alone.

**Two corrections to the numbers as published.**

1. The 7B.3 table quoted the prior at **0.8412**; the committed baseline path
   (`run_phase7.run_baselines` → `linear_prior`, 9 training cancers) gives **0.8348**, which is
   exactly the `linear_prior` row of the 7B.1 aggregate. The 7B.3 figure came from an ad-hoc
   analysis whose prior differed slightly from the committed code. Every number in this section is
   from the committed path. The direction and significance are unchanged; the gap over the prior is
   larger under the corrected baseline, not smaller.
2. The rank-normalised column is a genuine robustness check only with **average ranks for ties**.
   The prior is `mT + λ(A_n − mN)` with mT, mN means of 9 binary matrices, so it takes ~200 distinct
   values over 2M cells and is massively tied; ranking with a bare `argsort` breaks those ties in
   index order and injects noise at the magnitude of the signal. With that bug the check reported
   +0.0089 (p 0.15) and looked like a scale artefact; with average ranks it reports +0.0297
   (p 1.3e-04, 11/11) and agrees with the z-scored column.

**Standing caveat:** the validation cancer already drove that fold's early stopping, so selecting α
on it reuses validation. That is what validation is for and the test cancer is untouched, but it is
not a fresh selection set.

## Where Phase 7 ends

The architecture learns cancer-specific structure that a two-parameter rule cannot express — adding
it to the best simple baseline improves that baseline on every one of the 11 cancers, under two
independent normalisers and strict validation-only selection. **Alone it still loses to that rule**
(0.7869 vs 0.8348), and the most likely reason remains architectural: `Z W Zᵀ` has no additive
per-pair term, so it cannot represent the input-independent average-tumour component that dominates
the target. That was not fixed here — Phase 7 is frozen at this result and the decoder question
carries into Phase 8, where the task prior is an explicitly separated, task-specific term rather
than something the universal backbone must learn.

Full tables: `results/phase7/verify.md`, rows in `results/phase7/verify.json`.

## Gate 7

- [x] Gate 7.0 passed on all three probe cancers
- [x] `pytest tests/test_tcga.py` green (10 passed); all rows at one pinned code commit
- [x] identity exactly 0.5 on both changed-edge strata in every fold
- [x] changed-edge table with paired model − baseline deltas (n = 11)
- [x] Phase 7C: strict validation-only selection reproduces the combination result (11/11, both
      normalisers); `pytest tests/test_tcga.py` green; **phase frozen**
- [ ] VALIDATED line — user

VALIDATED ____
