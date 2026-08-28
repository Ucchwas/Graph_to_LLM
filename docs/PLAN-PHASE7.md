# Phase 7 MVP — normal graph → tumour graph translation on TCGA

Written 2026-08-28, approved before any row exists. This is the first phase aimed at the PI's
actual goal (Dr. Islam, 2026-08-27): **raw graph in → graph out, end to end.** Phases 1–6
established that the architecture trains and that saturated link-prediction benchmarks cannot
discriminate it; this phase changes the *task*, not the model.

## 1. The model — unchanged

```
A_normal [N,N] raw rows + edge_index = A_normal.nonzero()
  → shared GCNConv(N → 1024) → residual GCN block → LayerNorm → bilinear decoder
  → Â_tumour [N,N]
```

`GraphLLM.embed` builds the edge set from the matrix it is handed, so passing `A_normal` is what
enforces the leak rule. The tumour matrix appears only as a supervision target and a metric
input. No node features, no embeddings, no architecture change.

## 2. Data (public; `g2l/tcga.py`)

UCSC Toil recompute `TcgaTargetGtex_rsem_gene_tpm` (log2(TPM+0.001), Ensembl ids) with
`TcgaTargetGTEX_phenotype.txt`, from `https://toil.xenahubs.net/download/`. One uniform pipeline
across every TCGA cohort — which is what makes cross-cancer comparison defensible. It is
**uniformly processed, not batch corrected**: cohort, donor and source effects remain and no
claim is made otherwise.

**Vial letters are unavailable.** Every Xena TCGA hub (Toil and the per-cohort `HiSeqV2` files
alike, both checked) uses 15-character sample ids — participant plus sample-type code, with the
vial letter dropped. `11A` cannot be distinguished from `11B`. The operational definitions are
therefore `-11` (Solid Tissue Normal) and `-01` (Primary Tumor); Xena already carries one column
per (participant, sample type). Vial-level selection would require going to GDC directly under a
different pipeline, which this MVP does not do.

**Inclusion (verified 2026-08-28 from the phenotype table):** a cancer is included iff it has
≥ 30 `-11` normals. Eleven qualify — normal / tumour / matched participants:

| cancer | normal | tumour | matched |
|---|---|---|---|
| Breast Invasive Carcinoma | 113 | 1,092 | 112 |
| Kidney Clear Cell Carcinoma | 72 | 530 | 72 |
| Lung Adenocarcinoma | 59 | 513 | 58 |
| Thyroid Carcinoma | 59 | 504 | 59 |
| Prostate Adenocarcinoma | 52 | 495 | 52 |
| Lung Squamous Cell Carcinoma | 50 | 498 | 50 |
| Liver Hepatocellular Carcinoma | 50 | 369 | 50 |
| Head & Neck Squamous Cell Carcinoma | 44 | 518 | 43 |
| Colon Adenocarcinoma | 41 | 288 | 26 |
| Stomach Adenocarcinoma | 36 | 414 | 33 |
| Kidney Papillary Cell Carcinoma | 32 | 288 | 32 |

(13 cancers at a threshold of 20, 14 at 15.) **One aggregate normal graph and one aggregate
tumour graph per cancer — 11 graph pairs. Individual patients are never graph pairs.**

## 3. Graph construction

Per cancer, per condition, on one fixed gene universe:

1. **Matched sample counts** — tumour samples are subsampled (seeded) to that cancer's normal
   count. Without this the tumour graph is estimated from ~500 samples and the normal graph from
   ~50, and part of the normal↔tumour difference is estimation noise rather than biology.
   All `-11` normals are used; the matched-participant counts are reported but the aggregate
   graphs use every available normal, since more samples give a better correlation estimate.
2. **Spearman correlation** across samples (rank transform + Pearson; numpy/torch only — scipy's
   linalg DLL is blocked on the laptop).
3. **Binarise at matched density** — exactly `ρ·N(N−1)/2` edges by |correlation|, the same ρ in
   both conditions, selected by a stable descending sort rather than a threshold, because rank
   correlations over a few dozen samples tie often and a threshold would admit unequal edge
   counts. Equal edge counts make the task *rewiring*, and force |gained| = |lost|.
4. Symmetric, binary, zero diagonal — what the current decoder and `edge_index` expect.

ρ = 0.01 is the main protocol (mean degree ≈ 20); ρ = 0.005 and 0.02 are reported as sensitivity,
not selected on.

**Gene universe:** protein-coding-scale filter — median log2(TPM+0.001) > 1.0 across the pooled
**normal** samples — then the top 2,000 by variance. One set for all cancers. It uses normal
expression only, never tumour expression and never the normal→tumour contrast being predicted;
it does see the held-out cancer's normals, so the selection is **mildly transductive**. A
per-fold gene set is the strictly clean alternative and is recorded as a robustness check.

## 4. Gate 7.0 — is there signal above noise? (CPU, before any GPU work)

For the three cancers with the most normals: bootstrap the samples within each condition
(B = 20) and take the pairwise Jaccard among bootstrap graphs = within-condition instability.

**Pass iff the normal-vs-tumour Jaccard is below the 5th percentile of the within-condition
bootstrap Jaccard, in all three cancers.** A low normal–tumour Jaccard on its own is not enough;
the between-condition change must clearly exceed within-condition resampling noise. If it fails,
Phase 7 stops and that is the finding — no GPU time is spent.

## 5. Leave-one-cancer-out

11 folds. Per fold: 1 test cancer, 1 validation cancer (the next in a fixed rotation), 9 train.
One shared model per fold; a step is one training cancer's `(A_normal → A_tumour)` pair;
selection on the validation cancer's changed-edge AUROC; ≤ 300 epochs, patience 30. Because the
held-out unit is a whole cancer, no within-graph split is needed — the test cancer's `A_tumour`
is scored in full and was never supervised.

**Loss:** pos-weighted BCE over every strict-upper-triangle cell of the target (≈ 2 M cells at
N = 2,000; no negative sampling, no sampling variance). Masking is *not* used — the target is a
different matrix from the input, so there is no copy shortcut to suppress, and copying is exactly
what the identity baseline measures.

## 6. Baselines (held-out cancer; training cancers only feed the means)

| baseline | prediction | what it tests |
|---|---|---|
| identity | `A_normal` | the graphs barely change |
| mean-tumour | mean of the training cancers' `A_tumour` | the input is irrelevant |
| **mean-change** | `A_normal + mean(A_tumour − A_normal)` over training cancers | the rewiring is the same in every cancer — the baseline most likely to win |
| common neighbours | `A_normal²` | non-learned structure |

## 7. Metrics

- **Overall** — AUROC / AP over all upper-triangle cells of the held-out `A_tumour`, plus AP@1:1
  on an equal seeded draw of non-edges (the column every earlier phase carries).
- **Changed-edge (headline)** — restricted to cells that flip. *gained*: among cells absent in
  the normal graph, rank tumour-present above tumour-absent. *lost*: among cells present in the
  normal graph, rank tumour-present above tumour-absent. **Identity is exactly 0.5 in both
  strata**, so copying earns nothing. *direction*: over changed cells only, separate gained from
  lost — identity scores exactly 0.0, chance is 0.5.
- Reported per fold, then mean ± SE over the 11 cancers, with paired model − baseline deltas
  (n = 11 folds).

## 8. Leakage prevention

- `edge_index` and the model input come only from `A_normal`; a unit test spies on the body's
  forward and asserts the tumour matrix never reaches it.
- The test cancer contributes nothing to training, validation, mean-tumour or mean-change.
- Tumour expression never enters gene selection; the tumour subsample seed is fixed per cancer.
- Graph construction is per cancer, so no cross-cancer normalisation couples folds.

## 9. Runs

N = 2,000: dense adjacency 16 MB, first-layer weight 8 MB, ~164 MB per conv layer at ρ = 0.01,
logits 16 MB — under 4 GB, smaller than every Phase-5 run.

| stage | where | cost |
|---|---|---|
| build (cancer table, gene universe, 11 pairs × 3 densities) | laptop CPU | ~1 h |
| Gate 7.0 (3 cancers × 2 conditions × 20 bootstraps) | laptop CPU | ~10 min |
| baselines (4 × 11 folds × 3 densities) | laptop CPU | minutes |
| translate (11 folds × 3 densities) | Marlowe, array 0-10 | ≈ 3–5 GPU-h |

## 10. Gate 7

- [ ] Gate 7.0 passed on all three probe cancers (or the phase stops and reports why)
- [ ] `pytest tests/test_tcga.py` green; all rows at one commit
- [ ] identity exactly 0.5 on both changed-edge strata in every fold
- [ ] changed-edge table with paired model − baseline deltas (n = 11)
- [ ] `results/phase7/RESULTS.md`, VALIDATED line

## 11. Known limitations (stated up front, not discovered later)

1. **11 graph pairs is not foundation-scale.** Paired tests over 11 folds give a direction, not a
   strong claim. This is a small translation benchmark.
2. **Co-expression graphs from 32–113 samples are noisy** — Gate 7.0 exists because the
   normal↔tumour difference may not exceed that noise. The most likely failure mode.
3. **mean-change may be unbeatable.** If rewiring is shared across cancers, a two-line baseline
   captures it; if it is entirely cancer-specific, leave-one-cancer-out cannot transfer. The
   phase only succeeds in between, and there is no guarantee we are there.
4. **Adjacent normal is not normal** — `-11` tissue carries field effects.
5. **Aggregate graphs discard patient-level heterogeneity**; patient-specific inference is out of
   scope.
6. **Gene selection is mildly transductive** (uses held-out-cancer normals).
7. **Binarisation destroys magnitude** — correlation-strength change is the biology, and the
   current symmetric binary decoder cannot represent it. A weighted target is the obvious
   follow-up, not attempted here.
8. **Density matching removes any real densification signal** along with the confound.
