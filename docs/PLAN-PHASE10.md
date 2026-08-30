# Phase 10 — the RRWP study: does the missing ingredient change the answer?

**Status: DRAFT, awaiting approval. No code written, no jobs submitted.**

## 0. Why this phase exists

Phases 8H and 9 compared the LGM against edge-aware GIN/GCN under identical treatment and found a
loss (molhiv, −0.033 test, 0/10 seeds) and a tie (TCGA, +0.006, p 0.72). Both comparisons were fair
and the harness is verified. What they did *not* test is the one ingredient every competitive
message-passing-free graph transformer has and ours lacks: **a relative positional encoding on the
dense node←node attention.** GRIT (Ma et al., ICML 2023) is the reference model of our class; its
ablation shows the relative random-walk encoding (RRWP) is load-bearing (ZINC MAE 0.059 → 0.081 with
node-only RWSE, → 0.117 with Graphormer's attention), and the KDD'26 positional-encoding benchmark
(Grötschla et al., 500+ configurations) finds RRWP "consistently achieves the best results".
CLAUDE.md §9 prescribes RRWP at K = 16 and §11 says the structural blind spot should have been
covered early. `lgm.py` today: node input is degree only; the dense block has zero relative
structural information.

This phase adds exactly that, to exactly that block, and reruns both benchmarks from scratch with
the baselines given the standard node-level counterpart (RWSE). Six arms on molhiv, four on TCGA.
The result is reported whichever way it goes.

**Pre-registered expectation.** On molhiv the literature says an RRWP transformer lands *at* GIN's
level, not above it (independent reproductions: GRIT 0.7707, GPS 0.7662, GIN-GUMP 0.7789; our
train-only-SSL GIN is 0.7776). The informative molhiv number is therefore LGM+RRWP − LGM: does the
missing ingredient close the −0.033 gap. TCGA is open: the LGM's global block gains structural gene
identity and the GCN gains node-level RWSE; the literature has no precedent for this task.

## 1. What is added — and what is not

### The LGM (unchanged core)

```
RawGraph(n, edge_index, edge_value, batch, x)
  → NodeEdgeProjection   node states [N,d] + edge states [M_u,d]      (unchanged)
  → L × ISETLayer        node←node dense  ⊕  node←edge / edge←node incidence, one shared softmax
                         ┗━ NEW: + bias_h(i,j) on the DENSE node←node logits only
  → task head            D1Bilinear (TCGA) / [mean nodes ‖ mean edges] → linear (molhiv)   (unchanged)
```

**RRWP.** From the edge set the model is handed, inside `forward`, nothing else:

```
A     = adjacency of g.edge_index (block-diagonal across a batch)      [N, N]
M     = D⁻¹ A                                                          row-stochastic
RRWP  = [I, M, M², …, M¹⁵]                                              [N, N, 16]   K = 16 fixed
bias  = Linear_l(16 → heads)(RRWP)                                      [N, N, heads], one per layer
lnn   = (q_i · k_j)/√d_h + bias                                         before the shared max
```

- Enters both code paths of `IncidenceAttention` (`forward` and the sequential `node_from_all`).
- Never touches node←edge, edge←node, edge states, the FFN, the projections, or any head.
- **Zero-initialised.** At initialisation LGM+RRWP is bit-identical to LGM at the same seed; the
  arm is a strict superset and the ablation is exact. Asserted by a test.
- Asymmetric, as in GRIT (`M^k[i,j] ≠ M^k[j,i]`; detailed balance makes the asymmetry a degree
  ratio). Output symmetry comes from the decoder and is asserted unchanged.

**Channel normalisation — a correction made during implementation, and the single most important
detail in this phase.** A k-step probability spreads over ~deg^k nodes, so a raw entry of `M^k` is
~deg^-k and the mean over the matrix is ~1/N. Measured on the real TCGA graphs (N = 2000, mean
degree 20) after 20 SSL epochs: the raw bias reached **6.7e-4 against attention logits of 1.64 — a
ratio of 4e-4.** The arm would have been numerically inert however long it trained, because a
zero-initialised weight would have to reach ~1e3 to compete, and the phase would have returned a
null result for the wrong reason. Published RRWP results are on molecules (N ≈ 25, degree ≈ 2)
where the raw values are already O(1) and the question never arises. So **every k ≥ 1 channel is
divided by its own root-mean-square over its own graph**; channel 0 is the identity and is already
O(1), so it is left alone. The same normalisation is applied to RWSE, so neither family is
advantaged by scaling. The statistic is permutation-invariant (a scalar per graph, so equivariance
is untouched), computed per graph and never per batch (no cross-graph coupling), and parameter-free
and computed at run time (size independence untouched) — all three asserted in `tests/test_rrwp.py`.
After the fix the same measurement gives **bias 8.8e-2 against logits 1.64, a ratio of 0.054** at 20
epochs, and it grows with training.
- Block-diagonal `M` makes `M^k` block-diagonal, so batching needs no per-graph loop and the
  cross-graph mask already in place is sufficient.
- Parameters added: `layers × (16 × heads + heads)` — 272 at the TCGA config. State-dict shapes
  are N-independent.

**Leakage rule.** RRWP is a function of `g.edge_index` *only*, computed in the forward pass. Whatever
edge set reaches the model is the edge set RRWP sees. On TCGA self-supervision, `ssl_mask` removes
hidden edges in both directions before the model is called, so RRWP of a hidden pair is `M[i,j] = 0`
(asserted). On molhiv, masking hides atom/bond *attributes*, never topology, so RRWP is identical
between the masked and unmasked batch and there is nothing to leak (asserted, to document it).

### The baselines (standard counterpart)

RWSE — the node-level part of the same object: `RWSE[i,k] = M^k[i,i]`, k = 1…16, → `Linear(16 → d)`,
zero-initialised, **added to the node input**:

- OGB GIN / GCN (`ogbnet.OGBGNN`): `h = AtomEncoder(x) + W_rwse · RWSE`. Everything else is OGB's
  reference architecture verbatim.
- TCGA edge-aware GCN (`EdgeGCNBaseline`): the same term inside its `NodeEdgeProjection`.

At initialisation GIN+RWSE ≡ GIN, GCN+RWSE ≡ GCN (asserted).

**This is a systems comparison, not an information-matched one.** RRWP is pairwise and lives in
attention logits; RWSE is node-level and lives in the input. That is how GRIT and the KDD'26
benchmark compare a transformer against MPNN baselines, and it is the accepted protocol — but it is
stated here so the write-up does not overclaim. Note also that GRIT additionally feeds the RRWP
diagonal into its node features; we deliberately do **not** (the LGM's only structural addition is
the bias), which if anything under-equips the LGM relative to GRIT.

### Not in this phase

No text, fingerprints, node-ID tables, external embeddings, pretrained encoders, GNN front-end, LLM.
No priors, Phase-7 models, scratch controls, extra ablations. K is fixed, not tuned. No RRWP
"update" / pair-state stream, no degree scaler, no edge←edge block — GRIT's remaining machinery is
out of scope. No change to the incidence layers, heads, decoder, losses, SSL objective, optimiser,
schedules, folds or evaluators.

## 2. Tests — written and green before anything else

New file `tests/test_rrwp.py`; existing test files extended where a parametrisation already exists.

| # | test | asserts |
|---|---|---|
| 1 | RRWP definition | equals explicit dense powers of `D⁻¹A`; channel 0 is `I`; rows of `M^k` sum to 1 off isolated nodes; RWSE equals the diagonal |
| 2 | RRWP equivariance | `RRWP(πG) = π RRWP(G) πᵀ` exactly |
| 3 | **leakage (TCGA)** | on a `ssl_mask`ed graph the model's RRWP equals RRWP of the masked adjacency, differs from the full one, and `M[i,j] = 0` for every hidden pair |
| 4 | leakage (molhiv) | masked and unmasked batches yield identical RRWP — attributes are masked, topology is not |
| 5 | **zero-init equivalence** | LGM+RRWP ≡ LGM, GIN+RWSE ≡ GIN, GCN+RWSE ≡ GCN, edgeGCN+RWSE ≡ edgeGCN at the same seed, `allclose` on outputs |
| 6 | strict superset | base state-dict loads into the +arm with only the new keys missing; new keys have N-free shapes |
| 7 | permutation equivariance | `test_phase9` parametrised over all four TCGA bodies (float64, 1e-9); molecule-batch equivariance in `test_equivariance.py` with RRWP on |
| 8 | symmetry | logits symmetric; endpoint-swap invariance of edge states unchanged; `edge_index` column shuffle leaves outputs unchanged |
| 9 | size independence | `test_size_independence.py` with RRWP on: strict cross-seed load, N ∈ {7, 23, 101, d}, state-dict shapes unchanged after every forward |
| 10 | batching | three graphs batched reproduce three single-graph runs with RRWP on (block-diagonality) |
| 11 | capacity (TCGA) | all four arms pairwise within 2 % at the real config |
| 12 | **memory at N = 2000** | laptop RTX 5070 Ti (12 GB): forward + backward, Phase-9 config, 2000 nodes / 20 k edges, with and without RRWP; peak recorded, delta asserted < 1 GB. **Measured: 1187 → 1428 MiB, delta 241 MiB**, matching the predicted `[2000,2000,16]` fp32 cost. Skipped without CUDA |
| 12b | normalisation | each k ≥ 1 channel has RMS exactly 1; scale independent of N (20/200/1000) and degree; a pure positive rescale that preserves zeros and sign; per graph, not per batch |
| 13 | launcher | refuses on a missing selection row; applies the rule per arm; dry-run lists six arms |
| 14 | smoke | every stage of both runners end-to-end on CPU with `--limit` / `--limit-genes` |

Plus the existing suite (168 tests as of `34e2580`; 12 unrelated modules fail at collection on the
laptop because of the scipy DLL block and are run on Marlowe's import check instead).

## 3. MolHIV — official classification, six arms

Protocol: **Phase 8H's, verbatim**, with three arms added and one rule generalised. Official scaffold
split, official `Evaluator`, all 9 atom and 3 bond attribute columns, `configs/phase8h.yaml`
hyper-parameters (100-epoch train-only masked-attribute SSL, batch 32, lr 1e-3, warm-up 1000 steps,
clip 1.0, patience 20, dropout 0.5, LGM linear head + node‖edge readout).

| arm | body | structural input | width grid (selection knob) |
|---|---|---|---|
| `lgm` | ISET, 4 layers | degree only | d ∈ {64, 128, 256} |
| `lgm_rrwp` | ISET, 4 layers | + RRWP bias | d ∈ {64, 128, 256} |
| `gin` | OGB GIN, 5 layers | — | d ∈ {100, 200, 300} |
| `gin_rwse` | OGB GIN, 5 layers | + RWSE | d ∈ {100, 200, 300} |
| `gcn` | OGB GCN, 5 layers | — | d ∈ {100, 200, 300} |
| `gcn_rwse` | OGB GCN, 5 layers | + RWSE | d ∈ {100, 200, 300} |

**Equal tuning budgets.** Every arm has exactly one knob (width), three values, three selection
seeds, on validation only. 8H gave the GNNs no knob; the GNN grids here put OGB's reference width
(300) at the top so the 8H configuration stays reachable, and the parameter spans align
(LGM 0.21 / 0.82 / 3.2 M; GIN 0.23 / 0.85 / 1.9 M; GCN 0.07 / 0.25 / 0.54 M).

**Selection rule — pre-registered, executed by the launcher, no human in the loop:** per arm,
width = argmax of the validation mean over seeds {0, 1, 2}. This is 8H's rule, kept so the three
old arms replicate 8H directly (expected: LGM d=256 ≈ 0.7446, GIN ≈ 0.7776, GCN ≈ 0.7625). 8H
documented that this rule favours the largest width and that size predicts a worse test score on
this split; the bias applies identically to all six arms. *(Alternative for approval: the
one-standard-error rule — smallest width within one SE of the best. Not recommended: with 3 seeds it
degenerates to "always the smallest", and it breaks the 8H anchor.)*

**Test discipline.** Stages 1–4 cannot reach the test split. Stage 5 trains the frozen
configuration, restores the best-validation checkpoint, and reads test once per seed, guarded by
`--confirm-final`. 10 final seeds {0…9}, mean ± unbiased std (OGB's rule).

**Inference parameter count** = `sum(p.numel())` of the fine-tuned classifier at test time — body,
readout, head. SSL prediction heads excluded (they are discarded). The widened mask-token rows the
pretrained checkpoint carries are included, as in 8H.

### Stages (Slurm, Marlowe, one chain)

Every stage is **one task per arm** (6 tasks), each looping that arm's three widths. Per-run costs
are Phase 8H's measured wallclocks: SSL 0.63–0.70 h (LGM) / 0.26–0.28 h (GIN, GCN); fine-tune
0.13–0.30 h (LGM) / 0.08–0.10 h (GNN).

| stage | file | tasks | per task (worst = LGM) | what |
|---|---|---|---|---|
| 1 | `phase10m_pretrain_select.sbatch` | 6 | 9 SSL runs, ≈ 6.5 h | selection-seed checkpoints |
| — | `phase10m_chain.sbatch` | 1 | — | `afterok:1`; submits 2 and 3 once stage 1 has drained |
| 2 | `phase10m_select.sbatch` | 6 | 9 fine-tunes, ≈ 2.7 h | validation rows |
| 3 | `phase10m_launch.sbatch` | 1 | `launch10 --step select` | reads 18 rows, writes `winners.json`, submits 4 + 5 |
| 4 | `phase10m_pretrain_rest.sbatch` | 6 | 7 SSL runs, ≈ 4.9 h | remaining checkpoints |
| 5 | `phase10m_final.sbatch` | 6 | 10 fine-tunes + one test read each, ≈ 3 h | the table, `afterok:4` |

≈ 96 SSL runs + 114 fine-tunes ≈ **40–45 GPU-h** (8H: 14 GPU-h for 36 + 45), wall-clock ≈ 7 h of
compute plus queue time. RRWP's own cost is negligible at both scales (≈ 1 min of matmul over a
100-epoch molhiv SSL run; ≈ 15 s over a TCGA one).

### Report — `results/phase10/molhiv/RESULTS.md`

One table, three columns, six rows: **Test ROC-AUC · Validation ROC-AUC · #Params**, each
mean ± std over 10 seeds. Below it, the pre-registered paired per-seed contrasts on test (Δ, seeds
won, t):

1. `lgm_rrwp − lgm` — does RRWP help the LGM (primary molhiv question)
2. `gin_rwse − gin`, `gcn_rwse − gcn` — does RWSE help the baselines
3. `lgm_rrwp − gin_rwse` — best-equipped system vs best-equipped system
4. `lgm − gin` — 8H replication check

## 4. TCGA — normal → tumour, four arms

Protocol: **Phase 9's, verbatim** (`g2l/tcga9.py`), with two bodies added and seeds extended.
Fixed 2,000-gene universe, ρ = 0.01 binarised Spearman graphs with signed-ρ edge values, per-gene
expression mean/std as `x` (x_dim 2), 11 leave-one-cancer-out folds (test / val = next in rotation /
9 train), 100-epoch train-only edge-masking SSL with model-independent masks, stratified
translation loss, selection by early stopping on the validation cancer, test cancer scored once.

| arm | body | structural input | params |
|---|---|---|---|
| `lgm` | ISET d=128, 4 layers, 4 heads | degree | 812,176 |
| `lgm_rrwp` | same | + RRWP bias | 812,432 |
| `edgegcn` | edge-aware GCN w=297, 4 layers | degree | 810,810 |
| `edgegcn_rwse` | same | + RWSE | 815,562 |

Measured worst pairwise gap **0.586 %**.

Widths are **fixed at Phase 9's** for all four arms so within-body contrasts differ in nothing but
the structural term; all pairwise parameter gaps < 2 % (test 11). No width knob on TCGA, exactly as
in Phase 9.

**Seeds {0, 1, 2, 3, 4}** — all four arms rerun from scratch at the Phase-10 commit; Phase 9's rows
are not reused. 11 folds × 5 seeds × 4 arms = **220 runs**, one Slurm array of 11 tasks (one held-out
cancer per task, all seeds and arms inside), ≈ 45 min per task, ≈ **8 GPU-h**.

### Report — `results/phase10/tcga/RESULTS.md`

Four rows: **changed-edge AUROC · changed-edge AP · overall AUROC · direction · #Params**, mean ± SE
over the 11 held-out cancers (seeds averaged within a cancer first). Paired per-cancer table
(Δ ± SE, t, p, wins/11) for the four contrasts:

1. `lgm_rrwp − lgm`
2. `edgegcn_rwse − edgegcn`
3. `lgm_rrwp − edgegcn_rwse` — best-equipped vs best-equipped (primary TCGA question)
4. `lgm − edgegcn` — Phase 9 replication check

Secondary: pairing every (cancer, seed), n = 55. Per-cancer table for the primary contrast.

## 5. Queue plan (the 32-submitted-jobs account cap is shared)

QOS `medium`: `MaxSubmitJobsPerAccount` = 32, shared with teammates, and **array tasks count
individually** — measured, not assumed (`sacctmgr show qos`). Wave one is TCGA (11) + molhiv stage 1
(6) + the chain job (1) = **18**; the chain then adds stage 2 (6) + stage 3 (1); stage 3 adds
stages 4 (6) + 5 (6). Peak footprint 18, leaving ample room for the ~5 jobs teammates typically hold.
All jobs carry `G2L_COMMIT=<hash>`; the Marlowe checkout is pinned to that hash for the whole phase.

## 6. Files

**New:** `g2l/rrwp.py` (RRWP / RWSE from `edge_index`), `g2l/run_phase10.py` (molhiv runner: `--arm
--stage --d --seeds --confirm-final`), `g2l/launch10.py`, `g2l/aggregate10.py` (both benchmarks),
`configs/phase10_molhiv.yaml`, `configs/phase10_tcga.yaml`, six `slurm/phase10*.sbatch`,
`tests/test_rrwp.py`, this plan, two `RESULTS.md`.

**Modified (additive flags only, every existing arm bit-identical at the defaults):** `g2l/lgm.py`
(`rrwp_k` on `IncidenceAttention` / `ISETLayer` / `ISETBody` / `LGM`; `rwse_k` on
`NodeEdgeProjection`), `g2l/ogbnet.py` (`rwse_k` on `OGBGNN`), `g2l/multigraph.py` (`rwse_k` on
`EdgeGCNBaseline`), `g2l/molclass.py` (`rrwp_k` pass-through on `LGMClassifier`), `g2l/tcga9.py`
(`build_model` gains `lgm_rrwp` / `edgegcn_rwse`), `g2l/run_phase9.py` (`--bodies` reads the config).
Phase 8H's and Phase 9's runners, configs, rows and results are not touched.

## 7. Execution order

1. Tests (§2) written first; whole suite green on the laptop; import check on Marlowe.
2. Laptop CPU smoke of every stage of both runners.
3. **One commit** — code, tests, configs, sbatch, plan. No Claude trailer. Push. Hash recorded here
   and in every row.
4. Ship `tcga9.pt` is already staged on Marlowe; verify. Submit TCGA array, then molhiv stages 1–3
   with `--dependency=afterok` chaining. Record job IDs here.
5. Monitor to completion; any failed task is inspected, never papered over (the launcher refuses
   on a missing row).
6. Aggregate, write the two `RESULTS.md`, one results commit. **Stop and report.**

## 8. Decisions needing your approval

1. Selection rule on molhiv: **argmax validation mean over 3 seeds** (8H's rule; recommended) — or
   one-SE.
2. Equal budgets via a width knob for the GNNs: **{100, 200, 300}** with OGB's 300 at the top.
3. TCGA: widths fixed at Phase 9's; **5 seeds**; all four arms rerun from scratch.
4. The LGM's *only* structural addition is the bias (no RWSE on its nodes), per your spec.

Everything else above follows from your instruction verbatim.
