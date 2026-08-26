# Phase 3 — Graph-aware attention: the per-head SPD bias

Written 2026-08-26 after Gate 2. Phase 2 established, on Cora masked edge prediction with
adjacency-row tokens and the masked-cell loss: no body 0.920 AUROC; frozen Llama-3.2-1B
0.878 (−0.043, p 0.004); random-init frozen body 0.887 (pretrained − random −0.009, p 0.11,
10 seeds); scratch transformer ≤ 0.873. Without a structural channel into attention the body
receives N unrelated tokens and can only hurt. **Phase 3 opens that channel** and is judged
on three pre-stated questions (§7). Per the user's instruction after Phase 2, the whole phase
is one probe + one array (§4); nothing is designed to need a second round.

## 1. Mechanism (implemented, tested)

`g2l/bias.py::SPDBias`: a learned table `[heads, 10]` indexed by shortest-path distance on the
**model's own input graph** — bucket 0 self (fixed 0), 1…8 exact distance, 9 = farther or
unreachable (learned, never −inf). Zero-init, so at step 0 every biased model *is* its Phase-2
counterpart (`test_zero_init_bias_is_inert`, bit-exact). `spd_matrix_fast` computes SPD by
boolean matmul on the GPU (exact vs scipy, 34 ms on Cora), so it is recomputed from the
masked input every training step and from the test-time input at eval — the leak rule holds
by construction (`test_bias_depends_only_on_observed_graph`).

Injection: the `[1, 32, N, N]` bias is the 4-D float `attention_mask` (Implementation A,
layer-shared, all 16 layers); gradients reach the table through transformers 5.15.1's mask
path (`test_bias_gradient_and_param_group`, tiny Llama and scratch). The scratch body takes
the same tensor as `nn.TransformerEncoder`'s float mask `[H, N, N]` (zero-mask parity and
gradient verified). Permutation equivariance of bias + body holds (`test_biased_model_is_
permutation_equivariant`). The table trains in its own AdamW group (`lr_bias`, no weight
decay); the run asserts the table is non-zero after 100 steps and records the final table
per head in the row (an interpretable result: how each head weights distance).

Verified in the installed transformers 5.15.1 (research pass, 2026-08-26): a 4-D float mask
early-exits `masking_utils` untouched, is added once to the scaled scores before an fp32
softmax (`modeling_llama.py:201-208`), is the same tensor object at every layer, and a
zero-init table behind it receives gradients (bf16 body under autocast included). One
caveat: *reentrant* gradient checkpointing fails with a shared non-leaf mask; we run without
checkpointing and `FrozenBody.checkpointing` is non-reentrant.

Not in Phase 3: LoRA (3B), per-layer bias (needs Implementation B), RRWP / magnetic
Laplacian biases, GaLA's per-head λ calibration (the learned per-head table subsumes it),
encoders E2–E7.

## 2. Arms

| arm | body | bias | encoder LR | trainables |
|---|---|---|---|---|
| pretrained + SPD | frozen Llama-3.2-1B | learned table (32 heads) | 1e-2 (Phase-2 selection) and ×⅓, ×3 | 9.82 M + 320 |
| random + SPD | same config, seeded init | learned table | 3e-3 | 9.82 M + 320 |
| scratch d256 L4 + SPD | 4-layer pre-LN transformer | learned table (8 heads) | 1e-3 | 3.9 M + 80 |
| scratch d256 L1 + SPD | one transformer block — Tan et al.'s LLM2Trsf control | learned table | 1e-3 | ~1 M + 80 |
| none | (Phase 2) | – | 3e-4 | 9.75 M |

Everything else (E1, LayerNorm gain T/√d, D1, decoder-input LayerNorm, masked-cell loss,
warm-up, clip, patience 200 / 2000 epochs, evaluation) is the Phase-2 recipe unchanged.

## 3. Controls

- **Shuffled A** for pretrained + SPD: the bias sees the rewired graph too. Real input must
  beat it by a margin comparable to Phase 2's +0.19.
- **Zero-init inertness** (test) and **bias non-zero after 100 steps** (assert in every run).
- **LLM2Trsf** (Tan et al. 2024, arXiv 2406.16964): scratch d256 with one transformer block +
  the same bias. If 16 frozen pretrained layers + bias do not beat one trained block + bias,
  the LLM is not contributing. Pre-registered in PLAN-PHASE2 §9 as the first Phase-3
  control if Phase 2's pretrained-vs-random delta was null — it was (−0.009, p 0.11).
- **Bias-off references at 10 seeds** so every paired delta has n = 10: none seeds 5–9,
  scratch L4 seeds 5–9, scratch L1 seeds 0–9 (bias off). Pretrained and random already have
  10 seeds from Phase 2.
- **Data-fraction sweep** (pre-registered in PLAN-PHASE2 §9 / docs/research/fpt.md,
  *conditional*): train edges kept at 10 / 25 / 50 % (val/test unchanged) for pretrained +
  SPD, random + SPD, none at the selected LRs — run only if pretrained + SPD ≈ random + SPD
  after the main stage (the question it answers is whether a pretrained prior matters when
  data is scarce; GPT4TS-style evidence says that is the only corner where it should).

## 4. The array (one submission)

Main stage, 240 runs, all 10 seeds, `python -m g2l.run_phase3 --stage main --list`:

| block | runs |
|---|---|
| pretrained + SPD, enc 1e-2 × bias LR {3e-3, 1e-2, 3e-2, 1e-1} | 40 |
| pretrained + SPD, enc {3e-3, 3e-2} × bias LR {1e-2, 3e-2} | 40 |
| random + SPD, enc 3e-3 × bias LR grid | 40 |
| scratch L4 + SPD, scratch L1 + SPD × bias LR grid | 80 |
| pretrained + SPD shuffled-A, bias LR {1e-2, 3e-2} | 20 |
| bias-off references: none s5–9, scratch L4 s5–9, scratch L1 s0–9 | 20 |

Bias LR grid spans 1.5 decades around GTLM's 5e-3 … 4e-2. Estimated ≈ 20 GPU-hours
(140 frozen-body runs; the probe measured the bias path at 0.76 s/epoch as a gather, cut to
≈ 0.35 s/epoch by the one-hot-matmul formulation); `sbatch --array=0-15 --time=08:00:00
slurm/phase3_grid.sbatch main 15` — 16 tasks, 15 runs each, inside the 32-job account cap;
the time limit covers a chunk of 15 frozen runs that all hit the 2000-epoch cap.
Before it: one probe job (`--probe`: biased arm 1 on the 512-node subgraph — loss halves,
table moves, gradient at the table; full Cora s/epoch and peak memory with the bias).

## 5. Selection and statistics

Per arm, (encoder LR, bias LR) is chosen by mean val AUROC over the 10 seeds; the edge rule
applies to both grids (a point at the boundary is extended once, `extend:` stage). Test
metrics reported at the selected setting; every comparison is a paired per-seed delta on the
shared splits with SE, t, p and MDE (n = 10 everywhere). Phase-2 rows are the bias-off
references (same seeds, same splits, same scorer; their commit `1f1f33a` differs from
Phase 3's and is recorded as such).

## 6. Steps

1. Commit; rsync the tree to Marlowe (new phase → new pinned commit); cluster `pytest`.
2. Probe job; read its log (memory, s/epoch, table movement) before anything else.
3. Main array. Pull rows; aggregate (`g2l/aggregate3.py`); apply the edge rule if it fires.
4. Fill `selected:`; run the fraction stage only under the §3 condition.
5. Walkthrough (laptop): gate tests live; rebuild pretrained + SPD seed 0 from its
   checkpoint; recompute full-graph logits vs the saved ones; print the learned table.
6. `results/phase3/RESULTS.md` — claim, table, per-seed rows, paired deltas, the LR
   surface, the learned bias tables (per head, per distance), controls, probe, gate
   checklist, shows / does not show, VALIDATED line. Cleanup, commit, user pushes.

## 7. Gate 3 — the questions, answered in this order

- **Q1 — does the bias help the frozen body?** Δ(pretrained + SPD − pretrained) over 10
  paired seeds. Success: > 0 and significant.
- **Q2 — does the LLM path now match or beat the linear model?** Δ(pretrained + SPD − none).
  The project goal is met at this stage if this is ≥ 0 (the 0.043 is recovered); it is a
  positive result if it is significantly > 0.
- **Q3 — does pretraining matter once structure is visible?** Δ(pretrained + SPD − random +
  SPD), and pretrained + SPD vs scratch L1 + SPD. Success: both > 0 and significant.
  If Q1 holds but Q3 does not, the bias is a generic graph-transformer effect and the
  language pretraining is still not being used — the data-fraction sweep then runs, and the
  honest framing for the write-up is "a frozen transformer with structural attention bias",
  not "an LLM".
- Controls pass: shuffled-A clearly below; inertness; every run's table non-zero.
- Failure of Q1 (bias does not help, or hurts) is a stop-the-line for the current design and
  triggers a diagnosis (learned tables, per-head λ, first-half-of-layers-only variant, Impl B
  per-layer) before anything else is built.

## 8. Decisions taken (by Claude, per the user's "optimal solution" instruction — veto any)

1. SPD table only (no RRWP / magnetic Laplacian) — the mechanism every reference shares;
   other bias forms are ablations for later.
2. Layer-shared, all layers (Implementation A) — per-layer / first-half variants deferred.
3. Encoder LR fixed at the Phase-2 selection per arm, with a ×⅓ / ×3 neighbourhood for the
   pretrained arm only; bias LR on a 4-point grid — one array, no expected extension.
4. 10 seeds for everything; scratch at d256 (Phase 2's best width) with 4 and 1 layers.
5. Data-fraction sweep conditional (§3), not in the main array.
6. No LoRA until the bias result is known (3B).
