# Graph-In / Graph-Out LLM — Phased Build Plan

## Context

Dr. Islam approved the architecture 2026-08-25: raw adjacency → trainable encoder (one token
per node, no tokenizer/embedding table) → **frozen Llama-3.2-1B** with a graph-derived
additive attention bias → trainable decoder → predicted adjacency. Primary task: masked edge
prediction on Cora. This plan turns ~570 KB of verified research (docs/research/, docs/hpc/)
into strict sequential phases. Each phase ends with a **gate the user personally validates**
before the next begins. Where CLAUDE.md and the research disagree, the research wins (the
superseded list is at the bottom).

User decisions taken: **backbone = Llama-3.2-1B** (user accepts the HF license + creates a
token when Phase 2 needs it; Qwen3-0.6B is the later second point); **hosting = private
GitHub repo**. Code style: clean, minimal, few comments, no unnecessary machinery. Logging:
JSONL files, no W&B.

Compute: laptop (RTX 5070 12 GB, Python 3.14, torch 2.13+cu130, transformers 5.15.1) = dev,
baselines, N≤512 smoke tests, gate walkthroughs. **Marlowe H100 80 GB** (account
`marlowe-m000211-pm06`, partition `batch`, job arrays, no debug partition → bring-up happens
locally, Marlowe enters at the first *sweep*) = all real training. Sherlock = backup.
Full-graph Cora training is Marlowe-only; full-graph *inference* fits the laptop (no_grad) —
used so gates never mean "trust the cluster logs".

## The architecture in one block

```
A_obs [N,N] (15% hidden) ──► encoder E* ──► tokens [N,2048]
                                             │  LayerNorm, gain=T/√d  (T = mean L2 of embed rows)
                                             ▼
   frozen Llama-3.2-1B (AutoModel, no lm_head, embed_tokens stubbed, bf16, eager)
   score(i,j) = qk/√d + bias(i,j)   bias: per-head [B,32,N,N] SPD table, zero-init,
   passed as the 4-D attention_mask kwarg ("Implementation A" — v5 masking_utils
   early-exits 4-D masks untouched; layer-SHARED by construction)
   position_ids = zeros (RoPE = exact identity)   config.is_causal = False
                                             │
                                             ▼  H [N,2048]
   decoder D1 (our own σ(H W Hᵀ), ~10 lines) and D3 (pair MLP) ──► Â logits [N,N]
```

Trainable (5 param groups, differential LR): encoder (1e-3..1e-4, swept) · decoder · SPD
table (5e-3, wd=0) · RMSNorm affine (1e-4) · LoRA r=16 α=32 (3e-5, Phase 3B only).
Loss: BCE-with-logits on masked cells only, pos_weight computed from the training split at
runtime. Structural features (SPD etc.) computed from **A_observed only** — never the
complete graph (the leak rule; unit-tested).

## Evaluation protocol (fixed for the whole project)

Three columns on every table, computed by one `evaluate()` on one split object:
1. **AUROC** (balance-invariant bridge),
2. **AP@1:1** — RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True,
   split_labels=True, add_negative_train_samples=False): 527 test positives + 527 sampled
   negatives — the number comparable to published GAE 91.0/92.0 and MaskGAE 96.45/95.95,
3. **AP@true-prevalence** on all masked cells ex-diagonal, + lift (=AP/base-rate).

Rules: sklearn `average_precision_score` on **logits** (never interpolated PR-AUC, never
probabilities, never threshold 0.5); seeded negative draws averaged ≥5; ≥3 seeds; per-config
LR sweep on any frozen-vs-random comparison (the FPT-rebuttal lesson); identity + random
baselines recomputed on every table (leak canary). GAE ~92 AP is the floor, MaskGAE ~96 is
the real bar.

## Repo layout

```
graph2llm/                      (git init here; private GitHub remote)
  configs/                      yaml per experiment
  g2l/
    data.py                     [B,N,N] + node-mask + padding API from day one (B=1 for Cora);
                                both masking views (edge-split + matrix-fraction, upper-tri+mirror)
    metrics.py                  the 3-column evaluate()
    encoders.py                 E1..E7, one interface; orthogonal init gain 1.41; out-LayerNorm
    decoders.py                 D1, D3 (D2/D4 later)
    bias.py                     SPD from A_obs only; per-head; zero-init; unreachable=own bucket
    llm.py                      wrapper: inputs_embeds, 4-D-mask injection + canary,
                                attn_mode {bidirectional,causal}, Impl-B fallback documented
    model.py  train.py  eval.py  walkthrough.py
  baselines/                    heuristics.py gae.py gat.py feature_only.py scratch_transformer.py
  slurm/                        train_1gpu.sbatch + array sweeps (account/partition per docs/hpc/marlowe.md)
  tests/                        the ~12 named tests below
  results/phaseN/RESULTS.md     per-phase deliverable
  requirements-local.txt  requirements-cluster.txt   (transformers==5.15.1 pinned in both)
```

## Phases

### Phase 0 — Protocol foundation (local, ~3–4 days)
Build `data.py`, `metrics.py`, `tests/`. Multi-graph `[B,N,N]`+padding API even though Cora
is B=1 (cheapest hedge against Phase-4 molecular rework). Both masking views; diagonal
excluded; message-passing/supervision cells disjoint. Tests: disjointness, identity-canary,
bias-equality-under-held-out-edge-changes (armed now, exercised Phase 3), multi-graph batch,
`torch_geometric` imports on py3.14/Windows. Also: git init + private GitHub remote + pinned
requirements.
**GATE 0:** all tests green; identity baseline ≈ base rate; user reproduces the three metric
columns by hand on a tiny synthetic graph.

### Phase 1 — Baselines (local) ∥ Marlowe onboarding (~1 week)
Laptop: heuristics (CN/AA/RA/PPR) → feature-only logreg → GAE/VGAE → GAT → scratch
adjacency-row transformer **capacity sweep** d_model∈{128…2048} (the curve is Phase 2's
matched-params reference). All rows through the same `evaluate()`, 3 seeds.
Marlowe (parallel, first SSH session — user provides Duo): verify allocation
(`sacctmgr`/`scontrol show partition` — the docs' 20 unknowns checklist), build venv on
/projects, pre-stage Llama-3.2-1B + Cora on login node (user's HF token), submit the
already-locally-validated scratch sweep as the first job array, `HF_HUB_OFFLINE=1`.
**GATE 1:** GAE reproduces 90.6±0.9 AUC / 91.2±1.0 AP; full baseline table; Marlowe array
job reproduces a local number; RESULTS.md signed.

### Phase 2 — Frozen LLM core: no bias, no LoRA (~1–1.5 weeks)
**Detailed plan: `docs/PLAN-PHASE2.md` (2026-08-25) — supersedes this section where they differ: masked-cell loss is primary, arm 4 re-run under it, 5 seeds paired, decoder-input norm, E4 deferred to Phase 4.**
Local bring-up: E1+D1/D3+frozen Llama (loading recipe above), N≤512 overfit smoke;
`attn_mode` config field exists now (AR hedge). Tests landing: 4-D-mask canary (runs at
every training start, both machines), zero-bias==no-mask parity, base-LLM parity,
`attention_scaling==1.0` assert, frozen-base assert, grad-existence-after-backward assert
(enable_input_require_grads is a no-op with inputs_embeds), encoder-output-norm≈T.
Marlowe: full-graph Cora (gradient checkpointing) — the **4-way comparison**, per-config LR
sweep × 3 seeds, one array (~16–24 runs):
(1) frozen **pretrained** body, (2) **identity body** (w/o LLM — same ~12.7M trainables),
(3) frozen **random-init** body, (4) scratch curve at matched params.
**GATE 2 (honest framing):** the deliverable is the **delta table**, not a win — row 2 will
be strong (E1+D1 ≈ matrix factorization ≈ GAE-class). (1)≈(2) pre-bias is a legitimate
outcome; Phase 3 is the intervention meant to create the gap. Stop-the-line only if nothing
trains or the protocol is broken. Record: Δ(1−2), Δ(1−3) ± σ.

### Phase 3 — Graph-aware attention (~1 week)
Day-one probe (local): parameter-free 1/SPD bias, single scalar λ, N=512 — fastest signal
the injection moves metrics. Then the learned per-head SPD table: zero-init +
`_is_hf_initialized=True`, diagonal zeroed, unreachable = learned bucket (never −inf),
computed from A_observed (leak test now runs for real), layer-shared (Impl A), differential
LRs, `max|bias|/max|qk/√d|` logged with >0.5 alert, bias-nonzero-after-100-steps assert.
Controls: shuffled-A (shuffled everywhere incl. SPD), zero-init inertness (step 0 ==
Phase-2 model). Exit item: flex `score_mod` parity vs eager (~1e-3 bf16; bias reaches
score_mod as a non-leaf tensor). Training stays eager through Cora.
**GATE 3:** bias delta over Phase 2 with seeds+LR sweep; shuffled-A degrades; inertness
verified; flex parity green.

### Phase 3B — LoRA (~2–3 days)
r=16 α=32 dropout 0.05, all 7 projections, `task_type=None`, 3e-5. Why here: ~11.3M params
would contaminate Phase 2's question; after Phase 4 it could invalidate the encoder ranking.
**GATE 3B:** LoRA delta on E1 known; recorded decision whether the encoder study runs with
or without it (default: without; top-2 encoders re-run with).

### Phase 4 — Encoder study E1–E7 (the PI deliverable, ~1.5–2 weeks)
Marlowe arrays: 7 encoders × per-encoder LR sweep × 3 seeds on Cora (E5 uses the
*normalized* Laplacian recipe — never raw eigenvalues). Laptop parallel track: molecular
masked edge prediction (ogbg-molhiv/bbbp/bace, N≈24–34) — the size-agnostic claim of
E3/E6/E7 is untestable on single-graph Cora. Crossing control lands here (needs E4):
(A=I, features-only) × (X=0, topology-only) + feature permutation. Permutation test scoped:
bias+body always; full-model only E6/E7 (E1–E5 fail by design — documented, not "fixed").
**GATE 4:** E1–E7 table on {Cora, molecular} with param counts; crossing quadrant; user can
state per encoder what it adds and what it cannot represent.

### Phase 5 — Task expansion
Masked node prediction (same infra). Then AR: `attn_mode='causal'`, right-shifted input, no
masking (causality hides the future), ordering = swept config {BFS, DFS, degree, k-core,
random} (BFS is NOT a safe default), ordering computed leak-free.
**GATE 5:** causality test (`grad(logits[t], tokens[>t])==0`) + ordering-leak test green; AR
vs masked-edge comparison; ordering-sweep spread reported.

### Phase 6 — Scale + full grid (Marlowe only)
PubMed, ogbn-arxiv (FlexAttention now load-bearing; subgraph sampling; identity canary
rerun under subgraph batching), decoder sweep D2/D4, per-layer bias ablation (triggers
Impl B), LLM2Attn/LLM2Trsf, fully-trained-random (labelled separately), full §8 grid via
arrays.
**GATE 6:** flex-vs-eager parity re-verified at scale; grid complete.

### Phase 7 — Biomedical (deferred)
Directed graphs (magnetic Laplacian becomes non-trivial). Requires a data-classification
review first — both clusters are Low/Moderate-risk only.

## Per-phase deliverable (the user's validation loop)
1. `results/phaseN/RESULTS.md` (~1 page): claim in one sentence; the table (identity+random
   rows recomputed); gate checklist; "shows / does not show" bullets (pasteable to a PI
   email); locked config values; `VALIDATED <date>` line.
2. `python -m g2l.walkthrough --phase N` (<15 min, laptop): recomputes headline numbers from
   saved predictions/checkpoints and re-runs gate assertions live.
3. One slide appended to `slides/` → a ready-made PI progress deck.

## Top rework risks and hedges
1. **Implementation A is behavior, not API** (4-D-mask early-exit). Hedge: the canary at
   every training start on both machines; transformers pinned; Impl B kept as same-signature
   fallback inside llm.py.
2. **Protocol/leak error in Phase 0** poisons everything. Hedge: Phase 0 is its own gate;
   identity canary is a permanent table row; bias-equality + disjointness tests.
3. **Fixed-N single-graph assumptions.** Hedge: [B,N,N]+padding API and multi-graph test
   from Phase 0; SPD per-sample; headline Cora numbers always from full-graph training.

## Superseded CLAUDE.md instructions (research verdicts, sources in docs/research/)
eager-only mandate (FlexAttention works; eager stays as parity oracle) · "5,429 edges"
(5,278 undirected / 10,556 directed; pos_weight 693.7 computed at runtime) · "AUPRC not
AUROC" + balanced-set contradiction (→ 3-column protocol) · universal weighted-BCE (AR task:
pos_weight over the actual scored set) · BFS default ordering (worst in 2 of 3 sources → 
swept) · AR under bidirectional mask (vacuous → attn_mode config) · "import
InnerProductDecoder" (no W hook → own 10-line D1) · LoRA 32/64 (→16/32) · "bias params start
random" (→ zero-init) · SPD-first priority + "magnetic dim 32" (MLP width; magnetic is a
no-op on undirected graphs → deferred) · "GAE is the number to beat" (floor; MaskGAE 96+ is
the bar) · shuffled-A decisive (→ crossing control) · single random-init control (→ two,
labelled) · FPT cited unqualified (→ cite rebuttal 2107.12460; only frozen-pretrained >
frozen-random survives) · missing RMSNorm trainables (→ 5th param group) · permutation test
as written (guaranteed-fail for E1–E5 → scoped).

## Verification
Every phase: `pytest tests/` green locally + walkthrough script live in front of the user +
RESULTS.md signed before the next phase's first commit. Phase 1 externally anchors the
whole protocol by reproducing published GAE numbers; Phase 2+ Marlowe runs are re-verified
on the laptop via full-graph no_grad inference of the downloaded checkpoints.
