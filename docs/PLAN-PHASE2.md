# Phase 2 — Frozen LLM core (no attention bias, no LoRA): step-by-step plan

Written 2026-08-25 after Phase 1 closed, from (i) `docs/PLAN.md` Phase 2, (ii) the research
notes, (iii) the transformers **5.15.1** source installed on both machines, (iv) what Phase 1
measured, and (v) an adversarial review of a first draft, three of whose findings were
confirmed by re-running them on Cora (§1). Where this file and `docs/PLAN.md` differ, this
file wins for Phase 2; every difference is listed in §1 with its reason.

## 0. The question, and the honest expectation

**Does a frozen, language-pretrained transformer body add anything when a graph enters it
through a learned linear projection of adjacency rows?**

One model, four bodies. Encoder E1, decoder-input norm, decoder D1, loss, optimizer, split,
selection rule and metrics are identical; only the block in the middle changes:

| arm | body | trainable | isolates |
|---|---|---|---|
| 1 | **frozen pretrained** Llama-3.2-1B | E1 + norms + D1 (9.82 M) | the thing we are building |
| 2 | **none** (encoder → decoder) | E1 + norms + D1 (9.75 M) | is the body doing anything at all |
| 3 | **frozen random-init** Llama-3.2-1B (same config, seeded init) | as arm 1 | pretraining vs. random fixed features — the one FPT claim that survived replication |
| 4 | **trained-from-scratch** pre-LN transformer, width ∈ {128 … 2048} | E1 + norms + D1 + body | the matched-params curve, re-measured under this phase's loss |

Deliverable = the **delta table**: for each pair of arms the per-seed difference on the same
split (paired), its mean, SE and a paired t-test, on all three metric columns, plus the
minimum detectable effect. `Δ ≈ 0` is a legitimate outcome; Phase 3 (the attention bias) is
the intervention meant to create a gap. Stop-the-line only if nothing trains or a protocol
test fails.

**What to expect (measured while planning, laptop, seed 0, this recipe):** arm 2 alone
reaches **0.915 AUROC / 0.927 AP@1:1** (three seeds in review: 0.913–0.922 / 0.923–0.934),
i.e. *above* GAE/GAT (0.905–0.910 / 0.911–0.912). The bar for arm 1 is therefore arm 2,
not GAE. PPR (0.850 / 0.900 / AP-sparse 0.031) stays as the heuristic reference.

## 1. What changed since `docs/PLAN.md` and the first draft, and why

| item | before | Phase 2 does | why |
|---|---|---|---|
| **loss** | draft proposed `recon_bce` (Phase-1 baselines' loss) "for comparability" | **masked-cell objective** (CLAUDE.md §5.1, PLAN.md): each step hide a fresh 15 % of the train graph's upper triangle (`mask_matrix`, mirrored), BCE-with-logits on exactly those cells, `pos_weight` = non-edge/edge ratio of the train graph (≈ 815.7, recorded) | with adjacency-row tokens the training label of pair (i,j) is *inside token i*, so `recon_bce` is solved by copying the input: train-visible AUROC → 0.999 while validation stalls at ~0.82 and decays (verified). Phase 1's scratch rows were early-stopped pre-memorisation states. Masking removes the shortcut: arm 2 goes from 0.84 to 0.915 |
| arm 4 | Phase-1 scratch curve reused | **re-run** inside the same wrapper (same E1, norms, D1, loss, selection; body = pre-LN transformer, dropout 0) | Phase-1 curve was measured under the shortcut-able loss with a different input gain and a seed-0-only LR rule; it is not comparable. Phase-1 rows stay as the Phase-1 record |
| encoder LayerNorm gain | draft: T/√d for arms 1/3, 1 for arm 2 | **same everywhere**: encoder LN gain `T/√d` (T from arm 1's embedding table, ≈ 0.022 numerically, recorded); plus a **decoder-input LayerNorm**, gain `1/√d`, in every arm | gain is a scale knob worth +0.05 AUROC on arm 2 (verified: 0.862 vs 0.915); it must not differ between arms. The decoder-input norm makes every arm's decoder see unit-norm rows (arm 1's post-RMSNorm rows have norm ≈ 45 and are anisotropic; arm 2's have norm ≈ 1) |
| seeds | 3 | **5** (0–4), paired across arms on the same split; pre-registered extension to 10 seeds for arms 1/3 at the selected LR if the paired Δ is below the MDE | unpaired MDE at n=5, σ≈0.012 is ≈ 0.024 AUROC; pairing on shared splits cuts the SE 30–60 % (Phase-1 rows: GAE vs GAT paired SE 0.003 vs 0.008) |
| LR selection | seed-0 val, then seeds | full grid × 5 seeds; LR per arm by **mean val AUROC over seeds**; **edge rule**: if the argmax sits on a grid edge, extend one decade that way (5 more runs) before the arm counts | Phase 1 selected an edge LR for 4 of 5 widths; the FPT rebuttal's point is that orderings change with LR |
| trainable count | ~12.7 M | 9,746,432 + 4,096 (decoder norm) + 67,584 RMSNorm (arms 1/3) | computed from the code |
| E4 (features) | draft: conditional extension | **dropped from Phase 2** | its premise ("E1 arms sit at a 0.85 ceiling") was false; it changes the question (topology+features) and has no arm 4; features come in Phase 4 with the crossing control |
| shuffled-adjacency control | Phase 3 | **added now** for arms 1 and 2 at the selected LR (CLAUDE.md §8: "from day one") | cheap (10 runs); degree-preserving rewiring of the observed graph must drop every arm to the degree-only level |
| encoder-gradient hook (GWC) | available behind a flag | **dropped** | AdamW updates are invariant to a uniform gradient scale; a 65× smaller encoder gradient is not a vanishing gradient. Encoder grad norm is logged as a > 0 sanity check only |
| local bring-up | laptop N ≤ 512 overfit | Marlowe probe job | user rule: anything beyond seconds-long runs on Marlowe |

Mechanics verified on the installed 5.15.1 (line numbers in the investigation record): a 4-D
`attention_mask` is returned untouched by `create_causal_mask` (we build it: float, bf16, on
device); `attention_mask=None` is **causal** under eager unless `config.is_causal=False`;
`position_ids` must be an explicit `[B,N]` zeros (default `arange` turns RoPE on);
`attention_scaling == 1.0` for rope type `llama3`; `repeat_kv` precedes the score matmul
(per-head tensors have 32 heads); `use_cache` defaults on; `LlamaModel(cfg)` yields **fp32**
params and **SDPA** unless `_from_config(cfg, dtype=bf16, attn_implementation='eager')` is
used; `from_pretrained` takes `dtype=`; a body with fp32 RMSNorm weights runs **only under
autocast**; `GradientCheckpointingLayer` checkpoints only in `train()` mode (safe: no
dropout); gradient reaches `inputs_embeds` through a frozen body without
`enable_input_require_grads`; `LlamaModel(cfg)` consumes the global RNG; the stub embedding
must be created in bf16 or `llm.dtype` reports fp32.

## 2. Locked configuration

**Backbone.** `meta-llama/Llama-3.2-1B`, revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`
(current `main`, resolvable without a token; re-checked at staging). Load-time asserts:
hidden 2048, 16 layers, 32 heads, 8 KV heads, head_dim 64, rope type `llama3`,
`rotary_emb.attention_scaling == 1.0`, `config._attn_implementation == 'eager'`, every
non-RMSNorm body parameter bf16.

**Loading.** `cfg = AutoConfig.from_pretrained(id, revision=SHA)`; `cfg.is_causal = False`.
Arm 1: `AutoModel.from_pretrained(id, config=cfg, revision=SHA, dtype=torch.bfloat16,
attn_implementation='eager')`. Arm 3: `torch.manual_seed(seed)`;
`LlamaModel._from_config(cfg, dtype=torch.bfloat16, attn_implementation='eager')`
(N(0, 0.02) init; `body_seed = seed`, recorded). Both: `requires_grad_(False)`; `T` = mean L2
norm of non-zero rows of `embed_tokens.weight` (arm 3: its own random table, ≈ 0.905);
`embed_tokens` replaced by `nn.Embedding(1, 2048, dtype=torch.bfloat16)`; the 33 RMSNorm
weights re-cast to fp32 with `requires_grad_(True)`. Then, in **every arm**,
`torch.manual_seed(seed)` immediately before constructing E1 / norms / D1, so trainable
inits are bit-identical across arms at each seed (tested).

**Forward (single entry point in `llm.py`, always under autocast — training, validation,
canary, tests, walkthrough).**
```
tokens = encoder(A_obs)                                          # [B,N,2048] fp32, row norm ≈ T
with torch.autocast(device_type, dtype=torch.bfloat16):
    H = llm(inputs_embeds=tokens.to(torch.bfloat16),             # differentiable cast at the boundary
            attention_mask=zeros(B, 1, N, N, dtype=torch.bfloat16),   # bidirectional; Phase-3 injection point
            position_ids=zeros(B, N, dtype=torch.long),          # RoPE = identity
            use_cache=False).last_hidden_state                   # post-final-RMSNorm
Z = dec_norm(H.float())                                          # LayerNorm(2048), gain 1/√d
logits = decoder(Z)                                              # D1: Z W Zᵀ, W = 0.1·I
```
Asserts inside: mask dtype bf16 (never derived from `llm.dtype`); body in `train()` whenever
checkpointing is enabled. `attn_mode ∈ {bidirectional, causal}` exists as a config field
(Phase-5 hedge); only `bidirectional` is used. Arm 2: `Z = dec_norm(tokens)`. Arm 4: body =
`nn.TransformerEncoder` (pre-LN, GELU, 8 heads, 4 layers, dropout 0, no positional
embeddings, no final norm) on `tokens`, trained.

**Precision.** Body bf16; encoder, both norms, decoder and RMSNorm weights fp32; AdamW on fp32
params (a 1e-4 update on a bf16 parameter underflows — verified); no GradScaler.

**Encoder E1.** `Linear(2708, 2048)`, orthogonal init gain 1.41, bias 0; `LayerNorm(2048)`
weight `T/√2048`, bias 0. Input = dense observed adjacency rows (train graph, masked per
step, during training; train+val graph at test); no features; isolated nodes give zero
tokens (no NaN). **Decoder-input norm.** `LayerNorm(2048)`, weight `1/√2048`, bias 0, all
arms. **D1.** `Z W Zᵀ`, `W = 0.1·I`. **D3** (secondary). `MLP([z_i ‖ z_j ‖ z_i⊙z_j])`
6144 → 512 → 1 on the supervised cells during training, on all 3.66 M scorable cells in
chunks at test (chunked == dense, tested).

**Loss.** Per step: `A_obs, sup = mask_matrix(A_train, 0.15, seed=step)`;
`BCE_with_logits(logits[sup], A_train[sup], pos_weight=815.7)`. Eval input: the unmasked
train graph (validation) / train+val graph (test). Full-batch, one graph per step.

**Optimizer.** One AdamW, default betas, groups: encoder + norms + decoder (LR ∈ {1e-3, 3e-4,
1e-4} + edge rule, wd 0.01), RMSNorm affine (1e-4, wd 0; empty in arms 2/4), scratch body
(arm 4, same LR as encoder), `bias_table` / `lora` asserted empty; any unclassified trainable
raises. Grad-clip 1.0; 100-step linear warmup, constant after; max 2000 epochs; early
stopping on validation AUROC, patience 200, best state restored.

**Evaluation.** `evaluate_edge_split(logits, get_split(seed))`, seeds 0–4, the six Phase-1
columns (AP@1:1 on the split's own 527 negatives; caption says so). Identity and random rows
recomputed into the table. Reference rows shown with the loss named per row.

**Statistic.** For arms A, B: `d_s = metric_A(s) − metric_B(s)` on the same split s; report
mean(d), SE(d), paired t (n = 5), for AUROC, AP@1:1, AP@sparse; MDE reported alongside.

**Per-run record (JSON row).** encoder, arm, decoder, loss, lr, seed, body_seed, T,
val AUROC, best_epoch, epochs run, trainable count, peak GPU memory, s/epoch, wall-clock,
`git rev-parse HEAD`, config hash, encoder grad norm at steps 0/100 (> 0), first-batch
decoder-input row norm and D1 logit stats (diag mean, off-diagonal std, train-edge mean),
and for arms 1/3/4 the per-layer hidden-state RMS on the first batch.

## 3. Run grid (each task = 1 GPU; Slurm arrays over an explicit run list)

| stage | tasks | note |
|---|---|---|
| probe (step 4) | 1 | measures s/epoch and peak memory; sets `--time` |
| D1 grid: arms 1, 2, 3 × LR × 5 seeds | 45 | arm 2 is seconds |
| arm 4 sweep: widths {128, 256, 512, 1024, 2048} × LR × 5 seeds | 75 | Phase-1's sweep took 11 min in total |
| edge-rule extensions | 5 per triggered arm | |
| D3: arms 1, 2, 3 at selected LR × 5 seeds | 15 | full grid if the probe's s/epoch allows |
| controls: shuffled-A (arms 1, 2, selected LR × 5); `recon_bce` shortcut row (arm 2 × LR × 5); encoder-gain-1 row (arm 2 × LR × 5) | 40 | arm-2 rows are seconds |
| seed extension (pre-registered, only if paired Δ < MDE) | 10 per arm 1/3 | |

Estimated s/epoch from FLOPs (≈ 19 TFLOP per epoch incl. the validation forward): 0.1–0.4 s
⇒ 3–13 min per 2000-epoch run, typically 2–4 min with early stopping; ≈ 30 GPU-h total
against ~10,000 GPU-h/cycle. Treated as unknown until the probe measures it.

Row files: `$SCR/results/phase2/rows/<encoder>_<arm>_<decoder>_<loss>_<lr>_<seed>.json`
(full key in the name; written to `.tmp` then renamed; **outside the rsync'd repo copy**),
pulled with `rsync marlowe:$SCR/results/phase2/rows/ results/phase2/rows/`. A task skips only
if a row with the same key **and the same commit hash** exists (`--force` overrides);
`aggregate` refuses a table whose rows span more than one commit. Arrays use
`--output=$SCR/logs/%x-%A_%a.out`; failed/timed-out ids are resubmitted with
`sbatch --array=<ids>`. Checkpoints (trainables only, ≈ 39 MB) at
`$SCR/runs/phase2/<key>.pt`; arm 1 seed 0 at the selected LR also saves its test logits
(29 MB) for the walkthrough.

## 4. Steps

**Step 0 — prerequisites (user).** Sign Gate 1 and push; accept the Llama 3.2 Community
License and create a read token; approve this plan (§8). No Phase-2 code before that.

**Step 1 — skeleton + tests (laptop, seconds).** Files in §6. Tests use
`LlamaConfig(hidden 64, 2 layers, 4 heads, 2 KV heads, initializer_range 0.2, eager)` with
random weights in fp32 (tolerance 1e-5), so they need no download: mask canary; zero-mask
== `is_causal=False`; `[B,1,N,N]` == `[B,32,N,N]` zero mask; **body** equivariance
`body(tokens[p]) == body(tokens)[p]` exactly, with the negative control that it *fails* when
`position_ids` is omitted; **full-model row-permutation** `model(A[p]) == model(A)[p][:,p]`
exactly (column relabelling is not equivariant for E1 by design — recorded, not tested);
eager enforced on both body constructions; frozen-base membership; grad existence with and
without checkpointing; E1/norm/D1 weights bit-identical across arms at a seed; encoder
output norm within 25 % of T on real Cora rows; D3 chunked == dense; stray trainable raises;
`mask_matrix` + `pos_weight` loss on a tiny graph equals a hand computation. Plus one
seconds-long real-data check on the laptop GPU: arm 2 through the unified wrapper on Cora
seed 0 must reproduce ≥ 0.91 AUROC (the number measured during review).

**Step 2 — weights.** Marlowe login node, user present: `hf auth login` (prompted, never
scripted) then `hf download meta-llama/Llama-3.2-1B --revision <SHA> --include config.json
generation_config.json model.safetensors` (2.3 GB; the `original/` duplicate excluded) into
`HF_HOME=/scratch/m000211-pm06/uutsha/hf`; copy `config.json` into `results/phase2/`.
Laptop: same download into the default `~/.cache/huggingface` (already outside OneDrive).
Jobs run `HF_HUB_OFFLINE=1` and start with `test -d $HF_HOME/hub/models--meta-llama--Llama-3.2-1B/snapshots/<SHA>`
so a scratch purge fails in seconds, not after a queue wait.

**Step 3 — real-weight checks (laptop, seconds).** Load asserts; **mechanics parity** on
pure-bf16 weights (before the fp32 recast): stub + zero 4-D mask + `position_ids=0` vs stock
model with `is_causal=False, attention_mask=None` → `torch.equal`; **precision drift**:
recipe model (fp32 norms, autocast) vs pure bf16 on the same inputs → relative Frobenius
error `‖ΔH‖/‖H‖ < 5e-2`, value recorded in RESULTS.md; canary with real weights. Then rsync
and **`pytest` on the Marlowe login node** (Python 3.10, CPU) — green before any `sbatch`;
the sbatch preamble also does `python -c "import g2l.llm, g2l.train, g2l.run_phase2"`.

**Step 4 — probe job (Marlowe, 1 GPU, 20 min).** Arm 1: (a) N=512 subgraph, 300 steps: loss
falls below half its step-0 value, encoder grad norm > 0 at every logged step; (b) full
Cora, 3 epochs, with and without gradient checkpointing (body in `train()`, checkpoint-call
count logged), `max_memory_allocated` and s/epoch; (c) arm 3, one epoch, per-layer RMS
profile. Expected: checkpointing off (≈ 28–35 GB estimated). Sets `--time` = 3× the
measured 2000-epoch time. Nothing else is submitted before this log is read.

**Step 5 — D1 grid + arm-4 sweep (arrays of 45 and 75).** Pull rows; per-arm LR curves;
apply the edge rule; select LR per arm; build the paired delta table. This is the table
Gate 2 is judged on.

**Step 6 — D3 (array of 15, or the full grid if cheap).**

**Step 7 — controls (array of 40).** Shuffled-A (degree-preserving rewiring of the observed
graph everywhere the model sees it; must collapse to the degree-only level ≈ 0.65); the
`recon_bce` row for arm 2 (documents the shortcut: train-visible AUROC ≈ 1, validation
collapse, test ≈ 0.84); the encoder-gain-1 row for arm 2 (documents the scale effect).

**Step 8 — close.** Pull rows → verify (single commit, all keys present) → aggregate →
`python -m g2l.walkthrough --phase 2` on the laptop: re-runs the gate assertions live (load
asserts, canary, parity), loads the frozen body + arm-1 seed-0 trainables, recomputes the
full-graph logits under `no_grad` (≈ 5 GB on the 12 GB laptop GPU), compares to the saved
Marlowe logits (max |Δ| and AUROC Δ reported; the tolerance is *measured* this first time,
then locked) → `results/phase2/RESULTS.md` (claim; table with identity, random, PPR,
GAE/GAT reference rows and the loss named per row; per-seed rows; paired deltas with SE, t,
MDE; per-LR curves; probe numbers; measured T; locked config incl. the parity procedure;
shows / does not show; VALIDATED line) → one slide → cleanup in this order: delete
superseded rows/logs/checkpoints on the laptop → commit → delete on the cluster → user pushes.

**Gate 2 checklist.** Probe log recorded (peak memory, s/epoch, checkpoint decision) ·
all new tests + 22 old green on both machines · canary + load asserts pass in every job ·
identity 0.500 / lift 1.0 and random at chance · every arm either converged or has a
recorded plateau diagnosis (arm 3 may legitimately plateau: extended LR grid tried, RMS
profile and encoder grad norms attached) · no LR at a grid edge without the extension ·
paired D1 delta table with SE/t/MDE on three columns · arm-4 curve under this loss · D3
table · shuffled-A collapses · `recon_bce` and gain-1 rows recorded · T and drift numbers
recorded · walkthrough run by the user and signed.

## 5. Tests landing in Phase 2 (exact assertions)

| test | assertion |
|---|---|
| mask canary (every job start) | `not allclose(f(t, mask=0), f(t, mask=randn·0.5))` |
| zero-mask parity | `torch.equal(f(t, mask=0), f(t, mask=None, is_causal=False))` |
| head-broadcast parity | `torch.equal(f(t, zeros[B,1,N,N]), f(t, zeros[B,32,N,N]))` |
| body equivariance (+ negative control) | `body(tokens[p]) == body(tokens)[p]` (atol 1e-5 fp32); `max|Δ| > 0.1` with `position_ids=None` |
| full-model row permutation | `model(A[p]) == model(A)[p][:,p]` (atol 1e-5) |
| eager + dtype at construction | `_attn_implementation == 'eager'`; non-norm body params bf16; stub bf16; mask bf16 |
| RoPE | `rotary_emb.attention_scaling == 1.0` |
| frozen-base membership | `p.requires_grad == (name ∈ encoder ∪ norms ∪ decoder ∪ RMSNorm-affine ∪ scratch body)` |
| grad existence | after one backward every trainable has a grad; encoder grad norm > 0; with and without checkpointing |
| paired init | E1, norms, D1 weights `torch.equal` across arms at seed 0 |
| encoder scale | `|mean ‖E1(A_cora)‖ − T| / T < 0.25` |
| loss | `mask_matrix` + `pos_weight` BCE on a 6-node graph equals a hand computation |
| D3 | chunked == dense scoring |
| param groups | stray trainable raises; `bias_table`/`lora` empty |
| mechanics parity (real weights, step 3) | `torch.equal` stub+zero-mask vs stock `is_causal=False` |
| precision drift (real weights, step 3) | `‖ΔH‖/‖H‖ < 5e-2`, recorded |
| arm-2 reproduction (real data, step 1) | seed 0 test AUROC ≥ 0.91 |

## 6. Files

Create: `g2l/encoders.py` (E1 behind the `GraphEncoder` interface), `g2l/decoders.py` (D1,
D3), `g2l/llm.py` (loader for arms 1/3, asserts, canary, the autocast forward, Impl-B
fallback documented), `g2l/model.py` (arm assembly: body ∈ {llama, none, scratch}, both
norms), `g2l/train.py` (recipe, row record, checkpoint of trainables), `g2l/run_phase2.py`
(run-list expansion, key-named row files, commit-aware skip), `configs/phase2.yaml`,
`slurm/phase2_probe.sbatch`, `slurm/phase2_grid.sbatch` (array), `tests/test_llm.py`,
`tests/test_encoders.py`, `results/phase2/RESULTS.md`.
Modify: `baselines/aggregate.py` (Phase-2 rows, LR selection, paired statistics, single-commit
check), `g2l/walkthrough.py` (phase 2), `scripts/marlowe_setup.sh` (weights staging),
`docs/PLAN.md` (pointer to this file). `baselines/scratch_transformer.py` stays untouched as
the Phase-1 record; the Phase-2 scratch body lives in `model.py`.
Laptop-side binaries (checkpoint, logits, config copy) under `%LOCALAPPDATA%\g2l\phase2\`,
never in the repo.

## 7. Risks and stop-the-line rules

- **Input-visible labels.** Any objective that scores cells present in the input row is
  shortcut-able with row tokens; only masked cells are supervised. The `recon_bce` control
  row keeps the failure mode on record.
- **Injection is behaviour, not API.** transformers pinned at 5.15.1 on both machines; the
  canary at every job start; Impl B kept as the same-signature fallback.
- **Silent no-gradient** (`.detach()`, `no_grad`, frozen encoder under reentrant
  checkpointing): grad-existence assert in every job.
- **Stale rows across code versions**: enforced by the commit hash in each row and the
  single-commit check in `aggregate`, not by discipline alone. Superseded rows, logs and
  checkpoints are deleted on both machines once corrected ones are verified.
- **Arm 3 may collapse** (N(0, 0.02) init, no depth scaling). Procedure: extended LR grid;
  if the plateau persists, it is the arm's result, recorded with its RMS profile.
- **Memory/time unverified** until the probe. Nothing is submitted before its log is read.
- **Gated weights**: no token in code, docs, logs or shell history; staging on the login
  node with the user present; Qwen3-0.6B (ungated) as the bring-up fallback if access is
  blocked for days.
- **Dev/prod split**: code written on Python 3.14 runs on 3.10 — cluster `pytest` after every
  rsync, before every `sbatch`; tests are device-agnostic.

## 8. Decisions needed before code is written

1. Confirm the reversal on the loss: masked-cell objective as the primary protocol, arm 4
   re-run under it, `recon_bce` kept only as the arm-2 shortcut control.
2. D3 scope: selected-LR only (15 runs, default) or the full grid (45).
3. Seeds: 5 with the pre-registered extension to 10 for arms 1/3 if the paired Δ is below
   the MDE (default), or 10 from the start (+45 runs).
4. Confirmation that the Llama 3.2 license is accepted and a token exists on your side.

## 9. Not in Phase 2 (recorded in RESULTS.md "does not show")

Attention bias (Phase 3), LoRA (3B), encoders E2–E7 incl. features/E4 with the crossing
control (Phase 4), masked-node and autoregressive tasks (Phase 5), LLM2Attn/LLM2Trsf and the
training-edge data-fraction sweep (Phase 6; pre-registered as the first Phase-3 controls if
Phase 2's deltas are null), fully-trained random body (Phase 6; arm 4 at d2048 approximates
it), MaskGAE re-run at our masking rate, RMSNorm-frozen sub-ablation.
