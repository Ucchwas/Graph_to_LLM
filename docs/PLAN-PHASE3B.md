# Phase 3B — LoRA on the frozen body, with the structural bias

Written 2026-08-26 while the Phase-3 top-up (job 450168) runs; numbers below are the Phase-3
state at that moment and will be replaced by the final Phase-3 selection before submission.
Approved direction (user, 2026-08-26): finish Phase 3, run LoRA once for a definitive answer,
then move to the encoder study / molecular datasets regardless of the LoRA outcome.

## 1. Why this phase, and why now

Phase 3 established, at n = 20 paired seeds on Cora masked-edge prediction:

| contrast | Δ AUROC | p |
|---|---|---|
| pretrained + SPD − pretrained (Q1) | +0.029 | < 0.001 |
| pretrained + SPD − none (Q2) | −0.009 | 0.001 |
| pretrained + SPD − random + SPD (Q3) | +0.019 | < 0.001 |
| random + SPD − random | +0.002 | 0.46 |

The bias turned the frozen pretrained body from a 0.037 cost into a 0.009 cost, and the
pretrained body now beats the random-init one — but the LLM path is still (slightly,
significantly) below the body-less linear model. The one lever never pulled is adapting the
body itself: every Phase-2/3 arm trained 320 bias parameters and 33 RMSNorm gains inside the
LLM and nothing else. Phase 3B answers, once, whether low-rank adaptation closes the last gap,
and whether pretraining still matters when the body is adapted (a random-init body with LoRA
is the "train a transformer from a random start with a low-rank budget" control).

## 2. Mechanism

`peft` 0.20.0 (`inject_adapter_in_model`, in place, `task_type=None`): LoRA r = 16, α = 32,
dropout 0.05 on all seven projections (q, k, v, o, gate, up, down) of every layer —
≈ 11.3 M trainable parameters. B is zero-initialised, so at step 0 a LoRA model *is* the
Phase-3 model (inertness test, exact under `no_grad`). LoRA parameters form their own AdamW
group (`lr_lora`, weight decay 0); encoder / decoder at 1e-2, RMSNorm gains at 1e-4, bias table
at the Phase-3 selection, all unchanged. Backward already traverses the whole body to reach
the encoder, so LoRA adds only the adapter matmuls (probe measures the actual cost; expected
+10–20 % per step, no extra activation memory).

## 3. Arms — one array, 20 seeds everywhere (no seed rule needed)

| arm | bias | LoRA lr | runs |
|---|---|---|---|
| pretrained + SPD + LoRA | Phase-3 selection | {3e-5, 1e-4, 3e-4} | 60 |
| random + SPD + LoRA | Phase-3 selection for random | {3e-5, 1e-4, 3e-4} | 60 |
| pretrained + LoRA (bias off) | – | {3e-5, 1e-4, 3e-4} | 60 |
| pretrained + SPD + LoRA, shuffled-A | Phase-3 selection | 1e-4 | 20 |
| references at this commit: pretrained + SPD, random + SPD, none | Phase-3 selections | – | 60 |

260 runs; ≈ 20 GPU-h. The LoRA-lr grid is a decade around GTLM's 6e-5–3e-4; edge rule one
step, applied in the same submission style as Phase 3 (pre-registered here: if the best cell
sits on the grid edge, one ×3 step is run at 20 seeds before the write-up).

Selection: per arm, LoRA lr by mean val AUROC on seeds 0–9; test reported on all 20 seeds.

## 4. Gate 3B — questions

- **Q1′ does LoRA help?** Δ(pretrained + SPD + LoRA − pretrained + SPD) > 0.
- **Q2′ does the LLM path now reach or beat the linear model?** Δ(pretrained + SPD + LoRA −
  none) ≥ 0. This is the project's "LLM earns its place on Cora" criterion.
- **Q3′ does pretraining matter when the body is adapted?** Δ(pretrained + SPD + LoRA −
  random + SPD + LoRA) > 0.
- **2 × 2**: pretrained + LoRA without the bias vs with — is the structural channel still
  needed once the weights can move?
- Controls: shuffled-A clearly below; LoRA-B zero-init inertness (test + init-stats check);
  Phase-3 references reproduce at this commit.

Whatever the answers, Phase 4 (encoder study E1–E7 on Cora and the molecular datasets)
follows; Phase 3B fixes *which* body configuration the encoder study uses (default: the best
Q2′ configuration if it beats `none`, otherwise bias-only without LoRA).

## 5. Steps

1. Code: `FrozenBody(lora=LoraConfig|None)`; `make_optimizer(lr_lora)`; `run_phase3` gains
   `lora` / `lr_lora` keys (`_lora{lr:.0e}` in the run key); `configs/phase3b.yaml`;
   tests: LoRA params exist and receive gradients, base weights untouched after a step,
   zero-init inertness, param-group classification.
2. Laptop smoke on the 512-node subgraph; commit; rsync to Marlowe (new pinned commit —
   only after the Phase-3 top-up has finished writing rows); cluster `pytest`.
3. Probe job (step time, peak memory, LoRA gradient norm, loss halves) — read before the array.
4. Array; aggregate (`aggregate3` with a "+ LoRA" tag); edge rule; walkthrough; RESULTS.md;
   cleanup; commit; user pushes.
