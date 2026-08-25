# FPT — Can a frozen language-pretrained transformer process a non-text modality that enters through a learned projection?

Research notes for the **Graph-In / Graph-Out LLM** project.
Topic owner: `fpt`. Written 2026-08-25. Every claim below is tagged with the primary source I actually fetched.

**One-paragraph summary.** The FPT paper is real, its recipe is exactly what `CLAUDE.md` §9 says it is,
and its *narrow* claim — a **frozen** language-pretrained transformer beats a **frozen** randomly-initialised
one of identical architecture — survived independent replication. Its *headline* claim — frozen-pretrained
matches or beats fully training the transformer — **did not survive**: Rothermel et al. showed it was an
artefact of running every configuration at a single learning rate of `1e-3`. A closely analogous lineage in
time series (GPT4TS → Tan et al.) went the same way, and there the pretrained LLM turned out to contribute
essentially nothing once ablated properly. So FPT is still the right citation for "why should this work at
all", but it must be cited for the narrow claim, with the rebuttal cited by us rather than by a reviewer.
Separately, I found concrete numeric guidance on the thing `CLAUDE.md` currently says nothing about: the
encoder's output scale must be made to match the LLM's input-embedding norm, or the frozen body sees badly
out-of-distribution inputs and the graph tokens acquire "representational inertia".

---

## Verified facts (with source next to each)

### A. FPT — identity and provenance

| Fact | Source |
|---|---|
| arXiv **2103.05247** resolves. Title *"Pretrained Transformers as Universal Computation Engines"*. Authors Kevin Lu (UC Berkeley), Aditya Grover (FAIR), Pieter Abbeel (UC Berkeley), Igor Mordatch (Google Brain). | `arxiv.org/abs/2103.05247` |
| v1 = 9 Mar 2021, v2 = 30 Jun 2021 (current). v2 added §3.3 LSTM analysis, §3.10 depth/token-mixing, §3.11 freezing strategies; the GitHub repo was updated alongside. | `arxiv.org/abs/2103.05247`, Appendix A of ar5iv HTML |
| Peer-reviewed version: **AAAI 2022, vol. 36 no. 7, pp. 7628–7636**, DOI `10.1609/aaai.v36i7.20729`, retitled ***"Frozen Pretrained Transformers as Universal Computation Engines"***. | `ojs.aaai.org/index.php/AAAI/article/view/20729` |
| Repo `github.com/kzl/universal-computation` exists; 23 files; core is `universal_computation/fpt.py` (194 lines), `experiment.py` (254), `trainer.py` (95), `scripts/run.py` (34). | GitHub trees API + raw file fetches |

### B. Exactly which parameters are finetuned — VERBATIM from §2.2

`CLAUDE.md` §9 says "new linear input and output layers, layernorm and positional params finetuned".
**This is correct, including the layernorm-affine-only detail.** Verbatim from the paper:

> **Output layer:** "it is crucial to finetune the output layer since we are transferring to a completely
> new task – we use the simplest possible instantiation of an output network, being a single linear layer
> applied to the last output token output by the transformer... `n_dim × d_out` parameters."
>
> **Input layer:** "it is important to reinitialize a new input layer since we are reading in a new
> modality; in essence, **we are learning how to query the transformer**... `d_in × n_dim` parameters for
> the weight matrix/embeddings, and an additional `n_dim` parameters if there is a bias term."
>
> **Layer norm parameters:** "as is standard practice in other finetuning works (Rebuffi et al. 2017;
> Houlsby et al. 2019), we also finetune the **affine layer norm parameters (scale and bias)**, which
> adapt to the statistics of the downstream task in a new domain. In GPT-2, layer norm is applied twice
> per block, so these are a total of `4 × n_dim × n_layers` parameters."
>
> **Positional embeddings:** "While we observe that positional embeddings can be surprisingly universal
> between modalities... we generally see a small benefit to finetuning the positional embeddings which
> have a cheap parameter cost of `l × n_dim`."

**Frozen:** all self-attention and feedforward layers of every residual block. Also frozen (and unused):
the token embedding table `wte` — the code drives the model through `inputs_embeds`, never the tokenizer.
*(This exactly matches our `CLAUDE.md` §13 "No tokenizer, no embedding table.")*

> "Note that, crucially, **all communication between tokens in the model are frozen**. The data in each
> datapoint is chunked into discrete tokens... and can only reference each other via the frozen attention
> connections, which are not trained; additionally, neither the output nor the input layers are connected
> to multiple tokens." — §2.2

Fraction of parameters trained: **0.086 %** of the 124M base CIFAR-10 model, dropping to **0.029 %** for
GPT-2 XL. Table 8 gives absolute trained-parameter counts: 106K (12L), 190K (24L), 300K (36L).

### C. Tasks / modalities (seven, not five)

| Task | Sequence given to the transformer | Source |
|---|---|---|
| Bit Memory | 5 bitstrings of length 1000, one shown masked at p=0.5 → **120 tokens of dim 50** | §2.1 |
| Bit XOR | 2 bitstrings of length 5, one bit per token → **10 tokens of dim 1** | §2.1 |
| ListOps (from LRA) | **512 tokens of dim 15** | §2.1 |
| MNIST | 32×32 image, 4×4 patches → **64 tokens of dim 16** | §2.1 |
| CIFAR-10 | 4×4 patches → paper says "64 tokens of dimension 16"; **code says `input_dim = 3*patch_size**2 = 48`** | §2.1 vs `experiment.py:73` |
| CIFAR-10 LRA | grayscale, flattened, 1 pixel per token → **1024 tokens of dim 1** | §2.1 |
| Remote Homology (TAPE) | protein fold prediction; paper says 25 amino-acid types and 1195 labels; **code uses `input_dim, output_dim = 30, 1200`** | §2.1 vs `experiment.py:87` |

`CLAUDE.md` lists "bit strings, XOR, MNIST, CIFAR and protein folding" — correct but omits ListOps and
CIFAR-10 LRA. Those two matter to us because they are the *long-sequence, low-inductive-bias* tasks, which
is the regime our N≈500–2708 adjacency-row tokens live in.

### D. THE NUMBERS — frozen pretrained vs random init (our most important baseline)

**Table 2, §3.2 — all rows are 12-layer base models, all frozen, all training only
{input, output, positional, layernorm} parameters.** This is the apples-to-apples comparison.

| Model | Bit Memory | XOR | ListOps | MNIST | CIFAR-10 | C10 LRA | Homology |
|---|---|---|---|---|---|---|---|
| **FPT** (language-pretrained) | **100 %** | 100 % | **38.4 %** | **98.0 %** | **68.2 %** | **38.6 %** | **12.7 %** |
| **Random** (GPT-2 default init, frozen) | 75.8 % | 100 % | 34.3 % | 91.7 % | 61.7 % | 36.1 % | 9.3 % |
| Bit-Memory-pretrained (frozen) | 100 % | 100 % | 35.4 % | 97.8 % | 62.6 % | 36.7 % | 7.8 % |
| ViT (ImageNet-21k, frozen) | 100 % | 100 % | 37.4 % | 97.8 % | 72.5 % | 43.0 % | 7.5 % |

**Deltas of pretrained over random-init:** Bit Memory **+24.2**, XOR **0.0**, ListOps **+4.1**,
MNIST **+6.3**, CIFAR-10 **+6.5**, CIFAR-10 LRA **+2.5**, Homology **+3.4**.

Two things to internalise:
1. The gap is real but **modest on the long-sequence tasks** (+2.5 on C10 LRA, +4.1 on ListOps) — the tasks
   most like ours.
2. The paper concedes the random transformer is strong: *"while the transformer architecture might be
   naturally conducive to these evaluations, the attention mechanisms used to transfer may be nontrivial and
   not fully specified by the architecture."* On MNIST it notes random transformers *"achieve similar
   performance to a linear classifier on top of raw features (92 %)"*.

**Table 1, §3.1 — FPT vs fully-trained transformer vs fully-trained LSTM.** Beware: heterogeneous models.

| Model | Bit Memory | XOR | ListOps | MNIST | CIFAR-10 | C10 LRA | Homology |
|---|---|---|---|---|---|---|---|
| FPT | 100 % | 100 % | 38.4 % | 98.0 % | **72.1 %** | 38.6 % | 12.7 % |
| Full (trained from scratch) | 100 % | 100 % | 38 % | 99.1 % | 70.3 % | 42 % | 9 % |
| LSTM (fully trained) | 60.9 % | 50.1 % | 17.1 % | 99.5 % | 73.6 % | 11.7 % | 12 % |

Appendix D.1 caveats, which the summary table hides:
- **CIFAR-10 FPT 72.1 % is a 36-layer GPT-2-large**, not the base model (base = 68.2 %). "Full" for
  CIFAR-10 is a **3-layer** GPT-2 trained end-to-end, chosen because 12 layers was unstable.
- ListOps and C10 LRA "Full" numbers are **quoted from Tay et al. 2020** (3-layer vanilla transformer with
  *"extensive sweeps over different hyper-parameters"*), not run by the authors.
- Homology "Full" is quoted from Rao et al. 2019 (12-layer, 512-dim).
- The authors state they *did not tune FPT hyperparameters at all* except for Remote Homology.

### E. Ablations — what FPT says about WHY it works

**Table 15/19, §3.12 — successive parameter ablation, pretrained frozen transformer.** Layernorm is the
single most important thing to unfreeze.

| Task | output only | + layernorm | + input | + positions |
|---|---|---|---|---|
| Bit Memory | 76 % | 94 % | 100 % | 100 % |
| Bit XOR | 56 % | 98 % | 98 % | 100 % |
| ListOps | 15 % | 36 % | 36 % | 38 % |
| MNIST | 23 % | 96 % | 98 % | 98 % |
| CIFAR-10 | 25 % | 54 % | 60 % | 68 % |
| CIFAR-10 LRA | 17 % | 39 % | 39 % | 39 % |
| Homology | 2 % | 9 % | 10 % | 13 % |

Paper's own words: *"Similar to a study of random CNNs by Frankle et al. 2020, we generally find the layer
norm parameters to be most important."*

**Table 21 — the same ablation on a RANDOM frozen transformer.** This is the deflationary result:

| Task | output only | + layernorm | + input | + positions |
|---|---|---|---|---|
| Bit XOR | 50 % | **100 %** | 100 % | 100 % |
| ListOps | 17 % | **35 %** | 36 % | 37 % |
| MNIST | 25 % | **83 %** | 92 % | 92 % |
| CIFAR-10 | 20 % | **46 %** | 56 % | 62 % |
| C10 LRA | 11 % | **34 %** | 36 % | 36 % |

Turning on layernorm alone takes a *random* frozen GPT-2 from 20 % → 46 % on CIFAR-10 and 11 % → 34 % on
C10 LRA. Most of the "magic" is the affine LN parameters modulating random features, not the pretrained
weights. Compare **Frankle, Schwab & Morcos, ICLR 2021 (arXiv 2003.00152)**: training *only* BatchNorm
affine parameters on a **randomly initialised** ResNet reaches **82 % on CIFAR-10** and 32 % ImageNet top-5
— higher than FPT's 72.1 %. FPT §4.3 acknowledges this: *"Their numbers are stronger than ours on
CIFAR-10, but include significantly more inductive bias via a convolutional architecture."*

**Table 16, §3.13 — is finetuning layernorm necessary?** CIFAR-10, only input+output trained:

| Init | Frozen LayerNorm | Finetuned LayerNorm |
|---|---|---|
| Pretrained | 61.5 % | 68.2 % |
| Random | 55.0 % | 61.7 % |

The pretrained-over-random gap is **+6.5 with LN tuned and +6.5 with LN frozen** — so the *gap* is not
caused by LN tuning, but LN tuning is worth ~6.5 points on top for both. This is the specific evidence
against the "the gains are just layernorm finetuning" critique, and it is worth quoting if a reviewer
raises it.

**Table 9, §3.8 — "is it just better initialisation statistics?"** They re-initialised a random transformer
using the pretrained model's per-layer mean and std.

| Init | Memory | XOR | ListOps | MNIST | C10 | C10 LRA | Homology |
|---|---|---|---|---|---|---|---|
| Pretrained | 100 % | 100 % | 38.4 % | 98.0 % | **68.2 %** | **38.6 %** | 12.7 % |
| Statistics Only | 100 % | 100 % | 37.4 % | 97.2 % | 56.5 % | 33.1 % | 11.0 % |
| Default random | 75.8 % | 100 % | 34.3 % | 91.7 % | 61.7 % | 36.1 % | 9.3 % |

Statistics-matching recovers most of the gap on 5/7 tasks but is **worse than plain random init** on
CIFAR-10 (56.5 vs 61.7) and C10 LRA (33.1 vs 36.1). Paper's conclusion: the benefit *"cannot be recovered
with a simple better initialisation scheme."* Honest read: it *mostly* can, except on the two image tasks.

**Table 11/12, §3.10 — depth and token mixing (ListOps).** With only output+layernorm trained:

| Layers | Pretrained | Random |
|---|---|---|
| 1 | 17 % | 17 % |
| 2 | **36 %** | 16 % |
| 6 | 38 % | 35 % |

With **only** the output layer trained (pure reservoir / linear probe; input layer random and frozen):

| Layers | Pretrained | Random |
|---|---|---|
| 1 | 12 % | – |
| 3 | 18 % | – |
| 6 | 33 % | – |
| 12 | **33 %** | 17 % |
| 24 | – | 17 % |

This is the strongest positive evidence in the whole paper: with *nothing* trainable that can mix tokens,
pretrained attention reaches 33 % on ListOps and random attention is stuck at 17 % regardless of depth.
**Pretrained attention mixes tokens usefully; random attention does not.** For us that is precisely the
claim that matters — our encoder produces per-node tokens and *all* cross-node computation happens in the
frozen attention.

**Table 6, §3.4 — compute efficiency (gradient steps to convergence, same LR and batch size).**

| | Memory | XOR | ListOps | MNIST | C10 | C10 LRA | Homology |
|---|---|---|---|---|---|---|---|
| FPT | 1e4 | 5e2 | 2e3 | 5e3 | 4e5 | 3e5 | 1e5 |
| Random | 4e4 | 2e4 | 6e3 | 2e4 | 4e5 | 6e5 | 1e5 |
| **Speedup** | 4× | 40× | 3× | 4× | 1× | 2× | 1× |

**Table 13, §3.11 — unfreezing more.** *Read this before deciding what LoRA touches.*

| Model | Memory | XOR | ListOps | MNIST | C10 | C10 LRA | Homology |
|---|---|---|---|---|---|---|---|
| FPT | 100 % | 100 % | 38.4 % | 98.0 % | 68.2 % | 38.6 % | 12.7 % |
| + Feedforward | 100 % | 100 % | 36.0 % | 98.3 % | **76.6 %** | 38.2 % | 13.1 % |
| + Attention | 100 % | 100 % | 36.8 % | 89.0 %† | 47.7 %† | 23.0 % | 10.9 % |
| + Both | 100 % | 100 % | 35.8 % | 93.1 %† | 32.9 % | 21.0 % | 10.5 % |

† = training diverged, number reported before divergence. Table 14: FPT 68.2 → + all FF layers 76.6 →
+ last attention layer **80.0 %** on CIFAR-10. Caveat, verbatim: *"We do not use a per-layer learning
scheme/etc."* — i.e. the divergence is a single-LR artefact, the same disease as the headline result.

**Table 8, §3.7 — scaling.** Frozen scaling is monotone and stable, unlike full training.

| Size | Layers | Total params | Trained params | FPT | Random |
|---|---|---|---|---|---|
| Small (base) | 12 | 117M | 106K | 68.2 % | 61.7 % |
| Medium | 24 | 345M | 190K | 69.8 % | 64.0 % |
| Large | 36 | 774M | 300K | 72.1 % | 65.7 % |

**Table 17, §3.14 — backbone swap (CIFAR-10 / ListOps).** GPT-2 68.2 / 38.4; BERT 68.8 / 38.3;
T5-encoder 64.7 / **15.4**; Longformer 66.8 / **17.0**. Backbone choice matters a lot on the long-sequence
task — and note that the two non-causal / long-context backbones are the ones that collapsed.

**Table 10, §3.9 — output-layer-only "reservoir" mode.** ListOps 32.8 % (vs FPT 38.4), C10 LRA 24.7 %
(vs 38.6), with a **500–2000× wall-clock speedup per epoch** after the first (features cached).

**Table 7, §3.6 — FPT underfits.** C10 LRA: FPT (12L) test 38.6 / train 38.5; vanilla transformer (3L)
test 42 / train 70; Linformer (3L) test 39 / train 97.

### F. FPT hyperparameters (Appendix C + code)

- Optimizer **Adam**, PyTorch defaults (β=(0.9, 0.999), eps=1e-8), **no weight decay**.
- **LR = 1e-3 for all transformer models, no schedule.** 1e-4 for Remote Homology only.
- LSTMs: 3e-4.
- Batch size *"largest that fits on an RTX 2080 Ti, somewhere between 2 and 16"*, no gradient accumulation
  (the released code does support `grad_accumulate`).
- Gradient clipping `clip_grad_norm_(..., 1.0)` (from `trainer.py`; not stated in the paper).
- Dropout 0.1; input projection uses **orthogonal init with gain 1.41**, bias zeroed.
- HuggingFace `transformers`; models trained *"to convergence"*, evaluated on a held-out test set.
- **No validation set** in the released code — test accuracy is reported directly (flagged by the rebuttal).

---

## Corrections to CLAUDE.md

### C1. §9 FPT entry — factually correct, but the framing overclaims. **(certain)**

> "**This is the justification that a frozen text LLM can process a non-text modality entering through a
> linear projection.** Cite it when asked why this should work at all."

The parameter list and modality list in `CLAUDE.md` are accurate (verified verbatim above). The framing is
not safe. **FPT's headline claim was overturned in a direct replication.**

Rothermel, Li, Rocktäschel & Foerster, ***"Don't Sweep your Learning Rate under the Rug: A Closer Look at
Cross-modal Transfer of Pretrained Transformers"***, arXiv **2107.12460** (FAIR / UW / UCL, 26 Jul 2021).
They swept LR logarithmically from `1e-6` to `1e-2`, 3 seeds, created proper validation splits, used
val-based early stopping. Their Table 1 (test accuracy, mean ± SEM over 3 seeds, GPT-2 base 12L):

| | ListOps | MNIST | CIFAR10 | CIFAR10-LRA |
|---|---|---|---|---|
| Frozen Random | 38.9 ± 0.3 | 98.0 ± 0.0 | 61.8 ± 0.2 | 44.2 ± 0.3 |
| **Frozen Pretrained** | 46.1 ± 0.3 | 98.5 ± 0.1 | 66.3 ± 0.0 | 54.7 ± 1.4 |
| Unfrozen Random | 57.6 ± 0.8 | 98.7 ± 0.0 | **77.8 ± 0.2** | 62.0 ± 0.7 |
| **Unfrozen Pretrained** | 56.3 ± 0.9 | **99.0 ± 0.0** | 77.7 ± 0.1 | **67.8 ± 0.3** |

Verbatim: *"we find that this result is, in fact, an artefact of not tuning the learning rates... when
tuning learning rates properly, pretrained transformers do outperform or match training from scratch in all
of our tasks, but only as long as the entire model is fine-tuned."* And:
*"Since 1×10⁻³ is just before a precipitous drop in the performance of the unfrozen transformers, had the
authors picked a lower learning rate they would have arrived at very different conclusions."*
On ListOps: *"Each of the LRs evaluated between 1×10⁻⁵ and 1×10⁻³ results in different orderings."*

**What survives, and is what we should cite FPT for:**
- **Frozen Pretrained > Frozen Random on all four tasks they re-ran.** Explicitly confirmed:
  *"For each task, the Frozen Pretrained setting outperforms the Frozen Random setting."* Their gaps are
  *larger* than FPT's own: ListOps +7.2, CIFAR-10 +4.5, C10-LRA +10.5, MNIST +0.5.
- Pretraining improves compute efficiency for the frozen variants (their Table 2: Frozen Random **never**
  reaches Frozen Pretrained's best accuracy on any task).
- Pretraining also helps the *unfrozen* variant on C10-LRA (+4.8 %) and MNIST.

**What does not survive:** "frozen matches or beats fully trained". Also overturned: FPT §3.11's claim that
unfreezing FF+attention hurts (that is Rothermel's best configuration once LR is swept), and FPT §3.6's
claim that frozen models always underfit (*"the Frozen Pretrained variant has the largest train/test gap
of all settings on ListOps"*).

Setup details worth copying: max gradient steps ListOps 3e5, MNIST 4e5, CIFAR10 4e5, C10-LRA 3e5; batch
size 8. Also they **fixed two bugs in Lu et al.'s ListOps loader** (sequences truncated to 512 when the
dataset has 500–2000 tokens; the closing parenthesis silently dropped because it is not alphanumeric) — so
the released FPT ListOps numbers are on a broken dataset.

**Suggested replacement wording for `CLAUDE.md` §9:**
> FPT is the precedent that a **frozen** language-pretrained transformer, fed a non-text modality through a
> trained linear input layer, beats an identically-frozen **randomly initialised** transformer of the same
> size on seven non-text tasks (Lu et al. 2021, Table 2: +6.5 CIFAR-10, +4.1 ListOps, +2.5 CIFAR-10-LRA;
> independently replicated with larger gaps by Rothermel et al. 2021, Table 1). FPT's stronger claim — that
> frozen matches full training — was shown by Rothermel et al. (arXiv 2107.12460) to be an artefact of a
> single fixed learning rate of 1e-3. Cite FPT for the narrow claim and cite the rebuttal ourselves.

### C2. §7 baseline 6 and §8 "Random-init LLM" conflate two different controls. **(certain)**

`CLAUDE.md` §7.6 says "Transformer from scratch — adjacency-row tokens into a randomly-initialised
transformer, no pretrained weights", and §8 says "Random-init LLM — same architecture, pretrained weights
discarded". These are two *different* experiments and the FPT literature shows they give **opposite
answers**:

- **Frozen-random** (FPT's "Random"): body randomly initialised **and frozen**; only encoder/decoder/bias/LN
  trained. This is the control that isolates pretrained knowledge from architecture. Pretrained wins here
  (Lu Table 2; Rothermel Table 1, all four tasks).
- **Unfrozen-random** (train the whole thing end to end): pretrained does **not** reliably win.
  Rothermel: Unfrozen Random 77.8 vs Unfrozen Pretrained 77.7 on CIFAR-10, and 57.6 vs 56.3 on ListOps.

**Action: run both, and label them distinctly.** §8's `Random-init LLM` row should become two rows:
`Random-init LLM (frozen)` and `Random-init LLM (fully trained)`.

### C3. §8 is missing the two decisive ablations a modern reviewer will demand. **(certain)**

Tan, Merrill, Gupta, Althoff & Hartvigsen, ***"Are Language Models Actually Useful for Time Series
Forecasting?"***, arXiv **2406.16964**, **NeurIPS 2024 Spotlight**. This is the closest thing in the
literature to a dry run of our project's failure mode: GPT4TS / Time-LLM / CALF all use the FPT recipe on a
non-text modality, and this paper ablated the LLM out.

Their three ablations, all of which we should implement:
- **`w/o LLM`** — delete the language model entirely, pass the encoder's tokens straight to the head.
- **`LLM2Attn`** — replace the LLM with **one randomly-initialised multi-head attention layer**.
- **`LLM2Trsf`** — replace the LLM with **one randomly-initialised transformer block**.

Result, verbatim: *"removing the LLM component or replacing it with a basic attention layer does not
degrade forecasting performance — in most cases, the results even improve!"* Across 13 datasets × 2 metrics,
ablations beat **Time-LLM in 26/26**, **CALF in 22/26**, **OneFitsAll in 19/26**.

Their Table 5 (CALF / GPT-2 backbone, 8 datasets × {MAE, MSE} = 16 cells, count of wins):

| Pre + FT | woPre + FT | Pre + woFT | woPre + woFT |
|---|---|---|---|
| **3** | **8** | 5 | 0 |

i.e. *randomly initialising the LLM and training it from scratch won more often than using the pretrained
weights*. They also shuffled the input sequence (`sf-all`, `sf-half`, `ex-half`) and found LLM-based methods
were **no more sensitive to shuffling than their ablations** — i.e. the LLM was not contributing sequence
modelling. *This is exactly our §8 "Shuffled adjacency" control, and it is the control that killed a whole
subfield.* Few-shot: at 10 % of data, LLaMA vs `w/o LLM` was 8 wins each.

**Counter-evidence, for honesty:** Zhou, Niu, Wang, Sun & Jin, ***"One Fits All: Power General Time Series
Analysis by Pretrained LM"*** (GPT4TS), arXiv **2302.11939**, **NeurIPS 2023 Spotlight** — the direct target
of the above — reports the opposite in the **low-data regime**. Their Table 7 (10 % data, MSE averaged over
4 prediction lengths):

| Dataset | GPT2(6) FPT | GPT2(0) | No Freeze | No Pretrain |
|---|---|---|---|---|
| Weather | **0.237** | 0.263 | 0.273 | 0.277 |
| ETTh1 | **0.427** | 0.874 | 0.753 | 1.326 |
| ETTh2 | **0.346** | 0.666 | 0.447 | 0.502 |

At 5 % data (their Table 20), `No Pretrain` on ETTh1 blows up to MSE 2.968 at horizon 336 vs 0.754 for the
frozen pretrained model. **Reconciliation: pretraining appears to buy a lot in the low-data / few-shot
regime and roughly nothing at full data.** Cora edge prediction is a *small* dataset — that regime is on our
side — but this must be argued explicitly, not assumed.

GPT4TS is also a useful template: it is the FPT recipe applied to a continuous non-text modality. Verbatim:
*"Self-attention and Feedforward layers in the transformer blocks are frozen while only the embedding
layer, normalization layers, and output layer require training"*, and *"we fine-tune the positional
embeddings and layer normalization layer, which is considered a standard practice (Lu et al. 2022; Houlsby
et al. 2019)"*. They add **reverse instance normalisation** on the input — a normalisation trick worth
stealing (see Mechanism below).

### C4. §11 has no pitfall about the encoder's output scale. **(certain — a documented failure mode)**

Nothing in `CLAUDE.md` says the encoder's output distribution must match the LLM's input-embedding
statistics. The frozen body was trained on inputs with a very specific scale; hand it vectors 30–70× too
long and the graph tokens will barely move through the network. Numbers and the fix are in
"CONCRETE: what our encoder should do at init" below. **Add a §11 pitfall.**

### C5. §7 / §12 do not mandate a learning-rate sweep. **(certain)**

Given C1 — the single most-cited result in this exact space was overturned purely because of a fixed LR —
running our headline comparison (`frozen pretrained LLM` vs `random-init` vs `no LLM`) at one LR would make
the paper indefensible. **Add to §12: every configuration in the §8 grid gets its own LR sweep, ≥3 seeds, a
real validation split, and val-based early stopping. Report mean ± SEM.** Note FPT's released code has no
validation split at all.

### C6. Minor factual nits in the FPT paper itself — do not copy its arithmetic. **(certain)**

- §2.2 states `4·768·12 = 36684`. Correct value is **36864**.
- §2.2 states `64·768 = 49512`. Correct value is **49152**.
- §2.2 calls the base model "124M parameters"; Appendix B.5 Table 22 says **117M** for the 12-layer base.
  (124M is GPT-2-small *including* `wte`; 117M excludes it.) With the corrected arithmetic,
  `7680 + 13056 + 36864 + 49152 = 106,752`, and `106752 / 124M = 0.086 %` — so the 0.086 % figure is right
  and is computed against 124M.
- §2.1 says CIFAR-10 gives "64 tokens of dimension 16", but `experiment.py` uses
  `input_dim = 3 * patch_size**2` = 48 for RGB. Homology: paper says 25 input types / 1195 labels, code uses
  `input_dim, output_dim = 30, 1200`.

### C7. Things in §9 / §1 I checked and found CORRECT — no change needed.

Every arXiv id resolves to the stated title (I fetched each and read `citation_title`):

| id | Resolved title |
|---|---|
| 2103.05247 | Pretrained Transformers as Universal Computation Engines ✓ |
| 2605.10247 | Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning ✓ |
| 2606.15633 | Formalizing and Mitigating Structural Distortion in LLM Attention for Graph Reasoning — live, but the title does not contain "GaLA"; verify the shorthand with the owner of that topic |
| 2402.05862 | Let Your Graph Do the Talking: Encoding Structured Data for LLMs ✓ (= GraphToken) |
| 2503.01805 | Depth-Width tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers ✓ |
| 2501.01073 | Graph Generative Pre-trained Transformer ✓ (= G2PT) |
| 2310.04560 | Talk like a Graph: Encoding Graphs for Large Language Models ✓ |

---

## Mechanism / math (exact formulas, tensor shapes)

### FPT's data path, in our shapes

```
x            [B, L, d_in]     L tokens of raw modality        (ours: [B, N, N] adjacency rows)
in_net(x)    [B, L, n_dim]    trained linear (+ optional MLP) (ours: GraphEncoder -> [B, N, d_model])
  + wpe      [B, L, n_dim]    positional embeddings, trained  (ours: RoPE reset / structural bias)
GPT2Model(inputs_embeds=..)   FROZEN attn + FFN; LN affine trained
             [B, L, n_dim]
out_net(h)   [B, L, d_out]    trained linear                  (ours: GraphDecoder -> [B, N, N])
```

The tokenizer and `wte` never appear. `GPT2Model.forward(inputs_embeds=x)` is the entire interface.

### The frozen attention block being reused (FPT Appendix B.1–B.3)

```
y_i = sum_j softmax_j( q_i . k_j / sqrt(d_k) ) * v_j       # causal mask in GPT-2
LayerNorm:  y~_i = (x_i - mean(x)) / std(x)
            y_i  = gamma_i * y~_i + beta_i                 # gamma, beta = the ONLY trained body params
```

LN is applied twice per block (pre-attention `ln_1`, pre-MLP `ln_2`), plus a final `ln_f`; hence
`4 * n_layers * n_dim` (counting gamma and beta at both sites). Our attention bias enters at exactly the
pre-softmax point: `score(i,j) = q_i·k_j/sqrt(d) + bias(i,j)`, compatible with everything above.

### Parameter budget, instantiated for our project

Let `d_model` = LLM hidden size, `L` = layers, `N` = nodes, `F` = feature dim.

```
encoder  E1 Linear(N -> d_model)              N * d_model + d_model
encoder  E4 Linear(N+F -> d_model)            (N+F) * d_model + d_model
LN affine (FPT-style, if we tune them)        4 * L * d_model
decoder  D1 W                                 d_model^2      (or 2*r*d_model low-rank)
bias table (GTLM-style)                       ~1.7e5
LoRA r=32                                     ~2*r*d_model*(#target matrices)*L
```

For Llama-3.2-1B (`d_model=2048`, `L=16`) and Cora (`N=2708`, `F=1433`):
`E4 ≈ (2708+1433)·2048 ≈ 8.48M`, `D1 = 2048² ≈ 4.19M`, `LN affine = 4·16·2048 = 131K`.
Encoder + decoder alone is ~12.7M trainable — **two orders of magnitude more trainable parameters than FPT
used (106K)**. We are much further from "the frozen body does all the work" than FPT was, so the `w/o LLM`
ablation (C3) is not optional: a 12.7M-parameter encoder+decoder pair may well solve Cora link prediction
on its own.

### Norm alignment at the modality interface (arXiv 2512.08374)

Target norm, computed once from the frozen LLM's embedding matrix `W_e`:

```
T = (1/|W*|) * sum_{w in W*} ||w||_2 ,   W* = { w in W_e : ||w||_2 > eps }
```

Insert a `LayerNorm(d_model)` after the encoder, with the **gain initialised to a scalar**:

```
g_init = T / sqrt(d_model)        # scalar, broadcast to all d_model entries
beta_init = 0
```

Rationale: after LayerNorm's normalise step `||x_hat||_2 ≈ sqrt(d_model)` (unit variance per dim), so
multiplying by `g = T/sqrt(d_model)` gives output L2 norm ≈ `T`, matching the text embeddings the frozen
body expects.

**The trap:** `T ≈ 1` even at `d_model = 4096`, so `g_init ≈ 0.01–0.02`. The backward pass through the LN
gain is `grad_x_hat = grad_y ⊙ g`, so a gain of 0.015 attenuates the encoder's gradient ~65×. Verbatim:
*"a minute initialization triggers a vanishing gradient problem, detaching the vision encoder from
supervision."* Their fix is a backward hook — **Global Weight Compensation (GWC)**:

```
g_bar = (1/D) * sum_i |g_i|
mu    = max(g_bar, delta),  delta = 1e-3
grad_x_hat  <-  (grad_y (*) g) * (1 / mu)     # forward scale preserved, backward scale restored to ~1
```

### Reverse instance normalisation (GPT4TS / RevIN, from Kim et al. 2022)

*"This normalization block simply normalizes the input time series using mean and variance, and then adds
them back to the output."* — GPT4TS. The graph-domain analogue: normalise the adjacency row (e.g. by degree)
before the encoder, and add the degree statistic back before the decoder, so the frozen body only ever sees
standardised rows. Cheap to try, and it makes E1's behaviour independent of graph density.

---

## CONCRETE: what our encoder should do at init

This is the actionable answer to task item 4, with sources.

**1. Orthogonal init on every linear layer in the encoder, gain 1.41, bias zero.**
FPT §3.12 verbatim: *"we initialize the input layers as Gaussian if embeddings are used, or use an
**orthogonal initialization** for linear layers; in particular, we find orthogonal initialization to be
**very important** when input parameters are not trained."* Default in their code: `orth_gain=1.41`
(≈ `sqrt(2)`, the ReLU gain). ORCA independently uses `xavier_uniform_(w, gain=sqrt(2))` for its conv
projection, bias 0. Same family.

**2. Put a LayerNorm on the encoder output. This is not optional in our setting.**
- ORCA's `Embeddings1D` does `projection -> transpose -> LayerNorm(embed_dim) -> + positional embeddings`.
- LLaVA does **not** normalise, and LLaVA-v1.5 is the model with the *worst* measured norm mismatch:
  visual tokens `39.96 ± 45.58` vs text `1.08` (arXiv 2512.08374 Table 2). Its projector actively made the
  norm mismatch worse (28.71 before → 39.96 after).
- Adding the LayerNorm on top of LLaVA-1.5 with Llama-3.2-3B gave +3.47 MM-Star, +4.40 SEED-Bench-2,
  +4.90 OCRBench, and it also improved **text-only** MMLU (arXiv 2512.08374 Table 4). That last part matters
  to us: getting the graph tokens' scale right stops them from wrecking the frozen model's own machinery,
  which is exactly the concern GTLM addresses from the other direction with its intra-node zero bias.

**3. Initialise that LayerNorm's gain to `T / sqrt(d_model)`, not to 1.**
`T` = mean L2 norm of the non-zero rows of the LLM's `embed_tokens.weight`. Published values of `T`:
Qwen2.5-7B **0.80** (D=3584), Llama3.2-3B **1.09** (D=3072), Qwen3-8B **1.38** (D=4096) — so
`g_init` lands around **0.014–0.022**. Ablation evidence that this matters (arXiv 2512.08374 Table 5): with
default `gain=1`, after LLaVA stage-1 pretraining the gain's mean |g| was still 0.9609 — *"the parameters of
the default-initialized layer remained largely unchanged from their initial state, indicating that the
optimization process failed to begin effectively without a reasonable starting point."* With the
norm-matched init, mean |g| moved to 0.0400.

**4. Watch for the vanishing gradient into the encoder, and apply GWC if it appears.**
On Llama-3.2 the naive small-gain init was fine. On Qwen2.5 it was not: multimodal metrics *dropped*
(MMBench −1.20, MM-Star −2.26) until they added the backward-hook compensation. **Diagnostic to log from
step 0:** `encoder_weight.grad.norm()`. If it is orders of magnitude below the decoder's, switch on GWC.

**5. Assert the output statistics at init, in a unit test.**
The whole point is that this is a *measurable* property, not a vibe.

```python
def test_encoder_output_matches_embedding_scale(encoder, llm, tol=0.25):
    W = llm.get_input_embeddings().weight.detach().float()
    n = W.norm(dim=-1); T = n[n > 1e-6].mean()
    toks = encoder(A_batch, X_batch)                       # [B, N, d_model]
    got  = toks.detach().float().norm(dim=-1).mean()
    assert abs(got - T) / T < tol, f"encoder RMS {got:.3f} vs embedding {T:.3f}"
```

**6. Optional but cheap: an ORCA-style distribution-alignment warm-up before task training.**
ORCA trains the embedder *alone* first, to minimise a distributional distance between its outputs and real
in-modality features, before any task loss. Verbatim finding: *"as the dataset distance decreases, the
fine-tuning accuracy increases"*, and *"models with smaller final OTDDs have better fine-tuning accuracy"*.
Cost: *"our embedder learning time is only ~10 % of the fine-tuning time."* They compared three metrics —
pairwise Euclidean, MMD, and OTDD — and **all three beat no alignment**; OTDD was best. Our task has no
per-token discrete label, so plain MMD or moment matching against a random sample of `embed_tokens.weight`
rows is the practical version.

**7. Do NOT copy the LLM's embedding-level LayerNorm — GPT-2 and Llama don't have one.**
ORCA's `embedder_init` copies `source.LayerNorm.weight/bias` from RoBERTa/BERT, which *do* normalise their
embedding output. GPT-2's first norm is `h.0.ln_1` (inside the block), and Llama's is
`layers.0.input_layernorm`. There is nothing at the embedding to copy, so the `T/sqrt(D)` construction is
the substitute. **This is the one place ORCA's recipe does not port directly.**

**8. Corroborating evidence from soft prompting.** Lester et al. (arXiv 2104.08691, "The Power of Scale for
Parameter-Efficient Prompt Tuning") ablated soft-prompt initialisation: random uniform `[-0.5, 0.5]` vs
embeddings sampled from the 5,000 most common vocabulary tokens vs class-label embeddings. Class-label
init was best, and: *"At smaller model sizes, there are large gaps between the different initializations,
but once the model is scaled to XXL size, those differences disappear."* Read honestly: initialising in the
embedding distribution matters **most at small model scale** — and we are planning on a 1B backbone, i.e.
the regime where it matters.

---

## Code we can reuse (real snippets, real signatures)

### 1. FPT's model wrapper — `kzl/universal-computation/universal_computation/fpt.py`

Real constructor signature, verbatim:

```python
class FPT(nn.Module):
    def __init__(
            self,
            input_dim, output_dim,
            model_name='gpt2',            # 'gpt2'|'gpt2-medium'|'gpt2-large'|'gpt2-xl'|'vit'|'lstm'
            pretrained=False,
            return_last_only=True,
            use_embeddings_for_in=False,
            in_layer_sizes=None,          # None => single Linear; [32,32] => 2-layer MLP
            out_layer_sizes=None,
            freeze_trans=True,
            freeze_in=False, freeze_pos=False, freeze_ln=False,
            freeze_attn=True, freeze_ff=True, freeze_out=False,
            dropout=0.1,
            orth_gain=1.41,               # orthogonal initialization of input layer
    ):
```

**The freeze logic — copy this pattern for our `models/llm_wrapper.py`.** Note it is substring matching on
parameter names, and the `else` branch is what silently freezes `wte`:

```python
if freeze_trans:
    for name, p in self.sequence_model.named_parameters():
        name = name.lower()
        if 'ln' in name or 'norm' in name:
            p.requires_grad = not freeze_ln
        elif 'wpe' in name or 'position_embeddings' in name or 'pos_drop' in name:
            p.requires_grad = not freeze_pos
        elif 'mlp' in name:
            p.requires_grad = not freeze_ff
        elif 'attn' in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = False      # <-- this is what freezes wte (the token embedding table)
```

**Porting to Llama:** the substrings differ but the same block works. Llama uses `input_layernorm`,
`post_attention_layernorm`, `model.norm` (all match `'norm'`), `self_attn` (matches `'attn'`),
`mlp.gate_proj`/`up_proj`/`down_proj` (match `'mlp'`), `embed_tokens` (falls to `else`), and has **no
`wpe`** (RoPE is computed, not stored) — so `freeze_pos` becomes a no-op. Positional information for us has
to come from the graph bias, exactly as `CLAUDE.md` §2 already specifies.

**The input projection — note there is NO normalisation on its output:**

```python
in_layers = []
last_output_size = input_dim
for size in self.in_layer_sizes:
    layer = nn.Linear(last_output_size, size)
    if orth_gain is not None:
        torch.nn.init.orthogonal_(layer.weight, gain=orth_gain)
    layer.bias.data.zero_()
    in_layers.append(layer); in_layers.append(nn.ReLU()); in_layers.append(nn.Dropout(dropout))
    last_output_size = size

final_linear = nn.Linear(last_output_size, embedding_size)
if orth_gain is not None:
    torch.nn.init.orthogonal_(final_linear.weight, gain=orth_gain)
final_linear.bias.data.zero_()
in_layers.append(final_linear); in_layers.append(nn.Dropout(dropout))
self.in_net = nn.Sequential(*in_layers)
```

**The forward — the only interface into the frozen body:**

```python
x = self.in_net(x)
transformer_outputs = self.sequence_model(inputs_embeds=x, return_dict=True)
x = transformer_outputs.last_hidden_state
if self.return_last_only:
    x = x[:, -ratio:]
x = self.out_net(x)
```

`inputs_embeds=` is the whole trick. For us:
`llm(inputs_embeds=tokens, attention_mask=bidirectional_mask)` plus our bias hooks.

### 2. FPT's trainer — the thing that caused the problem

```python
self.optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   # ONE LR for everything
...
torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
self.optim.step()
self.optim.zero_grad()
```

`model.parameters()` at a **single** LR of `1e-3` covers the input layer, output layer, positional
embeddings *and* the LN affine params. **Do not copy this.** Use param groups (§6 below).

### 3. ORCA's embedder — the best template for our encoder (`sjunhongshen/ORCA/src/embedder.py`)

A 1-D encoder for an arbitrary continuous modality feeding a frozen RoBERTa. It **does** normalise.

```python
class Embeddings1D(nn.Module):
    def __init__(self, input_shape, embed_dim=768, target_seq_len=64, config=None, dense=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.stack_num = self.get_stack_num(input_shape[-1], target_seq_len)
        self.norm = nn.LayerNorm(embed_dim)                       # <-- LayerNorm on projector output
        self.padding_idx = 1
        self.position_embeddings = nn.Embedding(target_seq_len, embed_dim, padding_idx=self.padding_idx)
        self.projection = nn.Conv1d(input_shape[1], embed_dim,
                                    kernel_size=self.stack_num, stride=self.stack_num)
        conv_init(self.projection)

    def forward(self, x=None, inputs_embeds=None, *args, **kwargs):
        if x is None: x = inputs_embeds
        b, c, l = x.shape
        x = self.projection(x).transpose(1, 2)   # [B, S, D]
        x = self.norm(x)                          # <-- normalise BEFORE the frozen body
        position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        self.ps = self.position_embeddings(position_ids)
        x = x + self.ps
        return x
```

and its init (`src/utils.py`) — note it copies the pretrained model's own embedding-LayerNorm and
positional embeddings into the new embedder, and has an explicit statistics-matching option:

```python
def conv_init(m):
    if m.__class__.__name__.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None: init.constant_(m.bias, 0)

def embedder_init(source, target, train_embedder=False, match_stats=False):
    if train_embedder:
        if hasattr(source, 'patch_embeddings'):
            if match_stats:
                weight_mean, weight_std = (source.patch_embeddings.projection.weight.mean(),
                                           source.patch_embeddings.projection.weight.std())
                nn.init.normal_(target.projection.weight, weight_mean, weight_std)
                bias_mean, bias_std = (source.patch_embeddings.projection.bias.mean(),
                                       source.patch_embeddings.projection.bias.std())
                nn.init.normal_(target.projection.bias, bias_mean, bias_std)
            ...
            target.norm.weight.data.copy_(source.norm.weight.data)
            target.norm.bias.data.copy_(source.norm.bias.data)
        else:
            target.norm.weight.data.copy_(source.LayerNorm.weight.data)   # RoBERTa embeddings.LayerNorm
            target.norm.bias.data.copy_(source.LayerNorm.bias.data)
            target.position_embeddings = copy.deepcopy(source.position_embeddings)
    else:
        for n, m in target.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)                          # GPT-2/BERT-style std
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
```

ORCA's OTDD call, for the optional alignment warm-up:

```python
def otdd(feats, ys=None, src_train_dataset=None, exact=True):
    ys = torch.zeros(len(feats)) if ys is None else ys
    dataset = torch.utils.data.TensorDataset(feats, ys)
    dist = DatasetDistance(src_train_dataset, dataset,
                           inner_ot_method='exact' if exact else 'gaussian_approx',
                           debiased_loss=True, inner_ot_debiased=True,
                           p=2, inner_ot_p=2, entreg=1e-1, ignore_target_labels=False,
                           device=feats.device, load_prev_dyy1=None)
    return dist.distance(maxsamples=len(src_train_dataset))
```

### 4. LLaVA's projector — `haotian-liu/LLaVA/llava/model/multimodal_projector/builder.py`

Verbatim. Note: **no normalisation anywhere**, default PyTorch init.

```python
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    ...
```

The same file also contains an **unused** `SimpleResBlock` (`LayerNorm -> Linear -> GELU -> Linear`,
residual) — the LLaVA authors wrote a pre-norm residual projector and then shipped the un-normalised one.
`mlp2x_gelu` = `Linear(1024→4096) -> GELU -> Linear(4096→4096)` for Vicuna-13B.

### 5. What I recommend we write — `models/encoders/base.py`

```python
class GraphEncoder(nn.Module):
    """A: [B,N,N], X: [B,N,F] or None  ->  tokens [B,N,d_model]"""

    def __init__(self, d_model, llm_embed_weight=None, orth_gain=1.41, dropout=0.1):
        super().__init__()
        self.proj = ...                                    # E1..E7, ends in nn.Linear(*, d_model)
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self._init_projection(orth_gain)
        self._init_norm_to_embedding_scale(llm_embed_weight, d_model)

    def _init_projection(self, orth_gain):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=orth_gain)   # FPT §3.12: "very important"
                if m.bias is not None:
                    m.bias.data.zero_()

    @torch.no_grad()
    def _init_norm_to_embedding_scale(self, W_e, d_model, eps=1e-6):
        """arXiv 2512.08374 §5.1:  g_init = T / sqrt(D),  beta = 0."""
        nn.init.zeros_(self.out_norm.bias)
        if W_e is None:
            nn.init.ones_(self.out_norm.weight)
            return
        norms = W_e.detach().float().norm(dim=-1)
        T = norms[norms > eps].mean()                 # mean L2 norm of non-zero embedding rows
        self.out_norm.weight.fill_(float(T / d_model ** 0.5))
        self.register_buffer('target_embed_norm', T.clone())

    def forward(self, A, X=None, mask=None):
        h = self.proj(A if X is None else torch.cat([A, X], dim=-1))
        return self.out_norm(self.drop(h))
```

Global Weight Compensation, if the encoder's gradients vanish (arXiv 2512.08374 §5.1 + Appendix):

```python
class GWCLayerNorm(nn.LayerNorm):
    """Forward: LayerNorm with a tiny gain (norm alignment).
       Backward: gradient into the input rescaled by 1/mean(|g|) so the encoder still learns."""
    DELTA = 1e-3

    def forward(self, x):
        xhat = F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        if xhat.requires_grad:
            mu = self.weight.detach().abs().mean().clamp_min(self.DELTA)
            xhat.register_hook(lambda g, mu=mu: g / mu)
        return xhat * self.weight + self.bias
```

### 6. Optimizer param groups (differential LR) — the pattern every source agrees on

```python
groups = [
    {'params': encoder.parameters(), 'lr': 1e-3},   # random init, needs a big LR
    {'params': decoder.parameters(), 'lr': 1e-3},
    {'params': bias_params,          'lr': 5e-3},   # GTLM: bias >> LoRA
    {'params': ln_affine_params,     'lr': 1e-4},   # FPT: LN affine is the sensitive knob
    {'params': lora_params,          'lr': 3e-5},   # starts from refined pretrained weights
]
optim = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.98))
torch.nn.utils.clip_grad_norm_(all_params, 1.0)
```

---

## Numbers to beat / hyperparameters to copy

### Hyperparameters, by source

| Source | Module trained | Optimizer | LR | Schedule | Batch | Other |
|---|---|---|---|---|---|---|
| **FPT** (Lu 2021 App. C) | in/out/LN/pos of GPT-2 | Adam, PyTorch defaults | **1e-3** (1e-4 homology) | **none** | 2–16 | dropout 0.1, orth init gain **1.41**, grad clip 1.0, no weight decay, no val split |
| **Rothermel** (2107.12460 §3) | all four variants | Adam | **sweep 1e-6 → 1e-2**, log-spaced | early stop on val | 8 | 3 seeds, mean ± SEM, max steps 3e5–4e5 |
| **Frozen** (Tsimpoukelli 2021 §4) | NF-ResNet-50 vision encoder only, 7B LM frozen | Adam β=(0.9, **0.95**) | **3e-4 constant** | none | 128 | early stop just after 1 epoch on 3M pairs; finetuned/scratch variants *"preferred a smaller learning rate of 1e-5"* |
| **LLaVA-1.5** stage 1 (projector only) | `mlp2x_gelu` | AdamW | **1e-3** (LLaVA-v1 linear used **2e-3**) | cosine, **warmup_ratio 0.03** | 256 | weight decay **0**, 1 epoch, bf16, ZeRO-2 |
| **LLaVA-1.5** stage 2 (LLM + projector) | all | AdamW | **2e-5** | cosine, warmup 0.03 | 128 | 1 epoch, ZeRO-3 |
| **BLIP-2** (2301.12597 §3.4) | Q-Former + 1 FC layer | AdamW β=(0.9, **0.98**), wd **0.05** | peak **1e-4**, min 5e-5 (stage 2) | cosine, **linear warmup 2k steps** | — | 32 queries × 768 dims |
| **ORCA** `configs/deepsea.yaml` | embedder, then all | Adam β=(0.9, **0.98**), wd 0 | **1e-5** | WarmupLR, warmup 10 epochs | 16 | clip 1.0, `embedder_epochs: 60`, `epochs: 13`, `target_seq_len: 512`, `objective: otdd-exact` |
| **GraphToken** (2402.05862 §3) | GNN encoder only, PaLM-2-S frozen | **Lion** | **0.05** | — | — | soft graph tokens prepended, LLM fully frozen |
| **GTLM** (per `CLAUDE.md` §9) | bias + LoRA | — | bias 5e-3 vs LoRA 3e-5 | — | — | consistent with everything above |

**The single most transferable rule, holding across all eight rows:** *the newly-initialised cross-modal
module gets a learning rate 30–1000× larger than anything touching pretrained weights.* LLaVA 1e-3 vs 2e-5
= 50×. GTLM 5e-3 vs 3e-5 = 167×. GraphToken Lion 0.05 with the LLM entirely frozen.

**Second rule, from LLaVA-1.5 verbatim:** *"we halve the learning rate in pretraining due to the usage of
the MLP projection layer instead of the original linear projection layer design."* (2e-3 linear → 1e-3 MLP.)
So E2 (MLP encoder) should get roughly half of E1's (Linear encoder) LR — do **not** reuse E1's LR across the
encoder sweep or the sweep will be confounded with an LR effect.

### Token counts and projector depths in the analogous literature

| System | # continuous tokens into the LLM | Projector |
|---|---|---|
| **Frozen** (Tsimpoukelli 2021) | **2** per image — *"We experimented using different number of tokens, specifically 1, 2 and 4 and found that 2 performs best"* | one Linear mapping the pooled NF-ResNet-50 vector to `D*n`, reshaped to `n × D` |
| **BLIP-2** | **32** learnable queries, each 768-dim inside the Q-Former | Q-Former + *"a fully-connected (FC) layer to linearly project the output query embeddings Z into the same dimension as the text embedding of the LLM"*, then prepended as *"soft visual prompts"* |
| **LLaVA-v1** | 256 (CLIP ViT-L/14 @224) | 1 × `nn.Linear` |
| **LLaVA-1.5** | **576** (CLIP ViT-L/14 @336) | 2-layer MLP with GELU |
| **GraphToken** | fixed small number, prepended | GNN readout + *"a final dense layer"* |
| **FPT** | 10 (XOR) … 1024 (C10 LRA, Homology) | 1 × `nn.Linear` (+ Dropout) |
| **Ours** | **N** (one per node): 2708 for Cora | E1–E7 |

Our N=2708 is **4.7× more continuous tokens than LLaVA-1.5, 1354× more than Frozen**, and past FPT's longest
demonstrated sequence (1024). This is a real extrapolation. `CLAUDE.md` §11 already says to cap at 500–1000
nodes per batch — that lands inside the demonstrated range, which is an extra argument for doing it.

### Norm statistics at the modality interface (arXiv 2512.08374, Tables 1–3)

**Table 1 — vision encoder outputs vs LLM text embeddings, mean ± std L2 norm:**

| Modality | Model | Dim | Avg L2 norm |
|---|---|---|---|
| Visual | CLIP-ViT-large-patch14 | 1024 | 29.30 ± 17.12 |
| Visual | SigLIP-SO-400m-patch14-384 | 1152 | 71.78 ± 13.95 |
| Visual | SigLIP2-SO-400m-patch14-384 | 1152 | 59.37 ± 106.08 |
| Visual | MoonViT-SO-400M | 1152 | 72.17 ± 7.13 |
| **Text** | **Qwen2.5-7B-Instruct** | 3584 | **0.80** |
| **Text** | **Qwen3-8B-Instruct** | 4096 | **1.38** |
| **Text** | **Llama3.2-3B-Instruct** | 3072 | **1.09** |

**Table 2 — does the projector fix it? Mostly no.**

| Model | Visual before proj. | Visual after proj. | Text (embedding) |
|---|---|---|---|
| LLaVA-v1.5 | 28.71 ± 16.87 | **39.96 ± 45.58** (worse) | 1.08 |
| Qwen-2.5-VL | 3484.24 ± 3882.82 | 56.88 ± 25.73 | 0.86 |
| KimiVL | 137.93 ± 34.43 | 4.78 ± 2.21 | 0.85 |
| GLM-4.1V | 47.44 ± 4.58 | 4.58 ± 2.03 | 0.80 |

Verbatim: *"a substantial disparity between visual and text token norms persists in all analyzed models...
With the exception of LLaVA-v1.5, all other models incorporate internal norm layers, yet their final output
norms differ by an order of magnitude."* And: *"these architectural choices and their impact on cross-modal
norm alignment are seldom, if ever, addressed in the models' respective technical reports."*

**Table 3 — even discrete visual vocabularies don't fix it:** Ovis-2.5 SigLIP2 visual vocab 64.00 ± 0.71 vs
text vocab 1.38 ± 0.32. But their update dynamics *were* synchronised, suggesting discrete tokenisation
mitigates the inertia even without norm parity.

**Table 4 — what norm alignment buys (LLaVA-1.5 framework, SigLIP-SO400M-384, Llama-3.2-3B-Instruct):**

| Method | MMBench-dev | MM-Star | POPE | SEED-Bench-2 | OCRBench |
|---|---|---|---|---|---|
| w/o Norm | 71.39 | 37.72 | 88.14 | 42.86 | 40.70 |
| **w/ Norm (naive init)** | **72.16** (+0.77) | **41.19** (+3.47) | **88.88** (+0.74) | **47.26** (+4.40) | **45.60** (+4.90) |

For Qwen2.5-7B the naive small-gain init **hurt** multimodal metrics (MMBench −1.20, MM-Star −2.26) because
of the vanishing-gradient effect, and GWC was needed to recover.

**Table 5 — why the *init* matters, not just the layer.** Learned LN params after LLaVA stage 1:

| Parameter | Metric | Default init (g=1) | Norm-matched init |
|---|---|---|---|
| Gain g | L2 norm | 53.2500 | 2.2812 |
| Gain g | mean of \|g\| | 0.9609 ± 0.0005 | 0.0400 ± 0.0001 |
| Bias β | mean of \|β\| | 0.0175 ± 0.0002 | 0.0152 ± 0.0001 |

### The table we should produce, in this shape

Masked edge prediction, AUPRC on masked entries, Cora, mean ± SEM over ≥3 seeds, LR swept per row:

```
                                                 Cora AUPRC (masked entries)
  identity / random / feature-only (CLAUDE.md §7.1-3)      ???
  GAE / VGAE (CLAUDE.md §7.4)                              ???   <- the number to beat
  encoder -> decoder, NO LLM                               ???   <- Tan et al. "w/o LLM"
  encoder -> 1 random attn layer -> decoder                ???   <- Tan et al. "LLM2Attn"
  encoder -> 1 random transformer block -> decoder         ???   <- Tan et al. "LLM2Trsf"
  frozen random-init LLM   (LN + bias trained)             ???   <- FPT "Random" = our floor
  frozen pretrained LLM    (LN + bias trained)             ???   <- FPT = the claim
  frozen pretrained LLM + LoRA                             ???
  fully-trained transformer, same shape, random init       ???   <- FPT "Full" / Rothermel "Unfrozen Random"
  fully-trained pretrained LLM                             ???   <- Rothermel's winner
```

**Effect sizes to expect if FPT transfers:** +2.5 to +6.5 points of frozen-pretrained over frozen-random on
long-sequence tasks (Lu Table 2), or +4.5 to +10.5 in Rothermel's properly-swept replication (Table 1). If
our gap is smaller than the seed noise across 3 seeds, we have nothing.

---

## Open questions

1. **Does the FPT effect survive at N ≈ 1000 tokens with a bidirectional mask and no positional
   embeddings?** FPT's pretrained-vs-random gap is smallest on exactly the longest-sequence task
   (C10 LRA +2.5). We additionally remove causal masking and reset RoPE — two things GPT-2/Llama were never
   trained without. FPT Table 17 shows the two long-context/bidirectional-ish backbones (T5-encoder 15.4 %,
   Longformer 17.0 %) collapsed on ListOps while GPT-2 got 38.4 %. **Unverified for our setting.**
2. **Low-data regime.** GPT4TS shows pretraining is worth a lot at 5–10 % data and Tan et al. show it is
   worth nothing at 100 %. Cora link prediction with 15 % masked entries — which regime is that? Needs a
   data-fraction sweep (10 / 25 / 50 / 100 % of training edges). That sweep is our strongest *potential*
   result if pretraining wins in the low-data corner.
3. **`T` for Llama-3.2-1B.** The `g_init = T/sqrt(D)` derivation assumes LayerNorm's unit-variance output.
   Llama uses RMSNorm on its residual stream but has **no norm on the embedding output at all**, so there is
   no pretrained norm to copy, and I could not verify `T` numerically for the 1B model (only 3B is published:
   1.09 at D=3072). We must compute it ourselves:
   `W = model.get_input_embeddings().weight; n = W.float().norm(dim=-1); T = n[n>1e-6].mean()`.
   No `torch` in this environment, so I could not run it.
4. **Is a LayerNorm on the encoder output even the right shape for a graph token?** LayerNorm destroys the
   per-node magnitude, which for adjacency rows carries degree information. RevIN's answer (GPT4TS) is
   normalise then add the statistic back at the output. We may need to feed degree in as a separate channel
   (which `CLAUDE.md` E5 already does) so LayerNorm can be applied without information loss. **Untested.**
5. **Does ORCA-style pre-alignment help when the target has no discrete labels?** ORCA's best metric (OTDD)
   needs labels; for dense/continuous targets they cluster to get pseudo-labels. Our masked-edge task has
   binary labels per *entry*, not per *token*. MMD or moment matching against a sample of
   `embed_tokens.weight` rows is the fallback, and ORCA found all three metrics beat no alignment, so the
   cheap version is probably fine — but the mapping is not obvious.
6. **Does FPT's "orthogonal init is very important" finding hold for a sparse 0/1 adjacency row?** An
   orthogonal `Linear(N, d_model)` with `N > d_model` is only row-orthogonal, and the effective gain on an
   input with ~4 non-zeros (Cora average degree ≈ 4) is very different from a dense image patch. **Needs an
   empirical check of output RMS at init**, which is exactly what the unit test in §5 above measures.
7. **Did Lu et al. ever respond to Rothermel et al.?** I found no rebuttal-to-the-rebuttal. The AAAI 2022
   camera-ready postdates the July 2021 rebuttal, so the rebuttal was public before publication, but I did
   not obtain the AAAI PDF to check whether it addresses the LR issue. **Unverified.**
8. **Is there a graph-specific replication of the FPT claim?** I did not find one. GraphToken, GTLM and
   GaLA all use frozen LLMs but none of them, as far as I could find, runs the frozen-random-init control
   that FPT's claim rests on. If that is right, **our §8 grid would be the first**, which is a contribution
   worth naming explicitly in the writeup.

---

## Sources fetched

**Primary papers (full text read, not just abstract):**
- `https://arxiv.org/abs/2103.05247` — FPT, abstract + version history
- `https://ar5iv.labs.arxiv.org/html/2103.05247` — FPT full text incl. all 22 tables and Appendices A–D
- `https://ojs.aaai.org/index.php/AAAI/article/view/20729` — AAAI 2022 published version metadata
- `https://ar5iv.labs.arxiv.org/html/2107.12460` — Rothermel, Li, Rocktäschel & Foerster, "Don't Sweep your Learning Rate under the Rug", full text + Appendices A–E
- `https://ar5iv.labs.arxiv.org/html/2302.05738` — ORCA, "Cross-Modal Fine-Tuning: Align then Refine" (ICML 2023)
- `https://ar5iv.labs.arxiv.org/html/2106.13884` — Frozen (Tsimpoukelli et al., NeurIPS 2021), full text + Appendix A
- `https://ar5iv.labs.arxiv.org/html/2310.03744` — LLaVA-1.5, full text + Appendix A.3 hyperparameters
- `https://ar5iv.labs.arxiv.org/html/2301.12597` — BLIP-2
- `https://ar5iv.labs.arxiv.org/html/2104.08691` — Lester, Al-Rfou & Constant, "The Power of Scale for Parameter-Efficient Prompt Tuning"
- `https://arxiv.org/html/2512.08374v1` — "The Unseen Bias: How Norm Discrepancy in Pre-Norm MLLMs Leads to Visual Information Loss" (submitted 9 Dec 2025), full text + Appendix
- `https://arxiv.org/abs/2512.08374` — abstract page (for submission date / author list)
- `https://arxiv.org/html/2406.16964v2` — Tan et al., "Are Language Models Actually Useful for Time Series Forecasting?" (NeurIPS 2024 Spotlight)
- `https://arxiv.org/html/2302.11939v3` — Zhou et al., "One Fits All" / GPT4TS (NeurIPS 2023 Spotlight)
- `https://arxiv.org/html/2402.05862v1` — GraphToken, "Let Your Graph Do the Talking"
- `https://arxiv.org/abs/2003.00152` — Frankle, Schwab & Morcos, "Training BatchNorm and Only BatchNorm" (ICLR 2021)

**Raw code read line by line:**
- `https://raw.githubusercontent.com/kzl/universal-computation/master/universal_computation/fpt.py` (194 lines, complete)
- `https://raw.githubusercontent.com/kzl/universal-computation/master/universal_computation/experiment.py` (254 lines)
- `https://raw.githubusercontent.com/kzl/universal-computation/master/universal_computation/trainer.py` (95 lines)
- `https://raw.githubusercontent.com/kzl/universal-computation/master/scripts/run.py` (34 lines)
- `https://raw.githubusercontent.com/kzl/universal-computation/master/README.md`
- `https://api.github.com/repos/kzl/universal-computation/git/trees/master?recursive=1`
- `https://raw.githubusercontent.com/sjunhongshen/ORCA/main/src/embedder.py` (439 lines)
- `https://raw.githubusercontent.com/sjunhongshen/ORCA/main/src/utils.py`
- `https://raw.githubusercontent.com/sjunhongshen/ORCA/main/src/configs/deepsea.yaml`
- `https://api.github.com/repos/sjunhongshen/ORCA/git/trees/main?recursive=1`
- `https://raw.githubusercontent.com/haotian-liu/LLaVA/main/llava/model/multimodal_projector/builder.py`
- `https://raw.githubusercontent.com/haotian-liu/LLaVA/main/scripts/v1_5/pretrain.sh`
- `https://raw.githubusercontent.com/haotian-liu/LLaVA/main/scripts/v1_5/finetune.sh`
- `https://raw.githubusercontent.com/haotian-liu/LLaVA/main/scripts/pretrain.sh`

**arXiv id existence checks (HTTP 200 + `citation_title` read):** 2605.10247, 2606.15633, 2402.05862,
2503.01805, 2501.01073, 2310.04560 — all resolve.

**Not fetched / could not verify:**
- The AAAI 2022 camera-ready PDF (metadata only) — so I cannot say whether it addresses the LR rebuttal.
- Numerical embedding-norm statistics for our actual backbone — no `torch` / `transformers` in this
  environment (checked; all three of `torch`, `transformers`, `torch_geometric` are missing).
