# Graph-In / Graph-Out LLM — Verified Knowledge Base

**Status:** master reference. Supersedes ad-hoc reading of the individual notes files.
**Assembled:** 2026-08-25, from eight deep-dive research passes + two adversarial critic passes + one local hardware/software survey.
**Authority:** where this document and `CLAUDE.md` disagree, **this document wins**, and §1 says why.

---

## 0. How to read this

### 0.1 Confidence tags

Every non-obvious claim below carries an inline source. Tags used:

| Tag | Meaning |
|---|---|
| *(no tag)* | Verified by at least one agent fetching a primary source (paper HTML/PDF, library source, or a direct computation), and not contradicted. |
| **[2x]** | Independently verified by two or more agents, or by an agent *and* a critic, via separate fetches. Treat as settled. |
| **[COMPUTED]** | Arithmetic done in this document or in a notes file from verified inputs. Recheck the arithmetic, not the provenance. |
| **[CONFLICT]** | A researcher and a critic disagreed. The resolution and the reason are stated. |
| **[UNVERIFIED]** | Nobody fetched a primary source. Do not put it in a paper. Do not build load-bearing code on it. |
| **[SINGLE-SOURCE]** | One agent fetched it; nobody else checked. Usable, but re-fetch before citing. |
| **[FABRICATED]** | An agent attached a real section number to advice that is not in that section. Named explicitly so it does not propagate. |

### 0.2 The three things to internalise before reading further

1. **The hardware is a 12 GB laptop GPU, not an 80 GB A100.** `environment.md` measured it: RTX 5070 **Laptop**, 12,227 MiB VRAM, 80 W power cap, Blackwell sm_120, Windows 11, Python 3.14. GTLM ran on a single A100/H100 80 GB (arXiv 2605.10247, Limitations). We have ~6.7x less memory and a small fraction of the throughput. **Subgraph sampling is the architecture, not a fallback.** Almost every "just use FlexAttention / just train on the full graph" recommendation in the research digest was written without knowing this.
2. **`transformers` installs as v5.15.1, not v4.** Every attention-injection recipe in a 2024–2025 repo (GTLM's included) predates the v4→v5 masking/dispatch rewrite. §4 is written against v5 source and was verified against it by two independent agents.
3. **`CLAUDE.md` contains five confirmed factual errors and about a dozen overstatements.** §1 is the most valuable part of this document. Read it in full before writing any code.

### 0.3 Notes-file index

All paths relative to `c:/Users/ucchw/OneDrive - Stanford/Graph to LLM/docs/research/`.

| File | Topic | Read it when |
|---|---|---|
| [`environment.md`](environment.md) | Local hardware, VRAM math, installable stack, transformers-v5 hazard | Before planning any experiment grid. **Shortest and most decision-relevant file.** |
| [`hf-mechanics.md`](hf-mechanics.md) | `inputs_embeds`, 4-D mask bias trick, GQA shape, RoPE=0, `is_causal=False`, LoRA, param groups, grad checkpointing, backbone table | Writing `models/llm_wrapper.py`. The engineering spine. |
| [`pyg-baselines.md`](pyg-baselines.md) | Cora loader, `RandomLinkSplit` exact shapes, GAE/VGAE source, AP-vs-AUROC math, `pos_weight`, memory arithmetic | Writing `data/`, `eval.py`, `baselines/`. |
| [`gtlm.md`](gtlm.md) | Closest prior work: bias families, SPD/RRWP code, FlexAttention rebuttal, differential LR, tiny checkpoints | Writing `models/bias/`. |
| [`bias-mechanism.md`](bias-mechanism.md) | Graphormer / GaLA / GRIT / SAN / Magnetic Laplacian lineage; bias magnitude limits | Deciding *which* bias and *how strong*. |
| [`depthwidth.md`](depthwidth.md) | Theory of adjacency-row tokenization; Thms 4.2/4.3/4.4; encoder recipes | Justifying the input format; sizing `d_model` against `N`. |
| [`fpt.md`](fpt.md) | Frozen-pretrained-transformer evidence *and its rebuttal*; encoder init + norm matching | Writing `models/encoders/base.py`; designing the headline ablation. |
| [`competitors.md`](competitors.md) | Verbalization / soft-prompt lineage; published Cora numbers by protocol; novelty audit | Writing related work; picking baselines. |
| [`autoregressive.md`](autoregressive.md) | GraphRNN / GRAN / G2PT; node ordering; MMD evaluation | Only when you reach task §5.3. Defer. |

### 0.4 Where researchers and critics disagreed

Policy: side with whoever cited a fetched primary source. Nine conflicts arose; all are resolved inline and flagged **[CONFLICT]**.

| # | Conflict | Resolution | Why |
|---|---|---|---|
| 1 | FlexAttention: can a learned bias be captured in `score_mod`? | **Both are right about different things.** See §3.4. | Researchers read `torch/_higher_order_ops/flex_attention.py` (`score_mod_other_buffers can require grad`); Critic 2 cited open issue pytorch#145460 (a leaf `nn.Parameter` raises "Captured buffers that require grad are not yet supported"). Reconciliation: a **non-leaf** tensor (the output of a bias MLP, which is exactly what GTLM passes) works; a **leaf Parameter** does not. Moot on our hardware — see §3.4. |
| 2 | GaLA all-heads-vs-per-head ablation numbers | **Downgraded to [UNVERIFIED].** | The bias agent marked it "(certain)"; Critic 1 fetched the same HTML and reported the table was not visible in the render. An attempted-and-failed fetch outranks an unsupported confidence tag. |
| 3 | GTLM `MIXED_BIAS.md` "\|b\|max 9–240, scored 10 pp below the no-bias floor" | **Downgraded to [UNVERIFIED].** | Two agents cited it as certain; Critic 1 fetched the file and confirmed the *mechanism* (NaN divergence at bias_lr 2e-2, quartic growth of degree-2 biases, AdamW defeating grad clipping, the value 52.85) but **not** those specific figures. |
| 4 | "E5 spectral recipe from arXiv 2605.22471 §A.7" | **[FABRICATED] provenance; the advice is retained as ours.** | Critic 1 fetched §A.7/§3.2 twice: the paper uses the **unnormalized** Laplacian `L = D − A` and a token that **explicitly includes raw eigenvalues**. Normalization, sign flips and standardization appear nowhere. The false attribution is baked into a code docstring at `depthwidth.md:768-769`. |
| 5 | 2605.22471 Thm 5 "Laplacian tokens ill-conditioned for edge prediction" | **Materially overstated.** | Critic 1 fetched the theorem: the `‖X‖² = Ω(n²)` step is qualified *"for **dense** graphs where `d_max = Θ(n)`"*. Cora's `d_max = 168`, nowhere near `Θ(2708)`. |
| 6 | GTLM's 173,056 bias params — in the paper or not? | **In the paper. Confirmed.** | `hf-mechanics` called it unverified having read only the abstract; the `gtlm` agent and Critic 1 both fetched the body and quote the Limitations sentence verbatim. Fetchers win. |
| 7 | RRWP max steps: 8 or 16? | **16.** | Repo *default* in `src/models/config.py` is 8; paper Table 6 **and** the actual `src/experiments/*/config.py` run configs say 16. Defaults are not run configs. |
| 8 | Is the bidirectional mask a cost driver? | **Unresolved. Measure it ourselves.** | Paper Limitations says the bidirectional mask contributes to the ~3x slowdown (Critic 1 verified verbatim). The repo's `REBUTTAL.md` decomposition says the mask is free (sdpa+mask = 1.00x plain causal). Critic 1 verified the paper side only. A genuine paper-vs-repo contradiction. |
| 9 | Cora edge count | **5,278 unique undirected / 10,556 directed.** `CLAUDE.md`'s 5,429 is wrong. | Three independent computations from `ind.cora.graph` (depthwidth agent, pyg agent, Critic 1) agree exactly. |

---

## 1. Corrections to CLAUDE.md

Deliberately blunt. Every row is a place where building to spec produces a wrong number, a wasted week, or a claim a reviewer can kill.

### 1.1 Hard factual errors

| # | CLAUDE.md claim | Status | Correction | Source |
|---|---|---|---|---|
| **F1** | §6, §11: "Cora. 2,708 nodes, **5,429 edges**" | **WRONG** | PyG hands you `edge_index [2, 10556]` = **5,278 unique undirected simple edges**. 5,429 is the pre-deduplication count from Sen et al.; 302 directed entries are dropped by `coalesce`. | **[2x]** Direct computation from `raw.githubusercontent.com/tkipf/gcn/.../ind.cora.graph` by the depthwidth agent, the pyg agent and Critic 1 independently; corroborated by pyg discussion #6203 |
| **F2** | §11: "~5,429 edges out of 2,708² ≈ 7.3M — **about 0.1% positive**" | **WRONG (mixes conventions)** | Density is **0.14395%** (10,556 / 7,333,264). "0.1%" comes from dividing an *undirected* count by a *symmetric* denominator. **`pos_weight` = 693.70** (full N²) or **693.44** (off-diagonal). This is the single hyperparameter §11 itself says will silently kill the model. | **[COMPUTED, 2x]** `pyg-baselines.md` and Critic 1, same raw file |
| **F3** | §9: "Theorem 4.2 — `m·p·H·L = Ω(N)`. **Check `d_model` against N**" | **Gloss is wrong** | The theorem bounds the **product**, not `m`. Llama-3.2-1B: `2048·16·32·16 = 16,777,216` vs `N = 2708` — satisfied by **6,196x**. Llama-3.1-8B: 24,782x. Even the no-residual form `mpH` exceeds 10⁶. **Theorem 4.2 imposes zero constraint on any real LLM. Delete the instruction.** | **[2x]** `arxiv.org/html/2503.01805v3` Thm 4.2 |
| **F4** | §9: "Theorem 4.4 — width `O(d·log N)` … `d_model` 4096 **clears it comfortably**" | **Understates by ~2x; does not generalise to Cora** | App. B.4's construction is `m = 2p + 2` with `p = Ω(d log n)` — roughly **twice** the headline, with an unspecified big-O constant. At Cora's `d_max = 168`: `2·168·log₂(2708) + 2 = 3,833`. **`d_model = 2048` does NOT clear it at hub nodes**; 4096 barely does. At Cora's p99 degree of 19 it is 435, trivially cleared — so **~27 hub nodes are under-width and the rest are fine**. `CLAUDE.md`'s own example (`d=20, N=1500` → 210) is arithmetically right but is a toy. | **[2x, COMPUTED]** App. B.4 + `d_max = 168` computed from the raw Cora file |
| **F5** | §9: "Theorem 4.3 — an `O(L)`-layer transformer with **width `O(N)`**" | **Hides a factor of 3** | App. B.3 verbatim: `X = [A; I; 0; 0] ∈ R^{3d×d}` — three N-wide blocks (adjacency row, one-hot node ID, `A^L` accumulator). For Cora that is **8,124**, 4x Llama-3.2-1B's 2048. The construction also **requires an explicit one-hot node-ID block**, which is precisely what destroys permutation equivariance. | **[2x]** App. B.3 |

### 1.2 Claims that are true of the source but wrong as instructions

| # | CLAUDE.md claim | Status | Correction | Source |
|---|---|---|---|---|
| **I1** | §11, §13: "**Do not use FlashAttention. It cannot take the custom bias. Use the eager attention path.**" + §9's "~3x slower training" | **Accurate about the submitted GTLM paper; obsolete about the method; and *right by accident* on our hardware** | FlashAttention-2 genuinely cannot take an arbitrary additive bias. But GTLM's repo ships `src/models/flex_attn/REBUTTAL.md`, titled *"the 'eager attention only' limitation is resolved"*, with a working `gtlm_flex` backend: cora 836.9→226.8 ms (**3.69x**), ogbn-arxiv 2006.2→208.3 ms (**9.63x**), peak memory at L≈4.5k **30 GB → 0.2 GB**, parity tests passing, *"No number in the paper changes."* **However — §3.4 — FlexAttention requires `torch.compile`, which is very likely unavailable on this Windows + cp314 + sm_120 machine.** The instruction "use eager" survives *for us*, for an entirely different reason than the one given. | Paper Limitations **[2x]**; `REBUTTAL.md` **[2x]**; the Windows/`torch.compile` consequence is **[UNVERIFIED — day-1 test, §9]** |
| **I2** | §11: "Report **AUPRC, not accuracy and not AUROC**" + "Evaluate on held-out masked entries **plus an equal number of sampled true non-edges**" | **Two incompatible protocols in adjacent sentences, and the AUROC ban is backwards** | These are two metrics differing by **~50x**. Measured with sklearn's exact AP on a scorer pinned to AUROC≈0.91 with 527 positives: AP = **0.9103** at 1:1, 0.6446 at 10:1, 0.2562 at 100:1, **0.0174** at the true 6952:1 — while AUROC stayed 0.90–0.92 throughout. Every published AP (GAE, VGAE, MaskGAE, ARGA, SEAL) is on a **balanced** set. AUROC is the *balance-invariant bridge* that makes the two comparable; banning it removes the only tool that solves §11's own problem. **Report three columns always.** | **[COMPUTED]** `pyg-baselines.md` simulation + sklearn AP docs + Davis & Goadrich ICML 2006 |
| **I3** | §7: "GAE / VGAE … **This is the number to beat**" | **That is the floor, not the ceiling** | MaskGAE (arXiv 2205.10053) Table 3: Cora **96.45 AUC / 95.95 AP** (path masking), 96.42/95.91 (edge masking), under a comparable 85/5/10 balanced-negatives split. ARGA 92.40/93.23 and SEAL 92.22/93.12 also beat GAE. Separately, PyG's **own** GAE reproduces *below* the published table: 90.6±0.9 AUC / 91.2±1.0 AP over 30 runs. **Aiming at 92 produces a result no reviewer accepts.** | **[2x]** MaskGAE Table 3, fetched by the competitors agent and Critic 1; PyG reproduction from arXiv 2107.02658 |
| **I4** | §11: "GTLM's fix … **restores equivariance. Adopt it, and write a unit test**" | **The prescribed test is guaranteed to fail for E1/E2/E3/E4/E5 and it is NOT a bug** | Under permutation `P`, `A → P A Pᵀ`, so token `i`'s **contents** permute, not just its position: `t'_{π(i)} = W·(P A Pᵀ)[π(i),:] = W·Pᵀ A[i,:] ≠ W·A[i,:]`. A fixed `[d_model, N]` matrix assigns a different learned direction to every *column index*, i.e. to every node id. GTLM's fixes restore equivariance of the **transformer body only**. arXiv 2605.22471 states it flatly: *"adjacency tokenization is not permutation-equivariant, unlike the spectral and random-walk variants."* **Decide the framing before writing the test.** Scope the strict test to E6/E7 and bias-only configs; give E1–E5 a weaker index-consistency test. | **[2x]** arXiv 2605.22471 §3.2/Table 1; the argument is elementary linear algebra |
| **I5** | §9: "LoRA rank **32 or 64**, alpha = 2r" | **Wrong for our setting** | Paper Table 7 (**GraphQA — abstract topology, no node text; the setting closest to ours**) gives **r = 16 / alpha = 32**, and the repo default `src/experiments/graphqa/config.py:252` is `lora_r: int = 16`. r ∈ {32,64} is Table 8 (text-attributed node classification); r=64/alpha=128 for the 3B/8B scaling runs. **`alpha = 2r` is correct everywhere.** The paper omits and the repo shows: `lora_dropout=0.05`, `bias='none'`, `target_modules` = **all seven** projections. | **[2x]** Tables 7/8 + repo config |
| **I6** | §9: "Reason: **bias params start random**, LoRA starts from refined pretrained weights" | **Contradicted by the released code** | `SPDBias` is `nn.Parameter(torch.zeros(max_spd, num_heads))`; `RRWPBias`'s output layer is `nn.init.zeros_(...)`. `LINEAR_BIAS.md` has an explicit *"Zero-init inertness"* test. Only `LaplacianBias`/`RWSEBias` use `randn*0.02`. **The differential-LR advice stands — zero-init makes a high bias LR *more* necessary — but the stated reason is false.** Zero-init also makes §8's "no attention bias" ablation a *true nested baseline* rather than a different model. | **[2x]** `graph_model/src/models/bias.py`, fetched by three parties |
| **I7** | §11: "connectivity needs depth `Ω(log N)` or width `Ω(N)`" cited as a proved fact | **It is unproved, conditional discussion prose** | Appendix A of 2503.01805 is proof-free comparative discussion; the claim inherits from Conjecture 13 of Sanford et al. 2024a, which the paper's own §4.1 labels a conjecture and whose Thm 4.1 shows is broken by linear width. Contrast Thm 4.2, which the paper calls *unconditional*. **Cite instead** arXiv 2605.22471 Thm 4: *"Assuming TC⁰ ⊊ L, any Transformer requires Ω(log n) depth to solve global graph connectivity, **even when provided with full Θ(n) Adjacency rows**."* Standard assumption, and it shows **width does not rescue you — depth does**. Every candidate backbone clears `⌈log₂ 2708⌉ = 12`. | **[2x]** both papers fetched |
| **I8** | §2: "**RoPE disabled / reset per node**" as two alternatives | **A choice where none exists** | GTLM never disables RoPE — it resets `position_ids` to 0 at node boundaries **in the collator**; zero RoPE code is modified. And for our **one-token-per-node** design the two are *numerically identical*: every token gets `position_id = 0`, `cos = 1·attention_scaling`, `sin = 0`, so `apply_rotary_pos_emb` is the exact identity. **Write no RoPE code**: `position_ids = torch.zeros(B, N, dtype=torch.long)`. Conditional on `attention_scaling == 1.0` — **assert it at startup**. | **[2x]** `modeling_llama.py` + `modeling_rope_utils.py` |
| **I9** | §2 lists two equivariance ingredients; §11 lists three. **They contradict each other.** | **§2 is incomplete** | Property 1's proof needs **three** conditions; the third is that the bias be computed from **static graph topology, independent of flattened sequence index**. Not pedantry: a bias derived from node *index* rather than node *structure* would look fine and silently break equivariance. Also: GTLM proves equivariance over **prefix nodes only** (prompt node pinned last, causal). We have no prompt node, so our claim is strictly **stronger** — say so rather than citing theirs as identical. | GTLM App. B.1 |
| **I10** | §9 orders SPD first ("our hop-bias table"); §10 puts `spd.py` first | **The cited source's own ablation inverts this** | GTLM Table 5 (GraphQA): removing **Magnetic** collapses Edge Count **56.9 → 7.1** and Shortest-Path variance ±1.7 → ±12.7. Removing **SPD** *improves* Shortest Path **90.1 → 92.0** and Cycle Check 96.7 → 97.2. Removing RRWP costs Edge Count 56.9 → 48.9. **Magnetic is load-bearing; SPD is the least essential.** Honest caveat: measured on GraphQA for GraphQA tasks, not Cora link prediction. But the spec's priority ordering is not supported by the source it cites. **And see trap T3 — Magnetic is provably a no-op on undirected graphs.** | **[2x]** GTLM Table 5 |
| **I11** | §13: "**No unweighted BCE**" stated as universal | **Both reference implementations we benchmark against use unweighted BCE** | GraphRNN: `binary_cross_entropy_weight` defaults `has_weight=False`. GRAN: `edge_weight: 1.0e+0` in every released config. They can, because they score inside an M-band / block where the positive rate is far higher. **Rephrase:** weighted BCE for §5.1/§5.2 on the full matrix; for §5.3 compute `pos_weight` over the *actual scored index set* and report what it was. | GraphRNN + GRAN repos |
| **I12** | §8: "**Shuffled adjacency is the decisive control**" | **Not decisive — it changes two things at once** | Degree-preserving rewiring on Cora destroys **homophily** along with topology, and on a citation network with 1433-dim BoW features homophily *is* most of the link signal. A drop therefore proves only "the input correlates with the target", which the identity baseline already establishes. **The decisive crossing is (A=I, features only) × (X=0, topology only)**, plus feature-permutation-with-A-fixed, plus rewiring that preserves the endpoint feature-similarity distribution. One afternoon, and it decides whether this is a graph paper or a feature-similarity paper. | Critic 2's analysis; no source claims "decisive". Reasoning accepted. |
| **I13** | §9: "Our decoder D1 is `InnerProductDecoder` with a learned W inserted. **Import it, do not reimplement.**" | **Not literally possible** | `InnerProductDecoder.forward` is exactly `(z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)` — no `__init__`, no `nn.Parameter`, no hook where a `W` could go. Import it for the **GAE/VGAE baseline** (correct and free there); write our own ~10-line module for D1. | `torch_geometric/nn/models/autoencoder.py` (master) |
| **I14** | §7.6 / §8 "Random-init LLM" described as one experiment | **Two experiments that give OPPOSITE answers in the literature** | **Frozen**-random isolates pretrained knowledge, and pretrained wins (Rothermel Table 1, 4/4: ListOps +7.2, MNIST +0.5, CIFAR10 +4.5, C10-LRA +10.5). **Fully-trained**-random is different, and pretrained does **not** reliably win (Unfrozen Random 77.8 vs Unfrozen Pretrained 77.7 on CIFAR-10; 57.6 vs 56.3 on ListOps). §7.6's *text* describes the fully-trained version but its stated *purpose* is what the frozen version does. **Split into two rows and run both.** | **[2x]** arXiv 2107.12460 Table 1 |
| **I15** | §9 FPT: "**This is the justification** that a frozen text LLM can process a non-text modality" | **The headline result was overturned; only a narrower claim survives** | Rothermel, Li, Rocktäschel & Foerster (arXiv 2107.12460): *"we find that this result is, in fact, an artefact of not tuning the learning rates."* Lu et al. ran every configuration at a single LR of 1e-3, one seed, **no validation split**. With a 1e-6→1e-2 sweep over 3 seeds, **unfrozen beats frozen on all four tasks** (CIFAR10 77.7 vs 66.3; C10-LRA 67.8 vs 54.7). **Cite FPT only for what replicated — frozen-pretrained > frozen-random, +2.5 to +10.5 points — and cite the rebuttal ourselves, in the same sentence.** | **[2x]** arXiv 2107.12460 |
| **I16** | §2 "Trainable: encoder, attention-bias, LoRA, decoder" | **Omits the single most important trainable group in the FPT recipe** | FPT Tables 15/19: output-only → +**layernorm affine** takes CIFAR-10 **25% → 54%**, MNIST 23% → 96%, ListOps 15% → 36%. Most important trainable component they have. Cost on Llama-3.2-1B is trivial: RMSNorm has **weight only, no bias**, so `16 × 2 × 2048 + 2048 = ` **67,584 params [COMPUTED]**. (The fpt agent's "131K" assumed weight+bias, a GPT-2 LayerNorm count, not Llama RMSNorm.) Add as a fifth trainable group at lr ≈ 1e-4. This is **not** "unfreezing the body" and does not violate §13. | **[2x]** FPT §3.12; RMSNorm shape from `modeling_llama.py` |

### 1.3 Missing content — things the spec should say and does not

| # | Gap | Why it matters | Source |
|---|---|---|---|
| **M1** | **No pitfall about the encoder's output scale.** | Measured L2 norms at the modality interface: vision-encoder tokens **29.3–72.2** vs LLM text embeddings **0.80–1.38**. Projectors do *not* fix it — LLaVA-v1.5 made it *worse* (28.71 before → 39.96 after, vs text 1.08). High-norm tokens acquire "representational inertia" and transform semantically far slower than text tokens. **Fix:** `nn.LayerNorm(d_model)` on every encoder output, gain initialised to the **scalar** `g_init = T / sqrt(d_model)`, bias 0, where `T` = mean L2 norm of the non-zero rows of `embed_tokens.weight`. Worth +3.47 MM-Star / +4.40 SEED-Bench-2 / +4.90 OCRBench on LLaVA-1.5 + Llama-3.2-3B, and it **also improved text-only MMLU**. | arXiv 2512.08374 §4–§5, Tables 1–5 **[SINGLE-SOURCE — the fpt agent fetched it; Critic 1 confirmed the paper exists but did not verify the numbers]** |
| **M2** | **No mandate for LR sweeps, seeds, or a validation split.** | The closest paper to our claim was overturned *purely* for this. Rothermel: *"Each of the LRs evaluated between 1e-5 and 1e-3 results in different orderings and, hence, conclusions about the optimal architecture variant."* A single-LR comparison of our headline claim is indefensible. | **[2x]** arXiv 2107.12460 |
| **M3** | **Three ablations a 2024-onward reviewer will demand.** | Tan et al. (NeurIPS 2024 Spotlight, arXiv 2406.16964) ablated the LLM out of three FPT-style systems: *"removing the LLM component or replacing it with a basic attention layer does not degrade forecasting performance — in most cases, the results even improve!"* Ablations beat Time-LLM **26/26**, CALF 22/26, OneFitsAll 19/26. **Our encoder+decoder+LoRA is ~13 M trainable params [COMPUTED, §7] vs FPT's 106 K** — the "small module did all the work" risk is ~100x larger for us. Add **(a) w/o LLM**, **(b) LLM2Attn** (one random MHA layer), **(c) LLM2Trsf** (one random transformer block). Week one, not the end. | **[2x]** arXiv 2406.16964 |
| **M4** | **Nothing about how SPD/RRWP interact with 15% edge masking.** | **The highest-risk unknown in the design.** Structural features must come from *some* adjacency. From the complete graph → you leak every held-out edge straight into the attention bias. From the observed graph → masking inflates distances and manufactures spurious unreachability *precisely on the pairs we are asked to predict as edges*, so the bias actively asserts "these nodes are far apart" about the labels. **No source paper faces this**, because none does masked edge prediction. Decided treatment in §3.5. | Original to this project; flagged independently by the bias agent and both critics |
| **M5** | **No zero-parameter heuristic baselines.** | Common Neighbors, Adamic-Adar, Resource Allocation, Personalized PageRank: five lines each, no training, ~0.80–0.85 AUC on citation graphs. Neo-GNN (arXiv 2206.04216): *"heuristic methods often show competitive performance compared to GNNs, with neighborhood overlap-based heuristic methods being even better than GNN models in several datasets."* **Every reviewer's first question about a Cora link-prediction number is "what does Adamic-Adar get".** | **[SINGLE-SOURCE]** Critic 2 |
| **M6** | **D1/D2 sit under a proved expressiveness ceiling the spec never mentions.** | Zhang, Li, Xia, Wang & Jin, *"Labeling Trick"*, NeurIPS 2021: aggregating single-node representations into a node-set representation has *"a fundamental constraint — the inability to capture dependence between nodes in the node set."* Automorphic pairs `(i,j)` and `(i,k)` get identical scores **no matter how good H is**. This is the mechanism behind the GAE-91 vs SEAL/BUDDY/NBFNet gap. **Consequence: our headline ablation would be measured through a decoder bottleneck that suppresses the very effect we are trying to detect.** Promote **D3** into the week-one decoder set. Pair with arXiv 2402.06662 (sign-rank limits of inner-product decoders). | **[SINGLE-SOURCE]** Critic 2 |
| **M7** | **§5.3 is architecturally incompatible with §2, and has an unflagged ordering leak.** | Under a **bidirectional** mask `h_t` sees nodes `t+1..N`, so the AR loss is **vacuous**. With RoPE reset to 0 there is no inter-node positional signal, so the model cannot know which step it is on. Separately: computing a BFS/DFS/k-core ordering on the **complete** graph makes **the ordering itself a function of the hidden edges** — a node's sequence position leaks its held-out neighbours, and §11's "score only what was masked" rule does not catch it because the leak lives in the permutation. `llm_wrapper.py` needs `attn_mode ∈ {bidirectional, causal}` and `inter_node_pos ∈ {none, rope, learned}`, task-selected. | `autoregressive.md`; the incompatibility is elementary |
| **M8** | **BFS is presented as the safe default ordering. In two of three primary sources it is the worst option tested.** | GRAN App. Table 4 (grid, B=1, K=20): **DFS alone Deg-MMD 1.54e-5 vs BFS alone 0.16** — four orders of magnitude, BFS last of five. Order Matters Table 2 (Cora subgraphs): GraphRNN+BFS 1.125/1.002/0.427 vs uniform random 0.188/0.206/0.200 — **BFS 6x worse than random**. Only G2PT (molecules) finds BFS good. **Make ordering a swept config field.** | `autoregressive.md` **[SINGLE-SOURCE — Critic 1 explicitly did not verify these two tables]** |
| **M9** | **`enable_input_require_grads()` is a COMPLETE NO-OP here, and nothing warns about it.** | It registers a forward hook on `get_input_embeddings()` = `embed_tokens`. **We pass `inputs_embeds` and never call `embed_tokens`. The hook never fires.** `gradient_checkpointing_enable`'s automatic call is a no-op for the same reason. What saves us is *incidental*: our trainable encoder produces `inputs_embeds` with `requires_grad=True`. **It vanishes the moment that stops being true** — a frozen-encoder ablation, a `torch.no_grad()`, a stray `.detach()`, a cached-embeddings speedup. Then with `use_reentrant=True` the LoRA params get `grad=None` **with no error** and loss goes flat. | **[2x]** `transformers/modeling_utils.py` lines 2216–2246, 3226–3234 |
| **M10** | **No compute budget anywhere.** | Now supplied by `environment.md`: 12 GB laptop GPU at 80 W. **The §8 ablation grid crossed with per-config LR sweeps and 3 seeds is arithmetically impossible as specified.** See §7 and §9. | `environment.md` (measured locally) |

### 1.4 Traps and false provenance — do not propagate these

| # | Item | Status | Detail |
|---|---|---|---|
| **T1** | "E5 spectral recipe **from arXiv 2605.22471 §A.7**" — including the code docstring at `depthwidth.md:768-769` and the summary at `:964` | **[FABRICATED] provenance** | Critic 1 fetched §A.7/§3.2 twice. The paper uses the **unnormalized** `L = D − A` and a token that **explicitly includes raw eigenvalues**: `x_v = (u_1(v),…,u_n(v), λ_1,…,λ_n) ∈ R^{2n}`. No sign flips, no standardization, no dropping of the trivial eigenvector. **The advice is good engineering — it is OURS, not theirs.** Fix the docstring in place before anyone codes from it. |
| **T2** | `DarioVajda/graph_model` → `src/models/configs/gtlm_llama_v0_1b.json` | **Legacy trap; first file you find** | It is a **v0** config and every graph hyperparameter is wrong vs the paper: `max_spd 16` (not 8), `max_rw_steps 4` (not 16), `magnetic_dim 16` (not 32), `magnetic_q 0.2` (not 0.25), nested `graph_attn_bias` dict replaced by a flat mixin. Used only by the backward-compat experiment. The **current mixin defaults** in `src/models/config.py` (`max_spd=32`, `max_rw_steps=8`) *also* differ from the paper. **Read `src/experiments/*/config.py` for what was actually run.** |
| **T3** | "Magnetic Laplacian, `q = 0.25`, `dim 32`" | **Misreads a knob, and the whole family may be a no-op for us** | (a) `magnetic_dim: int = 32  # model bias-MLP hidden width` — **not** 32 eigenvectors. The spectral dimension is a *separate* knob `magnetic_m`, default `m=0` = **keep all N eigenpairs**, i.e. a full `O(N³)` complex Hermitian eigendecomposition per graph. Trivial on ≤60-node ego subgraphs; **not** at N=2708. (b) **`Θ_uv = 2πq(A_uv − A_vu)`, so for an undirected graph `Θ ≡ 0`, `exp(iΘ) = 1`, and `L_N` collapses to the ordinary normalized Laplacian for ANY q.** Cora and the OGB molecular sets are undirected. **GTLM's strongest ablation (I10) may not transfer to Phase 1 or 2 at all.** (c) Geisler et al. (arXiv 2302.00049), who introduced the encoding, recommend `q = q'/d_G` with `q' ∈ {0.1, 0.25}` and warn performance *"drops severely (corresponds to absolute q > 0.05)"* — GTLM's absolute 0.25 is **5x above that threshold**. Revisit at Phase 4 (gene regulatory networks *are* directed); sweep q. |
| **T4** | "Total 173,056 bias params = **0.015%** of Llama-3.2-1B" | **Number right, percentage loose** | 173,056 is exact and verified three ways (4,096 SPD + 50,688 RRWP + 118,272 Magnetic), quoted verbatim in the paper's Limitations. But `173,056 / 1,235,814,400 = ` **0.0140%**. Against non-embedding params (973,146,112) it is 0.0178%. **Quote 0.014%, or write "~0.015% as reported" and state the denominator.** |
| **T5** | §9: "GaLA — arXiv 2606.15633" cited as if GaLA were the title | **Bibliography error, plus a motivation that does not transfer** | Real title: *"Formalizing and Mitigating Structural Distortion in LLM Attention for Graph Reasoning"*. **GaLA (Graph-aligned Language Attention)** is the method in §6. ID, authors (Loveland, Trivedi, Weinstein, Huang, Koutra) and venue (KDD '26) are correct **[2x]**. More importantly: GaLA's central premise is that **RoPE turns graph linearization into bandwidth-dependent attention decay** — it is a repair for *verbalization*. We never verbalize and RoPE is the identity, so **the pathology GaLA fixes is one we never create.** The mechanism transfers; the motivation and the token→node mapping do not. |
| **T6** | §9 GaLA: "Forcing the bias onto every head **damages the language machinery**" | **[UNVERIFIED] [CONFLICT #2]** | The bias agent asserted the *opposite* as "(certain)" from Table 2 (all-heads *improves* on every column, e.g. Cora 83.3→89.1); Critic 1 fetched the same HTML and could not see the table. **Neither the spec's claim nor the correction is verifiable right now.** Per-head remains the right design (Graphormer, GTLM and T5 are all per-head) — cite a different reason. GaLA also publishes **no numeric value for β**, only *"selected using a small validation set of 200 samples"*, and there is **no public code**. |
| **T7** | §9 GaLA: "applied only in the first half of layers", presented as a result to copy | **Mechanism verified; justification is one prose clause with no ablation** | The paper's only ablations are head selection and runtime. The layer schedule rests on *"leaving later layers to translate these representations into natural language outputs"* — **a rationale specific to a model that must emit text. We decode a graph.** GTLM and Graphormer bias *every* layer. **Sweep `{all, first half, last half, every other}`.** |
| **T8** | §9 Graphormer: "learnable embedding indexed by shortest-path distance" | **Two omitted traps** | (a) The paper says unreachable pairs get *"a special value, i.e., −1"*; the shipped Cython uses **510**, and the collator adds +1 so it indexes row 511 (hence `num_spatial=512`). Implementing "−1" literally means indexing an embedding with a negative number. (b) Graphormer does not merely soft-bias distant pairs — it **HARD-MASKS** pairs at `spatial_pos ≥ spatial_pos_max` to `-inf`. **Do NOT copy the hard mask.** Under 15% edge masking, a hidden edge can make a pair look unreachable, and `-inf` makes that edge **unrecoverable**. Also, use `torch.finfo(dtype).min`, never `float('-inf')`. |
| **T9** | §1: "**Every existing method converts the graph into something else before the LLM sees it**" | **Overstated — see §2** | GTLM, GaLA, UniGTE, GL-Fusion, HLM-G and **Graph Language Models (Plenz & Frank, ACL 2024, arXiv 2401.07105 — 16 months before GTLM)** all inject graph structure directly into a pretrained LM's attention. Plenz & Frank **re-index T5's own relative-position bias table by graph distance**. The attention-bias half of our architecture has substantial prior art. |
| **T10** | §1: GraphToken / GraphGPT / LLaGA described as "**GNN soft prompts**" | **Two of three are not** | GraphToken Table 3: **NodeSet and EdgeSet have ZERO body parameters** — linear set encoders, no message passing — and Table 2 shows them **winning** node count (0.996) and edge count (0.426). LLaGA's Neighborhood Detail template is explicitly *"parameter-free"*. Only GraphGPT genuinely uses message passing. **Our E6 "DeepSets, not a GNN" is close to GraphToken's NodeSet — we cannot claim that space is unexplored.** GraphToken's paper also **contradicts itself** on projection size (§4.2.3 "approximately 80,000 parameters" vs Table 3 "1.1e7"): **never cite a GraphToken projection param count.** Its soft-token count is never stated anywhere. |
| **T11** | §13: "No tokenizer, no embedding table" framed as fully novel | **One partial precedent** | InstructGLM (arXiv 2308.07134, EACL 2024 Findings) expands the LLM vocabulary with one new token per node and **initialises those embeddings with raw node feature vectors**. So "non-text vectors have never been placed in an LLM's embedding space for graphs" would be false. Keep the design; soften the claim: nobody puts a projected **adjacency row** there, and nobody bypasses the tokenizer entirely. |
| **T12** | §1: verbalization has "O(N²) token blowup" | **O(N²) only in the dense limit** | For sparse Cora the realistic verbalizations are edge-list O(E) and incident-list O(N+E). Running TLAG's own encoders on N=2708, E=5278: ~79.5k chars (edge list), ~155k chars (incident) → roughly **22.7k–31.8k and 44.3k–62k tokens** vs our **2,708**. **State it as "8–16x on Cora, and O(N²) only in the dense limit."** Those counts are chars/token **estimates** — re-measure with the real Llama tokenizer before publishing. |
| **T13** | §1 frames Talk Like a Graph as a method whose numbers we beat | **TLAG trains nothing** | A prompting study on frozen PaLM 62B over synthetic graphs of single- to low-double-digit node count (~100x smaller than Cora). **Beating its scores proves nothing.** Cite the **failure modes**: best edge-existence accuracy **49.0% against a 53.96% majority baseline** (their own admission, p.4), and **~0% on disconnected nodes** — *"LLMs lack a global model of a graph"* (p.7 §3.5). **[2x]** — Critic 1 read the actual PDF pages. |
| **T14** | §9 Depth-Width: "Code in supplementary material" | **Nothing is retrievable** | The v3 NeurIPS checklist says *"We will provide in the final version a Github project with the code."* No repo exists. OpenReview returns HTTP 403 / `ChallengeRequiredError` for both the PDF and the supplementary. **Budget for reimplementation (~40 lines; a version is in `depthwidth.md`).** The paper also **never states** its pooling method, output head, optimizer, or the §6.3 head count — searched in v1 and v3 for `pool`, `readout`, `CLS`, `Adam`, `optimiz`: zero hits. **Do not fabricate these from that paper.** |
| **T15** | §9's Depth-Width empirical targets (61.87 / 67.63 / 68.64) | **A straw man** | All nine numbers are exactly right, but they come from a deliberately plain transformer whose width grid was only `{32, 64}`. The same authors' 2026 follow-up (arXiv 2605.22471 Table 1) runs identical tokenization with a proper grid: **Adj/Pad ROC-AUC 71.31 (HIV) / 67.90 (BBBP) / 74.88 (BACE)**. Use those. **Honest caveat Critic 1 insisted on:** adjacency tokenization **LOSES to GIN on molhiv** (71.31 vs 73.20), and a **DeepSet that ignores edges entirely** gets 74.82 (HIV) and 75.22 (BACE), beating both GIN and adjacency tokenization. Our feature-only baseline is already vindicated in the literature. |

---

## 2. Literature landscape and an honest novelty assessment

This section is the related-work map and the exact wording of the novelty claim that survives
review. It compresses `competitors.md`, the positioning halves of `gtlm.md`, `fpt.md`,
`depthwidth.md` and `autoregressive.md`, and keeps every caveat the critics attached.

### 2.1 The five families, and where each one stops short of us

| Family | Representatives | What the LLM sees | What comes out | Where it stops short |
|---|---|---|---|---|
| **Verbalization** | Talk Like a Graph (arXiv 2310.04560, ICLR 2024); InstructGLM (2308.07134) | sentences about edges | text / a label | Best edge-existence accuracy **49.0%** vs a **53.96%** majority baseline; **~0%** on disconnected nodes — *"LLMs lack a global model of a graph"* (TLAG p.4 §3.1.1, p.7 §3.5, **[2x]** — Critic 1 read the PDF pages). Costs 8–16× our tokens on Cora, O(N²) only in the dense limit (T12). TLAG trains nothing; beating it proves nothing (T13). |
| **Compressed soft prompts** | GraphToken (2402.05862); GraphGPT (2310.13023); LLaGA (2402.08170); GraphPrompter (2402.10359); `<SOG_k>` (2602.01771) | a few projected vectors prepended to a text prompt | text / a label | The one genuine LLM link-prediction AUC in this family is GraphGPT's PubMed **0.8246 AUC / 0.8026 AP** — ~14 AUC points behind Kipf's 2016 GAE (96.4/96.5) (`competitors.md` §B2). GraphPrompter does not beat plain GAT on Cora LP (90.10 vs 90.71 accuracy). NOTE: this family is **not** all GNNs — GraphToken's NodeSet/EdgeSet have zero body params, LLaGA-ND is parameter-free (T10). |
| **Attention-bias injection into a pretrained LM** | Graph Language Models (Plenz & Frank, ACL 2024, 2401.07105 — **16 months before GTLM**); GTLM (2605.10247); GaLA (2606.15633, KDD '26); UniGTE (2510.16885, NeurIPS 2025); GL-Fusion (2412.06849); HLM-G (2410.22372) | node **text**, attention biased by structure | text / a label | This is the half of our architecture with substantial prior art (T9). Every member feeds text tokens and outputs text or a classification. **None decodes a graph.** |
| **Adjacency-row tokenization** | Depth-Width (2503.01805, NeurIPS 2025 Spotlight); Lost in Tokenization (2605.22471) | rows of A into a **from-scratch** transformer | a label / a count | The input-format precedent, with theory (§3.1 below). Not a pretrained LLM; never a graph out. |
| **Autoregressive graph generation** | GraphRNN (1802.08773); GRAN (1910.00760); G2PT (2501.01073); AutoGraph (2502.02216); BiGG (2006.15502) | n/a — trained from scratch | a graph | Graph-out exists here, but with no pretrained language weights anywhere. G2PT's own Table 3: adjacency-matrix tokens vs edge-sequence tokens is V.U.N. **94 vs 95** — our representation is not the bottleneck their prose implies (`autoregressive.md` C8/C9). |

The frozen-LLM-on-a-foreign-modality recipe itself (linear in → frozen body → linear out, LN
affine trained) is old and contested: FPT (2103.05247) established the narrow claim
frozen-pretrained > frozen-random (+2.5…+6.5; replicated at +4.5…+10.5 by the rebuttal
2107.12460 — which simultaneously overturned the headline "frozen matches full training", I15),
and Tan et al. (2406.16964, NeurIPS 2024 Spotlight) then showed three FPT-style time-series
systems lose nothing when the LLM is deleted (M3). GPT4TS (2302.11939) holds the low-data
counter-evidence: pretraining is worth a lot at 5–10% data, ~nothing at 100% (`fpt.md` C3).
Cora link prediction is small-data — that regime is on our side, but it must be argued with a
data-fraction sweep, not assumed.

### 2.2 The two claims that survive, verbatim

Checked against targeted arXiv full-text queries (all zero hits) and two 2026 surveys
(2605.03514, 2606.06865) that enumerate the field and contain no counterexample
(`competitors.md` §C1):

> **(a) Input:** the LLM's per-node token is a learned projection of the **raw adjacency row**
> — not text, not a message-passing summary, not a set-pooled embedding.
>
> **(b) Output:** a trainable decoder produces an **adjacency matrix from the LLM's hidden
> states**. No published work decodes a graph out of a pretrained LLM. This is the stronger
> claim; no counterexample of any kind was found.

Supporting quotable: GTLM's own Future Work names autoregressive graph generation with
`<ADD_EDGE>` control tokens as the open problem (verbatim in `gtlm.md` §1.6). That is the
closest group in the world telling reviewers our output half is unsolved.

A third, smaller claim: the **frozen-random-init control inside a graph-LLM system** appears
never to have been run (GraphToken, GTLM, GaLA all skip it) — our §8 grid would be the first
(`fpt.md` OQ8, [UNVERIFIED] as a negative — absence of evidence).

### 2.3 What we may NOT claim (the eaten-novelty ledger)

1. **"Every existing method converts the graph before the LLM sees it"** — false as stated
   (T9). Six papers inject structure directly into pretrained-LM attention. Say "converts the
   graph *content* into text or pooled vectors" and cite them.
2. **Attention-bias-in-a-frozen-LLM** — Plenz & Frank re-index T5's own relative-position bias
   table by graph distance (ACL 2024); GTLM, GaLA, UniGTE all do learned/fixed additive biases.
   Our bias is engineering, not novelty. (`competitors.md` correction 3.)
3. **One token per node** — GraphToken node-level readout, GraphGPT (n tokens per subgraph),
   LLaGA all already do a per-node token budget. **Provenance of the token is the
   differentiator, not the count** (T10, `competitors.md` C2(iii)).
4. **Non-message-passing encoders into a frozen LLM** — GraphToken's NodeSet (0 body params)
   already won node-count in its own paper. E6 must cite it as the nearest neighbour (T10).
5. **"No tokenizer / no embedding table" as fully novel** — InstructGLM initialises new
   per-node vocabulary embeddings with raw feature vectors (T11). Keep the narrower claim: no
   one puts a projected adjacency row there, and no one bypasses the tokenizer entirely.
6. **SPD-bias + frozen decoder + reconstruction + permutation invariance in one system** —
   UniGTE has all four (its reconstruction is a natural-language paraphrase through a 64-token
   bottleneck, not an adjacency matrix). **UniGTE is the single biggest novelty threat and must
   be in every related-work section** (`competitors.md` correction 4).

### 2.4 The empirical case against the incumbents (introduction material)

- *Revisiting Graph-Tokenizing LLMs* (2605.03514): accuracy drops 3–50% under
  semantics-preserving rephrasing; under relabeling most models fall below 50%, some under 10%.
- *When Graph Tokens Sink* (2606.03712): soft graph tokens become high-activation sink tokens
  with a "severe decoupling between activation-level saliency and graph-semantic utility";
  pruning the top-2 costs 0–1% accuracy. **Also a risk for us** — run their pruning diagnostic
  on our tokens (§8 R11).
- GraphGPT PubMed LP 0.8246 AUC vs GAE 96.4 — the graph-out side is unsolved in the LLM
  literature (`competitors.md` §B2).
- Heuristic reality check: neighbourhood-overlap heuristics beat GNNs on several link-prediction
  datasets (Neo-GNN, 2206.04216, [SINGLE-SOURCE] M5); a DeepSet that ignores edges beats GIN on
  ogbg-molbace/molhiv (2605.22471 Table 1, T15). Our Phase-1 table exists to pre-empt exactly
  this question.

### 2.5 Comparison targets by task (who we are actually racing)

| Our task | The bar | Not the bar |
|---|---|---|
| Masked edge prediction, Cora | **MaskGAE 96.45 AUC / 95.95 AP** (2205.10053 Table 3, P2 protocol); SEAL 92.22/93.12, ARGA 92.40/93.23 | GAE 91.0/92.0 (floor, I3); TLAG (trains nothing, T13); LLaGA/GraphPrompter accuracies (different metric + protocol, §6) |
| Molecular masked edge / property (Phase 4 parallel track) | Adj/Pad 71.31 / 67.90 / 74.88 ROC-AUC (2605.22471 Table 1) + GIN + DeepSet in the same harness | the Depth-Width 61.87/67.63/68.64 straw man (T15) |
| AR generation (Phase 5) | G2PT_base V.U.N. 100/99/100/100 on Planar/Tree/Lobster/SBM; GRAN/GraphRNN MMDs under their exact kernels | any cross-kernel MMD comparison (`autoregressive.md` comparability rules — GraphRNN grid Deg 1e-5 vs GRAN's re-eval of the same model 1.12e-2) |

---

## 3. Architecture decisions, decided

Every subsection states the decision as locked by the approved plan
(`~/.claude/plans/can-you-do-more-noble-sunset.md`), then the evidence. Where a notes file
disagrees with the plan, the plan wins and the disagreement is flagged.

### 3.1 Backbone

**DECIDED: Llama-3.2-1B (`meta-llama/Llama-3.2-1B`, gated — user accepts the HF license and
creates a token when Phase 2 needs it). Qwen3-0.6B is the later second point, not an
alternative first point.** (Plan, user decision.)

Config, verified from the byte-identical `unsloth/Llama-3.2-1B` mirror (`pyg-baselines.md`,
`hf-mechanics.md` **[2x]**):

```
hidden_size 2048 · num_hidden_layers 16 · num_attention_heads 32 (query) ·
num_key_value_heads 8 (GQA, G=4) · head_dim 64 · intermediate_size 8192 ·
vocab_size 128256 · rope_theta 500000 · rope_type "llama3" (attention_scaling = 1.0) ·
tie_word_embeddings true · total params 1,235,814,400 · non-embedding 973,146,112
```

Backbone comparison table (all figures read from shipped `config.json`s; `hf-mechanics.md`
§"Backbone comparison" — the §6 forward-reference from §1 resolves HERE):

| Model | `d_model` | `L` | `H` (Q) | `H_kv` | `head_dim` | `L·H` | bf16 weights | License |
|---|---|---|---|---|---|---|---|---|
| **Llama-3.2-1B** | 2048 | 16 | **32** | 8 | 64 | 512 | ~2.5 GB | Llama 3.2 Community (gated) |
| Llama-3.2-3B | 3072 | 28 | 24 | 8 | 128 | 672 | ~6.4 GB | gated |
| Qwen3-0.6B | 1024 | 28 | 16 | 8 | **128** (≠ 1024/16!) | 448 | ~1.2 GB | Apache-2.0 |
| Qwen3-1.7B | 2048 | 28 | 16 | 8 | 128 | 448 | ~3.4 GB | Apache-2.0 |
| Qwen3-4B | 2560 | 36 | 32 | 8 | 128 | 1152 | ~8.0 GB | Apache-2.0 |
| SmolLM2-1.7B | 2048 | 24 | 32 | **32 (MHA)** | 64 | 768 | ~3.4 GB | Apache-2.0 |
| Gemma-3-1B | 1152 | 26 | 4 | 1 | 256 | 104 | ~2.0 GB | gated |
| Llama-3.1-8B | 4096 | 32 | 32 | 8 | 128 | 1024 | ~16 GB | gated |

Why Llama-3.2-1B first (`hf-mechanics.md` recommendation, adopted by the plan):
1. **GTLM comparability** — the one external system whose numbers we can check ourselves
   against runs this exact backbone; its 173,056 bias params factor as 16×32×338 against this
   model's L and H, and its 5e-3/3e-5 LR split transfers without reinterpretation.
2. `d_model=2048` keeps our modules cheap (E1 ≈ 5.5M, D1 ≈ 4.2M; §7).
3. 32 query heads give per-head bias enough resolution (Gemma's 4 do not).
4. Memory fits (§7): full-graph Cora eager ≈ 45 GB attention, H100-viable with gradient
   checkpointing.

Disqualifications: **Gemma-3-1B** — 5 of 6 layers are sliding-window-512, global graph
attention is the whole point, and 4 heads is too few. **Qwen2.5-0.5B** — a generation behind,
odd G=7. **SmolLM2-1.7B** is kept in reserve as the only MHA candidate: if GQA broadcasting is
ever suspected as a bug, it removes the variable (`hf-mechanics.md`).

Theory sizing against Cora (depthwidth.md, F3/F4/F5 recomputed):
- Thm 4.2 (`mpHL = Ω(N)`): satisfied by **6,196×** — imposes nothing (F3).
- Thm 4.4 constructive width `2d·log₂N+2` at max degree 168 = **3,833 > 2048**: the ~27 hub
  nodes (p99 degree 19 → 435, fine) are under-width. Diagnostic, not blocker: bucket AUPRC by
  node degree and check whether error concentrates on hubs (depthwidth.md OQ5).
- Depth: connectivity needs Ω(log n) depth even at full width (2605.22471 Thm 4, conditional on
  TC⁰⊊L); ⌈log₂ 2708⌉ = 12 ≤ 16 layers. **Every candidate clears depth; width is marginal only
  at hubs.**
- Thm 4.3's 3N-wide construction (8,124) is unreachable for every candidate — but its only job
  is to make softmax(QK) ≈ A, which our bias supplies directly. That exemption is a
  **plausible, unproven conjecture** (depthwidth.md ⚠, OQ1) — never write it as settled.

Scale argument for staying at 1B: GTLM's own scaling table is flat (Cora 88.99 → 89.17 → 89.25
across 1B/3B/8B; `gtlm.md` §4.1). Spend the compute on the encoder study instead. If width per
FLOP ever becomes the binding issue, Qwen3-4B (d=2560, L=36) is the better trade than 8B
(depthwidth.md §6.1 reading).

### 3.2 Encoder + output normalization

**DECIDED: E1–E7 behind one interface in `g2l/encoders.py`; orthogonal init gain 1.41 on every
projection, bias 0; a LayerNorm on every encoder output with gain initialised to the scalar
`T/√d_model` (T = mean L2 norm of the non-zero rows of `embed_tokens.weight`, computed at
runtime from the loaded checkpoint); per-encoder LR sweep (encoder group 1e-3…1e-4).** (Plan;
`fpt.md` §CONCRETE; M1.)

The encoder menu, with the corrections baked in:

| E | Definition (input → `[B,N,2048]`) | Params (Cora) | Corrections applied |
|---|---|---|---|
| E1 | `Linear(2708, d)` on each row of A_obs | 5,548,032 | reference point; ties model to N |
| E2 | 2–3-layer MLP | ~9–13M | LR ≈ half of E1's — LLaVA-1.5 halved the projector LR when moving linear→MLP; reusing E1's LR confounds the sweep (`fpt.md` hyperparameter table, second rule) |
| E3 | `A @ R` (fixed `R∈R^{N×k}`, `R_ij~N(0,1)` sampled once) → `Linear(k,d)`; **k=512**, plus the node's own codeword block (a second `nn.Embedding(N,k)` or fixed codebook) | ~1.1M + codebook | k=512 ≈ ln³N (AGM-flavoured); 2605.22471's d_tr=8 is a brittleness probe, not a recipe; the codeword block comes from Thm 4.4's construction (depthwidth.md C6, "Design lesson for E3"). **Do not claim AGM proves connectivity is retained** — their sketch needs a specific decoder; ours is a JL projection (C6). |
| E4 | `Linear(N+F, d)` on `[A_i ‖ x_i]`, F=1433 | 8,482,816 | verbatim the Depth-Width Appendix-E recipe (their one exactly-right CLAUDE.md claim) |
| E5 | `[A_i ‖ U_i ‖ deg_i ‖ x_i]` with the **normalized** spectral recipe | ~9M | **The recipe is OURS, not 2605.22471's** (T1, [FABRICATED] provenance): symmetric normalized Laplacian, drop the trivial eigenvector, random sign flips at load, per-feature standardization, **never raw eigenvalues** (2605.22471 Thm 5: raw-eigenvalue tokens ⇒ ‖X‖²=Ω(n²) on dense graphs ⇒ softmax saturation or weight explosion; qualified "dense", Cora d_max=168 — CONFLICT #5 — but the normalization costs nothing and removes the trap). Eigvecs alone cannot even encode degree (depthwidth.md §3 bonus fact) — the row stays in the token. |
| E6 | DeepSets over the neighbour set: embed neighbour ids/features, one permutation-invariant aggregation, project. **NOT a GNN — one aggregation, no propagation** (CLAUDE.md §13 stands) | small | nearest published neighbour is GraphToken's NodeSet — cite it, do not claim the space is unexplored (T10) |
| E7 | sparse/bucketed: embedding table over nonzero indices + positional hashing | small | size-agnostic; exercised on the molecular track (Phase 4) since single-graph Cora cannot test size-agnosticism |

Output normalization (M1, the modality-interface fix — `fpt.md` §CONCRETE, arXiv 2512.08374
[SINGLE-SOURCE for the numbers, mechanism uncontested]):

1. `nn.LayerNorm(d_model)` after every encoder, **gain filled with the scalar `T/√d`, bias 0**.
   Measured T values: Qwen2.5-7B 0.80, Llama-3.2-3B 1.09, Qwen3-8B 1.38 ⇒ `g_init` ≈
   0.014–0.022. **T for Llama-3.2-1B is unpublished — compute it at load time** (code in §4.1),
   never hardcode.
2. The small gain attenuates the encoder's backward by ~1/g ≈ 65×. Log
   `encoder.weight.grad.norm()` from step 0; if it is orders of magnitude below the decoder's,
   switch on the GWC backward hook (`fpt.md` §5 code — divide the incoming grad by
   `mean(|g|).clamp_min(1e-3)`). On Llama-3.2-3B naive init was fine; on Qwen2.5 it was not.
3. Unit test (Phase-2 list): mean encoder-output L2 within 25% of T at init.
4. Caveat to carry: LayerNorm destroys per-node magnitude, which for an adjacency row is
   degree. E5 feeds degree explicitly; for E1 this is a known information loss, RevIN-style
   "normalize then add the statistic back" is the fallback (`fpt.md` OQ4, unresolved).

Permutation honesty (I4, settled framing): E1–E5 are **not** permutation-equivariant and
cannot be made so — the row's *contents* permute (`t'_{π(i)} = W·PᵀA[i,:]`), a fixed `[d,N]`
matrix is a node-id-indexed encoding. This is accepted for the fixed-node-set setting exactly
as 2503.01805 v3 accepts it. The strict permutation test is scoped to bias+body always and to
E6/E7 full-model; E1–E5 get the weaker index-consistency test (permute nodes AND the encoder's
input columns → output must permute). Documented, not "fixed" (plan Phase 4).

### 3.3 Bias mechanism + injection point

**DECIDED: one learned per-head SPD lookup table, layer-shared, zero-initialised, diagonal
forced to zero, unreachable = its own learned bucket (never −inf), computed from A_observed
only (§3.5), delivered as the 4-D `attention_mask` kwarg ("Implementation A"), differential LR
5e-3 with weight-decay 0. Phase 3 opens with a parameter-free `1/SPD × λ` probe before any
learned table. RRWP/Laplacian-distance are candidate additions, magnetic Laplacian is deferred
to directed data.** (Plan Phase 3; `bias-mechanism.md`; `gtlm.md`.)

Shape and site: `bias ∈ [B, 32, N, N]` float, added to the already-`1/√d`-scaled logits before
softmax — the exact site every lineage member uses (Graphormer `q *= scaling` then
`attn_weights += attn_bias`, G6; GTLM T1; GaLA A2). **32 = query heads, not 8 KV heads** — see
§4.4.

The table itself (GTLM's `SPDBias` shape, adapted):

- `nn.Embedding(max_dist + 2, num_heads)` = `10 × 32` = **320 params** at `max_dist = 8`
  (buckets: distances 1…8, one bucket for >8, one for unreachable-in-observed-graph; distance 0
  = diagonal is multiplied to zero, not a table row — GTLM's convention, and the config
  validator rationale: a self-row is a checkpoint-breaking change, `gtlm.md` §2.2/§2.5).
  **Current `g2l/bias.py` merges ">max_dist" and "unreachable" into one bucket (both →
  max_dist+1).** Splitting them is a cheap Phase-3 ablation; Graphormer keeps them distinct
  (sentinel 510 → row 511, G7/G9), GaLA zeroes unreachable (A3). Either is defensible;
  merged is the current code's choice — record it in the run config.
- **Zero-init + `_is_hf_initialized = True`** (I6, B14–B16): the model at step 0 is
  bit-identical to the no-bias run; "no attention bias" becomes a true nested baseline; and
  `post_init()` cannot re-randomize the zeros. The paper's "bias params start random" is
  contradicted by GTLM's own shipped code — the differential-LR advice survives, the stated
  reason does not.
- **Never a hard mask.** Graphormer hard-masks pairs at `spatial_pos ≥ spatial_pos_max` to
  −inf (T8). Under edge masking a hidden edge can make a pair look unreachable; −inf makes that
  edge unrecoverable. Soft learned bucket, always. Where a mask is genuinely needed (padding,
  causal), use `torch.finfo(dtype).min`, never `float('-inf')`, and never sum two finfo.min
  masks (overflow → NaN; `hf-mechanics.md` overflow trap).
- **Diagonal always open** — NaN guard (a fully-masked softmax row is NaN) and it removes a
  free per-head sharpness knob that would confound the "does the graph do anything" ablation
  (`gtlm.md` §2.5).
- **Layer-shared by construction under Implementation A** (one tensor for the whole stack).
  Precedent: Graphormer shares one table across all layers (G2/G11), T5 owns the table on
  block 0 only (B5). GTLM is per-layer; nobody has ablated shared-vs-per-layer — it is our
  cheap publishable ablation (Phase 6, triggers Impl B). 320 vs 5,120 params.
- **Weight-decay 0 on the table** (B12 mechanism: AdamW's update is scale-invariant, clipping
  bounds the step not the trajectory; a decayed zero-init table fights its own high LR). Our
  param groups are built by name in our own code, so HF's name-based "bias"-exemption trap
  (`gtlm.md` §2.11) does not apply — but keep the shape-based rule if HF Trainer ever enters.

Magnitude discipline (B6–B13, the numbers that decide whether this works):
- Reference scale: `q·k/√d` sits at **O(1–10)** pre-softmax (B8). Working biases in the one
  comparable system: **0.021–0.309**; 52.85 saturated softmax outright; logits >1e3 start
  degrading, >1e4 always diverged (Wortsman B6).
- **Working band: |bias| ≈ 0.05–0.5.** Log `max|bias|` and `max|q·k/√d|` per layer every N
  steps; alert at ratio > 0.5 (plan Phase 3, exact instrumentation).
- Keep the bias **degree-1 in learned parameters** (lookup table: yes; MLP on fixed features:
  yes; product of two trainable factors: no — quartic growth, NaN divergence in 3 GTLM runs at
  bias_lr 2e-2, B10). If a bilinear form is ever unavoidable, L2-normalise before the inner
  product (B13).
- **`bias-nonzero-after-100-steps` assert** (plan): a disconnected feature gate produces a run
  with no bias that trains cleanly and reads as a clean negative result (GTLM's own documented
  failure mode, `gtlm.md` §3.8).

Evidence-ordered roadmap of bias families (`bias-mechanism.md` §"cheap high-value ordering",
adopted):
1. no bias (the floor — B11 shows a bad bias lands *below* it);
2. `1/SPD` with one scalar λ (bounded in (0,1] by construction, cannot blow up — GaLA's shape;
   the plan's day-one Phase-3 probe at N=512);
3. learned per-head SPD table (the decided mechanism);
4. Laplacian-distance bias (`[H]` params = 32 scalars, `w_h · ‖U_i − U_j‖₂`, reuses E5's
   eigenvectors; GTLM repo form nobody's paper highlights);
5. RRWP MLP (subsumes SPD in theory — GRIT Prop 3.1(a), R2; check whether in practice.
   Cost note: dense `P` is `[N,N,16]` ≈ 470 MB fp32 at N=2708 and O(K·N³) dense bmm to build —
   fine once on H100, needs sparse/offline computation, `bias-mechanism.md` OQ6);
6. magnetic Laplacian **only when directed data arrives** (T3: `Θ = 2πq(A_uv − A_vu) ≡ 0` on
   undirected graphs — the mechanism is literally the ordinary normalized Laplacian for any q
   on Cora and OGB molecular; GTLM's "Magnetic is most load-bearing" ablation (I10) was
   measured on GraphQA and cannot transfer to Phase 1–4 data; when it does arrive, sweep q —
   Geisler et al. warn absolute q > 0.05 "drops severely", GTLM's 0.25 is 5× that, T3(c)).

Layer schedule: bias every layer (Graphormer, GTLM) — GaLA's first-half-only has **no ablation
behind it** and its rationale ("leave later layers to translate into natural language") does
not apply to a graph decoder (T7). `{all, first half, last half, every other}` is a Phase-6
sweep. Per-head vs head-shared (`[B,1,N,N]` broadcast) is the built-in ablation knob; the
per-head *justification* is Graphormer/GTLM/T5 precedent, **not** GaLA's "damages the language
machinery" claim, which is [UNVERIFIED] (T6, CONFLICT #2).

### 3.4 Attention backend: eager oracle + FlexAttention on Marlowe

**DECIDED: training runs the eager path through all Cora phases (2–4). Eager is the permanent
numerical-parity oracle. FlexAttention is brought up on Marlowe (Linux + Triton + H100) as a
Phase-3 exit item (parity test only) and becomes load-bearing at Phase 6 scale. SDPA is never
used. FlashAttention-2 is impossible.** (Plan; environment.md; supersedes the *hardware* clause
of I1 and CONFLICT #1's "moot on our hardware" — those were written under the withdrawn
laptop-training premise. §0.2 item 1 and M10 are likewise superseded by `environment.md`'s
H100 rewrite; the *content* of §0–§1 is preserved above unchanged, this section is the
correction of record.)

Why each backend is what it is:
- **Eager**: `attn_weights = attn_weights + attention_mask`, verbatim, no coercion (F1). The
  only path where `output_attentions=True` works (needed for the heads-learn-message-passing
  analysis later). Cost at full Cora: ~45 GB of attention activations without checkpointing,
  ~8 GB with (§7) — H100-viable. That is the whole Phase 2–4 story.
- **FlexAttention**: the bias never materialises at token level — a `score_mod` closure
  gathers `node_bias[b,h,q_idx,kv_idx]` inside the Triton kernel. GTLM's measured numbers on
  H100: cora eager 836.9 → flex 226.8 ms (3.69×), ogbn-arxiv 2006.2 → 208.3 ms (9.63×), peak
  memory at L≈4.5k 30 GB → 0.2 GB, "No number in the paper changes" (`gtlm.md` §2.10,
  REBUTTAL.md **[2x]**).
- **CONFLICT #1 resolution, operationalised**: gradients flow to tensors captured in
  `score_mod` (`torch/_higher_order_ops/flex_attention.py`: "score_mod_other_buffers can
  require grad"; the mask_mod-only restriction is current `main`), **but a leaf `nn.Parameter`
  captured directly can raise** (pytorch#145460). The bias must reach `score_mod` as a
  **non-leaf tensor** — e.g. the output of the embedding lookup / a `weight * 1.0` view —
  which is exactly what GTLM passes. The plan's Phase-3 exit item encodes this.
- **SDPA**: the mem-efficient backward is "buggy (GQA + per-head bias)" per GTLM's own
  dispatch docstring, corroborating pytorch#125674 (NaN grads); flash rejects any non-null
  `attn_mask` (`check_for_attn_mask` in `sdp_utils.cpp`, verbatim in `pyg-baselines.md`).
  Never pass our bias through `F.scaled_dot_product_attention`.

Flex bring-up checklist (from GTLM's optimization log + PyG notes, for the Phase-3 parity item
and Phase 6):
1. `torch.compile` required ⇒ Marlowe only; caches on scratch (`TRITON_CACHE_DIR`,
   `TORCHINDUCTOR_CACHE_DIR` — set in the sbatch template); **never `module load nvhpc`**
   without re-exporting `CC=gcc CXX=g++` (Inductor shells out to `$CXX`; `marlowe.md` §7.2).
2. **Pad N to a multiple of 128** (2708 → 2816) and mask: ~14× perf cliff otherwise, and open
   NaN bug pytorch#153799 when block_mask + score_mod meet a non-multiple length
   (`pyg-baselines.md` risk list).
3. `torch._dynamo.config.cache_size_limit` above the number of (L,N) shape buckets, or it
   silently falls back to eager (default 8).
4. `BLOCK_SIZE` 128 at k_hop=0; compile mode `max-autotune-no-cudagraphs`.
5. **Parity gate: flex vs eager agree to ~1e-3 in bf16 on a small graph, and bias-parameter
   gradients match** (GTLM has `test_flex_bias_grad_parity`; ours is the Phase-3/Phase-6 gate
   item). Re-verified at Phase-6 scale before any flex-trained number is reported.
6. Laptop (Windows / cp314 / sm_120): flex assumed unavailable; nothing depends on it — the
   laptop runs eager N≤512 smoke tests only. [UNVERIFIED whether compile works there; nobody
   needs it to.]

### 3.5 THE M4 LEAK DECISION: all structural features from A_observed only

**DECIDED (the answer to M4, stated once, enforced everywhere): every structural feature —
SPD, RRWP, Laplacian eigenvectors, degree, orderings, anything derived from adjacency — is
computed from the OBSERVED adjacency only. Never from the complete graph. No exceptions, no
per-feature carve-outs. `g2l/bias.py`'s module docstring states the rule; `tests/test_leaks.py`
enforces it; the Phase-0 gate armed it and Phase 3 exercises it for real.** (Plan, locked.)

The tension this resolves (M4 was "the highest-risk unknown in the design"): structural
features must come from *some* adjacency. From the complete graph, every held-out edge leaks
straight into the attention bias — SPD(u,v)=1 announces the label. From the observed graph,
masking inflates distances and manufactures spurious unreachability *precisely on the pairs we
are asked to predict as edges* — the bias actively asserts "far apart" about the positives. No
source paper faces this because none does masked edge prediction.

Why observed-only wins, in order:
1. **A leak is a validity failure; distortion is a measurable signal cost.** A complete-graph
   bias invalidates every number produced with it, silently and unrecoverably. Distortion just
   caps how much the bias helps — and Phase 3's delta table measures exactly that.
2. **Test-time consistency.** At evaluation the complete graph does not exist; a bias trained
   on complete-graph SPD would face a train/test feature shift on exactly its input. Training
   and testing both on observed-graph features is the only distribution-consistent choice.
3. **Symmetry with the baselines.** CN/AA/RA/PPR in `baselines/heuristics.py` all run on the
   observed (train+val at test time) graph. One rule for every row of every table.
4. **The distortion is partially learnable.** Because the unreachable/far bucket is a *learned
   scalar*, not −inf (§3.3), the model is free to learn that "far apart in the observed graph"
   is weak evidence, or even positive evidence, of a masked edge. A hard mask would make the
   error unrecoverable — this is why T8's warning and this decision are the same decision.

Mechanics under the two masking views (§5.4):
- **Edge-split view**: "observed" = the split's own message-passing graph — train edges during
  training, train+val edges at test scoring (RandomLinkSplit semantics; this is what
  `baselines/common.py` already does).
- **Matrix-fraction view**: `mask_matrix` zeroes both directions of every hidden pair; SPD is
  recomputed from that A_obs per masking draw. Cost: `scipy.sparse.csgraph.shortest_path` on
  N=2708 — seconds per draw, negligible (§7).

Enforcement (the tests, exactly as they exist in `tests/test_leaks.py`):
- `test_spd_depends_only_on_observed_graph` — build A₁, mask it, then flip **every held-out
  cell** to make A₂ (a materially different complete graph with an identical observed part);
  assert `spd_matrix(A₁_obs) == spd_matrix(A₂_obs)` exactly. Any structural-feature function
  added later gets the same flip test.
- `test_identity_scores_at_chance` — the permanent canary: a model that outputs its own input
  scores AUROC 0.500 / lift 1.000 to 1e-9 on held-out cells. Runs in every phase's table
  (identity row recomputed) and in the walkthrough.
- Phase-3 model-level version (armed at Phase 0, exercised Phase 3): two graphs differing only
  in held-out cells must produce **bit-identical bias tensors** end-to-end through
  `GraphAttentionBias`.
- Phase-5 extension: orderings are structural features too — `test_ordering_does_not_leak_labels`
  (two graphs differing only in held-out edges ⇒ same ordering), and for the AR task the clean
  answer is **no masking at all** (causality hides the future; ordering computed on the full
  training graph is then legal) (`autoregressive.md` C4 option 3, adopted by plan Phase 5).

Known residual (open, §10): under AR, `bias(t,j)` computed on the full graph leaks the future
through the bias even when the mask is causal; prefix-recomputation is O(N) SPD passes. Parked
until Phase 5 (`autoregressive.md` OQ2).

Rejected alternatives, recorded: (a) complete-graph features (leak, above); (b) zero-bias on
masked pairs (tells the model which cells are queried — not a label leak under the matrix
view, but it changes the task and cannot be replicated at deployment where the query set is
"every unobserved pair"; kept as a Phase-3 ablation only); (c) per-draw feature recomputation
was a cost concern on the laptop premise — on H100/CPU it is noise.

### 3.6 RoPE / mask

**DECIDED: `position_ids = torch.zeros(B, N, dtype=long)` — no RoPE code, no module surgery;
startup assert `attention_scaling == 1.0`. `config.is_causal = False` set statically at load.
The 4-D bias mask already implies bidirectionality (it early-exits before any causal mask is
built); `is_causal=False` is belt-and-braces for the no-bias ablation where
`attention_mask=None`. `attn_mode ∈ {bidirectional, causal}` is a config field from Phase 2
(the AR hedge).** (Plan; I8; F3/F4.)

- With one token per node, "RoPE reset per node" and "RoPE disabled" coincide *exactly*:
  position 0 ⇒ `freqs=0` ⇒ `cos=1·attention_scaling, sin=0` ⇒ `apply_rotary_pos_emb` is the
  bit-exact identity, **conditional on `attention_scaling == 1.0`** — true for rope_type
  `llama3` and `default`, false for yarn/longrope (F4). The assert runs at every training
  start on both machines (Phase-2 test list).
- `position_ids` must be `[B, N]` and must be passed explicitly — omitting it auto-builds
  `arange(N)` and silently re-enables RoPE (`hf-mechanics.md` pitfall 4).
- Consequence worth restating (from `gtlm.md` §2.8): with all positions equal and no bias,
  attention is fully permutation-invariant over a set — **the attention bias is the only
  channel through which pairwise structure reaches the attention map**, which is what makes the
  no-bias ablation sharp.
- Equivariance bookkeeping (I9): the body-level property needs all three ingredients — position
  reset, bidirectional mask, and bias computed from static topology (not sequence index). We
  have no prompt node, so our body-level claim is strictly stronger than GTLM's prefix-only
  proof — say so, don't cite theirs as identical. Encoder-level equivariance is a separate,
  mostly false, property (§3.2).
- Causal mode (Phase 5): right-shifted input rows (`X_in[t] = L[t-1]`, row 0 = all-zeros SOG),
  causal mask, **no masking of A** (§3.5), ordering swept over {BFS, DFS, degree-desc, k-core,
  random} — BFS is the *worst* single ordering in two of three primary sources (M8; GRAN
  Table 4: DFS 1.54e-5 vs BFS 0.16 Deg-MMD) and the plan supersedes CLAUDE.md's "use BFS".
  Causality test: `grad(logits[t], tokens[>t]) == 0` (Gate 5).
- Padding (multi-graph batches, Phase 4 molecular): fold the padding into the same 4-D bias
  tensor — `finfo.min/1` at padded **key** columns, diagonal kept open, never two stacked
  finfo.min masks. Or sidestep with B=1 + gradient accumulation (legitimate; `hf-mechanics.md`
  OQ7). `collate()` already guarantees padded cells never enter loss or metrics.

### 3.7 Decoder

**DECIDED: D1 (own ~10-line `σ(H W Hᵀ)`) and D3 (pair-MLP) from Phase 2, both; D2 (low-rank)
and D4 (autoregressive) later.** (Plan; I13; M6.)

- **D1**: `logits = H @ W @ H.T`, `W ∈ R^{2048×2048}` (4,194,304 params), init `eye × 0.1`
  (matches `baselines/scratch_transformer.py` so the scratch curve is apples-to-apples).
  Logits are symmetrized `(L + Lᵀ)/2` before scoring — `evaluate_edge_split` already does
  this, so direction conventions cannot matter. PyG's `InnerProductDecoder` **cannot** host a
  W (its forward is literally `(z[i]*z[j]).sum(-1)`, no `__init__`, I13) — it stays imported
  for the GAE baseline only.
- **D3**: `MLP([h_i ‖ h_j ‖ h_i·h_j])`, input 6144. Why it is in week one and not "later":
  the labeling-trick theorem (M6, [SINGLE-SOURCE] but the mechanism is standard) — any decoder
  that scores from two independently-computed node states gives automorphic pairs identical
  scores no matter how good H is; this is the mechanism behind the GAE-91 vs SEAL/MaskGAE-96
  gap. **Measuring the LLM's contribution through a D1-only bottleneck could suppress the very
  effect Phase 2–3 exist to detect.** D3 relaxes the bottleneck at the pair level. MaskGAE's
  `EdgeDecoder` (Hadamard-product-only MLP) is the published nearest form
  (`competitors.md` §Code 1). Cost: never materialise all 7.3M pairs × 6144 — score only the
  loss/eval cells (matrix view: ~550k cells; edge-split training: positives + sampled
  negatives), chunked. A hidden width of 512 costs ~3.1M params [our choice, not locked —
  record in config].
- **D2** (rank 32–128 bilinear) is a regularised D1 — Phase 6 sweep. **D4** — Phase 5, with
  GRAN's mixture-of-Bernoullis (K=20) as the reference head and the `num_mix>1` path
  restricted to molecular N (it materialises `[B,N,N,d]`; `autoregressive.md` R3 warning).
- Auxiliary losses (CLAUDE.md §5): when they land, MaskGAE's tuned degree-loss weight is
  **α ≈ 3e-3** — not 0.1, not 1.0 (`competitors.md` §Code 1). `L_spectral` (MSE on top-k
  eigenvalues of Â) is a different object from GRAN's spectral MMD (histogram, TV kernel) —
  never conflate in a table (`autoregressive.md` C7).

### 3.8 Trainable groups + differential LRs

**DECIDED: five parameter groups, one AdamW, differential LRs; grad-clip 1.0; per-group grad
norms logged; every group asserted non-empty at construction.** (Plan, verbatim; `fpt.md` §6;
`gtlm.md` §2.11.)

| Group | Members | LR | Weight decay | Why |
|---|---|---|---|---|
| encoder | E* projections + out-LayerNorm | **1e-3…1e-4, swept per encoder** | 0.01 | random init, far to travel; E2 gets ~½ of E1's (LLaVA rule, §3.2) |
| decoder | D1 W / D3 MLP | 1e-3 (swept with encoder) | 0.01 | same regime as encoder |
| SPD table | the 320-param lookup | **5e-3** | **0** | GTLM GraphQA value; zero-init needs the high LR *more*, not less (I6); decay would fight it (B12) |
| RMSNorm affine | 16×2 per-layer + final = **67,584 params** | **1e-4** | 0 | the single most important trainable in the FPT recipe (I16: output-only → +LN takes CIFAR-10 25%→54%); RMSNorm has weight only, no bias — the fpt agent's 131K assumed GPT-2 LayerNorm |
| LoRA (Phase 3B+) | r=16, α=32, dropout 0.05, all 7 projections | **3e-5** | 0.01 | starts from refined pretrained weights, must barely move |

- The cross-source constant: *the newly-initialised cross-modal module gets an LR 30–1000×
  larger than anything touching pretrained weights* (LLaVA 50×, GTLM 167–667×, GraphToken Lion
  0.05 with a fully frozen LLM — `fpt.md` hyperparameter table across eight systems).
- **The ratio, not the values, is what transfers.** GTLM's bias competes with pretrained
  attention over *text* tokens; ours competes over encoder outputs with no pretrained meaning.
  5e-3/3e-5 is the starting point of a sweep, not a constant (`gtlm.md` OQ3).
- M2 is a standing rule: **any frozen-vs-random or with-vs-without comparison gets its own LR
  sweep, ≥3 seeds, a validation split, val-based selection** — the single-LR shortcut is what
  killed FPT's headline claim ("Each of the LRs evaluated between 1e-5 and 1e-3 results in
  different orderings", Rothermel).
- Precision: frozen base in bf16; **all trainable groups in fp32**, cast at the LLM boundary —
  a 5e-3-LR bias table updated in bf16 rounds many updates to zero (`hf-mechanics.md` §8
  bf16 notes). No GradScaler (bf16 base, nothing fp16).
- Optimizer details not locked by the plan (recorded as defaults, sweepable): AdamW betas
  (0.9, 0.95), cosine-with-min-lr schedule, warmup = total_steps//10 (GTLM's repo schedule,
  `gtlm.md` §1.3).
- LoRA staging rationale (plan Phase 3B): its ~11.3M params would contaminate Phase 2's
  "what does the frozen body contribute" question, and introduced after Phase 4 it could
  invalidate the encoder ranking. Default: encoder study without LoRA, top-2 encoders re-run
  with (Gate 3B records the decision).

---

## 4. The engineering spine

Real code against `transformers==5.15.1` (pinned in both requirements files). Every mechanism
here was verified against v5 source by at least one agent, most by two (`hf-mechanics.md`
F1–F4 and §§3–8; `gtlm.md` §3). This is the §4 that §1's forward references ("§4 v5 injection
recipe") point at. Two v4 idioms are dead: `cache_position` is not a `LlamaModel.forward`
parameter, and `_update_causal_mask` was replaced by `masking_utils.create_causal_mask`.

### 4.1 Loading the backbone (no tokenizer, no lm_head) + computing T

```python
import torch
from transformers import AutoModel, AutoConfig

MODEL_ID = "meta-llama/Llama-3.2-1B"          # gated: needs HF_TOKEN once, then HF_HUB_OFFLINE=1

config = AutoConfig.from_pretrained(MODEL_ID)
config.is_causal = False                       # F3: sticky bidirectional (belt-and-braces)

llm = AutoModel.from_pretrained(               # AutoModel -> LlamaModel: no 128256x2048 lm_head
    MODEL_ID, config=config,
    dtype=torch.bfloat16,                      # `torch_dtype=` is the deprecated v4 spelling
    attn_implementation="eager",               # the additive-bias path; flex later via Impl B
)
llm.requires_grad_(False)

# --- T for the encoder out-LayerNorm gain: BEFORE stubbing embed_tokens ---
W = llm.embed_tokens.weight.detach().float()
n = W.norm(dim=-1)
T = n[n > 1e-6].mean()                         # unpublished for the 1B; compute, never hardcode

# --- reclaim the unused embedding table: 262.7M params = 525 MB bf16 (21% of the model) ---
llm.embed_tokens = torch.nn.Embedding(1, config.hidden_size)   # 1-row stub keeps the attr alive
# (keep the attribute: get_input_embeddings() must return something or
#  gradient_checkpointing_enable() warns. save_pretrained will dislike the tied weights —
#  we checkpoint our own modules separately anyway, GTLM-style tiny checkpoints, sec 4.9.)

assert llm.rotary_emb.attention_scaling == 1.0  # F4 precondition; yarn/longrope would break it
```

`AutoModel.forward` returns `.last_hidden_state` `[B,N,2048]` **already post-final-RMSNorm** —
exactly the `H` the decoder wants.

### 4.2 The forward: `inputs_embeds` + 4-D mask injection

```python
class GraphLLM(nn.Module):
    """encoder -> frozen bias-steered Llama -> decoder.  Implementation A:
    the per-head graph bias IS the `attention_mask` argument (4-D float)."""

    def forward(self, A_obs, X=None, spd=None):
        tokens = self.encoder(A_obs, X)                        # [B,N,2048] fp32 (our modules)
        tokens = tokens.to(torch.bfloat16)                     # cast at the boundary only
        B, N, _ = tokens.shape
        dev = tokens.device

        if self.training and not tokens.requires_grad:
            tokens.requires_grad_(True)                        # grad-ckpt guard, sec 4.8

        bias = (self.bias(spd, dtype=torch.bfloat16)           # [B,32,N,N], zero-init table
                if spd is not None else
                torch.zeros(B, self.num_heads, N, N, dtype=torch.bfloat16, device=dev))

        out = self.llm(
            inputs_embeds  = tokens,                           # never together with input_ids (XOR guard)
            attention_mask = bias,                             # 4-D -> masking_utils early-exit, F2
            position_ids   = torch.zeros(B, N, dtype=torch.long, device=dev),  # F4: RoPE == identity
            use_cache      = False,                            # mandatory under grad ckpt
        )
        H = out.last_hidden_state                              # [B,N,2048]
        return self.decoder(H)                                 # [B,N,N] logits
```

Why the mask trick works, mechanically (F1+F2, both verified against v5 source):
`masking_utils._preprocess_mask_arguments` line ~811 early-exits any 4-D tensor **before**
bool-coercion, `finfo.min` clamping, or the attn-implementation check — the tensor reaches
`eager_attention_forward` untouched, where it is added: `attn_weights = attn_weights +
attention_mask`. Zero library patching. Because zeros mean "attend everywhere with no bias",
the 4-D path is bidirectional by construction and step 0 is bit-identical to the base model.

**Implementation A is behavior, not API** (top rework risk #1 in the plan). Hence:

### 4.3 The canary — runs at every training start, on both machines

```python
def mask_injection_canary(llm, d_model=2048, H=32, N=16, dev="cuda"):
    t = torch.randn(1, N, d_model, dtype=torch.bfloat16, device=dev)
    p = torch.zeros(1, N, dtype=torch.long, device=dev)
    b0 = torch.zeros(1, H, N, N, dtype=torch.bfloat16, device=dev)
    b1 = torch.randn(1, H, N, N, dtype=torch.bfloat16, device=dev) * 0.5
    h0 = llm(inputs_embeds=t, attention_mask=b0, position_ids=p, use_cache=False).last_hidden_state
    h1 = llm(inputs_embeds=t, attention_mask=b1, position_ids=p, use_cache=False).last_hidden_state
    assert not torch.allclose(h0, h1), "4-D MASK IGNORED - the v5 early-exit is gone; switch to Impl B"
    hN = llm(inputs_embeds=t, attention_mask=None, position_ids=p, use_cache=False).last_hidden_state
    assert torch.allclose(h0, hN, atol=1e-2), "zero-bias != no-mask: is_causal/config drift"
```

The first assert detects a transformers upgrade removing the early exit (the failure is loud,
which is the whole argument for A over B — B's failure mode is silent mask loss). The second
is the zero-bias==no-mask parity from the Phase-2 test list; it requires `config.is_causal =
False` or the None-mask run is causal and differs.

### 4.4 GQA shape note — H = 32 query heads, never 8 KV heads

`repeat_kv` runs **before** the score matmul in `eager_attention_forward`, so by the time the
mask is added the KV heads are already broadcast to 32:

```
bias shape = [B, 32, N, N]      # num_attention_heads
         NOT [B, 8, N, N]       # num_key_value_heads -> silent mis-broadcast or shape error
```

Consequences: GQA saves **no** bias memory or parameters — budget `L·H`, not `L·H_kv`. A
`[B,1,N,N]` tensor broadcasts fine (the head-shared ablation). If tying within KV groups as a
regularizer, use `bias_kv.repeat_interleave(G, dim=1)` — `repeat()` tiles instead of
interleaving and silently misaligns every head (`hf-mechanics.md` GQA section).

### 4.5 position_ids and the scaling assert

```python
position_ids = torch.zeros(B, N, dtype=torch.long, device=dev)   # [B,N], never [N]
```

Omit it and Llama auto-builds `arange(N).unsqueeze(0)` — the causal-LM default, silently
re-enabling RoPE and breaking body-level equivariance. The value must be 0 exactly (at p≠0 the
common rotation cancels in q·k but rotates q,k themselves). The exactness is conditional on
`attention_scaling == 1.0` — asserted at startup (§4.1); `llama3` and `default` rope types
return 1.0, yarn/longrope do not (F4).

### 4.6 is_causal

Three routes, all official, in the order we use them: (1) the 4-D mask itself (implicit —
`create_causal_mask` is never consulted); (2) `config.is_causal = False` at load (sticky; our
default); (3) `is_causal=False` as a per-forward kwarg (documented in `TransformersKwargs`;
used by the no-bias ablation, and `is_causal=True`/causal-mask composition by the Phase-5 AR
mode via the `attn_mode` config field). Note `merge_with_config_defaults` mutates the shared
config during forward and restores in a `finally` — thread-unsafe by construction, and a
possible `torch.compile` guard-failure source; prefer the sticky route if we ever compile
(`hf-mechanics.md` OQ6). Do not confuse the kwarg with the `LlamaAttention.is_causal`
attribute, which eager never reads — eager behaviour is 100% the mask tensor.

### 4.7 LoRA config (Phase 3B)

```python
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=16, lora_alpha=32,                        # plan-locked (GTLM GraphQA Table 7; alpha=2r -> scaling 2.0)
    lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type=None,                             # NOT "CAUSAL_LM": no lm_head, no labels ->
)                                               # plain PeftModel forwards **kwargs untouched
llm = get_peft_model(llm, lora_cfg)
```

- peft's default `target_modules` for llama is `["q_proj","v_proj"]` only — too narrow for a
  new modality; all seven, explicitly (GTLM repo `src/train/model.py` uses all seven too).
- peft defaults are r=8/α=8 (scaling 1.0) — pass both or the scaling is wrong.
- `TaskType.CAUSAL_LM` wraps in `PeftModelForCausalLM`, which expects `labels` and an
  `lm_head` we deleted. `task_type=None` gives the plain wrapper.
- Under Implementation A the bias module lives outside the wrapped LLM, so peft never touches
  it. Under Impl B, attach bias modules to the raw layers **before** `get_peft_model`, reach
  them via `.base_model.model.layers` after (`hf-mechanics.md` §6).
- Frozen-base assert (Phase-2 test list): every parameter's `requires_grad` must equal
  "belongs to encoder/decoder/bias/out-norm/RMSNorm-affine/lora_" — catches both an
  accidentally-unfrozen body and an accidentally-frozen head.

### 4.8 Param groups

```python
def param_groups(model, lr_enc, lr_dec, lr_bias=5e-3, lr_ln=1e-4, lr_lora=3e-5, wd=0.01):
    g = {"enc": [], "dec": [], "bias": [], "ln": [], "lora": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:      continue
        elif "encoder" in n:         g["enc"].append(p)
        elif "decoder" in n:         g["dec"].append(p)
        elif "bias_table" in n:      g["bias"].append(p)      # our SPD module's name
        elif "lora_" in n:           g["lora"].append(p)
        elif "layernorm" in n or n.endswith("norm.weight"):   # RMSNorm affine (weight only)
                                     g["ln"].append(p)
        else: raise ValueError(f"unclassified trainable param: {n}")
    for k, v in g.items():
        assert v or k in ("bias", "lora"), f"empty param group: {k}"   # bias/lora phase-gated
    return [
        {"params": g["enc"],  "lr": lr_enc,  "weight_decay": wd},
        {"params": g["dec"],  "lr": lr_dec,  "weight_decay": wd},
        {"params": g["bias"], "lr": lr_bias, "weight_decay": 0.0},    # never decay the table
        {"params": g["ln"],   "lr": lr_ln,   "weight_decay": 0.0},
        {"params": g["lora"], "lr": lr_lora, "weight_decay": wd},
    ]

opt = torch.optim.AdamW(param_groups(model, 1e-3, 1e-3), betas=(0.9, 0.95))
```

The `raise` on unclassified params is deliberate: a renamed module otherwise lands in a
default group at the wrong LR and "the bias didn't help". Log `||grad||` per group every N
steps; `||g_bias|| ~ 1e-6` means the 5e-3 is doing nothing (`hf-mechanics.md` §6). HF
Trainer's name-based decay rule (excludes anything containing "bias") is not in play — we own
the optimizer — but is recorded in case Trainer ever enters (`gtlm.md` §2.11).

### 4.9 Gradient-checkpointing gotcha (M9 — the most dangerous silent failure)

```python
llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

- `enable_input_require_grads()` hooks `get_input_embeddings()` = `embed_tokens`, **which we
  never call** (we pass `inputs_embeds`). It is a complete no-op here, and
  `gradient_checkpointing_enable`'s automatic invocation of it is a no-op for the same reason
  (`modeling_utils.py` 2216–2246 / 3226–3234, **[2x]**).
- What saves us is *incidental*: the trainable encoder makes `inputs_embeds` require grad. It
  vanishes under a frozen-encoder ablation, a stray `.detach()`, a `torch.no_grad()` wrapper,
  or cached embeddings — then, with reentrant checkpointing, LoRA params get `grad=None` with
  **no error** and loss goes flat.
- Defence, all four (Phase-2 tests): (1) `use_reentrant=False` explicitly (v5 default, pinned
  anyway); (2) the `tokens.requires_grad_(True)` guard in `forward` (§4.2); (3)
  `use_cache=False`; (4) the grad-existence assert after the first backward:

```python
loss.backward()
missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
assert not missing, f"no gradient reached: {missing[:5]}"
```

Checkpointing removes the ×16-layers factor, not the within-layer `[B,32,N,N]` peak. If we
ever adopt per-batch context objects (Impl B), **do not clear them at the end of forward** —
the backward recompute reads them (`gtlm.md` §3.4 verbatim warning).

### 4.10 Tiny checkpoints + the strict-load guard

Trainables are ~10–21M (§7); never save the 2.5 GB frozen body. Save
`{n: p for n, p in model.named_parameters() if p.requires_grad}` plus the config and split
seed. On load, port GTLM's guard: `load_state_dict(strict=False)` **silently loads nothing**
when names drift — raise if `result.unexpected_keys` is non-empty (`gtlm.md` §3.6). This is
also what makes laptop re-verification of Marlowe checkpoints cheap (a checkpoint is ~40–85 MB
fp32).

### 4.11 Implementation B — the documented fallback (same signature, kept in `g2l/llm.py`)

Triggered by: the canary failing after a forced upgrade, per-layer bias tables (Phase 6
ablation), or FlexAttention (Phase 6 scale). Registered attention function + **mandatory**
mask-interface registration:

```python
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward

def graph_eager(module, query, key, value, attention_mask=None, *, scaling, dropout=0.0, **kw):
    bias = module.graph_bias(module._graph_ctx["spd"], dtype=query.dtype)   # per-layer module
    mask = bias if attention_mask is None else attention_mask + bias
    return eager_attention_forward(module, query, key, value, mask, scaling=scaling, dropout=dropout)

AttentionInterface.register("graph_eager", graph_eager)
AttentionMaskInterface.register("graph_eager", eager_mask)   # forget this -> attention_mask=None,
                                                             # every constraint silently dropped
```

Flex variant = GTLM's `score_mod` gather, trivial for us because token index == node index:

```python
def make_score_mod(node_bias):                 # node_bias: [B,H,N,N], NON-LEAF (sec 3.4)
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + node_bias[b, h, q_idx, kv_idx]
    return score_mod
```

Never override `LlamaAttention.forward` (GTLM's own v0 was 61 KB; the registered-function
rewrite is 5.2 KB — `gtlm.md` §3.2 "Strategy B", the single best architectural lesson in
that repo).

---

## 5. Data and evaluation protocol

The measuring stick, fixed for the whole project at Phase 0 (gate signed criteria in
`results/phase0/RESULTS.md`). Everything here is implemented in `g2l/data.py` and
`g2l/metrics.py` and enforced by `tests/`.

### 5.1 Cora, corrected facts (the numbers actually in the tensors)

| Quantity | Value | Source |
|---|---|---|
| nodes | 2,708 | Planetoid **[2x]** |
| `edge_index` | `[2, 10556]` directed entries | F1 correction; verified live in `tests/test_cora.py` |
| unique undirected edges | **5,278** (CLAUDE.md's 5,429 is the pre-dedup Sen et al. count; 302 directed entries dropped by `coalesce`) | **[2x]** three independent computations, CONFLICT #9 |
| features / classes | 1,433 bag-of-words / 7 | Planetoid |
| density | **0.14395%** (10,556/7,333,264); upper-tri positive rate 0.14400% | [COMPUTED, 2x] (F2) |
| `pos_weight` | **693.70** full matrix / **693.44** off-diagonal — computed at runtime from the training split, never hardcoded (breaks on PubMed/molecular otherwise) | [COMPUTED, 2x] |
| degrees | max **168** (node 1358), mean 3.898, p50 3, p90 7, p99 19, p99.9 65; no isolated nodes | computed from raw `ind.cora.graph` (`depthwidth.md` C7) |
| Kipf's loss companion | `norm = 0.50072` outer rescale; copying `pos_weight` without it ≈ 2× Kipf's effective LR — we log it, we do not silently copy | `tkipf/gae/train.py` verbatim |

`load_cora()` applies `NormalizeFeatures` and reads `PYG_DATA_ROOT` (laptop `data/pyg`,
Marlowe `/projects/m000211/data/pyg`) — never a hardcoded path.

### 5.2 The split: RandomLinkSplit, exact parameters and exact shapes

```python
RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True,
                split_labels=True, add_negative_train_samples=False)
```

Applied **once** per seed in `edge_split()` — never as a dataset transform (transforms rerun
on every `dataset[0]` access and hand out a fresh split each time; the loader docstring says
so and `pyg-baselines.md` lists it as the #1 gotcha). `torch.manual_seed(seed)` immediately
before; the seed varies split and init together (`baselines/common.py`).

Resulting shapes (ratios are of the **5,278** row≤col edges, not 10,556 — verified against
`RandomLinkSplit.forward` source and asserted in `tests/test_cora.py::test_split_sizes`):

```
train.edge_index            [2, 8976]   train edges, both directions   (message passing, train)
train.pos_edge_label_index  [2, 4488]   supervision positives; NO neg_* attrs (absent, not empty)
val.edge_index              [2, 8976]   = TRAIN edges only             (message passing, val)
val.pos/neg_edge_label_index[2, 263] each
test.edge_index             [2, 9502]   = train+val edges              (message passing, test)
test.pos/neg_edge_label_index[2, 527] each      <- the canonical 527/527 balanced test set
```

Negatives are sampled once against the **full** original edge set (a sampled negative is a
true non-edge in every split); no held-out edge is ever visible in the graph encoded from —
`tests/test_cora.py::test_supervision_disjoint_from_message_passing` asserts it including
reversed directions.

### 5.3 The three-column protocol, exactly as `g2l/metrics.py` implements it

One scoring path. Feed **logits**, never probabilities (float32 sigmoid tail-collapse creates
ties that change AP — I2/module docstring); sklearn `average_precision_score` only, never
trapezoidal PR-AUC (Davis & Goadrich: linear interpolation overstated 0.031 as 0.50 in exactly
our imbalance regime).

**`evaluate(logits, target, mask, seed, n_draws=5)`** — metrics on `mask==True` cells only:
- `auroc` — balance-invariant bridge (column 1);
- `ap_balanced` (± std) — all positives + an **equal seeded draw** of negatives from the same
  masked set, averaged over `n_draws=5` (column 2, the published-comparable balance);
- `ap_sparse`, `base_rate`, `lift = ap_sparse/base_rate` — every masked cell at true
  prevalence (column 3); plus `n_pos`, `n_scored`;
- raises on a single-class mask (no silent NaN).

**`evaluate_pairs(pos_logits, neg_logits)`** — the GAE-comparable 527+527 view; used for
val-based model selection in the baselines (selection on `(auroc+ap)/2`).

**`evaluate_edge_split(logits_full, split, seed)`** — the one table-row function. Symmetrizes
`L=(logits+logitsᵀ)/2` so direction conventions cannot matter, then reports:
- `auc`, `ap` — on `test.pos_edge_label_index` vs `test.neg_edge_label_index` (the split's own
  sampled negatives): **the column comparable to published GAE 91.0/92.0, MaskGAE 96.45/95.95**;
- `auroc_sparse`, `ap_sparse`, `lift`, `base_rate` — via `sparse_eval_mask`: scorable cells =
  strict upper triangle minus train/val edge cells = **3,660,527 cells, 527 positives, base
  rate 1.4397e-4** (matches every row of `results/phase1/laptop_reference_table.jsonl`), with
  an internal assert that no test edge collides with a train/val cell.

Protocol fine print vs the plan: the plan's "seeded negative draws averaged ≥5" is implemented
in `evaluate()`'s balanced column; `evaluate_edge_split`'s `ap` column instead uses the
split's own single 527-negative draw — the draw varies with the split seed, and every table
row runs ≥3–5 seeds, so the averaging happens across seeds. Both honest; the caption states
which. (Recorded as a deliberate deviation, not an accident.)

### 5.4 The two masking views (both required; different jobs)

1. **Edge-split view** (above) — the evaluation protocol and the baseline-comparable one.
   Baseline training uses GAE-style `recon_bce` (positives + freshly sampled negatives per
   step, `baselines/common.py`).
2. **Matrix-fraction view** — `mask_matrix(A, frac=0.15, seed)`: picks 15% of the **strict
   upper triangle**, mirrors the hiding into both directions of the input
   (`A_obs = A·(1−hidden)`, `hidden = sup ∨ supᵀ`), scores each pair **once** (sup_mask is
   upper-tri only), diagonal excluded. On Cora: 549,791 masked cells, ~792 true edges. This is
   CLAUDE.md §5.1's task as the LLM model's training objective — per step, hide a fresh 15% of
   the *train* graph's cells and take pos-weighted BCE on exactly those cells; at test time
   feed the train+val graph and score under the edge-split protocol. The model is **never told
   which cells are queried** (no mask channel into the encoder) — consistent with test time,
   where the query set is "every unobserved pair".
   The 15% rate is a CLAUDE.md carry-over, not evidence-locked: MaskGAE's tuned rate is
   **0.7** — masking-rate is a Phase-3+ sweep axis, and MaskGAE must be re-run at our rate for
   an apples-to-apples floor before any claim (`competitors.md` OQ4).

Guarantees under test (`tests/test_masking.py`, `test_batch.py`): sup_mask strictly
upper-triangular; both directions hidden; `A_obs` stays symmetric; diagonal never masked even
at frac=1.0; seed-reproducible; padded cells in `collate()`d batches never enter loss or
metrics; single-graph round-trip exact. The `[B,N,N]`+node-mask+padding API exists from
day one even though Cora is B=1 (the Phase-4 molecular hedge — plan rework risk #3).

### 5.5 The comparability trap, stated plainly

**AP@1:1 and AP@true-prevalence are the same metric on the same scorer differing by ~50×.**
Measured (I2, [COMPUTED]): a scorer pinned at AUROC≈0.91 gets AP 0.9103 at 1:1, 0.6446 at
10:1, 0.2562 at 100:1, **0.0174** at the true 6952:1 — while AUROC never leaves 0.90–0.92.
Every published AP (Kipf, MaskGAE, ARGA, SEAL) is balanced. Therefore:

- Never put numbers from different denominators in one column. Every table carries all three
  columns and the caption names the protocol (P1/P2/P3/P4 labels, §6).
- AUROC is the *bridge*, not the enemy — CLAUDE.md §11's ban is superseded (I2).
- The sparse column has **no published reference anywhere** — we generate it (our GAE at
  ap_sparse ≈ 0.006–0.008, lift 42–54, is the first such number; §6.2). Novel but uncheckable;
  the balanced column is where external anchoring lives (Gate 1 = reproduce 90.6/91.2).
- Identity + random rows are recomputed on **every** table (leak canary): identity must sit at
  AUROC 0.500/lift 1.0 to 1e-9 (`tests/test_leaks.py`); random at base rate.
- Cross-protocol accuracy numbers (LLaGA 86.82, GraphPrompter 90.10) are yes/no-question
  accuracies on different preprocessing — never in the same table as AUC (P3/P4, §6).
- AR phase inherits its own version: never compare MMDs across kernels (GraphRNN's grid Deg
  1e-5 vs GRAN's re-eval 1.12e-2, same model+data) and Cora-the-benchmark ≠ Cora-the-
  subgraph-generation-set (`autoregressive.md` comparability rules).

---

## 6. Numbers to beat

### 6.1 Cora link prediction — published, with protocol labels

Protocols (never mix columns across them):
```
P1  Kipf & Welling: 85/5/10 edge split, test = held-out positives + EQUAL sampled non-edges;
    roc_auc_score + average_precision_score. (What our evaluate_edge_split `auc`/`ap` reproduce.)
P2  MaskGAE: same 85/5/10 balanced-negatives AUC/AP protocol (their AP IS sklearn average
    precision — the "AUPRC" CLAUDE.md demands, on a balanced set).
P3  GraphPrompter: yes/no question to an LLM, ACCURACY on a balanced set. Not AUC.
P4  LLaGA: yes/no ACCURACY, different Cora preprocessing again. Not AUC.
```

| Method | AUC | AP | Protocol | Source |
|---|---|---|---|---|
| Spectral Clustering | 84.6 | 88.5 | P1 | Kipf & Welling 1611.07308 Table 1 (fetched verbatim) |
| DeepWalk | 83.1 | 85.0 | P1 | same |
| GAE* / VGAE* (no features) | 84.3 / 84.0 | 88.1 / 87.7 | P1 | same — the features alone are worth ~4 AP points |
| **GAE (published)** | **91.0 ± 0.02** | **92.0 ± 0.03** | P1 | same |
| **VGAE (published)** | 91.4 ± 0.01 | 92.6 ± 0.01 | P1 | same — but the PyG reproduction reverses the GAE/VGAE order inside 1σ: **never claim VGAE > GAE** (pyg C9) |
| **GAE (PyG's own code, 30 runs)** | **90.6 ± 0.9** | **91.2 ± 1.0** | P1 | arXiv 2107.02658 Table 2 — **the number you can actually reproduce**, 600 epochs, final state, no selection; Gate 1's target |
| VGAE (PyG reproduction) | 89.8 ± 0.9 | 90.3 ± 1.0 | P1 | same |
| GAE / VGAE (MaskGAE re-run) | 91.09 / 91.40 | 92.83 / 92.60 | P2 | MaskGAE 2205.10053 Table 3 **[2x]** |
| ARGA | 92.40 | 93.23 | P2 | same |
| SEAL | 92.22 | 93.12 | P2 | same |
| **MaskGAE (edge mask, p=0.7)** | **96.42** | **95.91** | P2 | same |
| **MaskGAE (path mask)** | **96.45** | **95.95** | P2 | same — **the real bar** (I3). Caveat before claiming: they mask 70% of edges; re-run MaskGAE at our rate first (`competitors.md` OQ4) |
| GAT / GraphPrompter+LoRA | 90.71 / 90.10 | — (accuracy) | P3 | 2402.10359 — the frozen-LLM system does not beat its own GAT baseline |
| LLaGA-HO-7B / GCN | 86.82 / 81.59 | — (accuracy) | P4 | 2402.08170 Table 1 |
| Random | 50.0 | 50.0 (balanced) | any | analytic |

**Reading order: GAE ~91–92 AP is the floor. MaskGAE ~96 is the bar. Reaching 92 is not a
result** (I3). PubMed for later phases: GAE 96.4/96.5 vs the best LLM-soft-prompt number ever
published, GraphGPT 82.46/80.26 — the ~14-point gap that motivates the whole graph-out side.

### 6.2 Heuristics + our own measured reference table (laptop, pre-gate)

Published anchor for heuristics: neighbourhood-overlap heuristics reach ~0.80–0.85 AUC on
citation graphs and beat GNNs on several datasets (Neo-GNN 2206.04216; M5,
[SINGLE-SOURCE]). Our own measured rows — `results/phase1/laptop_reference_table.jsonl`,
5 seeds each, all through the one `evaluate_edge_split()` on the one split; **status: measured
on the laptop, Gate 1 not yet signed, Marlowe reproduction pending** — do not cite outside the
repo yet:

| model | n | AUC | AP@1:1 | AUROC(sparse) | AP(sparse) | lift |
|---|---|---|---|---|---|---|
| common_neighbors | 5 | 0.735±0.008 | 0.732±0.009 | 0.736±0.007 | 0.0124 | 86 |
| adamic_adar | 5 | 0.738±0.008 | 0.739±0.008 | 0.737±0.007 | 0.0235 | 163 |
| resource_allocation | 5 | 0.737±0.007 | 0.737±0.008 | 0.737±0.007 | 0.0227 | 157 |
| ppr | 5 | 0.849±0.011 | 0.903±0.006 | 0.850±0.012 | 0.0309 | 214 |
| feature_only_logreg | 5 | 0.618±0.009 | 0.643±0.021 | 0.618±0.008 | 0.0003 | 2 |
| gae (early stop on val) | 5 | 0.907±0.006 | 0.911±0.011 | 0.906±0.007 | 0.0060 | 42 |
| gae_600ep (2107.02658 protocol) | 5 | 0.905±0.005 | 0.911±0.004 | 0.907±0.007 | 0.0078 | 54 |
| vgae | 5 | 0.896±0.008 | 0.900±0.006 | 0.896±0.009 | 0.0051 | 35 |
| gat (LP config, dropout 0.2) | 5 | 0.908±0.012 | 0.918±0.006 | 0.907±0.010 | 0.0071 | 49 |

What the table already shows (and Gate 1 will certify): the GAE reproduction lands on
2107.02658's 90.6±0.9 / 91.2±1.0 — the external anchor of the whole protocol. **PPR's
lift (214) is 4–5× GAE's (42–54)** at 6 points lower AUC: the balanced and sparse columns
genuinely rank models differently, which is the comparability trap made concrete. The
GAT-for-link-prediction footnote is recorded in `baselines/gae.py`: node-classification
dropout 0.6 collapsed AP to 0.72; the LP config uses dropout 0.2 (the widely-quoted GAT
hyperparameters were for node classification and are [UNVERIFIED] for LP — pyg gotcha).

### 6.3 OGB molecular (Phase-4 parallel track) — use the follow-up numbers, not the straw man

| Model / tokenization | molhiv | molbbbp | molbace | Source |
|---|---|---|---|---|
| AdjRows, deliberately plain transformer (width grid {32,64}) | 61.87 ± 1.10 | 67.63 ± 2.57 | 68.64 ± 2.34 | 2503.01805 Table 1 — **a straw man for absolute claims** (T15); cite only for the tokenization *ordering* (AdjRows > EdgeList on 3/3) |
| **Adj/Pad, properly tuned GT** | **71.31 ± 1.26** | **67.90 ± 2.35** | **74.88 ± 2.21** | 2605.22471 Table 1 — **the real bar** |
| GIN (same harness) | 73.20 | 66.46 | 72.73 | same — adjacency tokenization **loses to GIN on molhiv** (Critic 1's insisted caveat) |
| DeepSet, ignores edges entirely | 74.82 | 61.37 | 75.22 | same — **beats both GIN and adjacency on 2/3.** The feature-only baseline is already vindicated in the literature; run it first, every dataset |
| Comb (row ‖ Lap ‖ RW), best HIV | 76.44 ± 1.70 (Trunc) | 68.99 | 74.78 | same — direct empirical support for E5/E5b concatenation |

Also on the record for Phase 4: adjacency tokenization dominates on the two tasks "governed
by explicit local constraints" (MaxClique F1 25.20 vs ≤10.8; TopoOrd MAE 0.196) — the
follow-up's own words, and the strongest published justification of our input format for a
task that *is* exact edge constraints (depthwidth.md, "Lost in Tokenization" §Table 1
reading). GTLM's GraphQA Edge Existence 99.8 vs GraphToken 73.8 is *not* comparable to any of
this (text question about a fully visible graph; `gtlm.md` §4.1 caveat).

### 6.4 Cross-references

The backbone table lives in §3.1 (as promised in §1). AR-phase numbers-to-beat (GRAN/G2PT
MMD + V.U.N., kernels and all) live in `autoregressive.md` §"Numbers to beat" and are pulled
into the Phase-5/6 RESULTS.md when those phases open — with the mandatory train-vs-test MMD
scale row (O'Bray et al. 2106.01098) and the kernel-comparability rules of §5.5.

---

## 7. Compute reality

Supersedes M10 and §0.2 item 1: the laptop-training premise was withdrawn 2026-08-25
(user decision, recorded in `environment.md` — "anything beyond seconds-long tests runs on
Marlowe"). The grid that was "arithmetically impossible" on 12 GB is a rounding error on the
Marlowe allocation.

### 7.1 The two machines

| | Laptop (dev only) | Marlowe (all training) |
|---|---|---|
| GPU | RTX 5070 Ti Laptop, 12 GB, sm_120, 80 W cap | H100 80 GB SXM5 ×248 (8/node, 31 DGX nodes) |
| Software | Win 11, py 3.14.7, torch 2.13.0+cu130, transformers 5.15.1, PyG 2.8.0.post1 (pinned, `requirements-local.txt`) | venv on `/projects/m000211/envs/g2l`; transformers==5.15.1 pinned in `requirements-cluster` path of `scripts/marlowe_setup.sh`; no python/pytorch modules exist — pip wheels |
| Role | pytest, gate walkthroughs, N≤512 smoke, **full-graph no_grad inference** of downloaded checkpoints (fits 12 GB) | every sweep and every headline number |
| Slurm | — | account `marlowe-m000211-pm06`, partition `batch` (Medium: ≤16 nodes, **2-day walltime**, non-preemptible), no debug partition → bring-up is local, Marlowe enters at the first sweep |
| Budget | — | ~10,000 GPU-h / 12-week cycle; $0.30/GPU-h from 2026-09-01; machine at ~90% allocation → **queue wait, not budget, is the binding constraint** (`marlowe.md` §2, §3.6) |

### 7.2 H100 memory table (from `environment.md`, cross-checked against `pyg-baselines.md`)

Eager attention materialises `[B, 32, N, N]` per layer; worst case ≈ `12·B·H·N²` bytes/layer
(two bf16 score copies + fp32 softmax output + bf16 cast + bias kept for backward).

| N | per layer | all 16 layers | verdict |
|---|---|---|---|
| 512 | 0.1 GB | 1.6 GB | trivial (laptop smoke ceiling) |
| 1024 | 0.4 GB | 6.4 GB | easy |
| **2708 (full Cora)** | **2.8 GB** | **45 GB** | + ~2.0 GB frozen weights (embed stubbed) + 0.47 GB bias (+grad ~0.9 GB) + misc ≈ **~41 GB total, fits 80 GB at ~51% with no checkpointing**; with gradient checkpointing ≈ **8 GB** — the standard configuration |
| PubMed 19,717 | 149 GB | — | FlexAttention only (Phase 6) |
| ogbn-arxiv 169,343 | — | — | flex + subgraph sampling (identity canary re-run under subgraph batching — plan Phase 6) |

Full-graph Cora **training** is Marlowe-only; full-graph **inference** (no_grad, bf16) fits
the laptop — which is what keeps gates independent of cluster logs.

### 7.3 Trainable-parameter budget, Llama-3.2-1B + Cora ([COMPUTED], arithmetic shown)

```
E1 encoder      Linear(2708 -> 2048) + bias      2708*2048 + 2048        =  5,548,032
   out-LayerNorm 2*2048                                                  =      4,096
D1 decoder      W in R^{2048x2048}               2048^2                  =  4,194,304
RMSNorm affine  16 layers * 2 sites * 2048 + final 2048                  =     67,584
SPD bias table  (8 distances + >8 + unreachable) * 32 heads = 10*32     =        320
                                                                 -----------------
Phase 2 (no bias) .............................................  9,814,016
Phase 3 (+ table) .............................................  9,814,336
LoRA r=16, alpha=32, all 7 projections, per layer:
   q 16*(2048+2048)=65,536   k 16*(2048+512)=40,960   v 40,960   o 65,536
   gate 16*(2048+8192)=163,840   up 163,840   down 163,840
   per layer 704,512  * 16 layers ................................ 11,272,192  (~11.3M)
Phase 3B total ................................................. 21,086,528  (~21.1M)

Variants: E4 = (2708+1433)*2048 + 2048 = 8,482,816 (+2.9M over E1)
          D3 (hidden 512, our choice)  = 6144*512+512 + 512+1 ≈ 3,146,753
Frozen body: 1,235,814,400 params (973.1M non-embedding after stubbing embed_tokens).
Trainables are 0.8% (Phase 2) to 1.7% (3B) of the base -> tiny checkpoints (sec 4.10):
~39 MB fp32 at Phase 2, ~84 MB at 3B.
```

⚠ Inconsistency with the plan, flagged: the plan's Phase-2 line "identity body (w/o LLM —
same **~12.7M** trainables)" carries the E4-based estimate from `fpt.md` (8.48M + 4.19M).
With the E1 configuration Phase 2 actually trains **~9.8M**. The *point* of the row —
matched trainables between arms (1) and (2) — is unaffected; the matched number is whatever
the arm's encoder makes it. Record the actual count per run (CLAUDE.md §12's logging rule).

Context for M3's warning: FPT trained 106K params; we train ~10–21M — the "small module did
all the work" risk is ~100× larger for us, which is why the w/o-LLM arm is a Phase-2 row and
not an afterthought.

### 7.4 GPU-hours for the Phase 2/3/4 grids

[COMPUTED] from stated assumptions; the per-step time is [UNVERIFIED] until Phase-2 bring-up
measures it.

Assumption to be measured on day one of Phase 2: full-graph Cora eager step (fwd+bwd, grad
ckpt, B=1) ≈ **1–3 s** on H100. Basis: GTLM's measured 836.9 ms eager step at L≈663 with
three bias-family MLPs (78% of their overhead is bias-module compute we don't have); our N²
is 16.7× theirs but our bias is a table lookup; eager attention is bandwidth-bound (~0.5 TB
of traffic per step at N=2708 → ~0.3–0.5 s) plus ~0.1 s of MLP FLOPs. Full-batch training =
1 step/epoch; assume 500–2,000 epochs per run → **15–100 min/run; plan at 2 h/run** with
early stopping on val AUC+AP.

| Phase | Runs (plan) | Arithmetic | GPU-h @2 h/run |
|---|---|---|---|
| 2 — 4-way comparison | ~16–24 (arms 1–3 × LR sweep × 3 seeds; arm 4 is the local Phase-1 curve) | 24 × 2 h | ~48 |
| 3 — bias | probe + learned table + shuffled-A + inertness (+flex parity) ≈ 4–5 configs × 3 LRs × 3 seeds ≈ 36–45 | 45 × 2 h | ~90 |
| 3B — LoRA | 3 LRs × 3 seeds ≈ 9–12, ~1.3× slower steps | 12 × 2.5 h | ~30 |
| 4 — encoder study | 7 encoders × 3 LRs × 3 seeds = 63, + top-2 with LoRA = 18 | 81 × 2 h | ~162 |
| **Total 2→4** | ~160 runs | | **~330 GPU-h ≈ 3.3% of one cycle ≈ $100** |

Sensitivity: if the measured step is 4× the assumption (8 h/run), the total is ~1,300 GPU-h =
13% of the cycle — still not binding. What *is* binding: queue wait on a 90%-allocated
machine and our own validation throughput (one gate per phase). Consequence already in the
plan: many small 1-GPU jobs, never multi-GPU reservations.

### 7.5 Slurm array strategy (the shape of every sweep)

From `slurm/phase1_scratch_sweep.sbatch` (the committed, working template) + `marlowe.md` §4:

```
#SBATCH --account=marlowe-m000211-pm06  --partition=batch
#SBATCH -N1 -n1 --gpus=1 --cpus-per-task=16 --mem=192G      # 1/8 node fair share
#SBATCH --time=04:00:00                                     # per-run bound, << 2-day cap
#SBATCH --array=0-80%6                                      # e.g. Phase 4: 81 tasks, 6 concurrent
#SBATCH --output=/scratch/m000211/%u/logs/%x-%A_%a.out
module purge && module load slurm mps                       # mps: stale /tmp/nvidia-mps insurance
export HF_HOME=$SCR/hf TORCH_HOME=$SCR/torch XDG_CACHE_HOME=$SCR/xdgcache
export TRITON_CACHE_DIR=$SCR/triton TORCHINDUCTOR_CACHE_DIR=$SCR/inductor   # flex, Phase 6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYG_DATA_ROOT=/projects/m000211/data/pyg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs/sweep_list.txt)
srun python -u -m g2l.train --config "$CFG"
```

Rules of the road (all doc-sourced in `marlowe.md`; [UNKNOWN]s resolved by
`scripts/marlowe_verify.sh` at first login):
- **One GPU per job, many jobs.** Arrays schedule far better on a 90%-utilised machine, and
  Slurm charges for allocation not use. The `%6` throttle is courtesy on a 16-node shared
  partition; `MaxArraySize` is [UNKNOWN] until verified.
- **Task → (config, seed) via a sweep list file**; results appended to per-phase JSONL with
  the `(model, seed)`-idempotency skip already used by `baselines/run_phase1.py` — a re-queued
  array task never duplicates a row and a partially-failed array is resubmitted as-is.
- **Checkpoint/resume**: save trainables every K steps to `$SCR/runs/$SLURM_JOB_ID`; runs are
  ≤ hours, the 2-day `batch` cap is never in play, but the resume path exists for requeues
  and for `preempt`-partition experiments (bare account `-A marlowe-m000211` for uncharged
  preempt cycles is documented-but-[UNKNOWN]; never assumed).
- **Everything offline in the job**: models + Cora pre-staged on a login node
  (`scripts/marlowe_setup.sh`), `HF_HUB_OFFLINE=1` — a network blip cannot kill a queued
  experiment and 20 array tasks don't hammer huggingface.co (login nodes are the documented
  place for downloads; compute-node internet is [UNKNOWN]).
- **Storage discipline**: `$HOME` is 32 GB — nothing lands there. Code+venv+datasets on
  `/projects` (backed up, tiny-file-friendly NFS); caches/logs/runs on `/scratch` Lustre
  (10 TB, **not backed up, purge policy unpublished** — anything worth keeping is copied to
  `/projects/m000211/results/` at job end; §8 R14).
- **Policy tripwires** (Usage Violations page, verbatim-sourced): no training or compile
  warm-ups on login nodes; no tunnels (SSHX/Tailscale/`code tunnel` are named as prohibited;
  Claude remote-control likewise); ControlMaster multiplexing from WSL is the sanctioned
  Duo-tamer; one long `tmux` per login node as fallback.
- **Provenance per job**: job id, node, `git rev-parse --short HEAD`, `nvidia-smi` name/VRAM
  — already in the committed template; every RESULTS.md row must be traceable to a commit.
- Sherlock is the backup cluster; scripts do **not** port unchanged (no `--account`,
  heterogeneous GPU fleet needs `-C` constraints, `$SCRATCH` has a **90-day purge**, 15 GB
  `$HOME`, `sh_dev` exists — full 24-row diff table in `sherlock.md` §9).

---

## 8. Risks, ranked

Ranked by (probability × cost of late discovery). Each row: the cheap falsifying experiment
and the go/no-go criterion. The plan's top-3 rework risks are R1–R3 verbatim; the rest are
this document's additions from the notes.

| # | Risk | Cheap falsifier | Go / no-go |
|---|---|---|---|
| **R1** | **Implementation A is behavior, not API** — the 4-D early-exit is an inline comment, not a documented contract; a transformers bump silently reroutes the mask. (Plan risk 1; `hf-mechanics.md` OQ1) | The §4.3 canary: 4 forwards, <10 s, at every training start on both machines. | Canary fails ⇒ stop the line, switch to the same-signature Impl B fallback in `g2l/llm.py`; transformers stays pinned at 5.15.1 either way. |
| **R2** | **Protocol/leak error in Phase 0 poisons everything downstream** — a leak looks exactly like a great result. (Plan risk 2; M4's neighbourhood) | Identity row recomputed on every table (must be 0.500/1.0 to 1e-9); disjointness test; flip-held-out-cells bias-equality test; Gate-0 hand-reproduction of the three columns on a 6-node path graph (`g2l/walkthrough.py` phase0). | Any canary off by >1e-9 ⇒ no experiment runs until root-caused. This gate already passed (Phase 0 RESULTS.md). |
| **R3** | **Fixed-N / single-graph assumptions calcify** and Phase-4 molecular forces a rewrite. (Plan risk 3) | `tests/test_batch.py` multi-graph padding round-trip (already green); SPD computed per-sample; E3/E6/E7 exist precisely to break the N-tie. | Any new module that cannot take `[B,N,N]`+node-mask is rejected at review; headline Cora numbers always from full-graph runs. |
| **R4** | **The LLM contributes nothing** — a ~10M-param encoder+decoder may solve Cora alone (Tan et al. killed a subfield this way; M3. E1+D1 ≈ matrix factorization ≈ GAE-class, so arm 2 *will* be strong). | Phase 2's 4-way table IS the falsifier: (1) frozen pretrained vs (2) identity body vs (3) frozen random vs (4) scratch curve, per-config LR sweep × 3 seeds. | This is a **measurement, not a pass/fail gate** (Gate 2's honest framing): Δ(1−2)≈0 pre-bias is a legitimate publishable outcome; Phase 3 is the intervention meant to create the gap. Stop-the-line only if nothing trains or the protocol is broken. If after Phase 3 Δ(1−3) ≤ seed σ, the honest paper is the ablation paper. |
| **R5** | **The M4 distortion eats the bias** — observed-graph SPD asserts "far apart" about exactly the masked positives, so the bias helps less than Graphormer-lineage numbers suggest, or hurts. | The plan's day-one Phase-3 probe: parameter-free 1/SPD, single scalar λ, N=512, hours not days. Bounded in (0,1], cannot blow up. | λ-probe moves val AUC by ≥1σ ⇒ proceed to the learned table. Moves nothing ⇒ the learned table gets one shot with the leak tests live; still nothing ⇒ record "bias does not transfer to masked-edge prediction" as a finding and lean on the encoder study (the PI deliverable is Phase 4 regardless). |
| **R6** | **FPT-style gains vanish in our regime** — the pretrained-vs-random gap was smallest on the longest sequences (C10-LRA +2.5), and we additionally remove causality and positions; T5-enc/Longformer collapsed on ListOps (`fpt.md` OQ1). | Same Phase-2 table, Δ(1−3) column; plus the data-fraction sweep (10/25/50/100% of training edges) where GPT4TS says pretraining should shine (`fpt.md` C3). | Pretrained > random at low data fractions but not full data ⇒ reframe the claim around the low-data regime (legitimate, and Cora is small); no gap anywhere ⇒ R4's exit. |
| **R7** | **Silent no-gradient failures** — M9 (checkpointing no-op) and its cousins: empty param groups, disconnected bias gate, LoRA grads None. All train cleanly and read as clean negative results. | Grad-existence assert after first backward; per-group grad-norm logs; `bias-nonzero-after-100-steps` assert; empty-group assert at optimizer build (§4.8–4.9). | Any assert fires ⇒ fix before the run counts. These run in every job, not just tests. |
| **R8** | **Bias magnitude blow-up or bias inertness** — |b| ≫ 1 replaces attention (10 pp below the no-bias floor in GTLM's own logs, B11); |b| pinned at 0 means the LR ratio failed. | `max|bias|/max|qk/√d|` logged per layer, alert >0.5; zero-init inertness check (step-0 == Phase-2 model, Gate 3). | Ratio alert ⇒ drop bias LR one decade; table still at 0 after 100 steps ⇒ raise it one decade; either persisting ⇒ inspect the SPD feature pipeline before touching the model. |
| **R9** | **Encoder norm mismatch / vanishing encoder gradient** — M1: 30–70× norm mismatch is the documented multimodal failure; the T/√d gain fix attenuates encoder grads ~65×. | Init-time norm unit test (mean ‖token‖₂ within 25% of T); `encoder.grad.norm()` logged from step 0. | Norm test red ⇒ fix init before any run. Encoder grads ≪ decoder grads for >200 steps ⇒ enable the GWC hook (`fpt.md` §5) and rerun the smoke test. |
| **R10** | **D1's expressiveness ceiling caps the measurable effect** — automorphic pairs get identical scores under any inner-product decoder (M6), the exact mechanism behind GAE-91 vs MaskGAE-96. | D3 runs beside D1 from Phase 2 (both in `g2l/decoders.py`); compare deltas under both. | If the LLM/bias delta appears under D3 but not D1, report D3 as primary and D1 as the ablation — decided by data, not preference. |
| **R11** | **Our graph tokens become attention sinks** — high activation, no semantic utility (2606.03712 found pruning the top-2 sink tokens costs 0–1%). | Their diagnostic, ported: activation norms + attention mass per token; prune top-2 highest-activation tokens, re-evaluate. An afternoon on a Phase-3 checkpoint. | Pruning costs <1% AUC ⇒ we have their disease; investigate before writing any "the LLM reasons over structure" sentence. |
| **R12** | **Dev/prod split** — Windows/py3.14/cu130 vs Linux/cluster python/whatever pip resolves; CRLF; path conventions. (`environment.md` §4) | First Marlowe job reproduces a locally-known number (the Phase-1 scratch sweep — already scripted as `slurm/phase1_scratch_sweep.sbatch`); `.gitattributes` forces LF (committed); `PYG_DATA_ROOT` env-var everywhere. | Marlowe-vs-laptop disagreement beyond seed noise on the reproduction job ⇒ freeze sweeps until the stack diff is root-caused (torch/CUDA version first). |
| **R13** | **Walltime / requeue losses** — `batch` hard-stops at 2 days; nodes fail; arrays get requeued. | Runs sized ≤4 h; JSONL idempotency skip (re-run costs nothing); checkpoint-resume path exercised once deliberately (kill a job mid-run, resume, diff the curves). | Resume test not bit-clean ⇒ fix before Phase-4's 81-task array. |
| **R14** | **Scratch purge** — `/scratch/m000211` is not backed up and its purge policy is **unpublished** ([UNKNOWN] #10 in `marlowe.md`). | `scripts/marlowe_verify.sh` checks `/etc/motd` + quota tools at first login; RESULTS-bound artifacts copied to `/projects/.../results/` at job end, HF snapshot mirrored to `/projects/m000211/models/` if purge proves aggressive. | Nothing irreplaceable ever lives only on scratch — standing rule, not a gate. |
| **R15** | **Queue wait starves iteration** — machine at ~90% allocation; no debug partition. | `squeue --start` on the first probe jobs; measure real wait for 1-GPU 2-h jobs during Phase-1's Marlowe onboarding. | Median wait > ~half a day for 1-GPU jobs ⇒ shift bring-up further onto the laptop (N≤512), batch gates weekly, and consider uncharged `preempt` (bare account, [UNKNOWN]) for non-critical sweeps with checkpointing. |
| **R16** | **Gated-Llama access friction** — `meta-llama/Llama-3.2-1B` returns 401 anonymously; token handling on a shared cluster. | User accepts license + creates token before Phase 2's Marlowe entry (plan, user-owned); staged download on login node; token never in git (`marlowe.md` §9.4). | Blocked >a few days ⇒ Qwen3-0.6B (ungated, Apache-2.0) becomes the bring-up backbone; Llama numbers re-run once access lands. GTLM comparability is the only thing deferred. |
| **R17** | **The bar is misjudged** — beating GAE-92 and calling it a win (I3), or comparing against MaskGAE's 70%-mask number at our mask rate. | MaskGAE re-run under our exact protocol and mask rate (one afternoon with their repo, `competitors.md` OQ4) before any "beats SOTA" sentence. | Any claim sentence names its protocol column and its re-run bar, or it doesn't ship. |

Deliberately not risks: budget (§7.4 — 3–13% of one cycle worst case); Cora fitting in H100
memory (§7.2 — settled arithmetic); the AR task's incompatibility with the bidirectional mask
(M7 — already designed around via `attn_mode`, right-shift, and no-masking-under-AR, §3.6).

---

## 9. Revised build order

Mirrors the approved plan's phases and gates (the plan is the authority; this is the
KB-resident summary). Status as of 2026-08-25, from the repo — **no results are stated here
beyond what files in the repo record.**

| Phase | Content (condensed) | Gate | Status |
|---|---|---|---|
| **0 — Protocol foundation** (local) | `data.py` ([B,N,N]+padding from day one), `metrics.py` (3-column), both masking views, diagonal excluded, MP/supervision disjoint; leak tests armed; git + private GitHub + pinned requirements | **GATE 0**: all tests green; identity ≈ base rate; user reproduces the three columns by hand on a synthetic graph (`python -m g2l.walkthrough --phase 0`) | **COMPLETE** — `results/phase0/RESULTS.md`: 18/18 tests green, Cora facts verified live, identity at exactly 0.500/1.000; commit `082ad39`. User `VALIDATED` signature line still blank in the file. |
| **1 — Baselines ∥ Marlowe onboarding** | Laptop: CN/AA/RA/PPR → feature-only logreg → GAE/VGAE (+600ep reference protocol) → GAT(LP config) → scratch adjacency-row transformer capacity sweep d∈{128…2048} (Phase 2's matched-params curve). Marlowe: first SSH (user provides Duo), `marlowe_verify.sh` resolves the 20 [UNKNOWN]s, venv on /projects, pre-stage Llama+Cora, submit the locally-validated scratch sweep as the first array, `HF_HUB_OFFLINE=1` | **GATE 1**: GAE reproduces 90.6±0.9/91.2±1.0; full table; a Marlowe array job reproduces a local number; RESULTS.md signed | **IN PROGRESS** — baselines implemented (commits `1dfb862`, `d917875`); laptop reference table exists with 9 models × 5 seeds (`results/phase1/laptop_reference_table.jsonl`, §6.2 — GAE row lands on target); onboarding kit committed (`ea0511b`: verify/setup scripts, WSL ssh config, sbatch). Scratch-sweep results and the Marlowe reproduction are **not yet in the repo**; `results/phase1/RESULTS.md` not yet written; gate not signed. |
| **2 — Frozen LLM core, no bias, no LoRA** | Local: E1+D1/D3+frozen Llama (§4.1 recipe), N≤512 overfit smoke; `attn_mode` field exists now. Tests landing: 4-D canary, zero-bias==no-mask parity, base-LLM parity, `attention_scaling==1.0`, frozen-base, grad-existence, encoder-norm≈T. Marlowe: full-graph Cora (grad ckpt), the 4-way comparison — (1) frozen pretrained / (2) identity body / (3) frozen random / (4) scratch curve — per-config LR sweep × 3 seeds, one array | **GATE 2** (honest framing): the deliverable is the **delta table** Δ(1−2), Δ(1−3) ± σ, not a win; (1)≈(2) pre-bias is legitimate; stop-the-line only if nothing trains or the protocol is broken | not started |
| **3 — Graph-aware attention** | Day-one 1/SPD·λ probe (local, N=512) → learned per-head SPD table (zero-init, `_is_hf_initialized`, diagonal zeroed, unreachable=learned bucket, A_observed-only — the leak test now runs for real, layer-shared, differential LRs, magnitude logging). Controls: shuffled-A (shuffled everywhere incl. SPD), zero-init inertness. Exit item: flex `score_mod` parity vs eager (~1e-3 bf16, bias as non-leaf). Training stays eager | **GATE 3**: bias delta over Phase 2 with seeds+LR sweep; shuffled-A degrades; inertness verified; flex parity green | not started |
| **3B — LoRA** | r=16 α=32 dropout 0.05, all 7 projections, `task_type=None`, 3e-5. Staged here so ~11.3M params contaminate neither Phase 2's question nor Phase 4's ranking | **GATE 3B**: LoRA delta on E1 known; recorded decision whether the encoder study runs with/without (default: without; top-2 re-run with) | not started |
| **4 — Encoder study E1–E7** (the PI deliverable) | Marlowe arrays: 7 × per-encoder LR sweep × 3 seeds on Cora (E5 = normalized-Laplacian recipe). Laptop parallel: molecular masked-edge (molhiv/bbbp/bace) — the size-agnostic claim of E3/E6/E7 is untestable on single-graph Cora. Crossing control lands here (needs E4): (A=I, features-only) × (X=0, topology-only) + feature permutation — this, not shuffled-A, is the decisive control (I12). Permutation tests scoped per §3.2 | **GATE 4**: E1–E7 table on {Cora, molecular} with param counts; crossing quadrant; user can state per encoder what it adds and cannot represent | not started |
| **5 — Task expansion** | Masked node prediction (same infra). AR: `attn_mode='causal'`, right-shifted input, **no masking** (causality hides the future), ordering = swept {BFS, DFS, degree, k-core, random} computed leak-free — BFS is NOT a safe default (M8) | **GATE 5**: causality test (`grad(logits[t], tokens[>t])==0`) + ordering-leak test green; AR vs masked-edge on a shared index set; ordering-sweep spread reported | not started |
| **6 — Scale + full grid** (Marlowe only) | PubMed, ogbn-arxiv (flex now load-bearing; subgraph sampling; identity canary re-run under subgraph batching), decoder sweep D2/D4, per-layer bias ablation (triggers Impl B), LLM2Attn/LLM2Trsf, fully-trained-random (labelled separately from frozen-random — I14), full CLAUDE.md-§8 grid via arrays | **GATE 6**: flex-vs-eager parity re-verified at scale; grid complete | not started |
| **7 — Biomedical** (deferred) | Directed graphs — the magnetic Laplacian becomes non-trivial for the first time (T3). Requires a data-classification review first: both clusters are Low/Moderate-risk only | — | deferred |

**Day-1 tests** (the list §1's "[UNVERIFIED — day-1 test, §9]" forward-references; each is
cheap and phase-tagged):

1. *Every phase, every training start, both machines*: 4-D-mask canary (§4.3); identity-at-
   chance; grad-existence after first backward.
2. *Phase 2 start*: `attention_scaling == 1.0`; zero-bias==no-mask parity; base-LLM parity
   (our loading recipe vs a stock `AutoModel` forward on the same random embeds);
   frozen-base assert; encoder-output-norm ≈ T.
3. *Phase 3 start*: zero-init inertness (step-0 outputs == the Phase-2 checkpoint's);
   bias-equality under held-out-edge changes (the armed Phase-0 test, now against the real
   bias module); bias-nonzero-after-100-steps.
4. *Marlowe first login*: `scripts/marlowe_verify.sh` (partitions, assoc, array caps, quotas,
   driver, compute-node internet probe). *Marlowe first GPU session*: the
   `torch.compile(flex_attention)` smoke block from `marlowe.md` §5 — this is the test that
   retires I1's [UNVERIFIED] clause on the machine where flex actually matters.
5. *Phase 5 start*: D4 causality + ordering-leak tests (`autoregressive.md` R10).

Per-phase deliverable loop (unchanged from the plan): 1-page RESULTS.md with the claim in one
sentence, the table with identity+random rows recomputed, shows/does-not-show bullets, locked
config values, `VALIDATED <date>`; `python -m g2l.walkthrough --phase N` recomputes headline
numbers live on the laptop (<15 min); one slide appended to `slides/`.

Bookkeeping note: current `tests/` holds 21 test functions (Phase-0's 18 + Phase-1's
`test_protocol.py` 3); `environment.md`'s "all 26 tests green" does not match a
`grep -c "^def test_"` count of the committed files — [UNVERIFIED], recount with
`pytest --collect-only` and fix whichever number is stale.

---

## 10. Open questions we could not resolve

Deduplicated from the open-question sections of all eight notes files plus this document's
own analysis. Grouped; sources in parentheses; items already *decided* by the plan (masking
protocol, ordering-as-config, backbone, hosting) are not re-opened here.

**Architecture / theory**
1. Does the attention-bias path actually escape Theorem 4.2's communication lower bound? The
   two-party-protocol argument may extend to a biased transformer; if it does not, that is a
   theorem worth writing — if it does, we must stop implying the bias solves the width
   problem. Unproven either way; never state as settled (depthwidth.md OQ1/OQ2).
2. Does an SPD bias help at all with one token per node and no verbalization? Every existing
   datum is from a from-scratch graph transformer (big win) or a frozen LLM eating serialized
   text (mixed). Nobody has measured our configuration — this is arguably the paper
   (bias-mechanism.md OQ1).
3. Is the SPD bias redundant with an adjacency-row encoder? E1's token already *is* the 1-hop
   indicator; Graphormer's tokens carry no adjacency, which may fully explain its 37% gain.
   Needs the bias-on/off × row-visible/row-hidden crossed ablation (bias-mechanism.md OQ2).
4. Per-layer vs layer-shared bias tables: nobody in the lineage has ablated it (Graphormer/T5
   share; GTLM doesn't; 320 vs 5,120 params). Our Phase-6 ablation; cheap and publishable
   (bias-mechanism.md OQ4, hf-mechanics.md OQ2).
5. Layer schedule {all, first half, last half, every other}: GaLA's first-half-only has no
   ablation behind it and a text-generation rationale we don't share (T7).
6. Does the differential-LR ratio (167–667×) transfer when the tokens the bias competes with
   have no pretrained meaning? Sweep; treat GTLM's values as a starting point only
   (gtlm.md OQ3).
7. Hub-node hypothesis: Thm 4.4 predicts error concentrates on the ~27 nodes with degree
   above ~89 at d_model=2048. Bucket AUPRC by degree; compare against a 4096-wide backbone if
   it shows (depthwidth.md OQ5).
8. Is a LayerNorm the right interface for a magnitude-carrying token (degree)? RevIN-style
   re-injection vs E5's explicit degree channel — untested (fpt.md OQ4).
9. E5b = `row ‖ Lap ‖ RW ‖ X`: the follow-up's `Comb` tokenization wins 3 of 8 tasks;
   evidence says yes, param cost says measure (depthwidth.md OQ7).

**Task / evaluation**
10. Which regime is Cora masked-edge prediction — GPT4TS's low-data (pretraining matters) or
    Tan's full-data (it doesn't)? The 10/25/50/100% training-edge sweep decides, and is our
    strongest potential positive result (fpt.md OQ2).
11. MaskGAE at our mask rate: does 96.45 survive p=0.15? Must be measured before any bar is
    claimed (competitors.md OQ4; §8 R17).
12. No published sparse-view AUPRC exists for anything — our GAE-sparse rows are the first
    and have no external check; mitigation is the balanced column's anchor
    (pyg-baselines.md OQ8).
13. AR-phase bias causality: `bias(t,j)` from the full graph leaks the future through the
    bias under a causal mask; per-prefix recomputation is O(N) SPD passes; unaddressed in any
    literature we found (autoregressive.md OQ2; §3.5 residual).
14. Which ordering for molecular AR (G2PT: BFS≈degree≫DFS; GRAN: DFS≫BFS on grids;
    molecules are neither ours nor theirs) — the Phase-5 sweep exists because no ordering
    wins everywhere (autoregressive.md OQ4).
15. Does the frozen LLM help under a *causal* mask (closer to pretraining than our
    bidirectional mode)? Run the scratch control for §5.3 specifically, not just §5.1
    (autoregressive.md OQ1).

**Engineering**
16. Does the 4-D early-exit have a deprecation plan? Watch `masking_utils.py` on any
    (deliberate) upgrade; the canary is the tripwire (hf-mechanics.md OQ1).
17. FlexAttention with our exact gather (`table[h, spd[q,kv]]` after the non-leaf hoist):
    performance and the #153799 NaN edge at padded-2816 — measured at the Phase-3 parity
    item, not before (pyg-baselines.md OQ4/OQ5).
18. Does `torch.compile` tolerate `merge_with_config_defaults`'s config mutation? Prefer the
    sticky `config.is_causal=False` if compiling (hf-mechanics.md OQ6).
19. Ragged batching via finfo.min-in-the-bias vs B=1+grad-accum: untested; B=1 is the
    Phase-2 default (hf-mechanics.md OQ7).
20. `NeighborLoader`'s induced-subgraph argument name (Phase 6 sampling) — unverified
    signature (pyg-baselines.md OQ7).
21. Marlowe's 20 first-login [UNKNOWN]s (array caps, purge policy, compute-node internet,
    bare-account preempt billing, driver version…) — `scripts/marlowe_verify.sh` exists to
    close them; until run, every dependent statement in §7.5 is provisional
    (marlowe.md §10).

**Positioning**
22. GraphToken's projection parameter count (80K vs 1.1e7 — self-contradictory) and its
    soft-token count (never stated): do not cite either until resolved (T10;
    competitors.md OQ2/OQ3).
23. Whether GaLA has code (none found), what β it used (never published), and whether its
    entropy heuristic means anything on non-text tokens — the gradient variant is the safer
    port if we ever calibrate per-head strength (T6; bias-mechanism.md OQ7/OQ8/OQ9).
24. Whether anyone has verbalized a 2,708-node graph into a frozen LLM (we found no attempt;
    state as our own measurement if we run the ablation, not as a literature claim)
    (competitors.md OQ7).
25. Whether the 2-cycle/directed results transfer at all to undirected masked-edge
    prediction — most of the Depth-Width theory speaks about directed detection tasks; we use
    it for sizing, not for task guarantees (depthwidth.md, standing caveat).
26. GTLM's publication status (NeurIPS review as of the repo's July-2026 rebuttal docs):
    check for a camera-ready before citing the Limitations section, which is slated to
    change (gtlm.md OQ8).
