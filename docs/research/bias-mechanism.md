# Bias Mechanism

Domain notes on the attention-bias lineage our model implements: raw adjacency -> structural
features -> a scalar added to pre-softmax attention logits inside a frozen LLM.

Everything below was fetched from a primary source (arXiv full text or raw GitHub) during this
research pass. Where I could not verify something I say so explicitly. "The paper says X" and
"I believe X" are kept apart.

---

## Verified facts (with source next to each)

### Graphormer (Ying et al., NeurIPS 2021, arXiv 2106.05234)

| # | Fact | Source |
|---|---|---|
| G1 | Spatial encoding equation is **Eq. (6)**: `A_ij = (h_i W_Q)(h_j W_K)^T / sqrt(d) + b_phi(v_i,v_j)` | ar5iv 2106.05234 §3.1.2 |
| G2 | `b_phi(v_i,v_j)` is "a learnable **scalar** indexed by phi(v_i,v_j), and **shared across all layers**" | ar5iv 2106.05234, text directly under Eq. (6) |
| G3 | `phi` = shortest-path distance if the two nodes are connected. "If not, we set the output of phi to be a special value, i.e., **-1**." | ar5iv 2106.05234 §3.1.2 |
| G4 | Centrality encoding is **Eq. (5)**: `h_i^(0) = x_i + z^-_{deg^-(v_i)} + z^+_{deg^+(v_i)}` — added to the *input embedding*, not to the logits | ar5iv 2106.05234 §3.1.1 |
| G5 | Edge encoding is **Eq. (7)**: `A_ij = (h_i W_Q)(h_j W_K)^T/sqrt(d) + b_phi(i,j) + c_ij`, with `c_ij = (1/N) * sum_{n=1..N} x_{e_n} (w_n^E)^T` — the mean dot-product of edge features along the shortest path | ar5iv 2106.05234 §3.1.3 |
| G6 | In code the bias is added **after** the `1/sqrt(d_head)` scaling: `q *= self.scaling` (line 145) then `attn_weights = bmm(q,k)` then `attn_weights += attn_bias` (line 182) | raw `graphormer/modules/multihead_attention.py` |
| G7 | Unreachable pairs are **510** in the actual code, not -1 | raw `graphormer/data/algos.pyx` lines 26-33, 47-53 |
| G8 | `spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)` — **per-head**, one table, `num_spatial=512` default | raw `graphormer_layers.py:101`; `graphormer/tasks/graph_prediction.py` `num_spatial: int = field(default=512)` |
| G9 | The collator applies `+1` to SPD (`pad_spatial_pos_unsqueeze`) so index 0 is reserved for padding; self-pairs (SPD 0) -> index 1; unreachable (510) -> index 511 | raw `graphormer/data/collator.py` |
| G10 | Pairs with raw SPD `>= spatial_pos_max` are **hard-masked to -inf**, not softly biased. Function default is 20; the fairseq task default is **1024** | `collator.py:` `attn_biases[idx][1:,1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")`; `graph_prediction.py` `spatial_pos_max: int = field(default=1024)` |
| G11 | The bias tensor is computed **once** and handed to every layer | raw `graphormer_graph_encoder.py:223` `attn_bias = self.graph_attn_bias(batched_data)`, then `self_attn_bias=attn_bias` at line 250 |
| G12 | Ablation Table 5 (12-layer models, 100K iters, PCQM4M-LSC valid MAE) — exact rows below | ar5iv 2106.05234 Table 5 |
| G13 | Setting `b_phi = 0` when `phi = 1` and `b_phi = -inf` otherwise, plus `W_Q = W_K = 0`, `W_V = I`, makes self-attention compute exact MEAN neighbour aggregation | ar5iv 2106.05234 Appendix (proof that GNNs are special cases) |

**Graphormer Table 5, verbatim** (`valid MAE`, lower better). Columns: Laplacian PE / Spatial /
Centrality / edge-via-node / edge-via-Aggr / edge-via-attn-bias.

| LapPE | Spatial | Centrality | via node | via Aggr | via attn bias | valid MAE |
|:--:|:--:|:--:|:--:|:--:|:--:|--:|
| - | - | - | - | - | - | **0.2276** |
| yes | - | - | - | - | - | **0.1483** |
| - | yes | - | - | - | - | **0.1427** |
| - | yes | yes | - | - | - | **0.1396** |
| - | yes | yes | yes | - | - | **0.1328** |
| - | yes | yes | - | yes | - | **0.1327** |
| - | yes | yes | - | - | yes | **0.1304** |

Deltas that matter to us:

* **no encoding -> spatial encoding: 0.2276 -> 0.1427, a 37.3% relative MAE reduction.** This is by
  far the largest single contribution in the table. The SPD attention bias is doing the heavy lifting.
* Laplacian PE -> spatial encoding: 0.1483 -> 0.1427, only 3.8% relative. So the *bias* form beats the
  *input-feature* form of the same spectral information, but not by much.
* + centrality (degree embedding on the input): 0.1427 -> 0.1396, 2.2% relative. Small.
* + edge encoding as attention bias: 0.1396 -> 0.1304, 6.6% relative; and as a bias it beats the two
  conventional edge treatments (0.1328 via node, 0.1327 via Aggr) by ~1.8%.
* Final full Graphormer (bigger, longer training) reaches 0.1234 valid MAE vs GIN-VN 0.1395.

### GaLA (arXiv 2606.15633, KDD '26)

**The arXiv ID in CLAUDE.md resolves and is correct.** I fetched it.

| # | Fact | Source |
|---|---|---|
| A1 | Real title: **"Formalizing and Mitigating Structural Distortion in LLM Attention for Graph Reasoning"**. GaLA = "Graph-aligned Language Attention" is the *method* inside it. Authors Donald Loveland, Puja Trivedi, Ari Weinstein, Edward W Huang, Danai Koutra. v1 2026-06-14, v2 2026-06-17. Accepted KDD '26 (Aug 9-13 2026, Jeju) | arxiv.org/abs/2606.15633, arxiv.org/html/2606.15633v2 |
| A2 | Attention equation: `a~^(l,h)_{i,j} = (q~_i^(l,h))^T k~_j^(l,h) / sqrt(d) + lambda_h * B_{i,j}` where q~,k~ are the **RoPE-rotated** query/key | html v2 §6.1 |
| A3 | `B_{i,j} = 1 / d_G(M(i), M(j))` when `M(i),M(j) in V`, `M(i) != M(j)`, and `d_G < inf`; **`B_{i,j} = 0` otherwise**. So unreachable pairs and self-pairs get exactly zero, not a special learned value | html v2 §6.1 |
| A4 | Injection applied only in early layers, `l in [0, L/2]` | html v2 §6.1 |
| A5 | Label-free entropy calibration: `lambda_h = beta * sqrt( H(h) / max_{h' in H_l} H(h') )`, `H(h) = E_{S in D_cal}[ -(1/T) sum_{i=1..T} sum_{j=1..i} A^(h)_{i,j} log A^(h)_{i,j} ]`. Inner sum restricted to `j <= i` for the causal mask. Normalisation is **per layer** (`max` over `H_l`, heads in layer l) | html v2 §6.2 |
| A6 | Label-based gradient calibration: attach per-head scale `g_h` **initialised to 1** to each head's structural bias, freeze everything else, one backward pass. `lambda_h = beta * max{0,-G(h)} / max_{h'} max{0,-G(h')}`, `G(h) = E_{S in D_cal}[ d/dg_h (-log p(y|S)) ]` | html v2 §6.2 |
| A7 | `beta` is selected on a validation set of **200 samples** disjoint from test | html v2 Appendix C |
| A8 | Calibration cost: ~30-39 s one-time (entropy 30/35/33 s, gradient 39/28/29 s on Cora/PubMed/Arxiv). Per-sample shortest-path adds 0.001/0.013/0.066 s | html v2 Table 3 |
| A9 | Backbones: Qwen2.5-3B-Instruct, Ministral-3B-Instruct, Qwen2.5-7B-Instruct. All methods use **BFS linearization** to minimise edge stretch | html v2 §7 |
| A10 | Head-selection ablation (Qwen2.5-3B), Table 2 | html v2 Table 2 |

**GaLA Table 2, verbatim** (accuracy %):

| Method | Head selection | Semi-Cora | Semi-PubMed | Semi-Arxiv | Real-Cora | Real-PubMed | Real-Arxiv |
|---|---|--:|--:|--:|--:|--:|--:|
| BFS Lin. | - | 83.3 | 90.3 | 80.3 | 65.9 | 77.5 | 35.2 |
| GaLA | All | 89.1 | 91.2 | 82.4 | 67.7 | 80.1 | 35.2 |
| GaLA | Random | 87.8 | 90.9 | 83.1 | 66.8 | 80.2 | 37.0 |
| GaLA | Entropy | 89.1 | 91.5 | 83.1 | 67.0 | 79.2 | 36.0 |
| GaLA | Gradient | 90.4 | 92.0 | 82.4 | 67.4 | 82.5 | 36.8 |

### GTLM (arXiv 2605.10247) — the closest existing work

| # | Fact | Source |
|---|---|---|
| T1 | Attention: `A^(l,h)_{i,j} = Q_i.K_j^T / sqrt(d_head) + b_SPD^(l,h)(u,v) + b_RRWP^(l,h)(u,v) + b_Mag^(l,h)(u,v)` for tokens i in node u, j in node v | html 2605.10247 §3.2 |
| T2 | `b_spd in R^{L x H x max_spd}` — **per layer AND per head**, with `b_SPD^(l,h)(u,u) = 0` | html §3.2 |
| T3 | `b_RRWP^(l,h)(u,v) = 0` if `u=v`, else `[MLP_RRWP^(l)(P_{u,v})]_h`, `MLP_RRWP^(l): R^K -> R^n_heads` | html §3.2 |
| T4 | `b_Mag^(l,h)(u,v) = 0` if `u=v`, else `[MLP_Mag^(l)([Re(K_{u,v}) || Im(K_{u,v})])]_h`, with kernel `K = V diag(phi(lambda)) V^dagger in C^{NxNxd_Mag}` and DeepSet `phi(lambda_i) = sigma(W_2 [Linear(lambda_i) (+) mean(Linear(lambda))])` | html §3.2, Appendix A |
| T5 | Bias hyperparameters (Table 6): **SPD max dist 8, max RRWP steps 16, Magnetic Lap. q = 0.25, Magnetic Lap. dim 32** | html Appendix C Table 6 |
| T6 | **173,056** bias parameters = **4,096 SPD + 50,688 RRWP + 118,272 Magnetic** = **0.015%** of the 1B base model | html §5 "Parameter Efficiency" |
| T7 | Backward compatibility verified numerically: max abs logit difference **2.1e-5 +/- 8.3e-6** over 5 trials | html Appendix B.2 |
| T8 | Differential LR, GraphQA (Table 7): `eta = 3e-5`, `eta_bias = 5e-3`, **LoRA r = 16, alpha = 32** | html Appendix D Table 7 |
| T9 | Differential LR, TAG benchmarks (Table 8): `eta` from {6e-5, 2e-4, 3e-4}, `eta_bias` from {0.01, 0.04}, LoRA r from {32, 64}, alpha = 2r | html Appendix E Table 8 |
| T10 | FlashAttention incompatible -> O(N^2) eager attention; bidirectional mask adds more; **~3x longer training** than baseline LLM at ~1,000-token sequences | html §5 "Limitations" |
| T11 | **Ablation (Table 5, GraphQA): the Magnetic Laplacian bias is the most crucial; SPD and RRWP give "more modest gains only in some specific tasks"** | html §4.5 |

**GTLM Table 5 (leave-one-bias-out, GraphQA accuracy %)** — the single most consequential table for
our plan:

| Method | Node Count | Edge Count | Cycle Check | Tri. Count | Node Deg. | Conn. Nodes | Reachability | Edge Exist. | Shortest Path |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Standard | 100+/-0.0 | 56.9+/-1.7 | 96.7+/-0.6 | 30.9+/-0.6 | 99.7+/-0.1 | 90.7+/-0.3 | 99.0+/-0.0 | 99.8+/-0.2 | 90.1+/-1.7 |
| w/o SPD | 100+/-0.0 | 52.6+/-5.9 | **97.2**+/-0.3 | 29.9+/-0.6 | 99.1+/-1.0 | 81.4+/-2.3 | 98.8+/-0.5 | 99.7+/-0.1 | **92.0**+/-2.6 |
| w/o RRWP | 100+/-0.0 | 48.9+/-1.5 | **97.4**+/-0.2 | 30.7+/-0.6 | 99.5+/-0.4 | 87.7+/-1.7 | 97.5+/-0.6 | 99.6+/-0.4 | 90.0+/-1.8 |
| w/o Magnetic | 100+/-0.0 | **7.1**+/-1.3 | 93.6+/-2.0 | **20.2**+/-1.2 | 99.6+/-0.2 | 91.0+/-1.4 | 97.0+/-0.4 | 99.5+/-0.1 | **76.3**+/-12.7 |

Read it carefully: dropping SPD **improves** Cycle Check (96.7 -> 97.2) and Shortest Path
(90.1 -> 92.0). Dropping Magnetic collapses Edge Count (56.9 -> 7.1) and Triangle Counting
(30.9 -> 20.2). The SPD hop-bias table — the thing CLAUDE.md §2 centres the design on — is the
*weakest* of the three in the only LLM-side ablation that exists.

### Other bias families

| # | Fact | Source |
|---|---|---|
| R1 | GRIT RRWP: `M := D^-1 A`; `P_{i,j} = [I, M, M^2, ..., M^{K-1}]_{i,j} in R^K` | ar5iv 2305.17589 §3 |
| R2 | Proposition 3.1(a): there exists an MLP `R^K -> R` acting independently per pair such that `MLP(P)_{ij}` approximates `SPD_{K-1}(i,j)`. RRWP + MLP subsumes the SPD table | ar5iv 2305.17589 |
| R3 | GRIT attention: `e_hat_{i,j} = sigma( rho( (W_Q x_i + W_K x_j) (*) W_Ew e_{i,j} ) + W_Eb e_{i,j} )` with signed square root `rho(x) = (ReLU(x))^{1/2} - (ReLU(-x))^{1/2}`. Note GRIT *replaces* the dot product; it is not an additive bias on top of one | ar5iv 2305.17589 Eq. 2 |
| R4 | GRIT degree injection: `x_i^{out'} = x_i^{out} (*) theta_1 + log(1+d_i) * x_i^{out} (*) theta_2` | ar5iv 2305.17589 Eq. 5 |
| R5 | GRIT K values in experiments: 21 primary; also 7, 14, 18, 24, 42 | ar5iv 2305.17589 Table 6 |
| R6 | GRIT code: `add_full_rrwp(data, walk_length=8, add_identity=True, spd=False)` computes `deg_inv = 1/adj.sum(dim=1)` (inf -> 0), `adj = adj * deg_inv.view(-1,1)`, stacks `[I, M, M^2, ...]` to `n x n x k`, takes the **diagonal as node PE** and the off-diagonal as **relative PE** | raw `LiamMa/GRIT/grit/transform/rrwp.py` |
| M1 | Magnetic Laplacian, unnormalized: `L_U^(q) = D_s - A_s (*) exp(i Theta^(q))`; normalized: `L_N^(q) = I - (D_s^{-1/2} A_s D_s^{-1/2}) (*) exp(i Theta^(q))`; phase `Theta^(q)_{u,v} := 2*pi*q*(A_{u,v} - A_{v,u})` | ar5iv 2302.00049 (Geisler et al., ICML 2023) |
| M2 | `q = 0` recovers the ordinary combinatorial Laplacian. For undirected graphs any q recovers it (the phase term vanishes because `A = A^T`). Each directed edge induces a phase rotation of `2*pi*q` | ar5iv 2302.00049 |
| M3 | **Geisler et al. recommend `q = q' / d_G` with `q' in {0.1, 0.25}`, where `d_G` normalises by the max number of directed edges on a simple path, and report that for high `q'` "performance drops severely (corresponds to absolute q > 0.05)"** | ar5iv 2302.00049 |
| M4 | Sign/phase ambiguity fix: normalise eigenvectors to unitary, set max real magnitude positive, then rotate all eigenvectors so the phase at a root/source node is zero | ar5iv 2302.00049 |
| S1 | SAN's gamma is **not** an additive logit bias. It reweights two separate attention pathways: real edges get `1/(1+gamma) * softmax(...)`, non-edges get `gamma/(1+gamma) * softmax(...)`, with separate `Q1,K1,E1` vs `Q2,K2,E2`. `gamma = 0` = pure sparse attention | ar5iv 2106.03893 |
| S2 | SAN LPE: concat the m lowest eigenvalues with their eigenvectors -> `2 x m` per node -> linear to k -> Transformer encoder over the m pairs -> sum pool. Laplacian is `D^{-1/2} L D^{-1/2}` | ar5iv 2106.03893 |
| P1 | GraphGPS layer: `X_M^(l+1), E^(l+1) = MPNN_e^l(X^l, E^l, A)`; `X_T^(l+1) = GlobalAttn^l(X^l)`; `X^(l+1) = MLP^l(X_M^(l+1) + X_T^(l+1))`. Global attention sees **no** edge features | ar5iv 2205.12454 |
| P2 | GraphGPS taxonomy: local/global/relative x PE/SE. **Relative PE** = shortest-path distances, heat kernels, random walks, Green's function, graph geodesic — this is the cell our bias module lives in | ar5iv 2205.12454 |
| P3 | GraphGPS experimental encodings: LapPE dim 8-16 via DeepSet; RWSE dim 16-20 processed linearly | ar5iv 2205.12454 |

### Practical: how hard can you push a bias before the LLM degrades

| # | Fact | Source |
|---|---|---|
| B1 | ALiBi: `softmax(q_i K^T + m * [-(i-1), ..., -2, -1, 0])`. Slope `m` is a **geometric sequence starting at `2^(-8/n)` with that same value as ratio**, n = number of heads. 8 heads -> `{1/2, 1/4, ..., 1/256}`. 16 heads -> `{2^-0.5, 2^-1, 2^-1.5, ..., 2^-8}` | ar5iv 2108.12409 |
| B2 | ALiBi slopes are **fixed, not learned**. Learning them "did not yield strong extrapolation results" and slowed training by 3%. Slopes that work best lie in (0,1) with density increasing near 0 | ar5iv 2108.12409 |
| B3 | T5 relative position bias: `nn.Embedding(relative_attention_num_buckets=32, n_heads)`, `max_distance=128`. Half the buckets are exact small distances, half are **log-spaced** up to max_distance; everything beyond collapses into the last bucket | raw HF `modeling_t5.py:212,217-262` |
| B4 | T5 init: `init.normal_(module.relative_attention_bias.weight, mean=0.0, std=factor * d_model**-0.5)` — **not** zero-init | raw HF `modeling_t5.py:616` |
| B5 | T5 owns the bias table only on **block 0** (`has_relative_attention_bias=bool(i == 0)`) and threads the computed `position_bias` through every later block — one table, all layers | raw HF `modeling_t5.py:648` |
| B6 | Wortsman et al. 2023: "we first noticed that **all points with attention logits above 1e4 diverged**." Transplanting a forced max attention logit `kappa`: "loss deteriorates around `kappa = 1e3`, and by `kappa = 1e4` the loss exceeds that of a zero-layer bigram model" | ar5iv 2309.14322 §3.3 |
| B7 | qk-layernorm (LayerNorm on q and k before the dot product) is the standard fix for attention-logit growth; it let them train a 1.2B model at LR 0.3 | ar5iv 2309.14322 §3.1.1 |
| B8 | GTLM's own diagnostic states the reference scale plainly: "attention logits are q.k/sqrt(d_head) with d_head=64, i.e. **O(1-10) before softmax**. A bias bound far above that does not nudge attention, it replaces it." | raw `DarioVajda/graph_model` `.../landmark/diagnose_scale.py` docstring |
| B9 | Measured working bias magnitudes in GTLM at their operating point: **phase channel 0.309, magnitude channel 0.021**. At a 6x trunk scale the magnitude channel reaches **52.85**, which "saturates softmax outright" | raw `src/models/biases/MIXED_BIAS.md` §5.7 |
| B10 | A bias that is **degree-2 in learned parameters** (a product of two trainable factors) grows as `k^4` under trunk scaling and diverged to NaN in 3 runs at `bias_lr = 2e-2`; the degree-1 form (`k^2`) never diverged at either LR | raw `MIXED_BIAS.md` §5.7 |
| B11 | Documented failure: an unbounded product-form bias scored **0.357 F1 against a 0.462 no-bias floor** — a 10 pp *regression* — consistent across dims, LRs and seeds, and got **worse** with higher LR (0.357 -> 0.211 at bias_lr 2e-2) | raw `.../landmark/diagnose_scale.py` docstring |
| B12 | `max_grad_norm = 1.0` does **not** prevent this: AdamW's update `lr * m_hat/(sqrt(v_hat)+eps)` is approximately scale-invariant, so clipping bounds the step, not the trajectory. With `bias_weight_decay = 0` each bias parameter can drift up to `bias_lr` per step, unopposed | raw `MIXED_BIAS.md` §5.7 |
| B13 | The fix that worked: **L2-normalise before the inner product**. Scaling the trunk by k, the bias went 0.0227 -> 0.363 -> 7.40 -> 136.1 unnormalised (k=1,2,4,8) versus 0.0756 -> 0.0968 -> 0.1038 -> 0.1038 normalised. Splitting into `Z_Q || Z_K` does **not** fix it (still quartic) | raw `MIXED_BIAS.md` §5.7 |
| B14 | GTLM's `SPDBias` is **zero-initialised**: `nn.Parameter(torch.zeros(max_spd, num_heads))`, and marked `_no_weight_decay = True` | raw `src/models/bias.py:103-107` |
| B15 | GTLM's `RRWPBias` MLP **zero-initialises its output layer** (`nn.init.zeros_(self.proj[2].weight)` and `.bias`) so the bias starts at exactly 0 | raw `src/models/bias.py:161-163` |
| B16 | GTLM has an explicit design test for this: "**Zero-init inertness** — at step 0 the bias is exactly 0 and logits match the no-bias model" | raw `src/models/biases/LINEAR_BIAS.md` test 9 |

---

## Corrections to CLAUDE.md

### 1. GaLA's arXiv ID is correct, but the paper is not called "GaLA" — CONFIRMED, minor

CLAUDE.md §9 lists "**GaLA** — arXiv 2606.15633 (KDD '26) Loveland, Trivedi, Weinstein, Huang, Koutra."
The ID resolves. The paper's title is **"Formalizing and Mitigating Structural Distortion in LLM
Attention for Graph Reasoning"**; GaLA (Graph-aligned Language Attention) is the method proposed in
§6. Cite it by the real title or a reviewer will not find it.

### 2. "Forcing the bias onto every head damages the language machinery" — NOT SUPPORTED

CLAUDE.md §9 asserts this as GaLA's justification for per-head `lambda_h`. **GaLA's own Table 2 shows
the opposite of "damages."** All-head injection beats the BFS baseline on all six columns except a tie
on real-world Arxiv:

* semi-synth: 83.3 -> 89.1 (Cora), 90.3 -> 91.2 (PubMed), 80.3 -> 82.4 (Arxiv)
* real-world: 65.9 -> 67.7 (Cora), 77.5 -> 80.1 (PubMed), 35.2 -> 35.2 (Arxiv)

Even *random* head selection beats all-heads on Arxiv (83.1 vs 82.4; 37.0 vs 35.2). The paper's own
text says "nearly every choice improves over the BFS baseline... the benefit is not tied to a single
heuristic." The defensible claim is: **per-head calibration buys roughly 1-2 pp over uniform all-head
injection** (best case Cora semi-synth 89.1 -> 90.4, real PubMed 80.1 -> 82.5), not that uniform
injection is harmful. Confidence: **certain** — I read Table 2 and the surrounding text.

The real evidence that a too-strong bias damages the model is elsewhere and is much stronger: GTLM's
`MIXED_BIAS.md` §5.7 and `diagnose_scale.py` (facts B9-B13). Cite those instead.

### 3. GaLA's "first half of layers" has NO supporting ablation

CLAUDE.md reports it as a fact to copy. It *is* what the paper does (`l in [0, L/2]`, verified), but the
justification is one sentence of prose — "allowing graph structure to shape contextual representations
while leaving later layers to translate these representations into natural language outputs" — and
**there is no layer-range ablation anywhere in the paper.** I grepped the full HTML: the only ablations
are head-selection (Table 2) and runtime (Table 3). Confidence: **certain**.

Consequence for us: treat the layer schedule as an **open hyperparameter to sweep**, not a settled
result. Note also that GaLA's rationale is specific to a model that must emit *natural language* at the
end. **We decode a graph, not text.** The argument for leaving late layers untouched is weaker for us,
and GTLM — the architecture closest to ours — biases **every** layer.

### 4. `beta` is never given a numeric value in GaLA

CLAUDE.md doesn't claim one, but if we try to copy the recipe: the paper says only that `beta` "is
selected using a small validation set of 200 samples disjoint from the test data." No value, no grid,
no sensitivity plot. Confidence: **certain** (grepped every occurrence of beta). We will have to find
our own scale. Facts B8/B9 give the right order of magnitude to start from: `|lambda_h * B|` in the
**0.02-0.5** range against logits of O(1-10).

### 5. GaLA's motivating problem does not apply to our architecture

CLAUDE.md takes "one idea" from GaLA, which is fine, but the framing should be recorded. GaLA's entire
premise (§1, §5) is that **RoPE turns graph linearization into bandwidth-dependent attention decay** —
graph-adjacent nodes get pushed far apart in the serialized token sequence and RoPE suppresses their
attention. GaLA is a *repair* for verbalization.

Our design has **one token per node, RoPE disabled/reset, and no verbalization at all** (CLAUDE.md §1,
§2, §13). The pathology GaLA fixes is one we never create. What transfers is the mechanism (per-head
`lambda_h`, entropy/gradient calibration, `1/SPD` as a parameter-free bias shape); what does not
transfer is the motivation, the `M: token -> node` mapping (trivially the identity for us), and the
first-half-of-layers rationale. Confidence: **certain** on the paper's framing.

### 6. Graphormer's bias table is shared across ALL layers — CLAUDE.md is ambiguous

CLAUDE.md §2 says "bias comes from graph structure, added at every layer" and §9 says Graphormer's
`spatial_pos_encoder` "is our hop-bias table." Both true, but they hide a design fork:

* **Graphormer**: one table, `b_phi` "shared across all layers" (paper, under Eq. 6), computed once in
  `graphormer_graph_encoder.py:223` and passed to every layer. Params = `num_spatial x num_heads`.
* **GTLM**: `b_spd in R^{L x H x max_spd}` — a **separate table per layer**. Params = `L x H x max_spd`.
* **T5**: one table on block 0, threaded through all blocks (B5). Same as Graphormer.

For Llama-3.2-1B (L=16, H=32): layer-shared SPD with max_spd=8 is 256 params; GTLM's per-layer is 4,096.
Both are negligible. But per-layer is 16x the capacity and 16x the number of things that can blow up.
CLAUDE.md should state which we are building. Confidence: **certain**.

### 7. Graphormer's unreachable handling: paper says -1, code says 510

CLAUDE.md doesn't mention this at all, and it is a real trap. The paper says "we set the output of phi
to be a special value, i.e., **-1**." The shipped Cython (`algos.pyx`) sets unreachable to **510**, and
the collator adds +1 so it indexes embedding row **511** — which is why `num_spatial` defaults to
**512**. Do not implement "-1" literally; you will index a table with a negative number.

Worse, and more important: Graphormer **hard-masks** pairs at `spatial_pos >= spatial_pos_max` to
`-inf` (G10). So beyond a cutoff Graphormer does not softly down-weight distant pairs, it forbids
attention between them entirely. That is a different mechanism from what CLAUDE.md §8 describes as
"soft hop bias," and it means unreachable pairs in Graphormer are **masked out**, not given a learned
"unreachable" bias value. Confidence: **certain** — read from `algos.pyx`, `collator.py`,
`graph_prediction.py`.

### 8. GTLM's parameter and hyperparameter claims are all CORRECT

Every number in CLAUDE.md §9's GTLM block checks out against the paper:

* SPD max distance 8, RRWP max steps 16, Magnetic q = 0.25, dim 32 — Table 6. **Verified.**
* 173,056 bias params = 0.015% of Llama-3.2-1B — §5. **Verified**, and the split is 4,096 SPD +
  50,688 RRWP + 118,272 Magnetic.
* I independently reconstructed two of the three from the shapes, which cross-validates both the paper
  and the repo: SPD `L x H x max_spd = 16 x 32 x 8 = 4,096` exactly. RRWP per layer, using the repo's
  `hidden = 4 * max_rw_steps`: `Linear(16,64) = 16*64+64 = 1,088`, `Linear(64,32) = 64*32+32 = 2,080`,
  total 3,168, times 16 layers = **50,688** exactly. The reported number, the paper's stated `K=16`,
  and the repo's MLP shape are mutually consistent.
* Differential LR 5e-3 vs 3e-5 on GraphQA — Table 7. **Verified.** 0.01-0.04 vs 6e-5-3e-4 on node
  classification — Table 8. **Verified.**
* Intra-node zero bias, difference ~2e-5 — Appendix B.2 says **2.1e-5 +/- 8.3e-6**. **Verified.**
* FlashAttention incompatible, ~3x slower, ~1,000-token sequences — §5. **Verified.**

One incompleteness: CLAUDE.md says "LoRA rank 32 or 64, alpha = 2r." That is Table 8 (TAG benchmarks).
GraphQA (Table 7) used **r = 16, alpha = 32**. Confidence: **certain**.

### 9. CLAUDE.md over-weights the SPD hop-bias relative to the evidence

This is the correction with the most consequence for the build order. CLAUDE.md §2 makes the attention
bias central, §9 calls Graphormer's spatial encoding "the origin of the mechanism GTLM and GaLA both
use," and §10 lists `models/bias/spd.py` first. The evidence is split:

* **For SPD**: Graphormer Table 5, no-encoding -> spatial encoding is 0.2276 -> 0.1427, a 37% relative
  MAE cut — the largest single delta in that table. In a **from-scratch** graph transformer, the SPD
  bias is the dominant structural signal.
* **Against SPD**: GTLM Table 5, the only leave-one-out ablation run **inside a frozen pretrained LLM**,
  finds Magnetic Laplacian most crucial, and removing SPD *improves* 2 of 9 tasks (Cycle Check
  96.7 -> 97.2, Shortest Path 90.1 -> 92.0). Their own words: SPD and RRWP "demonstrated more modest
  gains only in some specific tasks."
* **Theory against a standalone SPD table**: GRIT Prop 3.1(a) — an MLP on K-step RRWP can approximate
  `SPD_{K-1}(i,j)`. The RRWP bias **subsumes** the SPD table up to K-1 hops. A separate SPD table is
  partly redundant with an RRWP head.

Confidence: **likely** that a pure SPD table under-delivers for us. The honest statement is that the
one experiment closest to our setting says SPD is the weakest of three biases, and CLAUDE.md presents
it as the core. Recommendation in "Numbers to beat" below.

### 10. GTLM's paper says bias params "are initialized randomly"; the repo zero-inits them

CLAUDE.md §9 repeats the paper's reasoning: "bias params start random, LoRA starts from refined
pretrained weights" as the motivation for the differential learning rate. But the released code
zero-inits `SPDBias.weights` (B14) and zero-inits the `RRWPBias` MLP output layer (B15), and
`LINEAR_BIAS.md` has an explicit "zero-init inertness" test asserting the bias is exactly 0 at step 0.
Only `LaplacianBias` and `RWSEBias` use `randn(num_heads) * 0.02`.

So the paper text and the shipped code disagree. This does **not** invalidate the differential-LR
advice — if anything zero-init makes a high bias LR *more* necessary, because a zero table has no
gradient-scale head start. But do not repeat "bias params start random" as a fact about the
implementation. Confidence: **certain** on both readings (paper Appendix C; `bias.py:103, 161-163`).

### 11. "Hard 1-hop mask = GAT-style attention" (§8) — needs one qualifier

Graphormer's appendix proves that `b_phi = 0` for `phi = 1` and `-inf` otherwise, **combined with
`W_Q = W_K = 0` and `W_V = I`**, gives exact MEAN aggregation (i.e. GCN-style, not GAT-style). Keeping
the learned `W_Q, W_K` and applying only the `-inf` mask is the GAT-style variant. Both are worth
running; label them distinctly in `ablate.py` or the results will be ambiguous. Confidence: **certain**
on the Graphormer proof; the labelling point is mine.

### 12. Could not verify: no public GaLA code

I searched and found no GitHub release for GaLA. The paper has no code-availability statement that I
could find in the HTML. Everything in the GaLA section above is from the paper text only — we will be
reimplementing from the equations, not porting. Confidence: **likely** (absence of evidence; a repo may
exist unindexed).

### 13. Caution on q = 0.25 for the Magnetic Laplacian

GTLM uses **absolute q = 0.25** (Table 6, and `magnetic_lap.py` default). Geisler et al. — who
introduced the encoding — recommend `q = q'/d_G` with `q' in {0.1, 0.25}` normalised by graph size, and
report performance "drops severely" for **absolute q > 0.05** (M3). These are not the same number.
GTLM's graphs are small ego-subgraphs (30-60 neighbours), so the mismatch may not have bitten them. If
we adopt the Magnetic Laplacian on Cora-scale graphs, **sweep q — do not inherit 0.25 blindly.**
Confidence: **certain** that the two papers state different things; **uncertain** which is right at our
scale.

---

## Mechanism / math (exact formulas, tensor shapes)

Notation: `B` batch, `N` nodes, `H` heads, `L` layers, `d` head dim. Our setting is one token per node,
so token index == node index and the token->node map is the identity.

### The common form

```
score[b,h,i,j] = (q[b,h,i,:] . k[b,h,j,:]) / sqrt(d)  +  bias[b,h,i,j]
```

`bias` is `[B,H,N,N]`, float32 (cast to the attention dtype at the last moment), added **after** the
`1/sqrt(d)` scaling and **before** softmax. Confirmed in Graphormer's kernel (G6) and stated in both
GTLM (T1) and GaLA (A2).

### 1. SPD bias (hop table)

Compute from A:

```
D = all_pairs_shortest_path(A)     # [B,N,N] int, 0 on the diagonal, INF where unreachable
```

Graphormer's convention (from `algos.pyx`, verbatim logic):

```
M[i][j] = 0    if i == j
M[i][j] = 510  if A[i][j] == 0 and i != j     # pre-Floyd sentinel
... Floyd-Warshall ...
M[i][j] = 510  if M[i][j] >= 510              # post-pass: unreachable stays 510
```

Then `spatial_pos = M + 1` at collate time, so:

| SPD | table index | meaning |
|---|---|---|
| (padding) | 0 | `padding_idx`, permanently zero, no gradient |
| 0 | 1 | self-pair |
| 1 | 2 | adjacent |
| ... | ... | |
| 510 | 511 | unreachable |

Learnable part: `nn.Embedding(num_spatial, H, padding_idx=0)` -> **`[num_spatial, H]`**, `num_spatial=512`.
Lookup gives `[B,N,N,H]`, then `.permute(0,3,1,2)` -> `[B,H,N,N]`.

GTLM's variant is cleaner for our purposes and is what I would copy:

```
idx = clamp(spd - 1, 0, max_spd - 1)
b   = embedding(idx, weights)             # [B,N,N,H]
b   = b.permute(0,3,1,2) * (spd > 0)      # zero wherever spd == 0
```

Learnable part: **`[max_spd, H]`** per layer (GTLM) or shared (Graphormer). With `max_spd=8, H=32`:
256 params per layer, 4,096 for 16 layers.

**Handling unreachable pairs — three real options, all attested:**

1. **Own bucket** (Graphormer): map INF to a dedicated index and let the model learn it. Needs a table
   of size `max_spd + 1`.
2. **Zero bias** (GaLA, A3): `B_{i,j} = 0` when `d_G = INF`. Equivalent to "no opinion." Also what
   GTLM's `(spd > 0)` mask does if you encode unreachable as 0.
3. **Hard mask to -inf** (Graphormer's `spatial_pos_max`, G10): forbid attention entirely.

For masked *edge* prediction these differ a lot. Option 3 is dangerous for us: at 15% masking the SPD
matrix is computed on the *observed* graph, so a masked edge can make a pair look unreachable, and
`-inf` would make it impossible to recover. **Use option 1 or 2. Do not hard-mask.**

**Bucketing for large diameter.** Cora has N=2708 and a diameter around 19; `max_spd=8` saturates
quickly. T5's scheme (B3) is the right pattern — exact buckets for small distances, log-spaced beyond:

```
max_exact = num_buckets // 2
is_small  = d < max_exact
d_large   = max_exact + (log(d / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)).long()
bucket    = where(is_small, d, min(d_large, num_buckets - 1))
```

### 2. RRWP bias

```
M     = D^-1 A                    # row-stochastic; set 1/0 -> 0 for isolated nodes
P     = stack([I, M, M^2, ..., M^{K-1}], dim=-1)     # [B,N,N,K]
```

Learnable part: an MLP `R^K -> R^H` applied per pair. GTLM's exact widths:

```
hidden = 4 * K
proj   = Sequential(Linear(K, hidden), SiLU(), Linear(hidden, H))
```

Params per layer `= K*4K + 4K + 4K*H + H`. With `K=16, H=32`: `1088 + 2080 = 3,168`.
Output `[B,N,N,H]` -> permute -> `[B,H,N,N]`, diagonal zeroed.

Cost note: `P` is `[B,N,N,K]` dense. At N=2708, K=16, fp32 that is **~470 MB per graph**. This is the
real memory wall, not the attention matrix. Options: compute `P` sparsely, cap K, or subsample nodes.

### 3. Magnetic Laplacian bias (directed)

```
A_s      = 0.5 * (A + A^T)
d_s      = A_s.sum(-1);  d_s^{-1/2} with 0 where d_s == 0
Theta    = 2*pi*q * (A - A^T)                       # [B,N,N] real
L_N      = I - (d_s^{-1/2} A_s d_s^{-1/2}) (*) exp(i*Theta)     # Hermitian, complex
lam, V   = eigh(L_N)                                # V complex [B,N,N]
```

Keep the `m` lowest eigenpairs. GTLM then forms a basis-invariant kernel:

```
K = V diag(phi(lam)) V^dagger  in  C^{N x N x d_Mag}
phi(lam_i) = sigma(W_2 [ Linear(lam_i) (+) mean(Linear(lam)) ])   # DeepSets over eigenvalues
b_Mag^(l,h)(u,v) = [ MLP_Mag^(l)( [Re(K_uv) || Im(K_uv)] ) ]_h    # 0 when u == v
```

Learnable part: the DeepSets `phi` (input 1 -> `d_Mag`) plus `MLP_Mag: R^{2 d_Mag} -> R^H`.
118,272 params over 16 layers in GTLM with `d_Mag = 32`.

For **undirected** graphs `A = A^T`, so `Theta = 0`, `exp(i*Theta) = 1`, and `L_N` collapses to the
ordinary normalized Laplacian **for any q** (M2). Cora and the OGB molecular sets are undirected.
**The magnetic machinery buys us nothing in Phase 1 and Phase 2** — it only matters once we hit
directed data (gene regulatory networks in Phase 4). Note this is exactly where GTLM's ablation says
most of the value lives, which is a warning that their result may not transfer to our undirected
benchmarks.

### 4. Laplacian-distance bias (cheap, undirected, and unused by anyone above)

GTLM's repo has a form nobody's paper highlights, and it is trivially cheap:

```
dist = cdist(lap_eigvecs, lap_eigvecs, p=2)          # [B,N,N]
b    = dist.unsqueeze(1) * w.view(-1,1,1)            # w: [H]
```

Learnable part: **`[H]`**, i.e. 32 scalars. This is the minimum-viable spectral bias and pairs naturally
with encoder E5 (CLAUDE.md §3), which already needs the eigenvectors.

### 5. GaLA's parameter-free bias

```
B[i,j] = 1/d_G(i,j)   if i != j and d_G < inf
       = 0            otherwise
score += lambda_h * B
```

Learnable part: **nothing** in `B`; only `lambda_h`, `[H]` per layer (or a single `beta` plus a fixed
per-head profile from calibration). Range of `B` is `(0, 1]` — bounded by construction, which is why it
is safe to apply to a frozen model with no training. This boundedness is the property facts B9-B13 say
we should preserve.

### 6. Per-head vs all-head, and layer schedules — the design space

| Scheme | Learnable shape | Attested by |
|---|---|---|
| One scalar per SPD bucket, shared over heads and layers | `[max_spd]` | — (nobody does this) |
| Per-head, shared over layers | `[max_spd, H]` | Graphormer (G2, G11), T5 (B5) |
| Per-head, per-layer | `[L, max_spd, H]` | GTLM (T2) |
| Fixed shape x per-head scale, first half of layers only | `[H]` per layer, `l in [0, L/2]` | GaLA (A4, A5) |
| Fixed shape x fixed per-head slope, all layers | 0 (non-learned) | ALiBi (B1, B2) |

---

## Code we can reuse (real snippets, real signatures)

### Graphormer's `GraphAttnBias` — the canonical implementation

From `raw.githubusercontent.com/microsoft/Graphormer/main/graphormer/modules/graphormer_layers.py`:

```python
class GraphAttnBias(nn.Module):
    """Compute attention bias for each head."""

    def __init__(self, num_heads, num_atoms, num_edges, num_spatial,
                 num_edge_dis, hidden_dim, edge_type, multi_hop_max_dist, n_layers):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"], batched_data["spatial_pos"], batched_data["x"],
        )
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        ...
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset
        return graph_attn_bias
```

Two things to note. The `[:, :, 1:, 0]` and `[:, :, 0, :]` rows are the **[VNode] virtual token** —
a learned per-head "distance to the graph token." If we add a global readout token we need the same.
And the last line adds `attn_bias` a **second** time (it was already cloned in); `attn_bias` carries
`-inf` for padding, and `-inf + -inf = -inf`, so it is a deliberate re-assertion of the mask after the
additions, not a bug.

The corresponding init (note `nn.Embedding` is **normal(0, 0.02)**, not zero):

```python
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
```

### Graphormer's Floyd-Warshall and unreachable sentinel

From `graphormer/data/algos.pyx` — this is the ground truth on the 510 value:

```cython
    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyed algo
    for k in range(n):
        ...

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510
```

### Graphormer's collator — the +1 offset and the hard distance cutoff

```python
def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    ...
    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
```

### Where the bias enters attention (Graphormer's MHA)

```python
        self.scaling = self.head_dim ** -0.5      # line 52
        ...
        q *= self.scaling                          # line 145
        ...
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)
```

This settles the "relative to q.k/sqrt(d)" question from the task: the bias is added to the
**already-scaled** logits. Our bias magnitude must be compared against `q.k/sqrt(d)`, not `q.k`.

### GTLM's bias module — the best template for our `models/bias/`

From `raw.githubusercontent.com/DarioVajda/graph_model/main/src/models/bias.py`. Note the plugin
protocol (`config_key`, append to `BIAS_TYPES`) — that is exactly the swappable interface CLAUDE.md §3
asks for, applied to biases instead of encoders.

```python
class BaseBias(nn.Module):
    config_key: str = ""
    shared: bool = False

    @classmethod
    def is_enabled(cls, bias_config) -> bool:
        return bool(getattr(bias_config, cls.config_key, False))


class SPDBias(BaseBias):
    """Learnable lookup table indexed by shortest-path distance."""

    config_key = 'spd'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.max_spd = getattr(bias_config, 'max_spd', 32)
        self.weights = nn.Parameter(torch.zeros(self.max_spd, num_heads))
        # 2-D by shape but semantically an additive logit lookup (64 globally
        # shared values per head - it cannot memorize examples): exempt from the
        # trainer's shape-based weight-decay rule.
        self.weights._no_weight_decay = True

    def forward(self, *, dtype, device, spd=None, **kwargs) -> Optional[torch.Tensor]:
        if spd is None:
            return None
        non_zero = (spd > 0).unsqueeze(1)
        idx = torch.clamp(spd - 1, 0, self.max_spd - 1)
        b = F.embedding(idx, self.weights).permute(0, 3, 1, 2).to(dtype)
        return b * non_zero                                         # (B, H, N, N)


class RRWPBias(BaseBias):
    """Small MLP applied to multi-hop random-walk probability vectors."""

    config_key = 'rrwp'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        max_rw_steps = getattr(bias_config, 'max_rw_steps', 8)
        self.bias_self_node = getattr(bias_config, 'bias_self_node', False)
        hidden = 4 * max_rw_steps
        self.proj = nn.Sequential(
            nn.Linear(max_rw_steps, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, num_heads, bias=True),
        )
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)
        self.proj[2]._is_hf_initialized = True

    def forward(self, *, dtype, device, rrwp=None, **kwargs) -> Optional[torch.Tensor]:
        if rrwp is None:
            return None
        b = self.proj(rrwp).permute(0, 3, 1, 2).contiguous()      # (B, H, N, N)
        if self.bias_self_node:
            return b
        diag = torch.eye(b.shape[-1], device=device, dtype=torch.bool)
        return b.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)


class LaplacianBias(BaseBias):
    """Learnable scalar weight x pairwise L2 distance between spectral embeddings."""

    config_key = 'laplacian'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_heads) * 0.02)

    def forward(self, *, dtype, device, laplacian=None, **kwargs) -> Optional[torch.Tensor]:
        if laplacian is None:
            return None
        dist = torch.cdist(laplacian, laplacian, p=2.0)            # (B, N, N)
        return dist.unsqueeze(1) * self.weights.view(-1, 1, 1).to(dtype)
```

The intra-node / diagonal convention, factored out so it cannot drift:

```python
def finalize_node_bias(b: torch.Tensor, device, bias_self_node: bool) -> torch.Tensor:
    """``(B, N, N, H)`` -> ``(B, H, N, N)``, zeroing the intra-node diagonal unless
    ``bias_self_node``."""
    b = b.permute(0, 3, 1, 2).contiguous()                        # (B, H, N, N)
    if bias_self_node:
        return b
    diag = torch.eye(b.shape[-1], device=device, dtype=torch.bool)
    return b.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)
```

The accumulator, including the k-hop hard gate (CLAUDE.md §11's suggested memory fallback):

```python
        node_bias = None
        for module in self.bias_modules:
            b = module(dtype=dtype, device=device, num_nodes=num_nodes, spd=spd, ...)
            if b is not None:
                node_bias = b if node_bias is None else node_bias + b

        node_bias = self._apply_k_hop_gate(node_bias, k_hop_mask, dtype, device)

    def _apply_k_hop_gate(self, node_bias, k_hop_mask, dtype, device):
        """Apply -inf to positions outside the K-hop neighbourhood."""
        if self.k_hop == 0 or k_hop_mask is None:
            return node_bias
        gate = k_hop_mask.unsqueeze(1)                             # (B, 1, N, N)
        if node_bias is None:
            B, N, _ = k_hop_mask.shape
            node_bias = torch.zeros(B, self.num_heads, N, N, dtype=dtype, device=device)
        return node_bias.masked_fill(~gate, torch.finfo(dtype).min)
```

Note `torch.finfo(dtype).min`, **not** `float("-inf")`. Under bf16/fp16 autocast a true `-inf` in an
added bias produces `NaN` as soon as a whole row is masked. Copy this.

### GRIT's RRWP computation — real, tested, ready to lift

From `raw.githubusercontent.com/LiamMa/GRIT/main/grit/transform/rrwp.py`:

```python
@torch.no_grad()
def add_full_rrwp(data, walk_length=8, attr_name_abs="rrwp",
                  attr_name_rel="rrwp", add_identity=True, spd=False, **kwargs):
    device = data.edge_index.device
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(num_nodes, num_nodes))

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)          # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1)     # n x k   <- node-level PE
    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)
    ...
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)
    return data
```

The `deg_inv[deg_inv == float('inf')] = 0` line handles isolated nodes. The `abs_pe = pe.diagonal()`
line is free RWSE — the diagonal of the same tensor is the node-level random-walk structural encoding,
which feeds encoder E5.

### Magnetic Laplacian — GTLM's batched implementation

From `raw.githubusercontent.com/DarioVajda/graph_model/main/src/utils/magnetic_lap.py`:

```python
def get_magnetic_laplacian_coords(graphs, q=0.25, use_gpu=True, m=0):
    ...
    As = 0.5 * (A + A.transpose(1, 2))
    ds_diag = As.sum(dim=2)
    ds_inv_sqrt = torch.where(ds_diag > 0, 1.0 / torch.sqrt(ds_diag), 0.0)

    # theta_ij = 2pi * q * (a_ij - a_ji)
    thetas = 2 * np.pi * q * (A - A.transpose(1, 2))

    rotation = torch.exp(1j * thetas.to(torch.complex64))
    normalized_As = ds_inv_sqrt.unsqueeze(2) * As * ds_inv_sqrt.unsqueeze(1)
    eye = torch.eye(max_n, device=device).unsqueeze(0)
    L_N = eye.to(torch.complex64) - (normalized_As.to(torch.complex64) * rotation)

    try:
        eigvals, eigvecs = torch.linalg.eigh(L_N)
        if not (torch.isfinite(eigvals).all() and torch.isfinite(eigvecs).all()):
            raise RuntimeError("non-finite eigendecomposition")
    except RuntimeError:
        ev, vec = torch.linalg.eigh(L_N.cpu().to(torch.complex128))
        eigvals = ev.to(torch.float32).to(device)
        eigvecs = vec.to(torch.complex64).to(device)
```

Their comment documents two real cuSOLVER failure modes on complex64 — non-convergence (LAPACK 17) on
near-degenerate spectra, and **silent NaN eigenvectors for isolated degree-0 nodes**. Cora has isolated
nodes after edge masking. **Copy the fallback, not just the formula.**

### T5's log-spaced bucketing — for Cora-scale diameters

From HF `modeling_t5.py`:

```python
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True,
                                  num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
```

For SPD (always non-negative) drop the `bidirectional` branch and keep the log half.

### GaLA — no code exists; here is the mechanism transcribed

```python
# B: [N,N] float, parameter-free. Precompute once per graph.
#   B[i,j] = 1/spd[i,j]  for i != j and spd finite;  0 otherwise.
B = torch.zeros(N, N)
finite = (spd > 0) & (spd < INF)
B[finite] = 1.0 / spd[finite].float()

# lambda_h: [H] per layer, from calibration. Applied only for l in [0, L/2].
score = q @ k.transpose(-1, -2) / math.sqrt(d) + lam.view(1, -1, 1, 1) * B
```

Entropy calibration (label-free), transcribed from A5:

```python
# H(h) = E_{S in D_cal}[ -(1/T) sum_i sum_{j<=i} A[h,i,j] * log A[h,i,j] ]
# lambda_h = beta * sqrt( H(h) / max_{h' in layer l} H(h') )
ent = -(attn * attn.clamp_min(1e-9).log()).sum(-1).mean(-1)   # [H], causal-masked
lam = beta * torch.sqrt(ent / ent.max())
```

Gradient calibration (label-based), transcribed from A6:

```python
g = torch.ones(H, requires_grad=True)          # per-head scale on the bias, init 1.0
loss = -log_p_of_gold_label(model_with_bias(g * B))
loss.backward()                                 # ONE backward pass, weights frozen
G = g.grad                                      # [H]
s = (-G).clamp_min(0)
lam = beta * s / s.max()
```

---

## Numbers to beat / hyperparameters to copy

### Copy directly

| Hyperparameter | Value | Why / source |
|---|---|---|
| `max_spd` | **8** (GTLM) or bucket to 32 with T5 log-spacing | T5 Table 6; Cora diameter ~19 needs bucketing |
| `num_spatial` (Graphormer style) | 512, `padding_idx=0` | G8 |
| SPD table init | **zeros** | B14, B16 — bias is exactly 0 at step 0, model == base LLM |
| SPD table weight decay | **exempt** (`_no_weight_decay = True`) | B14 — 8x32 globally shared scalars cannot memorise |
| `max_rw_steps` (K) | **16** (GTLM) / 21 (GRIT primary) | T5, R5 |
| RRWP MLP | `Linear(K, 4K) -> SiLU -> Linear(4K, H)`, **output layer zero-init** | B15 |
| Magnetic `q` | 0.25 (GTLM) — but **sweep it**, see correction 13 | T5 vs M3 |
| Magnetic `d_Mag` | 32 | T5 |
| Bias LR | **5e-3** (GraphQA) or **0.01-0.04** (node classification) | T8, T9 |
| LoRA LR | **3e-5** (GraphQA) or **6e-5 - 3e-4** (node classification) | T8, T9 |
| LoRA rank / alpha | r=16, alpha=32 (GraphQA); r in {32,64}, alpha=2r (TAG) | T8, T9 |
| Bias weight decay | 0, but **then you must bound the bias form** | B12 |
| Mask fill value | `torch.finfo(dtype).min`, never `float("-inf")` | GTLM `_apply_k_hop_gate` |

The differential learning rate is the single most important item. **5e-3 / 3e-5 is a 167x ratio; 0.04 /
6e-5 is a 667x ratio.** CLAUDE.md is right that ignoring this means the bias will not converge.

### Bias magnitude budget — the numbers that decide whether this works

* `q.k/sqrt(d)` with `d_head = 64` sits at **O(1-10)** before softmax (B8).
* Bias values that *worked* in a trained frozen-LLM graph model: **0.021 to 0.309** (B9).
* Bias value that **saturated softmax outright**: **52.85** (B9).
* Max attention logit above **1e4** always diverged; loss starts deteriorating around **1e3** (B6).
* GaLA's `B` is bounded in `(0, 1]` by construction, so `lambda_h` *is* the magnitude budget.

**Working rule for us: target `|bias|` in the 0.05-0.5 band, i.e. roughly 1-10% of the logit scale.
Log `max|bias|` and `max|q.k/sqrt(d)|` every N steps and alert if the ratio exceeds ~0.5.** This is a
cheap guard and B11 shows what it costs to skip it: a 10 pp regression *below* the no-bias baseline.

**Keep the bias at most degree-1 in learned parameters.** A lookup table (SPD) is degree-1. An MLP on a
fixed feature (RRWP, magnetic) is degree-1 in the sense that matters. A product of two trainable
factors is degree-2 and grows as `k^4` — three of GTLM's runs diverged to NaN this way at
`bias_lr = 2e-2` (B10). If a bilinear form is unavoidable, **L2-normalise before the inner product**
(B13: 136.1 -> 0.1038 at k=8).

### Numbers to beat

| Benchmark | Number | Source |
|---|---|---|
| Graphormer PCQM4M-LSC valid MAE, full model | **0.1234** (GIN-VN 0.1395) | ar5iv 2106.05234 Table 1 |
| Graphormer relative gain from spatial encoding alone | **0.2276 -> 0.1427 = 37.3%** | Table 5 |
| GaLA over BFS-linearized zero-shot, semi-synth Cora | **83.3 -> 90.4 (+7.1 pp)** | Table 2 |
| GaLA over baseline, real-world PubMed | **77.5 -> 82.5 (+5.0 pp)** | Table 2 |
| GaLA overhead | ~18% wall clock (262 s vs ~225 s at n=1000) vs ~20x for CoT | Table 3 |
| GTLM bias parameter budget | **173,056 = 0.015%** of a 1B model | T6 |
| GTLM backward-compat delta on a single-node graph | **2.1e-5 +/- 8.3e-6** | T7 |
| GTLM training slowdown from custom bias | **~3x** | T10 |

That last row is the one to plan around. CLAUDE.md §11 already says "measure this before building
anything elaborate." GTLM measured 3x at ~1,000 tokens. Cora at N=2708 is 7.3x more attention entries
than that.

### A cheap, high-value ablation ordering

Given correction 9, I would order the bias ablation by evidence strength rather than by CLAUDE.md's
listing order:

1. **No bias** (the floor — B11 shows a bad bias lands *below* it, so this number is load-bearing).
2. **GaLA-style `1/SPD` with a single scalar `lambda`** — zero learnable structure, bounded in `(0,1]`,
   cannot blow up. Fastest possible signal on whether biasing helps at all.
3. **SPD lookup table, per-head, zero-init** (Graphormer/GTLM form).
4. **Laplacian-distance bias** — `[H]` params, reuses E5's eigenvectors, undirected-friendly.
5. **RRWP bias** — subsumes SPD in theory (R2); check whether it subsumes it in practice.
6. Magnetic Laplacian **only when directed data arrives** (it degenerates to the ordinary Laplacian on
   Cora and the molecular sets).

Steps 1-2 are a day's work and settle the central question before we invest in learned bias tables.

---

## Open questions

1. **Does an SPD bias help at all when there is one token per node and no verbalization?** Every result
   we have is from either a from-scratch graph transformer (Graphormer: big win) or a frozen LLM eating
   *serialized text* (GTLM, GaLA: mixed). Nobody has measured our exact configuration. This is
   genuinely unknown and is arguably the paper.
2. **Is the SPD bias redundant with an adjacency-row encoder?** Our encoder E1 already feeds row `A_i`
   into each token. `A_i` *is* the 1-hop indicator. Graphormer's node input carries no adjacency at all,
   so its spatial encoding is the only source of structure — which may fully explain the 37% gain and
   mean it does not transfer to us. Worth an explicit ablation: bias-on/off crossed with encoder E1
   (row) vs an encoder that hides the row.
3. **What layer schedule?** GaLA says first half with no ablation (correction 3); GTLM and Graphormer
   say all layers. We decode a graph, not text, so GaLA's "leave later layers for language" rationale
   may not apply. Sweep `{all, first half, last half, every other}`.
4. **Per-layer vs layer-shared bias tables?** Graphormer and T5 share one table across layers; GTLM
   uses `L` separate tables. Nobody ablates this. At `max_spd=8, H=32, L=16` the difference is 256 vs
   4,096 params — cheap enough to test both.
5. **How do we compute SPD when 15% of entries are masked?** SPD must come from the *observed* graph or
   we leak the label. But masking edges inflates distances and can create spurious unreachability — the
   bias then actively tells the model "these are far apart" about exactly the pairs it must predict as
   edges. This is a leakage-vs-signal tension that none of the source papers face (none do masked edge
   prediction). **This is the highest-risk unknown in the whole bias design.** Options to test: compute
   SPD on the observed graph and accept the distortion; recompute per masking draw; or exclude masked
   pairs from the bias entirely (bias 0 on masked entries).
6. **Memory for RRWP.** `[B,N,N,K]` at N=2708, K=16, fp32 is ~470 MB per graph — larger than the
   attention matrix. Needs a sparse formulation or node subsampling before Phase 3.
7. **`beta` / `lambda_h` scale for our task.** GaLA never publishes a value. We have the magnitude band
   (0.05-0.5) from GTLM's diagnostics but no task-matched number.
8. **Does entropy-based head selection mean anything when the LLM's input is not text?** GaLA's entropy
   heuristic rests on Clark et al. 2019's finding that low-entropy heads do syntax. Our tokens are
   adjacency-row projections, not words. The heads' entropy profile on our inputs may bear no relation
   to their linguistic role. The gradient variant has no such dependency and is probably the safer one
   for us.
9. **Unverified**: whether GaLA has released code. I found none.

---

## Sources fetched

Primary papers (full text read, not just abstract):

* `https://arxiv.org/abs/2106.05234` and `https://ar5iv.labs.arxiv.org/html/2106.05234` — Graphormer,
  Ying et al., NeurIPS 2021. Eqs. 5/6/7, Table 5 ablation, Appendix GNN-special-case proof.
* `https://arxiv.org/abs/2606.15633` and `https://arxiv.org/html/2606.15633v2` — Loveland, Trivedi,
  Weinstein, Huang, Koutra, "Formalizing and Mitigating Structural Distortion in LLM Attention for
  Graph Reasoning" (GaLA), KDD '26. §6.1, §6.2, §6.3, §7, Appendix C/D, Tables 1-3.
* `https://arxiv.org/abs/2605.10247` and `https://arxiv.org/html/2605.10247v1` — Vajda, GTLM. §3.2,
  §4.5 Table 5, §5, Appendix A/B.2/C Table 6/D Table 7/E Table 8.
* `https://arxiv.org/abs/2305.17589` and `https://ar5iv.labs.arxiv.org/html/2305.17589` — GRIT, Ma et
  al., ICML 2023. RRWP definition, Prop 3.1, Eq. 2/5, Table 6.
* `https://ar5iv.labs.arxiv.org/html/2302.00049` — Geisler et al., "Transformers Meet Directed Graphs,"
  ICML 2023. Magnetic Laplacian, role of q.
* `https://ar5iv.labs.arxiv.org/html/2106.03893` — SAN, Kreuzer et al., NeurIPS 2021. LPE, gamma.
* `https://ar5iv.labs.arxiv.org/html/2205.12454` — GraphGPS, Rampasek et al., NeurIPS 2022. PE/SE
  taxonomy.
* `https://ar5iv.labs.arxiv.org/html/2108.12409` — ALiBi, Press et al., ICLR 2022. Slopes.
* `https://ar5iv.labs.arxiv.org/html/2309.14322` — Wortsman et al., "Small-scale proxies for
  large-scale Transformer training instabilities." §3.1.1, §3.3.

Raw source code (downloaded and read in full):

* `raw.githubusercontent.com/microsoft/Graphormer/main/graphormer/modules/graphormer_layers.py`
* `raw.githubusercontent.com/microsoft/Graphormer/main/graphormer/data/algos.pyx`
* `raw.githubusercontent.com/microsoft/Graphormer/main/graphormer/data/wrapper.py`
* `raw.githubusercontent.com/microsoft/Graphormer/main/graphormer/data/collator.py`
* `raw.githubusercontent.com/microsoft/Graphormer/main/graphormer/modules/multihead_attention.py`
* `raw.githubusercontent.com/microsoft/Graphormer/main/graphormer/modules/graphormer_graph_encoder.py`
* `raw.githubusercontent.com/microsoft/Graphormer/main/graphormer/models/graphormer.py`
* `raw.githubusercontent.com/microsoft/Graphormer/main/graphormer/tasks/graph_prediction.py`
* `raw.githubusercontent.com/DarioVajda/graph_model/main/src/models/bias.py` (1,720 lines)
* `raw.githubusercontent.com/DarioVajda/graph_model/main/src/utils/magnetic_lap.py`
* `raw.githubusercontent.com/DarioVajda/graph_model/main/src/models/biases/MIXED_BIAS.md` (§5.7)
* `raw.githubusercontent.com/DarioVajda/graph_model/main/src/models/biases/LINEAR_BIAS.md`
* `raw.githubusercontent.com/DarioVajda/graph_model/main/src/experiments/bias_experiments/landmark/diagnose_scale.py`
* `raw.githubusercontent.com/LiamMa/GRIT/main/grit/transform/rrwp.py`
* `raw.githubusercontent.com/LiamMa/GRIT/main/grit/layer/grit_layer.py`
* `raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/t5/modeling_t5.py`

Searched, nothing found: a public code release for GaLA.
