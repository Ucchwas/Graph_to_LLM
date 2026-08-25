# GTLM — "Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning"

Deep-dive reference for the closest existing work to our graph-in/graph-out project.
Research date: **2026-08-25**. Everything below was fetched and read directly; nothing is from memory.

- Paper: **arXiv:2605.10247** (v1), Dario Vajda, Faculty of Computer and Information Science, University of Ljubljana. Submitted **11 May 2026**, primary class **cs.LG**.
- Repo: **github.com/DarioVajda/graph_model** (branch `main`, 1,135 files, tree fetched via GitHub API, `truncated: false`).
- Status: appears to be under **NeurIPS** review — the repo contains `src/models/flex_attn/REBUTTAL.md`, a rebuttal-planning document dated internally to 2026-07-26.

**The single most important thing on this page:** the paper's headline limitation (FlashAttention-incompatible → eager-only → O(N²) → ~3× slowdown), which CLAUDE.md §11 tells us to design around, **has been solved in the repo since submission** via a `torch.nn.attention.flex_attention` backend. See §2.7 and the Corrections section. Do not copy the eager-only path.

---

## 1. Verified facts (with source next to each)

### 1.1 Existence and identity

| Claim | Verdict | Source |
|---|---|---|
| Paper exists at arXiv 2605.10247 | **CONFIRMED** | fetched `arxiv.org/abs/2605.10247` — title, author, abstract all match CLAUDE.md |
| Title is exactly *"Teaching LLMs to See Graphs: Unifying Text and Structural Reasoning"* | **CONFIRMED** | `arxiv.org/abs/2605.10247` |
| Author is Dario Vajda (sole author) | **CONFIRMED** | abstract page + BibTeX in repo README (`@misc{vajda2026teachingllmsgraphsunifying}`) |
| Repo `github.com/DarioVajda/graph_model` exists and is the paper's code | **CONFIRMED** | repo README links `arXiv:2605.10247` and states "This repository contains all of the code used for this research project" |
| Model name is GTLM = Graph Transformer Language Model | **CONFIRMED** | abstract |
| Base model is Llama-3.2-1B | **CONFIRMED** | `src/experiments/backward_compatibility/__main__.py` line `model_name = "meta-llama/Llama-3.2-1B"`; repo README training example `--model_name meta-llama/Llama-3.2-1B`; scaling to Llama-3.2-3B and Llama-3.1-8B (paper Table 4) |

**Verbatim abstract** (`arxiv.org/abs/2605.10247`):

> Using Large Language Models (LLMs) to process graph-structured data is an active research area, yet current state-of-the-art approaches typically rely on multi-step pipelines with Graph Neural Network (GNN) encoders that compress rich textual attributes into solitary tokens, creating a significant semantic bottleneck. In this paper, we introduce the Graph Transformer Language Model (GTLM), a novel architecture that enables pretrained LLMs to natively process graph topologies while entirely eliminating this compressive bottleneck. GTLM is exceptionally parameter-efficient: by injecting graph-aware attention biases directly into the LLM's attention modules, it introduces only 0.015% additional parameters relative to the base model. We theoretically prove that our bidirectional attention prefix preserves node permutation equivariance while maintaining exact backward compatibility with the pretrained base model. Extensive evaluations demonstrate that a 1B-parameter GTLM matches or exceeds the performance of 7B-parameter state-of-the-art models on standard Text-Attributed Graph benchmarks, while significantly surpassing baselines on GraphQA. Finally, we demonstrate that GTLM attention heads implicitly learn to simulate message passing, explaining its superior performance on algorithmic tasks. This paradigm shift enables true algorithmic reasoning within LLMs and provides a scalable foundation for next-generation GraphRAG and relational deep learning.

### 1.2 The bias parameter count — independently recomputed from source, exact match

CLAUDE.md claims **173,056 bias parameters**. This is not just "as reported" — I recomputed it from the actual module constructors in `src/models/bias.py` using Llama-3.2-1B's architecture (`num_hidden_layers=16`, `num_attention_heads=32`, `head_dim=64`, from `src/models/configs/gtlm_llama_v0_1b.json`) and the paper's Table 6 hyperparameters:

```
SPD      per layer = max_spd * H           =  8 * 32                    =    256
RRWP     per layer = (K*4K + 4K) + (4K*H + H)
                   = (16*64 + 64) + (64*32 + 32)                        =  3,168
Magnetic per layer = lambda_lin(1→64)  128
                   + deep_set(128→32)  4,128
                   + proj[0](64→32)    2,080
                   + proj[2](32→32)    1,056                            =  7,392

x 16 layers:  SPD 4,096 | RRWP 50,688 | Magnetic 118,272   TOTAL = 173,056
```

The paper's Appendix reports **exactly** 4,096 / 50,688 / 118,272 / 173,056. This is a three-way agreement between the paper text, the repo source, and an independent recomputation — and it **retroactively confirms** `max_spd = 8`, `max_rw_steps = 16`, and `magnetic_dim = 32`, because no other values reproduce those counts.

### 1.3 Hyperparameters — paper vs. repo defaults, both checked

| Knob | Paper (Table 6/7/8) | Repo source | Verdict |
|---|---|---|---|
| SPD max distance | 8 | `src/experiments/tag_benchmarks/config.py:171` `max_spd: int = 8`; `graphqa/config.py:176` same | **CONFIRMED** |
| RRWP max steps K | 16 | `tag_benchmarks/config.py:173` `max_rw_steps: int = 16`; graphqa same | **CONFIRMED** |
| Magnetic Laplacian q | 0.25 | `config.py:176` `magnetic_q: float = 0.25`; `src/utils/magnetic_lap.py` `def get_magnetic_laplacian_coords(graphs, q=0.25, ...)` | **CONFIRMED** |
| Magnetic dim | 32 | `config.py:175` `magnetic_dim: int = 32  # model bias-MLP hidden width` | **CONFIRMED** (but see Correction C4 — it is a *hidden width*, not a spectral dimension) |
| GraphQA base/LoRA LR | 3e-5 | `graphqa/config.py:259` `lr: float = 3e-5` | **CONFIRMED** |
| GraphQA bias LR | 5e-3 | `graphqa/config.py:260` `bias_lr: float = 5e-3` | **CONFIRMED** |
| GraphQA LoRA rank | **16**, alpha 32 | `graphqa/config.py:252` `lora_r: int = 16`; `lora_alpha = lora_r * 2` | **CONFIRMED — contradicts CLAUDE.md, see C1** |
| TAG base LR | {6e-5, 2e-4, 3e-4} | swept | **CONFIRMED** |
| TAG bias LR | {0.01, 0.04} | repo default `bias_lr: float = 5e-2` (`tag_benchmarks/config.py:215`) | paper CONFIRMED; repo default is higher |
| TAG LoRA rank | {32, 64}, alpha = 2r | `tag_benchmarks/config.py:207` `lora_r: int = 32` | **CONFIRMED** |
| LoRA dropout | not in paper | `lora_dropout: float = 0.05` | repo only |
| LoRA target modules | not in paper | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` (`src/train/model.py`) | repo only — **all 7 projections** |
| k_hop | 0 for every paper result | `k_hop: int = 0  # 0 disables the mask (the historical setting)` | **CONFIRMED** |
| Optimizer schedule | not in paper | `lr_scheduler_type="cosine_with_min_lr"`, `min_lr = lr/10`, `warmup_steps = total_steps//10`, `weight_decay=0.1` | repo only (`src/train/trainer.py`) |
| Ego-subgraph neighbours | Cora 60, PubMed 30, ogbn-arxiv 60, Reddit 30 | — | paper Table 8 |

### 1.4 Numerical verification claims — cross-checked against raw experiment output

CLAUDE.md says GTLM "verified numerically (difference ~2×10⁻⁵, floating-point noise)". **Confirmed, and tighter than stated.**

The paper (Appendix B.2) reports backward compatibility max logit difference **2.1×10⁻⁵ ± 8.3×10⁻⁶** over five trials. The repo's `src/experiments/backward_compatibility/__main__.py` ends with a docstring containing the five raw random-string runs:

```
Maximum absolute difference in logits: 0.00001192
Maximum absolute difference in logits: 0.00001240
Maximum absolute difference in logits: 0.00002480
Maximum absolute difference in logits: 0.00003019
Maximum absolute difference in logits: 0.00002575
```
mean = 2.101e-5, sample std = 8.34e-6 → **exactly reproduces the paper's 2.1e-5 ± 8.3e-6.**

Permutation equivariance: paper reports **2.77×10⁻⁵ ± 2.87×10⁻⁶**. `src/experiments/permutation_equivariance/__main__.py` docstring:

```
3.075599670410156e-05
2.47955322265625e-05
2.5153160095214844e-05
2.7298927307128906e-05
3.063678741455078e-05
```
mean = 2.771e-5, sample std = 2.87e-6 → **exact match.** The paper's numbers are honestly reported.

### 1.5 Limitations — verbatim from the paper

> A primary limitation of GTLM is its current computational scaling with respect to sequence length. The injection of custom structural attention biases into the transformer layers is currently incompatible with hardware-accelerated exact attention kernels, such as FlashAttention (Dao et al., 2022). Consequently, our implementation must fall back to standard 𝒪(N²) attention calculation. Furthermore, the bidirectional attention mask required for GTLM's structural awareness natively incurs a training overhead compared to standard causal masking. Combined, these factors result in approximately 3× longer training duration than an equivalent baseline LLM. While this computational overhead is highly manageable for the sequence lengths evaluated in our experiments (approximately 1,000 tokens), scaling training to extended context windows (e.g., ≥4096 tokens) would incur significant memory and time costs. We note that **this limitation is an implementation artifact rather than a fundamental architectural constraint**, and it could be resolved by writing specialized CUDA attention kernels that support our structural biases.
>
> Secondly, the core architectural strength of GTLM—processing the full, uncompressed textual attributes of every node—inherently increases sequence lengths. [...] it limits the model's scalability to massively large graphs or datasets with exceptionally long per-node documents. Future work could explore sparse attention mechanisms or more advanced sampling methods to alleviate these context constraints.

Hardware: **"single NVIDIA A100 or H100 GPU (80GB VRAM)"**. Training times: GraphQA standard 10 min, GraphQA incidence 40 min, TAG 6–8 h (H100), ogbn-arxiv ~20 h. 3B ≈ 2× slowdown, 8B ≈ 3× slowdown. Sequence length ~1,000 tokens; ≥4,096 flagged as problematic. **All matches CLAUDE.md.**

### 1.6 Future Work — verbatim, confirms CLAUDE.md's claim

> **K-hop Attention Mask.** To mitigate the quadratic scaling bottleneck of processing full node sequences, future work could enforce topological sparsity via a K-hop attention mask. Given our empirical observation that ego subgraphs of 60 neighbors are sufficient for state-of-the-art performance, masking out distant nodes would drastically reduce the memory footprint and computational complexity from 𝒪(N²) to a nearly linear scale with respect to graph size.
>
> **Autoregressive Graph Generation.** A compelling future direction is extending GTLM beyond predictive tasks to full Text-Attributed Graph generation. This could be achieved by coupling GTLM with an external graph-generation module, or natively training it to emit structural topology. **By introducing control tokens (e.g., `<ADD_EDGE>`), the model could alternate between generating textual attributes and edge connections.** This unified capability would unlock novel applications such as knowledge graph synthesis and complex molecular design.

CLAUDE.md §9's claim — "their Future Work section explicitly names autoregressive graph generation with `<ADD_EDGE>` control tokens as an open problem. That is our project." — is **CONFIRMED verbatim.** This is a strong, quotable positioning sentence for our writeup. Note the nuance: they frame it as *text-attributed graph* generation alternating text and edges, whereas we do pure structure in / structure out with no text at all. Our framing is a strict subset in modality but goes further in one respect: they only propose it, and they propose it as a *generation* task; we also do masked (BERT-style) edge prediction, which they do not mention at all.

---

## 2. Mechanism / math (exact formulas, tensor shapes)

### 2.1 Where the bias enters

Paper §3.2, verbatim structure:

```
A_{i,j}^{(l,h)} = (Q_i^{(l,h)} · K_j^{(l,h)}ᵀ) / sqrt(d_head)
                + b_SPD^{(l,h)}(u,v)
                + b_RRWP^{(l,h)}(u,v)
                + b_Mag^{(l,h)}(u,v)
```

- **Pre-softmax logits.** Added to the scaled dot product before softmax.
- **Per-layer AND per-head.** Both `l` and `h` index the bias. Not shared across layers, not shared across heads. (CLAUDE.md §9's GaLA note says "our bias table should be per-head, not shared" — GTLM already is, in both axes.)
- **Node-level, then broadcast to tokens.** `i` is a token, `u` is the node token `i` belongs to. The bias is computed at node granularity `(B, H, N, N)` and lifted to token granularity `(B, H, L, L)` by gathering with a `node_ids: (B, L)` map. This is the single most important structural idea to steal — see §2.6.
- Additivity: the three families are summed, so adding a fourth costs nothing architecturally.

Code confirming the entry point (`src/models/dispatch.py`):

```python
def graph_attention_dispatch(impl, module, query, key, value,
                             structural_mask,   # (B, 1, q, kv) additive float
                             token_soft_bias,   # (B, H, q, kv) additive float, or None
                             scaling, dropout):
    if impl == "eager":
        attn_mask = structural_mask
        if token_soft_bias is not None:
            attn_mask = attn_mask + token_soft_bias              # (B, H, q, kv)
        return eager_attention_forward(
            module, query, key, value, attn_mask, scaling=scaling, dropout=dropout
        )
```

The hard mask (`0` / `finfo.min`) and the soft bias are **summed into one additive tensor** and handed to HuggingFace's own `eager_attention_forward`. They never reimplement softmax. That is the cleanest possible integration and we should copy it exactly.

### 2.2 SPD bias — learned lookup on shortest-path distance

Paper:
```
b_SPD^{(l,h)}(u,v) = b_spd^{(l,h)}(D_hops(u,v)),   b_spd ∈ R^{L × H × max_spd},   b_SPD^{(l,h)}(u,u) = 0
```

Real code (`src/models/bias.py`):

```python
class SPDBias(BaseBias):
    """Learnable lookup table indexed by shortest-path distance."""
    config_key = 'spd'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.max_spd = getattr(bias_config, 'max_spd', 32)
        self.weights = nn.Parameter(torch.zeros(self.max_spd, num_heads))
        # 2-D by shape but semantically an additive logit lookup (64 globally
        # shared values per head — it cannot memorize examples): exempt from the
        # trainer's shape-based weight-decay rule.
        self.weights._no_weight_decay = True

    def forward(self, *, dtype, device, spd=None, **kwargs):
        if spd is None:
            return None
        non_zero = (spd > 0).unsqueeze(1)
        idx = torch.clamp(spd - 1, 0, self.max_spd - 1)
        b = F.embedding(idx, self.weights).permute(0, 3, 1, 2).to(dtype)
        return b * non_zero                                         # (B, H, N, N)
```

Details that matter:
- **Zero-initialized** (`torch.zeros`), so at step 0 the bias is exactly 0 and the model *is* the base LLM. Everything trains up from the pretrained function rather than perturbing it randomly. This is the deep reason the differential LR is needed (§2.8).
- Index is `clamp(spd - 1, 0, max_spd - 1)`, i.e. distance 1 → row 0, distance ≥ 8 → row 7. **There is no table row for self-distance 0**; instead `* (spd > 0)` zeroes it. A repo comment explicitly warns that adding a self-row is a checkpoint-breaking change.
- Unreachable pairs: whatever the SPD feature encodes as its "infinite" value gets clamped into the last bucket. (Worth checking in their dataset code if we care — for us, disconnected components in Cora exist.)
- `_no_weight_decay = True` — the trainer decides decay by *shape*, not name, and this opts a 2-D parameter out.

### 2.3 RRWP bias — K-step random-walk probabilities through an MLP

Paper (Appendix A.1), with `M = D⁻¹A`:
```
P_{u,v} = [I, M, M², ..., M^{K-1}]_{u,v} ∈ R^K

b_RRWP^{(l,h)}(u,v) = 0                                  if u = v
                    = [MLP_RRWP^{(l)}(P_{u,v})]_h        if u ≠ v
MLP_RRWP^{(l)} : R^K → R^{n_heads}
```

Real code (`src/models/bias.py`):

```python
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

    def forward(self, *, dtype, device, rrwp=None, **kwargs):
        if rrwp is None:
            return None
        b = self.proj(rrwp).permute(0, 3, 1, 2).contiguous()      # (B, H, N, N)
        if self.bias_self_node:
            return b
        diag = torch.eye(b.shape[-1], device=device, dtype=torch.bool)
        return b.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)
```

- Hidden width is **4K** (= 64 at K=16). SiLU. Output layer **zero-initialized** → bias exactly 0 at step 0.
- `_is_hf_initialized = True` prevents `post_init()` from re-randomizing the zeroed layer. **This is a real HF footgun: without it, `PreTrainedModel.post_init()` / `_init_weights` will overwrite your careful zero-init.** Copy this.
- The RRWP *feature* is computed offline. `src/utils/rrwp.py` `compute_rrwp(graphs, max_distance=8, use_gpu=True)`: builds padded `A [B,N,N]`, adds self-loops to sink nodes (out-degree 0) to avoid division by zero, sets padded-node degrees to 1.0 (safe because A is 0 there), forms `M = A / out_degrees`, then iterates `current_power = bmm(current_power, M)` writing channel `d`, with **channel 0 = identity**. Output `(B, N, N, K)` float. Naive O(K·N³) dense — fine for N≤2048, will need sparse for Cora-full.

### 2.4 Magnetic Laplacian bias — directed spectral

Paper (Appendix A.2):
```
H^(q)   = A_s ⊙ exp(i·Θ^(q))
L_N^(q) = I − (D_s^{-1/2} A_s D_s^{-1/2}) ⊙ exp(i·Θ^(q))

φ(λ_i)  = σ( W_2 [ Linear(λ_i) ⊕ mean(Linear(λ)) ] )          # permutation-invariant Deep Sets over eigenvalues
K       = V diag(φ(λ)) V†   ∈ C^{N×N×d_Mag}

b_Mag^{(l,h)}(u,v) = 0                                                   if u = v
                   = [ MLP_Mag^{(l)}( [Re(K_{u,v}) ⊕ Im(K_{u,v})] ) ]_h  if u ≠ v
```

Feature computation (`src/utils/magnetic_lap.py`, verbatim core):

```python
As = 0.5 * (A + A.transpose(1, 2))                       # symmetrized adjacency
ds_diag = As.sum(dim=2)
ds_inv_sqrt = torch.where(ds_diag > 0, 1.0 / torch.sqrt(ds_diag), 0.0)

# theta_ij = 2*pi*q*(a_ij - a_ji)          <-- the direction signal
thetas = 2 * np.pi * q * (A - A.transpose(1, 2))
rotation = torch.exp(1j * thetas.to(torch.complex64))

normalized_As = ds_inv_sqrt.unsqueeze(2) * As * ds_inv_sqrt.unsqueeze(1)
eye = torch.eye(max_n, device=device).unsqueeze(0)
L_N = eye.to(torch.complex64) - (normalized_As.to(torch.complex64) * rotation)

eigvals, eigvecs = torch.linalg.eigh(L_N)                # Hermitian
V_final = torch.stack([eigvecs.real, eigvecs.imag], dim=-1)   # (B, N, M, 2)
```

Two failure modes they hit and handle — **copy the fallback, we will hit both on Cora**:
> The single-precision (complex64) GPU driver has two rare failure modes on a magnetic Laplacian, both pure solver-stability artifacts (the matrix is genuinely Hermitian and correctly built): (a) cuSOLVER raises a non-convergence error (LAPACK code 17) on a near-degenerate spectrum; and (b) it returns NaN eigenvectors — silently, no exception — for the decoupled trivial block produced by an isolated (degree-0) node. In both cases CPU LAPACK in double precision (complex128) is robust, so we retry that decomposition there.

Note `q = 0.25` means `theta = pi/2 * (a_ij - a_ji)`: for an undirected edge `theta = 0` (rotation = 1, reduces to the ordinary normalized Laplacian); for a one-way edge `theta = ±pi/2` (rotation = ±i, pure imaginary). **For an undirected graph like Cora, the magnetic Laplacian degenerates exactly to the normalized symmetric Laplacian and the imaginary part is identically zero.** That halves its value for us — see Open Questions.

Model side (`src/models/bias.py`, `MagneticBias.__init__`):

```python
magnetic_dim = getattr(bias_config, 'magnetic_dim', 32)
self.lambda_lin = nn.Linear(1, head_dim, bias=True)                 # eigenvalue → head_dim
self.deep_set = nn.Sequential(
    nn.Linear(head_dim * 2, magnetic_dim, bias=True), nn.SiLU(),    # [h_i ‖ mean(h)] → phi
)
self.proj = nn.Sequential(
    nn.Linear(magnetic_dim * 2, magnetic_dim, bias=True), nn.SiLU(),
    nn.Linear(magnetic_dim, num_heads, bias=True),
)
nn.init.zeros_(self.proj[2].weight); nn.init.zeros_(self.proj[2].bias)
self.proj[2]._is_hf_initialized = True
```

The Deep Sets step over eigenvalues (`phi = deep_set([h_i ‖ mean_over_valid(h)])`) is what makes the spectral encoding **basis-invariant** — it does not depend on the arbitrary sign/phase of individual eigenvectors, which is the classic failure of naive Laplacian-eigenvector positional encodings. **This is directly relevant to CLAUDE.md §3 E5 (row + Laplacian eigenvectors): a raw eigenvector concatenation is sign-ambiguous and will not be permutation-equivariant. The Deep Sets fold is the fix.**

A notable optimization they document (`_folded_spectral`), worth stealing if we go this route: because `proj[0]` is linear, project `phi` **before** the N² einsums instead of concatenating `Re(K) ‖ Im(K)` afterward, halving the largest per-layer intermediate:

```python
W1, b1 = self.proj[0].weight, self.proj[0].bias        # (out, 2m), (out)
m = W1.shape[1] // 2                                    # split at HALF THE INPUT width
phiR = phi @ W1[:, :m].T
phiI = phi @ W1[:, m:].T
return (
    torch.einsum('bil,bjl,blk->bijk', V_real, V_real, phiR)
  + torch.einsum('bil,bjl,blk->bijk', V_imag, V_imag, phiR)
  + torch.einsum('bil,bjl,blk->bijk', V_imag, V_real, phiI)
  - torch.einsum('bil,bjl,blk->bijk', V_real, V_imag, phiI)
) + b1                                                  # (B, N, N, m)
```

### 2.5 The intra-node zero bias trick

All three families are **forced to exactly 0 when u = v**:

```python
def finalize_node_bias(b, device, bias_self_node):
    """(B, N, N, H) → (B, H, N, N), zeroing the intra-node diagonal unless bias_self_node."""
    b = b.permute(0, 3, 1, 2).contiguous()                        # (B, H, N, N)
    if bias_self_node:
        return b
    diag = torch.eye(b.shape[-1], device=device, dtype=torch.bool)
    return b.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)
```

**Why it preserves base-LLM behaviour (Property 2, Appendix B.1):**
> For any graph consisting of a single node (representing a standard text sequence), the GTLM forward pass is mathematically identical to that of the unmodified base LLM.

Proof sketch: with N = 1, every token pair is intra-node, so every bias term is 0 by construction; the structural mask degenerates to plain causal; and per-node RoPE reset degenerates to `arange(len)`, i.e. ordinary RoPE. Therefore the computation *is* stock Llama. Verified numerically at 2.1e-5 ± 8.3e-6 (fp32 accumulation noise). This is what makes the whole thing safe: pretrained capability is not merely "mostly preserved", it is **provably exactly preserved on the pretraining data distribution**, and the graph machinery only activates once N > 1.

**Critical translation to our project.** In GTLM a node is a *span of text tokens*, so "intra-node" is a block on the diagonal. In our design a node is **exactly one token**, so intra-node pairs are exactly the matrix diagonal `i = i`. The trick reduces to "the diagonal of the bias must be 0". But the *guarantee* it buys GTLM (single-node graph ⇒ base LLM) is **worthless to us**, because we never feed a single-node graph and we never feed text at all. We should still zero the diagonal — for a different reason: it removes a free per-head self-attention-strength parameter that would otherwise let the model trivially sharpen or flatten every row's softmax, which is a confound in the "does the graph do anything" ablation (CLAUDE.md §8).

They expose `bias_self_node: bool = False` to flip this, and the config validator refuses to combine it with SPD because SPDBias has no table row for distance 0.

### 2.6 The node→token expansion (the reusable core idea)

`src/models/structural_mask.py`, verbatim:

```python
def expand_node_to_token_bias(
    node_bias: torch.Tensor,   # (B, H, N, N)
    node_ids:  torch.Tensor,   # (B, full_seq_len)
    q_len:     int,
    kv_len:    int,
) -> torch.Tensor:             # (B, H, q_len, kv_len)
    """Lift a node-level bias to token level: token pair (q, k) inherits the bias
    of node pair ``(node[q], node[k])``."""
    B, H = node_bias.shape[0], node_bias.shape[1]
    device = node_ids.device
    node_ids_q  = node_ids[:, -q_len:]
    node_ids_kv = node_ids[:, -kv_len:]
    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
    h_idx = torch.arange(H, device=device).view(1, H, 1, 1)
    q_idx = node_ids_q.view(B, 1, q_len, 1)
    k_idx = node_ids_kv.view(B, 1, 1, kv_len)
    return node_bias[b_idx, h_idx, q_idx, k_idx]                  # (B, H, q, kv)
```

**For us this function is the identity** (one token per node ⇒ `node_ids = arange(N)` ⇒ `L = N`). That is a genuine simplification: our bias tensor is already `(B, H, N, N)` = `(B, H, L, L)` and needs no gather. We inherit their math without their indexing machinery.

### 2.7 The structural mask — bidirectional prefix, causal suffix

`src/models/structural_mask.py`, verbatim:

```python
def build_dense_structural_mask(node_ids, prompt_node, pad_mask, k_hop_mask,
                                k_hop, q_len, kv_len, dtype, device) -> torch.Tensor:
    B = node_ids.shape[0]
    node_q = node_ids[:, -q_len:]
    node_k = node_ids[:, -kv_len:]
    prompt = prompt_node.view(B, 1).to(node_ids.device)

    q_pos = torch.arange(kv_len - q_len, kv_len, device=device).view(q_len, 1)
    k_pos = torch.arange(kv_len, device=device).view(1, kv_len)

    causal      = (k_pos <= q_pos)                                 # (q, kv)
    is_prefix_q = (node_q != prompt).unsqueeze(2)                  # (B, q, 1)
    is_prefix_k = (node_k != prompt).unsqueeze(1)                  # (B, 1, kv)
    both_prefix = is_prefix_q & is_prefix_k                        # (B, q, kv)

    allowed = causal.unsqueeze(0) | both_prefix                    # <-- the relaxation

    pad_k = pad_mask[:, -kv_len:].bool().unsqueeze(1)
    allowed = allowed & pad_k

    if k_hop > 0 and k_hop_mask is not None:
        b_idx = torch.arange(B, device=device).view(B, 1, 1)
        nq = node_q.view(B, q_len, 1); nk = node_k.view(B, 1, kv_len)
        allowed = allowed & k_hop_mask[b_idx, nq, nk]

    diag = (k_pos == q_pos).unsqueeze(0)                           # (1, q, kv) NaN guard
    allowed = allowed | diag

    min_val = torch.finfo(dtype).min
    return torch.where(allowed,
                       torch.zeros((), dtype=dtype, device=device),
                       torch.full((), min_val, dtype=dtype, device=device)
                      ).unsqueeze(1)                               # (B, 1, q, kv)
```

Points worth noting:
1. `allowed = causal | both_prefix` — one OR turns a causal decoder into a **prefix-LM**. Everything that is not the prompt node attends bidirectionally; the prompt node stays causal so text generation still works.
2. `allowed = allowed | diag` — **always leave the diagonal open as a NaN guard.** A fully-masked softmax row is NaN, which propagates silently through the whole model. This is a two-line defence we must copy; with a k-hop mask an isolated node would otherwise produce a NaN row.
3. Mask uses `torch.finfo(dtype).min`, **not `-inf`** — `-inf` plus a finite bias can produce NaN in bf16/fp16 accumulation; `finfo.min` is safe under addition with a bounded bias.
4. Shape is `(B, 1, q, kv)` — head-broadcast — while the soft bias is `(B, H, q, kv)`. They add and broadcast.

**For our project the prefix relaxation is simpler still:** there is no prompt node, so the mask is fully bidirectional and `allowed` is just `pad_k | diag`. In modern transformers there is now a first-class way to do this without touching masks at all: `config.is_causal = False`, or `model(**inputs, is_causal=False)` per call (see §5).

### 2.8 RoPE reset per node — and the surprising consequence for us

Paper §3.2: *"Position IDs for RoPE are counted from zero onwards inside of each node individually."*

**Where it actually happens: in the data collator, not the model.** `src/utils/text_graph_collator_v2.py`:

```python
def _pack_one(self, item: TextGraph) -> dict:
    """Build packed token/position/node tensors for one graph (prompt last)."""
    graph_input_ids = [torch.as_tensor(ids, dtype=torch.long) for ids in item['input_ids']]
    prompt_idx = int(item['prompt_node'])
    order = [j for j in range(len(graph_input_ids)) if j != prompt_idx] + [prompt_idx]

    base_offset = self._node_base_offsets(item, graph_input_ids, prompt_idx)

    tokens    = torch.cat([graph_input_ids[j] for j in order])
    positions = torch.cat([base_offset[j] + torch.arange(graph_input_ids[j].shape[0])
                           for j in order])
    nodes     = torch.cat([
        torch.full((graph_input_ids[j].shape[0],), j, dtype=torch.long) for j in order
    ])
```

with `node_position_mode="reset"` giving `base_offset[j] = 0` for every node. The model then just forwards the collator's `position_ids` into stock Llama RoPE — **no RoPE code is modified anywhere.** That is a much cleaner mechanism than "disabling RoPE" and it is the right one to copy.

The collator also exposes a second mode I did not see mentioned in the paper, documented in the source:

> `node_position_mode`: How each node's tokens' `position_ids` start (E3, TODO.md). `"reset"` (default) — every node restarts at local position 0, the historical behavior; **RoPE then carries zero inter-node relative-position signal.** `"spd_depth"` — a node's tokens start at `STRIDE * depth(node)`, where `depth` is its shortest-path distance from the prompt node (clamped to `max_spd`) [...] depth is a pure function of graph structure, invariant to input node-listing order.

**Consequence for us, and it is important.** We have exactly one token per node. Under `"reset"`, *every* token gets `position_id = 0`. RoPE at position 0 is the identity rotation. So for our architecture, **per-node RoPE reset is numerically identical to disabling RoPE entirely** — all queries and keys are unrotated. CLAUDE.md §2 writes "RoPE disabled / reset per node" as if they were alternatives; for us they are the same thing. Two practical notes:

- Implement it as `position_ids = torch.zeros(B, N, dtype=torch.long)` passed into the stock model. Zero model surgery, and it keeps the option of switching to `spd_depth`-style structural positions later by changing one tensor.
- Watch for degenerate behaviour: with all positions equal and no bias, attention is fully permutation-*invariant* over an unordered set, so the model literally cannot distinguish tokens except by their content (the encoder output). This is what we want for equivariance, but it also means **the attention bias is the only channel through which pairwise structure reaches the attention map.** That makes the "no attention bias" ablation in CLAUDE.md §8 a much sharper test than it might look.

### 2.9 The permutation-equivariance argument

Paper Property 1 (Appendix B.1):
> The internal representations and final outputs of GTLM are equivariant with respect to the serialization order of the prefix nodes.

The proof rests on **three** ingredients, all of which are necessary:
1. **RoPE indices reset at each node's start** — so no token's representation depends on where its node landed in the flattened sequence.
2. **Fully-visible non-causal mask across the prefix** — so the set of visible keys is the same regardless of order (a causal mask would make it order-dependent).
3. **Inter-node structural biases computed purely from static graph topology, independent of flattened sequence indices** — so `b(u,v)` moves with the nodes under a permutation π rather than staying pinned to sequence slots.

Given all three, the mask, the Q/K representations, and the bias are all equivariant under π, so softmax and the value-weighted sum are too, and equivariance holds exactly (up to fp accumulation order).

CLAUDE.md §11 says GTLM's fix is "RoPE reset plus bidirectional prefix mask plus structure-derived (not sequence-derived) bias" — **that is an accurate three-part summary. Confirmed.**

Their equivariance is only over *prefix* nodes; the prompt node is pinned last and generated causally. **We have no prompt node, so we get full equivariance over all N nodes** — a strictly stronger property than they prove, and easy to state as such.

Their test methodology (worth copying into `tests/`, and cheap): run the same graph under two random node permutations, un-permute the output logits back to canonical node order, take `max |Δ|`. `src/experiments/permutation_equivariance/__main__.py` does exactly this. For us it is even simpler — permute rows and columns of A, check `Â` permutes identically:

```python
# our version of their test
perm = torch.randperm(N)
A_p = A[perm][:, perm]
out_p = model(A_p)                       # [N, N]
assert torch.allclose(out_p, model(A)[perm][:, perm], atol=1e-4)
```

CLAUDE.md §11 already asks for this test. Note it will only pass if our **decoder** is equivariant too — D1 (inner product) is, D4 (autoregressive, BFS-ordered) is not.

### 2.10 FlashAttention / FlexAttention — the paper's limitation is obsolete

The repo contains a complete `torch.nn.attention.flex_attention` backend that the paper does not describe. `src/models/flex_attn/REBUTTAL.md` opens:

> **# Rebuttal note — the "eager attention only" limitation is resolved**
> Planning + evidence for the NeurIPS rebuttal on the FlexAttention backend, which did not exist at submission time. The submitted paper states that GTLM is forced onto eager attention and that training is correspondingly slow. **That limitation is now gone.**

The mechanism: keep the bias at node level and gather it **inside the Triton kernel** with a `score_mod` closure, so the `(B, H, L, L)` token-level tensor never exists. Verbatim from `src/models/flex_kernel.py`:

```python
def make_score_mod(
    node_bias: Optional[torch.Tensor],   # (B, H, N, N) float, or None
    node_ids: torch.Tensor,              # (B, kv_len) int
    q_offset: int = 0,
) -> Optional[Callable]:
    """Build the ``score_mod(score, b, h, q_idx, kv_idx)`` closure that gathers
    the token-expanded node bias on the fly. Returns None if there is no bias."""
    if node_bias is None:
        return None

    def score_mod(score, b, h, q_idx, kv_idx):
        nq = node_ids[b, q_idx + q_offset]
        nk = node_ids[b, kv_idx]
        return score + node_bias[b, h, nq, nk]

    return score_mod
```

That is the whole thing. From their design notes (`src/models/flex_attn/README.md`):

> **The node-bias gather is the core trick.** The per-layer soft bias stays at node level `(B,H,N,N)`; a flex `score_mod` gathers `node_bias[b,h,node[q],node[k]]` inside the kernel, so the dense token-level `(B,H,L,L)` blow-up never materializes. The kernel is agnostic to how `node_bias` was produced — **all bias types (SPD, magnetic, …) come for free**, no per-type flex work.

Measured results (H100, paper config, `k_hop=0`, gradient checkpointing on, B=1, bf16, all biases on, 24 real batches):

| dataset | L real | `eager` | `flex` | **speedup** | `flex-nobias` | `sdpa` (plain Llama) | flex/sdpa |
|---|--:|--:|--:|--:|--:|--:|--:|
| cora | 663 | 836.9 | 226.8 | **3.69×** | 143.0 | 118.6 | 1.91× |
| pubmed | 950 | 1,255.3 | 215.9 | **5.81×** | 137.2 | 112.1 | 1.93× |
| reddit | 976 | 406.3 | 201.3 | **2.02×** | 125.5 | 103.6 | 1.94× |
| ogbn-arxiv | 1,065 | 2,006.2 | 208.3 | **9.63×** | 126.8 | 103.7 | 2.01× |

Wall-clock, one epoch: **ogbn-arxiv 50.7 h → 5.3 h**; pubmed 4.12 h → 0.71 h.

Their own honest decomposition of the residual ~2× over a plain LLM (reddit):

| | step ms | Δ |
|---|--:|--:|
| `sdpa` — plain Llama-3.2-1B, plain causal | 103.6 | — |
| `sdpa+mask` — + GTLM's structural mask | 103.6 | **+0.0 ms** |
| `flex-nobias` — + flex kernel | 125.5 | +21.9 |
| `flex` — + per-layer bias modules | 201.3 | **+75.8** |
| `eager` — dense `(B,H,L,L)` instead of flex | 406.3 | +205.0 |

> **The mask is free.** `sdpa+mask` costs 0.90–1.04× plain causal across the four datasets even though GTLM's mask admits 0.70 of the L×L matrix against causal's 0.50 [...] So the ~2× is *not* the price of bidirectional prefix attention.
> **It is bias-module compute.** ~78% of the gap.

Isolated attention layer, scaling (H100, fwd+bwd ms):

| L | `eager` | `flex` (k=0) | `flex` (k=2) | `flash` floor |
|---|--:|--:|--:|--:|
| 1,216 | 21.5 | 3.0 | 2.1 | 0.5 |
| 4,478 | 263.8 | 26.9 | 15.3 | 2.0 |
| 17,506 | **OOM** | 411.6 | 110.0 | 22.4 |
| 69,121 | **OOM** | 5,511.5 | 282.7 | 330.6 |
| 276,067 | **OOM** | timeout | 1,834.0 | 5,404.4 |

Peak memory at L≈4.5k: eager 30 GB vs flex 0.2 GB.

Practical knobs they document (all directly reusable):

| knob | recommendation | trade |
|---|---|---|
| compile mode | `max-autotune-no-cudagraphs` (default) | 1.2–1.5× faster step for a ~320 s one-time compile per bucketed shape (vs ~16 s for `default`) |
| `BLOCK_SIZE` | **64 when k_hop>0, 128 when k_hop=0** | 64-blocks give 1.3–1.6× at k>0 but ~20% slower fwd at k=0. Sub-128 blocks **require** an autotune compile mode (inductor's default only generates 128-tile kernels, which must divide the mask block size) |
| `node_ids` dtype | **int32**, cast *before* capture | free and **bitwise-identical** to int64; net fwd+bwd −1.4% to −20.1%. int16 impossible (torch index-dtype rule) |
| `checkpoint_graph_bias` | **True** | ~free at small N; at N=2048 costs +20% step for **−36 GB** |
| padding | **pad L to a multiple of block size — mandatory** | ~14× cliff otherwise (L=8847 → 83 ms vs L=8960 → 5.7 ms, same work) |
| node ordering | **RCM (Reverse Cuthill–McKee)** | one-time O(V+E); makes k-hop neighbourhoods contiguous so more blocks are fully skippable |
| dynamo cache | raise `torch._dynamo.config.cache_size_limit` above the number of `(L, N)` buckets | default 8; past that it **silently falls back to eager** |

Known wall they flag: **L≈70k OOMs for every method including flash** — the unchunked LM-head cross-entropy is ~90 GB of logits at L=70k. Irrelevant to us (our decoder is `H @ W @ H.T`, not a 128k-vocab head) — one place where our architecture is *cheaper* than theirs.

Also: *"Realized speedup tracks block-level density, not element-level. A 95% element-sparse mask whose allowed pairs are scattered can be only ~50% block-sparse."* Relevant if we ever use a k-hop mask on Cora.

### 2.11 Differential learning rates

Paper Table 7 (GraphQA): η = 3e-5, η_bias = 5e-3, LoRA rank 16, alpha 32, 20 epochs.
Paper Table 8 (TAG): η ∈ {6e-5, 2e-4, 3e-4}, η_bias ∈ {0.01, 0.04}, LoRA rank ∈ {32, 64}, alpha = 2r.
Paper Table 9: Family Tree η=5e-5 / η_bias=1e-2; KG-QA η=5e-4 / η_bias=3e-2; both rank 32, alpha 64.
Paper Table 12 (scaling): 3B/8B, η_bias 2e-2 to 4e-2, LoRA rank 64, alpha 128.

**Ratio η_bias/η ranges from ~60× (GraphQA: 5e-3 / 3e-5 = 167×) to ~660× (TAG: 0.04 / 6e-5).** CLAUDE.md's numbers are right; its warning "Copy this or the bias parameters will not converge" is well-founded, and the code shows the mechanism explicitly.

Implementation (`src/utils/text_graph_trainer_v2.py`, `GraphTrainerV2.create_optimizer`) — the key excerpt:

```python
groups = {"base_decay": [], "base_no_decay": [], "bias_decay": [], "bias_no_decay": []}
active_p = self.active_params or []
for n, p in opt_model.named_parameters():
    if not p.requires_grad:
        continue
    is_active = any(act in n for act in active_p)
    if is_active:
        # HF's name-based rule would exempt every graph-bias param
        # (their module names contain "bias"). Decide by shape
        # instead: matrices decay, 1-D gains/offsets don't; modules
        # opt individual params out via `_no_weight_decay = True`.
        has_decay = p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)
    else:
        has_decay = n in decay_parameters
    key = ("bias" if is_active else "base") + ("_decay" if has_decay else "_no_decay")
    groups[key].append(p)

base_lr = self.args.learning_rate
bias_lr = self.bias_lr if self.bias_lr is not None else base_lr
grouped = [
    {"params": groups["base_decay"],    "weight_decay": self.args.weight_decay, "lr": base_lr, "is_bias": False},
    {"params": groups["base_no_decay"], "weight_decay": 0.0,                    "lr": base_lr, "is_bias": False},
    {"params": groups["bias_decay"],    "weight_decay": self.bias_weight_decay, "lr": bias_lr, "is_bias": True},
    {"params": groups["bias_no_decay"], "weight_decay": 0.0,                    "lr": bias_lr, "is_bias": True},
]
grouped = [g for g in grouped if g["params"]]
self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
```

Selection is by **substring match on parameter name**: `ACTIVE_PARAMS = ["graph_bias"]` (both `src/experiments/graphqa/train.py:32` and `src/experiments/tag_benchmarks/train.py:36`). So the bias modules must be named `graph_bias` on the attention module for the split to work.

Two subtle bugs they document and fixed, both of which we would otherwise hit:
- **HF's weight-decay rule is name-based and excludes anything containing "bias"** — which silently exempts every graph-bias parameter. They decide by *shape* instead (`p.ndim >= 2`), with an explicit `_no_weight_decay` opt-out on tensors that are 2-D by shape but semantically a lookup table (e.g. `SPDBias.weights`).
- Default `bias_weight_decay = 0.0`, deliberately, to preserve historical behaviour.

The stated reason for the LR gap, from CLAUDE.md and consistent with the code: bias parameters are **zero-initialized and must travel a long way**, while LoRA sits on top of refined pretrained weights and must barely move. There is a second reason visible in the code: the bias enters as an *additive logit* competing with `q·k/√d` values of O(1–10), so its gradient is small relative to its needed magnitude. Their `MIXED_BIAS.md` records an arm where the unnormalized landmark bias reached `|b|max = 9–240` against O(1–10) logits and **scored below the no-bias floor** — i.e. too large a bias destroys the language model. They log `bias_gain_absmax` per step to catch this. **We should log `max |bias|` per layer from day one.**

---

## 3. Code we can reuse (real snippets, real signatures)

### 3.1 File layout of `src/models/` (this is the layout to copy)

```
src/models/
  __init__.py                    # imports dispatch → registers gtlm_eager/gtlm_flex at import time
  config.py                      # GraphConfigMixin — FLAT graph-bias fields (round-trips through save_pretrained)
  attention.py                   # GraphAttentionMixin — owns the per-layer bias module, NO forward override
  bias.py                        # BaseBias + SPDBias/RRWPBias/MagneticBias/... + GraphAttentionBias coordinator (88 KB)
  causal_lm.py                   # GraphCausalLMMixin — builds GraphContext per batch, installs on every attn module
  context.py                     # GraphContext dataclass — the typed forward↔attention contract
  dispatch.py                    # gtlm_eager / gtlm_flex, registered into ALL_ATTENTION_FUNCTIONS
  structural_mask.py             # build_dense_structural_mask + expand_node_to_token_bias (pure tensor ops)
  flex_kernel.py                 # BlockMask build, score_mod, compiled flex_attention call
  io.py                          # save_bias_parameters / load_bias_parameters (tiny checkpoints)
  modeling_gtlm_llama.py         # 5-class thin adapter (130 lines) — the whole Llama wiring
  modeling_gtlm_gemma3.py        # layer-heterogeneous RoPE backbone
  modeling_gtlm_bloom.py         # ALiBi backbone probe
  configs/gtlm_llama_v0_1b.json  # LEGACY v0 config — do NOT copy its hyperparameters (see C5)
  biases/*.md                    # design docs per bias variant
  flex_attn/                     # benchmark suite + optimization log + REBUTTAL.md
```

Compare to CLAUDE.md §10's proposed `models/bias/{spd,rrwp,laplacian,per_head}.py`. **GTLM's flatter design is better**: one `bias.py` with a `BaseBias` protocol and a `BIAS_TYPES` registry, so adding a bias type is one class + one list entry and `GraphAttentionBias` picks it up automatically. Recommend we adopt the registry.

```python
class BaseBias(nn.Module):
    """
    Subclass protocol:
      config_key : str   — attribute on bias_config that enables this type.
      shared     : bool  — False (default): one instance per transformer layer.
                           True: ONE instance on the top-level model, computed once
                           per forward and added to every layer's bias.
      forward(**kwargs)  — consumes whatever it needs from the shared kwargs dict
                           and returns a (B, H, N, N) float tensor, or None when
                           its required input is absent.
    """
    config_key: str = ""
    shared: bool = False

    @classmethod
    def is_enabled(cls, bias_config) -> bool:
        return bool(getattr(bias_config, cls.config_key, False))
```

The `shared: bool` flag (compute once per forward vs. per layer) is a nice knob we should keep — CLAUDE.md's GaLA note about applying bias only in the first half of layers is the same axis. The repo generalizes it further with `magnetic_groups: int` — G instances, layer `l` served by group `l*G // num_layers`, spanning per-layer (G = L) to fully-shared (G = 1) with one integer.

### 3.2 The whole Llama wiring — 5 classes, no forward override

`src/models/modeling_gtlm_llama.py` (this is the entire model surgery; verbatim):

```python
class GTLMLlamaConfig(GraphConfigMixin, LlamaConfig):
    """LlamaConfig + the flat graph-bias fields (from GraphConfigMixin)."""
    model_type = "gtlm_llama"


class GTLMLlamaAttention(GraphAttentionMixin, LlamaAttention):
    def __init__(self, config: GTLMLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.init_graph_bias(config, layer_idx)


class GTLMLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: GTLMLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GTLMLlamaAttention(config, layer_idx)


class GTLMLlamaModel(LlamaModel):
    def __init__(self, config: GTLMLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GTLMLlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()


class GTLMLlamaForCausalLM(GraphCausalLMMixin, LlamaForCausalLM):
    """Graph-biased Llama causal LM with a standard HF I/O contract."""
    config_class = GTLMLlamaConfig
    graph_model_cls = GTLMLlamaModel
```

And the attention mixin is **parameters only** — it adds no math at all (`src/models/attention.py`, verbatim):

```python
class GraphAttentionMixin:
    def init_graph_bias(self, config, layer_idx: int) -> None:
        """Attach the per-layer trainable graph bias. Call from the backbone
        attention's ``__init__`` after ``super().__init__``.

        K-hop is enforced by the shared structural mask, so the bias module runs
        in soft-only mode (``k_hop=0``) and just accumulates the trainable biases.
        ``head_dim`` is read from ``config.head_dim`` (not ``hidden_size //
        num_attention_heads``) so backbones where they differ (e.g. Gemma2) are
        correct.
        """
        self.graph_bias = GraphAttentionBias(
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            layer_idx=layer_idx,
            bias_config=config,
            k_hop=0,
        )
        self._graph_ctx = None
```

Their own name for this is **"Strategy B"**:

> there is **no attention `forward` override**. The stock `LlamaAttention.forward` does q/k/v projection + RoPE + cache, then dispatches to the `gtlm_*` function named by `config._attn_implementation`, which the causal-LM forward pins per batch.

**This is the single best architectural lesson in the repo.** Overriding `forward` means re-implementing q/k/v projection, RoPE application, and KV-cache update — all of which change between transformers releases. Registering an attention *function* means you own only the score step. Their `legacy/modeling_gtlm_llama_v0.py` is 61 KB; the Strategy-B `modeling_gtlm_llama.py` is 5.2 KB. That is a 12× reduction from making the right structural choice, and it is the choice we should make on day one rather than after a rewrite.

### 3.3 Registering the attention function

`src/models/dispatch.py`, verbatim:

```python
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def gtlm_eager(module, query, key, value, attention_mask=None, *, scaling, dropout=0.0, **kwargs):
    return _gtlm_dense("eager", module, query, key, value, scaling, dropout)

def gtlm_flex(module, query, key, value, attention_mask=None, *, scaling, dropout=0.0, **kwargs):
    ctx = _require_ctx(module)
    node_bias = compute_node_bias(module, ctx, query.dtype, query.device)
    score_mod = make_soft_score_mod(node_bias, ctx.node_ids_flex)
    return flex_attention_forward(
        query, key, value,
        block_mask=ctx.block_mask, score_mod=score_mod,
        scaling=scaling, enable_gqa=True,
        compile_mode=module.config.flex_compile_mode,
    )

def register_gtlm_attention_functions() -> None:
    """Register the gtlm_* functions into HF's ``ALL_ATTENTION_FUNCTIONS`` so a
    backbone can select one via ``config._attn_implementation``. Idempotent.

    NB: set ``config._attn_implementation`` to these names *directly* (as the
    config mixin does); do not route a custom name through the public
    ``attn_implementation=`` kwarg, which validates against the built-in set.
    """
    ALL_ATTENTION_FUNCTIONS["gtlm_eager"] = gtlm_eager
    ALL_ATTENTION_FUNCTIONS["gtlm_flex"] = gtlm_flex

register_gtlm_attention_functions()   # at import time
```

Note the docstring caveat about `attn_implementation=` validating against a built-in set — see §5 for the sanctioned modern API (`AttentionInterface.register`) which does not have this problem.

Their `gtlm_*` functions **ignore the incoming `attention_mask` entirely** and substitute their own structural mask. That is deliberate and documented. In transformers v5 this interacts with `AttentionMaskInterface` — see §5.3.

Their honest note on why SDPA is not offered:

> The fused SDPA kernels are intentionally not offered: a custom dense bias makes flash ineligible and the mem-efficient kernel both buggy in backward (GQA + per-head bias) and pointless (the dense bias is already materialized), so it would only ever reduce to eager.

**"mem-efficient SDPA backward is buggy with GQA + per-head bias" is a landmine we would otherwise step on.** Do not pass a per-head `(B,H,L,L)` bias to `F.scaled_dot_product_attention` and expect correct gradients under GQA.

### 3.4 The per-batch context object

`src/models/context.py`, verbatim:

```python
@dataclass
class GraphContext:
    node_ids:        torch.Tensor                 # (B, kv_len) token → node index
    num_nodes:       Optional[torch.Tensor]       # (B,) node count per batch item
    cache:           Dict[str, Any]               # per-generation bias cache (shared dict)
    features:        Dict[str, Any]               # spd / laplacian / rwse / rrwp / magnetic
    structural_mask: Optional[torch.Tensor] = None  # (B, 1, q, kv) dense path
    block_mask:      Optional[Any] = None            # flex BlockMask
    node_ids_flex:   Optional[torch.Tensor] = None   # (B, kv_len) int32, flex path
    shared_node_bias: Optional[torch.Tensor] = None  # (B, H, N, N)
    shared_pair_features: Optional[torch.Tensor] = None  # (B, N, N, magnetic_dim)
    node_start_indices: Optional[torch.Tensor] = None  # (B, N) first-token pos per node
    group_bias:      Optional[Any] = None            # bias.GroupBiasCache

    def install_on(self, attn_modules) -> None:
        """Attach this context to each attention module so the registered
        ``gtlm_*`` function can read it. Not cleared afterwards: it must survive
        gradient-checkpointing recompute in backward; the next forward overwrites
        it."""
        for module in attn_modules:
            module._graph_ctx = self
```

**"Not cleared afterwards: it must survive gradient-checkpointing recompute in backward"** — if you clear the context at the end of forward, gradient checkpointing will recompute the layer in backward and find nothing. We will hit this the moment we enable checkpointing.

The alternative to this side-channel is to pass graph tensors as forward kwargs, which transformers v5 does support end-to-end (§5.2). The side-channel exists because their features are `(B,N,N,K)` sized and they want the bias computed once per layer, not threaded through every intermediate signature.

### 3.5 Gradient checkpointing of the bias modules

`src/models/dispatch.py`:

```python
def compute_node_bias(module, ctx, dtype, device) -> Optional[torch.Tensor]:
    """Compute this layer's node-level soft bias ``(B, H, N, N)`` (or ``None``).

    Optionally gradient-checkpointed: the bias compute is milliseconds, but
    autograd would otherwise keep every layer's ``(B, N, N, ·)`` intermediates
    alive. Safe to recompute — in training the bias cache is never read, so the
    recompute is deterministic.
    """
    feats = ctx.features
    def _bias():
        return module.graph_bias(dtype=dtype, device=device, num_nodes=ctx.num_nodes,
                                 spd=feats["spd"], rrwp=feats["rrwp"],
                                 magnetic=feats["magnetic"], ..., cache_dict=ctx.cache)

    if (getattr(module.config, "checkpoint_graph_bias", False)
            and module.training and torch.is_grad_enabled()):
        node_bias = torch.utils.checkpoint.checkpoint(_bias, use_reentrant=False)
    else:
        node_bias = _bias()
    ...
```

Their measured payoff: *"~40 GB at N=2048 for ms of recompute"*, and it "turns `2048×8` from OOM into runnable" for **+20% step time**. Default is `True`. At our Cora scale (N up to 2,708) the per-layer `(B,N,N,·)` intermediates are exactly the thing that will OOM us, so **turn this on from the start**.

### 3.6 Tiny checkpoints

`src/models/io.py`, verbatim — 173 K parameters do not need a 2.5 GB checkpoint:

```python
def save_bias_parameters(model, save_dir, params):
    os.makedirs(save_dir, exist_ok=True)
    custom_state_dict = {}
    for name, param in model.named_parameters():
        if any(param_name in name for param_name in params):
            custom_state_dict[name] = param.data
    torch.save(custom_state_dict, os.path.join(save_dir, "bias_parameters.pt"))

def load_bias_parameters(model, path):
    bias_path = os.path.join(path, "bias_parameters.pt")
    if os.path.exists(bias_path):
        state_dict = torch.load(bias_path, map_location="cpu", weights_only=True)
        result = model.load_state_dict(state_dict, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"bias_parameters.pt at {path} holds {len(result.unexpected_keys)} tensors "
                f"with no matching model parameter (e.g. {result.unexpected_keys[:3]}) — "
                "the checkpoint and the model disagree on parameter names, so the graph-bias "
                "weights were NOT restored.")
        return result
    return None
```

The `unexpected_keys` guard is the important half: `load_state_dict(strict=False)` silently loads nothing if names drift, and you discover it as "training from scratch again" three days later.

### 3.7 LoRA setup

`src/train/model.py`:

```python
def select_active_params(model, active_params=None, lora=None):
    if lora is not None:
        lora_config = LoraConfig(
            r=lora.get("r", 8),
            lora_alpha=lora.get("lora_alpha", 16),
            target_modules=lora.get("target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=lora.get("lora_dropout", 0.05),
            bias=lora.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if active_params == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif active_params is not None:
        for name, param in model.named_parameters():
            if any(p in name for p in active_params):
                param.requires_grad = True
    return model
```

Order matters: **freeze everything, then apply LoRA, then selectively unfreeze the bias params by name.** `get_peft_model` wraps the model and mangles parameter names (`base_model.model.…`), which is why the bias selection is substring-based rather than exact.

### 3.8 Their config style — flat fields, not nested

`src/models/config.py`:

```python
class GraphConfigMixin:
    def __init__(self, spd=False, laplacian=False, max_spd=32, rwse=False,
                 rrwp=False, max_rw_steps=8, magnetic=False, magnetic_dim=32,
                 magnetic_q=0.25, bias_self_node=False, k_hop=0,
                 k_hop_directed=False, graph_attn_impl="eager",
                 checkpoint_graph_bias=True,
                 flex_compile_mode="max-autotune-no-cudagraphs",
                 flex_block_size=None, flex_cache_size_limit=32, **kwargs):
        super().__init__(**kwargs)
        ...
```

With the rationale in the module docstring:

> The graph fields are stored **flat** (not nested) so standard `save_pretrained` / `from_pretrained` round-trips them with no custom code, and so the config can be handed directly to `GraphAttentionBias` as its `bias_config`.

Copy this. A nested `graph_attn_bias: {...}` dict (which their legacy v0 config used) does not round-trip cleanly through `PretrainedConfig`.

Their config `__init__` also does **eager validation with loud errors**, e.g. rejecting mutually exclusive magnetic placements with a message naming the clash. One of these is worth quoting because it describes a failure mode we will hit:

> a dataset/collator gate that misses one yields a run with **NO bias at all**, which trains cleanly and looks like a clean negative result.

That is exactly how our §8 ablation grid could silently lie to us. **Assert that the bias is actually non-zero at least once during training** (e.g. log `max|bias|` — if it is exactly 0 after 100 steps, something is disconnected).

---

## 4. Numbers to beat / hyperparameters to copy

### 4.1 Their reported results

**GraphQA** (accuracy; "GTLM (Standard)" / "GTLM (Incidence)" are two graph-serialization arms):

| task | GTLM | prior best (GraphToken) |
|---|--:|--:|
| Node Count | 100 ± 0.0 | 99.6 |
| Edge Count | 62.2 ± 2.0 | 42.6 |
| Cycle Check | 96.7 ± 0.6 | 96.4 |
| Triangle Counting | 31.6 ± 0.5 | **34.8** (they lose) |
| Node Degree | 99.7 ± 0.1 | 96.2 |
| Connected Nodes | 98.8 ± 0.3 | 26.4 |
| Reachability | 99.0 ± 0.0 | 94.4 |
| Edge Existence | 99.8 ± 0.2 | 73.8 |
| Shortest Path | 91.7 ± 1.8 | 63.8 |

**Edge Existence 99.8 vs 73.8 is the number most relevant to us** — it is the closest thing in their paper to our masked-edge-prediction task, though it is asked as a text question about a fully visible graph, not as reconstruction of a hidden entry. It is not a directly comparable number and we should not claim it as such.

**Text-Attributed Graphs** (node classification, Llama-3.2-1B):

| Dataset | Metric | GTLM best | GTLM mean | RGLM-Denoiser | LLaGA |
|---|---|--:|--:|--:|--:|
| Cora | Acc | 90.04 | 88.99 ± 1.12 | **90.22** | 88.75 |
| Cora | F1 | 92.14 | 91.31 ± 0.90 | 89.35 | 87.87 |
| PubMed | Acc | 94.70 | 94.57 ± 0.11 | 90.95 | 90.34 |
| PubMed | F1 | 94.84 | 94.76 ± 0.10 | 90.90 | 90.25 |
| ogbn-arxiv | Acc | 76.53 | 76.07 ± 0.54 | 75.10 | 74.61 |
| ogbn-arxiv | F1 | 70.02 | 69.40 ± 0.86 | 56.84 | 56.48 |
| Reddit | Acc | 68.09 | 67.80 ± 0.37 | 67.64 | 66.38 |
| Reddit | F1 | 68.00 | 67.71 ± 0.32 | 67.50 | 66.38 |

Note honestly: on **Cora accuracy they are second** (88.99 mean vs RGLM-Denoiser 90.22). Their win is on F1 and on the harder datasets.

**Scaling** (Table 4) — the interesting negative result: Cora 88.99 (1B) → 89.17 (3B) → 89.25 (8B); PubMed 94.57 → 94.52 → 94.67. **Essentially flat.** Model scale buys almost nothing here. That is a good argument for us to stay on 1B and spend the compute on the encoder study instead, and a good sentence to have ready when a reviewer asks why we did not use a bigger LLM.

**Beyond standard benchmarks** (Table 3):

| task | GTLM | Fine-tuned LLM | RGLM-Decoder |
|---|--:|--:|--:|
| Family Tree | 84.7 ± 1.8 | 81.7 ± 2.0 | 7.7 ± 0.3 |
| Knowledge Graph QA | 76.0 ± 2.6 | 58.1 ± 2.3 | 52.0 ± 2.7 |

**Ablation** (Table 5, GraphQA Standard) — which bias family matters:

| removed | Edge Count | Connected Nodes | Shortest Path |
|---|--:|--:|--:|
| none | 56.9 ± 1.7 | 90.7 ± 0.3 | 90.1 ± 1.7 |
| w/o SPD | 52.6 ± 5.9 | 81.4 ± 2.3 | **92.0 ± 2.6** (better!) |
| w/o RRWP | 48.9 ± 1.5 | 87.7 ± 1.7 | 90.0 ± 1.8 |
| w/o Magnetic | **7.1 ± 1.3** | 91.0 ± 1.4 | 76.3 ± 12.7 |

**Read this carefully, it is the most actionable table in the paper.** Removing the Magnetic Laplacian destroys Edge Count (56.9 → 7.1) and destabilizes Shortest Path (variance 1.7 → 12.7). Removing SPD *improves* Shortest Path. No single bias dominates; **Magnetic is load-bearing and is the one CLAUDE.md treats as most exotic.** If we implement only one bias, the ablation says SPD is the least essential of the three — which is the opposite of the intuitive ordering, and the opposite of CLAUDE.md §9's implicit ordering.

Caveat before we over-index on this: their Magnetic Laplacian is *directed*-graph machinery, and GraphQA graphs may be directed in a way Cora is not. See Open Questions.

### 4.2 Hyperparameter block to copy

```yaml
# bias
spd: true
max_spd: 8
rrwp: true
max_rw_steps: 16
magnetic: true
magnetic_dim: 32          # bias-MLP hidden width, NOT number of eigenvectors
magnetic_q: 0.25
bias_self_node: false     # zero the intra-node / diagonal bias
k_hop: 0

# optimization
learning_rate: 3e-5           # LoRA / base
bias_learning_rate: 5e-3      # ~170x higher; sweep {5e-3, 1e-2, 4e-2}
lr_scheduler_type: cosine_with_min_lr
min_lr: learning_rate / 10
warmup_steps: total_steps // 10
weight_decay: 0.1             # base params only; bias_weight_decay 0.0
lora_r: 16                    # {16, 32, 64}
lora_alpha: 2 * lora_r
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
gradient_accumulation_steps: 4       # 32 in the paper's TAG runs
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
checkpoint_graph_bias: true

# kernel
graph_attn_impl: flex
flex_compile_mode: max-autotune-no-cudagraphs
flex_block_size: 128          # 64 if k_hop > 0
flex_cache_size_limit: 32     # > number of distinct (L, N) buckets
```

### 4.3 Cost targets

- Their Cora step: **226.8 ms** (flex) vs 118.6 ms for plain Llama-3.2-1B at L≈663, B=1, H100, gradient checkpointing on. Target ~2× the plain-LLM floor.
- Cora one epoch: **0.10 h** (flex) vs 0.38 h (eager).
- Peak memory, Cora: flex 4.85 GB vs eager 6.33 GB.
- At L≈4.5k: eager 30 GB / flex 0.2 GB (isolated attention layer). **Full Cora at N=2,708 is squarely in the region where eager is painful and flex is trivial.**

---

## 5. Adding a custom additive attention bias to a modern HF causal LM — the general question

Researched independently of GTLM, from the current transformers docs (v5.15.1) and the PyTorch FlexAttention blog. There are four viable routes; GTLM uses (2) and (4).

### 5.1 Route 1 — pass a 4D float mask (simplest; no code changes)

Transformers accepts a user-supplied 4D `attention_mask` of shape `(B, 1, q_len, kv_len)` and **skips its own `create_*_mask` entirely**. Float convention: `0.0` = attend, `-inf` = masked, added to scores pre-softmax.

- **Works today with zero model surgery.** Because it is *added* to the logits, you can smuggle a bias in by putting non-zero finite values where you want a bias.
- **Fatal limitation for us: the head dimension is 1.** The doc says "`1` broadcasts the same mask across every attention head" — so a per-head bias cannot be expressed this way. CLAUDE.md wants a per-head bias (§9, GaLA note). Whether a `(B, H, q, kv)` mask is accepted is model-dependent and not documented as supported.
- **Also: it is one tensor for the whole model.** A per-layer bias cannot be expressed. GTLM's bias is per-layer.
- **Backend restriction, verbatim:** *"`sdpa` takes a boolean or a float mask. `eager` adds the mask to the scores, so it takes a float mask only. `flash_attention_2` and `flex_attention` consume their own formats (a 2D padding mask and a `BlockMask`) and don't accept a raw 4D mask."*
- Classic bug the docs call out: reusing the 2D padding convention (`1`/`0`) in a float 4D mask. `0.0` *keeps* a position; `1.0` merely adds a small bias. Nothing errors; the mask silently does nothing.

**Verdict: good enough for a shared, single-layer, head-uniform bias — i.e. a quick smoke test. Not good enough for the real model.**

### 5.2 Route 2 — register a custom attention function (the sanctioned modern way; GTLM's Strategy B)

```python
from transformers import AutoModelForCausalLM, AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

def custom_attention(
    module: torch.nn.Module,   # required
    query: torch.Tensor,       # required
    key: torch.Tensor,         # required
    value: torch.Tensor,       # required
    attention_mask: Optional[torch.Tensor],  # required
    a_new_kwarg=None,          # you can add as many kwargs as you need
    **kwargs,                  # required — models pass other args
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    ...
    return attn_output, attn_weights   # attn_weights optional

AttentionInterface.register("custom", custom_attention)
AttentionMaskInterface.register("custom", sdpa_mask)   # MUST have the same name
model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="custom")
model(torch.ones(1, 5, dtype=int), a_new_kwarg=...)
```

Why this is the right route:
- You own **only the score step**. q/k/v projection, RoPE, and KV-cache stay in the model's stock forward — the code most likely to change between releases.
- The attention function receives `module`, so `module.graph_bias` (a per-layer `nn.Module`) is reachable → **per-layer, per-head bias for free.**
- **Kwargs propagate:** "Models supporting AttentionInterface propagate kwargs to attention layers and the attention function. Pass arguments as kwargs in the model's forward function." So graph features can travel as forward kwargs rather than as GTLM's `_graph_ctx` side-channel. Prefer kwargs where the tensors are small; the side-channel earns its keep only when you want per-layer caching of a big derived tensor.

**The one warning that will bite, verbatim from the docs:**
> Register a matching attention mask function when you register a custom attention function. If the custom `attn_implementation` name is not registered in `AttentionMaskInterface`, Transformers skips mask creation and passes `attention_mask=None` to the attention layers. Your attention function must handle causal, padding, packing, or sliding-window constraints itself, **or those constraints can be silently dropped.**

GTLM's `gtlm_eager` ignores the incoming mask and builds its own, so this is harmless for them — but only because they do it deliberately. If you register a custom function, *assume the mask is `None`* until you have registered a mask formatter.

Version note: GTLM writes into `ALL_ATTENTION_FUNCTIONS` directly and sets `config._attn_implementation` (the private attribute) rather than passing `attn_implementation=`, with the comment *"do not route a custom name through the public `attn_implementation=` kwarg, which validates against the built-in set."* On current transformers, `AttentionInterface.register` + `attn_implementation="custom"` is the documented public path and does not have that problem. Use the public API; keep GTLM's trick in mind as a fallback for older pins.

### 5.3 Route 3 — FlexAttention `score_mod` (best performance; where the field has landed)

`score_mod` receives a scalar and its indices:

```python
def score_mod(score: f32[], b: i32[], h: i32[], q_idx: i32[], kv_idx: i32[]):
    return score          # no-op = standard attention
```

A learned per-head additive bias is the ALiBi example, verbatim from the PyTorch blog:

```python
alibi_bias = generate_alibi_bias()   # [num_heads]

def alibi(score, b, h, q_idx, kv_idx):
    bias = alibi_bias[h] * (kv_idx - q_idx)
    return score + bias
```

> This demonstrates one interesting piece of flexibility `torch.compile` provides — we can load from `alibi_bias` even though it *wasn't explicitly passed in as an input*! The generated Triton kernel will calculate the correct loads from the `alibi_bias` tensor and fuse it. Note that you could regenerate `alibi_bias` and we still wouldn't need to recompile.

And explicitly for a full 2D learned bias:

```python
bias = torch.randn(1024, 1024)
def score_mod(score, b, h, q_idx, kv_idx):
    return score + bias[q_idx][kv_idx]      # The bias tensor can change!
```

Gradients flow to captured tensors — the blog notes they "leveraged higher order ops, PyTorch's autograd to automatically generate the backwards pass". GTLM's `test_flex_bias_grad_parity` test confirms bias-parameter gradient parity against eager in practice.

**`score_mod` vs `mask_mod` — do not conflate them.** `mask_mod(b, h, q_idx, kv_idx) -> bool` goes into `create_block_mask` and produces a `BlockMask` that lets the kernel *skip whole blocks*. `score_mod` is strictly more expressive but is applied to every computed element. The blog is blunt: applying masking via `score_mod` costs "about a 15-20% degradation". **Structure that masks goes in `mask_mod`; structure that biases goes in `score_mod`.** GTLM does exactly this split — hard k-hop/causal/padding in the BlockMask, soft learned bias in `score_mod`.

Recompilation rules worth internalizing:
- Changing *captured tensor values* does **not** trigger recompilation.
- Changing *block sparsity* requires recomputing the `BlockMask` (hundreds of µs) but **not** recompiling.
- Changing *shapes* does trigger recompilation → bucket your `(L, N)` and raise `torch._dynamo.config.cache_size_limit` (default 8) or it silently falls back to eager.
- `create_block_mask` is expensive; build once per batch and thread it through every layer. Set `B=None`/`H=None` to broadcast where possible, and use `_compile=True` ("often an order of magnitude" cheaper).

### 5.4 Route 4 — override the attention `forward` (what GTLM did first, then abandoned)

Their `legacy/modeling_gtlm_llama_v0.py` is **61 KB**; the Strategy-B replacement is **5.2 KB**. That is the whole argument. Only do this if you need to change something *before* the score step (e.g. modify q/k themselves, as their `MagneticMagnitudeBias` does by appending structural dimensions to per-head Q/K).

### 5.5 Bidirectional attention on a causal model — first-class support now

For our non-causal requirement there is now a supported flag; no mask surgery needed:

```python
config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
config.is_causal = False
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B", config=config)
# or per call, temporarily overriding the config:
outputs = model(**inputs, is_causal=False)
```

Docs: *"This only works for causal (decoder) models. It does not turn encoder models into decoder models."* For a partially-bidirectional (prefix-LM) pattern, the composable route is `create_causal_mask(..., or_mask_function=...)` — with the doc's own warning that `or_mask_function`/`and_mask_function` "can express any attention pattern, but they're slower than the built-in patterns", noticeable on small models.

**Our project needs full bidirectionality, so `is_causal=False` is the cheapest correct answer** and we should try it before writing anything like `build_dense_structural_mask`.

### 5.6 Checklist of gotchas (assembled from both sources)

1. **`_is_hf_initialized = True`** on any layer you zero-initialize, or `post_init()` re-randomizes it.
2. **Mask with `torch.finfo(dtype).min`, not `-inf`** — `-inf` + finite bias → NaN under low-precision accumulation.
3. **Always leave the diagonal unmasked** as a NaN guard against fully-masked softmax rows.
4. **Do not clear per-batch state at the end of forward** if gradient checkpointing is on; backward recompute needs it.
5. **HF's weight-decay rule is name-based and excludes "bias"** — it will silently exempt your graph-bias parameters. Decide by shape.
6. **Mem-efficient SDPA backward is buggy with GQA + per-head bias.** Do not trust it.
7. **A registered custom attention name with no registered mask formatter gets `attention_mask=None`.** Silently.
8. **Bias magnitude is a real failure mode.** GTLM measured `|b|max = 9–240` against O(1–10) logits and scored *below* the no-bias floor. Log `max|bias|` per layer per step.
9. **A misconfigured feature gate yields a run with no bias at all, which trains cleanly and reads as a clean negative result.** Assert the bias is non-zero.
10. FlexAttention: **pad L to a multiple of `BLOCK_SIZE`** — ~14× cliff otherwise.
11. FlexAttention: raise `torch._dynamo.config.cache_size_limit` above your shape-bucket count or you silently fall back to eager.

---

## 6. Corrections to CLAUDE.md

### C1 — LoRA rank on GraphQA is 16, not 32 or 64 *(confidence: certain)*

CLAUDE.md §9: "LoRA rank 32 or 64, alpha = 2r."

Paper Table 7 gives **LoRA Rank 16, LoRA Alpha 32** for GraphQA; the repo default `src/experiments/graphqa/config.py:252` is `lora_r: int = 16`; the repo README's example command uses `--lora_r 16`. Rank {32, 64} is correct for the **TAG node-classification** benchmarks only (Table 8). Rank 32 / alpha 64 for Family Tree and KG-QA (Table 9); rank 64 / alpha 128 for the 3B/8B scaling runs (Table 12).

`alpha = 2r` **does** hold everywhere — that half of the claim is right. Fix: "LoRA rank 16 (GraphQA) to 64 (8B scaling), alpha = 2r throughout."

### C2 — The FlashAttention limitation is fixed; "use the eager path" is now bad advice *(confidence: certain)*

CLAUDE.md §11 says "Do not use FlashAttention. It cannot take the custom bias. Use the eager attention path." and §9 cites the ~3× slowdown as GTLM's standing limitation.

Both statements are accurate *about the submitted paper* and obsolete *about the method*. `src/models/flex_attn/REBUTTAL.md` is titled **"the 'eager attention only' limitation is resolved"** and states the limitation "did not exist at submission time... That limitation is now gone." The repo ships `gtlm_flex`, `src/models/flex_kernel.py`, and a benchmark suite showing 2.0×–9.6× over eager on the four TAG datasets, memory going from 30 GB (eager, L≈4.5k) to 0.2 GB (flex), and eager OOMing above L≈4.5k while flex runs to L≈256k.

The correct statement: FlashAttention-2 proper still cannot take an arbitrary additive bias, but **FlexAttention can, and is the right backend.** The bias is gathered inside the Triton kernel by a `score_mod` closure; the dense `(B,H,L,L)` tensor never materializes. Practical consequence for CLAUDE.md §11's memory paragraph: "Cora at N=2,708 is ~7.3M attention entries per head per layer. Measure this before building anything elaborate" — with flex, this concern largely evaporates, and the fallbacks listed (sample k-hop subgraphs, cap at 500–1,000 nodes) may be unnecessary. Keep eager as the *reference implementation* for numerical parity tests, exactly as GTLM does.

Also note their honest scoping: the ~2× residual over a plain LLM is **bias-module compute (78% of the gap), not the bidirectional mask** — `sdpa+mask` measured 1.00× plain causal. So CLAUDE.md's implication that bidirectional attention is a cost driver is wrong; it is free.

### C3 — "0.015% of Llama-3.2-1B" is slightly off *(confidence: likely)*

173,056 is exactly right (verified three ways, §1.2). But 173,056 / 1,235,814,400 = **0.0140%**, not 0.015%. Against non-embedding parameters (973,146,112) it is 0.0178%. 0.015% is reproduced only by a denominator around 1.15 B. This is a rounding/denominator quibble, not a fabrication — but if we quote the figure in our own writeup, quote **0.014%** or say "~0.015% as reported", and state the denominator.

### C4 — `magnetic_dim = 32` is an MLP hidden width, not a spectral dimension *(confidence: certain)*

CLAUDE.md §9: "Magnetic Laplacian q = 0.25, dim 32." This reads as though 32 eigenvectors are retained. It is not that.

`src/experiments/tag_benchmarks/config.py:175`, verbatim: `magnetic_dim: int = 32   # model bias-MLP hidden width`. It is the width of `deep_set`'s output and `proj[0]`'s output inside `MagneticBias`. The number of eigenpairs retained is a **separate** knob, `magnetic_m` (`get_magnetic_laplacian_coords(..., m=0)`), and its default `m = 0` means **keep all N eigenpairs**. A config comment in `tag_benchmarks/train.py` makes the distinction explicitly: the collator's magnetic-m "is NOT the model's magnetic_dim."

This matters for us: budgeting "32 eigenvectors" would be a materially different (and much cheaper) design than what they actually ran, which is a full N×N eigendecomposition per graph.

### C5 — Do not copy `src/models/configs/gtlm_llama_v0_1b.json` *(confidence: certain)*

The only committed model config in the repo is a **legacy v0** file with different values from the paper:

```json
"graph_attn_bias": {
  "spd": true, "max_spd": 16,
  "rrwp": true, "max_rw_steps": 4,
  "magnetic": true, "magnetic_dim": 16, "magnetic_q": 0.2
}
```

`max_spd 16` (not 8), `max_rw_steps 4` (not 16), `magnetic_dim 16` (not 32), `magnetic_q 0.2` (not 0.25), and a **nested** `graph_attn_bias` dict which the current `GraphConfigMixin` replaced with flat fields. It is used only by the backward-compatibility experiment. Anyone browsing the repo for "the config" will find this file first and get every number wrong.

Also note the current mixin's *defaults* (`max_spd=32`, `max_rw_steps=8`) differ from the paper's Table 6 values — the paper's values come from the experiment configs, not the model defaults. Read `src/experiments/*/config.py`, not `src/models/config.py`, for what was actually run.

### C6 — "RoPE disabled / reset per node" are the same thing for us *(confidence: certain)*

CLAUDE.md §2 lists "RoPE disabled / reset per node" as alternatives. GTLM never disables RoPE; it resets `position_ids` to 0 at each node boundary, in the **collator**, and forwards them to stock Llama RoPE (`src/utils/text_graph_collator_v2.py::_pack_one`). With our one-token-per-node design, every token gets `position_id = 0`, so RoPE becomes the identity rotation — reset and disabled coincide exactly. Implementation guidance: pass `position_ids = torch.zeros(B, N, dtype=torch.long)`; write no RoPE code.

Related: the repo has an unpublished `node_position_mode="spd_depth"` mode giving nodes `STRIDE * depth(node)` positions, which preserves permutation equivariance (depth is a function of topology) while restoring *some* RoPE signal. Not in the paper. Worth an ablation for us, since our "reset" degenerates completely.

### C7 — The equivariance proof needs three ingredients, and CLAUDE.md §2 lists only two *(confidence: certain)*

CLAUDE.md §2's architecture block lists "RoPE disabled / reset per node" and "bidirectional (non-causal) mask" but not the third necessary condition. Property 1's proof requires additionally that **the bias be computed from static graph topology, independent of flattened sequence index**. CLAUDE.md §11 does state all three correctly ("RoPE reset plus bidirectional prefix mask plus structure-derived (not sequence-derived) bias") — so §2 should be brought in line with §11. It matters because a bias derived from, say, node *index* rather than node *structure* would look fine and silently break equivariance.

Second nuance: GTLM proves equivariance over **prefix nodes only** (the prompt node is pinned last and generated causally). We have no prompt node, so our claim is strictly stronger. Worth saying explicitly in our writeup rather than citing theirs as if identical.

### C8 — The ablation ordering is the reverse of CLAUDE.md's emphasis *(confidence: certain)*

CLAUDE.md §9 lists SPD first and describes it as "our hop-bias table" (§9, Graphormer), treating the Magnetic Laplacian as the exotic extra. Their Table 5 ablation says the opposite: removing Magnetic collapses Edge Count from 56.9 to **7.1** and blows Shortest Path variance from ±1.7 to ±12.7, while **removing SPD *improves* Shortest Path** (90.1 → 92.0). If we implement one bias, the evidence does not point at SPD.

Caveat (stated as such): this is measured on GraphQA, on their graphs, for their tasks. It is not evidence about Cora link prediction. But it is enough to say CLAUDE.md's implicit priority ordering is not supported by the source it cites.

### C9 — Unverified: GaLA's arXiv id resolves to a differently-titled paper *(confidence: uncertain — outside my assigned topic)*

Flagging only because it surfaced during search. CLAUDE.md §9 cites "GaLA — arXiv 2606.15633 (KDD '26), Loveland, Trivedi, Weinstein, Huang, Koutra". A search result for that id returned the title **"Formalizing and Mitigating Structural Distortion in LLM Attention for Graph Reasoning"** (arxiv.org/html/2606.15633v2). I did not fetch or verify this paper — it is another agent's topic. It may simply be a retitled version. Someone should confirm the id/title/name correspondence rather than assume it.

---

## 7. Actionable summary for our build

1. **Use Strategy B.** Register a `gtlm_eager`-style attention function via `AttentionInterface.register`; do not override `LlamaAttention.forward`. 5 KB vs 61 KB of code, in their own repo history.
2. **Add flex from the start, keep eager as the parity oracle.** Copy `make_score_mod` verbatim — for us `node_ids = arange(N)` so it is even simpler.
3. **Set `config.is_causal = False`** rather than hand-building a bidirectional mask. Verify against a hand-built mask once.
4. **`position_ids = zeros(B, N)`.** No RoPE code.
5. **Zero-initialize every bias output layer** and set `_is_hf_initialized = True`. The model then starts exactly as the base LLM and trains upward.
6. **Zero the bias diagonal** (`bias_self_node = False` equivalent).
7. **Differential LR from day one**: bias LR ~100–600× the LoRA LR, split by substring match on `graph_bias`, weight decay decided by shape not name.
8. **Log `max |bias|` per layer per step.** It is how they caught a diverging arm, and how we will catch a silently-disconnected one.
9. **Turn on `checkpoint_graph_bias`.** The `(B,N,N,·)` intermediates are what will OOM us at Cora scale.
10. **Copy the permutation test and the "base-LLM parity" test into `tests/`** — cheap, and they are the two properties we will claim in the paper.
11. **Adopt the `BaseBias` + `BIAS_TYPES` registry pattern** instead of CLAUDE.md §10's one-file-per-bias layout.
12. Our decoder is `H @ W @ H.T`, not a 128k-vocab LM head — so **we dodge their L≈70k cross-entropy memory wall entirely.** Worth noting as an advantage.

---

## 8. Open questions

1. **Does the Magnetic Laplacian do anything on an undirected graph?** With `theta_ij = 2*pi*q*(a_ij - a_ji)`, an undirected graph gives `theta = 0` everywhere, `rotation = 1`, and `L_N` collapses to the ordinary normalized symmetric Laplacian with an identically-zero imaginary part. Cora is undirected. Their ablation says Magnetic is the most load-bearing bias — but that was measured on GraphQA, where graphs may be directed. **Before investing in the magnetic machinery for Cora, check whether their GraphQA graphs are directed.** If they are, their strongest ablation result may not transfer to us at all, and the honest move is to test SPD and RRWP first on Cora and treat Magnetic as a directed-graph feature (relevant later for gene regulatory networks, which *are* directed).
2. **What SPD value do they assign to unreachable pairs?** `clamp(spd - 1, 0, max_spd - 1)` folds "infinity" into the same bucket as distance 8. Cora has disconnected components. Whether they use a sentinel or just a large integer determines whether "far" and "unreachable" are distinguishable. Not resolved — would need `src/utils/` SPD computation and the dataset code, which I did not read.
3. **Does the differential LR survive our setting?** Their bias parameters compete with a *pretrained, frozen* attention pattern over text tokens. Our tokens are encoder outputs with no pretrained meaning, so the balance of "how much should the bias move" may be completely different. The 100–600× ratio is a starting point, not a transferable constant. Sweep it.
4. **Does GTLM's equivariance claim survive our encoder?** Their nodes have text; ours have adjacency rows. `A_i,:` as a token is **not** permutation-invariant — permuting nodes permutes the row's *entries*, not just the row's position. Our E1/E2/E4/E5 encoders are `Linear(N, d)` over the raw row, which is equivariant only if the projection commutes with permutation, which it does not. **CLAUDE.md §11 already flags this ("Raw adjacency-row tokenization is not permutation invariant on its own") but the GTLM fix does not solve it** — GTLM's fix restores equivariance of the *transformer body*, not of the encoder. Our permutation test will fail at the encoder unless E6 (DeepSets) or E7 is used. This is worth resolving before writing the test, or the test will look like a bug.
5. **What is `magnetic_m` set to in their actual runs?** Default `m = 0` = all N eigenpairs, i.e. a full `O(N³)` complex eigendecomposition per graph. Their graphs are ego-subgraphs of ≤60 neighbours so this is trivial; full Cora at N=2,708 is not. Their collator has a `magnetic_m_collate` knob (a prefix slice, kept off the dataset cache key) but I did not find the value used.
6. **Does the `spd_depth` position mode help?** Unpublished, in the repo only. For us, `"reset"` degenerates RoPE to the identity entirely — so a structural position scheme might be strictly better, and is cheap to try. No results exist for it anywhere I looked.
7. **How do they handle sparse/large graphs for RRWP?** `compute_rrwp` is dense `O(K·N³)` batched `bmm`. Fine at N≤60. At Cora's N=2,708 that is 16 dense 2708³ matmuls per graph. Will need a sparse rewrite or subgraph sampling.
8. **Is the paper accepted?** The `REBUTTAL.md` implies an active NeurIPS review as of 2026-07-26. Cite as a preprint; check for a v2 / camera-ready before we submit, since the limitations section is slated to change.

---

## 9. Sources fetched

Primary — paper:
- https://arxiv.org/abs/2605.10247 — abstract, authors, date, category
- https://arxiv.org/html/2605.10247v1 — full text: §3.2 mechanism, Appendix A.1/A.2 bias math, Appendix B.1/B.2 proofs + numerical verification, Tables 1–12, Limitations, Future Work

Primary — repo (`raw.githubusercontent.com/DarioVajda/graph_model/main/...`, fetched via curl so the text below is verbatim source, not a summary):
- `README.md`
- `src/models/README.md` (21 KB user manual)
- `src/models/__init__.py`
- `src/models/attention.py`
- `src/models/bias.py` (88 KB — read `BaseBias`, `finalize_node_bias`, `SPDBias`, `LaplacianBias`, `RWSEBias`, `RRWPBias`, `MagneticBias`, `BIAS_TYPES`, `GraphAttentionBias`)
- `src/models/causal_lm.py`
- `src/models/config.py`
- `src/models/context.py`
- `src/models/dispatch.py`
- `src/models/flex_kernel.py`
- `src/models/io.py`
- `src/models/modeling_gtlm_llama.py`
- `src/models/structural_mask.py`
- `src/models/configs/gtlm_llama_v0_1b.json`
- `src/models/flex_attn/README.md` (37 KB optimization log)
- `src/models/flex_attn/REBUTTAL.md` (25 KB)
- `src/models/flex_attn/flex_core.py`
- `src/train/trainer.py`
- `src/train/model.py`
- `src/utils/text_graph_trainer_v2.py`
- `src/utils/text_graph_collator_v2.py`
- `src/utils/rrwp.py`
- `src/utils/magnetic_lap.py`
- `src/experiments/backward_compatibility/{README.md,__main__.py}`
- `src/experiments/permutation_equivariance/{README.md,__main__.py}`
- `src/experiments/tag_benchmarks/{config.py,train.py}`
- `src/experiments/graphqa/{config.py,train.py}`
- `https://api.github.com/repos/DarioVajda/graph_model/git/trees/main?recursive=1` — full 1,135-file tree, `truncated: false`

Secondary — the general "custom attention bias in HF" question:
- https://huggingface.co/docs/transformers/en/attention_interface — `AttentionInterface.register`, `AttentionMaskInterface`, custom 4D masks, `is_causal=False`, `create_*_mask`
- https://pytorch.org/blog/flexattention/ — `score_mod` / `mask_mod` signatures, ALiBi and learned-bias examples, recompilation rules, BlockMask cost
