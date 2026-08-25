# HF Mechanics — feeding embeddings, biasing attention, killing RoPE, going bidirectional

Research notes for `models/llm_wrapper.py`. Everything below was checked against the
**actual `main` branch of `huggingface/transformers`** (fetched 2026-08-25) and against
`pytorch/pytorch` and `huggingface/peft` sources. Line-level quotes are verbatim.

> **Version context — read this first.** `transformers` is now on **v5** (PyPI latest
> `5.15.1`, checked 2026-08-25). `peft` latest is `0.20.0`. The v5 attention/masking
> stack is *substantially different* from the v4 code that almost every blog post and
> StackOverflow answer describes. Two v4 idioms are now **dead**:
> `cache_position` is no longer a parameter of `LlamaModel.forward`, and
> `_update_causal_mask` no longer exists (replaced by `masking_utils.create_causal_mask`).
> Nothing here was verified against a locally installed package — **there is no
> `transformers` or `peft` installed in this repo's environment** (`ModuleNotFoundError`
> for both). Pin `transformers>=5.10` in `requirements.txt` and re-run the smoke tests
> in §"Code we can reuse" before trusting any of it.

---

## Verified facts (with source next to each)

### The four load-bearing facts

| # | Fact | Source |
|---|---|---|
| F1 | Llama's **eager** attention adds the `attention_mask` tensor to the scores pre-softmax, with no dtype/shape coercion: `attn_weights = attn_weights + attention_mask` | `src/transformers/models/llama/modeling_llama.py`, `eager_attention_forward`, line 206 |
| F2 | A user-supplied **4-D** `attention_mask` is passed through **verbatim** — mask creation early-exits before any bool conversion or `finfo.min` clamping | `src/transformers/masking_utils.py`, `_preprocess_mask_arguments`, lines 810-812 |
| F3 | `is_causal=False` is a **first-class documented forward kwarg** that flips the whole model to bidirectional attention, no library edit needed | `src/transformers/utils/generic.py`, `TransformersKwargs.is_causal` + `merge_with_config_defaults` |
| F4 | With `position_ids` all equal to 0, RoPE is **exactly** the identity for Llama-3.2 and Qwen3 (`cos=1, sin=0`, `attention_factor == 1.0`) | `modeling_llama.py` `LlamaRotaryEmbedding.forward` + `modeling_rope_utils.py` `_compute_llama3_parameters` |

F1 + F2 together are the whole ballgame: **the per-head additive graph bias can be
delivered as the `attention_mask` argument. No monkey-patching is required for a
layer-shared bias.**

### F1 — eager adds the mask to the logits, verbatim

```python
# transformers/models/llama/modeling_llama.py  (main, lines 191-213)
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)      # <-- GQA expanded HERE
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask              # <-- F1. Plain add.

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

There is **no** `mask.to(torch.bool)`, **no** `masked_fill`, **no** `clamp` on this path.
Whatever tensor arrives is added. Qwen3's `eager_attention_forward` is byte-identical
apart from an extra `sliding_window` kwarg.

### F2 — the 4-D early exit

```python
# transformers/masking_utils.py  (main, lines 810-824), inside _preprocess_mask_arguments
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None, None, None

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None, None, None

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
```

and its caller:

```python
# transformers/masking_utils.py, create_causal_mask, lines 935-939
    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask
```

Three things worth spelling out:

* The `.to(dtype=torch.bool)` coercion on line 823 is **guarded by `ndim == 2`**. Our 4-D
  float tensor never reaches it.
* The `finfo(dtype).min` clamping lives in `eager_mask`, which is only reached when the
  early exit did *not* fire. Our tensor is never clamped.
* The 4-D check is the **first** statement in the function, before the
  `_attn_implementation` membership check. So a 4-D mask survives even with a custom
  registered attention implementation name.

### F3 — `is_causal=False` is a supported forward kwarg

```python
# transformers/utils/generic.py, TransformersKwargs docstring
        is_causal (`bool`, *optional*)
            Can be set to False to enable bi-directional attention, i.e. use decoder Attention modules as encoders.
```

```python
# transformers/utils/generic.py, merge_with_config_defaults
        # Maybe temporarily overwrite config value to create the correct mask - kwarg takes precedence
        is_causal = kwargs.get("is_causal", getattr(self.config, "is_causal", None))
        if is_causal is not None:
            is_causal_in_config = hasattr(self.config, "is_causal")
            if is_causal_in_config:
                is_causal_original_value = self.config.is_causal
            # Set it to both config and kwargs (it's needed in both, and can come from only 1 of the sources)
            self.config.is_causal = is_causal
            kwargs["is_causal"] = is_causal

        try:
            ...
                output = func(self, *args, **kwargs)
        # Restore original config value
        finally:
            if is_causal is not None:
                if is_causal_in_config:
                    self.config.is_causal = is_causal_original_value
                else:
                    del self.config.is_causal
```

`LlamaModel.forward` carries `@merge_with_config_defaults`, so the decorator sets
`config.is_causal = False` *for the duration of the call* and restores it in a `finally`.
`create_causal_mask` reads it on its very first line:

```python
# transformers/masking_utils.py, create_causal_mask, lines 915-926
    # Power feature: if `is_causal` is False, then fallback to bi-directional mask for bi-directional attention.
    # It allows to use decoder-only models with bi-directional attention as well
    if not getattr(config, "is_causal", True):
        return create_bidirectional_mask(
            config, inputs_embeds, attention_mask,
            past_key_values=past_key_values,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
            allow_is_bidirectional_skip=allow_is_causal_skip,
        )
```

and the bidirectional mask function is a literal "always true":

```python
# transformers/masking_utils.py
def bidirectional_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """This creates a full bidirectional mask."""
    return q_idx >= 0
```

This is **thread-unsafe by construction** (it mutates the shared config object during the
forward). Do not share one model instance across threads while relying on it.

### F4 — RoPE at position 0 is exactly the identity

```python
# transformers/models/llama/modeling_llama.py, LlamaRotaryEmbedding.forward
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
```

```python
# transformers/models/llama/modeling_llama.py, apply_rotary_pos_emb
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
```

With `position_ids == 0`: `freqs = 0` → `emb = 0` → `cos = 1 * attention_scaling`,
`sin = 0`. And `attention_scaling` is **1.0** for every RoPE type our candidate models use:

```python
# transformers/modeling_rope_utils.py, _compute_llama3_parameters (Llama-3.2's rope_type)
    attention_factor = 1.0  # Unused in this type of RoPE
    ...
    return inv_freq_llama, attention_factor
```

```python
# transformers/models/llama/modeling_llama.py, compute_default_rope_parameters (Qwen3's rope_type)
    attention_factor = 1.0  # Unused in this type of RoPE
```

So `q_embed = q*1 + rotate_half(q)*0 = q`. **Bit-exact identity, not an approximation.**
(`yarn` and `longrope` *do* return `attention_factor != 1.0` — see `modeling_rope_utils.py`
lines 418-439 — which would rescale q and k by that constant even at position 0. Neither
Llama-3.2 nor Qwen3 uses them, but check `config.rope_parameters["rope_type"]` if you swap
backbones.)

### Other verified facts

| Fact | Source |
|---|---|
| `LlamaModel.forward` signature is `(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, **kwargs)` — **`cache_position` is gone** | `modeling_llama.py` lines 367-376; `grep cache_position modeling_llama.py` → no hits |
| `create_causal_mask` docstring still lists `cache_position` as "Deprecated and unused" | `masking_utils.py` lines 890-891 |
| XOR guard: `if (input_ids is None) ^ (inputs_embeds is not None): raise` — you must pass exactly one | `modeling_llama.py` line 377 |
| SDPA `attn_mask` float semantics: "added to the attention score", "must match the dtype of query, key, and value"; shape "broadcastable to (N,...,L,S)" | PyTorch 2.9 SDPA docs |
| SDPA `enable_gqa` "currently works only for Flash_attention and math kernel on CUDA tensor" | same |
| SDPA silently disables `is_causal` when a mask is present: `is_causal = q_length > 1 and attention_mask is None and is_causal` | `integrations/sdpa_attention.py` line 124 |
| Transformers refuses `output_attentions=True` on any non-eager implementation | `configuration_utils.py`, `validate_output_attentions`, lines 484-489 |
| The `config._attn_implementation` **setter performs no validation** — it just assigns `_attn_implementation_internal` | `configuration_utils.py` lines 419-435 |
| Validation only happens in `get_correct_attn_implementation` (i.e. via `from_pretrained(attn_implementation=...)` / `set_attn_implementation`) | `modeling_utils.py` lines 1926-1943 |
| `ALL_ATTENTION_FUNCTIONS[k] = v` writes the **instance-local** mapping; `AttentionInterface.register(k, v)` is a `@classmethod` that writes the **global** mapping | `utils/generic.py`, `GeneralInterface.__setitem__` vs `.register` |
| `_preprocess_mask_arguments` checks `ALL_MASK_ATTENTION_FUNCTIONS._global_mapping` specifically — a *locally* registered mask fn is invisible to it | `masking_utils.py` line 819 |
| `gradient_checkpointing_enable` defaults to `use_reentrant: False` | `modeling_utils.py` line 3205 |
| `enable_input_require_grads` hooks **`get_input_embeddings()`** (i.e. `embed_tokens`) | `modeling_utils.py` lines 2216-2246 |
| peft LoRA scaling is `lora_alpha / r`, or `lora_alpha / sqrt(r)` when `use_rslora=True` | `peft/tuners/lora/layer.py` lines 278-281 |
| peft `LoraConfig` defaults: `r=8, lora_alpha=8, lora_dropout=0.0, bias="none", use_rslora=False` | `peft/tuners/lora/config.py` lines 613-643 |
| peft's *default* `target_modules` for `llama`, `qwen2`, `qwen3` is `["q_proj", "v_proj"]` only | `peft/utils/constants.py` lines 85, 103-104 |
| GTLM = arXiv **2605.10247**, "Teaching LLMs to See Graphs", Dario Vajda, submitted 2026-05-11, cs.LG | arXiv API `id_list=2605.10247`, HTTP 200 |
| The GTLM abstract does say "only 0.015% additional parameters relative to the base model" | same abstract, fetched verbatim |
| `github.com/DarioVajda/graph_model` exists and is public (repo id 1145492144) | GitHub API |
| GTLM implements exactly the "register into `ALL_ATTENTION_FUNCTIONS`" strategy | `src/models/dispatch.py`, `register_gtlm_attention_functions()` |

---

## Corrections to CLAUDE.md

### C1. "Do not use FlashAttention... Use the eager attention path." — right conclusion, incomplete menu (confidence: certain)

§11 is correct that FlashAttention-2 cannot take an arbitrary additive bias, and correct
that eager works. But it omits the third option, and **GTLM's own repo — the reference
codebase CLAUDE.md §9 tells us to study — implements it and says it is the only path that
actually goes faster**:

```
# DarioVajda/graph_model, src/models/dispatch.py, module docstring
Backends:
  * ``eager`` — add the soft bias onto the dense structural mask and delegate to
    HuggingFace's own ``eager_attention_forward`` (no softmax reimplementation).
    The fused SDPA kernels are intentionally not offered: a custom dense bias
    makes flash ineligible and the mem-efficient kernel both buggy in backward
    (GQA + per-head bias) and pointless (the dense bias is already materialized),
    so it would only ever reduce to eager. ...
  * ``flex`` — implemented over :mod:`src.models.flex_kernel`; keeps the soft
    bias at node level ``(B,H,N,N)`` and gathers it inside the kernel (no
    token-level expansion). This is the only path that actually accelerates over
    eager (sparse block masks), at the cost of compiling a kernel per shape.
```

Amend §11 to: **eager for correctness and for the first working run; FlexAttention
(`score_mod`) is the scaling path when N gets large; SDPA is never worth it; FA2 is
impossible.** Note also the extra datum in that docstring — the SDPA *mem-efficient*
backward is described as **buggy** with GQA + per-head bias, which independently
corroborates pytorch#125674 (NaN gradients from the mem-efficient backend).

### C2. "RoPE disabled / reset per node" conflates two different things (confidence: certain)

§2 says "RoPE disabled / reset per node" and §9 says "RoPE reset per node". For **our**
architecture these are not the same and the distinction matters.

GTLM has *multiple tokens per node* (they feed node text), so "reset per node" means a
per-node local position counter — confirmed in their code:

```python
# DarioVajda/graph_model, src/models/causal_lm.py, prepare_inputs_for_generation
            # for them: node_ids gets the prompt node, and position_ids *continues the
            # prompt node's per-node local counter* (positions reset per node, and the
            # prompt node is packed last -> its last local position is position_ids[:, -1];
```

**We have exactly one token per node** (CLAUDE.md §2: `tokens [N, d_model]`, one per
node). So "reset per node" degenerates to "every token gets position 0", which by F4 is
*exact* RoPE removal. We do not need to disable rotary at all — and we should not, because
disabling it requires touching the module. Rewrite §2 as:
**"RoPE neutralised by giving every graph token `position_id = 0`; exactly the identity
because `cos(0)=1, sin(0)=0` and `attention_scaling == 1.0` for this backbone."**

### C3. "Total 173,056 bias parameters = 0.015% of Llama-3.2-1B" — number is self-consistent but the percentage doesn't reproduce (confidence: likely)

I verified the **0.015%** claim appears verbatim in the GTLM abstract (arXiv 2605.10247).
I did **not** find the 173,056 figure in any source I fetched — it is not in the abstract,
and I did not fetch the full paper body. It is however arithmetically consistent with
Llama-3.2-1B:

```
173,056 = 16 layers x 32 heads x 338
```

(Llama-3.2-1B has exactly 16 layers and 32 attention heads — verified from
`unsloth/Llama-3.2-1B/config.json`.) That is a strong signal the number is real.

But the percentage does not reproduce: `173,056 / 1.24e9 = 0.0140%`, which rounds to
**0.014%**, not 0.015%. To get 0.015% you need a denominator of ~1.154e9. Possible
explanations: they count a different parameter total, or they round up, or 173,056 is one
configuration and 0.015% is another. **Do not quote "173,056 = 0.015%" as a single
verified pair.** Quote the 0.015% (sourced, abstract) and treat 173,056 as unverified.

### C4. "SPD max distance 8, RRWP max steps 16, Magnetic Laplacian q = 0.25, dim 32" — two of four match the repo defaults, one does not (confidence: likely)

From `DarioVajda/graph_model, src/models/config.py`:

```
22:        rrwp: bool = False,
23:        max_rw_steps: int = 8,          # <-- CLAUDE.md says 16
35:        magnetic_dim: int = 32,         # <-- matches
40:        magnetic_q: float = 0.25,       # <-- matches
44:        landmark_d_max: int = 8,
```

`magnetic_dim=32` matches and `magnetic_q=0.25` matches. But the repo's **default**
`max_rw_steps` is **8**, not 16. I could not locate an `spd_max_distance` field in
`config.py` to check the "SPD max distance 8" claim. Caveat: repo *defaults* are not
necessarily the *reported experimental* hyperparameters — the paper may well use 16 on
GraphQA. Flag it, re-check against the paper body before copying, and don't treat the repo
default as refuting the paper. (This is a bias-design detail, outside my topic; handing it
to whoever owns that.)

### C5. §10's `llm_wrapper.py` description is right but understates how little code it needs (confidence: certain)

"frozen LLM + bias injection + LoRA + RoPE control" reads like it needs a subclassed model.
Given F1-F4 it is roughly **40 lines with zero subclassing** for the layer-shared-bias case.
See "Code we can reuse" below.

### C6. Nothing in CLAUDE.md warns that `enable_input_require_grads()` is a no-op for us (confidence: certain)

This is the single most likely silent failure in the whole build. See the gradient
checkpointing section below. The standard PEFT + gradient-checkpointing incantation that
every tutorial recommends **does nothing in our architecture** because it hooks the token
embedding table, which we never call. We happen to be saved by a different mechanism; that
mechanism breaks the moment anyone runs a frozen-encoder ablation.

---

## Mechanism / math (exact formulas, tensor shapes)

### Notation

| symbol | meaning | Llama-3.2-1B |
|---|---|---|
| `B` | batch | 1-8 |
| `N` | graph tokens = nodes (one token per node) | 512-2708 |
| `d_model` | `config.hidden_size` | 2048 |
| `H` | `config.num_attention_heads` (**query** heads) | 32 |
| `H_kv` | `config.num_key_value_heads` | 8 |
| `G` | `num_key_value_groups = H / H_kv` | 4 |
| `d_h` | `config.head_dim` | 64 |
| `L` | `config.num_hidden_layers` | 16 |

### The attention computation, exactly as executed

```
q = q_proj(x).view(B, N, H,    d_h).transpose(1,2)     ->  [B, H,    N, d_h]
k = k_proj(x).view(B, N, H_kv, d_h).transpose(1,2)     ->  [B, H_kv, N, d_h]
v = v_proj(x).view(B, N, H_kv, d_h).transpose(1,2)     ->  [B, H_kv, N, d_h]

q, k = apply_rotary_pos_emb(q, k, cos, sin)            # identity when position_ids == 0

k = repeat_kv(k, G)                                    ->  [B, H, N, d_h]   # GQA expansion
v = repeat_kv(v, G)                                    ->  [B, H, N, d_h]

S = (q @ k.transpose(2,3)) * scaling                   ->  [B, H, N, N]
S = S + attention_mask                                 # <-- our bias lands here
A = softmax(S, dim=-1, dtype=float32).to(q.dtype)      ->  [B, H, N, N]
O = (A @ v).transpose(1,2).reshape(B, N, H*d_h)
out = o_proj(O)                                        ->  [B, N, d_model]
```

`scaling = head_dim ** -0.5`. Matches CLAUDE.md §2's
`score(i,j) = (q_i . k_j)/sqrt(d) + bias(i,j)` exactly.

### GQA and the shape of the per-head bias — the answer to §7's question

**The bias tensor is indexed by QUERY heads, `H`, never by `H_kv`.**

The reason is one line of ordering in `eager_attention_forward`:

```python
key_states = repeat_kv(key, module.num_key_value_groups)   # [B,H_kv,N,d] -> [B,H,N,d]
...
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling   # -> [B,H,N,N]
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask
```

`repeat_kv` runs **before** the score matmul. By the time the mask is added, the KV heads
have already been broadcast up to `H`. So:

```
bias tensor shape  =  [B, H, N, N]        # H = num_ATTENTION_heads = 32 for Llama-3.2-1B
                   NOT [B, H_kv, N, N]    # 8 would silently broadcast-fail or mis-broadcast
```

Concretely for Llama-3.2-1B: **32**, not 8. For Qwen3-0.6B/1.7B: **16**, not 8.
For Qwen3-4B: **32**, not 8.

Practical consequences:

* **GQA does not save you any bias memory or bias parameters.** A per-head bias table has
  `L * H` independent head slots regardless of `H_kv`. Budget `L * H`, not `L * H_kv`.
* If you *want* to tie the bias within a KV group (a legitimate regulariser: 8 groups
  instead of 32 free heads), do it explicitly:
  `bias = bias_kv.repeat_interleave(G, dim=1)` on a `[B, H_kv, N, N]` tensor. That
  reproduces `repeat_kv`'s exact interleaving (`expand` on a new axis then `reshape`), so
  head `h` reads group `h // G`. Do **not** use `bias_kv.repeat(1, G, 1, 1)`, which tiles
  instead of interleaving and silently misaligns every head.
* A shape of `[B, 1, N, N]` broadcasts fine and gives a head-shared bias. Useful for the
  "no per-head lambda" ablation. CLAUDE.md §9 (GaLA) argues per-head is the right call, so
  `[B, H, N, N]` is the default and `[B, 1, N, N]` is the ablation.

### Memory: the number that decides everything

The score matrix is `B * H * N^2` elements, materialised **per layer**, and
`softmax(..., dtype=torch.float32)` means an **fp32** copy is alive alongside the bf16 one.

Per-layer live tensors on the eager path, in bytes, ignoring q/k/v:

```
S (bf16, pre-add)        2*B*H*N^2
S+bias (bf16)            2*B*H*N^2
softmax fp32 output      4*B*H*N^2
softmax bf16 output      2*B*H*N^2
bias itself (bf16)       2*B*H*N^2     (if kept for backward)
                        -----------
                     ~= 12*B*H*N^2 bytes per layer, worst case
```

For Llama-3.2-1B (`H=32`, `L=16`), `B=1`:

| N | `B*H*N^2` elements | ~ per layer | ~ all 16 layers |
|---|---|---|---|
| 256 | 2.10 M | 25 MB | 0.40 GB |
| 512 | 8.39 M | 101 MB | 1.6 GB |
| 1024 | 33.6 M | 403 MB | 6.4 GB |
| 2048 | 134 M | 1.6 GB | 25.8 GB |
| **2708 (full Cora)** | **235 M** | **2.8 GB** | **45 GB** |

**Full-graph Cora on one 80 GB card is not happening on the eager path**, and it is
marginal even with gradient checkpointing (which drops the x16 but keeps x1 plus recompute).
This confirms CLAUDE.md §11's warning quantitatively and settles the fallback:
**cap at N ~= 512-1024 nodes per batch via k-hop subgraph sampling**, exactly as §11
suggests. Start at N=512.

Qwen3-0.6B is cheaper per node (`H=16`) but has more layers (`L=28`): `L*H = 448` vs
Llama-3.2-1B's `512`. Roughly a 12% memory win, not a category change.

### Overflow trap when composing two additive masks

HF uses `torch.finfo(dtype).min` (not `-inf`) for blocked positions:

```python
# masking_utils.py, eager_mask
        min_dtype = torch.finfo(dtype).min
        mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
```

`finfo(bfloat16).min = -3.3895e38`. Adding a *small* learned bias to that is a no-op
(rounding). But **adding two such masks overflows to `-inf`**, and a row that is entirely
`-inf` makes `softmax` produce `NaN`. If you ever sum a structural mask and a padding mask,
use `finfo.min / 2` for each, or compose them in bool and convert once.

GTLM defends against the empty-row case with an explicit diagonal escape hatch:

```python
# DarioVajda/graph_model, src/models/structural_mask.py
    diag = (k_pos == q_pos).unsqueeze(0)                          # (1, q, kv) NaN guard
    allowed = allowed | diag
```

Copy that guard. For us the diagonal is node `i` attending to itself, which we want open
anyway.

---

## Code we can reuse (real snippets, real signatures)

### 1. Loading the backbone (no tokenizer, no lm_head)

```python
import torch
from transformers import AutoModel, AutoConfig

MODEL_ID = "meta-llama/Llama-3.2-1B"   # or "Qwen/Qwen3-0.6B" (Apache-2.0, ungated)

config = AutoConfig.from_pretrained(MODEL_ID)
config.is_causal = False               # F3: bidirectional for every forward

llm = AutoModel.from_pretrained(       # AutoModel -> LlamaModel: NO lm_head at all
    MODEL_ID,
    config=config,
    dtype=torch.bfloat16,              # NB: `torch_dtype=` is the deprecated v4 spelling
    attn_implementation="eager",       # CLAUDE.md §11 - mandatory for the additive bias
)
llm.requires_grad_(False)              # freeze the body; LoRA re-enables its own params
```

`AutoModel` gives you `LlamaModel`, whose `forward` returns `BaseModelOutputWithPast` with
`.last_hidden_state` of shape `[B, N, d_model]` — already post-`self.norm` (the final
RMSNorm). That is exactly the `H` that CLAUDE.md §2 wants to hand to the decoder. Using
`AutoModelForCausalLM` instead would drag in a `128256 x 2048` `lm_head` we never call.

**Dead-weight note.** `embed_tokens` is `vocab x d_model` and we never call it:
`128256 x 2048 = 262.7 M` params = **525 MB of bf16 (21% of Llama-3.2-1B)**; for Qwen3-0.6B
it is `151936 x 1024 = 155.6 M` = 311 MB (**26%**). You can reclaim it after loading:

```python
llm.embed_tokens = torch.nn.Embedding(1, config.hidden_size)   # 1-row stub, keeps attr alive
```

Keep the attribute (don't `del` it) so `get_input_embeddings()` still returns something and
`gradient_checkpointing_enable()` doesn't emit its "does not expose input embeddings"
warning. **Do this only after `from_pretrained`, and expect `save_pretrained` to be
unhappy about tied weights** — save our encoder/decoder/bias/LoRA separately anyway.

### 2. `inputs_embeds` — the exact call

```python
tokens = encoder(A, X, mask)                              # [B, N, d_model], OUR module
tokens = tokens.to(llm.dtype)                             # bf16 to match the frozen body

B, N, _ = tokens.shape
position_ids = torch.zeros(B, N, dtype=torch.long, device=tokens.device)   # F4: RoPE off

out = llm(
    inputs_embeds = tokens,          # [B, N, d_model] float
    attention_mask = graph_bias,     # [B, H, N, N] float, additive -- see below
    position_ids  = position_ids,    # [B, N] long, all zeros
    use_cache     = False,           # no generation; also required under grad ckpt
)
H_out = out.last_hidden_state        # [B, N, d_model]
```

**Pitfalls, each one checked against source:**

1. **Never pass `input_ids` as well.** `if (input_ids is None) ^ (inputs_embeds is not
   None): raise ValueError` (line 377). Exactly one.
2. **Do not pass `cache_position`.** It is not a parameter of `LlamaModel.forward` in v5
   (`grep` returns nothing in `modeling_llama.py`). It would be swallowed by `**kwargs`
   and silently ignored — or rejected by a stricter wrapper. Every v4-era tutorial that
   tells you to build a `cache_position` is out of date.
3. **`attention_mask` here is our 4-D bias, not a 2-D padding mask.** If you *also* need
   padding (ragged graphs in a batch), you must fold it into the same 4-D tensor —
   you cannot pass both. Add `finfo.min` at padded key columns of the bias tensor.
4. **`position_ids` must be `[B, N]`**, not `[N]`. `LlamaModel` auto-builds
   `torch.arange(...).unsqueeze(0)` if you omit it — which is the *causal-LM* default and
   would re-enable RoPE. Always pass it explicitly.
5. **`use_cache=False`.** With `use_cache=True` and no `past_key_values`, the model
   constructs a `DynamicCache` and grows it; harmless but wasteful, and
   `GradientCheckpointingLayer` will warn and override it anyway.
6. **dtype match.** Cast the encoder output to `llm.dtype`. A fp32 `inputs_embeds` into a
   bf16 body works (the first `nn.Linear` upcasts) but wastes memory and a fp32 mask
   would then force the whole score matrix to fp32.

### 3(a). Implementation A — the bias-as-`attention_mask` trick (RECOMMENDED FIRST)

Zero library patching, zero subclassing. Works today on eager.

```python
import torch, torch.nn as nn

class GraphAttentionBias(nn.Module):
    """Per-head additive attention bias from shortest-path distance (Graphormer-style).

    Produces ONE [B, H, N, N] tensor that is handed to the frozen LLM as its
    `attention_mask`. Because transformers' `_preprocess_mask_arguments` early-exits on
    any 4-D mask (masking_utils.py:811), the tensor reaches `eager_attention_forward`
    untouched and is added to the pre-softmax logits (modeling_llama.py:206).

    NOTE: one tensor for the whole stack => the bias is SHARED ACROSS LAYERS.
    For per-layer bias tables use implementation A2 or B.
    """
    def __init__(self, num_heads: int, max_dist: int = 8):
        super().__init__()
        self.max_dist = max_dist
        # buckets: 0..max_dist, plus one bucket for "unreachable / >max_dist"
        self.spd = nn.Embedding(max_dist + 2, num_heads)
        nn.init.zeros_(self.spd.weight)      # start as an exact no-op vs. the base LLM

    def forward(self, spd: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """spd: [B, N, N] long, shortest-path distance, `max_dist+1` for unreachable."""
        idx  = spd.clamp(max=self.max_dist + 1)
        bias = self.spd(idx)                       # [B, N, N, H]
        bias = bias.permute(0, 3, 1, 2)            # [B, H, N, N]
        return bias.to(dtype).contiguous()
```

Wiring it in:

```python
bias = bias_module(spd, dtype=llm.dtype)          # [B, H, N, N]
out  = llm(inputs_embeds=tokens, attention_mask=bias, position_ids=position_ids,
           use_cache=False)
```

**Why zero-init matters.** `nn.init.zeros_` makes the bias tensor identically 0 at step 0,
which is exactly the "no mask" state — an all-zeros additive mask is a fully bidirectional
mask with no bias. So the model starts *bit-identical* to the frozen base run
bidirectionally. That is our version of GTLM's "intra-node zero bias / exact backward
compatibility" property (CLAUDE.md §9), and it makes the "No attention bias" ablation in §8
a one-liner (`bias = torch.zeros(...)`).

**Bidirectionality comes free.** Zeros mean "allowed everywhere". You do **not** need
`config.is_causal = False` when you pass a 4-D mask — the mask *is* the whole policy and it
early-exits before `create_causal_mask` ever decides anything. Set `config.is_causal=False`
anyway as belt-and-braces for the no-bias ablation where you pass `attention_mask=None`.

**Verify it end to end (do this on day one):**

```python
# The bias must actually change the output.
b0 = torch.zeros(1, H, N, N, dtype=llm.dtype, device=dev)
b1 = torch.randn(1, H, N, N, dtype=llm.dtype, device=dev) * 0.5
h0 = llm(inputs_embeds=t, attention_mask=b0, position_ids=p, use_cache=False).last_hidden_state
h1 = llm(inputs_embeds=t, attention_mask=b1, position_ids=p, use_cache=False).last_hidden_state
assert not torch.allclose(h0, h1), "BIAS IS BEING IGNORED - the trick broke"

# Zero bias == no mask at all (both are 'attend everywhere, no bias')
hN = llm(inputs_embeds=t, attention_mask=None, position_ids=p, use_cache=False).last_hidden_state
# (requires config.is_causal = False, otherwise hN is the CAUSAL run and differs)
assert torch.allclose(h0, hN, atol=1e-2)
```

That first assert is the regression test that tells us instantly if a `transformers`
upgrade ever removes the 4-D early exit. **Put it in `tests/` and run it in CI.**

### 3(a2). Implementation A2 — per-layer bias via forward pre-hooks

Still no subclassing, still no patched attention, but gives each layer its own bias table.
Uses `register_forward_pre_hook(..., with_kwargs=True)`, which is allowed to *return*
replacement `(args, kwargs)`.

```python
def install_per_layer_bias(llm, bias_modules):
    """bias_modules: nn.ModuleList, one GraphAttentionBias per decoder layer."""
    handles = []
    for layer, bmod in zip(llm.layers, bias_modules):
        def hook(module, args, kwargs, _b=bmod):
            spd = module._graph_spd                       # stashed per batch, see below
            kwargs["attention_mask"] = _b(spd, dtype=module._graph_dtype)
            return args, kwargs
        handles.append(layer.register_forward_pre_hook(hook, with_kwargs=True))
    return handles

# per batch, before calling llm(...):
for layer in llm.layers:
    layer._graph_spd, layer._graph_dtype = spd, llm.dtype
```

This works because `GradientCheckpointingLayer.__call__` ends in
`super().__call__(*args, **kwargs)` — plain `nn.Module.__call__` — which runs forward
pre-hooks normally. Two caveats: under gradient checkpointing the hook fires **again** on
the backward recompute (fine, it is deterministic — but do not put side effects in it), and
you are now paying `L` separate `[B,H,N,N]` bias tensors instead of one.

### 3(b). Implementation B — a registered custom attention function

This is what GTLM does. More code, more surface area, but it is the **documented,
supported extension point** and it is the only option once you need FlexAttention.

The official contract (`docs/source/en/attention_interface.md`):

```python
def custom_attention(
    module: torch.nn.Module,          # required
    query: torch.Tensor,              # required
    key: torch.Tensor,                # required
    value: torch.Tensor,              # required
    attention_mask: Optional[torch.Tensor],   # required
    **kwargs,                         # accept additional kwargs from models
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    ...
```

Our version:

```python
import torch
from transformers.modeling_utils import AttentionInterface, ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.models.llama.modeling_llama import eager_attention_forward


def graph_eager_attention(module, query, key, value, attention_mask=None,
                          *, scaling, dropout=0.0, **kwargs):
    """Eager attention + this layer's learned per-head graph bias.

    Reads `module.graph_bias` (an nn.Module owned by this attention layer) and
    `module._graph_ctx` (the per-batch structural features, installed by our wrapper).
    Delegates the actual softmax to HF's own eager kernel so there is exactly one
    implementation of the attention math.
    """
    ctx = module._graph_ctx
    if ctx is None:
        raise RuntimeError("graph_eager ran without _graph_ctx; call through GraphLLM.forward")

    bias = module.graph_bias(ctx["spd"], dtype=query.dtype)     # [B, H, N, N]
    mask = bias if attention_mask is None else attention_mask + bias
    return eager_attention_forward(
        module, query, key, value, mask, scaling=scaling, dropout=dropout
    )


# Register GLOBALLY (classmethod -> _global_mapping). Instance assignment
# `ALL_ATTENTION_FUNCTIONS["graph_eager"] = fn` only writes the LOCAL mapping, and
# masking_utils checks `ALL_MASK_ATTENTION_FUNCTIONS._global_mapping` explicitly.
AttentionInterface.register("graph_eager", graph_eager_attention)

# MANDATORY companion. From the official docs:
#   "If no mask formatter is registered, mask creation is skipped and attention_mask=None
#    passes to the attention layers."
# Registering `eager_mask` means HF still builds the proper float mask (bidirectional,
# because config.is_causal=False) and hands it to us to add our bias onto.
AttentionMaskInterface.register("graph_eager", eager_mask)
```

Attaching the per-layer bias parameters and selecting the implementation:

```python
for layer in llm.layers:
    layer.self_attn.graph_bias  = GraphAttentionBias(config.num_attention_heads)
    layer.self_attn._graph_ctx  = None

llm.set_attn_implementation("graph_eager")     # passes validation: it is in valid_keys()
# ...or, to bypass validation entirely (GTLM's approach, also valid):
# llm.config._attn_implementation = "graph_eager"   # the setter does NO validation
```

Per batch, before the forward:

```python
for layer in llm.layers:
    layer.self_attn._graph_ctx = {"spd": spd}
```

**The two traps in implementation B, both verified in source:**

1. **Forget `AttentionMaskInterface.register` and you silently lose all masking.**
   `_preprocess_mask_arguments` line 819:
   `if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping: return True, None, ...`
   → `create_causal_mask` returns `None` → your attention function gets
   `attention_mask=None` and every padding/causal constraint vanishes without an error.
   GTLM *deliberately* exploits this (they build the mask themselves in `_graph_ctx`), but
   it will bite anyone who doesn't know.
2. **`output_attentions=True` becomes illegal.**
   `validate_output_attentions` raises for any implementation not in `["eager", None]`.
   Since CLAUDE.md §9 wants the "heads implicitly learn message passing" analysis, keep a
   plain-`eager` code path (implementation A) available for interpretability runs. This is
   a real argument for A.

### Which is more maintainable?

**Implementation A, decisively, for this project.**

| | A (4-D mask) | B (registered fn) |
|---|---|---|
| Lines of glue | ~10 | ~60 |
| Library internals touched | none — a public kwarg | 2 private registries, `_graph_ctx` smuggled onto modules |
| Survives a `transformers` upgrade | depends on 1 documented behaviour (F2) | depends on 2 registries + the attn-fn signature |
| Per-layer bias | no (A2 hook: yes) | yes |
| `output_attentions=True` | **works** | raises |
| FlexAttention path later | no | yes |
| Breaks how? | loudly — the day-one assert fails | **silently** — forget the mask registration and masking just disappears |

That last row is the decider. A's failure mode is a failing assert; B's is a model that
trains and converges to something subtly wrong. Given CLAUDE.md §12's build order —
"3. ...no attention bias yet... get it training end to end. 4. Add the attention bias" —
**A is exactly the right size for step 4.** Adopt B only when you actually need per-layer
bias tables or FlexAttention, and keep A's assert as the numerical oracle to validate B
against (they must agree to ~1e-3 in bf16 on a small graph).

### 4. RoPE control

**(i) Same position id for every token — RECOMMENDED.**

```python
position_ids = torch.zeros(B, N, dtype=torch.long, device=dev)
```

Exact identity by F4. One line. Nothing to restore. Works with LoRA, checkpointing, and
`torch.compile`. The value must be `0`, not an arbitrary constant: at `p != 0` the rotation
`R(p)` is still *common* to all tokens, so `(R q).(R k) = q.k` holds by orthogonality and
the logits are unchanged — but `q` and `k` themselves are rotated, which perturbs nothing
downstream *only* because `v` is unrotated. `p = 0` makes it identity at the tensor level,
which is strictly safer and trivially unit-testable.

**(ii) Truly disabling rotary.** Replace the module so `cos=1, sin=0` always:

```python
class NoRoPE(torch.nn.Module):
    def __init__(self, head_dim):
        super().__init__(); self.head_dim = head_dim
    def forward(self, x, position_ids):
        B, N = position_ids.shape
        cos = torch.ones (B, N, self.head_dim, dtype=x.dtype, device=x.device)
        sin = torch.zeros(B, N, self.head_dim, dtype=x.dtype, device=x.device)
        return cos, sin

llm.rotary_emb = NoRoPE(config.head_dim)
```

**(i) is cleaner. Use it.** (ii) is strictly worse for three reasons: it replaces a module
that `@use_kernelized_func` / `@use_kernel_forward_from_hub` may have specialised; it
breaks `save_pretrained`/`from_pretrained` round-tripping; and it produces the *same
numbers* as (i) while being 15 lines instead of 1. Note the shape: `cos`/`sin` must be
`[B, N, head_dim]` (not `head_dim//2`) because `emb = torch.cat((freqs, freqs), dim=-1)`
duplicates, and `apply_rotary_pos_emb` does `cos.unsqueeze(1)` to reach `[B,1,N,d_h]`.

**Unit test either way** (this is CLAUDE.md §11's permutation test, and RoPE is the thing
most likely to break it):

```python
perm = torch.randperm(N)
h  = model(A, X)[0]
hp = model(A[perm][:, perm], X[perm])[0]
assert torch.allclose(h[perm], hp, atol=1e-2)   # bf16 tolerance
```

If RoPE is live, this fails — because token `i` at sequence position `i` gets a different
rotation than at position `perm[i]`.

### 5. Non-causal / bidirectional attention

Three routes, in order of preference:

**(1) Pass a 4-D mask (implicit).** An all-zeros `[B,H,N,N]` additive mask *is* full
bidirectional attention. If you are already using implementation A, you are already
bidirectional and there is nothing to do. Verified via the F2 early exit — the causal mask
is never even constructed.

**(2) `is_causal=False` as a forward kwarg (explicit, official).**

```python
out = llm(inputs_embeds=tokens, position_ids=position_ids,
          attention_mask=None, is_causal=False, use_cache=False)
```

Documented in `TransformersKwargs`. `merge_with_config_defaults` temporarily writes
`config.is_causal = False`, `create_causal_mask` sees it and delegates to
`create_bidirectional_mask`, and the value is restored in a `finally`. This is the route
for the **"no attention bias"** ablation in CLAUDE.md §8, where we pass no mask at all.

**(3) `config.is_causal = False` on the config (sticky).**

```python
config.is_causal = False   # before or after from_pretrained; read via getattr at mask time
```

Same mechanism, permanent. Cleanest for our use because the model is *never* used causally.
Set it at construction and keep route (2) in your back pocket for per-call overrides.

**What NOT to do:** do not try `model.model.config.is_causal` on a `ForCausalLM` and expect
it to differ from `model.config` — they are the same object. And do not confuse the
`is_causal` *kwarg* with `LlamaAttention.is_causal` (the `self.is_causal = True` attribute
on line 228); that attribute is only read by `sdpa_attention_forward` as a fallback default
and has no effect on eager, whose behaviour is 100% determined by the mask tensor.

**None of these require editing library code.** CLAUDE.md §5's framing ("without editing
library code") is satisfied by an officially documented kwarg.

### 6. LoRA via peft

**Target modules.** peft's built-in default for `llama`/`qwen2`/`qwen3` is
`["q_proj", "v_proj"]` only (verified in `peft/utils/constants.py`). That is the original
LoRA paper's choice and it is **too narrow for us** — we are asking the frozen body to
process a modality it has never seen, so the MLP needs adaptation too. Specify explicitly:

```python
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

Identical names for Llama-3.2 and Qwen3 — verified in both `modeling_llama.py`
(`LlamaMLP`: `gate_proj`/`up_proj`/`down_proj`) and `modeling_qwen3.py`. Qwen3 additionally
has `q_norm`/`k_norm` (RMSNorm on `head_dim`) which are **not** `nn.Linear` and cannot be
LoRA targets.

**Rank / alpha.** CLAUDE.md §9's "rank 32 or 64, alpha = 2r" is the standard convention and
gives `scaling = lora_alpha / r = 2.0` (verified: `peft/tuners/lora/layer.py:281`,
`self.scaling[adapter_name] = lora_alpha / r`). Note peft's *defaults* are `r=8,
lora_alpha=8` → scaling 1.0, so you must pass both.

```python
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r              = 32,
    lora_alpha     = 64,          # alpha = 2r  ->  scaling = 64/32 = 2.0
    lora_dropout   = 0.05,
    bias           = "none",
    target_modules = TARGET_MODULES,
    task_type      = None,        # NOT "CAUSAL_LM": we have no lm_head and no labels
)
llm = get_peft_model(llm, lora_cfg)
```

`task_type=None` matters. `TaskType.CAUSAL_LM` makes peft wrap the model in
`PeftModelForCausalLM`, which expects `labels` and an `lm_head`. We have neither. A plain
`PeftModel` wrapper forwards `**kwargs` through to `LlamaModel.forward` untouched, which is
what we want.

**Combining peft with implementation B.** Order matters:

```python
# 1. attach OUR bias modules to the raw layers
for layer in llm.layers:
    layer.self_attn.graph_bias = GraphAttentionBias(...)
# 2. THEN wrap with peft
llm = get_peft_model(llm, lora_cfg)
# 3. reach the real layers through .base_model.model afterwards
for layer in llm.base_model.model.layers:
    layer.self_attn._graph_ctx = ctx
```

peft replaces `q_proj` with a `lora.Linear` wrapper but leaves `LlamaAttention` itself
intact, so `module.graph_bias` and `module._graph_ctx` survive and
`ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, ...)` still
dispatches to our function. With **implementation A this is a non-issue entirely** — peft
never sees the bias, which lives in our own module outside the wrapped LLM. Another point
for A.

**Keeping our params trainable while the base is frozen.** `get_peft_model` sets
`requires_grad=False` on everything it wraps and `True` only on `lora_*`. Our
encoder/decoder/bias modules must live **outside** the peft-wrapped submodule (they do, in
our `GraphLLM` wrapper), so they are untouched. Assert it:

```python
def assert_frozen_base(model):
    for n, p in model.named_parameters():
        is_ours = any(k in n for k in ("encoder", "decoder", "graph_bias", "lora_"))
        assert p.requires_grad == is_ours, f"{n}: requires_grad={p.requires_grad}"
```

**Parameter groups with different learning rates.** CLAUDE.md §9 is emphatic that this is
load-bearing ("Copy this or the bias parameters will not converge") — bias params start
random, LoRA starts from refined pretrained weights.

```python
def param_groups(model, lr_bias=5e-3, lr_lora=3e-5, lr_io=1e-4, wd=0.01):
    """Differential LRs per CLAUDE.md §9 (GTLM: 5e-3 bias vs 3e-5 LoRA on GraphQA)."""
    groups = {"bias": [], "lora": [], "io": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "graph_bias" in n:    groups["bias"].append(p)
        elif "lora_" in n:       groups["lora"].append(p)
        else:                    groups["io"].append(p)   # encoder + decoder
    for k, v in groups.items():
        assert v, f"empty param group: {k}"         # catches a silently-frozen module
    return [
        {"params": groups["bias"], "lr": lr_bias, "weight_decay": 0.0},   # no WD on a bias table
        {"params": groups["lora"], "lr": lr_lora, "weight_decay": wd},
        {"params": groups["io"],   "lr": lr_io,   "weight_decay": wd},
    ]

opt = torch.optim.AdamW(param_groups(model), betas=(0.9, 0.95))
```

Two details worth keeping: **`weight_decay=0.0` on the bias group** (decaying a
zero-initialised bias table toward zero directly fights the high LR that is supposed to
move it), and the `assert v` on each group — a typo in a module name otherwise produces an
empty group that trains nothing, silently, and looks exactly like "the bias didn't help".

Log per-group grad norms every N steps. If `||g_bias||` is ~1e-6 the bias is not learning
and the 5e-3 is doing nothing.

### 7. The whole wrapper, assembled

```python
import torch, torch.nn as nn
from transformers import AutoModel, AutoConfig

class GraphLLM(nn.Module):
    """Encoder -> frozen bias-steered LLM -> decoder.  CLAUDE.md §2.

    Implementation A: the per-head graph bias is delivered as the 4-D `attention_mask`.
    No library patching. See docs/research/hf-mechanics.md F1/F2.
    """
    def __init__(self, model_id, encoder, decoder, bias_module, lora_cfg=None):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_id)
        cfg.is_causal = False                   # F3 (the 4-D mask already implies it)
        self.llm = AutoModel.from_pretrained(
            model_id, config=cfg, dtype=torch.bfloat16, attn_implementation="eager",
        )
        self.llm.requires_grad_(False)
        if lora_cfg is not None:
            from peft import get_peft_model
            self.llm = get_peft_model(self.llm, lora_cfg)

        self.encoder, self.decoder, self.bias = encoder, decoder, bias_module
        self.d_model   = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads    # H, NOT num_key_value_heads -- GQA note

    def forward(self, A, X=None, spd=None, mask=None):
        tokens = self.encoder(A, X, mask).to(torch.bfloat16)       # [B, N, d_model]
        B, N, _ = tokens.shape
        dev = tokens.device

        if self.training and not tokens.requires_grad:
            tokens.requires_grad_(True)          # grad-checkpointing guard, see below

        bias = (self.bias(spd, dtype=torch.bfloat16) if spd is not None
                else torch.zeros(B, self.num_heads, N, N, dtype=torch.bfloat16, device=dev))

        out = self.llm(
            inputs_embeds  = tokens,
            attention_mask = bias,                                  # [B, H, N, N]
            position_ids   = torch.zeros(B, N, dtype=torch.long, device=dev),   # F4
            use_cache      = False,
        )
        return self.decoder(out.last_hidden_state)                  # [B, N, N] logits
```

### 8. Gradient checkpointing + bf16 + frozen base

**The standard advice does not apply to us, and this is the most dangerous gotcha here.**

The famous incantation is `model.enable_input_require_grads()`. Here is what it actually
does:

```python
# transformers/modeling_utils.py, lines 2216-2246
    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        ...
            input_embeddings = module.get_input_embeddings()
        ...
            hooks.append(input_embeddings.register_forward_hook(make_inputs_require_grads))
```

It registers a **forward hook on `embed_tokens`**. We pass `inputs_embeds` and therefore
**never call `embed_tokens`**. The hook never fires. `enable_input_require_grads()` is a
**complete no-op for this architecture.**

And `gradient_checkpointing_enable` will not save us either:

```python
# transformers/modeling_utils.py, lines 3226-3234
        needs_embedding_grads = self.main_input_name == "input_ids"
        enable_input_grads = needs_embedding_grads or getattr(self, "_hf_peft_config_loaded", False)
        if enable_input_grads:
            self.enable_input_require_grads()
```

(It *would* fire the `_hf_peft_config_loaded` branch — and still be a no-op, for the same
reason.)

**Why we are nonetheless fine:** our encoder is trainable, so `tokens = encoder(...)` is a
non-leaf tensor with `requires_grad=True`. The checkpointed segment therefore has a
grad-requiring input by construction, autograd records the graph, and backward flows to
the LoRA params and to the encoder. **The thing that saves us is the trainable encoder,
not `enable_input_require_grads`.**

**When it breaks:** the moment `inputs_embeds` stops requiring grad. That happens in a
frozen-encoder ablation, under `torch.no_grad()` around the encoder, if anyone calls
`.detach()` while debugging, or in a cached-token-embeddings speedup. Then, with
`use_reentrant=True`, the checkpointed block has no grad-requiring input, autograd skips
it, and **the LoRA parameters get `grad=None` with no error message**. Loss goes flat.
Days get lost.

**Defence — do all four:**

```python
llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

1. **`use_reentrant=False` explicitly.** It is already the v5 default
   (`modeling_utils.py:3205`), but pass it anyway so a version change cannot flip it.
   Non-reentrant checkpointing tracks grad-requiring tensors captured in the closure and
   is far more forgiving of exactly this failure mode.
2. **Force it regardless of provenance**, so a frozen-encoder ablation cannot silently
   break: `tokens.requires_grad_(True)` right before the LLM call, when
   `self.training and not tokens.requires_grad` (already in the `GraphLLM` above).
3. **`use_cache=False`.** Mandatory — `GradientCheckpointingLayer.__call__` warns and
   overrides it, and `merge_with_config_defaults` also forces it off when
   `self.gradient_checkpointing and self.training`.
4. **Assert grads exist after the first `backward()`:**

```python
loss.backward()
n_none = sum(p.grad is None for p in model.parameters() if p.requires_grad)
assert n_none == 0, f"{n_none} trainable params got NO gradient - checkpointing/graph break"
```

**bf16 specifics, all fine:**

* Load the frozen base in bf16 (`dtype=torch.bfloat16`). It is never updated, so there is
  no master-weight problem and **no `GradScaler` is needed** (bf16 has fp32's exponent
  range). Do not use fp16 for a frozen base — the `finfo(float16).min = -65504` headroom
  interacts badly with additive masks.
* Keep **our** trainable modules (encoder, decoder, bias table) in **fp32** and cast only
  at the LLM boundary. The bias table's updates are `lr=5e-3 * small grads`; in bf16
  (8 bits of mantissa) many of those updates round to zero. This is a real risk for
  precisely the parameters CLAUDE.md §9 warns will not converge.
* `softmax(..., dtype=torch.float32)` already upcasts internally, so the attention
  softmax is fp32 regardless — good for numerical safety, and the reason for the x4 term
  in the memory table above.
* Checkpointing does **not** reduce the peak `[B,H,N,N]` allocation *within* a layer; it
  removes the xL factor across layers. The N<=512-1024 cap from the memory table stands
  either way.

---

## Numbers to beat / hyperparameters to copy

### Backbone comparison

All figures read from the shipped `config.json` (Llama and Gemma via the ungated `unsloth/`
mirrors; `meta-llama/*` and `google/*` return 401 to anonymous fetches).

| Model | `d_model` | `L` | `H` (Q heads) | `H_kv` | `G` | `head_dim` | `L*H` | bf16 weights | vocab | License |
|---|---|---|---|---|---|---|---|---|---|---|
| **Llama-3.2-1B** | 2048 | 16 | **32** | 8 | 4 | 64 | 512 | ~2.5 GB | 128256 | Llama 3.2 Community (gated) |
| Llama-3.2-3B | 3072 | 28 | **24** | 8 | 3 | 128 | 672 | ~6.4 GB | 128256 | Llama 3.2 Community (gated) |
| **Qwen3-0.6B** | 1024 | 28 | **16** | 8 | 2 | 128 | 448 | ~1.2 GB | 151936 | Apache-2.0 |
| Qwen3-1.7B | 2048 | 28 | **16** | 8 | 2 | 128 | 448 | ~3.4 GB | 151936 | Apache-2.0 |
| Qwen3-4B | 2560 | 36 | **32** | 8 | 4 | 128 | 1152 | ~8.0 GB | 151936 | Apache-2.0 |
| Qwen2.5-0.5B | 896 | 24 | 14 | 2 | 7 | 64 | 336 | ~1.0 GB | 151936 | Apache-2.0 |
| SmolLM2-1.7B | 2048 | 24 | 32 | **32** | **1 (MHA)** | 64 | 768 | ~3.4 GB | 49152 | Apache-2.0 |
| Gemma-3-1B | 1152 | 26 | 4 | 1 | 4 | 256 | 104 | ~2.0 GB | 262144 | Gemma (gated) |
| Llama-3.1-8B | 4096 | 32 | 32 | 8 | 4 | 128 | 1024 | ~16 GB | 128256 | Llama 3.1 Community (gated) |

`L*H` is the memory-relevant figure: attention-score memory scales as `L * H * N^2` and
per-head bias parameters scale as `L * H * buckets`.

**Notes that matter:**

* **Qwen3's `head_dim` (128) != `hidden_size / num_heads`** (1024/16 = 64) for the 0.6B and
  1.7B. `q_proj` is `1024 -> 16*128 = 2048`. Always read `config.head_dim`; never compute
  it. Transformers does this correctly
  (`getattr(config, "head_dim", hidden_size // num_heads)`) but our own code must too.
* **Gemma-3-1B is disqualified.** `sliding_window: 512` with `sliding_window_pattern: 6`
  means 5 of every 6 layers only attend within a 512-token window. Global graph attention
  is the entire point. Also `num_key_value_heads: 1` and only 4 query heads — 4 heads is
  far too few for a per-head bias with per-head `lambda_h` (CLAUDE.md §9/GaLA).
* **Qwen2.5-0.5B** has `sliding_window: 32768`, larger than any graph we will run, so it is
  effectively full attention — but `H_kv=2` with `G=7` is an unusual ratio and the model is
  a generation behind. Skip.
* **SmolLM2-1.7B is the only MHA candidate** (`H_kv = H = 32`, `G=1`). If GQA broadcasting
  ever proves to be the bug, this is the clean control that removes the variable entirely.
  Worth remembering; not worth starting with.
* **Qwen3 has no sliding-window layers** (`sliding_window: null`), so `has_sliding_layers`
  is False and the `causal_mask_mapping` dict collapses to a single `"full_attention"`
  entry. Our 4-D mask flows through untouched.
* Qwen3's `Qwen3Model.forward` accepts `attention_mask` as a **dict** keyed by layer type
  (`if not isinstance(causal_mask_mapping := attention_mask, dict)`). Passing a plain 4-D
  tensor is fine and takes the normal path. Passing a dict is the escape hatch if a future
  sliding-window Qwen ever matters.

### Recommendation for the first working run

**Llama-3.2-1B**, with **Qwen3-0.6B** as the immediate second point.

Reasons, in order:

1. **Direct comparability with GTLM.** CLAUDE.md §9 calls GTLM "the closest existing work";
   their reported numbers are on Llama-3.2-1B. The 173,056 bias-parameter figure factors
   exactly as `16 x 32 x 338` against *this* model's 16 layers and 32 heads (§C3), so their
   bias-parameter budget and their 5e-3 / 3e-5 LR split transfer without reinterpretation.
   Changing the backbone forfeits the one external number we can check ourselves against.
2. **Sits comfortably on one 40 GB card at N=512**: 2.5 GB frozen weights + ~1.6 GB
   attention (no checkpointing) + LoRA/optimizer/encoder/decoder. Even N=1024 fits at
   ~6.4 GB of attention.
3. `d_model = 2048` keeps our own modules cheap: E1 is `Linear(2708, 2048)` ~= 5.5 M params,
   D1's `W` is `2048^2` ~= 4.2 M.
4. 32 heads gives the per-head bias / per-head `lambda_h` idea (CLAUDE.md §9, GaLA) enough
   resolution to be meaningful. 4 heads (Gemma) would not.

**The one catch: `meta-llama/Llama-3.2-1B` is gated** — anonymous `config.json` fetches
return 401. Someone must accept the license on the Hub and set `HF_TOKEN`. If that is
friction, **Qwen3-0.6B is the drop-in fallback**: Apache-2.0, ungated, half the weights
(1.2 GB), 12% less attention memory (`L*H = 448` vs 512), and a *smaller* `d_model` (1024)
which makes our encoder/decoder cheaper still. Its only real cost is losing GTLM
comparability.

Run both. They differ on `H` (32 vs 16), `d_model` (2048 vs 1024), and license, so
agreement between them is decent evidence the effect is not a backbone artifact — a cheap
addition to CLAUDE.md §8's ablation grid.

### Hyperparameters to copy

| Knob | Value | Provenance |
|---|---|---|
| LoRA `r` | 32 | CLAUDE.md §9 (GTLM) |
| LoRA `alpha` | 64 (`= 2r` -> scaling 2.0) | CLAUDE.md §9; formula verified `peft/tuners/lora/layer.py:281` |
| LoRA `dropout` | 0.05 | convention (peft default is 0.0) |
| LoRA `target_modules` | all 7 proj matrices | peft's default `["q_proj","v_proj"]` is too narrow for a new modality |
| LR, bias params | **5e-3** | CLAUDE.md §9 (GTLM GraphQA) |
| LR, LoRA | **3e-5** | CLAUDE.md §9 (GTLM GraphQA) |
| LR, encoder/decoder | 1e-4 | our own; between the two — they start random like the bias but are much larger |
| weight decay | 0.01 on LoRA + encoder/decoder, **0.0 on bias** | see §6 |
| dtype | bf16 frozen base, **fp32 trainables** | see §8 |
| `attn_implementation` | `"eager"` | CLAUDE.md §11 |
| `use_cache` | `False` | required under checkpointing |
| N per batch | **512** to start, 1024 if memory allows | memory table above |
| SPD max distance | 8 (-> 10 buckets with self and unreachable) | CLAUDE.md §9; repo default unconfirmed (§C4) |
| bias init | **zeros** | makes step 0 bit-identical to the base model |

Per-head SPD bias table sizes at `max_dist=8` (10 buckets), per-layer-per-head:
Llama-3.2-1B `16 * 32 * 10 = 5,120` params; Qwen3-0.6B `28 * 16 * 10 = 4,480`.
Both are ~0.0004% of the base — three orders of magnitude below the LoRA budget, and far
below GTLM's 173,056 (which covers RRWP + magnetic Laplacian on top of SPD).

---

## Open questions

1. **Does the 4-D early exit have a deprecation plan?** It is load-bearing for
   implementation A and is documented only as an inline comment ("it was already prepared,
   or it is custom"), not in the public API docs. The day-one assert in §3(a) is our
   canary. Worth watching `masking_utils.py` on upgrades.
2. **Per-layer vs layer-shared bias — how much does it actually buy?** GTLM uses per-layer
   tables; GaLA applies bias only in the first half of layers and reports that forcing it
   everywhere "damages the language machinery". Implementation A gives us layer-shared for
   free. **Measure layer-shared first**; only pay for A2/B if the delta justifies it. This
   is a cheap, publishable ablation nobody seems to have run.
3. **Is FlexAttention worth it at our N?** GTLM's docstring says flex is "the only path
   that actually accelerates over eager", but it "compiles a kernel per shape" — and with
   variable-N subgraph sampling we would recompile constantly. Padding to a fixed N would
   fix that at the cost of wasted compute. Unresolved; benchmark before committing.
4. **Does `attention_scaling` stay 1.0?** Verified for `default` and `llama3` rope types
   today. If we ever move to a YaRN/longrope-scaled checkpoint, F4's exactness dies and
   `position_ids=0` starts rescaling q and k by a constant. Add a startup assert:
   `assert llm.rotary_emb.attention_scaling == 1.0`.
5. **Unverified from CLAUDE.md §9:** the 173,056 figure (§C3), the "SPD max distance 8" and
   "RRWP max steps 16" values (§C4), the "difference ~2e-5" backward-compatibility check,
   and the "~3x slower training" claim. I read the GTLM *abstract* and parts of the *repo*,
   **not the paper body**. Someone should fetch `arxiv.org/pdf/2605.10247` and check them.
6. **Does `torch.compile` survive `config.is_causal` mutation?**
   `merge_with_config_defaults` writes to the config object inside the forward, which
   smells like a guard failure or a recompile per call. Untested. Prefer route (3) — set
   `config.is_causal = False` once, statically — if we ever compile.
7. **Ragged batching.** Folding a padding mask into the same 4-D bias tensor (since we
   cannot pass both) is straightforward but untested, and it is where the `finfo.min`
   overflow trap will actually bite. Alternative: batch size 1 with gradient accumulation,
   which sidesteps it entirely and is probably right for the first run.

---

## Sources fetched

**transformers (github.com/huggingface/transformers, `main`, fetched 2026-08-25)**
- `src/transformers/models/llama/modeling_llama.py` (513 lines, read in full)
- `src/transformers/models/qwen3/modeling_qwen3.py`
- `src/transformers/masking_utils.py` (80 KB)
- `src/transformers/modeling_utils.py` (270 KB)
- `src/transformers/integrations/sdpa_attention.py` (171 lines, read in full)
- `src/transformers/modeling_layers.py`
- `src/transformers/modeling_rope_utils.py`
- `src/transformers/configuration_utils.py`
- `src/transformers/utils/generic.py`
- `docs/source/en/attention_interface.md`

**peft (github.com/huggingface/peft, `main`)**
- `src/peft/utils/constants.py`
- `src/peft/tuners/lora/config.py`
- `src/peft/tuners/lora/layer.py`

**GTLM (github.com/DarioVajda/graph_model, `main`)**
- `src/models/dispatch.py` (read in full — the registered-attention-function pattern)
- `src/models/attention.py`, `src/models/modeling_gtlm_llama.py`
- `src/models/structural_mask.py` (read in full)
- `src/models/causal_lm.py`, `src/models/config.py`
- repo metadata via GitHub API

**Other**
- arXiv API `id_list=2605.10247` — GTLM abstract, confirmed
- https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://github.com/pytorch/pytorch/issues/125674 (NaN grads, mem-efficient backend)
- PyPI JSON API: `transformers` 5.15.1, `peft` 0.20.0
- HF Hub `config.json`: `Qwen/Qwen3-{0.6B,1.7B,4B}`, `Qwen/Qwen2.5-0.5B`,
  `HuggingFaceTB/SmolLM2-1.7B`, `unsloth/Llama-3.2-{1B,3B}`, `unsloth/Llama-3.1-8B`,
  `unsloth/gemma-3-1b-pt`

**Not fetched (flagged as unverified above):** the GTLM paper body
(`arxiv.org/pdf/2605.10247`), GaLA (arXiv 2606.15633), Depth-Width (arXiv 2503.01805).
