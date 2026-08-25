# Environment & Compute Reality

Measured on the development machine, 2026-08-25. These are **local facts**, not literature —
the research agents cannot know them, so they are recorded here for the knowledge base.

---

## 1. Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 **Laptop** GPU |
| VRAM | **12,227 MiB (~12 GB)**, ~517 MiB already used by the desktop |
| Architecture | Blackwell, compute capability **sm_120** |
| Driver | 591.84, CUDA 13.1 |
| Power cap | 80 W (laptop part — sustained throughput will be well below a desktop 5070) |
| Disk free | 845 GB |
| OS | Windows 11 Home 26200, PowerShell + Git Bash |
| git | 2.55.0 — **project is not yet a git repo** |

### The single most important consequence

CLAUDE.md §9 notes GTLM ran on **a single A100/H100 80 GB**. We have **12 GB, on a
power-limited laptop part**. That is ~6.7x less memory. Every design decision below follows
from this. Full-graph Cora (N=2708) training is *not* the default path; **subgraph sampling is
the default path**, and full-graph is an evaluation-only mode.

---

## 2. Software stack — verified installable on this machine

Nothing is installed yet. Python 3.14.7 is the **only** interpreter present (no conda, no 3.11/3.12).
Python 3.14 is new enough that wheel availability was a genuine risk; it was checked with
`pip install --dry-run`. **Result: the entire stack resolves cleanly on cp314.**

| Package | Resolves to | Notes |
|---|---|---|
| `torch` (cu130 index) | **2.13.0+cu130**, `cp314` win_amd64 wheel | Required: Blackwell sm_120 needs cu128+. cu130 is the newest index and carries the newest torch. |
| `torchvision` | 0.28.0+cu130 | not needed, but resolves |
| `torch` (cu128 index) | 2.11.0+cu128 | older fallback |
| `torch-geometric` | **2.8.0.post1** | pure-python core; `torch-scatter`/`torch-sparse` compiled extras are **not** required for `Planetoid`, `GAE`, `GCNConv`, `GATConv`, `RandomLinkSplit` |
| `transformers` | **5.15.1** | ⚠️ **v5, not v4** — see hazard below |
| `peft` | 0.20.0 | |
| `accelerate` | 1.14.0 | |
| `torchmetrics` | 1.9.0 | |
| `ogb` | 1.3.6 | for Phase 2 molecular datasets |
| `scipy` / `scikit-learn` / `networkx` | 1.18.1 / 1.9.0 / 3.6.1 | |

Install command (not yet run — needs approval):

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install torch-geometric transformers peft accelerate torchmetrics ogb scipy scikit-learn networkx
```

> Do **not** install `torch` from default PyPI on Windows — those wheels are CPU-only.
> The `--index-url .../cu130` is what makes the GPU usable.

### ⚠️ Hazard: transformers is v5, not v4

CLAUDE.md and every reference codebase (GTLM, Graphormer-era code) predate transformers v5.
Attention dispatch, masking utilities, and the `attn_implementation` plumbing were all
reworked across the v4→v5 boundary. **Any attention-injection recipe found in a 2024–2025 repo
must be re-verified against the installed v5 source before we copy it.** This is the highest
engineering risk in the project and is tracked as such.

---

## 3. Memory math for the frozen backbone

> **⚠️ PARTIALLY SUPERSEDED — read §5 first.** This section assumes the **eager** attention path,
> because CLAUDE.md §11 says "Do not use FlashAttention... Use the eager attention path." The GTLM
> research notes ([gtlm.md](gtlm.md) C2) establish that this advice is **obsolete**: FlexAttention
> takes an arbitrary additive bias via a `score_mod` closure and never materialises the
> `(B,H,L,L)` tensor. GTLM's own benchmarks report eager 30 GB → flex **0.2 GB**. If flex works on
> this machine, the numbers below stop being binding. **But see §5 — flex needs Triton, and this is
> Windows.** Until that is settled, plan against the eager numbers below.


Working assumption: **Llama-3.2-1B** — config to confirm at load time, expected
`hidden=2048, layers=16, heads=32, kv_heads=8 (GQA), head_dim=64, vocab=128256`.

The custom additive bias forces the **eager** attention path (FlashAttention cannot take it,
per CLAUDE.md §11), so the `[B, H, N, N]` score matrix is **materialised**.

Per layer, per graph, batch 1, bf16 (2 bytes):

```
scores  [1, 32, N, N]  →  32·N²·2 bytes
probs   [1, 32, N, N]  →  32·N²·2 bytes   (saved for backward)
bias    [   32, N, N]  →  32·N²·2 bytes
```

| N | 32·N²·2 bytes (one tensor) | probs across all 16 layers | verdict at 12 GB |
|---|---|---|---|
| **2708** (full Cora) | **469 MB** | **7.5 GB** | ✗ no checkpointing; ✓ *only* with gradient checkpointing |
| 1024 | 67 MB | 1.07 GB | ✓ comfortable |
| **512** (proposed default) | **17 MB** | **268 MB** | ✓ trivial — room for larger batches |
| 256 | 4 MB | 67 MB | ✓ |

Budget at N=2708 **with gradient checkpointing** (recompute one layer at a time):

```
frozen weights, bf16                     ~2.5 GB
  minus unused embed_tokens + lm_head    -1.0 GB   ← we use the tokenizer/vocab for NOTHING
one layer's attention (scores+probs+bias) ~1.4 GB
saved hidden states, 16 × [2708, 2048]   ~0.18 GB
encoder/decoder/bias/LoRA + Adam states  ~0.2 GB
------------------------------------------------
peak                                     ~3.3 GB   ✓ fits
```

**Actionable consequences**

1. **Load `LlamaModel`, not `LlamaForCausalLM`.** The LM head is dead weight (~525 MB) and
   §13 of the spec says the vocab is unused. Dropping `embed_tokens` too saves another ~525 MB —
   careful, `tie_word_embeddings=True` on the 1B means they are the same tensor.
2. **Gradient checkpointing is mandatory** for any N > ~1024. Known gotcha: with a fully frozen
   base, checkpointing yields no gradients unless input embeddings require grad — call
   `model.enable_input_require_grads()`. In our case the encoder output already requires grad,
   which likely covers it, but verify explicitly.
3. **Default training config: k-hop subgraphs capped at N=512.** Full-graph N=2708 is an
   *evaluation* path, not a training path. This is the fallback CLAUDE.md §11 already lists;
   on this hardware it is not a fallback, it is the plan.
4. **GQA matters for bias shape.** `num_key_value_heads=8` vs `num_attention_heads=32`: the bias
   is added *after* KV expansion, so it must be shaped for **32 query heads**, not 8. Getting
   this wrong silently broadcasts the wrong bias to the wrong heads.
5. Speed, not memory, may be the real limit. Eager attention on an 80 W laptop part is
   bandwidth-bound on those `[32, N, N]` tensors, and gradient checkpointing doubles the forward
   pass. **Measure seconds/step at N=512 and N=2708 in week 1 before designing any experiment
   grid** — CLAUDE.md §12 already says to log wall-clock and peak memory at every stage. Do it
   from step one.

---

## 4. Open items for the local setup

- [ ] Confirm torch 2.13.0+cu130 actually initialises sm_120 on this driver (`torch.cuda.is_available()`, a matmul, and `torch.cuda.get_device_capability()` → expect `(12, 0)`).
- [ ] Confirm bf16 support and whether `torch.compile` works on Windows + cp314 (historically weak on Windows — assume **no** `torch.compile`).
- [ ] Decide gated-model access: Llama-3.2 requires a Hugging Face licence acceptance and token. **Qwen3-0.6B / 1.7B are ungated** and are the friction-free alternative — see the backbone comparison in the knowledge base.
- [ ] `git init` the project.
- [ ] Pin the whole stack in `requirements.txt` once the first end-to-end run succeeds.

---

## 5. The FlexAttention question — the decisive local unknown

[gtlm.md](gtlm.md) §5.3 and C2 show the reference implementation solved exactly our memory problem:
a `score_mod` closure gathers the bias **inside a Triton kernel**, so the dense `(B, H, N, N)` score
tensor is never allocated. Their measured result: eager 30 GB at L≈4.5k and OOM beyond it, versus
flex **0.2 GB** running to L≈256k, at 2.0×–9.6× the speed. That would make full-graph Cora
(N=2708) comfortable on 12 GB and make the subgraph fallback optional rather than mandatory.

**The catch is local, and the research agents could not have known it:** `torch.compile` and
FlexAttention depend on **Triton**, whose upstream support is Linux-first. On Windows it has
historically been unavailable or unofficial (`triton-windows`). torch 2.13 may or may not ship a
working Windows Triton for `cp314`.

**This single question decides the project's compute plan**, so it is the first thing to test after
install — before writing any model code:

```python
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
# 1. does a plain torch.compile work at all on this box?
# 2. does flex_attention run compiled on sm_120?
# 3. does a score_mod closure that gathers from a learned table produce gradients?
```

Three outcomes, three plans:

| Outcome | Training path | Consequence |
|---|---|---|
| Flex works | full-graph Cora N=2708 | subgraph sampling becomes optional; closest to GTLM |
| Flex compiles but is slow / unstable on Windows | eager at N≤512 subgraphs, flex for eval | as §3, gradient checkpointing on |
| No Triton at all | **eager only, N≤512 subgraphs, mandatory** | §3's numbers are the plan; full-graph Cora is eval-only under `no_grad` |

Keep the eager kernel regardless — GTLM keeps it as the numerical-parity oracle for the flex path,
and we should copy that. A `test_flex_matches_eager` test is cheap and catches silent bias-indexing
bugs, which are otherwise invisible.
