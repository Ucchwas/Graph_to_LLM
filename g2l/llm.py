"""The frozen Llama body and the injection route ("Implementation A").

Graph structure enters attention as a float 4-D `attention_mask` [1, 1 or H, N, N] that
transformers 5.15.1 adds to every layer's pre-softmax scores untouched
(masking_utils._preprocess_mask_arguments early-exits on 4-D masks). The tensor we pass is
the whole attention policy: zeros = bidirectional, no bias. This is library behaviour, not
API, so `FrozenBody.canary()` runs at every training start. If it ever fails, Implementation
B is the same-signature fallback: register an attention function with
`transformers.AttentionInterface.register` that adds the bias itself.

RoPE is the exact identity with position_ids = 0 for every token (cos = 1, sin = 0;
attention_scaling = 1 for rope type "llama3"); with a structure-only mask the body is then
permutation-equivariant over tokens.
"""
import torch
from torch import nn
from transformers import AutoConfig, AutoModel

MODEL_ID = "meta-llama/Llama-3.2-1B"


def load_config(model_id: str = MODEL_ID, revision: str | None = None):
    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    cfg.is_causal = False
    return cfg


def build_body(cfg, pretrained: bool, dtype=torch.bfloat16, seed: int = 0,
               model_id: str = MODEL_ID, revision: str | None = None):
    """Pretrained weights (arm 1) or the same config with seeded N(0, initializer_range)
    weights (arm 3). Eager attention and the dtype are fixed at construction."""
    cfg.is_causal = False
    if pretrained:
        return AutoModel.from_pretrained(model_id, config=cfg, revision=revision, dtype=dtype,
                                         attn_implementation="eager")
    torch.manual_seed(seed)
    return AutoModel.from_config(cfg, dtype=dtype, attn_implementation="eager")


def rmsnorm_names(llm) -> list[str]:
    return [n for n, _ in llm.named_parameters() if n.endswith("layernorm.weight") or n == "norm.weight"]


class FrozenBody(nn.Module):
    """Frozen body whose RMSNorm gains train (kept in fp32), embedding table stubbed, forward
    per the contract above. `T` = mean L2 norm of the original embedding rows."""

    def __init__(self, llm):
        super().__init__()
        assert llm.config._attn_implementation == "eager", "body must use eager attention"
        assert llm.rotary_emb.attention_scaling == 1.0, "RoPE identity needs attention_scaling == 1"
        llm.requires_grad_(False)
        W = llm.get_input_embeddings().weight
        norms = W.detach().float().norm(dim=-1)
        self.T = float(norms[norms > 1e-6].mean())
        self.dtype = W.dtype
        llm.embed_tokens = nn.Embedding(1, llm.config.hidden_size, dtype=self.dtype, device=W.device)
        llm.embed_tokens.requires_grad_(False)
        for name in rmsnorm_names(llm):
            p = llm.get_parameter(name)
            p.data = p.data.float()
            p.requires_grad_(True)
        self.llm = llm

    def checkpointing(self, on: bool):
        if on:
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.llm.gradient_checkpointing_disable()

    def _run(self, x, mask, is_causal=None):
        pos = torch.zeros(1, x.shape[1], dtype=torch.long, device=x.device)
        kw = {} if is_causal is None else {"is_causal": is_causal}
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=self.dtype == torch.bfloat16):
            out = self.llm(inputs_embeds=x, attention_mask=mask, position_ids=pos, use_cache=False, **kw)
        return out.last_hidden_state[0].float()

    def forward(self, tokens, bias=None):
        """tokens [N, d] -> H [N, d] fp32. bias: [1, 1 or H, N, N] in the body dtype, or None."""
        N = tokens.shape[0]
        x = tokens.to(self.dtype).unsqueeze(0)
        mask = torch.zeros(1, 1, N, N, dtype=self.dtype, device=x.device) if bias is None else bias
        assert mask.ndim == 4 and mask.dtype == self.dtype, "bias must be 4-D in the body dtype"
        return self._run(x, mask)

    @torch.no_grad()
    def canary(self, n: int = 16, atol: float = 1e-2):
        """The 4-D mask must reach the scores, and the zero mask must equal bidirectional no-mask."""
        d = self.llm.config.hidden_size
        dev = self.llm.embed_tokens.weight.device
        g = torch.Generator().manual_seed(0)
        t = torch.randn(n, d, generator=g).to(dev)
        h0 = self.forward(t)
        h1 = self.forward(t, bias=(torch.randn(1, 1, n, n, generator=g) * 0.5).to(dev, self.dtype))
        assert not torch.allclose(h0, h1), "4-D mask ignored: transformers changed; switch to Implementation B"
        hN = self._run(t.to(self.dtype).unsqueeze(0), None, is_causal=False)
        assert torch.allclose(h0, hN, atol=atol), "zero mask != bidirectional no-mask: config drift"
