"""Arm assembly: encoder -> body -> decoder-input LayerNorm -> decoder, on one graph [N, N].
Bodies: FrozenBody (pretrained or random Llama), None (arm 2), ScratchBody (arm 4)."""
import math

import torch
from torch import nn

from g2l.bias import SPDBias, spd_matrix_fast
from g2l.decoders import D1Bilinear, D3PairMLP
from g2l.encoders import E1Linear


class ScratchBody(nn.Module):
    """Pre-LN transformer trained from scratch: dropout 0, no positional embeddings, no final
    norm (the decoder-input LayerNorm follows). `bias` [1, heads, N, N] is added to every
    layer's attention logits, the same route as the frozen body."""

    def __init__(self, d_model: int, layers: int = 4, heads: int = 8):
        super().__init__()
        self.heads = heads
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, tokens, bias=None):
        return self.enc(tokens.unsqueeze(0), mask=None if bias is None else bias[0])[0]


class GraphLLM(nn.Module):
    """`bias`: an SPDBias or None. The bias is computed from the model's own input graph
    (the observed, masked adjacency at train time; the test-time input at eval), never from
    held-out cells -- the leak rule holds by construction."""

    def __init__(self, encoder, body, decoder, d_model: int, bias=None):
        super().__init__()
        self.encoder, self.body, self.decoder, self.bias = encoder, body, decoder, bias
        self.dec_norm = nn.LayerNorm(d_model)
        nn.init.constant_(self.dec_norm.weight, 1 / math.sqrt(d_model))
        nn.init.zeros_(self.dec_norm.bias)

    def attention_bias(self, A):
        if self.bias is None:
            return None
        b = self.bias(spd_matrix_fast(A, self.bias.max_dist))
        return b.to(self.body.dtype) if hasattr(self.body, "dtype") else b

    def embed(self, A):
        h = self.encoder(A)
        if self.body is not None:
            h = self.body(h, bias=self.attention_bias(A))
        return self.dec_norm(h)

    def forward(self, A):
        return self.decoder(self.embed(A))

    def pairs(self, A, i, j):
        return self.decoder.pairs(self.embed(A), i, j)


def build_model(arm: str, n_nodes: int, d_model: int, T: float, body=None, decoder: str = "d1",
                gain: float | None = None, scratch_layers: int = 4, d3_hidden: int = 512, seed: int = 0,
                bias: bool = False, max_dist: int = 8):
    """arm in {pretrained, random, none, scratch}; `body` is the FrozenBody for the frozen arms.
    torch is re-seeded here so encoder / decoder start bit-identical across arms at a seed;
    the bias table is zero-init, so `bias=True` starts at exactly the `bias=False` model."""
    torch.manual_seed(seed)
    enc = E1Linear(n_nodes, d_model, gain=T / math.sqrt(d_model) if gain is None else gain)
    dec = D1Bilinear(d_model) if decoder == "d1" else D3PairMLP(d_model, hidden=d3_hidden)
    if arm == "scratch":
        body = ScratchBody(d_model, layers=scratch_layers)
    elif arm == "none":
        body = None
    else:
        assert body is not None, f"arm {arm} needs a FrozenBody"
    spd_bias = None
    if bias:
        assert body is not None, "the attention bias needs a body"
        heads = body.heads if isinstance(body, ScratchBody) else body.llm.config.num_attention_heads
        spd_bias = SPDBias(heads, max_dist)
    return GraphLLM(enc, body, dec, d_model, bias=spd_bias)


def param_groups(model) -> dict[str, list]:
    """Trainables by group; any trainable outside the known groups raises."""
    groups = {"main": [], "rmsnorm": [], "scratch": [], "bias_table": [], "lora": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.split(".")[0] in ("encoder", "decoder", "dec_norm"):
            groups["main"].append(p)
        elif name.startswith("body.llm.") and (name.endswith("layernorm.weight") or name.endswith(".norm.weight")):
            groups["rmsnorm"].append(p)
        elif name.startswith("body.enc."):
            groups["scratch"].append(p)
        elif name == "bias.bias_table":
            groups["bias_table"].append(p)
        elif "lora_" in name:
            groups["lora"].append(p)
        else:
            raise ValueError(f"unclassified trainable parameter: {name}")
    assert groups["main"], "encoder/decoder are not trainable"
    return groups
