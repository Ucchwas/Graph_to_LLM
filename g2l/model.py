"""Arm assembly: encoder -> body -> decoder-input LayerNorm -> decoder, on one graph [N, N].
Bodies: FrozenBody (pretrained or random Llama), None (arm 2), ScratchBody (arm 4)."""
import math

import torch
from torch import nn

from g2l.decoders import D1Bilinear, D3PairMLP
from g2l.encoders import E1Linear


class ScratchBody(nn.Module):
    """Pre-LN transformer trained from scratch: dropout 0, no positional embeddings, no final
    norm (the decoder-input LayerNorm follows)."""

    def __init__(self, d_model: int, layers: int = 4, heads: int = 8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, tokens):
        return self.enc(tokens.unsqueeze(0))[0]


class GraphLLM(nn.Module):
    def __init__(self, encoder, body, decoder, d_model: int):
        super().__init__()
        self.encoder, self.body, self.decoder = encoder, body, decoder
        self.dec_norm = nn.LayerNorm(d_model)
        nn.init.constant_(self.dec_norm.weight, 1 / math.sqrt(d_model))
        nn.init.zeros_(self.dec_norm.bias)

    def embed(self, A):
        h = self.encoder(A)
        if self.body is not None:
            h = self.body(h)
        return self.dec_norm(h)

    def forward(self, A):
        return self.decoder(self.embed(A))

    def pairs(self, A, i, j):
        return self.decoder.pairs(self.embed(A), i, j)


def build_model(arm: str, n_nodes: int, d_model: int, T: float, body=None, decoder: str = "d1",
                gain: float | None = None, scratch_layers: int = 4, d3_hidden: int = 512, seed: int = 0):
    """arm in {pretrained, random, none, scratch}; `body` is the FrozenBody for the frozen arms.
    torch is re-seeded here so encoder / decoder start bit-identical across arms at a seed."""
    torch.manual_seed(seed)
    enc = E1Linear(n_nodes, d_model, gain=T / math.sqrt(d_model) if gain is None else gain)
    dec = D1Bilinear(d_model) if decoder == "d1" else D3PairMLP(d_model, hidden=d3_hidden)
    if arm == "scratch":
        body = ScratchBody(d_model, layers=scratch_layers)
    elif arm == "none":
        body = None
    else:
        assert body is not None, f"arm {arm} needs a FrozenBody"
    return GraphLLM(enc, body, dec, d_model)


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
        elif "bias_table" in name:
            groups["bias_table"].append(p)
        elif "lora_" in name:
            groups["lora"].append(p)
        else:
            raise ValueError(f"unclassified trainable parameter: {name}")
    assert groups["main"], "encoder/decoder are not trainable"
    return groups
