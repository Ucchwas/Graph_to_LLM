"""Arm assembly: encoder -> body -> decoder-input LayerNorm -> decoder, on one graph [N, N].
Bodies: FrozenBody (pretrained or random Llama), None (arm 2), ScratchBody (`scratch`; the
graph transformer `gt`), GNNBody (`gnn`, Phase 4: the graph-native backbone, later the GFM)."""
import math

import torch
from torch import nn

from g2l.bias import SPDBias, spd_matrix_fast
from g2l.decoders import D1Bilinear, D3PairMLP
from g2l.encoders import E1Linear

# nn.TransformerEncoderLayer's eval-mode fast path (torch._transformer_encoder_layer_fwd) casts a
# float `mask` to bool, which would turn the additive SPD bias into a hard mask at every val/test
# pass of the scratch arms; tests/test_bias.py::test_bias_survives_eval_mode guards this.
torch.backends.mha.set_fastpath_enabled(False)


class ScratchBody(nn.Module):
    """Pre-LN transformer, no positional embeddings, no final norm (the decoder-input LayerNorm
    follows). `bias` [1, heads, N, N] is added to every layer's attention logits, the same route
    as the frozen body.
    `scratch` (Phases 2-3): 8 heads, dropout 0, torch's cloned-layer init (every layer starts
    from the same draw), trained. `gt`: heads = width / 64, each layer its own draw
    (`clone_init=False`), optional dropout, optionally `frozen` at its random init -- then only
    the LayerNorm affines train, the counterpart of the frozen Llama's RMSNorm gains."""

    def __init__(self, d_model: int, layers: int = 4, heads: int = 8, dropout: float = 0.0,
                 frozen: bool = False, clone_init: bool = True):
        super().__init__()
        self.heads, self.frozen = heads, frozen

        def layer():
            return nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=dropout,
                                              batch_first=True, norm_first=True, activation="gelu")

        self.enc = nn.TransformerEncoder(layer(), layers, enable_nested_tensor=False)
        if not clone_init:
            self.enc.layers = nn.ModuleList([layer() for _ in range(layers)])
        if frozen:
            for name, p in self.enc.named_parameters():
                p.requires_grad = ".norm" in name

    def forward(self, tokens, bias=None):
        return self.enc(tokens.unsqueeze(0), mask=None if bias is None else bias[0])[0]


class GraphLLM(nn.Module):
    """`bias`: an SPDBias or None. The bias (and a GNN body's edge set) is computed from the
    model's own input graph (the observed, masked adjacency at train time; the test-time input
    at eval), never from held-out cells -- the leak rule holds by construction."""

    def __init__(self, encoder, body, decoder, d_model: int, bias=None):
        super().__init__()
        self.encoder, self.body, self.decoder, self.bias = encoder, body, decoder, bias
        self.dec_norm = nn.LayerNorm(d_model)
        nn.init.constant_(self.dec_norm.weight, 1 / math.sqrt(d_model))
        nn.init.zeros_(self.dec_norm.bias)

    def attention_bias(self, A):
        if self.bias is None:
            return None
        return self.bias(spd_matrix_fast(A, self.bias.max_dist), dtype=getattr(self.body, "dtype", torch.float32))

    def embed(self, A):
        h = self.encoder(A)
        if self.body is not None:
            h = self.body(h, A) if getattr(self.body, "needs_graph", False) else self.body(h, bias=self.attention_bias(A))
        return self.dec_norm(h)

    def forward(self, A):
        return self.decoder(self.embed(A))

    def pairs(self, A, i, j):
        return self.decoder.pairs(self.embed(A), i, j)


def build_model(arm: str, n_nodes: int, d_model: int, T: float, body=None, decoder: str = "d1",
                gain: float | None = None, scratch_layers: int = 4, d3_hidden: int = 512, seed: int = 0,
                bias: bool = False, max_dist: int = 8, heads: int | None = None, dropout: float = 0.0,
                frozen: bool = False, kind: str = "gcn"):
    """arm in {pretrained, random, none, scratch, gt, gnn}; `body` is the FrozenBody for the
    frozen Llama arms. torch is re-seeded here so encoder / decoder start bit-identical across
    arms at a seed; the bias table is zero-init, so `bias=True` starts at exactly the
    `bias=False` model. `heads` / `dropout` / `frozen` apply to `gt` (heads defaults to
    d_model / 64); `kind` / `dropout` / `scratch_layers` to `gnn`."""
    torch.manual_seed(seed)
    enc = E1Linear(n_nodes, d_model, gain=T / math.sqrt(d_model) if gain is None else gain)
    dec = D1Bilinear(d_model) if decoder == "d1" else D3PairMLP(d_model, hidden=d3_hidden)
    if arm == "scratch":
        body = ScratchBody(d_model, layers=scratch_layers)
    elif arm == "gt":
        body = ScratchBody(d_model, layers=scratch_layers, heads=heads or max(1, d_model // 64),
                           dropout=dropout, frozen=frozen, clone_init=False)
    elif arm == "gnn":
        from g2l.gnn import GNNBody
        assert not bias, "the attention bias has no place in a message-passing body"
        body = GNNBody(d_model, layers=scratch_layers, kind=kind, dropout=dropout)
    elif arm == "none":
        body = None
    else:
        assert body is not None, f"arm {arm} needs a FrozenBody"
    spd_bias = None
    if bias:
        assert body is not None, "the attention bias needs a body"
        heads_ = body.heads if isinstance(body, ScratchBody) else body.llm.config.num_attention_heads
        spd_bias = SPDBias(heads_, max_dist)
    return GraphLLM(enc, body, dec, d_model, bias=spd_bias)


def param_groups(model) -> dict[str, list]:
    """Trainables by group; any trainable outside the known groups raises. Trained body
    parameters (scratch / gt / gnn) form the `scratch` group; a frozen gt body's LayerNorm
    affines join `rmsnorm` (the frozen Llama's norm gains)."""
    groups = {"main": [], "rmsnorm": [], "scratch": [], "bias_table": [], "lora": []}
    body_frozen = getattr(model.body, "frozen", False)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.split(".")[0] in ("encoder", "decoder", "dec_norm"):
            groups["main"].append(p)
        elif name.startswith("body.llm.") and (name.endswith("layernorm.weight") or name.endswith(".norm.weight")):
            groups["rmsnorm"].append(p)
        elif "lora_" in name:
            groups["lora"].append(p)
        elif name.startswith("body.") and not name.startswith("body.llm."):
            groups["rmsnorm" if body_frozen else "scratch"].append(p)
        elif name == "bias.bias_table":
            groups["bias_table"].append(p)
        else:
            raise ValueError(f"unclassified trainable parameter: {name}")
    assert groups["main"], "encoder/decoder are not trainable"
    return groups
