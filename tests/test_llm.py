"""The injection route and the arm assembly, on a tiny random Llama in fp32 (no download).
initializer_range 0.2 makes attention non-uniform so RoPE and mask effects are visible."""
import pytest
import torch

pytest.importorskip("transformers")
from transformers import LlamaConfig

from g2l.llm import FrozenBody, build_body
from g2l.model import build_model, param_groups
from g2l.train import make_optimizer, masked_loss

N, D = 20, 64


def tiny_cfg():
    return LlamaConfig(hidden_size=D, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, vocab_size=100, initializer_range=0.2, attn_implementation="eager")


@pytest.fixture
def body():
    return FrozenBody(build_body(tiny_cfg(), pretrained=False, dtype=torch.float32, seed=0))


def graph(seed=0):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(N, N, generator=g) < 0.3).float(), diagonal=1)
    return up + up.T


def test_canary_and_exact_bidirectional_parity(body):
    body.canary(atol=1e-6)
    t = torch.randn(N, D)
    h0 = body(t)
    hN = body._run(t.unsqueeze(0), None)  # config.is_causal=False is sticky: no-mask is bidirectional
    hC = body._run(t.unsqueeze(0), None, is_causal=True)
    assert torch.equal(h0, hN) and not torch.allclose(h0, hC)


def test_head_broadcast_parity(body):
    t = torch.randn(N, D)
    z1 = torch.zeros(1, 1, N, N)
    zH = torch.zeros(1, 4, N, N)
    assert torch.equal(body(t, bias=z1), body(t, bias=zH))
    with pytest.raises(RuntimeError):
        body(t, bias=torch.zeros(1, 2, N, N))  # KV-head count is wrong: heads are query heads


def test_body_equivariance_and_rope_negative_control(body):
    t = torch.randn(N, D)
    p = torch.randperm(N)
    assert torch.allclose(body(t[p]), body(t)[p], atol=1e-5)
    x, xp = t.unsqueeze(0), t[p].unsqueeze(0)
    m = torch.zeros(1, 1, N, N)
    h = body.llm(inputs_embeds=x, attention_mask=m, use_cache=False).last_hidden_state[0]   # default position_ids
    hp = body.llm(inputs_embeds=xp, attention_mask=m, use_cache=False).last_hidden_state[0]
    assert (h[p] - hp).abs().max() > 0.1, "RoPE on (default position_ids) should break equivariance"


def test_full_model_row_permutation(body):
    model = build_model("pretrained", N, D, body.T, body=body, seed=0).eval()
    A = graph()
    p = torch.randperm(N)
    with torch.no_grad():
        assert torch.allclose(model(A[p]), model(A)[p][:, p], atol=1e-5)


def test_construction_asserts(body):
    llm = body.llm
    assert llm.config._attn_implementation == "eager"
    assert llm.rotary_emb.attention_scaling == 1.0
    assert llm.embed_tokens.weight.shape[0] == 1 and not llm.embed_tokens.weight.requires_grad
    for n, p in llm.named_parameters():
        if n.endswith("layernorm.weight") or n == "norm.weight":
            assert p.dtype == torch.float32 and p.requires_grad
        else:
            assert p.dtype == body.dtype and not p.requires_grad
    with pytest.raises(AssertionError):
        body(torch.randn(N, D), bias=torch.zeros(1, 1, N, N, dtype=torch.float64))


def test_frozen_membership_and_groups(body):
    model = build_model("pretrained", N, D, body.T, body=body, seed=0)
    g = param_groups(model)
    assert len(g["rmsnorm"]) == 2 * 2 + 1 and not g["scratch"] and not g["bias_table"] and not g["lora"]
    trainable = {id(p) for ps in g.values() for p in ps}
    for n, p in model.named_parameters():
        assert p.requires_grad == (id(p) in trainable)
        if n.startswith("body.llm.layers") and "layernorm" not in n:
            assert not p.requires_grad
    model.body.llm.layers[0].self_attn.q_proj.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        param_groups(model)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_gradients_reach_every_trainable(body, checkpointing):
    model = build_model("pretrained", N, D, body.T, body=body, seed=0)
    body.checkpointing(checkpointing)
    model.train()
    A = graph()
    i, j = torch.triu_indices(N, N, 1)
    loss = masked_loss(model.pairs(A, i, j), A[i, j], torch.tensor(3.0))
    loss.backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, n
    assert sum(p.grad.norm() for p in model.encoder.parameters()) > 0
    make_optimizer(model, 1e-3, 1e-4, 0.01).step()


def test_paired_init_across_arms(body):
    a1 = build_model("pretrained", N, D, body.T, body=body, seed=0)
    a2 = build_model("none", N, D, body.T, seed=0)
    a4 = build_model("scratch", N, D, body.T, scratch_layers=1, seed=0)
    for m in (a2, a4):
        assert torch.equal(a1.encoder.proj.weight, m.encoder.proj.weight)
        assert torch.equal(a1.decoder.W, m.decoder.W)
        assert torch.equal(a1.dec_norm.weight, m.dec_norm.weight)
    assert param_groups(a4)["scratch"] and not param_groups(a2)["rmsnorm"]


def test_random_body_is_seeded():
    b1 = build_body(tiny_cfg(), pretrained=False, dtype=torch.float32, seed=3)
    b2 = build_body(tiny_cfg(), pretrained=False, dtype=torch.float32, seed=3)
    b3 = build_body(tiny_cfg(), pretrained=False, dtype=torch.float32, seed=4)
    q = lambda b: b.layers[0].self_attn.q_proj.weight
    assert torch.equal(q(b1), q(b2)) and not torch.equal(q(b1), q(b3))
