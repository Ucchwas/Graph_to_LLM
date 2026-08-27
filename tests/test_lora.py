"""Phase 3B: LoRA on the frozen body, on the tiny fp32 Llama (no download)."""
import pytest
import torch

pytest.importorskip("peft")
from transformers import LlamaConfig

from g2l.llm import FrozenBody, build_body, rmsnorm_names
from g2l.model import build_model, param_groups
from g2l.train import make_optimizer

N, D = 20, 64
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def body(lora: bool):
    cfg = LlamaConfig(hidden_size=D, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=100, initializer_range=0.2, attn_implementation="eager")
    b = FrozenBody(build_body(cfg, pretrained=False, dtype=torch.float32, seed=0))
    return b.add_lora(r=4, alpha=8, dropout=0.0, targets=TARGETS) if lora else b


def graph(seed=0):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(N, N, generator=g) < 0.3).float(), diagonal=1)
    return up + up.T


def test_lora_params_groups_and_trainables():
    b = body(lora=True)
    names = [n for n, p in b.named_parameters() if p.requires_grad]
    lora = [n for n in names if "lora_" in n]
    assert len(lora) == 2 * 2 * len(TARGETS), "one A and one B per target projection per layer"
    assert all(n.endswith("layernorm.weight") or n.endswith(".norm.weight") for n in names if "lora_" not in n), \
        "only RMSNorm gains and adapters train"
    assert {"llm." + n for n in rmsnorm_names(b.llm)} <= set(names), "RMSNorm gains re-enabled after injection"
    m = build_model("random", N, D, 1.0, body=b, seed=0, bias=True)
    g = param_groups(m)
    assert len(g["lora"]) == len(lora) and g["rmsnorm"] and g["bias_table"]
    opt = make_optimizer(m, 1e-3, 1e-4, 0.01, lr_bias=1e-2, lr_lora=1e-4)
    assert any(pg["lr"] == 1e-4 and pg["weight_decay"] == 0.0 for pg in opt.param_groups)
    with pytest.raises(AssertionError):
        make_optimizer(m, 1e-3, 1e-4, 0.01, lr_bias=1e-2)  # LoRA present but no lr_lora


def test_lora_is_inert_at_init():
    A = graph()
    plain = build_model("random", N, D, 1.0, body=body(lora=False), seed=0, bias=True)
    lora = build_model("random", N, D, 1.0, body=body(lora=True), seed=0, bias=True)
    with torch.no_grad():
        assert torch.equal(plain(A), lora(A))
    lora.train()
    assert torch.allclose(plain(A), lora(A), atol=1e-6)


def test_lora_trains_and_base_stays_frozen():
    A = graph()
    m = build_model("random", N, D, 1.0, body=body(lora=True), seed=0, bias=True)
    base = {n: p.detach().clone() for n, p in m.body.llm.named_parameters() if not p.requires_grad}
    opt = make_optimizer(m, 1e-3, 1e-4, 0.01, lr_bias=1e-2, lr_lora=1e-2)
    m(A).sum().backward()
    lora = {n: p for n, p in m.named_parameters() if "lora_" in n}
    assert all(p.grad is not None for p in lora.values())
    assert all(p.grad.abs().sum() > 0 for n, p in lora.items() if "lora_B" in n), "B receives gradient"
    assert all(p.grad.abs().sum() == 0 for n, p in lora.items() if "lora_A" in n), "A's gradient is B^T (...) = 0 at init"
    opt.step()
    assert any(p.abs().sum() > 0 for n, p in lora.items() if "lora_B" in n), "B moved off zero"
    assert all(torch.equal(p, base[n]) for n, p in m.body.llm.named_parameters() if n in base), "base weights untouched"
    with torch.no_grad():
        out_after = m(A)
    plain = build_model("random", N, D, 1.0, body=body(lora=False), seed=0, bias=True)
    with torch.no_grad():
        assert not torch.allclose(plain(A), out_after, atol=1e-4), "the adapters change the function"
