"""Real-weight checks (Phase 2 step 3). Skipped unless the pinned Llama-3.2-1B snapshot is
in the local Hugging Face cache; seconds on GPU, under a minute on CPU."""
import math

import pytest
import torch
import yaml

pytest.importorskip("transformers")
from transformers import AutoModel

from g2l.encoders import E1Linear
from g2l.llm import FrozenBody, build_body, load_config

CFG = yaml.safe_load(open("configs/phase2.yaml"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = 64


@pytest.fixture(scope="module")
def stock():
    try:
        cfg = load_config(CFG["model_id"], CFG["revision"])
        m = AutoModel.from_pretrained(CFG["model_id"], config=cfg, revision=CFG["revision"], dtype=torch.bfloat16,
                                      attn_implementation="eager", local_files_only=True)
    except Exception as e:
        pytest.skip(f"weights not cached: {e}")
    return m.to(DEV).eval()


@pytest.fixture(scope="module")
def body(stock):
    cfg = load_config(CFG["model_id"], CFG["revision"])
    return FrozenBody(build_body(cfg, pretrained=True, model_id=CFG["model_id"], revision=CFG["revision"])).to(DEV)


def tokens(d):
    return torch.randn(N, d, generator=torch.Generator().manual_seed(0)).to(DEV)


def test_geometry_and_scale(stock, body):
    c = stock.config
    assert (c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (2048, 16, 32, 8, 64)
    assert c.rope_parameters["rope_type"] == "llama3" and stock.rotary_emb.attention_scaling == 1.0
    assert stock.config._attn_implementation == "eager"
    print(f"\nT (mean L2 norm of pretrained embedding rows) = {body.T:.6f}")
    assert 0.3 < body.T < 3.0


def test_mechanics_parity_exact(stock):
    """Zero 4-D mask == no mask + is_causal=False on the untouched bf16 model, bit for bit."""
    x = tokens(2048).to(torch.bfloat16).unsqueeze(0)
    pos = torch.zeros(1, N, dtype=torch.long, device=DEV)
    with torch.no_grad():
        h0 = stock(inputs_embeds=x, attention_mask=torch.zeros(1, 1, N, N, dtype=torch.bfloat16, device=DEV),
                   position_ids=pos, use_cache=False).last_hidden_state
        hN = stock(inputs_embeds=x, attention_mask=None, position_ids=pos, use_cache=False, is_causal=False).last_hidden_state
        hC = stock(inputs_embeds=x, attention_mask=None, position_ids=pos, use_cache=False, is_causal=True).last_hidden_state
    assert torch.equal(h0, hN)
    assert not torch.allclose(h0.float(), hC.float(), atol=1e-2)


def test_precision_recipe_drift(stock, body):
    """fp32 RMSNorm gains under autocast vs the pure-bf16 model: small relative drift, recorded."""
    t = tokens(2048)
    pos = torch.zeros(1, N, dtype=torch.long, device=DEV)
    with torch.no_grad():
        pure = stock(inputs_embeds=t.to(torch.bfloat16).unsqueeze(0), attention_mask=None, position_ids=pos,
                     use_cache=False, is_causal=False).last_hidden_state[0].float()
        recipe = body(t)
    rel = ((recipe - pure).norm() / pure.norm()).item()
    print(f"\nprecision-recipe relative Frobenius drift = {rel:.4e}")
    assert rel < 5e-2


def test_canary_real(body):
    body.canary(atol=1e-2)


def test_encoder_scale_matches_real_T(body):
    try:
        from baselines.common import observed_dense
        from g2l.data import edge_split, load_cora
        data = load_cora()
    except Exception as e:
        pytest.skip(f"Cora unavailable: {e}")
    A = observed_dense(edge_split(data, seed=0)[0], data.num_nodes)
    torch.manual_seed(0)
    tok = E1Linear(data.num_nodes, 2048, gain=body.T / math.sqrt(2048))(A)
    assert abs(tok.norm(dim=-1).mean().item() - body.T) / body.T < 0.25
