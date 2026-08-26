"""Phase 3: the graph-aware attention bias. Tiny Llama config (fp32) + the scratch body."""
import torch
from transformers import LlamaConfig

from g2l.bias import SPDBias, spd_matrix, spd_matrix_fast
from g2l.data import mask_matrix, subsample_edges
from g2l.llm import FrozenBody, build_body
from g2l.model import ScratchBody, build_model, param_groups
from g2l.train import make_optimizer

D = 64


def tiny_body():
    cfg = LlamaConfig(hidden_size=D, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=16, max_position_embeddings=64,
                      initializer_range=0.2, attn_implementation="eager")
    return FrozenBody(build_body(cfg, pretrained=False, dtype=torch.float32, seed=0))


def graph(n=24, p=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(n, n, generator=g) < p).float(), diagonal=1)
    return up + up.T


def test_spd_fast_matches_scipy():
    for n, p, seed in ((30, 0.1, 0), (80, 0.03, 1), (120, 0.01, 2)):
        A = graph(n, p, seed)
        assert torch.equal(spd_matrix_fast(A), spd_matrix(A))
    A = torch.zeros(6, 6)
    A[0, 1] = A[1, 0] = A[1, 2] = A[2, 1] = 1.0
    d = spd_matrix_fast(A)
    assert d[0, 0] == 0 and d[0, 2] == 2 and d[0, 5] == 9


def test_zero_init_bias_is_inert():
    A = graph()
    for arm, body in (("random", tiny_body()), ("scratch", None)):
        plain = build_model(arm, A.shape[0], D, 1.0, body=body, seed=0)
        biased = build_model(arm, A.shape[0], D, 1.0, body=body, seed=0, bias=True)
        assert biased.bias is not None and biased.bias.bias_table.abs().sum() == 0
        with torch.no_grad():
            assert torch.equal(plain(A), biased(A))
        assert torch.allclose(plain(A), biased(A), atol=1e-6)  # autograd may pick another attention kernel


def test_bias_is_per_head_off_diagonal_and_reaches_scores():
    A = graph()
    m = build_model("random", A.shape[0], D, 1.0, body=tiny_body(), seed=0, bias=True)
    with torch.no_grad():
        m.bias.bias_table[1, 1] = 3.0
    b = m.attention_bias(A)
    spd = spd_matrix(A)
    assert b.shape == (1, 4, A.shape[0], A.shape[0])
    assert b[0, 1][spd == 1].eq(3.0).all() and b[0, 1][spd != 1].eq(0).all()
    assert b[0, [0, 2, 3]].abs().sum() == 0 and b[0, 1].diagonal().abs().sum() == 0
    plain = build_model("random", A.shape[0], D, 1.0, body=tiny_body(), seed=0)
    assert not torch.allclose(plain(A), m(A))


def test_bias_gradient_and_param_group():
    A = graph()
    for arm, body in (("random", tiny_body()), ("scratch", None)):
        m = build_model(arm, A.shape[0], D, 1.0, body=body, seed=0, bias=True)
        assert param_groups(m)["bias_table"] == [m.bias.bias_table]
        opt = make_optimizer(m, 1e-3, 1e-4, 0.0, lr_bias=1e-2)
        assert any(g["lr"] == 1e-2 for g in opt.param_groups)
        m(A).sum().backward()
        assert m.bias.bias_table.grad is not None and m.bias.bias_table.grad.abs().sum() > 0
        assert m.bias.bias_table.grad[:, 0].abs().sum() == 0  # self bucket is fixed


def test_biased_model_is_permutation_equivariant():
    A = graph()
    m = build_model("random", A.shape[0], D, 1.0, body=tiny_body(), seed=0, bias=True)
    with torch.no_grad():
        m.bias.bias_table.normal_(0, 0.5)
        m.bias.bias_table[:, 0] = 0
    perm = torch.randperm(A.shape[0], generator=torch.Generator().manual_seed(1))
    H = m.embed(A)
    Hp = m.embed(A[perm][:, perm])
    # E1 is a fixed-N linear map, so permuting rows also permutes the encoder's input columns;
    # equivariance holds for the bias + body given permuted tokens, which is what we test:
    tokens = m.encoder(A)
    Hb = m.body(tokens, bias=m.attention_bias(A))
    Hbp = m.body(tokens[perm], bias=m.attention_bias(A[perm][:, perm]))
    assert torch.allclose(Hb[perm], Hbp, atol=1e-4)
    assert H.shape == Hp.shape


def test_bias_depends_only_on_observed_graph():
    A1 = graph(40, 0.1, 3)
    A_obs, sup = mask_matrix(A1, frac=0.2, seed=0)
    hidden = sup | sup.T
    A2 = A1.clone()
    A2[hidden] = 1.0 - A2[hidden]
    m = build_model("scratch", 40, D, 1.0, seed=0, bias=True)
    with torch.no_grad():
        m.bias.bias_table.normal_()
    assert torch.equal(m.attention_bias(A_obs), m.attention_bias(A2 * (~hidden).float()))


def test_subsample_edges():
    A = graph(100, 0.1, 4)
    B = subsample_edges(A, 0.3, seed=0)
    assert torch.equal(B, B.T) and ((B == 1) <= (A == 1)).all()
    frac = torch.triu(B, 1).sum() / torch.triu(A, 1).sum()
    assert 0.2 < frac < 0.4
    assert torch.equal(subsample_edges(A, 0.3, seed=0), B) and not torch.equal(subsample_edges(A, 0.3, seed=1), B)
    assert torch.equal(subsample_edges(A, 1.0, seed=0), A)


def test_bias_survives_eval_mode():
    """torch's eval-mode TransformerEncoderLayer fast path casts a float mask to bool -- the bias
    would become a hard mask at every val/test pass; g2l.model disables it. With a non-zero
    table, eval/no_grad output must equal train-mode output, and the scratch layer 0 must equal
    softmax(QK^T/sqrt(d) + bias) V computed by hand."""
    assert not torch.backends.mha.get_fastpath_enabled()
    N = 40
    A = graph(N, 0.1, 3)
    for arm, body in (("random", tiny_body()), ("scratch", None)):
        m = build_model(arm, N, D, 1.0, body=body, seed=0, bias=True)
        with torch.no_grad():
            m.bias.bias_table.normal_(0, 1.0)
            m.bias.bias_table[:, 0] = 0
        m.train()
        out_train = m(A).detach()
        m.eval()
        with torch.no_grad():
            out_eval = m(A)
        assert torch.allclose(out_train, out_eval, atol=1e-5), arm
    layer, H = m.body.enc.layers[0], m.body.heads  # the scratch model

    def heads(t):
        return t.view(1, N, H, D // H).transpose(1, 2)

    with torch.no_grad():
        x = m.encoder(A).unsqueeze(0)
        b = m.attention_bias(A)[0]  # [H, N, N]
        h = layer.norm1(x)
        q, k, v = (h @ layer.self_attn.in_proj_weight.T + layer.self_attn.in_proj_bias).chunk(3, -1)
        att = torch.softmax(heads(q) @ heads(k).transpose(-1, -2) / (D // H) ** 0.5 + b, -1) @ heads(v)
        att = att.transpose(1, 2).reshape(1, N, D) @ layer.self_attn.out_proj.weight.T + layer.self_attn.out_proj.bias
        ref = x + att
        ref = ref + layer.linear2(torch.nn.functional.gelu(layer.linear1(layer.norm2(ref))))
        assert torch.allclose(layer(x, src_mask=b), ref, atol=1e-5)
        assert not torch.allclose(layer(x, src_mask=b), layer(x), atol=1e-3)  # the bias is not a no-op
