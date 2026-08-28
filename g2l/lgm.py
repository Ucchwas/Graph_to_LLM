"""Phase 8 -- the LGM: an incidence-aware bidirectional Graph Transformer over node AND edge states.

The **external input is the raw graph** and nothing else:

    RawGraph(n, edge_index [2, M], edge_value [M, k])

Everything below is an internal hidden representation built by shared, size-independent
projections -- not a supplied embedding and not a node-ID lookup. No parameter is indexed by node
identity, so one state_dict runs at any N (tests/test_size_independence.py):

    raw graph -> node states [N, d] + edge states [M_u, d]
              -> L pre-LN bidirectional layers, attention structured by incidence
              -> node outputs [N, d] -> D1Bilinear -> [N, N]

There is no GCN and no message-passing layer anywhere. Attention has three blocks: node<-node is
DENSE (the global channel), node<-edge and edge<-node follow the incidence relation. Edge states are
keys AND values in the node softmax, so edge content reaches node outputs as a d-vector rather than
as a scalar attention bias -- the property that makes edges first-class here.

Two symmetry rules the architecture must obey, both load-bearing for permutation equivariance:

  * one state per UNDIRECTED edge, keyed by the canonical (min, max) pair. A node permutation can
    flip which endpoint is `min`, so nothing may distinguish an edge's two endpoints: the edge<-node
    block scores both with the SAME projection and softmaxes over the pair (swap the slots and the
    output is unchanged), and there is deliberately no src/dst role tag anywhere.
  * the edge featuriser is symmetric in its endpoints (sum and |difference| of endpoint degrees).

Node<-node and node<-edge share ONE softmax, computed with a common max so the two blocks combine as
their true joint normaliser, with a learned per-head offset on the edge block (the per-head bias
strength of GaLA, CLAUDE.md section 9).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from g2l.decoders import D1Bilinear

# nn.MultiheadAttention is not used here, but g2l.model sets this globally and this module must be
# safe when imported on its own (an eval-mode fast path would cast additive float masks to bool).
torch.backends.mha.set_fastpath_enabled(False)

NEG_INF = -1e30


@dataclass
class RawGraph:
    """The raw graph. `edge_index` is [2, M] with M directed entries; `edge_value` is [M, k] or
    None (k defaults to 1, all ones). `batch` [n] holds segment ids for block-diagonal batching.

    The dense adjacency is redundant given `edge_index` -- structure reaches the model through the
    edge states and the incidence relation, never through a dense [N, N] object -- so it is kept
    only as a convenience constructor."""
    n: int
    edge_index: torch.Tensor
    edge_value: torch.Tensor | None = None
    batch: torch.Tensor | None = None
    x: torch.Tensor | None = None      # [n, F] categorical node features (e.g. OGB atom features)

    @staticmethod
    def from_dense(A: torch.Tensor, edge_value: torch.Tensor | None = None) -> "RawGraph":
        return RawGraph(A.shape[0], A.nonzero().T, edge_value)

    def to(self, device) -> "RawGraph":
        return RawGraph(self.n, self.edge_index.to(device),
                        None if self.edge_value is None else self.edge_value.to(device),
                        None if self.batch is None else self.batch.to(device),
                        None if self.x is None else self.x.to(device))


def canonical(g: RawGraph, dtype=None):
    """Collapse the M directed entries to M_u undirected edge slots keyed by (min, max), averaging
    the edge values of the two directions. Returns (src, dst, value) with src < dst."""
    src, dst = g.edge_index[0], g.edge_index[1]
    lo, hi = torch.minimum(src, dst), torch.maximum(src, dst)
    uniq, inv = torch.unique(lo * g.n + hi, return_inverse=True)
    m_u = uniq.numel()
    val = g.edge_value if g.edge_value is not None else torch.ones(
        src.numel(), 1, device=src.device, dtype=dtype or torch.get_default_dtype())
    val = val.to(dtype or val.dtype)
    out = torch.zeros(m_u, val.shape[1], device=val.device, dtype=val.dtype)
    cnt = torch.zeros(m_u, 1, device=val.device, dtype=val.dtype)
    out.index_add_(0, inv, val)
    cnt.index_add_(0, inv, torch.ones_like(val[:, :1]))
    return uniq // g.n, uniq % g.n, out / cnt.clamp(min=1)


def concat_graphs(graphs: list[RawGraph], dtype=None) -> RawGraph:
    """Block-diagonal disjoint union: node indices are offset and segment ids recorded. This is
    masking, NOT padding -- no dummy node exists and no parameter sees the batch shape. Edges never
    cross graphs, so incidence stays within a graph and the dense block is masked by `batch`
    (tests/test_lgm_batching.py pins that a batch reproduces the single-graph runs exactly)."""
    dtype = dtype or torch.get_default_dtype()
    off, ei, ev, batch, xs = 0, [], [], [], []
    for b, g in enumerate(graphs):
        ei.append(g.edge_index + off)
        ev.append(g.edge_value.to(dtype) if g.edge_value is not None
                  else torch.ones(g.edge_index.shape[1], 1, dtype=dtype, device=g.edge_index.device))
        batch.append(torch.full((g.n,), b, dtype=torch.long, device=g.edge_index.device))
        if g.x is not None:
            xs.append(g.x)
        off += g.n
    return RawGraph(off, torch.cat(ei, 1), torch.cat(ev, 0), torch.cat(batch),
                    torch.cat(xs) if xs else None)


def segment_mean(x: torch.Tensor, batch: torch.Tensor | None, n_seg: int) -> torch.Tensor:
    """Per-graph mean, broadcast back over nodes. One graph is the common case."""
    if batch is None:
        return x.mean(0, keepdim=True).expand_as(x)
    tot = torch.zeros(n_seg, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, batch, x)
    cnt = torch.zeros(n_seg, 1, device=x.device, dtype=x.dtype).index_add_(
        0, batch, torch.ones_like(x[:, :1]))
    return (tot / cnt.clamp(min=1))[batch]


class NodeEdgeProjection(nn.Module):
    """Raw graph -> (node states [N, d], edge states [M_u, d]). Named `first` on the body so
    g2l/train.py:96's encoder-parameter probe finds it when the encoder is nn.Identity.

    Node features are structural and scale-free by design: a Fourier encoding of log1p(degree)
    plus degree relative to the graph's mean degree, so one checkpoint spans mean degree 2 (molecules)
    to 31 (Amazon Photo). Edge features are symmetric in the endpoints."""

    def __init__(self, d: int, k: int = 1, freqs: int = 8, degree_init: bool = True, d_rni: int = 0,
                 atom_dims: list[int] | None = None):
        super().__init__()
        self.k, self.degree_init, self.d_rni = k, degree_init, d_rni
        # One embedding per categorical node-feature column, summed -- OGB's AtomEncoder. Indexed by
        # FEATURE VALUE, never by node index: two nodes with the same atom type get the same vector,
        # so this cannot carry node identity and the model stays size-independent. That is the
        # distinction the "no node-ID embedding table" constraint draws.
        self.atom = nn.ModuleList([nn.Embedding(c, d) for c in atom_dims]) if atom_dims else None
        self.register_buffer("w", torch.logspace(-1, 1, freqs), persistent=False)
        self.f_node, self.f_edge = 2 * freqs + 3, k + 4
        self.u_node, self.u_edge = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.w_deg = nn.Linear(self.f_node, d)
        self.w_ev = nn.Linear(self.f_edge, d)
        self.w_rni = nn.Linear(d_rni, d, bias=False) if d_rni else None
        self.ln_node, self.ln_edge = nn.LayerNorm(d), nn.LayerNorm(d)

    def node_features(self, deg: torch.Tensor, batch, n_seg: int) -> torch.Tensor:
        ld = torch.log1p(deg).unsqueeze(-1)                       # [N, 1]
        mean_deg = segment_mean(deg.unsqueeze(-1), batch, n_seg)  # [N, 1]
        return torch.cat([torch.sin(ld * self.w), torch.cos(ld * self.w),
                          deg.unsqueeze(-1) / mean_deg.clamp(min=1e-6),
                          ld - torch.log1p(mean_deg),
                          (deg == 0).to(ld.dtype).unsqueeze(-1)], dim=-1)

    def edge_features(self, val, deg, src, dst, batch, n_seg) -> torch.Tensor:
        ls, lt = torch.log1p(deg[src]).unsqueeze(-1), torch.log1p(deg[dst]).unsqueeze(-1)
        md = segment_mean(deg.unsqueeze(-1), batch, n_seg)[src]
        return torch.cat([val, ls + lt, (ls - lt).abs(),
                          (src == dst).to(val.dtype).unsqueeze(-1), torch.log1p(md)], dim=-1)

    def forward(self, g: RawGraph, z: torch.Tensor | None = None):
        src, dst, val = canonical(g, dtype=self.u_node.dtype)
        n_seg = int(g.batch.max()) + 1 if g.batch is not None else 1
        deg = torch.zeros(g.n, device=src.device, dtype=val.dtype).index_add_(
            0, torch.cat([src, dst]), torch.ones(2 * src.numel(), device=src.device, dtype=val.dtype))
        fn = self.node_features(deg, g.batch, n_seg)
        if not self.degree_init:
            fn = torch.zeros_like(fn)
        h = self.u_node + self.w_deg(fn)
        if self.atom is not None:
            assert g.x is not None, "atom_dims set but the graph carries no node features"
            h = h + sum(emb(g.x[:, i]) for i, emb in enumerate(self.atom))
        if self.w_rni is not None:
            assert z is not None, "d_rni > 0: pass the random node states explicitly (see the tests)"
            h = h + self.w_rni(z)
        e = self.u_edge + self.w_ev(self.edge_features(val, deg, src, dst, g.batch, n_seg))
        return self.ln_node(h), self.ln_edge(e), src, dst


class IncidenceAttention(nn.Module):
    """One shared W_q/W_k/W_v/W_o over both state types.

    node<-node is dense; node<-edge is the incidence relation. They share one softmax: both blocks
    are exponentiated against a COMMON max, so the returned mixture is the true joint normaliser
    rather than two independent pools glued together. `beta` [heads] is a learned per-head offset
    on the edge block.

    edge<-node attends to exactly the two endpoints with the SAME projection for both slots, which
    is what keeps the layer symmetric under swapping an edge's endpoints."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.o = nn.Linear(d, d, bias=False)
        self.beta = nn.Parameter(torch.zeros(heads))

    def split(self, x):
        return x.view(x.shape[0], self.h, self.dh)

    def forward(self, hn, he, src, dst, batch):
        n, m = hn.shape[0], he.shape[0]
        scale = 1.0 / math.sqrt(self.dh)
        qn, kn, vn = self.split(self.q(hn)), self.split(self.k(hn)), self.split(self.v(hn))
        qe, ke, ve = self.split(self.q(he)), self.split(self.k(he)), self.split(self.v(he))

        # ---- node <- node, dense --------------------------------------------------------------
        lnn = torch.einsum("ihd,jhd->ijh", qn, kn) * scale                       # [N, N, H]
        if batch is not None:
            lnn = lnn.masked_fill((batch[:, None] != batch[None, :]).unsqueeze(-1), NEG_INF)
        mx = lnn.amax(dim=1)                                                     # [N, H]

        # ---- node <- edge, incidence ----------------------------------------------------------
        node_of = torch.cat([src, dst])                                          # [2 M_u]
        edge_of = torch.cat([torch.arange(m, device=he.device)] * 2)
        lne = (qn[node_of] * ke[edge_of]).sum(-1) * scale + self.beta            # [2 M_u, H]
        if m:
            mx = torch.maximum(mx, torch.full_like(mx, NEG_INF).index_reduce_(
                0, node_of, lne, "amax", include_self=False))

        enn = (lnn - mx.unsqueeze(1)).exp()                                      # [N, N, H]
        den = enn.sum(1)                                                         # [N, H]
        num = torch.einsum("ijh,jhd->ihd", enn, vn)
        if m:
            ene = (lne - mx[node_of]).exp()                                      # [2 M_u, H]
            den = den + torch.zeros_like(den).index_add_(0, node_of, ene)
            num = num + torch.zeros_like(num).index_add_(
                0, node_of, ene.unsqueeze(-1) * ve[edge_of])
        out_n = num / den.clamp(min=1e-30).unsqueeze(-1)

        # ---- edge <- node, the two endpoints, one shared projection ---------------------------
        le = torch.stack([(qe * kn[src]).sum(-1), (qe * kn[dst]).sum(-1)], dim=-1) * scale
        pe = le.softmax(-1)                                                      # [M_u, H, 2]
        out_e = pe[..., 0].unsqueeze(-1) * vn[src] + pe[..., 1].unsqueeze(-1) * vn[dst]

        d = self.h * self.dh
        return self.o(out_n.reshape(n, d)), self.o(out_e.reshape(m, d))


class ISETLayer(nn.Module):
    """Pre-LN block. The LayerNorms and the FFN are shared across both state types -- one
    transformer over one token set, with the type separation carried by u_node / u_edge."""

    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = IncidenceAttention(d, heads)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dropout = dropout

    def forward(self, hn, he, src, dst, batch):
        an, ae = self.attn(self.ln1(hn), self.ln1(he), src, dst, batch)
        hn, he = hn + F.dropout(an, self.dropout, self.training), he + F.dropout(ae, self.dropout, self.training)
        hn = hn + F.dropout(self.ff(self.ln2(hn)), self.dropout, self.training)
        he = he + F.dropout(self.ff(self.ln2(he)), self.dropout, self.training)
        return hn, he


class ISETBody(nn.Module):
    """`self.first` and `self.enc.layers` are the names g2l/train.py probes (lines 60 and 96)."""
    needs_graph = True

    def __init__(self, d: int, layers: int = 4, heads: int = 8, k: int = 1, dropout: float = 0.0,
                 degree_init: bool = True, d_rni: int = 0, edges: bool = True,
                 atom_dims: list[int] | None = None):
        super().__init__()
        self.first = NodeEdgeProjection(d, k=k, degree_init=degree_init, d_rni=d_rni,
                                        atom_dims=atom_dims)
        self.enc = nn.Module()
        self.enc.layers = nn.ModuleList([ISETLayer(d, heads, dropout) for _ in range(layers)])
        self.edges = edges  # False = the no-edge control: identical model with edge states removed

    def forward(self, g: RawGraph, z: torch.Tensor | None = None) -> torch.Tensor:
        hn, he, src, dst = self.first(g, z)
        if not self.edges:
            he, src, dst = he[:0], src[:0], dst[:0]
        for layer in self.enc.layers:
            hn, he = layer(hn, he, src, dst, g.batch)
        return hn


class LGM(nn.Module):
    """Raw graph -> [N, N] logits. `node_states` is the batched entry point: decode per graph."""

    def __init__(self, d: int = 256, layers: int = 4, heads: int = 8, k: int = 1,
                 dropout: float = 0.0, degree_init: bool = True, d_rni: int = 0,
                 edges: bool = True, seed: int | None = None,
                 atom_dims: list[int] | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.body = ISETBody(d, layers, heads, k, dropout, degree_init, d_rni, edges, atom_dims)
        self.dec_norm = nn.LayerNorm(d)
        nn.init.constant_(self.dec_norm.weight, 1 / math.sqrt(d))
        nn.init.zeros_(self.dec_norm.bias)
        self.decoder = D1Bilinear(d)

    def node_states(self, g: RawGraph, z=None) -> torch.Tensor:
        return self.dec_norm(self.body(g, z))

    def forward(self, g: RawGraph, z=None) -> torch.Tensor:
        return self.decoder(self.node_states(g, z))

    def pairs(self, g: RawGraph, i, j, z=None) -> torch.Tensor:
        return self.decoder.pairs(self.node_states(g, z), i, j)
