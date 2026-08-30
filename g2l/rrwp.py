"""Phase 10 -- relative random-walk probabilities (RRWP) and their diagonal (RWSE).

Computed from the edge set a model is handed, inside its forward pass, and from nothing else:

    A       adjacency of `edge_index`                       [N, N]   (block-diagonal across a batch)
    M       D^-1 A, the random-walk transition matrix       row-stochastic; an isolated node's row is 0
    RRWP    [I, M, M^2, ..., M^(K-1)]                        [N, N, K]  (GRIT, Ma et al. ICML 2023)
    RWSE    diag(M^k), k = 1..K                              [N, K]     (Dwivedi et al. 2022)

RRWP is the pairwise object: it enters the LGM only as a learned per-head bias on the DENSE
node<-node attention logits. RWSE is its node-level part and is what the message-passing baselines
receive, added to their node input -- the accepted transformer-vs-MPNN protocol (GRIT; Groetschla
et al. KDD'26). Neither is a node-ID table, an external embedding, or a pretrained encoder: both are
deterministic functions of the input topology, so the models stay size-independent and permutation-
equivariant (RRWP(pi G) = pi RRWP(G) pi^T), and whatever edges are hidden from the model are hidden
from these too -- the leakage rule holds by construction.

RRWP is asymmetric for irregular graphs (M^k[i,j] d_i = M^k[j,i] d_j by detailed balance); it is
used as GRIT uses it. Output symmetry comes from the decoder, not from here.

WHY THE CHANNELS ARE NORMALISED (`normalise=True`, the default and what the runs use). A k-step
probability spreads over ~deg^k nodes, so a raw entry of M^k is ~deg^-k and the mean over the matrix
is ~1/N. On the TCGA graphs (N = 2000, mean degree 20) that makes the raw bias 6.7e-4 against
attention logits of magnitude 1.7 -- a ratio of 4e-4 -- and the arm is numerically inert no matter
how long it trains, because a zero-initialised weight would need to reach ~1e3 to compete. Published
RRWP results are on molecules (N ~ 25, degree ~ 2), where the raw values are already O(1) and the
question never arises. So every k >= 1 channel is divided by its own root-mean-square over the
graph, giving unit scale at ANY N and ANY degree; channel 0 is the identity and is already O(1), so
it is left alone. The statistic is:

  * permutation-invariant  -- a scalar per graph, so equivariance is untouched;
  * per graph, never per batch -- graph A's scale cannot depend on graph B (no cross-graph leakage);
  * parameter-free and computed at run time -- size independence is untouched.

RWSE is normalised the same way for the same reason, so neither family is advantaged by scaling.
"""
import torch

EPS = 1e-12


def transition(edge_index: torch.Tensor, n: int, dtype=None, device=None) -> torch.Tensor:
    """M = D^-1 A on the given edge set. Multi-entries collapse to 1 (assignment, not a count)."""
    device = device or edge_index.device
    dtype = dtype or torch.get_default_dtype()
    A = torch.zeros(n, n, dtype=dtype, device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    deg = A.sum(1, keepdim=True)
    return A / deg.clamp(min=1.0)


def _seg_sizes(batch: torch.Tensor | None, n: int, n_seg: int, dtype, device):
    if batch is None:
        return torch.full((1,), float(n), dtype=dtype, device=device)
    return torch.zeros(n_seg, dtype=dtype, device=device).index_add_(
        0, batch, torch.ones(n, dtype=dtype, device=device))


def _rms_rows(P: torch.Tensor, batch: torch.Tensor | None, n_seg: int) -> torch.Tensor:
    """Per-graph RMS of a block-diagonal [N, N] matrix, broadcast back over its rows -> [N, 1].

    P is block-diagonal, so summing a row over ALL columns is the same as summing it over the row's
    own graph -- which is what keeps this per-graph rather than per-batch."""
    n = P.shape[0]
    sq = P.pow(2).sum(1)                                            # [N]
    cnt = _seg_sizes(batch, n, n_seg, P.dtype, P.device)            # [n_seg]
    if batch is None:
        tot = sq.sum().reshape(1)
    else:
        tot = torch.zeros(n_seg, dtype=P.dtype, device=P.device).index_add_(0, batch, sq)
    rms = (tot / (cnt * cnt).clamp(min=1.0)).clamp(min=EPS).sqrt()  # [n_seg]
    return (rms if batch is None else rms[batch]).reshape(-1, 1)


def rrwp(edge_index: torch.Tensor, n: int, K: int = 16, dtype=None, device=None,
         batch: torch.Tensor | None = None, normalise: bool = True) -> torch.Tensor:
    """[N, N, K]: channel k is M^k, k = 0 .. K-1 (channel 0 is the identity).

    With `normalise` (the default) every channel k >= 1 is divided by its own per-graph RMS, so the
    bias it feeds has unit scale at any N and any degree -- see the module docstring. Powers are
    normalised one at a time so no second [N, N, K] tensor is ever materialised."""
    M = transition(edge_index, n, dtype, device)
    n_seg = int(batch.max()) + 1 if batch is not None else 1
    P = torch.eye(n, dtype=M.dtype, device=M.device)
    out = [P]                                     # channel 0: the identity, already O(1)
    for _ in range(K - 1):
        P = P @ M
        out.append(P / _rms_rows(P, batch, n_seg) if normalise else P)
    return torch.stack(out, dim=-1)


def rwse(edge_index: torch.Tensor, n: int, K: int = 16, dtype=None, device=None,
         batch: torch.Tensor | None = None, normalise: bool = True) -> torch.Tensor:
    """[N, K]: the k-step return probabilities diag(M^k), k = 1 .. K.

    Normalised per graph and per channel, exactly as `rrwp` is, so neither model family gains an
    advantage from the scaling."""
    M = transition(edge_index, n, dtype, device)
    n_seg = int(batch.max()) + 1 if batch is not None else 1
    cnt = _seg_sizes(batch, n, n_seg, M.dtype, M.device)
    P, out = M, []
    for _ in range(K):
        d = P.diagonal()
        if normalise:
            if batch is None:
                tot = d.pow(2).sum().reshape(1)
            else:
                tot = torch.zeros(n_seg, dtype=M.dtype, device=M.device).index_add_(
                    0, batch, d.pow(2))
            rms = (tot / cnt.clamp(min=1.0)).clamp(min=EPS).sqrt()
            d = d / (rms if batch is None else rms[batch])
        out.append(d)
        P = P @ M
    return torch.stack(out, dim=-1)
