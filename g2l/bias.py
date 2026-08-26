"""Structural features for the attention bias.

THE LEAK RULE: every function here takes the OBSERVED adjacency only. Nothing in
this module may ever see held-out cells; tests/test_leaks.py enforces it.
"""
import numpy as np
import torch
from torch import nn


def spd_matrix_fast(A_obs: torch.Tensor, max_dist: int = 8) -> torch.Tensor:
    """`spd_matrix` by boolean matmul on A's device (frontier expansion, one matmul per hop):
    exact, int16 [N, N], ~30 ms on Cora on a GPU. Same bucket semantics as `spd_matrix`."""
    N = A_obs.shape[0]
    Af = (A_obs > 0).float()
    reach = torch.eye(N, dtype=torch.bool, device=A_obs.device)
    dist = torch.full((N, N), max_dist + 1, dtype=torch.int16, device=A_obs.device)
    dist.fill_diagonal_(0)
    frontier = reach.float()
    for k in range(1, max_dist + 1):
        new = (frontier @ Af > 0) & ~reach
        if not new.any():
            break
        dist[new] = k
        reach |= new
        frontier = new.float()
    return dist


class SPDBias(nn.Module):
    """Per-head learned lookup on shortest-path distance, added to every layer's pre-softmax
    scores (Graphormer's spatial encoding; GTLM's SPD bias). Zero-init, so the model starts
    exactly at the Phase-2 model. Bucket 0 (self) is fixed at 0, buckets 1..max_dist are exact
    distances, bucket max_dist + 1 is "farther or unreachable" -- learned, never -inf."""

    def __init__(self, heads: int, max_dist: int = 8):
        super().__init__()
        self.max_dist = max_dist
        self.bias_table = nn.Parameter(torch.zeros(heads, max_dist + 2))

    def forward(self, spd: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        """spd int [N, N] -> bias [1, heads, N, N] in `dtype`. Written as one-hot(spd) x table so
        the backward is a reduction GEMM instead of a scatter-add of N^2 x heads elements into
        10 slots (0.44 s/epoch on Cora as a gather, ~20 ms this way). Bucket 0 (self) is
        excluded, which also fixes the diagonal at 0."""
        buckets = torch.arange(1, self.max_dist + 2, device=spd.device).view(-1, 1, 1)
        onehot = (spd.unsqueeze(0) == buckets).to(dtype)  # [K-1, N, N]
        b = torch.einsum("hk,knm->hnm", self.bias_table[:, 1:].to(dtype), onehot)
        return b.unsqueeze(0)


def spd_matrix(A_obs: torch.Tensor, max_dist: int = 8) -> torch.Tensor:
    """All-pairs shortest-path distance on the observed graph, int16 [N, N].
    Distances > max_dist and unreachable pairs both map to max_dist + 1
    (a soft learned bucket downstream -- never a hard -inf mask)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    A = csr_matrix(A_obs.detach().cpu().numpy())
    d = shortest_path(A, method="auto", unweighted=True, directed=False)
    d[~np.isfinite(d)] = max_dist + 1
    return torch.from_numpy(np.minimum(d, max_dist + 1).astype(np.int16))
