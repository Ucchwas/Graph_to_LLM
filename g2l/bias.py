"""Structural features for the attention bias.

THE LEAK RULE: every function here takes the OBSERVED adjacency only. Nothing in
this module may ever see held-out cells; tests/test_leaks.py enforces it.
"""
import numpy as np
import torch


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
