"""Decoders: node states Z [N, d] -> edge logits. `forward` scores every cell, `pairs` an index set."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class D1Bilinear(nn.Module):
    """logits = Z W Z^T with W = 0.1 I at init (the Phase-1 scratch decoder)."""

    def __init__(self, d_model: int, chunk: int = 1_048_576):
        super().__init__()
        self.W = nn.Parameter(torch.eye(d_model) * 0.1)
        self.chunk = chunk

    def forward(self, Z):
        return Z @ self.W @ Z.T

    def pairs(self, Z, i, j):
        """Score an index set without materialising [N, N]. Chunked in both directions: the dense
        route costs N^2 (234 MB per step at Photo's N=7650), while scoring every pair at once would
        allocate [P, d] -- P = 0.15*N(N-1)/2 is 4.4M cells there, far worse than the matrix it
        replaces. Neither is affordable; chunks of ~1M pairs are."""
        ZW = Z @ self.W
        return torch.cat([(ZW[i[s:s + self.chunk]] * Z[j[s:s + self.chunk]]).sum(-1)
                          for s in range(0, i.numel(), self.chunk)])


class D3PairMLP(nn.Module):
    """MLP([z_i || z_j || z_i * z_j]) -> logit, scored in chunks of pairs. Training chunks are
    checkpointed so the [P, 3d] activations of ~550k supervised pairs are never all held."""

    def __init__(self, d_model: int, hidden: int = 512, chunk: int = 65536):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.chunk = chunk

    def _score(self, Z, i, j):
        zi, zj = Z[i], Z[j]
        return self.mlp(torch.cat([zi, zj, zi * zj], dim=-1)).squeeze(-1)

    def pairs(self, Z, i, j):
        out = []
        for s in range(0, i.numel(), self.chunk):
            a, b = i[s:s + self.chunk], j[s:s + self.chunk]
            if torch.is_grad_enabled():
                out.append(checkpoint(self._score, Z, a, b, use_reentrant=False))
            else:
                out.append(self._score(Z, a, b))
        return torch.cat(out)

    def forward(self, Z):
        N = Z.shape[0]
        rows = max(1, self.chunk // N)
        out = torch.empty(N, N, device=Z.device, dtype=Z.dtype)
        cols = torch.arange(N, device=Z.device)
        for s in range(0, N, rows):
            r = torch.arange(s, min(s + rows, N), device=Z.device)
            i = r[:, None].expand(-1, N).reshape(-1)
            j = cols[None, :].expand(len(r), -1).reshape(-1)
            out[s:s + len(r)] = self._score(Z, i, j).view(len(r), N)
        return out
