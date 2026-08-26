"""Graph encoders: raw adjacency rows -> one token per node in the LLM's embedding space."""
import torch
from torch import nn


class GraphEncoder(nn.Module):
    """A: [N, N] observed adjacency (0/1, hidden cells zeroed); X: [N, F] or None -> [N, d_model]."""

    def forward(self, A: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class E1Linear(GraphEncoder):
    """One linear map per adjacency row, then LayerNorm. The LayerNorm gain sets the token
    norm: gain = T / sqrt(d_model) puts every token at the LLM's embedding scale T."""

    def __init__(self, n_nodes: int, d_model: int, gain: float):
        super().__init__()
        self.proj = nn.Linear(n_nodes, d_model)
        nn.init.orthogonal_(self.proj.weight, gain=1.41)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(d_model)
        nn.init.constant_(self.norm.weight, gain)
        nn.init.zeros_(self.norm.bias)

    def forward(self, A, X=None):
        return self.norm(self.proj(A))
