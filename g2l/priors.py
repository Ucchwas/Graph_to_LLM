"""Non-parametric references for held-out-layer completion (Phase 6). Each scores a full [N, N]
matrix from exactly what the shared model could have seen: the layer's own partial input and
the train layers' visible pairs. A train layer is scored leave-one-out (its own pairs removed).
  identity          the input itself: every scored cell is 0, the leak canary (AUROC 0.5)
  common_neighbors  A_in @ A_in on the partial input
  pair_frequency    number of train layers containing the pair -- the memorisation reference;
                    globally hidden pairs score 0 by construction
  knn               Jaccard(input, train layer) weighted vote of the train layers' pairs
"""
import numpy as np
import torch

from g2l.layers import dense, pair_matrix


class Priors:
    def __init__(self, layers, N: int):
        self.N = N
        self.train = [l for l in layers if l.group == "train"]
        self.names, M, self.union = pair_matrix({l.name: l.train for l in self.train}, N)
        self.M = M.astype(np.float32)
        self.size = self.M.sum(1)
        self.row = {n: r for r, n in enumerate(self.names)}
        self.ui, self.uj = self.union // N, self.union % N

    def _scatter(self, s: np.ndarray) -> torch.Tensor:
        S = torch.zeros(self.N, self.N)
        v = torch.from_numpy(s.astype(np.float32))
        S[self.ui, self.uj] = v
        S[self.uj, self.ui] = v
        return S

    def _query(self, inp: torch.Tensor) -> np.ndarray:
        key = inp[0] * self.N + inp[1]
        pos = torch.searchsorted(self.union, key).clamp(max=self.union.numel() - 1)
        hit = pos[self.union[pos] == key]
        q = np.zeros(self.union.numel(), dtype=np.float32)
        q[hit.numpy()] = 1.0
        return q

    def identity(self, L, inp):
        return dense(inp, self.N)

    def common_neighbors(self, L, inp):
        A = dense(inp, self.N)
        return A @ A

    def pair_frequency(self, L, inp):
        c = self.M.sum(0)
        if L.name in self.row:
            c = c - self.M[self.row[L.name]]
        return self._scatter(c)

    def knn(self, L, inp):
        q = self._query(inp)
        inter = self.M @ q
        w = inter / (self.size + q.sum() - inter)
        if L.name in self.row:
            w[self.row[L.name]] = 0.0
        return self._scatter(w @ self.M)

    def score(self, name: str, L, inp):
        return getattr(self, name)(L, inp)
