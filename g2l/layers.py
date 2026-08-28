"""Aligned multi-layer graphs (Phase 6): many edge sets over one node index. `split_layers`
applies three splits at once -- layers into similarity groups (train / val / test layers), a
global set of hidden node pairs removed from every layer, and each layer's own pairs 85/5/10
with 1:1 negatives -- and every Layer exposes the (train, val, test) objects the single-graph
harness already consumes, so the per-layer and the shared arms score identically."""
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data


def canon(e: torch.Tensor) -> torch.Tensor:
    """[2, k] -> unique pairs with i < j, self-pairs dropped."""
    e = torch.stack([torch.minimum(e[0], e[1]), torch.maximum(e[0], e[1])])
    return torch.unique(e[:, e[0] != e[1]], dim=1)


def both(e: torch.Tensor) -> torch.Tensor:
    return torch.cat([e, e.flip(0)], 1)


def dense(e: torch.Tensor, N: int) -> torch.Tensor:
    A = torch.zeros(N, N)
    A[e[0], e[1]] = 1.0
    A[e[1], e[0]] = 1.0
    return A


def sample_pairs(avoid: torch.Tensor, n: int, g: torch.Generator, pool: torch.Tensor | None = None) -> torch.Tensor:
    """n distinct seeded pairs i < j from `pool` (every pair if None) that are not in `avoid`."""
    N = avoid.shape[0]
    ok = torch.triu(torch.ones(N, N, dtype=torch.bool), 1) & ~avoid
    if pool is not None:
        ok &= pool
    cand = ok.nonzero()
    return cand[torch.randperm(cand.size(0), generator=g)[:n]].T.contiguous()


@dataclass
class Layer:
    name: str
    group: str                 # train / val / test
    train: torch.Tensor        # [2, E] i < j: the layer's visible pairs (message passing + supervision)
    val_pos: torch.Tensor
    val_neg: torch.Tensor
    test_pos: torch.Tensor
    test_neg: torch.Tensor
    hidden_pos: torch.Tensor   # the layer's pairs that belong to the global hidden set
    hidden_neg: torch.Tensor   # as many hidden pairs absent from the layer
    n_pairs: int               # pairs in the layer before anything was removed

    def exclude(self, N: int, hidden: torch.Tensor) -> torch.Tensor:
        """Cells never supervised: the global hidden pairs and this layer's val / test pairs."""
        ex = hidden.clone()
        for e in (self.val_pos, self.test_pos):
            ex[e[0], e[1]] = ex[e[1], e[0]] = True
        return ex

    def split(self, N: int, hidden: torch.Tensor):
        """(train, val, test) as RandomLinkSplit would build them: val input = train pairs, test
        input = train + val pairs; the test object carries the hidden-pair control and the
        exclusion for the sparse column."""
        tr = Data(edge_index=both(self.train), num_nodes=N)
        tr.pos_edge_label_index = self.train
        va = Data(edge_index=both(self.train), num_nodes=N)
        va.pos_edge_label_index, va.neg_edge_label_index = self.val_pos, self.val_neg
        te = Data(edge_index=both(torch.cat([self.train, self.val_pos], 1)), num_nodes=N)
        te.pos_edge_label_index, te.neg_edge_label_index = self.test_pos, self.test_neg
        te.hidden_pos_edge_label_index, te.hidden_neg_edge_label_index = self.hidden_pos, self.hidden_neg
        te.exclude_mask = hidden
        return tr, va, te


def pair_matrix(pairs: dict[str, torch.Tensor], N: int):
    """Boolean [L, P] membership of every layer over the union of pairs (row order = sorted names)."""
    names = sorted(pairs)
    keys = [pairs[n][0] * N + pairs[n][1] for n in names]
    union, inv = torch.unique(torch.cat(keys), return_inverse=True)
    M = np.zeros((len(names), union.numel()), dtype=np.bool_)
    off = 0
    for r, k in enumerate(keys):
        M[r, inv[off:off + k.numel()].numpy()] = True
        off += k.numel()
    return names, M, union


def layer_groups(names: list[str], M: np.ndarray, tau: float, seed: int, fractions=(0.8, 0.1, 0.1)) -> dict[str, str]:
    """Connected components of the layer graph linked at Jaccard >= tau, assigned whole to
    train / val / test by cumulative layer count in a seeded component order."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    Mf = M.astype(np.float32)
    inter = Mf @ Mf.T
    sz = M.sum(1).astype(np.float32)
    J = inter / (sz[:, None] + sz[None, :] - inter)
    np.fill_diagonal(J, 0.0)
    n, lab = connected_components(sp.csr_matrix(J >= tau), directed=False)
    order = np.random.default_rng(seed).permutation(n)
    cum = np.cumsum(np.bincount(lab)[order])
    comp_split = np.zeros(n, dtype=int)
    comp_split[order[cum > fractions[0] * len(names)]] = 1
    comp_split[order[cum > (fractions[0] + fractions[1]) * len(names)]] = 2
    groups = {nm: ("train", "val", "test")[comp_split[lab[r]]] for r, nm in enumerate(names)}
    tr = np.array([groups[nm] == "train" for nm in names])
    te = np.array([groups[nm] == "test" for nm in names])
    groups["__max_cross_jaccard__"] = float(J[np.ix_(tr, te)].max()) if tr.any() and te.any() else 0.0
    groups["__n_components__"] = int(n)
    return groups


def split_layers(N: int, pairs: dict[str, torch.Tensor], seed: int = 0, tau: float = 0.2, hidden_frac: float = 0.10,
                 val_frac: float = 0.05, test_frac: float = 0.10, min_pairs: int = 0, groups: dict | None = None):
    """Returns (layers sorted by name, hidden [N, N] bool, info). `pairs`: name -> [2, E] with i < j."""
    pairs = {n: canon(e) for n, e in pairs.items() if e.size(1) >= min_pairs}
    names, M, union = pair_matrix(pairs, N)
    if groups is None:
        groups = layer_groups(names, M, tau, seed)
    g = torch.Generator().manual_seed(seed)
    hid = torch.rand(union.numel(), generator=g) < hidden_frac  # the global pair-disjoint holdout
    hidden = torch.zeros(N, N, dtype=torch.bool)
    hidden[union[hid] // N, union[hid] % N] = True
    hidden = hidden | hidden.T
    layers = []
    for k, nm in enumerate(names):
        e = pairs[nm]
        gl = torch.Generator().manual_seed(seed * 1_000_003 + k)
        A = dense(e, N).bool()
        is_hidden = hidden[e[0], e[1]]
        hidden_pos, rest = e[:, is_hidden], e[:, ~is_hidden]
        perm = torch.randperm(rest.size(1), generator=gl)
        n_test, n_val = int(test_frac * rest.size(1)), int(val_frac * rest.size(1))
        test_pos, val_pos, train = rest[:, perm[:n_test]], rest[:, perm[n_test:n_test + n_val]], rest[:, perm[n_test + n_val:]]
        layers.append(Layer(nm, groups[nm], train, val_pos, sample_pairs(A | hidden, n_val, gl), test_pos,
                            sample_pairs(A | hidden, n_test, gl), hidden_pos,
                            sample_pairs(A, hidden_pos.size(1), gl, pool=hidden), e.size(1)))
    info = {"n_layers": len(layers), "n_union_pairs": int(union.numel()), "n_hidden_pairs": int(hid.sum()),
            "groups": {k: sum(l.group == k for l in layers) for k in ("train", "val", "test")},
            "max_cross_jaccard": groups.get("__max_cross_jaccard__"), "n_components": groups.get("__n_components__")}
    return layers, hidden, info
