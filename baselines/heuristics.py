"""Zero-parameter link-prediction heuristics on the observed graph."""
import torch

from baselines.common import observed_dense


def common_neighbors(A):
    return A @ A


def adamic_adar(A):
    deg = A.sum(1)
    w = torch.where(deg > 1, 1.0 / torch.log(deg.clamp(min=2.0)), torch.zeros_like(deg))
    return A @ torch.diag(w) @ A


def resource_allocation(A):
    deg = A.sum(1).clamp(min=1.0)
    return A @ torch.diag(1.0 / deg) @ A


def ppr(A, alpha=0.15):
    deg = A.sum(1).clamp(min=1.0)
    P = A / deg[:, None]
    S = alpha * torch.linalg.inv(torch.eye(A.shape[0]) - (1 - alpha) * P)
    return (S + S.T) / 2


HEURISTICS = {
    "common_neighbors": common_neighbors,
    "adamic_adar": adamic_adar,
    "resource_allocation": resource_allocation,
    "ppr": ppr,
}


def make_score_fn(kind: str):
    def score(data, split, device):
        _, _, test = split
        A = observed_dense(test, data.num_nodes)  # train+val edges, the test-time graph
        return HEURISTICS[kind](A)

    return score
