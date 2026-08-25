"""Logistic regression on [x_i || x_j || x_i * x_j]; no graph structure at all.

The learned linear form decomposes as a_i + b_j + (x_i * w3) . x_j, so the full
[N, N] score matrix is one matmul -- no 3.66M-row feature matrix is ever built.
"""
import numpy as np
import torch


def score_fn(data, split, device):
    from sklearn.linear_model import LogisticRegression
    from torch_geometric.utils import negative_sampling

    train = split[0]
    X = data.x.numpy()
    pos = train.pos_edge_label_index
    neg = negative_sampling(pos, data.num_nodes, num_neg_samples=pos.shape[1])

    def pairs(ei):
        i, j = ei.numpy()
        both_i = np.concatenate([i, j])  # train both directions -> symmetric weights
        both_j = np.concatenate([j, i])
        return np.hstack([X[both_i], X[both_j], X[both_i] * X[both_j]])

    F = X.shape[1]
    feats = np.vstack([pairs(pos), pairs(neg)]).astype(np.float32)
    y = np.concatenate([np.ones(2 * pos.shape[1]), np.zeros(2 * neg.shape[1])])
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(feats, y)

    w = clf.coef_[0].astype(np.float32)
    a = X @ w[:F]
    b = X @ w[F : 2 * F]
    S = a[:, None] + b[None, :] + (X * w[2 * F :]) @ X.T + clf.intercept_[0]
    return torch.from_numpy(((S + S.T) / 2).astype(np.float32))
