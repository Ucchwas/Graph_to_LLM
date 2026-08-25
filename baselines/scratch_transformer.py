"""Adjacency-row tokens into a randomly initialised transformer, trained from
scratch: the control that isolates what a pretrained LLM contributes (Phase-2
row 4). Input = raw adjacency rows only, matching encoder E1. No positional
embeddings: at fixed N the row itself carries identity.
"""
import torch
from torch import nn

from baselines.common import observed_dense, recon_bce
from g2l.metrics import evaluate_pairs


class ScratchAdjTransformer(nn.Module):
    def __init__(self, n_nodes, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_nodes, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.W = nn.Parameter(torch.eye(d_model) * 0.1)

    def forward(self, A):
        h = self.enc(self.inp(A).unsqueeze(0)).squeeze(0)
        return h @ self.W @ h.T


def train_one(data, split, device, d_model, lr, max_epochs=300, patience=30, seed=0):
    torch.manual_seed(seed)
    train, val, test = split
    N = data.num_nodes
    A_train = observed_dense(train, N).to(device)
    A_val = observed_dense(val, N).to(device)  # = train edges
    pos = train.pos_edge_label_index.to(device)

    model = ScratchAdjTransformer(N, d_model=d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_auc, best_state, bad = 0.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = recon_bce(model(A_train), pos, N)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            Lv = model(A_val)
            m = evaluate_pairs(
                Lv[val.pos_edge_label_index[0], val.pos_edge_label_index[1]],
                Lv[val.neg_edge_label_index[0], val.neg_edge_label_index[1]],
            )
        if m["auroc"] > best_auc:
            best_auc, bad = m["auroc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(observed_dense(test, N).to(device)).cpu()
    n_params = sum(p.numel() for p in model.parameters())
    return logits, best_auc, n_params
