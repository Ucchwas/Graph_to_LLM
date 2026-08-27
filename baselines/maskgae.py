"""MaskGAE (Li et al., KDD 2023, arXiv 2205.10053; github.com/EdisonLeeeee/MaskGAE) in this
harness, following the reference train_linkpred.py defaults (verified against the master branch
2026-08-27): GCN encoder, one layer to 128 channels with BatchNorm (the README's --bn) / ELU /
input dropout 0.8; edge decoder = 2-layer MLP (64 hidden, dropout 0.2) on the Hadamard product;
degree decoder of the same shape regressing each node's masked-edge degree (weight alpha 0.003);
mask ratio p 0.7; negatives from PyG negative_sampling on the train graph, one per masked edge;
Adam 1e-2, weight decay 5e-5, gradient clip 1.0, batches of 2^16 masked edges, 500 epochs, the
state with the best validation AUC (checked every 10 epochs; no early stopping). The eval-time
encoder input is the split's edge_index (train + val at test), as in the reference.
Deviations: edge-wise masking (their MaskGAE_edge row, Cora 96.42 / 95.91; path masking needs
torch_cluster), applied to undirected pairs -- the reference masks directed entries and
re-symmetrises the remainder, so ~21 % of its targets stay visible to the encoder. `featureless`:
X = I, the published featureless-autoencoder convention (their script always feeds features)."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree, negative_sampling, to_undirected

from g2l.metrics import evaluate_pairs


def mlp(d_in, d_hidden, layers, dropout):
    dims = [d_in] + [d_hidden] * (layers - 1) + [1]
    mods = []
    for k in range(layers):
        mods += [nn.Dropout(dropout), nn.Linear(dims[k], dims[k + 1])]
        if k < layers - 1:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


class Encoder(nn.Module):
    def __init__(self, d_in, d_hidden=128, layers=1, dropout=0.8, bn=True):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(d_in if k == 0 else d_hidden, d_hidden) for k in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d_hidden) if bn else nn.Identity() for _ in range(layers)])
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(F.dropout(x, self.dropout, self.training), edge_index)))
        return x


@torch.no_grad()
def score_full(dec, z, chunk=1 << 20):
    """Every pair through the pair MLP, a block of rows at a time (N^2 x d is never held)."""
    N, d = z.shape
    out = torch.empty(N, N, device=z.device)
    rows = max(1, chunk // N)
    for s in range(0, N, rows):
        h = z[s:s + rows, None, :] * z[None, :, :]
        out[s:s + rows] = dec(h.reshape(-1, d)).view(-1, N)
    return out


def make_score_fn(featureless=True, p=0.7, epochs=500, lr=0.01, weight_decay=5e-5, alpha=0.003,
                  eval_period=10, batch_size=1 << 16, grad_norm=1.0, hidden=128, dec_hidden=64, seed=0):
    def score(data, split, device):
        train, val, test = split
        N = data.num_nodes
        x = torch.eye(N, device=device) if featureless else data.x.to(device)
        torch.manual_seed(seed)
        enc = Encoder(x.size(1), hidden).to(device)
        edec, ddec = mlp(hidden, dec_hidden, 2, 0.2).to(device), mlp(hidden, dec_hidden, 2, 0.2).to(device)
        params = [*enc.parameters(), *edec.parameters(), *ddec.parameters()]
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        pos_und = train.pos_edge_label_index.to(device)  # one direction per train edge
        ei = {k: s.edge_index.to(device) for k, s in zip(("train", "val", "test"), split)}
        vp, vn = val.pos_edge_label_index.to(device), val.neg_edge_label_index.to(device)

        def pairs(z, e):
            return edec(z[e[0]] * z[e[1]]).squeeze(-1)

        best, best_state = -1.0, None
        for epoch in range(epochs):
            m = torch.rand(pos_und.size(1), device=device) < p
            remaining, masked = to_undirected(pos_und[:, ~m], num_nodes=N), pos_und[:, m]
            deg = degree(masked[1], N).float()
            enc.train(), edec.train(), ddec.train()
            perm = torch.randperm(masked.size(1), device=device)
            for s in range(0, masked.size(1), batch_size):
                pe = masked[:, perm[s:s + batch_size]]
                ne = negative_sampling(ei["train"], num_nodes=N, num_neg_samples=pe.size(1))
                opt.zero_grad()
                z = enc(x, remaining)
                po, no = pairs(z, pe), pairs(z, ne)
                loss = (F.binary_cross_entropy_with_logits(po, torch.ones_like(po))
                        + F.binary_cross_entropy_with_logits(no, torch.zeros_like(no)))
                if alpha:
                    loss = loss + alpha * F.mse_loss(ddec(z).squeeze(-1), deg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, grad_norm)
                opt.step()
            if epoch % eval_period:
                continue
            enc.eval(), edec.eval()
            with torch.no_grad():
                z = enc(x, ei["val"])
                auc = evaluate_pairs(pairs(z, vp), pairs(z, vn))["auroc"]
            if auc > best:
                best = auc
                best_state = [{k: v.detach().clone() for k, v in mod.state_dict().items()} for mod in (enc, edec)]
        enc.load_state_dict(best_state[0]), edec.load_state_dict(best_state[1])
        enc.eval(), edec.eval()
        with torch.no_grad():
            return score_full(edec, enc(x, ei["test"])).cpu()

    return score
