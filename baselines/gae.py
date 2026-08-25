"""GAE / VGAE / GAT autoencoders via PyG, with early stopping on val AUC.

Reproduction targets (arXiv 2107.02658 Table 2): GAE 90.6+-0.9 AUC / 91.2+-1.0 AP,
VGAE 89.8+-0.9 / 90.3+-1.0. Encoding uses each split's own edge_index
(val = train edges, test = train+val), as RandomLinkSplit intends.
"""
import torch
from torch import nn

from g2l.metrics import evaluate_pairs


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=32, out=16):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out)

    def forward(self, x, edge_index):
        return self.conv2(self.conv1(x, edge_index).relu(), edge_index)


class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=32, out=16):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_dim, hidden)
        self.conv_mu = GCNConv(hidden, out)
        self.conv_logstd = GCNConv(hidden, out)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


class GATEncoder(nn.Module):
    """Community-consensus Cora GAT config (flagged UNVERIFIED in the research notes)."""

    def __init__(self, in_dim, out=16):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.conv1 = GATConv(in_dim, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(64, out, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        return self.conv2(torch.nn.functional.elu(self.conv1(x, edge_index)), edge_index)


def make_score_fn(kind: str, max_epochs=1000, patience=100):
    def score(data, split, device):
        from torch_geometric.nn import GAE, VGAE

        train, val, test = split
        variational = kind == "vgae"
        enc = {"gae": GCNEncoder, "vgae": VariationalGCNEncoder, "gat": GATEncoder}[kind](data.num_features)
        model = (VGAE(enc) if variational else GAE(enc)).to(device)
        opt = (torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4) if kind == "gat"
               else torch.optim.Adam(model.parameters(), lr=0.01))
        x = data.x.to(device)
        ei = {k: s.edge_index.to(device) for k, s in zip(("train", "val", "test"), split)}
        pos = train.pos_edge_label_index.to(device)

        best_auc, best_state, bad = 0.0, None, 0
        for epoch in range(max_epochs):
            model.train()
            opt.zero_grad()
            z = model.encode(x, ei["train"])
            loss = model.recon_loss(z, pos)
            if variational:
                loss = loss + model.kl_loss() / data.num_nodes
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                zv = model.encode(x, ei["val"])
                m = evaluate_pairs(
                    (zv[val.pos_edge_label_index[0]] * zv[val.pos_edge_label_index[1]]).sum(-1),
                    (zv[val.neg_edge_label_index[0]] * zv[val.neg_edge_label_index[1]]).sum(-1),
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
            zt = model.encode(x, ei["test"])
            return (zt @ zt.T).cpu()

    return score
