"""Phase 6, the shared arm: one model (raw rows -> shared first GCN -> residual GCN block ->
bilinear decoder; nothing per layer) trained on every train layer of an aligned layer set, one
layer per step with the masked-cell loss (held-out and globally hidden cells excluded from
supervision), selected on the mean validation AUROC of the val layers' completion, then scored
on every layer by the single-graph scorer -- the same rows the per-layer arm produces."""
import time

import torch

from g2l.data import mask_matrix
from g2l.layers import dense
from g2l.metrics import evaluate_edge_split, evaluate_pairs
from g2l.train import make_optimizer, masked_loss, pos_weight_of


@torch.no_grad()
def val_auroc(model, L, N, device):
    Lv = model(dense(L.train, N).to(device))
    return evaluate_pairs(Lv[L.val_pos[0], L.val_pos[1]], Lv[L.val_neg[0], L.val_neg[1]])["auroc"]


def train_shared(model, layers, hidden, N, cfg: dict, device, seed: int, log=print) -> dict:
    torch.manual_seed(seed)
    tr = [l for l in layers if l.group == "train"]
    va = [l for l in layers if l.group == "val"]
    assert tr and va, "the shared arm needs train and val layers"
    model.to(device)
    opt = make_optimizer(model, cfg["lr"], cfg["lr_norm"], cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    pw = [torch.tensor(pos_weight_of(dense(l.train, N)), device=device) for l in tr]
    g = torch.Generator().manual_seed(seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_epoch, bad, best_state, t0 = 0.0, 0, 0, None, time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        tot = 0.0
        for k in torch.randperm(len(tr), generator=g).tolist():
            L = tr[k]
            A_in = dense(L.train, N)
            A_obs, sup = mask_matrix(A_in, cfg["mask_frac"], seed=epoch * 1_000_003 + k, exclude=L.exclude(N, hidden))
            i, j = (t.to(device) for t in sup.nonzero(as_tuple=True))
            opt.zero_grad(set_to_none=True)
            loss = masked_loss(model.pairs(A_obs.to(device), i, j), A_in.to(device)[i, j], pw[k])
            loss.backward()
            if epoch == 0 and not tot:
                missing = [n for n, p in trainable.items() if p.grad is None]
                assert not missing, f"no gradient reached: {missing[:5]}"
            torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        v = sum(val_auroc(model, l, N, device) for l in va) / len(va)
        if v > best:
            best, best_epoch, bad = v, epoch, 0
            best_state = {n: p.detach().clone() for n, p in trainable.items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        log(f"epoch {epoch:3d} loss {tot / len(tr):.4f} val_layers_auroc {v:.4f} best {best:.4f}@{best_epoch}")
    epochs, wall = epoch + 1, time.time() - t0
    model.load_state_dict(best_state, strict=False)
    return {"val_auc": best, "best_epoch": best_epoch, "epochs": epochs, "sec_per_epoch": wall / epochs,
            "wallclock_s": round(wall, 1), "n_trainable": sum(p.numel() for p in trainable.values()),
            "n_train_layers": len(tr), "n_val_layers": len(va),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None}


def subsample(e: torch.Tensor, frac: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return e[:, torch.rand(e.size(1), generator=g) < frac]


@torch.no_grad()
def layer_rows(model, layers, hidden, N, device, fracs=(), seed: int = 0) -> list[dict]:
    """One scored row per layer from a shared model (input = the layer's train + val pairs, which
    a held-out layer's model never trained on); test layers also from reduced inputs (`fracs`
    of those pairs kept, the test pairs unchanged)."""
    model.eval()
    rows = []
    for L in layers:
        split = L.split(N, hidden)
        inp = torch.cat([L.train, L.val_pos], 1)
        row = evaluate_edge_split(model(dense(inp, N).to(device)).cpu(), split, seed=seed)
        row.update(layer=L.name, group=L.group, n_pairs=L.n_pairs, n_train=int(L.train.size(1)), n_test=int(L.test_pos.size(1)))
        if L.group == "test":
            for f in fracs:
                lf = model(dense(subsample(inp, f, seed), N).to(device)).cpu()
                m = evaluate_pairs(lf[L.test_pos[0], L.test_pos[1]], lf[L.test_neg[0], L.test_neg[1]])
                row[f"auc_in{f}"], row[f"ap_in{f}"] = m["auroc"], m["ap"]
        rows.append(row)
    return rows
