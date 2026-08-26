"""Phase-2 training: masked-cell objective, val-AUROC early stopping, one row per run.

Each step hides a fresh 15% of the train graph's upper triangle (mirrored into the input) and
supervises exactly those cells with pos-weighted BCE. Cells visible in the input are never
supervised: adjacency-row tokens make any such loss solvable by copying the input.
"""
import time

import torch
import torch.nn.functional as F

from baselines.common import observed_dense, recon_bce
from g2l.data import mask_matrix, rewire_degree_preserving, subsample_edges
from g2l.metrics import evaluate_edge_split, evaluate_pairs
from g2l.model import param_groups


def pos_weight_of(A_train: torch.Tensor) -> float:
    N = A_train.shape[0]
    n_edges = torch.triu(A_train, 1).sum().item()
    return (N * (N - 1) / 2 - n_edges) / n_edges


def masked_loss(logits, target, pos_weight):
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def make_optimizer(model, lr, lr_norm, weight_decay, lr_bias=None):
    """Differential LRs: encoder/decoder and the scratch body at `lr`; RMSNorm gains at
    `lr_norm`; the zero-init bias table at `lr_bias` (no weight decay -- it starts at 0)."""
    g = param_groups(model)
    assert not g["lora"], "LoRA is Phase 3B"
    assert bool(g["bias_table"]) == (lr_bias is not None), "bias table present iff lr_bias is set"
    groups = [{"params": g["main"], "lr": lr, "weight_decay": weight_decay}]
    if g["rmsnorm"]:
        groups.append({"params": g["rmsnorm"], "lr": lr_norm, "weight_decay": 0.0})
    if g["scratch"]:
        groups.append({"params": g["scratch"], "lr": lr, "weight_decay": weight_decay})
    if g["bias_table"]:
        groups.append({"params": g["bias_table"], "lr": lr_bias, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def _grad_norm(params):
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))


@torch.no_grad()
def first_batch_stats(model, A, pos):
    """Decoder-input scale, D1-style logit scale and per-layer hidden RMS before any training."""
    model.eval()
    Z = model.embed(A)
    L = model(A)
    off = L[~torch.eye(A.shape[0], dtype=torch.bool, device=A.device)]
    stats = {"z_norm": Z.norm(dim=-1).mean().item(), "logit_diag": L.diagonal().mean().item(),
             "logit_offdiag_std": off.std().item(), "logit_train_edge": L[pos[0], pos[1]].mean().item()}
    layers = getattr(getattr(model.body, "llm", None), "layers", None) or getattr(getattr(model.body, "enc", None), "layers", None)
    if layers is not None:
        rms = []
        hooks = [l.register_forward_hook(lambda m, i, o: rms.append((o[0] if isinstance(o, tuple) else o).float().pow(2).mean().sqrt().item())) for l in layers]
        model.embed(A)
        for h in hooks:
            h.remove()
        stats["layer_rms"] = rms
    return stats


def train_run(model, data, split, cfg: dict, device, seed: int, loss: str = "masked",
              shuffle_input: bool = False, train_frac: float = 1.0, log=print) -> tuple[dict, dict, torch.Tensor]:
    """Returns (row, trainable state at the best epoch, test logits [N, N]).
    `train_frac` < 1 keeps a seeded fraction of the train edges (input and supervision; val
    and test untouched) for the data-fraction sweep."""
    train, val, test = split
    N = data.num_nodes
    A_train = observed_dense(train, N)
    if train_frac < 1.0:
        A_train = subsample_edges(A_train, train_frac, seed)
    A_in = rewire_degree_preserving(A_train, seed) if shuffle_input else A_train
    pw = torch.tensor(pos_weight_of(A_train), device=device)
    A_train, A_in = A_train.to(device), A_in.to(device)
    pos = A_train.nonzero().T
    model.to(device)
    if cfg.get("checkpointing") and hasattr(model.body, "checkpointing"):
        model.body.checkpointing(True)
    if hasattr(model.body, "canary"):
        model.body.canary()
    opt = make_optimizer(model, cfg["lr"], cfg["lr_norm"], cfg["weight_decay"], cfg.get("lr_bias"))
    table = model.bias.bias_table if model.bias is not None else None
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    enc_params = list(model.encoder.parameters())
    row = {"init": first_batch_stats(model, A_in, pos), "grad_norm_enc": {}}

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_epoch, bad, best_state = 0.0, 0, 0, None
    t0 = time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        if loss == "masked":
            A_obs, sup = mask_matrix(A_in.cpu(), cfg["mask_frac"], seed=epoch)
            i, j = sup.to(device).nonzero(as_tuple=True)
            step_loss = masked_loss(model.pairs(A_obs.to(device), i, j), A_train[i, j], pw)
        else:  # recon_bce: the shortcut control (supervises cells visible in the input)
            step_loss = recon_bce(model(A_in), pos, N)
        step_loss.backward()
        if epoch in (0, 100):
            row["grad_norm_enc"][epoch] = _grad_norm(enc_params)
        if epoch == 0:
            missing = [n for n, p in trainable.items() if p.grad is None]
            assert not missing, f"no gradient reached: {missing[:5]}"
            assert row["grad_norm_enc"][0] > 0, "encoder received a zero gradient"
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
        opt.step()
        sched.step()
        if table is not None and epoch == 100:
            assert table.abs().max() > 0, "bias table still zero after 100 steps"

        model.eval()
        with torch.no_grad():
            Lv = model(A_in)
            m = evaluate_pairs(Lv[val.pos_edge_label_index[0], val.pos_edge_label_index[1]],
                               Lv[val.neg_edge_label_index[0], val.neg_edge_label_index[1]])
        if m["auroc"] > best:
            best, best_epoch, bad = m["auroc"], epoch, 0
            best_state = {n: p.detach().clone() for n, p in trainable.items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        if epoch % 100 == 0:
            log(f"epoch {epoch:4d} loss {step_loss.item():.4f} val_auroc {m['auroc']:.4f} best {best:.4f}@{best_epoch}"
                + (f" max|bias| {table.abs().max().item():.3f}" if table is not None else ""))
    epochs = epoch + 1
    wall = time.time() - t0

    res = model.load_state_dict(best_state, strict=False)
    assert not res.unexpected_keys
    # shuffled control: the same rewired graph is the input at train and test time, so tokens
    # keep their (structureless) node identity and only real neighbourhood structure is removed
    A_test = A_in if shuffle_input else observed_dense(test, N).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(A_test).cpu()
    row.update(evaluate_edge_split(logits, split, seed=seed))
    row.update(val_auc=best, best_epoch=best_epoch, epochs=epochs, n_trainable=sum(p.numel() for p in trainable.values()),
               pos_weight=pw.item(), n_train_edges=int(torch.triu(A_train, 1).sum().item()),
               sec_per_epoch=wall / epochs, wallclock_s=round(wall, 1),
               peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None)
    if table is not None:
        row["bias_table"] = [[round(v, 4) for v in h] for h in table.detach().cpu().tolist()]
    return row, best_state, logits
