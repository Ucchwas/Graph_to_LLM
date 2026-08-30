"""Phase 9 runner -- one fold per invocation; every seed and both bodies inside it.

  python -m g2l.run_phase9 --stage build                      (CPU, once: data/phase9/tcga9.pt)
  python -m g2l.run_phase9 --stage run --fold 3               (GPU: one held-out cancer)
  python -m g2l.run_phase9 --stage run --fold 3 --cpu --epochs 2 --limit-genes 200   (smoke)

Per (seed, body): SSL on the training cancers' graphs -> normal->tumour fine-tuning with selection
on the validation cancer -> the test cancer scored once. Rows: one JSON per (cancer, body, seed).
"""
import argparse
import json
import os
import pathlib
import subprocess

import torch
import yaml

from g2l import tcga9

ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase9/rows"))
BODIES = ("lgm", "edgegcn")


def commit_hash() -> str:
    if os.environ.get("G2L_COMMIT"):
        return os.environ["G2L_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def shrink(ds: dict, n_genes: int) -> dict:
    """Smoke only: keep the first `n_genes` genes of every graph."""
    out = {**ds, "n": n_genes, "genes": ds["genes"][:n_genes], "graphs": {}}
    for k, g in ds["graphs"].items():
        m = (g.edge_index < n_genes).all(0)
        out["graphs"][k] = tcga9.CGraph(n_genes, g.edge_index[:, m], g.edge_value[m], g.x[:n_genes])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase9.yaml")
    ap.add_argument("--stage", required=True, choices=["build", "run"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma list; default from the config")
    ap.add_argument("--bodies", default=None, help="comma list; default: the config's `bodies`, "
                    "else lgm,edgegcn (Phase 9)")
    ap.add_argument("--epochs", type=int, default=None, help="smoke: caps SSL and fine-tune epochs")
    ap.add_argument("--limit-genes", type=int, default=None, help="smoke: shrink every graph")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())

    if args.stage == "build":
        p7 = pathlib.Path(os.environ.get("G2L_PHASE7_ROOT", "data/phase7")) / "processed"
        ds = tcga9.build(p7 / f"pairs_rho{cfg['density']}.pt", p7 / "expr.pt", tcga9.data_path())
        print(f"wrote {tcga9.data_path()}: {len(ds['cancers'])} cancers x 2 conditions, N={ds['n']}")
        return

    if args.epochs:
        cfg["ssl_epochs"] = cfg["max_epochs"] = args.epochs
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    bodies = args.bodies.split(",") if args.bodies else list(cfg.get("bodies", BODIES))
    ds = tcga9.load()
    if args.limit_genes:
        ds = shrink(ds, args.limit_genes)
    fold = args.fold if args.fold is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    test_c, val_c, train_c = tcga9.folds(ds["cancers"])[fold]
    corpus = tcga9.ssl_corpus(ds, train_c)
    print(f"fold {fold}: test={test_c} val={val_c} train={len(train_c)} | N={ds['n']} "
          f"ssl graphs {len(corpus)} | device {device} | commit {commit_hash()[:8]}", flush=True)

    ROWS.mkdir(parents=True, exist_ok=True)
    counts = {}
    for seed in seeds:
        for body in bodies:
            model = tcga9.build_model(body, cfg, seed)
            counts[body] = tcga9.n_params(model)
            print(f"\n[seed {seed}] {body}: {counts[body]:,} params", flush=True)
            s = tcga9.pretrain(model, corpus, cfg, device, seed)
            f = tcga9.translate(model, ds, train_c, val_c, cfg, device, seed)
            r = tcga9.test(model, ds, test_c, device, seed)
            row = {"cancer": test_c, "val_cancer": val_c, "fold": fold, "body": body, "seed": seed,
                   "n_params": counts[body], "commit": commit_hash(), "density": ds["density"],
                   "n_genes": ds["n"], "smoke": bool(args.epochs or args.limit_genes), **s, **f, **r}
            key = f"{test_c.replace(' ', '_').replace('&', 'and')}_{body}_seed{seed}"
            (ROWS / f"{key}.json").write_text(json.dumps(row))
            print(f"[seed {seed}] {body}: val_changed {f['val_changed_auc']:.4f} | TEST changed AUROC "
                  f"{r['auc_changed']:.4f} AP {r['ap_changed']:.4f} overall AUROC {r['auc']:.4f} "
                  f"direction {r['auc_direction']:.4f} | wrote {key}.json", flush=True)
    names = list(counts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"parameter match: {a} {counts[a]:,} vs {b} {counts[b]:,} "
                  f"({100 * abs(counts[a] - counts[b]) / counts[a]:.2f}%)")


if __name__ == "__main__":
    main()
