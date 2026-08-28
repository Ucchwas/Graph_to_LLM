"""Phase-7 runner: normal-graph -> tumour-graph translation on TCGA co-expression graphs.

  python -m g2l.run_phase7 --stage build      (CPU: cancer table, gene universe, one graph pair per cancer)
  python -m g2l.run_phase7 --stage gate       (CPU: bootstrap stability -- must pass before any GPU work)
  python -m g2l.run_phase7 --stage baselines  (CPU: identity / mean-tumour / mean-change / CN, leave-one-cancer-out)
  python -m g2l.run_phase7 --stage translate --index 0   (GPU: one fold = one held-out cancer)

Rows land under $G2L_ROWS (default results/phase7/rows), one JSON per (cancer, arm, density).
"""
import argparse
import json
import os
import pathlib
import subprocess
import traceback

import numpy as np
import torch
import yaml

from g2l.model import build_model
from g2l.tcga import (cancer_table, condition_graph, gene_universe, jaccard, load_expression,
                      root, sha256, spearman, subsample, binarise)
from g2l.translate import baseline_scores, score_all, train_translation

CONFIG = pathlib.Path("configs/phase7.yaml")
ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase7/rows"))
RUNS = pathlib.Path(os.environ.get("G2L_RUNS", "results/runs/phase7"))
BASELINES = ("identity", "mean_tumour", "mean_change", "common_neighbors")


def commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def pairs_path(density: float) -> pathlib.Path:
    return root() / "processed" / f"pairs_rho{density}.pt"


def write_row(row: dict, key: str):
    ROWS.mkdir(parents=True, exist_ok=True)
    tmp = (ROWS / f"{key}.json").with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    os.replace(tmp, ROWS / f"{key}.json")


def build(cfg: dict):
    """One aggregate normal graph and one aggregate tumour graph per included cancer, on one
    fixed gene universe. Tumour samples are subsampled to the cancer's normal count so both
    correlation estimates carry the same sampling noise."""
    table = cancer_table(cfg["min_normal"])
    inc = [r for r in table if r["included"]]
    print(f"{len(inc)} cancers with >= {cfg['min_normal']} solid-tissue normals:")
    for r in inc:
        print(f"  {r['disease'][:45]:45s} normal {r['n_normal']:4d}  tumour {r['n_tumour']:5d}  matched {r['n_matched']:4d}")
    assert len(inc) >= cfg["min_cancers"], f"only {len(inc)} cancers pass -- Phase 7 reported infeasible"

    samples, cols = [], {}
    for r in inc:
        tum = subsample(r["tumour"], r["n_normal"], seed=cfg["seed"])
        cols[r["disease"]] = {"normal": r["normal"], "tumour": tum}
        samples += r["normal"] + tum
    samples = sorted(set(samples))
    print(f"\nstreaming expression for {len(samples)} samples ...", flush=True)
    genes, expr = load_expression(samples)
    pos = {s: k for k, s in enumerate(samples)}
    normal_cols = sorted({pos[s] for r in inc for s in r["normal"]})
    gene_idx = gene_universe(genes, expr, normal_cols, cfg["n_genes"], cfg["expr_floor"])
    print(f"genes {len(genes)} -> universe {len(gene_idx)} (expressed on normals, top variance)")

    for density in cfg["densities"]:
        out = {"cancers": [r["disease"] for r in inc], "genes": [genes[i] for i in gene_idx],
               "density": density, "normal": {}, "tumour": {}, "samples": cols}
        for r in inc:
            c = r["disease"]
            for cond in ("normal", "tumour"):
                out[cond][c] = condition_graph(expr, gene_idx, [pos[s] for s in cols[c][cond]], density)
            print(f"  rho={density} {c[:40]:40s} edges {int(out['normal'][c].sum() // 2):6d} "
                  f"jaccard(N,T) {jaccard(out['normal'][c], out['tumour'][c]):.4f}", flush=True)
        pairs_path(density).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, pairs_path(density))
    print("\nchecksums:")
    for f in sorted((root() / "raw").glob("*.gz")):
        print(f"  {f.name}  {f.stat().st_size} bytes  sha256 {sha256(f)}")


def gate(cfg: dict):
    """Gate 7.0: is the normal->tumour change bigger than resampling noise? Bootstrap the samples
    within each condition and compare the pairwise bootstrap Jaccard with the normal-vs-tumour
    Jaccard on the full sample sets."""
    o = torch.load(pairs_path(cfg["densities"][0]))
    samples = o["samples"]
    all_s = sorted({s for c in o["cancers"] for cond in ("normal", "tumour") for s in samples[c][cond]})
    full_genes, expr = load_expression(all_s)
    pos = {s: k for k, s in enumerate(all_s)}
    at = {g: i for i, g in enumerate(full_genes)}
    keep = [at[g] for g in o["genes"]]  # the saved universe, in its saved order
    rows = []
    for c in o["cancers"][:cfg["gate_cancers"]]:
        rec = {"cancer": c, "n_normal": len(samples[c]["normal"])}
        for cond in ("normal", "tumour"):
            cs = [pos[s] for s in samples[c][cond]]
            g = torch.Generator().manual_seed(cfg["seed"])
            boots = []
            for _ in range(cfg["bootstraps"]):
                pick = [cs[i] for i in torch.randint(len(cs), (len(cs),), generator=g).tolist()]
                boots.append(binarise(spearman(expr[keep][:, pick]), o["density"]))
            js = [jaccard(boots[i], boots[j]) for i in range(len(boots)) for j in range(i + 1, len(boots))]
            rec[f"{cond}_boot_jaccard_mean"] = float(np.mean(js))
            rec[f"{cond}_boot_jaccard_p5"] = float(np.percentile(js, 5))
        rec["normal_vs_tumour_jaccard"] = jaccard(o["normal"][c], o["tumour"][c])
        rec["pass"] = bool(rec["normal_vs_tumour_jaccard"] <
                           min(rec["normal_boot_jaccard_p5"], rec["tumour_boot_jaccard_p5"]))
        print(f"{c[:40]:40s} boot N {rec['normal_boot_jaccard_mean']:.4f} (p5 {rec['normal_boot_jaccard_p5']:.4f}) "
              f"boot T {rec['tumour_boot_jaccard_mean']:.4f} (p5 {rec['tumour_boot_jaccard_p5']:.4f}) "
              f"N-vs-T {rec['normal_vs_tumour_jaccard']:.4f}  {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
        rows.append(rec)
    verdict = all(r["pass"] for r in rows)
    (ROWS.parent).mkdir(parents=True, exist_ok=True)
    (ROWS.parent / "gate.json").write_text(json.dumps({"rows": rows, "pass": verdict,
                                                       "bootstraps": cfg["bootstraps"], "commit": commit_hash()}))
    print(f"\nGate 7.0: {'PASS' if verdict else 'FAIL'} "
          f"({sum(r['pass'] for r in rows)}/{len(rows)} cancers) -- written to {ROWS.parent / 'gate.json'}")


def folds(cancers: list[str]) -> list[tuple[str, str, list[str]]]:
    """Leave-one-cancer-out: (test, val, train). Validation is the next cancer in a fixed
    rotation, so no fold ever selects on its own test cancer."""
    return [(c, cancers[(k + 1) % len(cancers)],
             [x for x in cancers if x != c and x != cancers[(k + 1) % len(cancers)]])
            for k, c in enumerate(cancers)]


def run_baselines(cfg: dict, density: float):
    o = torch.load(pairs_path(density))
    commit = commit_hash()
    for test, _, train in folds(o["cancers"]):
        A_n, A_t = o["normal"][test], o["tumour"][test]
        tn, tt = [o["normal"][c] for c in train], [o["tumour"][c] for c in train]
        for name in BASELINES:
            row = score_all(baseline_scores(name, A_n, tn, tt), A_n, A_t, seed=cfg["seed"])
            key = f"{test.replace(' ', '_')}_{name}_rho{density}"
            row.update(cancer=test, arm=name, density=density, commit=commit, key=key,
                       n_train_cancers=len(train), n_genes=len(o["genes"]))
            write_row(row, key)
            print(f"{test[:35]:35s} {name:17s} auc {row['auc']:.4f} changed {row['auc_changed']:.4f} "
                  f"(gained {row['auc_gained']:.4f} lost {row['auc_lost']:.4f}) dir {row['auc_direction']:.4f}", flush=True)


def run_translate(cfg: dict, density: float, index: int, device: str, epochs=None):
    o = torch.load(pairs_path(density))
    fold = folds(o["cancers"])[index]
    test, val, train = fold
    commit = commit_hash()
    key = f"{test.replace(' ', '_')}_shared_rho{density}"
    print(f"fold {index}: test={test} val={val} train={len(train)} cancers @ {commit[:8]}", flush=True)
    N = len(o["genes"])
    model = build_model("gnn_direct", N, cfg["width"], 1.0, scratch_layers=cfg["layers"], seed=cfg["seed"], kind=cfg["kind"])
    tcfg = {**{k: cfg[k] for k in ("lr", "lr_norm", "weight_decay", "warmup", "patience")},
            "max_epochs": epochs or cfg["max_epochs"]}
    summary = train_translation(model, o, train, [val], tcfg, device, cfg["seed"])
    A_n, A_t = o["normal"][test].to(device), o["tumour"][test].to(device)
    model.eval()
    with torch.no_grad():
        row = score_all(model(A_n).cpu(), A_n.cpu(), A_t.cpu(), seed=cfg["seed"])
    row.update(summary, cancer=test, arm="shared", density=density, commit=commit, key=key,
               val_cancer=val, n_train_cancers=len(train), n_genes=N)
    write_row(row, key)
    if not epochs:
        RUNS.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), RUNS / f"{key}.pt")
    print(f"done {key} auc {row['auc']:.4f} changed {row['auc_changed']:.4f} "
          f"(gained {row['auc_gained']:.4f} lost {row['auc_lost']:.4f}) dir {row['auc_direction']:.4f} "
          f"[{row['wallclock_s']}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", required=True, choices=["build", "gate", "baselines", "translate"])
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if args.stage == "build":
        return build(cfg)
    if args.stage == "gate":
        return gate(cfg)
    densities = [args.density] if args.density else cfg["densities"]
    if args.stage == "baselines":
        for d in densities:
            run_baselines(cfg, d)
        return
    idx = args.index if args.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    failed = []
    for d in densities:
        try:
            run_translate(cfg, d, idx, device, args.epochs)
        except Exception:
            traceback.print_exc()
            failed.append(f"fold{idx}_rho{d}")
        if device == "cuda":
            torch.cuda.empty_cache()
    if failed:
        raise SystemExit(f"{len(failed)} run(s) failed: {failed}")


if __name__ == "__main__":
    main()
