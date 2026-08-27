"""Phase-5 table: one row per (dataset, model). results/phase5/rows -> results/phase5/aggregate.md

  python -m g2l.aggregate5 [--rows results/phase5/rows]
"""
import argparse
import glob
import json
import pathlib

COLS = [("auc", "AUROC"), ("ap", "AP@1:1"), ("auroc_sparse", "AUROC sparse"), ("ap_sparse", "AP sparse"),
        ("lift", "lift"), ("val_hits20", "val Hits@20"), ("hits20", "test Hits@20")]
ORDER = ["gnn_direct", "identity", "random", "common_neighbors", "ppr", "gae0", "vgae0", "maskgae0", "maskgae_x"]


def fmt(v, k):
    if v is None:
        return "-"
    return f"{v:.0f}" if k == "lift" else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/phase5/rows")
    args = ap.parse_args()
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{args.rows}/*.json"))]
    out = ["# Phase 5 -- dataset feasibility: direct GCN vs baselines, one seed", ""]
    for ds in sorted({r["dataset"] for r in rows}, key=["cora", "ddi", "photo"].index):
        sub = sorted([r for r in rows if r["dataset"] == ds], key=lambda r: (ORDER.index(r["model"]), r["seed"]))
        n = next((r.get("n_nodes") for r in sub if r.get("n_nodes")), "?")
        out += [f"## {ds}  (N = {n}; rows @ {', '.join(sorted({r['commit'][:8] for r in sub}))})", "",
                "| model | seed | " + " | ".join(c for _, c in COLS) + " | best epoch / epochs | s/epoch | peak GB | wall s |",
                "|---|---|" + "---|" * len(COLS) + "---|---|---|---|"]
        for r in sub:
            name = r["key"].split("_", 1)[1].rsplit("_s", 1)[0] if r["model"] == "gnn_direct" else r["model"]
            ep = f"{r['best_epoch']} / {r['epochs']}" if "epochs" in r else "-"
            spe = f"{r['sec_per_epoch']:.2f}" if "sec_per_epoch" in r else "-"
            mem = f"{r['peak_mem_gb']:.2f}" if r.get("peak_mem_gb") else "-"
            out.append(f"| {name} | {r['seed']} | " + " | ".join(fmt(r.get(k), k) for k, _ in COLS)
                       + f" | {ep} | {spe} | {mem} | {r.get('wallclock_s', '-')} |")
        out.append("")
    text = "\n".join(out)
    print(text)
    pathlib.Path(args.rows).parent.joinpath("aggregate.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
