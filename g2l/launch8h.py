"""Phase 8H launcher -- runs as a Slurm job after the selection stage, so the whole chain
(pretrain-select -> select -> THIS -> pretrain-rest -> final) needs no human in the loop.

  python -m g2l.launch8h --commit <hash>            (submits)
  python -m g2l.launch8h --commit <hash> --dry-run  (prints the sbatch commands only)

The one decision it makes: the LGM's width, by highest validation mean over the selection seeds.
GIN and GCN have no open knob in 8H (pretrained won both in 8G). It refuses to submit anything if
a selection row is missing -- a failed task must be looked at, not papered over -- and records
the decision in winners.json beside the rows.
"""
import argparse
import json
import os
import pathlib
import subprocess

ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase8h/rows"))
LGM_WIDTHS = (64, 128, 256)
REQUIRED = [f"lgm_d{d}_hlinear" for d in LGM_WIDTHS] + ["gin_pretrained", "gcn_pretrained"]


def select(rows: pathlib.Path) -> dict:
    got = {}
    for name in REQUIRED:
        f = rows / f"{name}.json"
        if not f.exists():
            raise SystemExit(f"selection row missing: {f} -- a select task failed; not submitting")
        r = json.loads(f.read_text())
        if len(r.get("val_rocauc", [])) != len(r["seeds"]):
            raise SystemExit(f"selection row incomplete: {f}")
        got[name] = r
    by_d = {d: got[f"lgm_d{d}_hlinear"]["val_mean"] for d in LGM_WIDTHS}
    lgm_d = max(by_d, key=by_d.get)
    return {"lgm_d": lgm_d, "lgm_val_by_d": by_d,
            "gin_val": got["gin_pretrained"]["val_mean"], "gcn_val": got["gcn_pretrained"]["val_mean"],
            "rule": "LGM width = argmax validation mean over select_seeds; GIN/GCN pretrained (8G)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    w = select(ROWS)
    (ROWS / "winners.json").write_text(json.dumps(w, indent=1))
    print("selection:", json.dumps(w), flush=True)

    export = f"ALL,G2L_COMMIT={args.commit},LGM_D={w['lgm_d']}"
    pre = ["sbatch", "--parsable", "--array=0-5", f"--export={export}", "slurm/phase8h_pretrain_rest.sbatch"]
    if args.dry_run:
        print("DRY:", " ".join(pre))
        print("DRY:", " ".join(["sbatch", "--parsable", "--array=0-8", "--dependency=afterok:<pre>",
                                f"--export={export}", "slurm/phase8h_final.sbatch"]))
        return
    a = subprocess.check_output(pre, text=True).strip().split(";")[0]
    print(f"PRETRAIN_REST_JOB={a}", flush=True)
    fin = ["sbatch", "--parsable", "--array=0-8", f"--dependency=afterok:{a}", f"--export={export}",
           "slurm/phase8h_final.sbatch"]
    b = subprocess.check_output(fin, text=True).strip().split(";")[0]
    print(f"FINAL_JOB={b}", flush=True)
    (ROWS / "winners.json").write_text(json.dumps({**w, "pretrain_rest_job": a, "final_job": b}, indent=1))


if __name__ == "__main__":
    main()
