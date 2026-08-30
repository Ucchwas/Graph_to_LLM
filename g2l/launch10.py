"""Phase 10 MolHIV chain driver -- runs inside Slurm so no human is in the loop.

Two steps, both submitted as tiny jobs by the previous stage's dependency:

  --step chain    after stage 1 (pretrain-select): submit stage 2 (select) and the stage-3 job
  --step select   after stage 2: choose each arm's width on VALIDATION, submit stages 4 and 5

The one decision made anywhere in the chain is stage-3's: per arm, the width with the highest
validation mean over the selection seeds. That is Phase 8H's rule, kept deliberately -- it makes
the plain `lgm`, `gin` and `gcn` arms a direct replication of 8H, and it applies identically to all
six arms, so it cannot favour one. 8H recorded that this rule prefers the largest width and that
size predicts a WORSE test score on molhiv's scaffold split; that bias is now shared by every arm
rather than borne by one.

Refuses to submit if any row is missing: a failed task must be looked at, not papered over.

The account's submitted-job cap is 32, shared, and array tasks count individually. The chain is
sized to respect it: stage 1 (18) + this chain job (1) + the TCGA array (11) = 30 at wave one;
stage 2 (18) + stage 3 (1) = 19 once stage 1 has drained; stages 4 (12) + 5 (12) = 24 after that.
"""
import argparse
import json
import os
import pathlib
import subprocess

import yaml

from g2l.run_phase10 import ARMS, widths

ROWS = pathlib.Path(os.environ.get("G2L_ROWS", "results/phase10/molhiv/rows"))


def sbatch(args: list[str], dry: bool) -> str:
    if dry:
        print("DRY:", " ".join(args))
        return "DRY"
    return subprocess.check_output(args, text=True).strip().split(";")[0]


def select(rows: pathlib.Path, cfg: dict) -> dict:
    """Per arm: the width with the highest validation mean. Every row must be present and complete."""
    by_arm, means = {}, {}
    for arm in ARMS:
        by_arm[arm] = {}
        for d in widths(arm, cfg):
            f = rows / f"sel_{arm}_d{d}.json"
            if not f.exists():
                raise SystemExit(f"selection row missing: {f} -- a select task failed; not submitting")
            r = json.loads(f.read_text())
            if len(r.get("val_rocauc", [])) != len(r["seeds"]):
                raise SystemExit(f"selection row incomplete: {f}")
            by_arm[arm][d] = r["val_mean"]
        means[arm] = by_arm[arm]
    return {"width": {a: max(means[a], key=means[a].get) for a in ARMS}, "val_by_width": means,
            "rule": "per arm: width = argmax validation mean over select_seeds (Phase 8H's rule)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["chain", "select"])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--config", default="configs/phase10_molhiv.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    export = f"ALL,G2L_COMMIT={args.commit}"

    if args.step == "chain":
        # stage 1 is already complete (this job depends on it); submit select, then the selector
        a = sbatch(["sbatch", "--parsable", "--array=0-17", f"--export={export}",
                    "slurm/phase10m_select.sbatch"], args.dry_run)
        print(f"SELECT_JOB={a}", flush=True)
        b = sbatch(["sbatch", "--parsable", f"--dependency=afterok:{a}", f"--export={export}",
                    "slurm/phase10m_launch.sbatch"], args.dry_run)
        print(f"LAUNCH_JOB={b}", flush=True)
        return

    w = select(ROWS, cfg)
    ROWS.mkdir(parents=True, exist_ok=True)
    (ROWS / "winners.json").write_text(json.dumps(w, indent=1))
    print("selection:", json.dumps(w["width"]), flush=True)
    a = sbatch(["sbatch", "--parsable", "--array=0-11", f"--export={export}",
                "slurm/phase10m_pretrain_rest.sbatch"], args.dry_run)
    print(f"PRETRAIN_REST_JOB={a}", flush=True)
    b = sbatch(["sbatch", "--parsable", "--array=0-11", f"--dependency=afterok:{a}",
                f"--export={export}", "slurm/phase10m_final.sbatch"], args.dry_run)
    print(f"FINAL_JOB={b}", flush=True)
    (ROWS / "winners.json").write_text(json.dumps({**w, "pretrain_rest_job": a, "final_job": b},
                                                  indent=1))


if __name__ == "__main__":
    main()
