# Handoff — for a Claude Code session running on Marlowe

Read this first, then `CLAUDE.md` (spec), `docs/PLAN.md` (the approved phased plan — its
decisions are locked), and `docs/research/00-KNOWLEDGE-BASE.md` §1 (where the research
overturned the spec). Deep references: `docs/research/*.md`, `docs/hpc/marlowe.md`.

## Where the project stands (2026-08-25)
- **Phase 0 complete** (protocol foundation): `g2l/data.py`, `g2l/metrics.py`, `g2l/bias.py`,
  26 tests green on the laptop. Gate signed off by the user via `python -m g2l.walkthrough --phase 0`.
- **Phase 1 code complete, results pending on Marlowe**: `baselines/` (heuristics, feature-only,
  GAE/VGAE/GAT, scratch adjacency-row transformer), `baselines/run_phase1.py`,
  `baselines/run_scratch_sweep.py`, `baselines/aggregate.py`.
  A laptop run of the 45-row baseline table exists off-repo as a reference; **Marlowe is the
  source of truth** — regenerate everything here. Known reference values from the laptop run:
  gae_600ep ≈ 0.905 AUC / 0.911 AP (target 90.6±0.9 / 91.2±1.0); PPR ≈ 0.85 / 0.90;
  GAT (dropout 0.2, lr 0.01) ≈ 0.90 / 0.91; feature-only ≈ 0.62 / 0.64.

## First session on Marlowe, in order
1. `bash scripts/marlowe_verify.sh` — read every line; it resolves the unknowns in
   docs/hpc/marlowe.md §10 (partition limits, quotas, array caps, driver version, internet).
2. Pick the torch CUDA index from the driver version, then `bash scripts/marlowe_setup.sh`
   (`TORCH_INDEX=...` if not cu126). Venv goes on `/projects/m000211/envs/g2l`; caches on
   `/scratch/m000211/$USER`; nothing large in `$HOME` (32 GB).
3. `pytest tests/` on the login node (seconds; allowed).
4. Write `requirements-cluster.txt` from `pip freeze` (keep `transformers==5.15.1`).
5. Smoke job: `srun -N1 -n1 -G1 -c8 --mem=64G -t 00:20:00 -A marlowe-m000211-pm06 -p batch --pty bash -l`
   then inside: `module load slurm mps`, activate venv, `python -c "import torch;print(torch.cuda.get_device_name(0))"`.
6. `sbatch slurm/phase1_scratch_sweep.sbatch`; also run `python -m baselines.run_phase1` as a
   job (write `slurm/phase1_baselines.sbatch` by copying the sweep script; 30 min, 1 GPU).
7. Monitor: `squeue -u $USER`, `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS`, logs in
   `/scratch/m000211/$USER/logs/`. Fix and resubmit on failure; do not leave stray jobs.
8. `python -m baselines.aggregate` → write `results/phase1/RESULTS.md` (format: results/phase0/RESULTS.md)
   with the full table, the gate checklist, "shows / does not show", and a VALIDATED line.
   Gate 1 passes when gae_600ep is within ±1 of 90.6/91.2 and the scratch capacity curve exists.

## Rules that are not in CLAUDE.md
- Commits: author is the user (`git config user.name Ucchwas`, `user.email ucchwas09@gmail.com`);
  **never add a Claude co-author trailer**; **the user pushes**, not Claude.
- Never store, print, or script passwords/tokens. The HF token (needed from Phase 2 for
  gated Llama-3.2-1B) lives only in the user's env or `huggingface-cli login`.
- Login nodes: edit, test, `pip install`, `huggingface-cli download`, `sbatch` only. No training,
  no `torch.compile` warm-ups, no preprocessing loops. One GPU per job; many jobs; job arrays.
- No VS Code tunnels / Claude remote-control features — Remote-SSH only (Marlowe policy).
- Every table row goes through `g2l.metrics.evaluate_edge_split` on the same split; identity
  and random rows are recomputed whenever the loader changes.
- Clean minimal code, few comments, no unnecessary machinery. JSONL logging, no W&B.
