# Handoff — for a Claude Code session working with Marlowe

Read this first, then `CLAUDE.md` (spec), `docs/PLAN.md` (the approved phased plan — its
decisions are locked), and `docs/research/00-KNOWLEDGE-BASE.md` §1 (where the research
overturned the spec). Deep references: `docs/research/*.md`. Cluster facts as observed:
`docs/hpc/marlowe-verified.md` (supersedes the desk-research `docs/hpc/marlowe.md`).

## Where the project stands (2026-08-25)
- **Phase 0 complete** (protocol foundation): `g2l/data.py`, `g2l/metrics.py`, `g2l/bias.py`;
  22 tests green on both machines. Gate 0 signed via `python -m g2l.walkthrough --phase 0`.
- **Phase 1 complete, Gate 1 awaiting the user's sign-off**: `baselines/` (identity/random
  controls, heuristics, feature-only, GAE/VGAE/GAT, scratch adjacency-row transformer).
  Results produced on Marlowe (jobs 447748/447749) in `results/phase1/{table,scratch_sweep}.jsonl`,
  write-up in `results/phase1/RESULTS.md`, live check `python -m g2l.walkthrough --phase 1`.
  The first scratch sweep (old recipe) stalled at a degree-only solution in 22/25 seeds; its
  numbers are recorded in RESULTS.md and the file was not kept.
- **Marlowe is set up**: `$HOME/Graph_to_LLM` (rsync'd), `$HOME/envs/g2l` (uv venv, Python
  3.10.12, torch 2.13.0+cu126, pinned in `requirements-cluster.txt`), data/caches/logs under
  `/scratch/m000211-pm06/$USER`. Phase 2 needs the user's HF token to pre-stage Llama-3.2-1B.

## Working with Marlowe from the laptop
- Connect once: `wsl -d Ubuntu -u root ssh -fNM marlowe` (the user types password + Duo;
  the master lives 8 h). Then `wsl -d Ubuntu -u root ssh marlowe '...'` needs no prompt.
  Non-interactive shells lack `module`: wrap Slurm commands as `bash -lc "module load slurm; ..."`.
- Sync code (from WSL, inside the repo; no `git clone` on the cluster — private repo, no credentials):
  `rsync -az --exclude .venv --exclude data/pyg --exclude __pycache__ --exclude slides -e ssh ./ marlowe:/users/uutsha/Graph_to_LLM/`
- Jobs: `sbatch slurm/<name>.sbatch` (1 GPU, 8 CPUs, 64 GB, partition `batch`);
  `squeue -u $USER`; `sacct -j <id> -X --format=JobID,State,Elapsed,ExitCode`; logs in
  `/scratch/m000211-pm06/$USER/logs/`. Runners are idempotent — rows already in the JSONL are
  skipped, so rename or remove a result file before re-running an experiment.
- Results come back by `scp` into `results/` and are committed; gate walkthroughs run on the laptop.
- Fresh environment: `bash scripts/marlowe_verify.sh`, then `bash scripts/marlowe_setup.sh`.

## Rules that are not in CLAUDE.md
- Commits: author is the user (`git config user.name Ucchwas`, `user.email ucchwas09@gmail.com`);
  **never add a Claude co-author trailer**; **the user pushes**, not Claude.
- Never store, print, or script passwords/tokens. The HF token (needed from Phase 2 for
  gated Llama-3.2-1B) lives only in the user's env or `hf auth login` (the `huggingface-cli`
  command is dead in huggingface-hub 1.x).
- The user validates every gate; ask before writing code for the next phase or submitting
  jobs outside the agreed phase tasks. Anything beyond seconds-long tests runs on Marlowe.
- Login nodes: edit, test, `uv pip install`, `hf download`, `sbatch` only. No
  training, no `torch.compile` warm-ups, no preprocessing loops. One GPU per job; many jobs.
- No VS Code tunnels / Claude remote-control features (Marlowe policy). VS Code Remote-SSH
  does not work with Duo either — use the multiplexed WSL connection above.
- Every table row goes through `g2l.metrics.evaluate_edge_split` on the same split; identity
  and random rows are part of every table (leak canary: identity must sit at chance).
- Clean minimal code, few comments, no unnecessary machinery. JSONL logging, no W&B.
