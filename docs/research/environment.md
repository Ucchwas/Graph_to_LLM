# Environment & Compute

Two environments with a hard division of labour. Decided by the user 2026-08-25:
**anything beyond seconds-long tests runs on Marlowe.** The laptop is for development only.

> History: an earlier version of this file planned training on the laptop's 12 GB GPU and
> concluded subgraph sampling was mandatory. That premise was withdrawn the same day; the
> memory math below is redone for the H100.

---

## 1. Local laptop — development, unit tests, tiny sanity runs

| Item | Value |
|---|---|
| GPU | RTX 5070 Ti Laptop, 12 GB, Blackwell sm_120, driver 591.84 / CUDA 13.1, 80 W cap |
| OS | Windows 11 Home, PowerShell + Git Bash; **WSL 2 Ubuntu** installed for SSH multiplexing |
| Python | 3.14.7 (only interpreter present) |
| Stack | `torch 2.13.0+cu130` · `transformers 5.15.1` · `torch-geometric 2.8.0.post1` · `peft 0.20.0` — pinned in `requirements-local.txt` |
| Verified | CUDA initialises on sm_120, bf16 matmul OK, all 21 tests green, Cora downloads |

Use for: `pytest`, gate walkthroughs (full-graph Cora **inference** fits in `no_grad`), N≤512
shape/overfit smoke tests. Not for sweeps or multi-minute training, even when they would fit.

### Hazard that applies everywhere: transformers is v5
Attention dispatch and masking utilities were reworked across the v4→v5 boundary. Every
injection recipe from a 2024–2025 repo (GTLM included) is re-verified against the installed
v5 source, and `transformers==5.15.1` is pinned on both machines. The day-one
"4-D mask changes logits" canary guards this at every training start.

---

## 2. Marlowe — all training (primary)

Facts from docs/hpc/marlowe.md (desk research; items marked *verify* are checked by
`scripts/marlowe_verify.sh` at first login).

| Item | Value |
|---|---|
| Login | `ssh uutsha@login.marlowe.stanford.edu` — password + Duo per connection; multiplexed via WSL (`Host marlowe`, ControlPersist 8h) |
| Account / partition | `marlowe-m000211-pm06` / `batch` (Medium Project: ≤16 nodes, 2-day walltime, not preemptible) |
| Hardware | DGX H100 SuperPOD: 248× H100 80 GB SXM5, 8/node, 112 cores + 2 TB RAM/node |
| Persistent | `/projects/m000211` — code (`Graph_to_LLM/`), venv (`envs/g2l/`), datasets (`data/pyg/`) |
| Scratch | `/scratch/m000211/uutsha` — HF cache, Triton/Inductor caches, logs, runs. Not backed up; purge policy *verify* |
| `$HOME` | 32 GB — dotfiles only |
| Software | Lmod; `module load slurm mps` in every GPU job; no python/pytorch module → venv + pip; Apptainer available |
| Internet on compute nodes | *verify*; regardless, models pre-staged on login node, jobs run `HF_HUB_OFFLINE=1` |
| Budget | ~10,000 GPU-h per 12-week cycle; $0.30/GPU-h from 2026-09-01 |

Policy: no training, compile warm-ups or preprocessing loops on login nodes; never create an
unauthenticated path in (no remote-control/tunnel tools); one GPU per job, many jobs.

### Memory math on an H100 80 GB (Llama-3.2-1B: 16 layers, 32 heads, d=2048)
Eager attention materialises `[B, 32, N, N]`; per layer ≈ 12·B·H·N² bytes worst case.

| N | per layer | all 16 layers | verdict |
|---|---|---|---|
| 512 | 0.1 GB | 1.6 GB | trivial |
| 1024 | 0.4 GB | 6.4 GB | easy |
| **2708 (full Cora)** | **2.8 GB** | **45 GB** | fits with gradient checkpointing (~8 GB) or FlexAttention |
| PubMed 19,717 | 150 GB | — | FlexAttention only |

FlexAttention (`score_mod` gathers the bias in-kernel, GTLM reports eager 30 GB → 0.2 GB) is
available on Marlowe (Linux + Triton). Eager stays the numerical-parity oracle; flex becomes
load-bearing at Phase 6 scale. Headline Cora numbers come from full-graph training.

---

## 3. Sherlock — secondary

`ssh uutsha@login.sherlock.stanford.edu`; group `tauhid`; `$SCRATCH=/scratch/users/uutsha`
(**90-day content-based purge**); GPU jobs need `-p gpu -G 1 -C "GPU_SKU:H100_SXM5|GPU_SKU:L40S"`
(the public `gpu` partition is heterogeneous and mostly pre-Ampere); no `--account`; `sh_dev`
for interactive. Full differences list: docs/hpc/sherlock.md §9.

---

## 4. Dev/prod split — the standing risk

Code is written on Windows/py3.14/torch 2.13 and run on Linux/cluster Python/whatever torch
pip resolves there. Mitigations: `requirements-local.txt` and `requirements-cluster.txt`
pinned separately with `transformers==5.15.1` in both; `.gitattributes` forces LF on all
scripts (CRLF breaks bash on the cluster); data roots via `PYG_DATA_ROOT`; every gate
walkthrough re-runs on the laptop against cluster checkpoints; the first Marlowe job
reproduces a locally-known number before any new experiment runs.
