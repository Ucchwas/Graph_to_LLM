# Marlowe — VERIFIED on the machine, 2026-08-25

First real login (`uutsha@login-03`). Supersedes the desk-research guesses in `marlowe.md`
wherever they conflict. Everything below was observed, not inferred.

## Access — the paths in the plan were WRONG

| Path | Reality |
|---|---|
| `/projects/m000211` | **DENIED.** Owned by group `marlowe-m000211` (gid 851743857) = {sakib, junmings, ridvan, tauhid}. We are not in it. |
| `/projects/m000211-pm06` | **Does not exist.** |
| **`/scratch/m000211-pm06`** | ✅ **writable** — group `marlowe-m000211-pm06` (gid 851763112) = {sakib, junmings, ridvan, atik, uutsha, armeen, tauhid}. Teammates `atik`/`sakib` already have dirs here. |
| `/scratch/m000211` | DENIED (same group mismatch as /projects). |
| `$HOME=/users/uutsha` | ✅ writable, 693 MB used. |

**Cause:** our account is provisioned into the *cycle* group `…-pm06`, but the `/projects`
space is owned by the *base* group. Other projects have `-pm06` scratch dirs (m000051-pm06,
m000060-pm06, …) but **none has a `-pm06` projects dir**, so `/projects` access requires being
added to the base group. **Open request to SRCC — see below.**

**Working layout adopted (deviates from PLAN.md §Repo layout):**
```
$HOME/Graph_to_LLM          code (rsync'd from the laptop; private repo, no creds on cluster)
$HOME/envs/g2l              venv (NFS handles many small files better than Lustre)
/scratch/m000211-pm06/uutsha/{hf,pyg,logs,runs,uvcache,triton,inductor}
```

## Slurm — confirmed, with two corrections

- Account **`marlowe-m000211-pm06`**, QOS **`medium`** → partition **`batch`**.
- `batch`: MaxNodes=16, **MaxTime=2-00:00:00**, AllowQos=medium ✅ (as documented)
- `hero`: MaxNodes=31, MaxTime=30 days, AllowQos=**large** — not us.
- `preempt`: **MaxTime=12:00:00** (marlowe.md said 4 h — *wrong*) and
  **`DenyQos=class,medium`** → **we cannot use `preempt` at all.** This settles the
  "free preempt cycles via the bare account" question: not available to us.
- `MaxArraySize=1001`, `MaxJobCount=10000` → job arrays up to 1000 tasks. (Was [UNKNOWN].)
- Slurm 25.05.2. **Not on PATH by default — every script needs `module load slurm`.**
- Queue at first login: 134 jobs in `batch`, 57 pending. Expect real waits.

## Hardware — confirmed
31 nodes; n13 has 7 GPUs, the other 30 have 8 → **247 GPUs**. Per node: 112 cores
(2×56), **RealMemory 1,950,000 MB (~1.9 TB)**. **`TmpDisk=0` — there is no node-local
scratch**, so Triton/Inductor caches must live on `/scratch` (marlowe.md's "try node-local
/tmp" suggestion is not available).

Per-GPU fair share of a node: 14 cores, ~243 GB RAM.

## Storage capacity
- `/projects/m000211`: 1022 G total, 500 G used (49%) — *visible via df only before the mount denied us*
- `/scratch` (Lustre): **10 T total, 8.7 T used, 1.4 T free (87% full)** — watch this.
- `/users` (NFS): 3.2 P filesystem; no per-user quota command available (`quota` not installed).

## Software — the environment story
- **No `python3-venv`**: `python3 -m venv` fails with *"ensurepip is not available"*.
  `--without-pip` works, but the clean fix is **`uv`** (self-contained, no ensurepip):
  `curl -LsSf https://astral.sh/uv/install.sh | sh` → `~/.local/bin/uv` (0.12.5).
- `module load conda` / `conda/24.3.0-0` **does not put conda on PATH** — do not rely on it.
- System Python is **3.10.12** (laptop runs 3.14.7 — pin per platform; our code is 3.10-compatible).
- Modules present: `slurm`, `gcc/13.1.0`, `python3`, `python39`, `conda/24.3.0-0`, plus
  `/cm/shared/modulefiles`. **No pytorch module.**
- `nvidia-smi` is absent on login nodes (no GPUs there) — driver version must be read from a
  compute node inside a job.

## Network
Login node has **outbound internet**: pypi.org, huggingface.co, github.com all return HTTP 200.
Compute-node egress still unverified → keep pre-staging models and running jobs with
`HF_HUB_OFFLINE=1`.

## Connectivity that actually works from Windows
- ❌ VS Code Remote-SSH — opens several connections, each triggering a separate Duo prompt.
- ❌ Git Bash / MSYS `ControlMaster` — dies at `mux_client_request_session: read from master
  failed`; MSYS's emulated Unix sockets cannot pass file descriptors.
- ❌ PowerShell `ssh.exe` — `getsockname failed: Not a socket` (no ControlMaster support).
- ✅ **WSL 2 Ubuntu**: `wsl -d Ubuntu -u root ssh -fNM marlowe` → one password+Duo, then
  8 h of prompt-free commands. Note `wsl.exe` expands `$HOME` in *its own* root context —
  use literal remote paths or pipe scripts over stdin.
- Private-repo clone from the cluster needs credentials we will not store → **sync with
  `rsync -az -e ssh ./ marlowe:~/Graph_to_LLM/`** over the authenticated connection instead.

## Open request for SRCC / the PI
> Our SUNet `uutsha` is in `marlowe-m000211-pm06` but `/projects/m000211` is owned by
> `marlowe-m000211`, and `/projects/m000211-pm06` does not exist. We can write to
> `/scratch/m000211-pm06` but have no persistent project space. Could `uutsha` be added to
> the `marlowe-m000211` group, or a `/projects/m000211-pm06` directory be created?

Until then: code and venv live in `$HOME`, everything large in `/scratch/m000211-pm06`, and
results are copied back to `$HOME`/git rather than left on scratch.
