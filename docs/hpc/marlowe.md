# Stanford Marlowe — practical notes for the Graph-to-LLM project

**Status:** desk research only. Compiled 2026-08-25 from the official Marlowe documentation site
(`marlowe-research.stanford.edu`) plus clearly-labelled secondary Stanford sources.
**Nothing here has been verified on the machine.** No SSH connection was made.

Legend used throughout:

- **[OFFICIAL]** — stated on `marlowe-research.stanford.edu` (the authoritative docs).
- **[STANFORD-OTHER]** — another Stanford site (Sherlock docs, rcpedia, HAI/DS pages). Not Marlowe-authoritative.
- **[INFERRED]** — my reasoning from official examples, not stated outright.
- **[UNKNOWN]** — not in the docs at all; must be checked live at first login. See §10.

Our account facts (given by the user, consistent with the documented naming scheme):

| Thing | Value |
| --- | --- |
| SUNet ID | `uutsha` |
| Login host | `login.marlowe.stanford.edu` |
| Project ID | `m000211` |
| Slurm account | `marlowe-m000211-pm06` |
| Tier | Medium Project → **`batch`** partition |
| Persistent storage | `/projects/m000211` |
| Scratch (Lustre) | `/scratch/m000211` |
| Home | `/users/uutsha` — **32 GB only** |

---

## 1. Connecting

### 1.1 The command

**[OFFICIAL]** The docs give exactly one form:

```bash
ssh uutsha@login.marlowe.stanford.edu
```

**[OFFICIAL]** "You will be prompted to provide both your password and duo two-factor authentication
credentials before it logs you in." So: **SUNet password + Duo, on every new TCP connection.**

**[OFFICIAL]** If you see `Connection closed by 171.67.99.202 port 22`, the account is not fully
activated yet — wait for the activation email, then ask in `#marlowe-researchers`.

**[INFERRED]** There is more than one login node: the official NGC example shows a prompt of
`you@login-02 $`. So `login.marlowe.stanford.edu` is almost certainly a load-balanced alias over
`login-01`, `login-02`, … (same pattern as Sherlock). Consequence: a `tmux` session lives on one
specific login node and you may land on a different one next time.

### 1.2 SSH public keys — do they work? Do they skip Duo?

**[UNKNOWN] / probably not.** The Marlowe docs say nothing about `~/.ssh/authorized_keys`.

**[STANFORD-OTHER]** On Sherlock — same operator (Stanford Research Computing), same
password+Duo+Kerberos design — the docs are explicit: *"SSH public-key authentication is not
supported on Sherlock."* Sherlock instead offers **GSSAPI/Kerberos**, which removes the *password*
prompt but you *"should directly see the two-factor (Duo) prompt"* — i.e. **Kerberos does not skip
Duo either.**

Treat Marlowe the same until proven otherwise: **assume no key auth, and assume nothing removes Duo.**
Test on first login (§10) but do not build tooling that depends on key auth.

**Therefore the only real lever we have against Duo-per-command is connection multiplexing.**

### 1.3 Connection multiplexing (ControlMaster) — the important bit

**[UNKNOWN for Marlowe]** — the Marlowe docs do not mention it. **[STANFORD-OTHER]** Sherlock
documents it as *the* way to avoid repeated Duo prompts, and it is a pure client-side OpenSSH
feature, so it works against any normal `sshd`. Very likely to work on Marlowe.

Client config (`~/.ssh/config` **on your machine**):

```
Host marlowe
    HostName login.marlowe.stanford.edu
    User uutsha
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 5
    TCPKeepAlive yes
```

Then: **one** `ssh marlowe` → one password + one Duo push. Every later `ssh marlowe <cmd>`,
`scp`, `rsync`, or `git` over that host alias reuses the socket with **no prompt** for 8 hours after
the last session closes.

Useful control commands:

```bash
ssh -O check marlowe     # is the master alive?
ssh -O exit  marlowe     # tear the master down deliberately
ssh -fN marlowe          # open a background master with no shell (one Duo push, then done)
```

> **CRITICAL WINDOWS FOOTGUN.** **[STANFORD-OTHER / upstream]** Microsoft's native Win32 OpenSSH
> (`C:\Windows\System32\OpenSSH\ssh.exe`, i.e. what you get in PowerShell) **does not implement
> ControlMaster.** It relies on Unix-domain sockets; on Windows the options are silently accepted but
> no master is created, or you get `getsockname failed: Not a socket`. This is a long-standing
> unresolved issue (Win32-OpenSSH #405, #1328).
>
> Since you are on Windows 11, the practical options are, best first:
>
> 1. **Use WSL2 (Ubuntu) as the SSH client.** ControlMaster works normally there. Put the config
>    above in `~/.ssh/config` *inside* WSL. This is the recommended setup — do this.
> 2. **One long-lived interactive session inside `tmux` on Marlowe.** One Duo push per work
>    session; do everything (edit, submit, monitor) inside that tmux. Note tmux is pinned to one
>    login node (§1.1).
> 3. **Open OnDemand** (`https://ood.marlowe.stanford.edu`) — browser, WebAuth/Duo once per browser
>    session, gives Jupyter Lab and Code Server. Zero SSH.
> 4. MobaXterm / PuTTY session reuse — works but is fiddly and not documented by Stanford.
>
> Do **not** try to script "one `ssh` per command" from PowerShell: that is one Duo push per command.

### 1.4 VPN, bastions, jump hosts

**[UNKNOWN]** No VPN requirement, bastion, or jump host is mentioned anywhere in the Marlowe docs.
**[STANFORD-OTHER]** Sherlock states access is not restricted to campus and recommends (but does not
require) the Stanford VPN from untrusted networks. **[INFERRED]** `login.marlowe.stanford.edu`
resolves to a public Stanford IP (`171.67.99.202` appears in the docs' own error message), so direct
off-campus SSH is expected to work. Verify at first login.

### 1.5 Windows guidance in the docs

**[OFFICIAL]** None. There is no Windows/PowerShell/PuTTY page anywhere on the Marlowe site.
Everything above about Windows is [STANFORD-OTHER] or [INFERRED].

---

## 2. The machine

**[OFFICIAL]** Marlowe is a **1 SU (Scalable Unit) NVIDIA DGX H100 SuperPOD**, operated by Stanford
Research Computing with Stanford HAI Research Data Science. 11.1 PFlop/s HPL.

| Property | Value | Source |
| --- | --- | --- |
| Compute nodes | **31** NVIDIA DGX H100 servers | [OFFICIAL] specs + about |
| GPUs total | **248 × NVIDIA H100 80 GB SXM5** | [OFFICIAL] about page |
| GPUs per node | **8** | [OFFICIAL] |
| GPU memory | **80 GB HBM** per GPU | [OFFICIAL] |
| CPU | 2 × Intel Xeon Platinum 8480C — **112 cores/node** | [OFFICIAL] |
| Host RAM | **2 TB per node** | [OFFICIAL] |
| Intra-node GPU fabric | **NVLink + NVSwitch** | [OFFICIAL] about page |
| Inter-node fabric | **NVIDIA InfiniBand NDR, 400 Gb/s** (compute) | [OFFICIAL] |
| Storage fabric | separate 400 Gb IB network to DDN ExaScaler Lustre | [OFFICIAL] |
| Home/project fabric | 100 Gb Ethernet to DDN IntelliFlash | [OFFICIAL] |
| External network | 100 Gb Ethernet to SUNet / the outside world | [OFFICIAL] |
| Data risk classification | **Low and Moderate Risk** (not High Risk / PHI) | [OFFICIAL] about page |
| Node-local NVMe | **[UNKNOWN]** — not documented anywhere | — |
| Login node specs / count | **[UNKNOWN]** — not documented | — |

Per-GPU fair share if you take 1 of 8 GPUs on a node: ~14 CPU cores, ~256 GB RAM. Use that as the
mental model when sizing `-c` and `--mem`.

**Utilisation reality check. [OFFICIAL]** The Marlowe landing page reports the machine at
**~90% allocated (30-day average)**, 180+ research groups, 2.9M GPU-hours delivered. Expect real
queue waits. Design our sweeps as **many small independent jobs / job arrays**, which schedule far
more easily than large reservations — which happens to be exactly our workload shape.

**Node naming. [OFFICIAL]** Compute nodes are `n01`, `n02`, …; `n12` appears in an FAQ error message.

---

## 3. Slurm

### 3.1 Accounts

**[OFFICIAL]** Accounts are `marlowe-<project ID>`. Medium and large projects get an extra suffix
(`-pmNN` / `-plNN`) which is **required** to use `batch` / `hero`. Without a valid account:

```
srun: error: ACCOUNT ERROR: Did you remember to set your account?
```

So for us: `-A marlowe-m000211-pm06`.

> **[OFFICIAL] Billing trap:** *"You will be charged against your GPU hours allocation if you submit
> a job with a medium/large project suffix to the `preempt` partition."*
> If we ever want free/uncharged preempt cycles, submit with the **bare** account
> `-A marlowe-m000211` (no `-pm06`). **[UNKNOWN]** whether the bare account is actually still valid
> for a Medium project — verify with `sacctmgr show assoc user=uutsha`.

### 3.2 Partitions

**[OFFICIAL]** Three partitions, mapped one-to-one onto the three access tiers:

| Partition | Tier | Node limit | Max walltime | Preemptible? |
| --- | --- | --- | --- | --- |
| `preempt` | Basic Access | no node limit | **4 hours** | **Yes** — killed within 15 min if `batch`/`hero` wants the node |
| `batch` | **Medium Project ← us** | 16 nodes | **2 days** | No |
| `hero` | Large Project | 31 nodes | 30 days | No |

**[OFFICIAL]** *"If you have a medium project allocation, you should submit to the `batch`
partition."* → **we use `-p batch`.** Non-preemptible scheduling is exactly what a Medium Project is
buying, and 2 days is far more than any of our jobs need.

**[OFFICIAL]** Preemption rule: *"Any jobs in the `preempt` queue can be preempted within 15 minutes
if a job in a higher priority partition (`batch` or `hero`) requests the node that the `preempt` job
is running on."*

> **These limits have changed over time.** An archived Feb-2026 version of the same page said
> `hero` 25 nodes / 24 h and `preempt` 8 nodes / **12 h**. The table above is the current page.
> Re-check with `scontrol show partition` on first login rather than trusting this file.

### 3.3 QOS, fairshare, per-user limits, priority

**[UNKNOWN].** The docs say **nothing** about QOS names, fairshare weights, `MaxJobs`,
`MaxSubmitJobs`, `MaxGRES` per user or per account, array size caps, or how priority is computed.
This matters directly for our sweep-of-many-small-jobs plan. Check live (§10).

### 3.4 GPU request syntax

Both forms appear in **[OFFICIAL]** examples:

- `-G N` (= `--gpus=N`) — used in the main Slurm page's `srun`, `salloc` and `sbatch` examples.
  **This is the house style; prefer it.**
- `--gres=gpu:8` — used in the official NGC NIM `srun` example.
- `--gpus-per-node=N` — **[UNKNOWN]**, never appears anywhere in the docs. It is standard Slurm and
  should work, but there is no reason for us to use it on a single-node job. Stick to `-G`.

`-N` = nodes, `-n` = tasks, `-c` = cpus-per-task. **[STANFORD-OTHER]** Stanford's rcpedia
Marlowe example uses `-G 1 -n 1 -c 4`, confirming `-n`/`-c` behave normally.

### 3.5 Required flags

**[OFFICIAL]** Strictly required: an **account** (`-A`) and, implicitly, a **partition** (`-p`) since
account and partition must match your tier. `--time` is in every example and you should always set
it. `module load slurm` appears inside every official job script (it is loaded by default at login,
but the official templates re-load it inside the job — harmless, and I keep it below for fidelity).

### 3.6 Checking our allocation burn

**[OFFICIAL]**

```bash
sreport cluster UserUtilizationByAccount -T gres/gpu \
    Start=2026-08-01 End=now account=marlowe-m000211-pm06 -t hours
```

**[OFFICIAL]** Recharge rates (Medium/Large, non-preemptible): **$0.25/GPU-hour** and
**$0.010/CPU-hour** through 2026-08-31; **$0.30/GPU-hour** from 2026-09-01. `/projects` storage
$20/TB/month. Medium Project = **up to 10,000 GPU-hours per ~12-week cycle**. At 10k GPU-hours, a
1-GPU 4-hour experiment costs 4 GPU-hours — we can afford ~2,000 such runs per cycle. Budget is not
our binding constraint; queue time and our own throughput are.

### 3.7 Useful queue commands

**[OFFICIAL]**

```bash
squeue -j <jobid> --start        # upper bound on start time
scontrol show jobid=<jobid>      # full detail on why it is pending
```

General Slurm (safe to assume):

```bash
squeue -u uutsha
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES,AllocTRES
scontrol show partition          # <-- run this first login, ground-truth the limits
sinfo -o "%P %.6D %.14F %G"      # partitions, node states, GRES
sacctmgr show assoc user=uutsha format=Account,Partition,QOS,MaxJobs,GrpTRES
```

---

## 4. A complete, copy-pasteable single-GPU sbatch template for our workload

Save as `/projects/m000211/graph2llm/slurm/train_1gpu.sbatch`.

```bash
#!/bin/bash
#SBATCH --job-name=g2l-cora
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --chdir=/projects/m000211/graph2llm
#SBATCH --output=/scratch/m000211/uutsha/logs/%x-%j.out
#SBATCH --error=/scratch/m000211/uutsha/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=uutsha@stanford.edu

set -euo pipefail

# ---------------------------------------------------------------- modules
module purge
module load slurm
module load mps                 # OFFICIAL: prevents CUDA_ERROR_MPS_CONNECTION_FAILED
# module load nvhpc             # ONLY if you need nvcc. See the warning in section 7.2.
# module load cudnn/cuda12/9.3.0.75

# If you DO load nvhpc, undo its CC/CXX hijack so torch.compile / Triton
# can still use gcc for its C++ codegen:
#   export CC=gcc CXX=g++

# ------------------------------------------------- caches: NEVER in $HOME
# $HOME is 32 GB. Every one of these will blow it if left at its default.
PROJ=/projects/m000211
SCR=/scratch/m000211/$USER

export HF_HOME="$SCR/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCH_HOME="$SCR/torch"
export TRITON_CACHE_DIR="$SCR/triton"              # OFFICIAL doc warns about ~/.triton
export TORCHINDUCTOR_CACHE_DIR="$SCR/inductor"     # torch.compile artefacts (FlexAttention)
export XDG_CACHE_HOME="$SCR/xdgcache"
export PYG_DATA_ROOT="$PROJ/data/pyg"              # our convention; these datasets are tiny
export WANDB_DIR="$SCR/wandb"
export WANDB_CACHE_DIR="$SCR/wandb-cache"
export TOKENIZERS_PARALLELISM=false

# Models were pre-fetched on a login node -> run offline so a network blip
# can never kill a queued experiment. See section 7.6.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$XDG_CACHE_HOME" "$PYG_DATA_ROOT" "$WANDB_DIR" "$SCR/logs"

# ------------------------------ threads: 112 cores/node, we asked for 16.
# Do not let OpenMP/MKL see all 112 and spawn 112 threads.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# ------------------------------------------------------------- environment
source "$PROJ/envs/g2l/bin/activate"        # venv on /projects  (see section 6)
# or:  module load conda && conda activate /projects/m000211/envs/g2l

# ------------------------------------------------------------- provenance
echo "job      : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "partition: $SLURM_JOB_PARTITION   account: $SLURM_JOB_ACCOUNT"
echo "gpus     : ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'dev',torch.cuda.get_device_name(0))"
git -C "$PROJ/graph2llm" rev-parse --short HEAD || true

# ------------------------------------------------------------------- run
srun python -u -m g2l.train \
    --config configs/cora_e1.yaml \
    --attn-impl flex \
    --out "$SCR/runs/$SLURM_JOB_ID"
```

Submit / watch / kill:

```bash
sbatch /projects/m000211/graph2llm/slurm/train_1gpu.sbatch
squeue -u uutsha
tail -f /scratch/m000211/uutsha/logs/g2l-cora-<jobid>.out
scancel <jobid>
```

### 4.1 Job arrays — the right shape for our E1–E7 / decoder / ablation sweeps

```bash
#SBATCH --job-name=g2l-e-sweep
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --array=0-20%6          # 21 configs, at most 6 running at once
#SBATCH --output=/scratch/m000211/uutsha/logs/%x-%A_%a.out
#SBATCH --error=/scratch/m000211/uutsha/logs/%x-%A_%a.err
# ... same body as above, then:
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs/sweep_list.txt)
srun python -u -m g2l.train --config "$CFG" --seed "$SLURM_ARRAY_TASK_ID"
```

**[UNKNOWN]** the site's `MaxArraySize` and any per-user running-job cap. The `%6` throttle is a
courtesy regardless — keep it modest so we do not monopolise `batch` (a 16-node partition shared with
every other Medium Project on the machine). Check with
`scontrol show config | grep -i maxarray`.

### 4.2 Notes specific to *our* memory profile

- Eager path materialises `[B, heads, N, N]` scores. At N=2708, 32 heads, bf16: ~469 MB/tensor,
  several GB live across layers. **Fits in 80 GB comfortably** for a Llama-3.2-1B; an 8B model with
  32 layers on the eager path will be much tighter — that is precisely the case for FlexAttention.
- FlexAttention needs `torch.compile` + Triton, which needs a **writable Triton and Inductor cache**
  on a filesystem that is not the 32 GB `$HOME`. Both are set above, on Lustre.
  **[INFERRED]** If compilation turns out to be slow or flaky on Lustre (many tiny files, and Lustre
  dislikes metadata churn), try pointing them at node-local `/tmp` instead — `/tmp` is definitely
  node-local, since the official MPS FAQ uses `/tmp/nvidia-mps-$SLURM_JOB_ID` — but then you lose
  cache reuse between jobs, which matters a lot across a 21-task array.
- Because our jobs are single-GPU and short, **never request `-N 2` or `-G 8`**. One GPU per job,
  many jobs. This also keeps GPU-hour burn honest: Slurm charges for what you allocate, not what you use.

---

## 5. Interactive / debug sessions

**[OFFICIAL]** There is **no dedicated debug or dev partition.** Use `srun --pty` or `salloc` on your
normal partition.

Official `srun` form (the docs' example uses `preempt`/`-G 4`; adapted to us):

```bash
# 1 GPU, 2 hours, on our Medium-Project partition:
srun -N 1 -n 1 -G 1 -c 16 --mem=192G -t 02:00:00 \
     -A marlowe-m000211-pm06 -p batch --pty bash -l
```

**[STANFORD-OTHER]** rcpedia's Marlowe recipe is the same shape:
`srun -p preempt -A marlowe-<project> -G 1 -n 1 -t 2:00:00 --pty /bin/bash`

Official `salloc` form:

```bash
salloc -N 1 -G 1 -c 16 -t 02:00:00 -A marlowe-m000211-pm06 -p batch
# then:  srun --pty bash -l    (or just run srun <cmd> from inside the allocation)
```

**Smoke-test recipe for us** (cheap, gets a GPU fast, exercises the exact risky path):

```bash
srun -N1 -n1 -G1 -c8 -t 00:30:00 -A marlowe-m000211-pm06 -p batch --pty bash -l
module load slurm mps
source /projects/m000211/envs/g2l/bin/activate
export HF_HOME=/scratch/m000211/$USER/hf
export TRITON_CACHE_DIR=/scratch/m000211/$USER/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/m000211/$USER/inductor
python - <<'PY'
import torch
from torch.nn.attention.flex_attention import flex_attention
print(torch.__version__, torch.cuda.get_device_name(0))
q = k = v = torch.randn(1, 32, 2708, 64, device='cuda', dtype=torch.bfloat16)
f = torch.compile(flex_attention)
print(f(q, k, v).shape)
print(torch.cuda.max_memory_allocated() / 2**30, 'GiB')
PY
```

If you want a browser IDE instead: **[OFFICIAL]** Open OnDemand at
`https://ood.marlowe.stanford.edu` offers **Jupyter Lab** and **Code Server**.
**[OFFICIAL] Code Server is capped at 4 CPU cores, 12 GB RAM, 8 hours, and cannot request GPUs**
("This is to prevent users from sitting on resources without utilizing them"). So Code Server is for
*editing only*; use Jupyter Lab (which can be given GPUs) or a real `srun` for anything GPU-bound.

**[OFFICIAL]** Marlowe does **not allow SSH port forwarding on compute nodes** — OOD is the only
route to a browser app running on a compute node. Do not plan on `ssh -L` into a compute node.

---

## 6. Storage — where everything goes

**[OFFICIAL]** three user filesystems:

| Path | Backing | Quota | Backup | Scope | Purge |
| --- | --- | --- | --- | --- | --- |
| `/users/uutsha` (`$HOME`) | DDN IntelliFlash (NFS, 100 GbE) | **32 GB** | every 24 h, replicated | per **user** | none documented |
| `/projects/m000211` | DDN IntelliFlash (NFS, 100 GbE) | **varies by allocation** | every 24 h at 21:15, replicated | per **project**, shared by all members | none documented |
| `/scratch/m000211` | DDN ExaScaler **Lustre** (400 Gb IB) | **10 TB default** | **NOT backed up, NOT replicated** | per **project** | **[UNKNOWN]** — no purge policy is published |

**[OFFICIAL]** `/scratch` is keyed on **project ID, not username** — the docs give `/scratch/m231631`,
`/scratch/m402630`. Same for `/projects`. So **scratch is shared per project**; by convention (and per
**[STANFORD-OTHER]** rcpedia's Marlowe example, which uses `/scratch/<project>/$USER`) each member
makes their own subdirectory:

```bash
mkdir -p /scratch/m000211/$USER
```

**[OFFICIAL]** `/projects` uses **autofs** — your directory *will not appear* in `ls /projects` until
you touch it. This confuses scripts. Fix: `ls /projects/m000211` (or just `cd` there) early in every
login and every job. The same quirk bites in Globus (§8).

**[OFFICIAL]** *"Your home directory only has 32GB of storage available. For any large files or conda
installs, we recommend using a different filesystem."* Symptom of ignoring this:
`disk quota exceeded`.

### 6.1 Our layout

```
/projects/m000211/                         # persistent, backed up, no purge risk
├── graph2llm/                             # git checkout (code)
├── envs/g2l/                              # Python venv or conda --prefix env   <-- HERE
├── data/pyg/                              # PyG datasets (Cora/PubMed/OGB: MBs to low GBs)
├── models/                                # curated HF snapshots we want to keep forever
└── results/                               # final metrics, checkpoints worth keeping

/scratch/m000211/uutsha/                   # 10 TB, Lustre, fast, NOT backed up
├── hf/                                    # $HF_HOME  (raw HF cache, regenerable)
├── triton/  inductor/  torch/  xdgcache/  # compile + framework caches
├── logs/                                  # slurm .out/.err
├── runs/                                  # per-job scratch, tensorboard, intermediate ckpts
└── apptainer_cache/                       # $APPTAINER_CACHEDIR

/users/uutsha/                             # 32 GB — dotfiles, .bashrc, .ssh ONLY
```

Rationale:

- **Python env on `/projects`.** It is backed up, has no purge exposure, and an env is small in bytes
  but has tens of thousands of tiny files — Lustre hates tiny-file metadata churn, IntelliFlash/NFS
  is the better fit. The **[OFFICIAL]** conda page uses `/projects/<id>/mycondadir` in its own example.
- **`HF_HOME` on `/scratch`.** Llama-3.2-1B is ~2.5 GB; an 8B is ~16 GB; plus blob/symlink
  duplication. Large and 100% regenerable → scratch. If purging turns out to be aggressive
  (**[UNKNOWN]**), keep a curated copy of the exact pinned revision under `/projects/m000211/models/`.
- **PyG datasets on `/projects`.** Cora is ~15 MB, PubMed ~50 MB, `ogbn-arxiv` ~200 MB, OGB molecular
  sets small. Tiny, and having them backed up makes runs reproducible. Set
  `root=os.environ["PYG_DATA_ROOT"]` in code so it is never hardcoded.
- **Never in `$HOME`:** `~/.cache`, `~/.triton`, `~/.conda`, the apptainer cache. **[OFFICIAL]** the
  NGC page explicitly tells you to symlink these out to scratch:
  ```bash
  ln -s /scratch/m000211/$USER/triton   ~/.triton
  ln -s /scratch/m000211/$USER/xdgcache ~/.cache
  ```
  Setting the env vars (as the template does) is cleaner; do both if you want belt and braces.

Quota checks — **[UNKNOWN]** which command Marlowe actually supports. Try, in order:
`quota -s`, `lfs quota -h -u $USER /scratch`, `lfs quota -h -p <projid> /scratch`,
`df -h /projects/m000211`.

---

## 7. Software environment

### 7.1 Lmod

**[OFFICIAL]** Yes — Lmod. `module avail` lists everything.
**[OFFICIAL]** Only `slurm` and `gcc/13.1.0` are loaded by default. Add `module load <name>` to
`~/.bashrc` for anything you always want.

Documented modules: `apptainer` (**[OFFICIAL]** actually just run `apptainer` directly — no module
load needed), `conda`, `conan`, `cudnn`, `cudatoolkit`, `stockcuda/12.6.2`, `java`, `nvhpc`, `mps`,
`gcc/64`, `slurm`. Notably **absent from the docs: any `python` or `pytorch` module.**

### 7.2 CUDA

**[OFFICIAL]**

- `module load nvhpc` — the **preferred** way to get `nvcc` and CUDA libs. *"Due to NVHPC overwriting
  the local CC and CXX variables, by default it is not loaded."*
- `module load cudatoolkit` — same as nvhpc **plus** it sets `CUDA_HOME`. Only updated when nvhpc is.
- `module load stockcuda/12.6.2` — stock CUDA, missing HPC libraries, "sporadically updated",
  explicitly discouraged.
- cuDNN is a **separate** module: `cudnn/cuda12/9.3.0.75`, `cudnn/cuda12/8.9.7.29`,
  `cudnn/cuda11/9.3.0.75`. `module load cudnn/cuda12` gets the default.

> **Gotcha aimed squarely at us.** `nvhpc` **overwrites `CC` and `CXX`** (to `nvc`/`nvc++`).
> `torch.compile`'s Inductor backend shells out to `$CXX` to build its C++ wrapper code — with
> `nvc++` that can fail with confusing errors. Since our FlexAttention path lives or dies on
> `torch.compile`, **do not load `nvhpc` unless you actually need `nvcc`**, and if you do,
> immediately `export CC=gcc CXX=g++` afterwards. This is [INFERRED], not documented — but it is the
> single most likely source of a mystifying compile failure on this project.
>
> A pip/conda PyTorch wheel bundles its own CUDA runtime and its own `ptxas` for Triton, so for pure
> PyTorch work you typically need **no CUDA module at all**. Verify with
> `python -c "import torch; print(torch.version.cuda)"`.

**[UNKNOWN]** the CUDA **driver** version, and therefore the maximum CUDA runtime a wheel can use.
Check `nvidia-smi` on a compute node before picking a `cu12x` wheel index.

### 7.3 Always load `mps`

**[OFFICIAL]** *"The `CUDA_ERROR_MPS_CONNECTION_FAILED` error can happen during any job that uses
CUDA under the hood (PyTorch, JAX, etc.) if a leftover `/tmp/nvidia-mps` directory from another job
is present on the node."* Fix, inside the job:

```bash
module load mps
```

It sets `CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$SLURM_JOB_ID` and
`CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$SLURM_JOB_ID`; a Slurm epilog cleans them up afterwards.
**Put `module load mps` in every GPU job.** It is free insurance against a random node poisoning a
run — and with a 21-task array, "a random node" is a near-certainty.

### 7.4 Conda — supported, not discouraged

**[OFFICIAL]** *"Marlowe uses an optimized version of Conda called Mamba. Every conda command works
with mamba. No code customization is needed."*

```bash
module load conda
mamba init          # then LOG OUT and LOG BACK IN, else `pip install` misbehaves
```

**[OFFICIAL]** Because `$HOME` is 32 GB, install environments elsewhere with `--prefix`:

```bash
conda create --prefix /projects/m000211/envs/g2l python=3.11
conda activate /projects/m000211/envs/g2l
```

**My recommendation for this project:** a plain **venv + pip** is simpler and lighter than conda for a
pure-PyTorch stack, and avoids the `mamba init` / logout dance entirely:

```bash
# on a login node (light work, explicitly allowed)
module load slurm
python3 -m venv /projects/m000211/envs/g2l
source /projects/m000211/envs/g2l/bin/activate
python -m pip install -U pip wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # match driver, §7.2
python -m pip install torch_geometric ogb transformers accelerate peft datasets \
                      safetensors hf_transfer huggingface_hub numpy scipy wandb
```

**[UNKNOWN]** whether a system `python3` of a suitable version exists on the login nodes, and which.
If not, `module load conda` and use mamba to provide the interpreter. Check `python3 -V` and
`module avail python` first.

### 7.5 Containers — supported and idiomatic, but not mandatory

**[OFFICIAL]** **Apptainer** (formerly Singularity). **Docker is not supported** *"due to the security
risks associated with it"*, but *"Apptainer supports running docker containers natively."*
No Enroot/Pyxis (`srun --container-image=...`) is mentioned anywhere. **[OFFICIAL]** There is **no
explicit "prefer containers" statement** — modules and containers are presented as equal peers. The
docs' most elaborate worked examples are, however, container-based (NGC NIM / Evo), and the RDS team
clearly leans on NGC PyTorch containers.

Set the cache out of `$HOME` **first** — **[OFFICIAL]**:

```bash
export APPTAINER_CACHEDIR=/scratch/m000211/$USER/apptainer_cache
mkdir -p "$APPTAINER_CACHEDIR"
```

Documented workflow:

```bash
# pull a Docker/NGC image into a .sif (login node is fine, no GPU needed)
cd /scratch/m000211/$USER
apptainer pull docker://nvcr.io/nvidia/pytorch:25.01-py3

# build from a definition file (also fine on the login node)
apptainer build g2l.sif g2l.def

# run with GPUs, binding our filesystems in
apptainer run   --nv --bind /scratch/m000211,/projects/m000211 g2l.sif
apptainer shell --nv --bind /scratch/m000211,/projects/m000211 g2l.sif
apptainer exec  --nv --bind /scratch/m000211,/projects/m000211 g2l.sif python -m g2l.train ...
```

For NGC images (`nvcr.io`) you need an NVIDIA developer API key — **[OFFICIAL]**:

```bash
export APPTAINER_DOCKER_USERNAME='$oauthtoken'
export APPTAINER_DOCKER_PASSWORD="<your NGC API key>"   # keep OUT of git; ~/.bash_profile, chmod 600
```

**[OFFICIAL] Triton warning, directly relevant to our FlexAttention plan:**
*"Several NVIDIA containers contain bad triton versions, which look for `libcuda.so` in the wrong
places, so delete if installed and reinstall latest"* — the docs' own `evo.def` does
`pip uninstall -y triton && pip install triton`, and notes that `pytorch:24.02-py3` was affected
while `pytorch:25.01-py3` is the one they recommend. **If we use an NGC PyTorch container and
`torch.compile`/FlexAttention breaks with a `libcuda.so` error, this is the cause.**

**[OFFICIAL]** `--bind` is required for anything outside the container, including module trees:
*"Apptainer is an isolated environment as a rule. This means you will need to bind the directory of
whatever module you are using into the apptainer container."*

**My recommendation:** start with the **venv on `/projects`** — simpler, and our stack is ordinary
pip-installable PyTorch. Keep a container definition file as the reproducibility artefact and
fallback if we hit driver-vs-wheel or Triton problems. **[OFFICIAL]** notes a `.sif` "can be shared
among your lab members or others and will help with reproducibility."

### 7.6 Internet access from compute nodes

**[UNKNOWN] — never stated explicitly. But the official docs strongly imply YES. [INFERRED]:**

1. The **[OFFICIAL]** NGC NIM example runs `apptainer run` of a NIM container **inside an `srun`
   allocation on `n01`** and says *"this will take about 10 minutes the first time"* — a NIM
   downloads its model weights from NGC on first run. That requires outbound HTTPS from a compute node.
2. The **[OFFICIAL]** `evo.def` `%runscript` does `git clone https://github.com/evo-design/evo.git`
   and is invoked via `apptainer run` from an **`sbatch` job on the `batch` partition**. Outbound git
   over HTTPS from a compute node.
3. **[STANFORD-OTHER]** rcpedia's Marlowe quick-start does `apptainer pull docker://ollama/ollama`
   and `ollama pull deepseek-r1:7b` **after** `srun`-ing onto a GPU node.

**Nevertheless, plan as if it might be restricted, rate-limited, or slow.** Our policy:

```bash
# ON A LOGIN NODE (light, allowed, and the documented place for data movement):
module load slurm
source /projects/m000211/envs/g2l/bin/activate
export HF_HOME=/scratch/m000211/$USER/hf
huggingface-cli login                     # for gated Llama; token lands in $HF_HOME, never in git
huggingface-cli download meta-llama/Llama-3.2-1B --revision <pin-the-sha>
huggingface-cli download Qwen/Qwen2.5-1.5B                    # ungated fallback
python -c "from torch_geometric.datasets import Planetoid; Planetoid('/projects/m000211/data/pyg','Cora')"
```

Then in the job, run **offline** (the template already sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`). This is the right call regardless of what the answer turns out to be: it
makes runs reproducible, removes a whole failure mode, and stops 20 array tasks all hammering
huggingface.co at the same instant.

Gated Llama caveat: `meta-llama/Llama-3.2-1B` requires accepting the licence on your HF account
first, and the token must be present. **Qwen2.5-1.5B / Qwen2.5-7B are ungated** and are the right
de-risking choice for early plumbing work — same architecture family (Llama-style decoder, RoPE,
GQA), so the attention-logit-bias hook and LoRA wiring transfer with minimal change.

---

## 8. Data transfer

**[OFFICIAL]** Two documented routes.

**Globus** — Marlowe has a Globus DTN, reachable through the Globus web UI; sign in with SUNet.
Three collections are exposed: `/scratch/`, `/projects/`, and `$HOME`. The same autofs quirk applies:
*"To get your `/projects/` directory to show up, select the 'Marlowe /projects directories' Globus
collection, then manually type in your project ID after the slash."* Use Globus for anything large
(multi-GB datasets in, checkpoints out).

**scp/rsync over SSH** — not given its own page, but the login nodes are **[OFFICIAL]** explicitly
*"for editing, compiling, submitting jobs, and moving data"*, so `scp`/`rsync` to a login node is
sanctioned. With the ControlMaster alias from §1.3 these cost **zero extra Duo pushes**:

```bash
# from WSL, after `ssh -fN marlowe` has established the master
rsync -avzP --delete \
      --exclude '.git' --exclude '__pycache__' --exclude '*.pt' \
      ~/dev/graph2llm/ marlowe:/projects/m000211/graph2llm/

scp marlowe:/scratch/m000211/uutsha/runs/12345/metrics.json ./
```

**Better for code: use git.** Push from Windows to GitHub, `git pull` on Marlowe (login nodes have
outbound HTTPS — that is how `apptainer pull` and `pip install` work there). Keeps provenance and
avoids rsync-vs-Lustre small-file pain.

**[UNKNOWN]** whether there is a separate named DTN hostname for scp/rsync (e.g. `dtn.marlowe...`).
Not documented; **[OFFICIAL]** no dedicated transfer node is mentioned other than the Globus DTN.

---

## 9. Policies, and what will get us in trouble

All **[OFFICIAL]**, from the Usage Violations page. Read it — it is short and unusually blunt.
Governing document: Stanford's Computer and Network Usage Policy, Administrative Guide 6.2.1.

### 9.1 Do not create a non-authenticated path into Marlowe

*"Marlowe is reached by SSH to the login nodes with a SUNet ID, a password, and Duo two-factor
authentication. Do not run anything that lets a person or a service reach Marlowe without passing
that gate."* Named examples of **prohibited** tools:

- websocket-based tunnels — **"SSHX or Claude Remote Control/Server"**
- userspace VPNs — Tailscale, ZeroTier
- **"the built-in web tunnels launched from editors such as VS Code, Cursor, and Windsurf"**
  (i.e. `code tunnel` / Remote Tunnels)

*"Ordinary SSH tunnels and port forwarding are fine, because they still require your password and Duo."*

> **Direct consequence for how we work.** Do **not** run Claude Code's remote-control/server mode,
> `sshx`, or `code tunnel` on Marlowe. ControlMaster multiplexing (§1.3) is fine — it is ordinary
> OpenSSH and the initial connection still passes password + Duo. `ssh -L` from your laptop to a
> **login** node is fine; note separately that **[OFFICIAL]** port forwarding is not allowed on
> **compute** nodes (§5), so use Open OnDemand for browser apps on compute nodes.

### 9.2 Do not degrade the login nodes

*"The login nodes are shared by everyone... They are for editing, compiling, submitting jobs, and
moving data. Short, light commands are fine; heavy or long-running work is not. If something will run
for a sustained period, or take a large share of a node's CPU or memory, it belongs on a compute
node, and may be subject to termination by Marlowe staff."*

Named examples: CPU benchmarks, storage benchmarks, deliberately thrashing swap, and
**"AI inference servers such as Ollama, vLLM, or SGLang"**.

For us that means, on a login node: git, editing, `sbatch`, `squeue`, `rsync`, `pip install`,
`apptainer pull`/`build`, `huggingface-cli download`. **Not**: training, `torch.compile` warmups,
dataset preprocessing loops, a Jupyter kernel doing real work, or any model serving.

### 9.3 Consequences

*"Abuse of Marlowe has consequences, up to and including loss of access for you and for your
project."* A sysadmin may temporarily suspend access; violations are reported to Stanford's
Information Security Office. This is a **PI-level** risk, not just a personal one.

### 9.4 Other things that will kill a job

- Exceeding partition walltime — `batch` hard-stops at **2 days** [OFFICIAL].
- Running on `preempt`: killed within **15 minutes** when `batch`/`hero` wants the node [OFFICIAL].
  Always checkpoint if you use `preempt`.
- `disk quota exceeded` [OFFICIAL] — the 32 GB `$HOME`. This is the #1 self-inflicted wound on this
  machine; §6 is the whole mitigation.
- `CUDA_ERROR_MPS_CONNECTION_FAILED` from a stale `/tmp/nvidia-mps` left by another job [OFFICIAL] —
  `module load mps` prevents it.
- **[OFFICIAL]** Data classification is **Low and Moderate Risk only.** No PHI / High Risk data.
  Relevant when we get to "biomedical networks": public benchmark graphs are fine; anything
  patient-derived is not, without checking first.
- **[OFFICIAL]** Never share credentials (Admin Guide 6.2.1 §2.b.1). Also: never commit an HF token
  or NGC API key. Keep them in `~/.bash_profile` (`chmod 600`) or in `$HF_HOME`, never in the repo.

### 9.5 Support

**[OFFICIAL]** Slack **`#marlowe-researchers`** on the Stanford Slack Grid is the *primary* channel
("Start with Slack") — the Marlowe team reads it. Email `srcc-support@stanford.edu` with "Marlowe" in
the subject for anything account-specific. There are also recurring **Zoom office hours** covering
"getting started, Slurm and scheduling, scaling a run across nodes, storage, and allocations", and
every PI is paired with a Research Data Scientist. Allocation/policy questions:
`marlowe-info@stanford.edu`.

Given that a Research Data Scientist is assigned to the project, the fastest route to resolving every
[UNKNOWN] below is one Slack post or one office-hours visit.

---

## 10. UNKNOWNS — verify these at first login

Run this block in the first session and paste the output back into this file.

```bash
# ---- 1. auth: are keys accepted at all? does anything skip Duo?
ls -la ~/.ssh/
grep -i -E 'pubkey|gssapi|password' /etc/ssh/sshd_config 2>/dev/null
hostname                       # which login node did the alias give me?

# ---- 2. slurm ground truth
scontrol show partition                       # real MaxTime / MaxNodes / preempt config per partition
scontrol show config | grep -i -E 'maxarray|maxjobcount|defmempercpu|maxmempercpu|preempt|schedulertype|priorityweight'
sacctmgr show assoc user=$USER format=Account,Partition,QOS,MaxJobs,MaxSubmit,GrpTRES,MaxTRES -p
sacctmgr show qos format=Name,Priority,MaxTRESPU,MaxJobsPU,MaxWall,Flags -p
sinfo -o "%P %.5a %.10l %.6D %.6t %N %G"      # does a debug/dev/beta partition exist?
sinfo -N -o "%n %c %m %G %f" | head           # cores / MB RAM / GRES / features per node
scontrol show node n01                        # incl. TmpDisk => node-local NVMe size

# ---- 3. defaults if I omit them (submit a probe job and read AllocTRES)
sbatch --wrap='sleep 20; scontrol show job $SLURM_JOB_ID' \
       -A marlowe-m000211-pm06 -p batch -N1 -n1 -G1 -t 00:02:00 \
       -o /scratch/m000211/$USER/probe-%j.out
# does --gpus-per-node work?  does --gres=gpu:1 work?
sbatch --test-only -A marlowe-m000211-pm06 -p batch -N1 --gpus-per-node=1 -t 00:05:00 --wrap='true'
sbatch --test-only -A marlowe-m000211-pm06 -p batch -N1 --gres=gpu:1      -t 00:05:00 --wrap='true'
# is the BARE account valid on preempt (i.e. uncharged)?
sbatch --test-only -A marlowe-m000211 -p preempt -N1 -G1 -t 00:05:00 --wrap='true'

# ---- 4. storage
ls /projects/m000211                          # force the autofs mount
df -h /users/$USER /projects/m000211 /scratch/m000211
quota -s 2>/dev/null; lfs quota -h -u $USER /scratch 2>/dev/null
lfs df -h /scratch 2>/dev/null
env | grep -i -E 'scratch|group|project'      # is there a $SCRATCH / $PROJECT / $GROUP_SCRATCH?
cat /etc/motd                                 # purge policies are often only announced here

# ---- 5. software
module avail 2>&1 | tee ~/module-avail.txt
module avail python; module avail pytorch; module avail cuda; module avail apptainer
python3 -V; which python3
nvidia-smi                                    # DRIVER version -> which cuXY wheel we may install

# ---- 6. compute-node facts + internet (the decisive test)
srun -N1 -n1 -G1 -c4 -t 00:10:00 -A marlowe-m000211-pm06 -p batch --pty bash -lc '
  hostname; nproc; free -g
  df -h /tmp; ls -ld /tmp            # node-local NVMe? tmpfs? how big?
  nvidia-smi topo -m                 # NVLink topology
  env | grep -i proxy
  curl -sS -o /dev/null -w "huggingface %{http_code} %{time_total}s\n" https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B
  curl -sS -o /dev/null -w "github     %{http_code}\n"                 https://github.com
  curl -sS -o /dev/null -w "pypi       %{http_code}\n"                 https://pypi.org/simple/
'
```

### Explicit list of things the official docs do NOT say

1. **Whether SSH public keys are supported**, and whether any auth method bypasses Duo.
   (Sherlock, same operator, says keys are *not* supported and Kerberos still hits Duo.)
2. **ControlMaster / ControlPersist guidance** — no mention at all. And on native Windows OpenSSH it
   does not work regardless, so **use WSL2**.
3. **Any Windows guidance whatsoever** (PowerShell, PuTTY, plink, WSL).
4. **VPN requirement, bastion, or jump host** — nothing stated. Assumed not required.
5. **Whether compute nodes have outbound internet**, and whether a proxy is needed. Strongly implied
   yes by three official examples; never stated. We pre-download regardless.
6. **QOS names, fairshare/priority policy, per-user or per-project MaxJobs / MaxSubmit / MaxGRES,
   MaxArraySize** — completely absent. Matters for our many-small-jobs plan.
7. **Default `--mem` and `--cpus-per-task` if omitted**, and whether `--mem` is enforced at all.
8. **Whether `--gpus-per-node` and `--gres=gpu:N` are accepted for sbatch** (only `-G` and a single
   `--gres` srun example appear).
9. **Whether the bare account `marlowe-m000211` (no `-pm06`) is usable on `preempt` without charging**
   the Medium allocation. The billing note implies yes; the tier text ("for basic access, you can
   only submit to preempt") muddies it.
10. **`/scratch` purge policy** — stated as not backed up and not replicated, but **no purge schedule
    or file-age policy is published**. Assume it can be purged; keep nothing irreplaceable there.
11. **`/projects/m000211` actual quota** — "varies depending on the amount approved". Check `df`.
12. **Node-local NVMe**: size, mount point, whether Slurm exposes `$TMPDIR`/`$L_SCRATCH`, and whether
    `/tmp` is tmpfs or disk. The only evidence is `/tmp/nvidia-mps-$SLURM_JOB_ID` in the MPS FAQ.
13. **CUDA driver version** on compute nodes → which PyTorch `cu12x` wheel we may install.
14. **Provided Python / PyTorch versions.** No `python` or `pytorch` module is listed anywhere; the
    docs' own PyTorch story is "use an NGC container or install it yourself".
15. **Login node count, names, and any per-user CPU/RAM cgroup limit** on them.
16. **Whether a debug/dev/`beta` partition exists.** An archived Feb-2026 copy of the Slurm page had
    an `srun` example using `-p beta`; the current page does not mention it. Check `sinfo`.
17. **Enroot / Pyxis (`srun --container-image=...`)** — never mentioned. Apptainer only, apparently.
18. **Maintenance windows / downtime cadence.** One "Planned Maintenance Window: February 24-25"
    update exists; no published recurring schedule. Subscribe to the updates feed and Slack.
19. **Whether `module purge` is safe** (it may drop `slurm`, which the official job templates
    re-load — my template does `module purge` then `module load slurm`, which should be equivalent,
    but confirm).
20. **Current partition limits.** They demonstrably changed between Feb 2026 and now. Trust
    `scontrol show partition`, not this file.

---

## 11. Quick start, condensed

```bash
# --- once, in WSL2 on Windows: put the ControlMaster block from section 1.3 in ~/.ssh/config
ssh -fN marlowe            # ONE password + ONE Duo push, good for 8h

# --- once, on Marlowe
ls /projects/m000211                          # kick autofs
mkdir -p /scratch/m000211/$USER/{hf,triton,inductor,torch,xdgcache,logs,runs,apptainer_cache}
mkdir -p /projects/m000211/{graph2llm,envs,models,results} /projects/m000211/data/pyg
cat >> ~/.bashrc <<'EOF'
export PROJ=/projects/m000211
export SCR=/scratch/m000211/$USER
export HF_HOME=$SCR/hf
export TRITON_CACHE_DIR=$SCR/triton
export TORCHINDUCTOR_CACHE_DIR=$SCR/inductor
export TORCH_HOME=$SCR/torch
export XDG_CACHE_HOME=$SCR/xdgcache
export APPTAINER_CACHEDIR=$SCR/apptainer_cache
export PYG_DATA_ROOT=$PROJ/data/pyg
alias sq='squeue -u $USER -o "%.10i %.20j %.9P %.8T %.10M %.6D %R"'
EOF

# --- env + data (login node: light work only)
python3 -m venv $PROJ/envs/g2l && source $PROJ/envs/g2l/bin/activate
pip install -U pip wheel && pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric ogb transformers accelerate peft datasets hf_transfer wandb
huggingface-cli download Qwen/Qwen2.5-1.5B          # ungated; Llama-3.2-1B needs licence acceptance
python -c "from torch_geometric.datasets import Planetoid; Planetoid('$PYG_DATA_ROOT','Cora')"

# --- smoke test on a real GPU (section 5), then
sbatch $PROJ/graph2llm/slurm/train_1gpu.sbatch
sq
sreport cluster UserUtilizationByAccount -T gres/gpu Start=2026-08-01 End=now \
        account=marlowe-m000211-pm06 -t hours
```

---

## Sources

Official (`marlowe-research.stanford.edu`):

- Documentation index — https://marlowe-research.stanford.edu/documentation/
- Connecting — https://marlowe-research.stanford.edu/documentation/getting-started/connecting/
- Get Access — https://marlowe-research.stanford.edu/documentation/getting-started/
- Filesystems — https://marlowe-research.stanford.edu/documentation/getting-started/filesystems/
- Globus — https://marlowe-research.stanford.edu/documentation/getting-started/globus/
- SLURM on Marlowe — https://marlowe-research.stanford.edu/documentation/slurm/
- Open OnDemand — https://marlowe-research.stanford.edu/documentation/openondemand/
- Software overview — https://marlowe-research.stanford.edu/documentation/software/
- Apptainer — https://marlowe-research.stanford.edu/documentation/software/apptainer/
- Conda — https://marlowe-research.stanford.edu/documentation/software/conda/
- CUDA Toolkit — https://marlowe-research.stanford.edu/documentation/software/cudatoolkit/
- cuDNN — https://marlowe-research.stanford.edu/documentation/software/cudnn/
- NVIDIA HPC SDK — https://marlowe-research.stanford.edu/documentation/software/nvhpc/
- Conan — https://marlowe-research.stanford.edu/documentation/software/conan/
- NGC NIM containers (Apptainer + Triton + cache guidance) — https://marlowe-research.stanford.edu/documentation/ngc-nims/
- FAQ — https://marlowe-research.stanford.edu/documentation/faq/
- Tech Specs — https://marlowe-research.stanford.edu/documentation/specs/
- Usage Violations — https://marlowe-research.stanford.edu/documentation/violations/
- Help & Support — https://marlowe-research.stanford.edu/documentation/help-and-support/
- About / detailed specs table — https://marlowe-research.stanford.edu/about/
- Access tiers & recharge rates — https://marlowe-research.stanford.edu/access/
- Project Application Guide — https://marlowe-research.stanford.edu/access/application-guide/
- Hero run cycle update — https://marlowe-research.stanford.edu/updates/hero-run-cycle-2026-07/

Other Stanford sources (secondary, labelled [STANFORD-OTHER] above):

- Sherlock — connection options, key auth, GSSAPI, ControlMaster — https://www.sherlock.stanford.edu/docs/advanced-topics/connection/
- Stanford Data Science — Logging into Marlowe — https://datascience.stanford.edu/logging-marlowe
- Stanford Data Science — Marlowe overview / utilisation stats — https://datascience.stanford.edu/marlowe
- Stanford Data Science — System Specifications — https://datascience.stanford.edu/system-specifications
- University IT — Marlowe service page (risk classification) — https://uit.stanford.edu/cloud-computing/marlowe
- UIT — Two-Step Authentication for SSH on Linux Servers — https://uit.stanford.edu/service/authentication/twostep/duo_ssl
- rcpedia (Stanford GSB DARC) — Running Ollama on Stanford Computing Clusters, incl. Marlowe `srun`/sbatch examples — https://rcpedia.stanford.edu/blog/2025/05/12/running-ollama-on-stanford-computing-clusters/
- Stanford Computer and Network Usage Policy, Admin Guide 6.2.1 — https://adminguide.stanford.edu/chapter-6/subchapter-2/policy-6-2-1

Non-Stanford (used only for the Windows ControlMaster limitation and doc-drift evidence):

- Win32-OpenSSH issue #405 "ControlPath in ssh_config fails" — https://github.com/PowerShell/Win32-OpenSSH/issues/405
- Win32-OpenSSH issue #1328 "Support for Control Master" — https://github.com/PowerShell/Win32-OpenSSH/issues/1328
- Archived Feb-2026 Marlowe Slurm page (evidence that partition limits change) — https://web.archive.org/web/20260213045809/https://docs.marlowe.stanford.edu/slurm.html
