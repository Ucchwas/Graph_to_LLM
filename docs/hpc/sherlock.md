# Stanford Sherlock — practical notes (SECONDARY cluster)

**Project:** Graph-in / graph-out — trainable encoder -> frozen Llama-3.2-1B-class LLM with a learned
graph-derived additive attention bias + LoRA -> trainable decoder. Masked edge prediction on Cora
(N=2708), later OGB / PubMed / ogbn-arxiv.

**Status:** research-only document. Written **entirely from the official public docs**
(<https://www.sherlock.stanford.edu/docs/>) plus the docs' own source repo
(`github.com/stanford-rc/www.sherlock.stanford.edu`, `includes/data/facts.yml` and
`includes/data/software.yml`). **No cluster was contacted. No SSH, no Duo push was fired.**

User specifics assumed (verify on first login):
- SUNet ID `uutsha`, host `login.sherlock.stanford.edu`
- `~/BioGlyph` -> `/home/groups/tauhid/ucchwas/BioGlyph` (PI group `tauhid`, i.e. `$GROUP_HOME=/home/groups/tauhid`)
- `$SCRATCH = /scratch/users/uutsha`

Anything marked **[NOT IN DOCS]** is inference or vendor spec, not a Sherlock documentation statement.
Everything else is doc-backed and quoted or paraphrased closely.

---

## 0. TL;DR for our workload

```bash
# one-time, from your laptop: ~/.ssh/config  (kills the per-command Duo push)
Host login.sherlock.stanford.edu
    ControlMaster auto
    ControlPath ~/.ssh/%l%r@%h:%p

# login
ssh uutsha@login.sherlock.stanford.edu       # password (SUNet) -> "Authenticated with partial success." -> Duo

# what can I actually use?
sh_part                    # partitions, limits, idle CPUs/GPUs, default+max walltime
sh_quota                   # quotas on HOME / GROUP_HOME / SCRATCH / GROUP_SCRATCH / OAK
node_feat -p gpu | grep GPU_            # the ONLY authoritative list of GPU constraint values
sh_node_feat -p owners | grep GPU_SKU   # what the tauhid/owners pool actually has

# quick interactive GPU for debugging (MIG slice, instant, 2h max)
sh_dev -g 1

# real interactive GPU
sh_dev -c 8 -m 64GB -g 1 -p gpu -t 2:00:00
# or: salloc -p gpu -G 1 -c 8 --mem=64GB -t 2:00:00

# batch: see the full template in section 7
sbatch train.sbatch
```

Three things that will bite us specifically:

1. **The public `gpu` partition is mostly pre-Ampere.** V100 / TITAN Xp / RTX 2080Ti / P100 have
   compute capability < 8.0, so **no bf16 and no usable FlexAttention/Triton path**. We must constrain
   to Ampere-or-newer (RTX 3090 / L40S / H100). See section 2.
2. **`gpu` partition caps you at 16 GPUs/user and 2 days**, and `--mem-per-cpu` max is 32 GB there.
3. **`$SCRATCH` is purged after 90 days of no *content* modification.** `touch` does not save a file.
   Our HF model cache and PyG datasets living on `$SCRATCH` will silently evaporate between paper pushes.

---

## 1. CONNECTING

### 1.1 The command

```bash
ssh uutsha@login.sherlock.stanford.edu
```

You land on the least-loaded of **20 load-balanced login nodes** (`sh0X-lnNN`). The prompt shows
`login!` in red as a reminder you are *not* on a compute node.

Host key fingerprints published in the docs (verify on first connect):

| Key type | Fingerprint |
|---|---|
| RSA | `SHA256:T1q1Tbq8k5XBD5PIxvlCfTxNMi1ORWwKNRPeZPXUfJA` |
| ECDSA | `SHA256:eB0bODKdaCWtPgv0pYozsdC5ckfcBFVOxeMwrNKdkmg` |

**[NOT IN DOCS at time of writing]** these two fingerprints were rendered from the docs page but are
served from an include file that did not resolve standalone; re-check the live page before trusting
them blindly.

### 1.2 Auth flow / Duo

Username = SUNet ID, password = **SUNet ID password**. Sherlock does *not* store your password; it
delegates to Stanford's central Kerberos service, and SRCC **cannot** reset it.

Sequence:

```
uutsha@login.sherlock.stanford.edu's password:
Authenticated with partial success.

Duo two-factor login for uutsha

Enter a passcode or select one of the following options:

 1. Duo Push to XXX-XXX-9999
 2. Phone call to XXX-XXX-9999
 3. SMS passcodes to XXX-XXX-9999 (next code starts with: 9)

Passcode or option (1-3):
```

then `Success. Logging you in...`

> **Danger, doc-quoted:** "Entering an invalid password multiple times will result in a (temporary) ban
> of your IP address." You then get `Connection refused`, auto-cleared "after a few minutes". Stanford
> VPN gives you a different IP as an escape hatch.
>
> Related trap the docs call out: **SSHFS on macOS** auto-reconnects after sleep without credentials and
> will get your IP blacklisted. Unmount before sleeping.

### 1.3 SSH keys — POLICY

> **"SSH public-key authentication is not supported on Sherlock."** (Advanced connection options page.)

So: **do not** try to install an `authorized_keys`-based login. There are exactly two documented
auth methods:

1. **Password + Duo** (recommended by the docs).
2. **GSSAPI / Kerberos** — no password per connection, but still a Duo prompt. Setup:

   ```bash
   sudo curl -o /etc/krb5.conf https://web.stanford.edu/dept/its/support/kerberos/dist/krb5.conf
   ```

   then in `~/.ssh/config` (indentation matters):

   ```
   Host login.sherlock.stanford.edu
       GSSAPIDelegateCredentials yes
       GSSAPIAuthentication yes
   ```

   `kinit uutsha@stanford.edu` once, `klist` to verify. **Kerberos tickets last 25 hours**, so `kinit`
   roughly daily. Existing SSH sessions are *not* killed when the ticket expires. `kdestroy` to drop it.

### 1.4 ControlMaster / connection multiplexing — the Duo-per-command fix

This is the documented answer to "avoiding multiple Duo prompts". Verbatim from the docs
(`Advanced connection options` -> `SSH options` -> `Avoiding multiple Duo prompts`):

```
Host login.sherlock.stanford.edu
    ControlMaster auto
    ControlPath ~/.ssh/%l%r@%h:%p
```

Docs' notes on it:

- It lets SSH "re-use an existing connection to Sherlock each time you open a new session ... thus
  avoiding subsequent 2FA prompts once the initial connection is established."
- **Downside stated in the docs:** once one connection is open, *all* subsequent connections pin to that
  same login node, "somewhat defeat[ing] the purpose of the load-balancing mechanism".
- If you hit `unix_listener: "..." too long for Unix domain socket` (a macOS path-length limit), replace
  the ControlPath line with:

  ```
  ControlPath ~/.ssh/%C
  ```

  The docs' `advanced-topics/connection` page actually shows the `%C` form in its own ControlMaster block
  too — either is fine; `%C` (hash) is the safer default everywhere.

**Practical add-ons [NOT IN DOCS]** — not documented by SRCC, but standard and safe:
`ControlPersist 4h` to keep the master alive after the last session closes, and `ServerAliveInterval 60`.
Recommended combined block for our laptop:

```
Host sherlock
    HostName login.sherlock.stanford.edu
    User uutsha
    ControlMaster auto
    ControlPath ~/.ssh/%C
    ControlPersist 4h          # NOT IN SHERLOCK DOCS - standard OpenSSH, keeps master alive
    ServerAliveInterval 60     # NOT IN SHERLOCK DOCS
```

Docs also explicitly say: **do not enable `Compression yes`** on fast networks (CPU overhead > bandwidth
saving); only consider it on slow/high-latency links. And **do not hand-tune `Ciphers`** — modern
OpenSSH already negotiates `aes128-gcm@openssh.com` / `chacha20-poly1305@openssh.com`.

### 1.5 Windows guidance

Docs' Windows section (`getting-started/index` -> SSH clients):

- Windows ships a built-in OpenSSH client usable from Windows Terminal — that works.
- SRCC recommends **WSL** for "the best compatibility with the Sherlock environment", and the GSSAPI
  instructions are explicitly footnoted as Linux/macOS, with **"For Windows, we recommend using the WSL"**.
- Other clients (PuTTY, MobaXterm...) are "available, but have not necessarily been tested with
  Sherlock, so your mileage may vary."

**Concrete recommendation for us (Windows 11):** use WSL, put the `~/.ssh/config` block above inside WSL,
so ControlMaster actually works (Windows-native OpenSSH ControlMaster support is unreliable
**[NOT IN DOCS]**).

### 1.6 Other connection facts

- Sticking to a specific login node (for `tmux`/`screen` persistence) is possible but **not recommended**:
  `ssh uutsha@ln21.sherlock.stanford.edu`.
- No geo/IP restriction; VPN recommended on untrusted networks.
- **OnDemand** web portal: <https://ondemand.sherlock.stanford.edu> — shell, file editor, job submission,
  Jupyter, VS Code (code-server), all without a local SSH client. Requesting a GPU in the `dev`
  partition from OnDemand gives you a MIG lightweight GPU instance.
- **Data transfer:** use the DTNs — `dtn.sherlock.stanford.edu` (7 of them, dedicated bandwidth, **no
  interactive shell**). Note the doc gotcha: on the DTNs the *default* destination path is not `$HOME`
  the way it is on login nodes — specify paths explicitly.

---

## 2. GPU RESOURCES

### 2.1 The fleet (from the docs' own `facts.yml`, i.e. the numbers rendered on `/docs/tech/`)

Cluster-wide: **2,073 compute nodes, 76,196 CPU cores, 1,240 GPUs, 14 GPU models, 5 GPU generations.**

**Public `gpu` partition — 47 nodes, 1,444 cores. This is the exact node inventory:**

| Nodes | CPU | Host RAM | IB | GPUs |
|---:|---|---:|---|---|
| 1 | 20c Intel E5-2640v4 | 256 GB | EDR | 4x Tesla **P100 PCIe** |
| 2 | 20c Intel E5-2640v4 | 256 GB | EDR | 4x Tesla **V100_SXM2** |
| 6 | 20c Intel E5-2640v4 | 512 GB | EDR | 4x GeForce **TITAN_Xp** |
| 2 | 20c Intel E5-2640v4 | 1024 GB | EDR | 4x GeForce **TITAN_Xp** |
| 1 | 24c Intel 5118 | 191 GB | EDR | 4x Tesla **V100_SXM2** |
| 2 | 24c Intel 5118 | 191 GB | EDR | 4x Tesla **V100 PCIe** |
| 16 | 32c AMD 7502P | 256 GB | HDR | 4x GeForce **RTX_2080Ti** |
| 4 | 32c AMD 7502P | 256 GB | HDR | 4x GeForce **RTX_3090** |
| 2 | 32c AMD 7502P | 256 GB | HDR | 4x Tesla **V100S PCIe** |
| 8 | 32c Intel 6426Y | 256 GB | NDR | 4x Tesla **L40S** |
| 2 | 64c Intel 8462Y+ | 1024 GB | NDR | 4x Tesla **H100_SXM5** |
| 1 | 64c Intel 8462Y+ | 2048 GB | NDR | 8x Tesla **H100_SXM5** |

**`dev` partition — 4 nodes:** 2x (20c E5-2640v4, 128 GB) and 2x (32c AMD 7543P, 256 GB, HDR) with
**32x Tesla A30_MIG-1g.6gb** GPU instances. That is what `sh_dev -g 1` hands you.

**`owners` partition — 1,732 nodes, 66,516 cores**, GPU-bearing configs include:
TITAN_Xp (4x and 8x), TITAN_V, P40, P100 PCIe, V100_SXM2 (4x and 8x), V100 PCIe, V100S PCIe,
RTX_2080Ti, RTX_3090, **A40 (22 nodes, 4x)**, **A100 PCIe (4 nodes, 4x)**,
**A100_SXM4 (13+5 nodes 4x, 10+7 nodes 8x)**, **L40S (26 nodes, 4x)**,
**H100_SXM5 (13 nodes 4x, 16 nodes 8x)**, **H200_SXM5 (17 nodes, 8x)**.

> The `owners` partition is where the modern GPUs live in bulk. It is preemptible (section 3).

### 2.2 GPU memory per model — **[NOT IN DOCS: vendor specs, must verify with `node_feat`]**

The Sherlock docs deliberately do **not** publish a model->memory table (the one on the GPU page is
explicitly footnoted "The lists of values provided in the table are non exhaustive" and is visibly
stale — it still only lists Pascal/Maxwell, P100/P40, 16GB/24GB). These are NVIDIA specs plus compute
capability, which is what actually matters for bf16 + Triton:

| GPU_SKU (approx.) | Mem | CC | bf16? | Where |
|---|---:|---:|:---:|---|
| P100 PCIe | 16 GB | 6.0 | no | gpu, owners |
| P40 | 24 GB | 6.1 | no | owners |
| TITAN_Xp | 12 GB | 6.1 | no | gpu, owners |
| TITAN_V | 12 GB | 7.0 | no | owners |
| V100 PCIe / V100_SXM2 | 16 GB (some 32) | 7.0 | no | gpu, owners |
| V100S PCIe | 32 GB | 7.0 | no | gpu, owners |
| RTX_2080Ti | 11 GB | 7.5 | no | gpu, owners |
| A30 MIG 1g.6gb | ~6 GB slice | 8.0 | yes | dev |
| A100 PCIe / A100_SXM4 | 40 or 80 GB | 8.0 | yes | owners |
| RTX_3090 | 24 GB | 8.6 | yes | gpu, owners |
| A40 | 48 GB | 8.6 | yes | owners |
| L40S | 48 GB | 8.9 | yes | gpu, owners |
| H100_SXM5 | 80 GB | 9.0 | yes | gpu, owners |
| H200_SXM5 | 141 GB | 9.0 | yes | owners |

**Consequence for this project:** in the *public* `gpu` partition, only **RTX_3090 (24 GB)**,
**L40S (48 GB)** and **H100_SXM5 (80 GB)** are Ampere+ — i.e. the only ones where bf16 is native and
where `torch.compile` + Triton FlexAttention is realistic. That is 4 + 8 + 3 = **15 of the 47 gpu nodes**.
Everything else (V100/TITAN Xp/2080Ti/P100 — 32 nodes) will either refuse bf16 or fall back to slow
emulation, and will force us onto the eager `[B, heads, N, N]` path we are trying to avoid.

### 2.3 Request syntax — what the docs actually say

**Canonical form on Sherlock is `--gpus` / `-G`, not `--gres`:**

```bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -G 1
```

> **Hard rule, quoted:** *"GPU resources MUST be requested explicitly. Jobs will be rejected at
> submission time if they don't explicitly request GPU resources."* Submitting `salloc -p gpu` with no
> GPU request gives:
> `srun: error: Unable to allocate resources: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)`

`--gres=gpu:1` **does** work (the Apptainer page uses `srun -p gpu -c 4 --gres gpu:1 --pty bash`), but the
docs standardise on `-G`/`--gpus`. Note there is **no documented `--gres=gpu:<model>:N` syntax on
Sherlock** — GPU model selection is done with `-C` constraints, not with gres type names. If you copy a
`--gres=gpu:h100:1` line from another cluster it will fail here **[NOT IN DOCS — inferred from the
complete absence of typed gres in every doc example]**.

Other documented GPU options (all valid for `srun`/`sbatch`/`salloc`):
`--cpus-per-gpu`, `--gpus-per-node`, `--gpus-per-task`, `--mem-per-gpu`, `--gpu-bind`, `--gpu-freq`.
Docs warn that mixing them creates conflicts that get the job rejected, e.g.
`sbatch --gpus-per-task=1 --cpus-per-gpu=2 --cpus-per-task=1` (first two imply cpus-per-task=2).

### 2.4 Selecting a specific GPU model: `-C` node features

Full documented feature-tag list (`advanced-topics/node-features`):

| Feature | Meaning | Documented example values |
|---|---|---|
| `CLASS:xxx` | node type in the Sherlock catalog | `CLASS:SH3_CBASE`, `CLASS:SH3_G4TF64` |
| `CPU_MNF:xxx` | CPU manufacturer | `INTEL`, `AMD` |
| `CPU_GEN:xxx` | CPU generation | `RME` (Rome), `SKX` (Skylake), `BDW`, `MLN` (Milan) |
| `CPU_SKU:xxx` | CPU model | `5118`, `7502P` |
| `CPU_FRQ:xxx` | base clock | `2.50GHz`, `2.75GHz` |
| `GPU_BRD:xxx` | GPU brand | `GEFORCE`, `TESLA` |
| `GPU_GEN:xxx` | GPU generation | `VLT` (Volta), `AMP` (Ampere), `PSC` (Pascal), `MXW` (Maxwell) |
| `GPU_SKU:xxx` | GPU model | `A100_SXM4`, `RTX_3090`, and from the docs' own `sh_node_feat -p gpu` output: `P100_PCIE`, `P40`, `RTX_2080Ti`, `V100_PCIE`, `V100S_PCIE`, `V100_SXM2` |
| `GPU_MEM:xxx` | GPU memory | `16GB`, `24GB`, `32GB`, `80GB` |
| `GPU_CC:xxx` | compute capability | `6.1`, `8.0`, `8.6` |
| `IB:xxx` | Infiniband | `EDR`, `HDR` (NDR also exists per facts.yml) |
| `NO_GPU` | tag on CPU-only nodes | — |

**WARNING — the docs are internally inconsistent on `GPU_SKU` spelling.** The GPU page shows
`TESLA_P100_PCIE` / `TESLA_P40`; the node-features page shows `P100_PCIE` / `P40` / `V100_SXM2`.
A wrong string = instant rejection:

```
srun: error: Unable to allocate resources: Requested node configuration is not available
```

So **the first thing to do on first login is dump the real values:**

```bash
node_feat -p gpu    | grep GPU_        # documented on the GPU page
sh_node_feat -p gpu | grep GPU_SKU     # documented on the node-features page
sh_node_feat -p owners | grep GPU_     # what tauhid/owners can reach
```

Both command names appear in the docs; treat whichever exists as authoritative. `node_feat -h` /
`sh_node_feat -h` for options. Note: it only lists features of partitions **you** have access to, so
output differs per user.

Boolean syntax (documented):

```bash
#SBATCH -C 'GPU_MEM:32GB&IB:HDR'                  # AND
#SBATCH -C "CPU_GEN:RME|CPU_GEN:MLN"              # OR (per node; multi-node jobs may be mixed)
#SBATCH -C "[CPU_FRQ:2.50GHz|CPU_FRQ:2.75GHz]"    # matching OR: all nodes identical
```

Documented examples directly relevant to us:

```bash
#SBATCH -G 1
#SBATCH -C GPU_MEM:80GB          # "to make sure that your training job will run on a GPU with 80GB"
```

and, because feature tags are **text, not numbers** (no `>=` possible), the documented workaround for
"CC >= 8.0" is to enumerate:

```bash
#SBATCH -C "GPU_CC:8.0|GPU_CC:8.6"
```

**Our recommended constraint** (enumerate the Ampere+ SKUs available in `gpu`) — verify strings first:

```bash
#SBATCH -C "GPU_SKU:H100_SXM5|GPU_SKU:L40S|GPU_SKU:RTX_3090"
```

or the memory-based equivalent `-C "GPU_MEM:80GB|GPU_MEM:48GB|GPU_MEM:24GB"` (note 24GB also matches the
Pascal **P40**, which is *not* Ampere — so SKU or `GPU_CC` is safer than `GPU_MEM` for us).

> Doc caveat repeated everywhere: *"Adding job constraints often increases job pending times in queue."*
> For our many-short-jobs profile, a tight constraint on 15 nodes may queue longer than the run itself.
> Strategy: constrain to Ampere+ but keep jobs short (<= 2 h) so backfill picks them up (section 3.5).

### 2.5 GPU compute mode

Default on Sherlock is **Exclusive Process** (one CUDA context per device). Override with `--gpu_cmode`:

| Mode | `--gpu_cmode` value | Meaning |
|---|---|---|
| "Default" (NVIDIA) | `shared` | multiple contexts per device |
| "Exclusive Process" | `exclusive` | **Sherlock default** |
| "Prohibited" | `prohibited` | no CUDA context allowed |

```bash
srun -p gpu -G 1 --gpu_cmode=shared nvidia-smi
```

Relevant if we ever want to pack several tiny encoder-sweep runs onto one GPU: we would need
`--gpu_cmode=shared`. Otherwise the second process fails to get a context.

### 2.6 GPU monitoring

```bash
ml load system nvtop     # htop-like live GPU/memory utilisation, for right-sizing requests
ml system ruse ; ruse -s -t10 --stdout ./myapp   # CPU/mem/time sizing, low overhead
seff <jobid> ; sacct -j <jobid>                  # post-mortem efficiency
```

---

## 3. PARTITIONS

### 3.1 Public partitions (documented table)

| Partition | Purpose | Resources | Limits |
|---|---|---|---|
| `normal` (default) | general compute | 20-64 cores/node, 6-8 GB RAM/core | default 2 h, **max 2 days**, up to **7 days with `--qos=long`** |
| `bigmem` | > 256 GB jobs | up to 4 TB RAM/node | **max 1 day** |
| `gpu` | GPU jobs | 20-64 cores/node, up to 2 TB RAM/node, 4 or 8 GPUs/node | **16 GPUs/user**, default 1 h, max 2 days |
| `dev` | dev/testing | dedicated nodes + MIG lightweight GPU instances | **2 h max, 4 cores + 2 GPUs/user** |
| `service` | lightweight recurring/admin tasks (transfers, cron-like, DB) | massively over-subscribed | **2 jobs, 16 cores/user, 2 days** |

Real `sh_part` output shown in the docs (illustrative numbers, structure is what matters):

```
 partition           || nodes         | CPU cores             | GPUs                 || job runtime     | mem/core        | per-node
 name         public ||   idle  total |   idle  total  queued |   idle  total queued || default maximum | default maximum |    cores   mem(GB)  gpus
-----------------------------------------------------------------------------------------------------------------------------------------------------
 normal*      yes    ||      0    218 |    438   5844    6949 |      0      0      0 ||      2h      7d |     6GB     8GB |    20-64   128-384     0
 bigmem       yes    ||      0     11 |    537    824     255 |      0      0      0 ||      2h      1d |     6GB    64GB |   24-256  384-4096     0
 gpu          yes    ||      0     33 |    354   1068     905 |     25    136    196 ||      1h      2d |     8GB    32GB |    20-64  191-2048   4-8
 dev          yes    ||      1      4 |     64    104       0 |     62     64      0 ||      1h      2h |     6GB     8GB |    20-32   128-256  0-32
 service      yes    ||      5      6 |    129    132       0 |      0      0      0 ||      1h      2h |     1GB     8GB |    20-32   128-256     0
-----------------------------------------------------------------------------------------------------------------------------------------------------
```

Read from this: `*` = default partition; `queued` = CPUs/GPUs requested by *pending* jobs (queue depth);
`mem/core` default vs max — **in `gpu`, default 8 GB/core, max 32 GB/core**; `per-node` shows the range.

### 3.2 Owner partitions and the group `tauhid`

Documented model:

- A PI group that has invested gets **a dedicated partition named after the PI's SUNet ID**. So if
  `tauhid` has bought nodes, expect `--partition=tauhid`. Jobs there run **up to 7 days with no special
  QOS**, and owners get **immediate, no-queue access** to their own nodes.
- All members of *any* owner group also get the shared **`owners`** partition, spanning nodes contributed
  by *all* PI groups (1,732 nodes / 66,516 cores). Much bigger pool, **at the price of preemption**.
- `--qos=high_p` (owner partitions only, no effect in public partitions) pushes one of your jobs ahead of
  others already queued in your own partition.

```bash
#SBATCH --partition=tauhid
#SBATCH --qos=high_p
```

**How to find out what our group owns — run these on first login:**

```bash
sh_part                                 # lists EVERY partition you can submit to, incl. a 'tauhid'
                                        # partition and 'owners' if the group has invested
sh_node_feat -p owners | grep GPU_      # GPU models reachable via owners
sh_node_feat -p tauhid                  # if the partition exists: exactly what tauhid bought
scontrol show partition tauhid          # NOT IN DOCS - standard Slurm, shows nodes/limits/QOS
sinfo -p owners -o '%20N %10c %10m %25f %10G'   # NOT IN DOCS - standard Slurm, nodes+features+gres
```

If `sh_part` shows no `tauhid` and no `owners` line, the group has a Sherlock **account** but has not
**invested**; we are then limited to `normal` / `gpu` / `dev` / `bigmem` / `service`.

### 3.3 How `owners` preemption works (documented)

- Owner groups get "immediate and exclusive access to the resources they purchased".
- When their nodes are idle, other owners may use them via `owners`.
- "when the purchasing owners want to use their resources, jobs from other owners that may be running on
  them are **preempted (i.e. killed and re-queued)**".
- The running-jobs page adds: "Jobs that are preempted are automatically requeued and will restart when
  resources are available again, so it is important to make sure your jobs can handle being interrupted
  and restarted (e.g. by **checkpointing regularly**)."

**For us:** `owners` is genuinely attractive (that is where the A100/H100/H200/A40/L40S bulk is), but only
if every training script checkpoints and resumes. Since our jobs are "many small-to-medium experiments",
the cheap correct answer is **keep each job short (30 min - 2 h) and idempotent** so a preemption costs
almost nothing, plus save a checkpoint at the end of each epoch to `$SCRATCH`.

### 3.4 Which partition should we use?

| Situation | Use |
|---|---|
| 5-minute smoke test, shape/dtype debugging, "does the bias tensor even fit" | `sh_dev -g 1` (`dev`, MIG A30 ~6 GB, instant, 2 h cap) |
| Interactive dev on a real GPU | `sh_dev -c 8 -m 64GB -g 1 -p gpu -t 2:00:00` or `salloc -p gpu -G 1 ...` |
| Cora / small-N sweeps (E1-E7, decoder sweep, ablation grid, seeds) | `-p gpu` with Ampere+ constraint, short walltime, job arrays |
| Same, but queue in `gpu` is jammed | add `-p gpu,owners` **if** we are in an owner group, and checkpoint |
| Guaranteed uninterrupted, up to 7 days | `-p tauhid` (if it exists) |
| Anything > 2 days without an owner partition | `-p normal --qos=long` (CPU only; **`long` is `normal`-only and only for non-owners**) |
| Big HF/dataset download, rsync, staging | `-p service` (2 days, 16 cores, 2 jobs) or a DTN |

Note: `--partition=gpu,owners` is legal Slurm (first available wins), but **watch the documented
whitespace trap**: `#SBATCH --partition=normal, owners` (space after comma) silently kills that and every
following `#SBATCH` line.

### 3.5 Queue limits, priority, backfill

- Submission limits per user and per group; hitting them yields
  `sbatch: error: MaxSubmitJobsPerUser` or `sbatch: error: MaxSubmitJobsPerAccount`.
  `sh_part` shows current limits.
- Priority = job age + requested resources + partition + **fairshare** (recent CPU/GPU-hour consumption
  lowers your priority; it decays over time). Pending reason `Priority` = higher-priority jobs first.
- **Backfill:** small/short jobs slip into gaps ahead of big reservations. Docs: "an accurate time limit
  makes a significant difference."
- Docs' own advice, which matches our workload exactly: **"Use job arrays for many independent tasks
  rather than requesting a large number of nodes at once"**, and pack multiple tasks per array element to
  stay under queue limits:

  ```bash
  #SBATCH --array=0-99:10
  #SBATCH -n 10
  for i in {0..9}; do srun -n 1 ./app $((SLURM_ARRAY_TASK_ID+i)) & done
  wait
  ```

---

## 4. INTERACTIVE SESSIONS

### 4.1 `sh_dev` — the documented recommended entry point

> "`sh_dev` is the recommended starting point for interactive work. It uses sensible defaults, runs on
> dedicated nodes, and typically gives you immediate access without any wait time."

Defaults: **1 core, 4 GB, 1 hour**, partition `dev`, X11 forwarding included (so GUI/plots work).

Full documented usage:

```
$ sh_dev -h
sh_dev: start an interactive shell on a compute node.

Usage: sh_dev [OPTIONS]
    Optional arguments:
        -c      number of CPU cores to request (OpenMP/pthreads, default: 1)
        -g      number of GPUs to request (default: none)
        -n      number of tasks to request (MPI ranks, default: 1)
        -N      number of nodes to request (default: 1)
        -m      memory amount to request (default: 4GB)
        -p      partition to run the job in (default: dev)
        -t      time limit (default: 01:00:00)
        -r      allocate resources from the named reservation (default: none)
        -J      job name (default: sh_dev)
        -q      quality of service to request for the job (default: normal)
```

Documented `dev`-partition limits — **the docs contradict themselves here, verify live**:

- Public-partitions table: `dev` = "2h max, **4 cores + 2 GPUs/user**".
- Footnote on the same page: "The dedicated partition that `sh_dev` uses by default only allows up to
  **2 cores and 8 GB** of memory per user at any given time."

So budget on ~2-4 cores / ~8 GB / 1-2 GPU instances / **2 h hard cap** in `dev`. Exceed it and the job is
rejected unless you pass `-p`.

### 4.2 Interactive GPU

```bash
sh_dev -g 1                                    # instant MIG A30 1g.6gb slice (dev partition)
sh_dev -c 4 -m 8GB -g 1 -p gpu                 # documented example: real GPU via sh_dev
salloc -p gpu --gpus 1                          # documented; then you get a shell on the GPU node
srun -p gpu -c 4 --gres gpu:1 --pty bash        # documented (Apptainer page) alternative
srun -p gpu -G 1 -C GPU_BRD:TESLA nvidia-smi -L # documented one-shot with a constraint
```

Documented `salloc` output shape:

```
$ salloc -p gpu --gpus 1
salloc: job 38068928 queued and waiting for resources
salloc: job 38068928 has been allocated resources
$ nvidia-smi --query-gpu=index,name --format=csv,noheader
0, Tesla V100-SXM2-16GB
```

Note that unconstrained `salloc -p gpu --gpus 1` really does hand you a 16 GB Volta — which is exactly
the wrong GPU for us. Always add the constraint.

**Important `salloc` semantics (documented):** `salloc ./script.sh` runs a script interactively, but
**`#SBATCH` directives inside the script are NOT interpreted by `salloc`** — you must pass every option
on the command line.

### 4.3 Attaching to a running job

- You **cannot** SSH to a compute node without an active job there:
  `Access denied by pam_slurm_adopt: you have no active jobs on this node`.
- Once you *do* have a job on `shXX-XXnXX`, `ssh shXX-XXnXX` works and drops you into the job's cgroup —
  documented caveat: **your SSH session's resource usage counts against the job's limits**, so a fat
  `htop`/python in that shell can OOM-kill your training run.

---

## 5. STORAGE

### 5.1 The filesystems

| Var | Type | Volume quota | Inode quota | Snapshots / backup | Retention | Scope |
|---|---|---:|---:|---|---|---|
| `$HOME` | NFS (`/home/users/uutsha`) | **15 GB** | n/a | yes / off-site replication | forever | cluster-wide, user |
| `$GROUP_HOME` | NFS (`/home/groups/tauhid`) | **1 TB** | n/a | yes / off-site replication | forever | cluster-wide, group |
| `$SCRATCH` | Lustre (`/scratch/users/uutsha`) | **100 TB** | **20 M** | **NO / NO** | **purged after 90 days** | cluster-wide, user |
| `$GROUP_SCRATCH` | Lustre | **100 TB** | **20 M** | **NO / NO** | **purged after 90 days** | cluster-wide, group |
| `$L_SCRATCH` | node-local SSD (`/lscratch/uutsha`) | n/a (~150 GB typical) | n/a | NO / NO | **deleted at job end** | that node only |
| `$OAK` | Lustre, opt-in purchase | as purchased (10 TB increments) | scales w/ volume | optional cloud backup / NO | forever | cluster-wide + external gateways |
| `$COMMON_DATASETS` | read-only shared | — | — | — | — | all users (AlphaFold3, BLAST, Ollama models) |

Global fail-safe caps on `/scratch` (documented): a **user** cannot exceed **125 TB / 25 M inodes** total,
a **group** cannot exceed **300 TB / 60 M inodes** total.

`$SCRATCH` performance: dedicated flash-backed Lustre over IB; the filesystems page says "aggregate
bandwidth of the filesystem is about 75 GB/s", `facts.yml`/tech page says the 14.6 PB `$SCRATCH` delivers
"over 600 GB/s" — the docs disagree; either way it is the fast parallel FS.

`$L_SCRATCH_JOB` is a per-job subdirectory of `$L_SCRATCH` (`/lscratch/uutsha/<jobid>`). `$L_SCRATCH`
itself survives until your *last* job on that node ends; `$L_SCRATCH_JOB` dies with the job.

### 5.2 Checking quotas

```bash
sh_quota                 # everything
sh_quota -f HOME         # one filesystem
sh_quota -f SCRATCH -j   # JSON (scriptable)
sh_quota -g tauhid       # group quotas in the context of a specific group
sh_quota -h
```

Sample documented output:

```
+---------------------------------------------------------------------------+
| Disk usage for user kilian (group: ruthm)                                 |
+---------------------------------------------------------------------------+
|   Filesystem |  volume /   limit                  | inodes /  limit       |
+---------------------------------------------------------------------------+
          HOME |   9.4GB /  15.0GB [||||||     62%] |      - /      - (  -%)
    GROUP_HOME | 562.6GB /   1.0TB [|||||      56%] |      - /      - (  -%)
       SCRATCH |  65.0GB / 100.0TB [            0%] | 143.8K /  20.0M (  0%)
 GROUP_SCRATCH | 172.2GB / 100.0TB [            0%] |  53.4K /  20.0M (  0%)
```

Finding what is eating quota:

```bash
du --human-readable --summarize *      # NB: du skips dotfiles - misses ~/.cache, ~/.conda !
ml system ncdu ; ncdu $HOME            # interactive; use -t 4 on an sh_dev -c 4 for big trees
ncdu --apparent-size $HOME/dir         # logical size vs on-disk
```

### 5.3 The `$SCRATCH` purge policy — read this twice

Documented, and it is unusual:

- Files **not modified in the last 90 days are automatically deleted**.
- "**contents need to change** for a file to be considered modified. The `touch` command does not modify
  file contents and thus does not extend a file's lifetime on the filesystem."
- Metadata changes do **not** reset the clock: reading, renaming, moving, chmod, chown, `touch`.
- Each content modification resets the 90-day countdown.
- Purging is a **continuous background process**, not a scheduled sweep. Example from the docs: a file
  created Feb 1 and never modified becomes eligible May 1 and can vanish any time after.
- The property used is internal and **not visible to users** — `ls` timestamps will lie to you.
- **Empty directory trees devoid of files for > 90 days are also cleaned up.**

Implication for us: an HF snapshot of Llama-3.2-1B or a PyG `Planetoid/Cora` download sitting untouched
on `$SCRATCH` for 90 days is **gone**, and `ls -l` will have shown a recent-looking mtime the whole time.

Mitigations:

- Keep an authoritative copy of anything expensive on `$GROUP_HOME` (1 TB, snapshotted, never purged) or
  `$OAK` if the group has it, and rsync/`dsync` it back to `$SCRATCH` when needed. The docs recommend
  exactly this for `$COMMON_DATASETS`:

  ```bash
  ml system mpifileutils
  srun dsync $GROUP_HOME/tauhid/uutsha/hf_cache $SCRATCH/hf_cache
  ```

- Or write a `service`-partition recurring job that re-touches *contents* — ugly, don't.

### 5.4 Where our things belong

Docs' own recommendation table:

| Use case | Filesystem |
|---|---|
| Personal scripts, config, source code | `$HOME` |
| Group-shared code, shared software installations | `$GROUP_HOME` |
| Active job input/output, large temp files, checkpoints | `$SCRATCH` / `$GROUP_SCRATCH` |
| High-IOPS job I/O, node-local temp | `$L_SCRATCH` |
| Long-term storage, big reference datasets, curated results | `$OAK` |

Concrete mapping for this project:

| Thing | Where | Why |
|---|---|---|
| `BioGlyph` repo / source | `~/BioGlyph` -> `/home/groups/tauhid/ucchwas/BioGlyph` (already correct) | `$GROUP_HOME`, snapshotted, replicated, 1 TB, never purged, shareable |
| **Python venv** | `$GROUP_HOME/tauhid/uutsha/envs/bioglyph` | Docs: "It is best to create them on `$GROUP_HOME` rather than in `$HOME`, to avoid running into space quota limits" (a torch+CUDA venv alone can exceed the 15 GB `$HOME` quota) |
| venv, when running **many concurrent jobs** | copy to `$L_SCRATCH` at job start, or install on `$SCRATCH` | Docs warn explicitly: "running multiple concurrent jobs using the same virtual environment can generate a lot of I/O on filesystems not designed to handle direct I/O from jobs (like `$HOME`, `$GROUP_HOME`, or `$OAK`)" |
| **HF model cache** (`HF_HOME`) | `$SCRATCH/hf` for working copies, **master copy on `$GROUP_HOME`** | tens of GB, must not touch `$HOME` (15 GB!); `$SCRATCH` is fast but 90-day purge |
| **PyG datasets** (Cora, PubMed, ogbn-arxiv, OGB mol) | `$SCRATCH/pyg` (working), master on `$GROUP_HOME` | Cora is tiny, ogbn-arxiv is not; PyG unpacks many small files -> inode pressure |
| Checkpoints, logs, `.out`/`.err` | `$SCRATCH/runs/$SLURM_JOB_ID/` | doc rule: "Store job data on `$SCRATCH`, not `$HOME`" |
| Final results / paper artifacts | `$GROUP_HOME` (or `$OAK`) | survives the purge |
| Apptainer/Enroot images | `$GROUP_HOME/tauhid/uutsha/simg` | docs' explicit recommendation, images are large |

**Never** point `--output`/`--error`, a training loop, or a dataloader at `$HOME` or `$GROUP_HOME`:

> "`$HOME` and `$GROUP_HOME` ... are NFS-based and not designed for large-file or parallel I/O: running
> many concurrent jobs against them degrades performance for all users on the cluster."

### 5.5 `~/.bashrc` hygiene (documented, and it will bite)

Slurm starts a shell on the compute node to build your environment before every job. A slow `~/.bashrc`
causes:

- hanging SSH logins, and
- **jobs held with `(user env retrieval failed requeued held)`**, needing manual
  `scontrol release <jobid>`. A burst of simultaneous job starts can trigger this "even on a healthy
  filesystem".

Documented anti-patterns to avoid in `~/.bashrc`:

- adding external-filesystem paths to `$PATH`,
- sourcing scripts on external filesystems,
- **auto-activating Conda envs** (and the `conda init` block that the installer appends;
  `conda config --set auto_activate_base false` to defuse it).

Test it:

```bash
time bash -i -c true      # "more than a few seconds" == your startup files are too heavy
```

Load modules and activate envs **in the job script**, not in `~/.bashrc`.

---

## 6. SOFTWARE

### 6.1 Lmod modules, and the `math`/`devel` category quirk

Software is **only** reachable through Lmod modules. Modules are grouped into **categories by scientific
field**, and you must load the category before the module is visible:

```bash
module load chemistry gromacs     # category first, then package
```

> **The quirk:** *"The `math` and `devel` categories are loaded by default"* — they are also **sticky
> (`S`)**, so `module purge` will NOT unload them; you need `ml --force purge`. `math` and `devel` give
> you compilers, languages, MPI and numerical libraries directly, with no category load. Everything
> else (`biology`, `chemistry`, `physics`, `system`, `staging`, `viz`, `contribs`) needs its category
> first. `system` in particular holds the low-level tools: `ml system nvtop`, `ml system ncdu`,
> `ml system ruse`, `ml system mpifileutils`, `ml system gcc/14.2.0`.

Cheat sheet:

| Command | Short | Meaning |
|---|---|---|
| `module avail` | `ml av` | list available (in loaded categories only!) |
| `module spider cuda` | `ml spider cuda` | search **all** modules incl. masked ones, and show how to load |
| `module keyword blas` | `ml key blas` | search names + descriptions |
| `module load cuda/12.6.1` | `ml cuda/12.6.1` | load a specific version |
| `module unload gcc` | `ml -gcc` | unload |
| `module swap gcc icc` | `ml -gcc icc` | swap |
| `module purge` | `ml purge` | remove all **except sticky** `math`/`devel` |
| — | `ml --force purge` | remove absolutely everything |
| — | `ml reset` | back to the login default (only `math`+`devel`) |
| `module save foo` / `restore foo` | `ml save/restore foo` | named collections |

Module property flags in `ml av`: `S` sticky, `L` loaded, `D` default, `r` restricted/licensed,
**`g` GPU-accelerated (will only run on GPU nodes)**, `m` MPI.

> **Reproducibility rule from the docs:** always pin the version. "When multiple versions of the same
> module exist, `module` will load the one marked as `Default (D)` ... defaults may evolve over time."

In job scripts, put `module load` right after the `#SBATCH` block. Slurm propagates your login
environment by default, but the docs say load explicitly anyway. `ml reset` first is the doc-shown
pattern.

### 6.2 Finding CUDA / Python versions

```bash
ml spider cuda        # all CUDA versions + how to load
ml spider python
ml spider py-pytorch
ml av                 # after ml system, etc.
```

Versions currently listed in the docs' own `software.yml` (i.e. the `/docs/software/list/` page) —
**verify with `ml spider` on the cluster, this changes**:

| Module | Versions available (docs' software list) |
|---|---|
| `cuda` | 8.0.61, 9.x, 10.x, 11.0.3, 11.1.1, 11.2.0, 11.3.1, 11.4.1, 11.5.0, 11.7.1, **12.0.0, 12.1.1, 12.2.0, 12.4.0, 12.6.1, 12.8.0** |
| `cudnn` | ... 8.9.0.131, 9.0.0.312, 9.4.0, **9.13.1.26, 9.14.0.64** |
| `python` | 2.7.13, 3.6.1, 3.9.0, **3.12.1, 3.14.2** |
| `py-pytorch` | ... 2.0.0_py39, 2.2.1_py312, **2.4.1_py312, 2.9.1_py314** |
| `py-torchvision` | 0.15.1_py39, 0.17.1_py312, 0.19.1_py312, 0.24.1_py314 |
| `nccl` | ... 2.23.4, **2.27.7** |
| `gcc` | 6.3.0 ... 10.3.0, **12.4.0, 14.2.0** |
| `uv` | 0.8.4, 0.9.5, **0.10.8** |
| `py-numpy` | ... 1.26.3_py312, 2.2.6_py312, 2.3.5_py314 |
| `nvtop` | 1.1.0, 2.0.3, 3.0.2 |

Note the naming scheme: `py-<package>/<version>_py<pyver>`, e.g. `py-numpy/1.26.3_py312`. Loading a
`py-*` module auto-loads the matching Python — "No need to load a `python` module explicitly."

There is **no `py-torch-geometric` / `py-transformers` module in the list** — we install those ourselves.

**`python` vs `python3`:** with a `python/3.x` module loaded, the interpreter is **`python3`**. Bare
`python` still resolves to the old system Python 2. (PEP-394 upstream decision, not Sherlock's.)

### 6.3 Conda — actively discouraged

The docs have a whole page titled with a danger admonition:

> **"Avoid using Anaconda on Sherlock. We recommend NOT using Anaconda on Sherlock, and instead consider
> other options like virtual environments or containers."**

Reasons given: installs sub-optimal duplicates of software already provided as modules; binaries not
optimised for Sherlock's CPUs; wrong assumptions about system library locations; **"installs software in
`$HOME` by default, where it writes large amounts of files. A single Anaconda installation can easily
fill up your `$HOME` directory quota"** (remember: 15 GB); can't be relocated; **modifies `~/.bashrc`**
(-> the job-hold problem in 5.5). If you must install it, do it on a compute node via `sh_dev` because
the installer opens too many files for a login node.

Documented conversion recipe: take the `environment.yml`, keep only real PyPI packages and everything
under `pip:`, **drop `cudatoolkit`, `cudnn`, `mkl`, `openblas`, compilers** (load Sherlock modules for
those instead, e.g. `ml cuda/12.6.1`), and `pip install` the rest into a venv.

### 6.4 The documented recommended way to build a Python environment

Two blessed paths.

**(a) `venv` + `pip`:**

```bash
sh_dev -c 4                                  # docs: build envs on a COMPUTE node, not a login node
ml python/3.12.1
python3 -m venv $GROUP_HOME/$USER/envs/bioglyph
source $GROUP_HOME/$USER/envs/bioglyph/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # NOT IN DOCS - PyTorch's own guidance
pip install torch_geometric ogb transformers peft datasets
deactivate
```

Docs' notes: build on a compute node ("Building virtual environments can be resource-intensive");
store on `$GROUP_HOME` not `$HOME`; `rm -rf myenv` to delete. For **`pip --user` installs** (no venv),
packages land in `$HOME/.local/lib/python<ver>/site-packages` — avoid for us, it eats the 15 GB `$HOME`.
Group-shared alternative that the docs describe:
`PYTHONUSERBASE=$GROUP_HOME/python pip install --user <pkg>` plus `PYTHONPATH`/`PATH` exports.

**(b) `uv` (documented and recommended as the modern path):**

```bash
ml python/3.12.1 uv gcc/12.4.0
export UV_PYTHON=$(which python3)      # docs: otherwise uv downloads its own interpreter
uv venv
source .venv/bin/activate
uv pip install torch torch_geometric transformers peft
# or project mode:
uv init --package bioglyph && cd bioglyph && uv add torch transformers
uv run --project /path/to/bioglyph python train.py
```

Docs warn: don't `pip install` into a `uv`-managed `.venv` after activating it — it desynchronises
`.venv` from `pyproject.toml`; use `uv add` / `uv pip`.

**(c) Containers**, when deps are painful: **Apptainer** (formerly Singularity) and **Enroot**
(NVIDIA, OCI/Docker in unprivileged userspace, native Slurm integration via **Pyxis**) are both
available. Notes from the docs:

- `apptainer pull docker://<image>:<tag>` works directly from Docker Hub / NGC.
- **Pull images on a compute node** (`sh_dev -c 4`) — it is multi-core and CPU-heavy. A GPU is *not*
  needed to pull a GPU image, only to run it.
- Store images in `$GROUP_HOME` (`$GROUP_HOME/$USER/simg`).
- Run GPU containers with **`apptainer ... --nv`** inside a GPU job.
- Stale-but-instructive doc warning: NGC containers were built for "supported GPU architectures (TITAN
  Xp, Tesla P40, P100 or V100)" and the docs suggest `#SBATCH -C "GPU_GEN:PSC|GPU_GEN:VLT"`. **For us
  that is exactly backwards** — a modern PyTorch NGC image wants Ampere+; ignore that specific
  constraint and use the Ampere+ constraint from section 2.4.

### 6.5 Internet from compute nodes

**[NOT EXPLICITLY DOCUMENTED — but strongly implied, and effectively yes]:**

- The Apptainer page has you `apptainer pull docker://...` **on a compute node** via `sh_dev`.
- The AI coding-agents page runs Claude Code / Codex / Gemini CLI **on compute nodes** talking to cloud
  APIs, and only warns against running them on *login* nodes.
- The Python page has you `pip install` / `uv add` from PyPI and `git+https://github.com/...` in
  interactive `sh_dev` sessions.

No proxy variables (`http_proxy` etc.) appear anywhere in the docs, and no statement of an egress
firewall exists. Conclusion: **outbound HTTPS from compute nodes works**, so `huggingface_hub` downloads
and PyG dataset downloads should work in-job. **Still design for offline-capable jobs** (pre-stage the
model and datasets, then run with `HF_HUB_OFFLINE=1`), because (a) this is not a documented guarantee,
and (b) N parallel array jobs all racing to download the same Llama snapshot is a self-inflicted
denial-of-service on both the HF endpoint and the filesystem. **Verify on first login:**
`sh_dev -c 2` then `curl -sSI https://huggingface.co | head -1`.

Also: gated Llama weights need an HF token. **Never** put the token in a file in the repo or in
`~/.bashrc`. Use `huggingface-cli login` interactively (it writes `~/.cache/huggingface/token`, mode 600)
or export `HF_TOKEN` in an interactive shell only. Ungated **Qwen** avoids the whole problem for the
sweep phase.

---

## 7. COMPLETE SINGLE-GPU SBATCH TEMPLATE

Save as `~/BioGlyph/slurm/train_gpu.sbatch`. Every line is doc-legal; the `[NOT IN DOCS]` markers flag
the parts that are our engineering choice rather than SRCC guidance.

```bash
#!/bin/bash
#
# ---- Sherlock single-GPU training job: graph-in/graph-out, frozen LLM + graph attention bias ----
# submit with:  sbatch slurm/train_gpu.sbatch
#      array:   sbatch --array=0-6 slurm/train_gpu.sbatch     (encoder sweep E1..E7)
#
# NOTE: all #SBATCH lines MUST be at the top, before any executable line, and MUST NOT
#       contain spaces in their values (a space silently voids this and every later directive).
#
#SBATCH --job-name=bgy-train
#SBATCH --output=/scratch/users/%u/runs/%x.%A_%a.out
#SBATCH --error=/scratch/users/%u/runs/%x.%A_%a.err
#
# --- resources ---
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#
# --- GPU selection: Ampere or newer ONLY.  bf16 + Triton/FlexAttention need CC >= 8.0.
#     The public gpu partition is mostly Volta/Pascal/Turing, which would silently
#     force us onto the eager [B,heads,N,N] path (or fail on bf16 outright).
#     >>> VERIFY THESE STRINGS FIRST:  node_feat -p gpu | grep GPU_SKU  <<<
#SBATCH --constraint="GPU_SKU:H100_SXM5|GPU_SKU:L40S|GPU_SKU:RTX_3090"
#
# --- notifications ---
#SBATCH --mail-type=END,FAIL
#
# --- If (and only if) the 'tauhid' group has invested in Sherlock, prefer:
#       #SBATCH --partition=tauhid           # 7 days, no preemption, no queue
#     or, for the big A100/H100/H200 pool at the price of preemption:
#       #SBATCH --partition=owners
#       #SBATCH --requeue                    # be preemption-safe: checkpoint + resume
#     Find out with: sh_part
#
# --- Do NOT add '#SBATCH --account=...' : Sherlock's docs never use -A / --account.

set -euo pipefail

# ---------------------------------------------------------------- environment
ml reset                                  # back to the login default (math+devel, sticky)
ml load python/3.12.1
ml load cuda/12.6.1                       # verify with: ml spider cuda
ml load cudnn/9.14.0.64                   # verify with: ml spider cudnn

VENV=$GROUP_HOME/$USER/envs/bioglyph      # built once on a compute node; NOT in $HOME (15GB quota)
source "$VENV/bin/activate"

# ---------------------------------------------------------------- caches OFF $HOME
# $HOME is 15 GB and NFS. Never let HF / torch / triton / pip caches land there.
export HF_HOME=$SCRATCH/hf
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$SCRATCH/torch
export TRITON_CACHE_DIR=$L_SCRATCH_JOB/triton        # node-local SSD: fast, dies with the job
export TORCHINDUCTOR_CACHE_DIR=$L_SCRATCH_JOB/inductor
export XDG_CACHE_HOME=$SCRATCH/.cache
export PYG_ROOT=$SCRATCH/pyg                         # our own var, consumed by the code
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PYG_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Pre-staged weights/datasets => no download race across array tasks. [NOT IN DOCS: our choice]
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}           # set to 0 for the one-off staging job
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1                            # docs: Python buffers stdout in batch jobs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK          # docs (ruse section): don't oversubscribe cores
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------------------------------------------------------------- run dir on $SCRATCH
# docs: "Store job data on $SCRATCH, not $HOME"
RUN=$SCRATCH/runs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}
mkdir -p "$RUN"
cd "$HOME/BioGlyph"                                  # -> /home/groups/tauhid/ucchwas/BioGlyph

# ---------------------------------------------------------------- provenance
echo "=== $(date -Is) ==="
echo "job          : $SLURM_JOB_ID  (array task ${SLURM_ARRAY_TASK_ID:-none})"
echo "node         : $SLURMD_NODENAME"
echo "partition    : $SLURM_JOB_PARTITION"
echo "cpus         : $SLURM_CPUS_PER_TASK"
echo "run dir      : $RUN"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
python3 -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda,"bf16",torch.cuda.is_bf16_supported(),"cc",torch.cuda.get_device_capability())'
echo "===================="

# ---------------------------------------------------------------- train
# --requeue-safe: resume from the last checkpoint if the job was preempted (owners partition)
CKPT="$RUN/last.pt"
srun python3 -u -m bioglyph.train \
    --config "configs/e${SLURM_ARRAY_TASK_ID:-1}.yaml" \
    --dataset cora \
    --data-root  "$PYG_ROOT" \
    --model-name meta-llama/Llama-3.2-1B \
    --attn-impl  flex \
    --dtype      bfloat16 \
    --out-dir    "$RUN" \
    --resume-if-exists "$CKPT" \
    --seed       "${SEED:-0}"

# ---------------------------------------------------------------- persist results
# $SCRATCH is purged after 90 days of no CONTENT modification. Copy anything we care about
# to $GROUP_HOME (1 TB, snapshotted, never purged).
KEEP=$GROUP_HOME/$USER/bioglyph-results/${SLURM_JOB_NAME}.${SLURM_JOB_ID}
mkdir -p "$KEEP"
cp -a "$RUN"/metrics.json "$RUN"/config.yaml "$RUN"/best.pt "$KEEP"/ 2>/dev/null || true

echo "=== done $(date -Is) ==="
```

Companion one-off **staging job** (run this once before the sweep; `service` partition, no GPU, has
network, 2-day limit):

```bash
#!/bin/bash
#SBATCH --job-name=bgy-stage
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/%u/runs/%x.%j.out

set -euo pipefail
ml reset; ml load python/3.12.1
source $GROUP_HOME/$USER/envs/bioglyph/bin/activate

export HF_HOME=$SCRATCH/hf
export HF_HUB_OFFLINE=0
mkdir -p "$HF_HOME" "$SCRATCH/pyg"

# gated Llama needs a token: run `huggingface-cli login` interactively BEFOREHAND.
# NEVER hardcode a token in this file.
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
for m in ["meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-1.5B"]:
    try:    print(m, snapshot_download(m))
    except Exception as e: print("SKIP", m, e)
from torch_geometric.datasets import Planetoid
r = os.environ["SCRATCH"] + "/pyg"
for d in ("Cora", "PubMed"):
    Planetoid(root=r, name=d)
PY

# master copy that survives the 90-day purge
mkdir -p $GROUP_HOME/$USER/stage
rsync -a --delete $SCRATCH/hf/  $GROUP_HOME/$USER/stage/hf/
rsync -a --delete $SCRATCH/pyg/ $GROUP_HOME/$USER/stage/pyg/
```

Array-job sweep pattern (docs' own advice for "many independent tasks"):

```bash
sbatch --array=0-6            slurm/train_gpu.sbatch          # encoder sweep E1..E7
sbatch --array=0-23%6         slurm/train_gpu.sbatch          # ablation grid, max 6 running at once
for s in 0 1 2 3 4; do sbatch --export=ALL,SEED=$s slurm/train_gpu.sbatch; done
```

---

## 8. GOTCHAS — the list that gets you warned, held, or purged

**Login nodes**

1. **"Login nodes are not for computing."** Acceptable use is exactly: lightweight file transfers,
   editing scripts/config, job submission and monitoring.
2. **"Resource limits are enforced ... Processes started there will automatically be terminated if their
   resource usage (including CPU time, memory and run time) exceed those limits."** No warning, just SIGKILL.
3. **No compiling on login nodes.** No `conda`/`miniconda` installer on login nodes (too many open files).
   No building venvs on login nodes. No `apptainer pull` on login nodes. No AI coding agents on login
   nodes. All of these are separate, explicit doc warnings — use `sh_dev`.
4. **No SSH to a compute node without a job there** (`pam_slurm_adopt` denies you). And when you do SSH
   in, that shell's memory counts against the job — a fat process there can OOM your training run.

**Scheduler**

5. **`gpu` partition rejects GPU-less jobs** with a misleading QOS-policy error. Always `-G N`.
6. **`#SBATCH` directives must be at the top**, before any non-comment line — later ones are *silently
   ignored*.
7. **A space in a `#SBATCH` value voids that directive and every one after it.**
   `#SBATCH --mem=16 G`, `#SBATCH --partition=normal, owners`, `#SBATCH --job-name=big job` — all silently
   downgrade your job to cluster defaults.
8. **`salloc ./script.sh` ignores the script's `#SBATCH` lines.** Pass options on the command line.
9. **Bad `-C` value = instant rejection**, not a fallback:
   `Requested node configuration is not available`. Verify feature strings with `node_feat`/`sh_node_feat`.
   Constraints also lengthen queue time.
10. **Conflicting GPU options get rejected** (e.g. `--gpus-per-task` + `--cpus-per-gpu` + explicit
    `--cpus-per-task`).
11. **Queue-limit errors** `MaxSubmitJobsPerUser` / `MaxSubmitJobsPerAccount` — the latter is your whole
    group's quota, so you can be blocked by a labmate. Use job arrays and pack tasks.
12. **`--qos=long` is `normal`-only and non-owners-only.** If we are in an owner group we cannot use it
    and don't need it (owner partitions already allow 7 days).
13. **`owners` jobs get killed and requeued** without warning. Checkpoint or keep jobs short.
14. **Fairshare:** heavy recent GPU-hour use lowers our priority for days. A hundred wasteful 2-day
    allocations for 20-minute jobs will visibly hurt the next sweep.

**Storage / filesystem etiquette**

15. **`$SCRATCH` 90-day purge is content-based.** `touch`, rename, chmod, and *reading* do not save a
    file. `ls` timestamps do not tell you the purge clock. Empty dirs are cleaned up too.
16. **`$HOME` is 15 GB.** A single torch+CUDA wheel set, an Anaconda install, or an unredirected
    `~/.cache/huggingface` blows it. `du` hides dotfiles — use `ncdu`.
17. **Never run job I/O against `$HOME` / `$GROUP_HOME` / `$OAK`.** Documented as degrading performance
    "for all users on the cluster" — this is the classic thing that gets a user emailed by SRCC.
18. **Many concurrent jobs sharing one venv on NFS = the same problem.** Copy the venv to
    `$L_SCRATCH` at job start, or keep it on `$SCRATCH`.
19. **`$L_SCRATCH_JOB` is deleted at job end** (and `$L_SCRATCH` when your last job on that node ends).
    Copy results out *before* the script exits.
20. **`~/.bashrc` must stay lightweight** or jobs get held with
    `(user env retrieval failed requeued held)` requiring `scontrol release`. No conda auto-activate, no
    external `$PATH`, no sourcing off slow filesystems. Check with `time bash -i -c true`.
21. **`$SCRATCH`/`$GROUP_SCRATCH` have no snapshots and no backups.** Deleted = gone.
22. **`$COMMON_DATASETS` is read-only and should not be computed against directly** — copy to `$SCRATCH`
    (`dsync` from `ml system mpifileutils`).
23. **External filesystems cannot be mounted on Sherlock.** Push Sherlock data outward (SSHFS from your
    machine to `dtn.sherlock.stanford.edu`), not the other way.
24. **Large transfers belong on the DTNs** (`dtn.sherlock.stanford.edu`), not login nodes. DTNs have no
    interactive shell, and their default destination path differs from the login nodes'.

**Policy / account**

25. **Sherlock is NOT approved for HIPAA / PHI / PII / High Risk data.** Printed in the MOTD. Biomedical
    networks: use de-identified/public data only, or this becomes a compliance incident.
26. **Sherlock is for research, not coursework** or general training sessions.
27. **No permanent services, no root**, even on owned nodes. Long-lived helpers go in the `service`
    partition as documented service jobs.
28. **"Sherlock is a compute cluster, not a storage system."** Don't use it as a backup target.
29. **Account closure:** `$HOME` is archived into `$SCRATCH` (and then purged after 90 days), `$SCRATCH`
    is purged, group spaces untouched. Retrieve data **before** leaving.
30. **Repeated bad passwords => temporary IP ban.** And unmount macOS SSHFS before sleep.

---

## 9. MARLOWE vs SHERLOCK — what differs, and what will break a copied script

Sherlock-side facts are doc-backed. The "if your Marlowe script..." framing is deliberate: **cross-check
each row against the Marlowe notes** rather than trusting anyone's memory of Marlowe.

| # | Area | Sherlock (documented) | What breaks if copied from Marlowe |
|---|---|---|---|
| 1 | **SSH keys** | **Public-key auth is NOT supported.** Password+Duo, or GSSAPI/Kerberos. | Any key-based automation, `authorized_keys` push, CI runner, agent-forwarding deploy, or `ssh -i key` workflow simply cannot work. This is the single biggest structural difference from most clusters. |
| 2 | **Duo per connection** | Every *new* TCP connection prompts Duo unless `ControlMaster` multiplexing is set up. | Scripts that open many separate ssh/scp/rsync connections fire a Duo push per connection. Set up ControlMaster **before** running any loop. |
| 3 | **Slurm account** | `--account` / `-A` **never appears anywhere in the Sherlock docs**. Access is by partition + group membership. | If the Marlowe script has `#SBATCH --account=<project>` or `-A`, remove it (or at minimum verify the account name exists here) — a bogus account is a submission failure. |
| 4 | **GPU request flag** | Canonical is `--gpus`/`-G`. `--gres=gpu:N` also works. **No typed gres** (`gpu:h100:1`) anywhere in the docs. | `--gres=gpu:h100:1` / `--gres=gpu:a100:2` style lines will fail. Convert to `-G N` **plus** `-C GPU_SKU:...`. |
| 5 | **GPU model selection** | Via `-C` node-feature tags (`GPU_SKU`, `GPU_MEM`, `GPU_CC`, `GPU_GEN`, `GPU_BRD`). | Any cluster where you pick GPUs by partition name (`-p h100`) or typed gres needs rewriting. |
| 6 | **GPU homogeneity** | **Heterogeneous fleet: 14 GPU models, 5 generations.** Unconstrained `-p gpu -G 1` can land you on a 12 GB Pascal TITAN Xp or a 16 GB V100. | A script written for a uniform H100 cluster gets wildly variable memory and *no bf16* here. Every GPU job needs an explicit Ampere+ constraint. |
| 7 | **Mandatory explicit GPU request** | `gpu` partition **rejects** jobs that don't request a GPU. | A script relying on a partition's implicit GPU-per-node allocation gets a confusing "violates accounting/QOS policy" error. |
| 8 | **Partition names** | `normal` (default), `dev`, `gpu`, `bigmem`, `service`, `owners`, `<pi_sunetid>`. | `-p batch`, `-p preempt`, `-p h100`, `-p gpu-shared`, project-named partitions: none exist. `sh_part` is the source of truth. |
| 9 | **Walltime caps** | `normal` 2 d (7 d w/ `--qos=long`), `gpu` **2 d**, `bigmem` 1 d, `dev` **2 h**, `service` 2 d, owner partitions 7 d. Defaults are short (1-2 h) if you omit `--time`. | A `--time=7-00:00:00` GPU job is **rejected** in `gpu`. Omitting `--time` gets you a 1 h GPU job, not an unlimited one. |
| 10 | **Per-user GPU cap** | **16 GPUs/user in `gpu`**. | A 32-GPU sweep fan-out silently queues forever / gets rejected. |
| 11 | **Memory syntax & caps** | `--mem` (per node) or `--mem-per-cpu`, never both. In `gpu`: default 8 GB/core, **max 32 GB/core**. | `--mem-per-cpu=64G` is over the `gpu` cap. And note `#SBATCH --mem=64 GB` (with a space) voids all following directives. |
| 12 | **Interactive** | `sh_dev` is the blessed wrapper (`-c -g -n -N -m -p -t -r -J -q`). `salloc`/`srun --pty` also work. | `srun --pty --gres=... bash` still works but ignores `sh_dev`'s sane defaults; also `salloc ./script.sh` **ignores** in-script `#SBATCH`. |
| 13 | **Env / modules** | **Lmod with mandatory categories.** `math`+`devel` preloaded and **sticky**; `system`, `biology`, etc. must be loaded first (`ml system nvtop`). `module purge` won't unload the sticky ones. | `module load nvtop` fails without `ml system`. `module purge` doesn't give a clean slate — use `ml --force purge` or `ml reset`. Also `python` != `python3`. |
| 14 | **Conda** | **Explicitly discouraged**, with a dedicated "avoid Anaconda" page; `venv`/`uv`/containers are the documented path. | A `conda activate` / `environment.yml` / `conda init`-in-bashrc workflow will (a) blow the 15 GB `$HOME` quota, (b) trigger the `(user env retrieval failed requeued held)` job-hold pathology. Convert to venv+`uv`, dropping `cudatoolkit`/`cudnn` in favour of `ml cuda/...`. |
| 15 | **Home directory size** | **`$HOME` = 15 GB**, `$GROUP_HOME` = 1 TB. | Any script that installs envs/caches into `$HOME` (default `pip --user`, default `HF_HOME`, default conda prefix) dies on quota. Redirect `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`. |
| 16 | **Scratch semantics** | `$SCRATCH=/scratch/users/<sunetid>`, 100 TB / 20 M inodes, **90-day content-based purge**, no backups. Also `$GROUP_SCRATCH`, `$L_SCRATCH` (node-local, dies with the job), `$OAK` (opt-in). | Hardcoded `/scratch/<project>/...` or `/projects/...` paths won't exist. A "permanent" scratch dataset directory will be purged. A `touch`-based keepalive cron **does not work here**. |
| 17 | **Path env vars** | `$HOME`, `$GROUP_HOME`, `$SCRATCH`, `$GROUP_SCRATCH`, `$L_SCRATCH`, `$L_SCRATCH_JOB`, `$OAK`, `$COMMON_DATASETS`. Docs insist on using the vars, not literal paths. | Marlowe-specific vars (project/workspace roots, `$PROJECT`, `$WORK`, etc.) are undefined here and expand to empty strings — `cd $PROJECT/foo` becomes `cd /foo`. Always `set -u` to catch this. |
| 18 | **Quota / info tools** | `sh_quota` (`-f -g -n -j`), `sh_part`, `node_feat` / `sh_node_feat`, `seff`, `ruse`. | `quota`, `mmlsquota`, `lfs quota`, or another cluster's helper scripts aren't the right answer here (plain `sinfo`/`scontrol`/`sacct` still work). |
| 19 | **Preemption** | Only in `owners` (and it *is* kill-and-requeue). Public partitions are not preemptible. | If Marlowe has a dedicated preemptible/spot partition, the equivalent here is `-p owners`; conversely a script assuming no preemption must not use `owners`. |
| 20 | **GPU compute mode** | **Exclusive Process by default**; need `--gpu_cmode=shared` to run 2+ processes on one GPU. | MPS / multi-process-per-GPU packing tricks fail with a cryptic CUDA context error unless you add `--gpu_cmode=shared`. |
| 21 | **Containers** | Apptainer **and** Enroot+Pyxis. `apptainer pull docker://...`, run with `--nv`. No Docker daemon, no root. | `docker run`, `podman`, `--privileged`, `sudo` — none available. `srun --container-image=...` (Pyxis) may work; verify against the Enroot page. |
| 22 | **Data transfer host** | `dtn.sherlock.stanford.edu` for bulk (no shell); `login.sherlock.stanford.edu` for small. | A transfer script pointed at a Marlowe DTN / Globus endpoint needs its host swapped; also Sherlock DTNs' default destination path differs. |
| 23 | **Compliance class** | **Not approved for HIPAA / PHI / PII / High Risk.** | If any biomedical network dataset is even arguably PHI, it cannot come here regardless of what Marlowe allows. Check before uploading. |
| 24 | **QOS names** | `long` (normal-only, non-owner-only), `high_p` (owner-partition-only), default `normal`. | Copying a `--qos=<something>` from Marlowe will fail or be silently ineffective. |

---

## 10. FIRST-LOGIN CHECKLIST (run these, in order, and paste the output back)

```bash
# --- 0. do this on the LAPTOP first (WSL), so you get ONE Duo push for the whole session
cat >> ~/.ssh/config <<'EOF'
Host sherlock
    HostName login.sherlock.stanford.edu
    User uutsha
    ControlMaster auto
    ControlPath ~/.ssh/%C
    ControlPersist 4h
    ServerAliveInterval 60
EOF
ssh sherlock

# --- 1. identity, group, paths
id; groups
echo "HOME=$HOME"; echo "GROUP_HOME=$GROUP_HOME"; echo "SCRATCH=$SCRATCH"
echo "GROUP_SCRATCH=$GROUP_SCRATCH"; echo "OAK=${OAK:-<none>}"
ls -ld ~/BioGlyph; readlink -f ~/BioGlyph      # confirm -> /home/groups/tauhid/ucchwas/BioGlyph

# --- 2. what can we submit to? does 'tauhid' own nodes?
sh_part
sh_quota

# --- 3. THE critical one: real GPU constraint values
node_feat -p gpu | grep GPU_        || sh_node_feat -p gpu | grep GPU_
sh_node_feat -p owners | grep GPU_  2>/dev/null
sinfo -p gpu -o '%20N %8c %10m %12G %40f'      # standard Slurm cross-check

# --- 4. module reality check
ml spider cuda | tail -30
ml spider python | tail -20
ml spider py-pytorch | tail -20
ml av 2>&1 | head -60

# --- 5. shell startup health (must be << a few seconds)
time bash -i -c true

# --- 6. compute-node sanity, incl. is there internet?
sh_dev -c 4 -m 16GB -t 1:00:00
  curl -sSI https://huggingface.co | head -1
  curl -sSI https://pypi.org       | head -1
  nvidia-smi -L
  exit

# --- 7. Ampere+ GPU sanity, with the constraint from step 3
srun -p gpu -G 1 -C "GPU_SKU:L40S|GPU_SKU:H100_SXM5|GPU_SKU:RTX_3090" -t 10:00 \
     nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

---

## 11. UNKNOWNS — what the public docs do NOT answer

1. **Does the `tauhid` group actually own nodes on Sherlock?** Not knowable from public docs. There is no
   public per-PI partition list. `sh_part` on first login is the only answer. Everything in this doc about
   `-p tauhid` / `-p owners` / `--qos=high_p` is conditional on that.
2. **The real `GPU_SKU` / `GPU_MEM` / `GPU_CC` strings.** The docs contradict themselves
   (`TESLA_P100_PCIE` vs `P100_PCIE`) and the GPU page's table is explicitly non-exhaustive and visibly
   stale. `node_feat -p gpu | grep GPU_` is authoritative. Also unknown: whether `GPU_CC:8.9` and
   `GPU_CC:9.0` tags exist for L40S/H100, and whether the L40S/H100 SKU tags carry a `TESLA_` prefix.
3. **Which of `node_feat` or `sh_node_feat` is the live command name** — both appear in the docs.
4. **GPU memory per model** is not published by SRCC at all. The table in 2.2 is vendor spec. In
   particular whether Sherlock's V100_SXM2 / V100_PCIE are 16 GB or 32 GB parts, and whether the owners'
   A100s are 40 or 80 GB, is unknown.
5. **Internet egress from compute nodes** is never stated explicitly. Strongly implied by
   `apptainer pull` / `pip install` / cloud-API coding-agent examples all running on compute nodes, but
   not guaranteed, and no proxy config is documented. Verify with `curl`.
6. **`dev`-partition per-user limits are self-contradictory in the docs**: "4 cores + 2 GPUs/user" vs a
   footnote saying "up to 2 cores and 8 GB". Real limit unknown.
7. **`$SCRATCH` aggregate bandwidth**: "about 75 GB/s" (filesystems page) vs "over 600 GB/s"
   (tech/facts). Unknown which is current.
8. **Exact per-user / per-group job submission (queue-depth) limits.** Docs just say "run `sh_part`".
9. **Whether `--account`/`-A` is accepted at all** (never mentioned). Assume not needed; verify only if a
   submission fails oddly.
10. **`--gpu_cmode=shared` interaction with MIG (`dev`) instances** — undocumented.
11. **Whether Enroot/Pyxis `srun --container-image=...` is enabled**, and its exact flags. The Enroot page
    exists but was not read in full here; read
    <https://www.sherlock.stanford.edu/docs/software/containers/enroot/> before choosing containers.
12. **FlexAttention / `torch.compile` / Triton viability per GPU**: no Sherlock doc mentions Triton,
    `torch.compile`, FlexAttention, or the `py-pytorch` build flags. Whether the module-provided
    `py-pytorch/2.4.1_py312` or `2.9.1_py314` ships a working Triton, and whether a pip-installed
    `torch+cu126` is preferable, must be tested in an `sh_dev` session.
13. **The precise host-key fingerprints** (served from a doc include that did not resolve standalone) —
    confirm from the live Connecting page at first login.
14. **`$OAK` availability for group `tauhid`** — opt-in purchase, unknown. `sh_quota` shows an OAK row if
    it exists.
15. **Whether a per-group `contribs` module repo exists for `tauhid`** (`ml contribs; ml av`).
16. **Windows-native OpenSSH `ControlMaster` reliability** — not addressed by the docs; SRCC recommends
    WSL. The WSL recommendation here is inference.
17. **Whether `$SCRATCH` (Lustre) tolerates our many-small-file PyG/HF trees well**, and whether striping
    tuning (`lfs setstripe`) is expected of users — Lustre striping is never mentioned in the docs.

---

## Sources

All read 2026-08-25. Rendered pages plus the docs' own markdown/YAML sources on GitHub.

- <https://www.sherlock.stanford.edu/docs/>
- <https://www.sherlock.stanford.edu/docs/getting-started/> (prerequisites, SSH clients, Windows/WSL, account closure)
- <https://www.sherlock.stanford.edu/docs/getting-started/connecting/> (ssh command, host keys, Duo, IP bans, MOTD)
- <https://www.sherlock.stanford.edu/docs/getting-started/submitting/> (`#SBATCH` syntax traps)
- <https://www.sherlock.stanford.edu/docs/advanced-topics/connection/> (**no public-key auth**, GSSAPI, ControlMaster block, ciphers, compression)
- <https://www.sherlock.stanford.edu/docs/user-guide/running-jobs/> (login-node rules, `sh_dev -h`, `sh_part`, partitions table, `long` QOS, owners + preemption, `high_p`, `ruse`)
- <https://www.sherlock.stanford.edu/docs/user-guide/gpu/> (`-G`, explicit-GPU rule, GPU types, `--gpu_cmode`, advanced GPU options, `nvtop`)
- <https://www.sherlock.stanford.edu/docs/advanced-topics/node-features/> (full feature-tag table, `sh_node_feat`, `-C` boolean syntax, `GPU_MEM:80GB` example)
- <https://www.sherlock.stanford.edu/docs/advanced-topics/submission-options/> (`#SBATCH` reference tables)
- <https://www.sherlock.stanford.edu/docs/advanced-topics/job-management/> (queue limits, `MaxSubmitJobs*`, array packing)
- <https://www.sherlock.stanford.edu/docs/advanced-topics/scheduling/> (backfill, fairshare)
- <https://www.sherlock.stanford.edu/docs/advanced-topics/service-jobs/>
- <https://www.sherlock.stanford.edu/docs/storage/> (quota table, `sh_quota`, where-to-store, `.bashrc` anti-patterns)
- <https://www.sherlock.stanford.edu/docs/storage/filesystems/> (per-FS characteristics, 90-day content-based purge, `$L_SCRATCH_JOB`)
- <https://www.sherlock.stanford.edu/docs/storage/data-transfer/> (DTNs, scp/rsync/sshfs)
- <https://www.sherlock.stanford.edu/docs/storage/common-datasets/> (`$COMMON_DATASETS`, `dsync`)
- <https://www.sherlock.stanford.edu/docs/software/modules/> (Lmod, sticky `math`/`devel`, properties, `ml reset`)
- <https://www.sherlock.stanford.edu/docs/software/install/> (build on compute nodes, `$GROUP_HOME` prefix, "Avoid Anaconda/Conda")
- <https://www.sherlock.stanford.edu/docs/software/using/python/> (pip `--user`, venv, `uv`, `UV_PYTHON`, buffering, env placement warnings)
- <https://www.sherlock.stanford.edu/docs/software/using/anaconda/> (why not conda; environment.yml conversion)
- <https://www.sherlock.stanford.edu/docs/software/containers/apptainer/> (`pull docker://`, `--nv`, `$GROUP_HOME/simg`, `--gres gpu:1` example)
- <https://www.sherlock.stanford.edu/docs/software/containers/enroot/> (listed, not read in full)
- <https://www.sherlock.stanford.edu/docs/software/list/> + `includes/data/software.yml` (cuda/cudnn/python/py-pytorch/uv/gcc versions)
- <https://www.sherlock.stanford.edu/docs/tech/> and <https://www.sherlock.stanford.edu/docs/tech/facts/> + `includes/data/facts.yml` (**exact per-partition node inventory incl. every GPU model**)
- <https://www.sherlock.stanford.edu/docs/concepts/> (investing/owners, preemption, limitations)
- <https://www.sherlock.stanford.edu/docs/user-guide/ondemand/> (<https://ondemand.sherlock.stanford.edu>, Jupyter, VS Code)
- <https://www.sherlock.stanford.edu/docs/user-guide/troubleshoot/> (how to file a ticket: srcc-support@stanford.edu)
- <https://www.sherlock.stanford.edu/docs/software/ai/coding-agents/> (agents on compute nodes, not login nodes)
- Docs source repo: <https://github.com/stanford-rc/www.sherlock.stanford.edu> (`src/docs/**.md`, `includes/data/facts.yml`, `includes/data/software.yml`, `mkdocs.yml` -> `purge_days: 90`)
