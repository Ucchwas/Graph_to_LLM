#!/bin/bash
# Usage: sbatch --array=0-1 slurm/phase8_gate.sbatch     (task 0 = lgm, task 1 = the no-edge control)
#   Gate 8.2: train ONE checkpoint on ogbg-molhiv graphs with N <= train_max_n, then score held-out
#   graphs at sizes it saw AND at sizes it never saw. The two arms share a parameter count exactly
#   (`edges=False` empties the edge states at run time and changes no shape), so any gap between
#   them is the edge channel, not capacity.
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p8gate
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase8/rows" G2L_RUNS="$SCR/runs/phase8"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, sklearn, ogb, g2l.lgm, g2l.multigraph, g2l.run_phase8"
ARMS=(lgm noedge)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
# The checkout is pinned for the phase and Phase-8 code is shipped by scp, so the recorded commit
# comes from $G2L_COMMIT (exported by the submitting shell), not from the checkout's HEAD.
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} arm $ARM on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset} | checkout $(git rev-parse --short HEAD)"
python -u -m g2l.run_phase8 --arm "$ARM"
