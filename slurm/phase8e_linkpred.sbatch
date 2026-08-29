#!/bin/bash
# Usage: sbatch --array=0-3 slurm/phase8e_linkpred.sbatch
#   Phase 8E, baseline track: can the LGM beat the edge-aware GCN's 0.9422 unseen-size AUROC?
#   Task 0-2 are the three attention-balance modes with the sequential update on; task 3 re-runs
#   edgegcn at three seeds, because the 0.9422 reference is a single seed and the GPU-noise floor
#   is ~0.002 while the gap to close is 0.012 -- a one-seed comparison could not settle it.
#   Every task runs seeds 0,1,2. QOS medium: 32 jobs per ACCOUNT, keep arrays small.
#SBATCH --job-name=g2l-p8elp
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase8e_lp/rows" G2L_RUNS="$SCR/runs/phase8e_lp"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.multigraph, g2l.run_phase8"
ARMS=(lgmseq lgmcount lgmsplit edgegcn)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} arm $ARM on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
for S in 0 1 2; do
  echo "=== $ARM seed $S ==="
  python -u -m g2l.run_phase8 --arm "$ARM" --seed "$S"
done
