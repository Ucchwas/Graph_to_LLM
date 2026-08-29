#!/bin/bash
# Usage: sbatch --array=0-2 slurm/phase8g_pretrain.sbatch
#   Phase 8G stage 1: masked-attribute self-supervision on the official MolHIV TRAINING split
#   only, run IDENTICALLY for all three families (same loop, same masks, same schedule, same
#   corpus -- g2l/molpretrain.py). Task 0 LGM, 1 GIN, 2 GCN; 10 seeds each -> 10 checkpoints.
#   No task here or in stage 2 touches the test split.
#SBATCH --job-name=g2l-p8gP
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase8g/rows" G2L_PRETRAIN="$SCR/results/runs/phase8g"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.molclass, g2l.molpretrain, g2l.ogbnet, g2l.run_phase8g"
T=${SLURM_ARRAY_TASK_ID:-0}
M=(lgm gin gcn)
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $T on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
python -u -m g2l.run_phase8g --stage pretrain --model "${M[$T]}"
