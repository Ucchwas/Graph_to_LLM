#!/bin/bash
# Usage: sbatch --array=0-1 slurm/phase8c_molhiv.sbatch    (task 0 = atom, task 1 = noatom)
#   Phase 8C: ogbg-molhiv graph classification on the OFFICIAL scaffold split with the official
#   Evaluator -- the one Phase-8 task with an external reference point. Task 0 uses the nine atom
#   features and is comparable to the OGB leaderboard; task 1 disables atom features only and is an
#   ablation. One seed each, selection on validation ROC-AUC, test scored once.
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p8c
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
export G2L_ROWS="$SCR/results/phase8/rows" G2L_RUNS="$SCR/runs/phase8"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.molclass, g2l.run_molhiv"
ARMS=(atom noatom)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} arm $ARM on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
python -u -m g2l.run_molhiv --arm "$ARM"
