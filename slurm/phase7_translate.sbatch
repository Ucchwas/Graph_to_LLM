#!/bin/bash
# Usage: sbatch --array=0-10 slurm/phase7_translate.sbatch    (task i = fold i = one held-out cancer)
#   Each task trains one shared model per density in configs/phase7.yaml on the other cancers and
#   scores the held-out one. Rows already written at this commit are skipped.
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p7
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
export G2L_PHASE7_ROOT="$SCR/phase7" G2L_ROWS="$SCR/results/phase7/rows" G2L_RUNS="$SCR/runs/phase7"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, sklearn, g2l.tcga, g2l.translate, g2l.run_phase7"
ls "$G2L_PHASE7_ROOT"/processed/pairs_rho*.pt >/dev/null || { echo "Phase-7 graph pairs not staged"; exit 1; }
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} on $SLURMD_NODENAME | commit $(git rev-parse --short HEAD)"
python -u -m g2l.run_phase7 --stage translate --index "${SLURM_ARRAY_TASK_ID:-0}"
