#!/bin/bash
# Usage: sbatch --array=0-7 slurm/phase6_decagon.sbatch     (independent per-layer models, 121 of the 963 layers per task)
#   The shared / priors / GAE stages run in slurm/phase6_shared.sbatch. Rows that already exist
#   at this commit are skipped, so a timed-out task is completed by resubmitting the same array.
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p6
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase6/rows" G2L_RUNS="$SCR/runs/phase6"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, scipy, g2l.decagon, g2l.shared, g2l.priors, g2l.run_phase6"
test -f "$PYG_DATA_ROOT/Decagon/processed/decagon.pt" || { echo "Decagon not staged"; exit 1; }
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} on $SLURMD_NODENAME | commit $(git rev-parse --short HEAD)"
python -u -m g2l.run_phase6 --stage independent --index "${SLURM_ARRAY_TASK_ID:-0}" --chunk 121
