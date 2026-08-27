#!/bin/bash
# Usage: sbatch --array=0-3 slurm/phase5_smoke.sbatch     (task i = dataset i of configs/phase5.yaml: cora, ddi, photo, ohmnet)
#   Each task runs its dataset's direct GCN and every baseline sequentially (row per run under $G2L_ROWS).
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p5
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase5/rows" G2L_RUNS="$SCR/runs/phase5"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import ogb, torch_geometric, g2l.datasets, g2l.run_phase5, baselines.maskgae"
for d in Cora/processed Photo/processed ogbl_ddi/processed PPT-Ohmnet/processed; do test -d "$PYG_DATA_ROOT/$d" || { echo "dataset not staged: $d"; exit 1; }; done
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} on $SLURMD_NODENAME | commit $(git rev-parse --short HEAD)"
python -u -m g2l.run_phase5 --config configs/phase5.yaml --index "${SLURM_ARRAY_TASK_ID:-0}"
