#!/bin/bash
# Usage: sbatch slurm/phase6_shared.sbatch      (one GPU: the shared model over the 770 train layers,
#   then the priors on all 963 layers, then featureless GAE on the 97 test layers)
#   Kept apart from the per-layer array so the long stage gets its own walltime and the short
#   chunks stay backfillable. Every stage skips rows that already exist at this commit, so a
#   timed-out run is completed by resubmitting the same script.
#SBATCH --job-name=g2l-p6s
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%j.out
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
echo "job ${SLURM_JOB_ID} on $SLURMD_NODENAME | commit $(git rev-parse --short HEAD)"
python -u -m g2l.run_phase6 --stage shared
python -u -m g2l.run_phase6 --stage priors
python -u -m g2l.run_phase6 --stage gae
