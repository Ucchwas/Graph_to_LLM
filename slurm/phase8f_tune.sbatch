#!/bin/bash
# Usage: sbatch --array=0-3 --dependency=afterok:<pretrain job> slurm/phase8f_tune.sbatch
#   Stage 2 of Phase 8F: a 2x2 over three paired seeds, VALIDATION ONLY. The test split is not
#   reachable from g2l/run_phase8f.py at all.
#     readout  node     = the Phase-8C/8D behaviour, which discards the final edge states
#              nodeedge = [mean(H_nodes) || mean(H_edges)] -> MLP
#     init     scratch / pretrained (stage 1's body, loaded strict=True)
#   The head is an MLP in EVERY arm, so the readout comparison isolates the edge states rather
#   than head depth. Capacity and dropout are fixed at the Phase-8D winner, d=64 / 0.5.
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p8fT
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
export G2L_ROWS="$SCR/results/phase8f/rows" G2L_PRETRAIN="$SCR/results/runs/phase8f"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.molclass, g2l.molpretrain, g2l.run_phase8f"
T=${SLURM_ARRAY_TASK_ID:-0}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $T on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
case $T in
  0) python -u -m g2l.run_phase8f --stage tune --readout node     --init scratch ;;
  1) python -u -m g2l.run_phase8f --stage tune --readout nodeedge --init scratch ;;
  2) python -u -m g2l.run_phase8f --stage tune --readout node     --init pretrained ;;
  3) python -u -m g2l.run_phase8f --stage tune --readout nodeedge --init pretrained ;;
esac
