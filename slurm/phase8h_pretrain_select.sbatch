#!/bin/bash
# Phase 8H stage 1 of 5: SSL (100 epochs, training split only) for the SELECTION seeds, all
# families. Tasks: 0-2 LGM d=64/128/256, 3 GIN, 4 GCN. Identical loop for everyone.
# Usage: sbatch --array=0-4 --export=ALL,G2L_COMMIT=<hash> slurm/phase8h_pretrain_select.sbatch
#SBATCH --job-name=g2l-p8hP1
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase8h/rows" G2L_PRETRAIN="$SCR/results/runs/phase8h"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.molclass, g2l.molpretrain, g2l.ogbnet, g2l.run_phase8g, g2l.launch8h"
T=${SLURM_ARRAY_TASK_ID:-0}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $T on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
R="python -u -m g2l.run_phase8g --config configs/phase8h.yaml"
case $T in
  0) $R --stage pretrain --model lgm --d 64  --seeds 0,1,2 ;;
  1) $R --stage pretrain --model lgm --d 128 --seeds 0,1,2 ;;
  2) $R --stage pretrain --model lgm --d 256 --seeds 0,1,2 ;;
  3) $R --stage pretrain --model gin --seeds 0,1,2 ;;
  4) $R --stage pretrain --model gcn --seeds 0,1,2 ;;
esac
