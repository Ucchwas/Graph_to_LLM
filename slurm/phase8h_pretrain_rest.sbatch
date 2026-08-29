#!/bin/bash
# Phase 8H stage 4 of 5: SSL for the remaining seeds 3-9, only at the LGM width that won.
# Tasks 0-3 LGM (seeds 3,4 / 5,6 / 7,8 / 9), 4 GIN (3-9), 5 GCN (3-9). Submitted by stage 3.
#SBATCH --job-name=g2l-p8hP2
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
echo "LGM_D=${LGM_D:?}"
case $T in
  0) $R --stage pretrain --model lgm --d "$LGM_D" --seeds 3,4 ;;
  1) $R --stage pretrain --model lgm --d "$LGM_D" --seeds 5,6 ;;
  2) $R --stage pretrain --model lgm --d "$LGM_D" --seeds 7,8 ;;
  3) $R --stage pretrain --model lgm --d "$LGM_D" --seeds 9 ;;
  4) $R --stage pretrain --model gin --seeds 3,4,5,6,7,8,9 ;;
  5) $R --stage pretrain --model gcn --seeds 3,4,5,6,7,8,9 ;;
esac
