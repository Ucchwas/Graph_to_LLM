#!/bin/bash
# Usage (after selection, winners passed via --export):
#   sbatch --array=0-5 --export=ALL,G2L_COMMIT=...,LGM_HEAD=...,GIN_INIT=...,GCN_INIT=... \
#          slurm/phase8g_final.sbatch
#   Phase 8G stage 3: the frozen configurations, 10 seeds each per OGB's submission rules, test
#   read ONCE per seed at the best-validation checkpoint. Tasks 0-1 LGM (seeds 0-4 / 5-9),
#   2-3 GIN, 4-5 GCN. Winners were chosen on validation in stage 2 and enter via env, so this
#   file contains no result-dependent editing.
#SBATCH --job-name=g2l-p8gF
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
S=("0,1,2,3,4" "5,6,7,8,9")
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $T on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
echo "winners: LGM_HEAD=${LGM_HEAD:?} GIN_INIT=${GIN_INIT:?} GCN_INIT=${GCN_INIT:?}"
case $T in
  0|1) python -u -m g2l.run_phase8g --stage final --model lgm --head "$LGM_HEAD" \
         --seeds "${S[$T]}" --confirm-final ;;
  2|3) python -u -m g2l.run_phase8g --stage final --model gin --init "$GIN_INIT" \
         --seeds "${S[$((T - 2))]}" --confirm-final ;;
  4|5) python -u -m g2l.run_phase8g --stage final --model gcn --init "$GCN_INIT" \
         --seeds "${S[$((T - 4))]}" --confirm-final ;;
esac
