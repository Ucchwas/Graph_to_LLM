#!/bin/bash
# Phase 8H stage 5 of 5: the frozen configurations, 10 seeds each (OGB's rule), train -> restore
# best-validation checkpoint -> read the TEST split once per seed. Tasks 0-4 LGM (2 seeds each),
# 5-6 GIN (5 each), 7-8 GCN (5 each). Submitted by stage 3 with the LGM width in LGM_D.
#SBATCH --job-name=g2l-p8hF
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
L=("0,1" "2,3" "4,5" "6,7" "8,9")
G=("0,1,2,3,4" "5,6,7,8,9")
case $T in
  0|1|2|3|4) $R --stage final --model lgm --d "$LGM_D" --head linear --seeds "${L[$T]}" --confirm-final ;;
  5|6)       $R --stage final --model gin --init pretrained --seeds "${G[$((T - 5))]}" --confirm-final ;;
  7|8)       $R --stage final --model gcn --init pretrained --seeds "${G[$((T - 7))]}" --confirm-final ;;
esac
