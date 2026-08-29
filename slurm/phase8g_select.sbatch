#!/bin/bash
# Usage: sbatch --array=0-5 --dependency=afterok:<pretrain job> slurm/phase8g_select.sbatch
#   Phase 8G stage 2: the ONE open knob per family, decided on VALIDATION over 3 paired seeds.
#     0-1: LGM head, linear vs MLP (readout=nodeedge and init=pretrained fixed by Phase 8F)
#     2-3: GIN, scratch vs pretrained     4-5: GCN, scratch vs pretrained
#   The test split is untouched; --stage select cannot reach it. The final 10-seed stage is
#   submitted separately once these rows pick each family's winner.
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p8gS
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
export G2L_ROWS="$SCR/results/phase8g/rows" G2L_PRETRAIN="$SCR/results/runs/phase8g"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.molclass, g2l.molpretrain, g2l.ogbnet, g2l.run_phase8g"
T=${SLURM_ARRAY_TASK_ID:-0}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $T on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
case $T in
  0) python -u -m g2l.run_phase8g --stage select --model lgm --head linear ;;
  1) python -u -m g2l.run_phase8g --stage select --model lgm --head mlp ;;
  2) python -u -m g2l.run_phase8g --stage select --model gin --init scratch ;;
  3) python -u -m g2l.run_phase8g --stage select --model gin --init pretrained ;;
  4) python -u -m g2l.run_phase8g --stage select --model gcn --init scratch ;;
  5) python -u -m g2l.run_phase8g --stage select --model gcn --init pretrained ;;
esac
