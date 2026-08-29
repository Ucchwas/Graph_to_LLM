#!/bin/bash
# Usage: sbatch --array=0-2 slurm/phase8d_final.sbatch
#   Test-split evaluation for the three configurations in results/phase8/SUMMARY.md, and nothing
#   else. Task 0 GIN, task 1 GCN (both reproduced inside our pipeline), task 2 the tuned LGM
#   d=64 dropout=0.5. Three fixed seeds each, identical loader/batcher/optimiser/Evaluator.
#
#   Each task trains on the official scaffold train split, early-stops on validation, restores the
#   best-validation checkpoint, then scores the official test split ONCE. Nothing is selected on
#   test. The validation numbers are re-derived here rather than reused, so val and test come from
#   the same run and the pair is internally consistent.
#
#   `--confirm-gate-passed` records operator authorisation. The Phase-8D validation gate did NOT
#   pass for the LGM (0.7525 vs GIN 0.8315); the user withdrew the stopping rule on 2026-08-29 and
#   asked for the test numbers regardless. That is why this array exists.
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p8dF
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase8d/rows"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.molclass, g2l.ogbnet, g2l.run_molhiv"
T=${SLURM_ARRAY_TASK_ID:-0}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $T on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
case $T in
  0) python -u -m g2l.run_molhiv --stage final --model gin --confirm-gate-passed ;;
  1) python -u -m g2l.run_molhiv --stage final --model gcn --confirm-gate-passed ;;
  2) python -u -m g2l.run_molhiv --stage final --d 64 --dropout 0.5 --confirm-gate-passed ;;
esac
