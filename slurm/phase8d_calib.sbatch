#!/bin/bash
# Usage: sbatch --array=0-10 slurm/phase8d_calib.sbatch
#   Bounded ogbg-molhiv calibration. Task 0-1 reproduce OGB's GIN / GCN inside our pipeline;
#   tasks 2-10 sweep LGM capacity x regularisation. Every task runs the three fixed seeds.
#   NO task in this array evaluates the test split -- `--stage tune` and `--stage baseline` cannot
#   reach it. The test set is read once, later, by an explicit `--stage final --confirm-gate-passed`
#   run, and only if a frozen config matched the reproduced baseline on validation.
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p8d
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
  0) python -u -m g2l.run_molhiv --stage baseline --model gin ;;
  1) python -u -m g2l.run_molhiv --stage baseline --model gcn ;;
  *)
    DS=(64 64 64 128 128 128 256 256 256)
    PS=(0.0 0.3 0.5 0.0 0.3 0.5 0.0 0.3 0.5)
    I=$((T - 2))
    python -u -m g2l.run_molhiv --stage tune --d "${DS[$I]}" --dropout "${PS[$I]}"
    ;;
esac
