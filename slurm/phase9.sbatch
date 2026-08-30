#!/bin/bash
# Usage: sbatch --array=0-10 --export=ALL,G2L_COMMIT=<hash> slurm/phase9.sbatch
#   Phase 9: TCGA normal -> tumour translation, one held-out cancer per task. Inside each task:
#   3 seeds x {LGM, capacity-matched edge-aware GCN}, each = SSL on the training cancers' graphs
#   -> fine-tune with selection on the validation cancer -> the test cancer scored once.
#   Data: $G2L_PHASE9_ROOT/tcga9.pt, built on the laptop from Phase 7's expression matrix and
#   sample lists (g2l.run_phase9 --stage build) and shipped -- the 300 MB expression file is not
#   needed here. QOS medium: 32 submitted jobs per ACCOUNT, shared -- this array is 11.
#SBATCH --job-name=g2l-p9
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
export G2L_PHASE9_ROOT="$SCR/phase9" G2L_ROWS="$SCR/results/phase9/rows"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
ls "$G2L_PHASE9_ROOT/tcga9.pt" >/dev/null || { echo "tcga9.pt not staged"; exit 1; }
python -c "import torch_geometric, g2l.lgm, g2l.multigraph, g2l.tcga9, g2l.run_phase9, g2l.aggregate9"
T=${SLURM_ARRAY_TASK_ID:-0}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $T on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
python -u -m g2l.run_phase9 --stage run --fold "$T"
