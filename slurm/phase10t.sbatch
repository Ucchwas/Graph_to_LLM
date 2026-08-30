#!/bin/bash
# Usage: sbatch --array=0-10 --export=ALL,G2L_COMMIT=<hash> slurm/phase10t.sbatch
#   Phase 10 TCGA: normal -> tumour translation, one held-out cancer per task. Inside each task:
#   5 seeds x 4 bodies {LGM, LGM+RRWP, edge-aware GCN, edge-aware GCN+RWSE}, each = SSL on the
#   training cancers' graphs -> fine-tune with selection on the validation cancer -> the test
#   cancer scored once. Phase 9's protocol verbatim; the bodies come from configs/phase10_tcga.yaml.
#SBATCH --job-name=g2l-p10t
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_PHASE9_ROOT="$SCR/phase9" G2L_ROWS="$SCR/results/phase10/tcga/rows"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
ls "$G2L_PHASE9_ROOT/tcga9.pt" >/dev/null || { echo "tcga9.pt not staged"; exit 1; }
python -c "import torch_geometric, g2l.rrwp, g2l.lgm, g2l.multigraph, g2l.tcga9, g2l.run_phase9, g2l.aggregate10"
T=${SLURM_ARRAY_TASK_ID:-0}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $T on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
python -u -m g2l.run_phase9 --config configs/phase10_tcga.yaml --stage run --fold "$T"
