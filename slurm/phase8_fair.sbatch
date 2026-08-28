#!/bin/bash
# Usage: sbatch --array=0-1 slurm/phase8_fair.sbatch    (task 0 = lgm, task 1 = edgegcn)
#   The corrected architecture comparison. Job 452985's GCN arm was fairly configured but used
#   torch_geometric GCNConv, which structurally cannot consume edge attributes, so its gap to the
#   LGM confounded architecture with edge access. `edgegcn` removes that single unfairness: an
#   OGB-style edge-aware GCN whose neighbour messages carry all three bond attributes. Everything
#   else is held identical -- node featuriser, edge encoder, decoder, split, masks, seed, budget.
#   Both arms train from scratch. Rows go to a SEPARATE directory so 452985's provenance stands.
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p8fair
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase8fair/rows" G2L_RUNS="$SCR/runs/phase8fair"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.multigraph, g2l.run_phase8"
ARMS=(lgm edgegcn)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} arm $ARM on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
python -u -m g2l.run_phase8 --arm "$ARM"
