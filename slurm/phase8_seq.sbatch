#!/bin/bash
# Usage: sbatch --array=0-1 slurm/phase8_seq.sbatch    (task 0 = lgmseq, task 1 = lgm control)
#   Tests the sequential edge-then-node update. In the default layer both updates read the same
#   input states, so topological information travels node->edge in one layer and edge->node in the
#   next -- two layers per hop. Sequential mode updates edges first and lets the node update read
#   them, so h_i absorbs h_j within one layer, matching a GCN's propagation rate at equal depth.
#   Task 1 re-runs the parallel LGM at THIS commit as a control, so the comparison cannot be
#   confounded by any drift since job 453058 (which measured lgm 0.9246 / edgegcn 0.9422).
#   Rows go to their own directory. QOS medium: 32 submitted jobs per ACCOUNT, keep arrays small.
#SBATCH --job-name=g2l-p8seq
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
export G2L_ROWS="$SCR/results/phase8seq/rows" G2L_RUNS="$SCR/runs/phase8seq"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.multigraph, g2l.run_phase8"
ARMS=(lgmseq lgm)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} arm $ARM on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
python -u -m g2l.run_phase8 --arm "$ARM"
