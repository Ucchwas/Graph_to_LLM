#!/bin/bash
# Usage: sbatch slurm/phase8f_pretrain.sbatch
#   Stage 1 of Phase 8F. Masked-attribute self-supervision on the official MolHIV TRAINING
#   molecules ONLY -- no external data and no extra graphs, so the later comparison against a
#   scratch model stays clean. Hides 15% of atom and 15% of bond ATTRIBUTES (never the topology)
#   and predicts them back: atoms off the node states, bonds off the edge states.
#   One task, three seeds, three body checkpoints. Stage 2 depends on this completing.
#SBATCH --job-name=g2l-p8fP
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%j.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase8f/rows" G2L_PRETRAIN="$SCR/results/runs/phase8f"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.molclass, g2l.molpretrain, g2l.run_phase8f"
echo "job ${SLURM_JOB_ID} on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
python -u -m g2l.run_phase8f --stage pretrain
