#!/bin/bash
# Phase 10 MolHIV stage 5 of 5: the frozen configuration per arm, 10 seeds (OGB rule),
# train -> restore the best-validation checkpoint -> read the TEST split ONCE per seed.
# 12 tasks: 5 seeds per task. Submitted by stage 3 with a dependency on stage 4.
#SBATCH --job-name=g2l-p10mF
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
#SBATCH --time=16:00:00
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase10/molhiv/rows" G2L_PRETRAIN="$SCR/results/runs/phase10"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.rrwp, g2l.lgm, g2l.molclass, g2l.molpretrain, g2l.ogbnet, g2l.run_phase10, g2l.launch10"
T=${SLURM_ARRAY_TASK_ID:-0}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $T on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
ARMS=(lgm lgm_rrwp gin gin_rwse gcn gcn_rwse)
LGMW=(64 128 256)
OGBW=(100 200 300)
R="python -u -m g2l.run_phase10 --config configs/phase10_molhiv.yaml"
A=${ARMS[$((T / 2))]}
S=("0,1,2,3,4" "5,6,7,8,9")
echo "arm=$A seeds=${S[$((T % 2))]}"
$R --stage final --arm "$A" --d best --seeds "${S[$((T % 2))]}" --confirm-final
