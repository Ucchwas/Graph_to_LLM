#!/bin/bash
#SBATCH --job-name=g2l-scratch-sweep
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000211/%u/logs/%x-%j.out

# First Marlowe job: reproduces the locally-validated Phase-1 scratch sweep.
# Prereqs (login node, once): venv at /projects/m000211/envs/g2l, repo cloned at
# /projects/m000211/Graph_to_LLM, Cora pre-downloaded. See docs/hpc/marlowe.md.
set -euo pipefail
module purge && module load slurm mps

SCR=/scratch/m000211/$USER
export HF_HOME="$SCR/hf" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYG_DATA_ROOT=/projects/m000211/data/pyg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "$SCR/logs"

source /projects/m000211/envs/g2l/bin/activate
cd /projects/m000211/Graph_to_LLM
echo "job $SLURM_JOB_ID on $SLURMD_NODENAME  commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun python -u -m baselines.run_scratch_sweep
