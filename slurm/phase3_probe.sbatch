#!/bin/bash
# Usage: sbatch slurm/phase3_probe.sbatch   (read $SCR/results/phase3/rows/probe.json before the array)
#SBATCH --job-name=g2l-p3-probe
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%j.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" HF_HOME="$SCR/hf" TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdgcache" TRITON_CACHE_DIR="$SCR/triton"
export TORCHINDUCTOR_CACHE_DIR="$SCR/inductor" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase3/rows" G2L_RUNS="$SCR/runs/phase3"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
echo "probe job $SLURM_JOB_ID on $SLURMD_NODENAME | commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -u -m g2l.run_phase3 --probe
