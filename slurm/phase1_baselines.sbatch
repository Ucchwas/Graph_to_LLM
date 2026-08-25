#!/bin/bash
#SBATCH --job-name=g2l-p1-base
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%j.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" HF_HOME="$SCR/hf" TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdgcache" TRITON_CACHE_DIR="$SCR/triton"
export TORCHINDUCTOR_CACHE_DIR="$SCR/inductor" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
echo "job $SLURM_JOB_ID on $SLURMD_NODENAME | commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'| device',torch.cuda.get_device_name(0),'| bf16',torch.cuda.is_bf16_supported())"
python -u -m baselines.run_phase1
