#!/bin/bash
# Usage: sbatch --array=0-44 slurm/phase2_grid.sbatch d1_grid   (task count from --list)
#SBATCH --job-name=g2l-p2
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
STAGE=${1:?stage name required}
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" HF_HOME="$SCR/hf" TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdgcache" TRITON_CACHE_DIR="$SCR/triton"
export TORCHINDUCTOR_CACHE_DIR="$SCR/inductor" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase2/rows" G2L_RUNS="$SCR/runs/phase2"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
REV=$(python -c "import yaml;print(yaml.safe_load(open('configs/phase2.yaml'))['revision'])")
test -d "$HF_HOME/hub/models--meta-llama--Llama-3.2-1B/snapshots/$REV" || { echo "weights not staged at $REV"; exit 1; }
python -c "import g2l.llm, g2l.train, g2l.run_phase2"
echo "job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID stage $STAGE on $SLURMD_NODENAME | commit $(git rev-parse --short HEAD)"
python -u -m g2l.run_phase2 --stage "$STAGE" --index "$SLURM_ARRAY_TASK_ID"
