#!/bin/bash
# Usage: sbatch --array=0-7 slurm/phase3b_grid.sbatch lora 33       (260-run Phase-3B main stage)
#        sbatch slurm/phase3b_grid.sbatch probe                      (LoRA probe: step time, memory, gradients)
#   task i runs run-list indices i*CHUNK .. i*CHUNK+CHUNK-1 sequentially (count from --list).
#   QOS medium allows 32 submitted jobs per ACCOUNT (shared with teammates): keep arrays small.
#SBATCH --job-name=g2l-p3b
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
STAGE=${1:?stage name required}
CHUNK=${2:-1}
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" HF_HOME="$SCR/hf" TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdgcache" TRITON_CACHE_DIR="$SCR/triton"
export TORCHINDUCTOR_CACHE_DIR="$SCR/inductor" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase3b/rows" G2L_RUNS="$SCR/runs/phase3b"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
CFG=configs/phase3b.yaml
REV=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['revision'])")
test -d "$HF_HOME/hub/models--meta-llama--Llama-3.2-1B/snapshots/$REV" || { echo "weights not staged at $REV"; exit 1; }
python -c "import peft, g2l.llm, g2l.train, g2l.run_phase3"
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} stage $STAGE chunk $CHUNK on $SLURMD_NODENAME | commit $(git rev-parse --short HEAD)"
if [ "$STAGE" = probe ]; then
  python -u -m g2l.run_phase3 --config "$CFG" --probe
else
  python -u -m g2l.run_phase3 --config "$CFG" --stage "$STAGE" --index "${SLURM_ARRAY_TASK_ID:-0}" --chunk "$CHUNK"
fi
