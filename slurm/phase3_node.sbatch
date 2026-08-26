#!/bin/bash
# Usage: sbatch --array=0-3 --time=08:00:00 slurm/phase3_node.sbatch main 15 4
#   GPUS runs in parallel per task, one process per allocated GPU; task i, GPU g runs
#   run-list chunk i*GPUS+g (same chunk semantics as phase3_grid.sbatch, so 4 tasks x 4 GPUs
#   x 15 runs cover the 240-run main stage). Fewer array elements against the 32-job cap.
#SBATCH --job-name=g2l-p3
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
STAGE=${1:?stage name required}
CHUNK=${2:-1}
GPUS=${3:-4}
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" HF_HOME="$SCR/hf" TORCH_HOME="$SCR/torch"
export XDG_CACHE_HOME="$SCR/xdgcache" TRITON_CACHE_DIR="$SCR/triton"
export TORCHINDUCTOR_CACHE_DIR="$SCR/inductor" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase3/rows" G2L_RUNS="$SCR/runs/phase3"
export OMP_NUM_THREADS=8
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
REV=$(python -c "import yaml;print(yaml.safe_load(open('configs/phase3.yaml'))['revision'])")
test -d "$HF_HOME/hub/models--meta-llama--Llama-3.2-1B/snapshots/$REV" || { echo "weights not staged at $REV"; exit 1; }
python -c "import g2l.llm, g2l.train, g2l.run_phase3"
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:?no GPUs allocated}"
echo "job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID stage $STAGE chunk $CHUNK gpus ${DEVS[*]} on $SLURMD_NODENAME | commit $(git rev-parse --short HEAD)"
pids=()
for g in $(seq 0 $((GPUS - 1))); do
  idx=$((SLURM_ARRAY_TASK_ID * GPUS + g))
  CUDA_VISIBLE_DEVICES="${DEVS[$g]}" python -u -m g2l.run_phase3 --stage "$STAGE" --index "$idx" --chunk "$CHUNK" \
    > "$SCR/logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.g$g.out" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit $fail
