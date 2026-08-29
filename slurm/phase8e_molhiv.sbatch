#!/bin/bash
# Usage: sbatch --array=0-6 slurm/phase8e_molhiv.sbatch
#   Phase 8E, leaderboard track. Two changes over Phase 8D, both motivated by measurement:
#     (1) warm-up + gradient clipping for the LGM. Phase 8D gave it plain Adam to match OGB's
#         reference recipe -- correct for the baselines, which reproduced faithfully, but it denied
#         the transformer its standard recipe and showed the symptom (LGM seed spread +-0.041
#         against GIN's +-0.0044). Task 0 is a `gin_warm` control that gives the BASELINE the same
#         treatment, so the asymmetry is measured rather than assumed.
#     (2) the attention-balance modes. Under `joint` the structural channel gets ~7% of a node's
#         attention mass because it competes on key count (~2 edge keys vs ~26 node keys).
#         `count` offsets each block by -log(its key count) -- still ONE softmax, architecture
#         claim intact. `split` normalises separately with a learned gate -- stronger control,
#         but the local half becomes an explicitly normalised neighbourhood pool.
#   The sequential edge-then-node update is on everywhere (worth +0.005, job 453148).
#   NO task here touches the test split. QOS medium: 32 jobs per ACCOUNT, keep arrays small.
#SBATCH --job-name=g2l-p8e
#SBATCH --account=marlowe-m000211-pm06
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000211-pm06/%u/logs/%x-%A_%a.out
set -euo pipefail
module load slurm
SCR=/scratch/m000211-pm06/$USER
export PYG_DATA_ROOT="$SCR/pyg" TORCH_HOME="$SCR/torch" XDG_CACHE_HOME="$SCR/xdgcache" HF_HUB_OFFLINE=1
export G2L_ROWS="$SCR/results/phase8e/rows"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$HOME/envs/g2l/bin/activate"
cd "$HOME/Graph_to_LLM"
python -c "import torch_geometric, ogb, g2l.lgm, g2l.molclass, g2l.ogbnet, g2l.run_molhiv"
T=${SLURM_ARRAY_TASK_ID:-0}
echo "job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $T on $SLURMD_NODENAME | code commit ${G2L_COMMIT:-unset}"
if [ "$T" -eq 0 ]; then
  python -u -m g2l.run_molhiv --stage baseline --model gin --warm
else
  MODES=(joint count split joint count split)
  DS=(64 64 64 128 128 128)
  PS=(0.5 0.5 0.5 0.3 0.3 0.3)
  I=$((T - 1))
  python -u -m g2l.run_molhiv --stage tune --attn "${MODES[$I]}" --d "${DS[$I]}" --dropout "${PS[$I]}"
fi
