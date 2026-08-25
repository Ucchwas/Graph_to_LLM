#!/bin/bash
# Marlowe one-time environment setup (runs on a LOGIN node).
# Layout per docs/hpc/marlowe.md: code+env on /projects (backed up), caches on /scratch.
set -euo pipefail
PROJ=/projects/m000211
SCR=/scratch/m000211/$USER
ls "$PROJ" >/dev/null   # autofs: touch the mount so it appears

mkdir -p "$SCR"/{hf,torch,triton,inductor,xdgcache,logs,runs} "$PROJ"/{envs,data}

# --- code ---
if [ ! -d "$PROJ/Graph_to_LLM" ]; then
  git clone https://github.com/Ucchwas/Graph_to_LLM.git "$PROJ/Graph_to_LLM"
else
  git -C "$PROJ/Graph_to_LLM" pull --ff-only
fi

# --- python env (venv on /projects; conda avoided per docs) ---
if [ ! -d "$PROJ/envs/g2l" ]; then
  python3 -m venv "$PROJ/envs/g2l"
fi
source "$PROJ/envs/g2l/bin/activate"
pip -q install --upgrade pip
# CUDA wheel index chosen AFTER we see the driver version from marlowe_verify.sh.
# cu126 is the safe default for H100 + recent drivers; adjust if verify says otherwise.
pip -q install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
pip -q install torch-geometric "transformers==5.15.1" peft accelerate torchmetrics ogb scipy scikit-learn networkx pytest pyyaml

# --- datasets (login node has internet; jobs run offline) ---
export PYG_DATA_ROOT="$PROJ/data/pyg"
python - <<'PY'
from torch_geometric.datasets import Planetoid
Planetoid("/projects/m000211/data/pyg", "Cora")
print("Cora staged")
PY

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
PY
echo "=== setup complete. Test with: sbatch slurm/phase1_scratch_sweep.sbatch ==="
