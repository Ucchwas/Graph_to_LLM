#!/bin/bash
# Marlowe one-time environment setup (runs on a LOGIN node; no GPU work).
# Layout per docs/hpc/marlowe-verified.md: code + venv in $HOME (NFS), everything
# large on /scratch/m000211-pm06/$USER. Code arrives by rsync from the laptop
# (private repo; no credentials on the cluster). Run after scripts/marlowe_verify.sh.
set -euo pipefail
SCR=/scratch/m000211-pm06/$USER
VENV=$HOME/envs/g2l
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SCR/uvcache" PYG_DATA_ROOT="$SCR/pyg"
mkdir -p "$SCR"/{hf,torch,triton,inductor,xdgcache,logs,runs,pyg,uvcache} "$HOME/envs"

cd "$HOME/Graph_to_LLM"
echo "code @ $(git rev-parse --short HEAD)"

# python3 -m venv is broken here (no ensurepip); uv is self-contained.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d "$VENV" ] || uv venv --python 3.10 "$VENV"
source "$VENV/bin/activate"

# torch from the cu126 index first (driver 580 on the nodes), the rest from PyPI.
uv pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
uv pip install -q torch-geometric "transformers==5.15.1" peft accelerate torchmetrics ogb scipy scikit-learn networkx pytest pyyaml
python -c "import torch,transformers,torch_geometric as g;print('torch',torch.__version__,'| cuda',torch.version.cuda,'| tf',transformers.__version__,'| pyg',g.__version__)"

# datasets: login node has internet; jobs run with HF_HUB_OFFLINE=1
python -c "
from torch_geometric.datasets import Planetoid
d=Planetoid('$SCR/pyg','Cora')[0];print('Cora:',d.num_nodes,'nodes',tuple(d.edge_index.shape))"

python -m pytest tests/ -q 2>&1 | tail -3
uv pip freeze > requirements-cluster.txt
echo "SETUP_COMPLETE -- next: sbatch slurm/phase1_baselines.sbatch"
