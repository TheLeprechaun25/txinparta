#!/bin/bash
# Run any darwinsnn command on one GPU. From the repo root:
#   sbatch --job-name=noise experiments/slurm/gpu.sbatch experiments/noise.py CONFIG --backend torch
# (the arguments are passed to the venv's python)
#SBATCH --partition=GPU          # adapt to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail
PY=${DARWINSNN_PY:-python}   # set DARWINSNN_PY to your environment's python
cd "$SLURM_SUBMIT_DIR"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) args=$*"
$PY -m pytest -q -p no:warnings tests/test_torch_equivalence.py
$PY "$@"
