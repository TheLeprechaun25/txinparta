#!/bin/bash
# One GPU job per (variant, seed): task id -> variant = id / SEEDS, seed = id % SEEDS.
#   VARIANTS="a b c" SEEDS=5 CONFIG=... OUT=... SEARCH_PLAN="16 32 4" \
#   sbatch --array=0-14%6 --export=ALL experiments/slurm/variants_seeds.sbatch
# (%6 = at most 6 GPUs at once). Merge with variants_summary.sbatch afterwards.
#SBATCH --job-name=darwinsnn_vseed
#SBATCH --partition=GPU          # adapt to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x_%A_%a.out

set -euo pipefail
read -r -a LIST <<< "$VARIANTS"
SEEDS=${SEEDS:-5}
VARIANT=${LIST[$((SLURM_ARRAY_TASK_ID / SEEDS))]}
SEED=$((SLURM_ARRAY_TASK_ID % SEEDS))
PY=${DARWINSNN_PY:-python}   # set DARWINSNN_PY to your environment's python
PLAN=()
if [ -n "${SEARCH_PLAN:-}" ]; then read -r -a P <<< "$SEARCH_PLAN"; PLAN=(--search-plan "${P[@]}"); fi
cd "$SLURM_SUBMIT_DIR"

echo "variant=$VARIANT seed=$SEED config=$CONFIG host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY -m pytest -q -p no:warnings tests/test_torch_equivalence.py
$PY experiments/variants.py "$CONFIG" --backend torch --seed-list "$SEED" \
    --pop-size "${POP_SIZE:-32}" --generations "${GENERATIONS:-50}" --variants "$VARIANT" \
    --out "$OUT" "${PLAN[@]}"
