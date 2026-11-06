#!/bin/bash
# One GPU job per search variant. Submit from the repo root, e.g.
#
#   VARIANTS="mutation_only clustered clustered_crossover_low" CONFIG=configs/shd.toml \
#   OUT=results/variants_shd SEEDS=5 GENERATIONS=50 SEARCH_PLAN="16 16 2" \
#   sbatch --array=0-2 --export=ALL experiments/slurm/variants.sbatch
#
# then merge with variants_summary.sbatch (same CONFIG/OUT, --dependency=afterany:<job>).
# Defaults: the four main variants on the synthetic task.
#SBATCH --job-name=darwinsnn_variants
#SBATCH --partition=GPU          # adapt to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm/%x_%A_%a.out

set -euo pipefail
read -r -a LIST <<< "${VARIANTS:-mutation_only crossover clustered clustered_crossover}"
VARIANT=${LIST[$SLURM_ARRAY_TASK_ID]}
PY=${DARWINSNN_PY:-python}   # set DARWINSNN_PY to your environment's python
CONFIG=${CONFIG:-configs/temporal_patterns.toml}
OUT=${OUT:-results/variants}
PLAN=()
if [ -n "${SEARCH_PLAN:-}" ]; then read -r -a P <<< "$SEARCH_PLAN"; PLAN=(--search-plan "${P[@]}"); fi
cd "$SLURM_SUBMIT_DIR"

echo "variant=$VARIANT config=$CONFIG host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
# refuse to run if the GPU backend does not reproduce the NumPy reference
$PY -m pytest -q -p no:warnings tests/test_torch_equivalence.py

$PY experiments/variants.py "$CONFIG" --backend torch --seeds "${SEEDS:-10}" \
    --pop-size "${POP_SIZE:-32}" --generations "${GENERATIONS:-50}" --variants "$VARIANT" \
    --out "$OUT" "${PLAN[@]}"
