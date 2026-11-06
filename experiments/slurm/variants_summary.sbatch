#!/bin/bash
# Merge the per-variant results of OUT into OUT/summary.md (same env vars as variants.sbatch)
#SBATCH --job-name=darwinsnn_summary
#SBATCH --partition=CPU          # adapt to your cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail
PY=${DARWINSNN_PY:-python}   # set DARWINSNN_PY to your environment's python
OUT=${OUT:-results/variants}
cd "$SLURM_SUBMIT_DIR"
$PY experiments/variants.py "${CONFIG:-configs/temporal_patterns.toml}" --summarize \
    --pop-size "${POP_SIZE:-32}" --generations "${GENERATIONS:-50}" --out "$OUT"
cat "$OUT/summary.md"
