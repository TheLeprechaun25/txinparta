#!/usr/bin/env bash
# Build libdarwinsnn_pultsito.so: Pultsito's CPU engine plus the ctypes shim.
#
#   native/build_pultsito.sh [PULTSITO_DIR] [OUTPUT_DIR]
#
# PULTSITO_DIR defaults to third_party/pultsito:
#   git clone https://github.com/AlexAmenabar/Pultsito-SNN-framework-in-C third_party/pultsito
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PULTSITO="${1:-$ROOT/third_party/pultsito}"
OUT="${2:-$ROOT/build}"
mkdir -p "$OUT"

SRC="$PULTSITO/src/pultsito"
# -ffp-contract=off: no fused multiply-add, so the float results match the
# NumPy reference bit for bit (see tests/test_pultsito_equivalence.py).
# -Wno-incompatible-pointer-types: Pultsito's legacy large_simulation() fails on GCC >= 14 otherwise.
gcc -O3 -fPIC -shared -fopenmp "${CFLAGS_ARCH:--march=native}" -ffp-contract=off \
    -Wno-incompatible-pointer-types \
    -I"$PULTSITO/include" -I"$PULTSITO/include/pultsito" -I"$PULTSITO/lib" \
    "$ROOT/native/pultsito_shim.c" \
    "$SRC/snn_library.c" "$SRC/load_data.c" "$SRC/helpers.c" \
    "$SRC/neuron_models/lif_neuron.c" "$SRC/training_rules/stdp.c" \
    "$SRC/simulations/simulations.c" "$SRC/metrics.c" \
    -lm -o "$OUT/libdarwinsnn_pultsito.so"
echo "built $OUT/libdarwinsnn_pultsito.so"
