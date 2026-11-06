"""Pultsito backend: runs the C engine through ctypes.

Build the library first:  native/build_pultsito.sh
Then point to it with simulator.library in the config, or with the
DARWINSNN_PULTSITO_LIB environment variable.

The shim currently drives Pultsito's batch *CPU* engine. Pultsito's GPU path
(simulate_batches_GPU) returns spike counts only, so binned readouts on GPU
use the torch backend instead.
"""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path

import numpy as np

from ..data import SpikeDataset
from ..network import Network
from .base import SimulationResult, n_bins

_DEFAULT = Path(__file__).resolve().parents[3] / "build" / "libdarwinsnn_pultsito.so"


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(ct.POINTER(ctype))


class PultsitoSimulator:
    def __init__(self, library: str = "", batch_size: int = 32, n_threads: int = 4):
        path = library or os.environ.get("DARWINSNN_PULTSITO_LIB") or str(_DEFAULT)
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found; run native/build_pultsito.sh")
        self.lib = ct.CDLL(path)
        self.lib.darwinsnn_pultsito_run.restype = ct.c_int
        self.batch_size = batch_size
        self.n_threads = n_threads

    def run(self, net: Network, weights: np.ndarray, data: SpikeDataset, idx: np.ndarray,
            time_steps: int, bin_size: int | None = None) -> SimulationResult:
        idx = np.asarray(idx)
        S, C, N = len(idx), net.n_inputs, net.n_neurons

        # inputs: spike times ordered by (sample, channel), plus counts per (sample, channel)
        n_spikes = np.zeros((S, C), np.uintp)
        times = []
        for k, i in enumerate(idx):
            t, c = data.times[i], data.channels[i]
            order = np.lexsort((t, c))
            times.append(t[order])
            n_spikes[k] = np.bincount(c, minlength=C)
        spike_times = np.ascontiguousarray(np.concatenate(times) if times else [], np.uintp)

        arr = np.ascontiguousarray
        v_th, v_rest = arr(net.v_thresh, np.float32), arr(net.v_rest, np.float32)
        refr, pre = arr(net.refractory, np.int32), arr(net.pre, np.int32)
        post, delay = arr(net.post, np.int32), arr(net.delay, np.int32)
        w = arr(weights, np.float32)

        counts = np.zeros((S, N), np.int32)
        binned = np.zeros((S, n_bins(time_steps, bin_size), N), np.int32) if bin_size else None
        status = self.lib.darwinsnn_pultsito_run(
            ct.c_size_t(C), ct.c_size_t(N), ct.c_size_t(net.n_synapses),
            _ptr(v_th, ct.c_float), _ptr(v_rest, ct.c_float), _ptr(refr, ct.c_int),
            _ptr(pre, ct.c_int), _ptr(post, ct.c_int), _ptr(w, ct.c_float), _ptr(delay, ct.c_int),
            ct.c_size_t(S), _ptr(n_spikes, ct.c_size_t), _ptr(spike_times, ct.c_size_t),
            ct.c_size_t(time_steps), ct.c_size_t(self.batch_size), ct.c_int(self.n_threads),
            ct.c_size_t(bin_size or 0),
            _ptr(counts, ct.c_int), _ptr(binned, ct.c_int) if binned is not None else None,
        )
        if status != 0:
            raise RuntimeError(f"Pultsito returned {status}")
        return SimulationResult(counts, binned, bin_size)
