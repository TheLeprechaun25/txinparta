"""Common interface for all simulator backends.

Model (identical in every backend, and the same as Pultsito's batch CPU path):

    for t in 0 .. T-1:
        I_i(t) = sum_{syn s -> i} w_s * spike_pre(s)(t - d_s)   if neuron i is not refractory
        v_i    = alpha * v_i + (1 - alpha) * v_rest_i + I_i      if neuron i is not refractory
        if v_i >= v_thresh_i:  spike, v_i = v_rest_i, refractory counter = r_i
        refractory counter -= 1

with alpha = 1 - 1/tau. Input channels act as "virtual neurons" whose spikes
come from the dataset. Delays are >= 1, so an input spike at step t
reaches its target at step t + d. Spikes arriving while a neuron is
refractory are lost. Every sample starts from v = v_rest with empty spike
buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..data import SpikeDataset
from ..network import Network


@dataclass
class SimulationResult:
    counts: np.ndarray              # [n_samples, n_neurons] spikes per neuron
    binned: np.ndarray | None       # [n_samples, n_bins, n_neurons], or None
    bin_size: int | None

    @property
    def total_spikes(self) -> np.ndarray:
        """Spikes per sample [n_samples]."""
        return self.counts.sum(axis=1)


class Simulator(Protocol):
    def run(self, net: Network, weights: np.ndarray, data: SpikeDataset, idx: np.ndarray,
            time_steps: int, bin_size: int | None = None) -> SimulationResult:
        """Simulate samples `idx` of `data` with signed `weights` [n_synapses].

        If `bin_size` is given, also return spike counts in consecutive windows
        of `bin_size` steps (bin_size=1 gives the full spike raster).
        """
        ...


def n_bins(time_steps: int, bin_size: int) -> int:
    return -(-time_steps // bin_size)
