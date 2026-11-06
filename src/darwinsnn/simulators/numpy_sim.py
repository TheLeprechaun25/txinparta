"""Reference simulator in NumPy/SciPy.

It is easy to read and needs no compilation, so it is the source of truth
the Pultsito backend is tested against (tests/test_pultsito_equivalence.py).
A whole batch of samples is simulated at once. Synapses are grouped by delay
into one sparse matrix per delay value, so each time step costs one sparse
mat-mul per distinct delay.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..data import SpikeDataset
from ..network import Network
from .base import SimulationResult, n_bins


class NumpySimulator:
    def __init__(self, tau: float = 20.0, batch_size: int = 64):
        self.alpha = 1.0 - 1.0 / tau
        self.batch_size = batch_size

    def run(self, net: Network, weights: np.ndarray, data: SpikeDataset, idx: np.ndarray,
            time_steps: int, bin_size: int | None = None) -> SimulationResult:
        idx = np.asarray(idx)
        delay_mats = self._delay_matrices(net, weights)
        counts = np.zeros((len(idx), net.n_neurons), np.int32)
        binned = (np.zeros((len(idx), n_bins(time_steps, bin_size), net.n_neurons), np.int32)
                  if bin_size else None)
        for start in range(0, len(idx), self.batch_size):
            sl = slice(start, start + self.batch_size)
            c, b = self._run_batch(net, delay_mats, data, idx[sl], time_steps, bin_size)
            counts[sl] = c
            if binned is not None:
                binned[sl] = b
        return SimulationResult(counts, binned, bin_size)

    @staticmethod
    def _delay_matrices(net: Network, weights: np.ndarray) -> dict[int, sparse.csr_matrix]:
        """One matrix W_d [n_neurons, n_inputs + n_neurons] per delay value d."""
        shape = (net.n_neurons, net.n_inputs + net.n_neurons)
        mats = {}
        for d in np.unique(net.delay):
            m = net.delay == d
            mats[int(d)] = sparse.csr_matrix((weights[m], (net.post[m], net.pre[m])), shape=shape)
        return mats

    def _run_batch(self, net, delay_mats, data, idx, T, bin_size):
        B, C, N = len(idx), net.n_inputs, net.n_neurons
        L = net.max_delay + 1                         # ring buffer length (as in Pultsito)
        inputs = data.dense(idx)                      # [B, T_in, C]
        T_in = inputs.shape[1]

        spikes = np.zeros((L, B, C + N), np.float32)  # spike history ring buffer
        v = np.tile(net.v_rest, (B, 1)).astype(np.float32)
        refr = np.zeros((B, N), np.int32)
        v_rest, v_th = net.v_rest[None, :], net.v_thresh[None, :]
        counts = np.zeros((B, N), np.int32)
        binned = np.zeros((B, n_bins(T, bin_size), N), np.int32) if bin_size else None

        for t in range(T):
            row = t % L
            spikes[row] = 0.0
            if t < T_in:
                spikes[row, :, :C] = inputs[:, t, :]

            # 1. input current from spikes emitted d steps ago
            current = np.zeros((B, N), np.float32)
            for d, W in delay_mats.items():
                if t - d >= 0:
                    current += (W @ spikes[(t - d) % L].T).T
            active = refr <= 0

            # 2. leaky integration (only for non-refractory neurons)
            v = np.where(active, self.alpha * v + (1 - self.alpha) * v_rest + current, v)

            # 3. firing, reset and refractory period
            refr -= 1
            fired = v >= v_th
            v = np.where(fired, v_rest, v)
            refr = np.where(fired, net.refractory[None, :], refr)
            spikes[row, :, C:] = fired
            counts += fired
            if binned is not None:
                binned[:, t // bin_size] += fired
        return counts, binned
