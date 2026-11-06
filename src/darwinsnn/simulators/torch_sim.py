"""GPU simulator in PyTorch, which simulates *many networks at once*.

One motif network has a few hundred neurons, far too few to keep a GPU
busy. But every individual of a generation is evaluated on the same samples
(common random numbers), so `run_many` stacks the networks into one
block-diagonal network that shares the input channels and simulates them all
in a single pass:

    combined presynaptic index: [ input channels | net 0 | net 1 | ... ]
    combined neuron index:      [ net 0 | net 1 | ... ]

The model and the update order are those of NumpySimulator (see base.py),
and tests/test_torch_equivalence.py checks the two give the same spikes.

    pip install torch            # any CUDA build; runs on CPU without a GPU
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from ..data import SpikeDataset
from ..network import Network
from .base import SimulationResult, n_bins

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")


class TorchSimulator:
    def __init__(self, tau: float = 20.0, batch_size: int = 256, device: str = "cuda",
                 max_neurons: int = 200_000):
        self.alpha = 1.0 - 1.0 / tau
        self.batch_size = batch_size
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available()
                                   else "cpu")
        self.max_neurons = max_neurons      # per pass; bigger populations are split

    def run(self, net: Network, weights: np.ndarray, data: SpikeDataset, idx: np.ndarray,
            time_steps: int, bin_size: int | None = None) -> SimulationResult:
        return self.run_many([net], [weights], data, idx, time_steps, bin_size)[0]

    def run_many(self, nets: list[Network], weights: list[np.ndarray], data: SpikeDataset,
                 idx: np.ndarray, time_steps: int, bin_size: int | None = None
                 ) -> list[SimulationResult]:
        """Simulate the same samples on every network; one result per network."""
        results, group, size = [], [], 0
        for k, net in enumerate(nets):
            if group and size + net.n_neurons > self.max_neurons:
                results += self._run_group(group, nets, weights, data, idx, time_steps, bin_size)
                group, size = [], 0
            group.append(k)
            size += net.n_neurons
        if group:
            results += self._run_group(group, nets, weights, data, idx, time_steps, bin_size)
        return results

    # ------------------------------------------------------------------ internals
    def _run_group(self, group, nets, weights, data, idx, T, bin_size):
        stacked = _stack([nets[k] for k in group], [weights[k] for k in group])
        idx = np.asarray(idx)
        counts = np.zeros((len(idx), stacked["n_neurons"]), np.int32)
        binned = (np.zeros((len(idx), n_bins(T, bin_size), stacked["n_neurons"]), np.int32)
                  if bin_size else None)
        mats = self._delay_matrices(stacked)
        for start in range(0, len(idx), self.batch_size):
            sl = slice(start, start + self.batch_size)
            c, b = self._run_batch(stacked, mats, data, idx[sl], T, bin_size)
            counts[sl] = c
            if binned is not None:
                binned[sl] = b
        out = []
        for lo, hi in stacked["ranges"]:
            out.append(SimulationResult(counts[:, lo:hi].copy(),
                                        binned[:, :, lo:hi].copy() if binned is not None
                                        else None, bin_size))
        return out

    def _delay_matrices(self, s) -> dict[int, torch.Tensor]:
        """One sparse matrix W_d [n_neurons, n_inputs + n_neurons] per delay value d."""
        shape = (s["n_neurons"], s["n_inputs"] + s["n_neurons"])
        mats = {}
        for d in np.unique(s["delay"]):
            m = s["delay"] == d
            ij = torch.tensor(np.stack([s["post"][m], s["pre"][m]]), dtype=torch.long)
            w = torch.tensor(s["weight"][m], dtype=torch.float32)
            mats[int(d)] = torch.sparse_coo_tensor(ij, w, shape).coalesce().to_sparse_csr() \
                .to(self.device)
        return mats

    @torch.no_grad()
    def _run_batch(self, s, mats, data, idx, T, bin_size):
        dev, C, N, B = self.device, s["n_inputs"], s["n_neurons"], len(idx)
        L = s["max_delay"] + 1                                   # ring buffer length
        inputs = torch.from_numpy(data.dense(idx)).to(dev)       # [B, T_in, C]
        inputs = inputs.permute(1, 2, 0).float()                  # [T_in, C, B]
        T_in = inputs.shape[0]

        v_rest = torch.from_numpy(s["v_rest"]).to(dev)[:, None]  # [N, 1]
        v_th = torch.from_numpy(s["v_thresh"]).to(dev)[:, None]
        r_period = torch.from_numpy(s["refractory"]).to(dev)[:, None]
        spikes = torch.zeros((L, C + N, B), device=dev)          # columns = samples
        v = v_rest.expand(N, B).clone()
        refr = torch.zeros((N, B), dtype=torch.int32, device=dev)
        counts = torch.zeros((N, B), dtype=torch.int32, device=dev)
        nb = n_bins(T, bin_size) if bin_size else 0
        binned = torch.zeros((nb, N, B), dtype=torch.int32, device=dev) if bin_size else None
        a, b = self.alpha, 1.0 - self.alpha

        for t in range(T):
            row = t % L
            spikes[row].zero_()
            if t < T_in:
                spikes[row, :C] = inputs[t]
            current = torch.zeros((N, B), device=dev)
            for d, W in mats.items():
                if t - d >= 0:
                    current += W @ spikes[(t - d) % L]
            active = refr <= 0
            v = torch.where(active, a * v + b * v_rest + current, v)
            refr -= 1
            fired = v >= v_th
            v = torch.where(fired, v_rest, v)
            refr = torch.where(fired, r_period, refr)
            spikes[row, C:] = fired.float()
            counts += fired
            if binned is not None:
                binned[t // bin_size] += fired

        counts = counts.T.cpu().numpy()                           # [B, N]
        binned = binned.permute(2, 0, 1).cpu().numpy() if binned is not None else None
        return counts, binned


def _stack(nets: list[Network], weights: list[np.ndarray]) -> dict:
    """Block-diagonal union of networks that share the same input channels."""
    C = nets[0].n_inputs
    assert all(n.n_inputs == C for n in nets), "networks must share the input channels"
    pre, post, delay, w, ranges = [], [], [], [], []
    offset = 0
    for net, wk in zip(nets, weights):
        is_neuron = net.pre >= C
        pre.append(np.where(is_neuron, net.pre + offset, net.pre))
        post.append(net.post + offset)
        delay.append(net.delay)
        w.append(wk)
        ranges.append((offset, offset + net.n_neurons))
        offset += net.n_neurons
    cat = np.concatenate
    return {
        "n_inputs": C, "n_neurons": offset, "ranges": ranges,
        "pre": cat(pre), "post": cat(post), "delay": cat(delay),
        "weight": cat(w).astype(np.float32),
        "v_rest": cat([n.v_rest for n in nets]).astype(np.float32),
        "v_thresh": cat([n.v_thresh for n in nets]).astype(np.float32),
        "refractory": cat([n.refractory for n in nets]).astype(np.int32),
        "max_delay": max(n.max_delay for n in nets),
    }
