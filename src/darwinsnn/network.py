"""Phenotype: the flat network that the simulators run.

Index convention (the same as Pultsito): presynaptic indices live in one
combined space, where `0 .. n_inputs-1` are the input channels and
`n_inputs + i` is network neuron `i`. Postsynaptic indices are always
network neurons `0 .. n_neurons-1`. Synapses are sorted by postsynaptic
neuron, so each neuron's incoming synapses are contiguous (CSR order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Network:
    n_inputs: int
    # neurons [n_neurons]
    v_thresh: np.ndarray            # float32
    v_rest: np.ndarray              # float32
    refractory: np.ndarray          # int32
    sign: np.ndarray                # int8, +1 excitatory / -1 inhibitory
    motif_of_neuron: np.ndarray     # int32, index of the motif each neuron belongs to
    is_output: np.ndarray           # bool, neuron is an output neuron of its motif
    # synapses [n_synapses], sorted by post
    pre: np.ndarray                 # int32, combined index space (inputs first)
    post: np.ndarray                # int32, network neuron index
    delay: np.ndarray               # int32, >= 1
    weight_magnitude: np.ndarray | None  # float32, None when weights are not evolved
    syn_sign: np.ndarray            # int8, sign of each synapse (from the presynaptic neuron)

    @property
    def n_neurons(self) -> int:
        return len(self.v_thresh)

    @property
    def n_synapses(self) -> int:
        return len(self.pre)

    @property
    def max_delay(self) -> int:
        return int(self.delay.max()) if self.n_synapses else 1

    def signed_weights(self, magnitudes: np.ndarray | None = None) -> np.ndarray:
        """Apply Dale's law: signed weight = sign of the presynaptic neuron * magnitude."""
        m = self.weight_magnitude if magnitudes is None else magnitudes
        if m is None:
            raise ValueError("network has no weights; sample magnitudes first")
        return (self.syn_sign * m).astype(np.float32)

    def out_degree(self) -> np.ndarray:
        """Number of outgoing synapses of each network neuron (used for SynOps)."""
        net_pre = self.pre[self.pre >= self.n_inputs] - self.n_inputs
        return np.bincount(net_pre, minlength=self.n_neurons)
