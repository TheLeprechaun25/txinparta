"""Shared test helpers."""

import numpy as np

from darwinsnn.config import GenomeConfig
from darwinsnn.genome import Genome, build_network


def check_network(genome: Genome, gc: GenomeConfig):
    net = build_network(genome)
    C, N = genome.n_channels, genome.n_neurons
    assert net.n_neurons == N
    assert net.n_synapses == genome.n_synapses
    assert np.all(np.diff(net.post) >= 0), "synapses must be sorted by post"
    assert net.pre.min() >= 0 and net.pre.max() < C + N
    assert net.post.min() >= 0 and net.post.max() < N
    assert np.all((net.delay >= gc.delay[0]) & (net.delay <= gc.delay[1]))
    # Dale's law: the synapse sign is the sign of its presynaptic neuron, inputs are excitatory
    from_neuron = net.pre >= C
    assert np.all(net.syn_sign[from_neuron] == net.sign[net.pre[from_neuron] - C])
    assert np.all(net.syn_sign[~from_neuron] == 1)
    return net
