"""Hand-built networks with known answers: delays, leak, refractoriness, inhibition."""

import numpy as np
import pytest

from darwinsnn.data import SpikeDataset
from darwinsnn.network import Network
from darwinsnn.simulators import NumpySimulator


def tiny_network(n_inputs, n_neurons, synapses, v_thresh=-50.0, v_rest=-65.0, refractory=0,
                 signs=None):
    """synapses: list of (pre, post, delay, signed weight); pre in the combined index space."""
    pre, post, delay, w = map(np.array, zip(*synapses))
    order = np.argsort(post, kind="stable")
    signs = np.ones(n_neurons, np.int8) if signs is None else np.array(signs, np.int8)
    net = Network(
        n_inputs=n_inputs,
        v_thresh=np.full(n_neurons, v_thresh, np.float32),
        v_rest=np.full(n_neurons, v_rest, np.float32),
        refractory=np.full(n_neurons, refractory, np.int32),
        sign=signs, motif_of_neuron=np.zeros(n_neurons, np.int32),
        is_output=np.ones(n_neurons, bool),
        pre=pre[order].astype(np.int32), post=post[order].astype(np.int32),
        delay=delay[order].astype(np.int32),
        weight_magnitude=np.abs(w[order]).astype(np.float32),
        syn_sign=np.sign(w[order]).astype(np.int8),
    )
    return net, w[order].astype(np.float32)


def one_sample(times, channels, n_channels, T):
    return SpikeDataset([np.array(times, np.int32)], [np.array(channels, np.int32)],
                        np.array([0]), n_channels, T)


def raster(net, w, data, T):
    r = NumpySimulator().run(net, w, data, np.array([0]), T, bin_size=1)
    return r.binned[0]                                    # [T, n_neurons]


def test_delay_shifts_spike():
    net, w = tiny_network(1, 1, [(0, 0, 3, 20.0)])        # strong input, delay 3
    out = raster(net, w, one_sample([5], [0], 1, 20), 20)
    assert np.flatnonzero(out[:, 0]).tolist() == [8]      # 5 + 3


def test_chain_of_neurons_adds_delays():
    # input -> n0 (d=2) -> n1 (d=4); neuron j has combined index 1 + j
    net, w = tiny_network(1, 2, [(0, 0, 2, 20.0), (1, 1, 4, 20.0)])
    out = raster(net, w, one_sample([1], [0], 1, 20), 20)
    assert np.flatnonzero(out[:, 0]).tolist() == [3]
    assert np.flatnonzero(out[:, 1]).tolist() == [7]


def test_leak_needs_coincidence():
    # each input spike adds 10 mV; threshold is 15 mV above rest
    net, w = tiny_network(1, 1, [(0, 0, 1, 10.0)], v_thresh=-50.0, v_rest=-65.0)
    together = raster(net, w, one_sample([3, 4], [0, 0], 1, 30), 30)
    apart = raster(net, w, one_sample([3, 25], [0, 0], 1, 40), 40)
    assert together[:, 0].sum() == 1
    assert apart[:, 0].sum() == 0


@pytest.mark.parametrize("refractory", [0, 2, 5])
def test_refractory_period(refractory):
    # input every step, strong enough to fire on its own: ISI = refractory + 1
    T = 30
    net, w = tiny_network(1, 1, [(0, 0, 1, 20.0)], refractory=refractory)
    out = raster(net, w, one_sample(list(range(T)), [0] * T, 1, T), T)
    isi = np.diff(np.flatnonzero(out[:, 0]))
    assert set(isi.tolist()) == {refractory + 1}


def test_inhibition_blocks_firing():
    # n0 inhibits n1 (sign from n0); both get the same input spike
    syn = [(0, 0, 1, 20.0), (0, 1, 3, 16.0), (1, 1, 1, -30.0)]
    net, w = tiny_network(1, 2, syn, signs=[-1, 1])
    out = raster(net, w, one_sample([0], [0], 1, 10), 10)
    assert out[:, 0].sum() == 1 and out[:, 1].sum() == 0


def test_batches_are_independent():
    net, w = tiny_network(1, 1, [(0, 0, 1, 20.0)])
    data = SpikeDataset([np.array([2]), np.array([7])], [np.array([0]), np.array([0])],
                        np.array([0, 1]), 1, 10)
    r = NumpySimulator(batch_size=1).run(net, w, data, np.array([0, 1]), 10, bin_size=1)
    assert np.flatnonzero(r.binned[0, :, 0]).tolist() == [3]
    assert np.flatnonzero(r.binned[1, :, 0]).tolist() == [8]
