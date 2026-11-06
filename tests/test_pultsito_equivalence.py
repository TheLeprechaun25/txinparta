"""The Pultsito backend must reproduce the NumPy reference spike for spike.

Weights are whole numbers, so the synaptic sums are exact in float32
whatever the summation order. Skipped if the shared library is not built
(native/build_pultsito.sh), unless DARWINSNN_REQUIRE_PULTSITO is set.
"""

import os

import numpy as np
import pytest

from darwinsnn.config import GenomeConfig
from darwinsnn.data import make_temporal_patterns
from darwinsnn.genome import build_network, random_genome
from darwinsnn.simulators import NumpySimulator

pultsito = pytest.importorskip("darwinsnn.simulators.pultsito")


@pytest.fixture(scope="module")
def backend():
    try:
        return pultsito.PultsitoSimulator(batch_size=7, n_threads=2)
    except FileNotFoundError as e:
        if os.environ.get("DARWINSNN_REQUIRE_PULTSITO"):   # set in CI: missing library = failure
            raise
        pytest.skip(str(e))


@pytest.mark.parametrize("seed", range(5))
def test_same_spikes_as_reference(backend, seed):
    rng = np.random.default_rng(seed)
    data = make_temporal_patterns(4, 20, 60, 5, 2.0, 0.01, rng)
    gc = GenomeConfig(n_motifs=(20, 60), weight=(3.0, 15.0), refractory=(0, 4))
    net = build_network(random_genome(gc, data.n_channels, rng))
    w = net.signed_weights(np.rint(rng.uniform(*gc.weight, net.n_synapses)).astype(np.float32))
    idx = np.arange(len(data))                       # 20 samples -> last batch is partial
    ref = NumpySimulator(batch_size=6).run(net, w, data, idx, 90, bin_size=5)
    out = backend.run(net, w, data, idx, 90, bin_size=5)
    assert ref.counts.sum() > 0
    np.testing.assert_array_equal(out.counts, ref.counts)
    np.testing.assert_array_equal(out.binned, ref.binned)
