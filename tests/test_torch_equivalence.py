"""TorchSimulator must reproduce the NumPy reference spike for spike, both for one
network and for several networks stacked into one pass (run_many).
Weights are whole numbers, so the sums are exact in float32 whatever the order.
Runs on the GPU if there is one, otherwise on the CPU. Skipped without torch.
"""

import numpy as np
import pytest

from darwinsnn.config import ExperimentConfig, GenomeConfig
from darwinsnn.data import make_temporal_patterns
from darwinsnn.evaluation import Evaluator
from darwinsnn.genome import build_network, random_genome
from darwinsnn.simulators import NumpySimulator

torch_sim = pytest.importorskip("darwinsnn.simulators.torch_sim")


def setup(seed, n_nets):
    rng = np.random.default_rng(seed)
    data = make_temporal_patterns(4, 20, 60, 5, 2.0, 0.01, rng)
    gc = GenomeConfig(n_motifs=(10, 50), weight=(3.0, 15.0), refractory=(0, 4))
    nets = [build_network(random_genome(gc, data.n_channels, rng)) for _ in range(n_nets)]
    ws = [n.signed_weights(np.rint(rng.uniform(*gc.weight, n.n_synapses)).astype(np.float32))
          for n in nets]
    return data, nets, ws


@pytest.mark.parametrize("seed", range(3))
def test_single_network(seed):
    data, (net,), (w,) = setup(seed, 1)
    idx = np.arange(len(data))
    ref = NumpySimulator(batch_size=6).run(net, w, data, idx, 90, bin_size=5)
    out = torch_sim.TorchSimulator(batch_size=7).run(net, w, data, idx, 90, bin_size=5)
    assert ref.counts.sum() > 0
    np.testing.assert_array_equal(out.counts, ref.counts)
    np.testing.assert_array_equal(out.binned, ref.binned)


@pytest.mark.parametrize("max_neurons", [200_000, 150])      # one pass / split into groups
def test_stacked_networks(max_neurons):
    data, nets, ws = setup(7, 5)
    idx = np.arange(len(data))
    sim = torch_sim.TorchSimulator(batch_size=8, max_neurons=max_neurons)
    many = sim.run_many(nets, ws, data, idx, 80, bin_size=10)
    for net, w, out in zip(nets, ws, many):
        ref = NumpySimulator().run(net, w, data, idx, 80, bin_size=10)
        np.testing.assert_array_equal(out.counts, ref.counts)
        np.testing.assert_array_equal(out.binned, ref.binned)


def test_population_evaluation_matches_one_by_one():
    cfg = ExperimentConfig()
    cfg.genome.n_motifs = (5, 20)
    cfg.genome.weight = (5.0, 5.0)               # constant whole-number weights: exact sums
    cfg.simulator.time_steps = 70
    cfg.fitness.n_reference = cfg.fitness.n_query = 3
    rng = np.random.default_rng(0)
    data = make_temporal_patterns(3, 10, 60, 8, 2.0, 0.0, rng)
    genomes = [random_genome(cfg.genome, data.n_channels, rng) for _ in range(4)]
    plan = Evaluator(cfg, None).make_plan(data, rng)
    one_by_one = [Evaluator(cfg, NumpySimulator()).evaluate(g, data, plan) for g in genomes]
    batched = Evaluator(cfg, torch_sim.TorchSimulator()).evaluate_population(genomes, data, plan)
    assert batched == one_by_one
