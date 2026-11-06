"""Txinparta in five steps, using the Python API directly (about 1 minute on a laptop).

    python examples/quickstart.py

1. load the synthetic temporal task   (spike timing is the only class information)
2. sample a random motif network       (the genome) and decode it into a Network
3. simulate it                         (NumPy reference simulator)
4. score random networks               (prototype readout on held-out test samples)
5. evolve a small population           and score the best network the same way
"""

import numpy as np

from darwinsnn.config import load_config
from darwinsnn.data import load_dataset
from darwinsnn.evaluation import Evaluator
from darwinsnn.evolution import Evolution
from darwinsnn.genome import build_network, random_genome
from darwinsnn.motifs import MOTIFS
from darwinsnn.simulators import NumpySimulator

cfg = load_config("configs/temporal_patterns.toml")
rng = np.random.default_rng(0)

# 1. data: 5 classes, 40 channels; every channel fires once, at a class-specific time
train, test = load_dataset(cfg.data, rng)
print(f"data: {len(train)} train / {len(test)} test samples, {train.n_channels} channels, "
      f"{train.n_classes} classes (chance = {1 / train.n_classes:.2f})")

# 2. a random genome (motifs + links) and its phenotype (flat arrays)
genome = random_genome(cfg.genome, train.n_channels, rng)
net = build_network(genome)
counts = {m.name: int(c) for m, c in zip(MOTIFS, genome.motif_counts())}
print(f"genome: {genome.n_motifs} motifs {counts}, {len(genome.links)} links")
print(f"network: {net.n_neurons} neurons, {net.n_synapses} synapses")

# 3. simulate 5 samples with one random weight draw (weights are not part of the genome)
weights = net.signed_weights(rng.uniform(*cfg.genome.weight, net.n_synapses).astype(np.float32))
result = NumpySimulator().run(net, weights, train, np.arange(5), cfg.simulator.time_steps)
print(f"spikes per sample: {result.total_spikes.tolist()}")

# 4. score: prototypes from training samples, accuracy on held-out test samples,
#    averaged over random weight draws. The same test plan is used for every network.
evaluator = Evaluator(cfg, NumpySimulator())
test_plan = evaluator.make_plan(train, np.random.default_rng(7), query_data=test, n_query=20)


def test_accuracy(g):
    return evaluator.evaluate(g, train, test_plan, test)["accuracy"]


random_scores = [test_accuracy(random_genome(cfg.genome, train.n_channels, rng)) for _ in range(4)]
print(f"random networks: test accuracy {np.mean(random_scores):.3f} "
      f"(each: {', '.join(f'{a:.2f}' for a in random_scores)})")

# 5. a short evolution (accuracy vs synaptic events) with a small, cheap search plan
cfg.fitness.objectives = ["accuracy", "synops"]
cfg.fitness.n_reference, cfg.fitness.n_query, cfg.fitness.n_repetitions = 8, 16, 2
cfg.evolution.pop_size, cfg.evolution.generations = 12, 10
cfg.evolution.out_dir = "results/quickstart"
front = Evolution(cfg, train, test, verbose=False).run()
best = max(front, key=lambda ind: ind.metrics["accuracy"])
print(f"evolved network: test accuracy {test_accuracy(best.genome):.3f} "
      f"({best.genome.n_motifs} motifs); logs in {cfg.evolution.out_dir}/")
