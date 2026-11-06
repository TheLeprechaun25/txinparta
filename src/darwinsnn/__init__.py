"""DarwinSNN: neuroevolution of motif-based, non-layered spiking neural networks.

Module map (read in this order):
    motifs.py      the 7 building-block circuits
    genome.py      genotype (motifs + links) and its decoding into a Network
    network.py     flat phenotype arrays shared by all simulators
    simulators/    NumPy reference and Pultsito (C) backends, same model
    data/          spike datasets: MNIST (Poisson), SHD, synthetic temporal patterns
    fitness.py     spike-pattern features, prototypes, separation/accuracy/energy
    evaluation.py  evaluating genomes with common random numbers
    mutation.py    structural and parametric mutation operators
    nsga2.py       non-dominated sorting, crowding, tournaments
    evolution.py   the main loop
"""

__version__ = "0.1.0"
