"""Experiment configuration.

Each experiment is described by a TOML file (see configs/). Every section of
the file maps to one dataclass below, and any value left out falls back to
the default given here. Ranges are written as [low, high], both inclusive.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class DataConfig:
    name: str = "temporal_patterns"     # "mnist" | "shd" | "temporal_patterns"
    path: str = ""                      # dataset location (unused for synthetic data)
    time_steps: int = 100               # length of the input spike trains
    dt_ms: float = 1.0                  # SHD only: ms per simulation step
    limit: int = 0                      # MNIST/SHD: use only the first `limit` samples (0 = all)
    max_rate: float = 0.2               # MNIST only: spike prob. per step for a white pixel
    n_classes: int = 10                 # synthetic data only
    n_channels: int = 50                # synthetic data only
    n_samples_per_class: int = 100      # synthetic data only
    jitter: float = 2.0                 # synthetic data only: timing noise (steps)
    noise_rate: float = 0.002           # synthetic data only: background spikes per step


@dataclass
class GenomeConfig:
    motif_types: list[str] | None = None        # None = all motifs in motifs.MOTIFS
    n_motifs: tuple[int, int] = (50, 200)       # initial number of motifs
    fan_out: tuple[int, int] = (2, 6)           # target motifs per motif at initialisation
    multiplicity: tuple[int, int] = (1, 2)      # parallel copies of a motif-to-motif link
    input_fan_out: tuple[int, int] = (1, 3)     # motifs reached by each input channel
    n_clusters: tuple[int, int] = (1, 1)        # clusters of motifs (1 = unstructured)
    p_inter_cluster: float = 0.1                # fraction of links between clusters
    v_thresh: tuple[float, float] = (-55.0, -45.0)
    v_rest: tuple[float, float] = (-75.0, -65.0)
    refractory: tuple[int, int] = (0, 5)        # time steps; sampled with a bias to low values
    delay: tuple[int, int] = (1, 10)            # time steps; must be >= 1 (causality)
    weight: tuple[float, float] = (10.0, 20.0)  # magnitude, the sign comes from the neuron
    evolve_weights: bool = False                # False = weight-agnostic search


@dataclass
class MutationConfig:
    # Relative probability of picking each operator for an offspring.
    p_add_motifs: float = 1.0
    p_remove_motifs: float = 1.0
    p_add_links: float = 1.0
    p_remove_links: float = 1.0
    p_rewire_inputs: float = 0.5
    p_neuron_params: float = 1.0
    p_delays: float = 1.0
    p_weights: float = 1.0              # only used when genome.evolve_weights = true
    # Strength of each operator.
    motif_fraction: tuple[float, float] = (0.02, 0.10)  # motifs added/removed, as a fraction
    link_fraction: tuple[float, float] = (0.02, 0.10)   # links added/removed, as a fraction
    param_fraction: tuple[float, float] = (0.02, 0.20)  # neurons/synapses perturbed
    v_step: float = 1.0                 # mV, std of the threshold / resting-potential perturbation
    weight_step: float = 1.0            # std of the weight perturbation
    n_operators: int = 1                # operators applied per offspring


@dataclass
class SimulatorConfig:
    backend: str = "numpy"              # "numpy" (reference) | "pultsito" (C) | "torch" (GPU)
    device: str = "cuda"                # torch only
    time_steps: int = 150               # simulated steps (>= data.time_steps)
    tau: float = 20.0                   # membrane time constant; Pultsito hard-codes 20
    batch_size: int = 32
    n_threads: int = 4                  # Pultsito only
    library: str = ""                   # Pultsito only: path to libdarwinsnn_pultsito.so


@dataclass
class FitnessConfig:
    objectives: list[str] = field(default_factory=lambda: ["separation", "spikes"])
    features: str = "binned"            # "counts" (rate code) | "binned" (temporal code)
    bin_size: int = 10                  # time steps per bin for "binned" features
    readout: str = "all"                # "all" neurons | "outputs" = motif output neurons only
    distance: str = "manhattan"         # "manhattan" | "euclidean"
    prototype: str = "medoid"           # "medoid" | "mean"
    dispersion: str = "mean"            # class spread: "radius" | "mean" | "median" | "p90"
    n_reference: int = 10               # samples per class used to build prototypes
    n_query: int = 10                   # held-out samples per class used to score accuracy
    n_repetitions: int = 3              # weight draws x sample draws averaged per evaluation


@dataclass
class EvolutionConfig:
    pop_size: int = 16
    generations: int = 20
    seed: int = 0
    out_dir: str = "results/run"
    reevaluate_parents: bool = True     # True = parents and offspring share one evaluation batch
    p_crossover: float = 0.0            # probability that an offspring comes from two parents
    crossover: str = "region"           # "region" (any genome) | "cluster" (needs n_clusters > 1)
    n_workers: int = 1                  # parallel processes evaluating individuals


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, values: dict):
    """Create dataclass `cls` from a dict, rejecting unknown keys (catches typos)."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"Unknown keys for [{cls.__name__}]: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if is_dataclass(default):
            value = _build(type(default), value)
        elif isinstance(default, tuple):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "rb") as f:
        cfg = _build(ExperimentConfig, tomllib.load(f))
    validate(cfg)
    return cfg


def validate(cfg: ExperimentConfig) -> None:
    g, s = cfg.genome, cfg.simulator
    assert g.delay[0] >= 1, "delays must be >= 1 time step"
    assert g.v_thresh[0] > g.v_rest[1], "every threshold must be above every resting potential"
    assert s.time_steps >= cfg.data.time_steps, "simulate at least as long as the input"
    if s.backend == "pultsito":
        assert s.tau == 20.0, "Pultsito hard-codes tau = 20 (alpha = 0.95)"
    if cfg.evolution.p_crossover > 0 and cfg.evolution.crossover == "cluster":
        assert g.n_clusters[0] > 1, "cluster crossover needs genome.n_clusters > 1"
    assert cfg.evolution.crossover in ("region", "cluster"), "unknown crossover"
    for name in cfg.fitness.objectives:
        assert name in ("separation", "accuracy", "spikes", "synops"), f"unknown objective {name}"
