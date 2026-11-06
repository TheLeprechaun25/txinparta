# Txinparta

[![CI](https://github.com/TheLeprechaun25/txinparta/actions/workflows/ci.yml/badge.svg)](https://github.com/TheLeprechaun25/txinparta/actions/workflows/ci.yml)

**Neuroevolution of non-layered, motif-based spiking neural networks.**

Txinparta searches for spiking networks built from small biological circuits
(*motifs*) wired together without layers. The networks classify by **spike
pattern** rather than with output neurons, and the search keeps their activity
low, as an energy proxy. The search is multi-objective (NSGA-II). By default
weights are not part of the genome, so it looks for structure that works with
random weights.

- **Motif networks:** 7 motif types (feed-forward, feedback, recurrent, lateral
  inhibition, pattern generator), optionally grouped into modular clusters.
- **Three simulators, the same spikes:**
  - a NumPy reference;
  - the C engine [Pultsito](https://github.com/AlexAmenabar/Pultsito-SNN-framework-in-C);
  - a PyTorch GPU backend that simulates a whole population in one pass.

  Equivalence tests keep them identical.
- **Temporal tasks:** spike-count or time-binned readouts. Ships with a synthetic
  timing task, MNIST (Poisson code) and the
  [Spiking Heidelberg Digits](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).
- **Careful evaluation:**
  - prototypes and scores use separate samples;
  - every individual of a generation is scored on the same samples and weight
    draws, and parents are re-scored;
  - the final front is scored on a test set.

The Python package is called `darwinsnn`.

## Quickstart

```bash
git clone --recurse-submodules https://github.com/TheLeprechaun25/txinparta.git
cd txinparta
pip install -e ".[dev]"
python examples/quickstart.py          # about 1 minute: random vs evolved networks
```

Then run a full search from a config file:

```bash
darwinsnn baseline configs/temporal_patterns.toml   # what the readout gets from the raw input
darwinsnn run      configs/temporal_patterns.toml   # evolve; writes to results/temporal_patterns
darwinsnn run      configs/temporal_patterns.toml --set evolution.pop_size=32 simulator.backend=torch
```

Every setting can be overridden with `--set section.key=value`. A run writes:
- `config.json`, `provenance.json` (code commits, seed, machine);
- `log.jsonl` (one line per evaluated individual);
- `final_front.pkl` (Pareto-front genomes with their test scores).

## Installation options

| extra | command | for |
|---|---|---|
| core | `pip install -e .` | NumPy/SciPy simulator, everything else |
| `dev` | `pip install -e ".[dev]"` | tests, ruff, pre-commit |
| `gpu` | `pip install -e ".[gpu]"` | PyTorch backend (`simulator.backend = "torch"`) |
| `shd` | `pip install -e ".[shd]"` | h5py, to read SHD |
| Pultsito | `native/build_pultsito.sh` | C backend (`simulator.backend = "pultsito"`); needs gcc and OpenMP |

For PyTorch, pick the build that matches your GPU. Recent cards (e.g. Blackwell,
sm_120) need a CUDA 12.8+ build.

## Configs

| config | task |
|---|---|
| `configs/temporal_patterns.toml` | synthetic: each channel fires once at a class-specific time, so only timing is informative. Laptop-sized. |
| `configs/shd.toml` | Spiking Heidelberg Digits: 20 spoken digits (English and German), 700 cochlea channels. GPU recommended. |
| `configs/mnist.toml` | MNIST, Poisson rate code, spike-count readout; large networks. |

## Repository layout

```
src/darwinsnn/
  motifs.py        the 7 motifs
  genome.py        genotype (motifs + links + clusters) and its decoding
  network.py       flat network arrays shared by all simulators
  simulators/      numpy (reference), pultsito (C, ctypes), torch (GPU, whole population)
  data/            spike datasets: MNIST, SHD, synthetic temporal patterns
  fitness.py       spike-pattern features, prototypes, objectives
  evaluation.py    evaluation with shared samples and weight draws
  mutation.py      structural and parametric mutations
  crossover.py     region and cluster crossover
  nsga2.py         non-dominated sorting, crowding, tournaments
  evolution.py     the main loop
  parallel.py      process-pool / GPU population evaluation
  readouts.py, analysis.py   ridge readout, control networks, hypervolume
native/            C shim exposing Pultsito to Python + build script
experiments/       baselines, search variants, evaluation noise, weight calibration, SLURM jobs
examples/          quickstart
tests/             unit tests and backend equivalence tests
third_party/       Pultsito (git submodule, pinned)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the model, the genome, the
evaluation protocol and the search in detail.

## Experiments

The scripts in `experiments/` run on a laptop with the NumPy backend, or on a GPU
cluster with `--backend torch`. SLURM templates are in `experiments/slurm/`.

| script | question |
|---|---|
| `baselines.py` | do evolved networks beat the raw input, random networks, and rewired / re-parametrised versions of themselves? |
| `variants.py` | mutation-only vs crossover vs clustered genomes, with the same budget |
| `noise.py` | how large must an evaluation be to rank good networks reliably? |
| `calibrate.py` | which weight range keeps random networks out of silence and saturation? |

## Development

```bash
pip install -e ".[dev]" && pre-commit install
make test        # pytest (the Pultsito/torch equivalence tests run if those backends are present)
make lint        # ruff
make pultsito    # build the C backend
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a motif, an objective or a
simulator backend.

## Acknowledgements

Txinparta builds on [DarwinSNNs](https://github.com/AlexAmenabar/DarwinSNNs) and
[Pultsito](https://github.com/AlexAmenabar/Pultsito-SNN-framework-in-C) by Alex Amenabar.
SHD is by Cramer et al. (2020), *The Heidelberg Spiking Data Sets for the Systematic
Evaluation of Spiking Neural Networks*, IEEE TNNLS.

If you use this code, please cite it (see [CITATION.cff](CITATION.cff)).
