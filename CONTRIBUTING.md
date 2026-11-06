# Contributing

## Setup

```bash
git clone --recurse-submodules https://github.com/TheLeprechaun25/txinparta.git
cd txinparta
pip install -e ".[dev]"        # add ,gpu and/or ,shd as needed
pre-commit install             # runs ruff on every commit
native/build_pultsito.sh       # optional: C backend, enables its equivalence tests
make test && make lint
```

CI (`.github/workflows/ci.yml`) runs ruff, all tests, the NumPy/Pultsito and
NumPy/torch equivalence tests (torch on CPU), and the quickstart example.

## Ground rules

- **The NumPy simulator is the reference.** Any change to the neuron model or the
  update order goes there first. The other backends must then reproduce it
  spike for spike: run `tests/test_pultsito_equivalence.py` and
  `tests/test_torch_equivalence.py`.
- **All randomness comes from a `numpy.random.Generator`** passed in explicitly. A run
  is fully determined by its config and seed. Evaluation randomness lives in the
  `EvalPlan`, so results do not depend on the number of workers.
- **Configs are dataclasses** (`config.py`). Unknown TOML keys are rejected, which
  catches typos. Add new settings there, with a default and a comment.
- Keep modules small and readable. Line length is 100 (ruff).

## Common extensions

**A new motif.** Add a `Motif(...)` to `MOTIFS` in `motifs.py`: neuron signs, internal
edges, input and output neurons. Everything else (genome, decoding, mutations) picks
it up. Add a check in `tests/test_genome_and_mutation.py`.

**A new objective.** Compute it in `evaluation.py` (`evaluate_population`), add its
name to `METRICS` there, and add its direction to `DIRECTION` in `fitness.py`:
+1 to minimise, −1 to maximise. Then allow it in `config.validate`.

**A new readout feature.** Extend `features()` and `input_features()` in `fitness.py`,
so the no-network baseline uses the same representation.

**A new simulator backend.** Implement `run(net, weights, data, idx, time_steps,
bin_size) -> SimulationResult`. Optionally also implement `run_many` to simulate
several networks in one call. Register it in `simulators/__init__.py`, then add an
equivalence test against `NumpySimulator` with whole-number weights.

**A new mutation or crossover.** Add a function to `mutation.py` (and to `OPERATORS`
plus a `p_<name>` probability in `MutationConfig`), or to `CROSSOVERS` in
`crossover.py`. `tests/test_genome_and_mutation.py` and `tests/test_crossover.py`
check that genomes stay valid.

## Updating Pultsito

```bash
git -C third_party/pultsito pull        # or check out a specific commit
native/build_pultsito.sh && pytest tests/test_pultsito_equivalence.py
git add third_party/pultsito && git commit -m "Update Pultsito"
```
