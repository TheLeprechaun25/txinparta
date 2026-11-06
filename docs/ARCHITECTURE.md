# Architecture

How Txinparta represents, simulates, scores and evolves networks. The code is
small; each section names the module where the details live.

```
genome (motifs + links) ──build_network──> Network (flat arrays)
        ▲                                        │ simulator.run / run_many
   mutation, crossover                           ▼
        │                         spike counts / time-binned spike counts
   NSGA-II selection <── objectives <── one prototype pattern per class
                         separation, accuracy, spikes, synops
```

## 1. Genome (`motifs.py`, `genome.py`)

A network is built from **motifs**, small circuits of 2–4 neurons with a fixed
internal wiring:

| motif | neurons | wiring | inputs → outputs |
|---|---|---|---|
| FFE feed-forward excitation | E E | 0→1 | 0 → 1 |
| FBE feedback excitation | E E | 0↔1 | 0 → 1 |
| FBI feedback inhibition | E E I | 0→1→2 ⊣0 | 0 → 1 |
| RCE recurrent excitation | E E E E | ring | all → all |
| RCI recurrent inhibition | I I I I | ring | all → all |
| LTI lateral inhibition | E I E | 0→1 ⊣2 | 0,2 → 0,2 |
| CPG pattern generator | E E E | all-to-all | all → all |

A `Genome` holds:

- `motifs`: uid → `MotifGene` (motif type, threshold, resting potential and refractory
  period per neuron, delays of the internal synapses, cluster id);
- `links`: (source uid, target uid) → `Link`. A link connects **every output neuron
  of the source to every input neuron of the target**, `multiplicity` times. It stores
  one delay per synapse (and one weight, if weights are evolved);
- `input_links`: (input channel, target uid) → `Link`.

Uids are stable, so adding or removing motifs never renumbers anything.
`prune_unreachable` drops motifs that no input can reach.

**Clusters.** With `genome.n_clusters > 1`, motifs belong to clusters. New links stay
inside the cluster, except for a fraction `p_inter_cluster` that goes to other clusters.

**Weights.** With `evolve_weights = false` (the default) the genome has no weights:
every evaluation draws them at random, so the search favours structure that works
for any weights. Signs follow Dale's law: a synapse has the sign of its presynaptic
neuron, and inputs are excitatory.

## 2. Network and simulators (`network.py`, `simulators/`)

`build_network` flattens a genome into arrays: one entry per neuron (threshold, rest,
refractory period, sign) and per synapse (pre, post, delay, weight), with synapses
sorted by postsynaptic neuron. Presynaptic indices use one combined space:
`0..C-1` are input channels and `C + i` is neuron `i`.

All backends run the same discrete-time LIF model, one step at a time:

```
I_i(t) = sum over synapses s -> i of  w_s * spike_pre(s)(t - d_s)     (if i not refractory)
v_i    = alpha * v_i + (1 - alpha) * v_rest_i + I_i                   (if i not refractory)
if v_i >= v_thresh_i:  spike;  v_i = v_rest_i;  refractory counter = r_i
refractory counter -= 1
```

with `alpha = 1 - 1/tau` (`tau = 20`). Delays are ≥ 1. Spikes that arrive during the
refractory period are lost. Every sample starts from rest.

| backend | where | use |
|---|---|---|
| `numpy` | `numpy_sim.py` | reference implementation; one sparse matrix per delay value |
| `pultsito` | `pultsito.py` + `native/pultsito_shim.c` | the C engine of [Pultsito](https://github.com/AlexAmenabar/Pultsito-SNN-framework-in-C), via ctypes |
| `torch` | `torch_sim.py` | GPU; stacks the whole population into one block-diagonal network |

The backends produce **identical spikes**: `tests/test_pultsito_equivalence.py` and
`tests/test_torch_equivalence.py` compare them with the NumPy reference, using
whole-number weights so that float sums are exact.

## 3. Readout and objectives (`fitness.py`)

There are no output neurons. A sample is described by the **spike pattern** of the
network:

- `features = "counts"`: spikes per neuron (rate code);
- `features = "binned"`: spikes per neuron per window of `bin_size` steps (temporal
  code; `bin_size = 1` is the full raster).

Each class gets one prototype pattern (`prototype = "medoid"` or `"mean"`), built from
*reference* samples. *Query* samples, which are never the reference ones, are then scored:

| objective | meaning |
|---|---|
| `accuracy` | fraction of queries whose nearest prototype is their own class |
| `separation` | mean over class pairs of D/(max(d_i, d_j) + D): D = distance between prototypes, d = spread of a class (`dispersion`: radius, mean, median or 90th percentile) |
| `spikes` | spikes per sample |
| `synops` | synaptic events per sample (spikes × out-degree), closer to hardware cost |

`readouts.py` adds a ridge classifier for analysis (not used during the search).

## 4. Evaluation (`evaluation.py`, `parallel.py`)

An evaluation is noisy: it depends on the samples drawn and on the weight draw. Each
generation draws one **EvalPlan**: `n_repetitions` × (reference samples, query samples,
weight seed). **All** individuals, parents included, are scored with it, so they are
compared on equal terms (common random numbers), and no individual survives on an
old lucky score. Metrics are averaged over the repetitions.

`experiments/noise.py` measures how well a plan of a given size ranks networks;
larger plans cost more simulation but rank good networks more reliably.

`PopulationEvaluator` evaluates the population with a process pool
(`evolution.n_workers`), or in one GPU pass with the torch backend. Results do not
depend on the number of workers.

## 5. Search (`nsga2.py`, `mutation.py`, `crossover.py`, `evolution.py`)

NSGA-II: non-dominated sorting + crowding distance, binary tournaments, and survivors
chosen from parents + offspring. An offspring is made by:

1. with probability `p_crossover`: **crossover** of two parents.
   - `region`: a connected region of each parent;
   - `cluster`: whole clusters of each parent.

   Links that left an inherited part are re-pointed into the other parent's part
   (so out-degrees are kept), and the fraction cut is logged;
2. **mutation**, `n_operators` of: add/remove motifs, add/remove links, rewire inputs,
   perturb neuron parameters, perturb delays, perturb weights (if evolved).

At the end, the Pareto front is scored on the test set and saved (`final_front.pkl`),
together with `log.jsonl` (one line per evaluated individual), `config.json` and
`provenance.json` (code commits, seed, machine).

## 6. Data (`data/`)

A `SpikeDataset` stores spike events (time step, channel) per sample, so encoded static
data and event data take the same path:

- `mnist`: MNIST images, Poisson rate code;
- `shd`: [Spiking Heidelberg Digits](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/),
  binned at `dt_ms`;
- `temporal_patterns`: synthetic task in which each channel fires once, at a
  class-specific time. Spike counts carry no class information, so only timing does.
