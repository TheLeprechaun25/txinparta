"""From spike patterns to objective values.

The network has no output layer. Each sample is described by the spike
pattern of the network (its "features"), and classification is done by
distance to one prototype pattern per class. The steps:

  1. features   counts per neuron (rate code), or counts per neuron per time
                bin (temporal code; with bin_size=1 this is the full raster)
  2. prototype  per class, from the *reference* samples: the medoid (the
                sample closest to all others of its class) or the mean pattern
  3. scores     on held-out *query* samples:
       separation  mean over class pairs of D_ij / (max(d_i, d_j) + D_ij),
                   D = distance between prototypes, d = class dispersion (radius /
                   mean / median / p90 of the distances to the own prototype)
       accuracy    fraction of queries whose nearest prototype is their own class
     energy
       spikes      mean spikes per sample
       synops      mean synaptic events per sample (spikes x out-degree), closer to
                   the cost on neuromorphic hardware

Prototypes and scores never use the same samples, so accuracy is never
measured on the samples that built the prototypes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .config import FitnessConfig
from .network import Network
from .simulators import SimulationResult

# objective name -> +1 to minimise, -1 to maximise (NSGA-II always minimises)
DIRECTION = {"separation": -1, "accuracy": -1, "spikes": +1, "synops": +1}


def features(result: SimulationResult, net: Network, cfg: FitnessConfig) -> np.ndarray:
    """Spike-pattern feature vector of every sample [n_samples, n_features]."""
    mask = net.is_output if cfg.readout == "outputs" else np.ones(net.n_neurons, bool)
    if cfg.features == "counts":
        return result.counts[:, mask].astype(np.float64)
    if cfg.features == "binned":
        return result.binned[:, :, mask].reshape(len(result.counts), -1).astype(np.float64)
    raise ValueError(f"unknown features {cfg.features}")


def input_features(data, idx, cfg: FitnessConfig) -> np.ndarray:
    """The same features computed on the *input* spike trains (the no-network baseline)."""
    raster = data.dense(idx).astype(np.float64)               # [n, T, channels]
    if cfg.features == "counts":
        return raster.sum(axis=1)
    pad = (-raster.shape[1]) % cfg.bin_size
    raster = np.pad(raster, ((0, 0), (0, pad), (0, 0)))
    n, T, C = raster.shape
    return raster.reshape(n, T // cfg.bin_size, cfg.bin_size, C).sum(axis=2).reshape(n, -1)


def distances(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    return cdist(a, b, {"manhattan": "cityblock", "euclidean": "euclidean"}[metric])


def prototypes(X: np.ndarray, y: np.ndarray, n_classes: int, cfg: FitnessConfig) -> np.ndarray:
    """One prototype pattern per class [n_classes, n_features]."""
    protos = np.zeros((n_classes, X.shape[1]))
    for c in range(n_classes):
        Xc = X[y == c]
        if cfg.prototype == "mean":
            protos[c] = Xc.mean(axis=0)
        else:  # medoid: the sample with the smallest mean distance to its class
            protos[c] = Xc[distances(Xc, Xc, cfg.distance).mean(axis=1).argmin()]
    return protos


def _dispersion(d: np.ndarray, kind: str) -> float:
    return float({"radius": np.max, "mean": np.mean, "median": np.median,
                  "p90": lambda x: np.percentile(x, 90)}[kind](d))


def separation_and_accuracy(X_ref, y_ref, X_query, y_query, n_classes, cfg: FitnessConfig):
    P = prototypes(X_ref, y_ref, n_classes, cfg)
    D_query = distances(X_query, P, cfg.distance)          # [n_query, n_classes]
    accuracy = float((D_query.argmin(axis=1) == y_query).mean())

    disp = np.array([_dispersion(D_query[y_query == c, c], cfg.dispersion)
                     for c in range(n_classes)])
    D_proto = distances(P, P, cfg.distance)
    i, j = np.triu_indices(n_classes, k=1)
    denom = np.maximum(disp[i], disp[j]) + D_proto[i, j]
    pair = np.divide(D_proto[i, j], denom, out=np.zeros_like(denom), where=denom > 0)
    return float(pair.mean()), accuracy


def energy(result: SimulationResult, net: Network) -> tuple[float, float]:
    spikes = float(result.counts.sum(axis=1).mean())
    synops = float((result.counts @ net.out_degree()).mean())
    return spikes, synops
