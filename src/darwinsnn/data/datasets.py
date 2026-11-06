"""Spike datasets.

Every dataset is turned into a SpikeDataset, a list of spike *events*
(time step, channel) per sample. This format suits both static data that
has been encoded (MNIST + Poisson) and data that is temporal from the start
(SHD, N-MNIST, ...), and it converts cheaply into the formats each simulator
needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpikeDataset:
    times: list[np.ndarray]         # per sample: int32 spike times (time steps)
    channels: list[np.ndarray]      # per sample: int32 channel of each spike
    labels: np.ndarray              # [n_samples] int
    n_channels: int
    time_steps: int                 # every spike time is < time_steps
    name: str = ""

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_classes(self) -> int:
        return int(self.labels.max()) + 1

    def subset(self, idx) -> SpikeDataset:
        idx = np.asarray(idx)
        return SpikeDataset([self.times[i] for i in idx], [self.channels[i] for i in idx],
                            self.labels[idx], self.n_channels, self.time_steps, self.name)

    def dense(self, idx) -> np.ndarray:
        """Boolean raster [len(idx), time_steps, n_channels] for the given samples."""
        out = np.zeros((len(idx), self.time_steps, self.n_channels), dtype=bool)
        for b, i in enumerate(idx):
            out[b, self.times[i], self.channels[i]] = True
        return out

    def split_by_class(self, rng: np.random.Generator, n_per_class: int, exclude=()) -> np.ndarray:
        """Pick `n_per_class` random samples of every class (without the ones in `exclude`)."""
        excluded = np.zeros(len(self), bool)
        excluded[list(exclude)] = True
        picks = []
        for c in range(self.n_classes):
            pool = np.flatnonzero((self.labels == c) & ~excluded)
            if len(pool) < n_per_class:
                raise ValueError(f"class {c}: needs {n_per_class} samples, has {len(pool)}")
            picks.append(rng.choice(pool, n_per_class, replace=False))
        return np.concatenate(picks)


# ===================================================================== encoders
def poisson_encode(values: np.ndarray, time_steps: int, max_rate: float,
                   rng: np.random.Generator, labels: np.ndarray, name="") -> SpikeDataset:
    """Rate code: feature value v in [0, 1] spikes with probability v * max_rate per step."""
    times, channels = [], []
    for x in values:
        spikes = rng.random((time_steps, x.size)) < (x * max_rate)[None, :]
        t, c = np.nonzero(spikes)
        times.append(t.astype(np.int32))
        channels.append(c.astype(np.int32))
    return SpikeDataset(times, channels, labels.astype(int), values.shape[1], time_steps, name)


# ===================================================================== datasets
def _read_idx(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(path: str, split: str, time_steps: int, max_rate: float,
               rng: np.random.Generator, limit: int | None = None) -> SpikeDataset:
    """MNIST from the original idx files, Poisson-encoded."""
    prefix = "train" if split == "train" else "t10k"
    images = _read_idx(Path(path) / f"{prefix}-images.idx3-ubyte").reshape(-1, 784) / 255.0
    labels = _read_idx(Path(path) / f"{prefix}-labels.idx1-ubyte")
    if limit:
        images, labels = images[:limit], labels[:limit]
    return poisson_encode(images, time_steps, max_rate, rng, labels, name=f"mnist-{split}")


def load_shd(path: str, split: str, dt_ms: float, time_steps: int,
             limit: int | None = None) -> SpikeDataset:
    """Spiking Heidelberg Digits (Cramer et al. 2020): 700 channels, 20 classes.

    `path` is the folder with shd_train.h5 / shd_test.h5. Spike times (seconds)
    are converted to steps of `dt_ms`, and spikes after `time_steps` are dropped.
    """
    import h5py  # optional dependency: pip install darwinsnn[shd]

    with h5py.File(Path(path) / f"shd_{split}.h5", "r") as f:
        all_t, all_c = f["spikes"]["times"], f["spikes"]["units"]
        labels = np.asarray(f["labels"], dtype=int)
        n = len(labels) if limit is None else min(limit, len(labels))
        times, channels = [], []
        for i in range(n):
            t = np.floor(np.asarray(all_t[i]) * 1000.0 / dt_ms).astype(np.int32)
            keep = t < time_steps
            times.append(t[keep])
            channels.append(np.asarray(all_c[i], dtype=np.int32)[keep])
    return SpikeDataset(times, channels, labels[:n], 700, time_steps, name=f"shd-{split}")


def make_temporal_patterns(n_classes: int, n_channels: int, time_steps: int,
                           n_per_class: int, jitter: float, noise_rate: float,
                           rng: np.random.Generator) -> SpikeDataset:
    """Synthetic *temporal* task that is cheap to run.

    Every class is a fixed spatio-temporal pattern in which each channel fires
    exactly once, at a time specific to that class. Samples add timing jitter
    and random background spikes. All classes have the same spike count per
    channel, so a pure rate readout cannot solve the task. Only spike timing
    separates the classes. Useful for tests and as a stand-in for SHD.
    """
    margin = int(np.ceil(3 * jitter)) + 1
    templates = rng.integers(margin, time_steps - margin, size=(n_classes, n_channels))
    times, channels, labels = [], [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            t = np.rint(templates[c] + rng.normal(0, jitter, n_channels)).astype(int)
            ch = np.arange(n_channels)
            noise = rng.random((time_steps, n_channels)) < noise_rate
            nt, nc = np.nonzero(noise)
            t, ch = np.concatenate([t, nt]), np.concatenate([ch, nc])
            keep = (t >= 0) & (t < time_steps)
            times.append(t[keep].astype(np.int32))
            channels.append(ch[keep].astype(np.int32))
            labels.append(c)
    return SpikeDataset(times, channels, np.array(labels), n_channels, time_steps,
                        name="temporal-patterns")


def load_dataset(cfg, rng: np.random.Generator) -> tuple[SpikeDataset, SpikeDataset]:
    """Return (search set, test set) as described by a DataConfig."""
    limit = cfg.limit or None
    if cfg.name == "mnist":
        return (load_mnist(cfg.path, "train", cfg.time_steps, cfg.max_rate, rng, limit),
                load_mnist(cfg.path, "test", cfg.time_steps, cfg.max_rate, rng, limit))
    if cfg.name == "shd":
        return (load_shd(cfg.path, "train", cfg.dt_ms, cfg.time_steps, limit),
                load_shd(cfg.path, "test", cfg.dt_ms, cfg.time_steps, limit))
    if cfg.name == "temporal_patterns":
        # one set of class templates, split into search and test samples
        full = make_temporal_patterns(cfg.n_classes, cfg.n_channels, cfg.time_steps,
                                      2 * cfg.n_samples_per_class, cfg.jitter,
                                      cfg.noise_rate, rng)
        idx = np.arange(len(full))
        test = (idx % (2 * cfg.n_samples_per_class)) >= cfg.n_samples_per_class
        return full.subset(idx[~test]), full.subset(idx[test])
    raise ValueError(f"unknown dataset {cfg.name}")
