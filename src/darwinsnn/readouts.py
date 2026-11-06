"""Readouts that are *not* used during the search, for analysis.

The search scores networks with a nearest-prototype readout (fitness.py). A
linear readout trained on many samples is a stronger reference. If it scores
far better than the prototype readout, the bottleneck is the readout and not
the network.
"""

from __future__ import annotations

import numpy as np


def _standardize(X_train: np.ndarray, X_test: np.ndarray):
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd


def _ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression weights [n_features + 1, n_classes] (last row = bias)."""
    X = np.hstack([X, np.ones((len(X), 1))])
    n, f = X.shape
    if f <= n:  # primal form
        return np.linalg.solve(X.T @ X + lam * np.eye(f), X.T @ Y)
    return X.T @ np.linalg.solve(X @ X.T + lam * np.eye(n), Y)   # dual form


def _predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (np.hstack([X, np.ones((len(X), 1))]) @ W).argmax(axis=1)


def ridge_accuracy(X_train, y_train, X_test, y_test, n_classes: int,
                   lambdas=(0.1, 1.0, 10.0, 100.0, 1000.0), rng=None) -> float:
    """Test accuracy of a ridge classifier on one-hot targets.

    The regularisation strength is chosen on a 25% validation split of the
    training set, and the final model is refitted on the full training set.
    """
    rng = rng or np.random.default_rng(0)
    Y = np.eye(n_classes)[y_train]
    perm = rng.permutation(len(y_train))
    val, fit = perm[: len(perm) // 4], perm[len(perm) // 4:]
    best_lam, best_acc = lambdas[0], -1.0
    for lam in lambdas:
        Xf, Xv = _standardize(X_train[fit], X_train[val])
        acc = (_predict(_ridge_fit(Xf, Y[fit], lam), Xv) == y_train[val]).mean()
        if acc > best_acc:
            best_lam, best_acc = lam, acc
    Xtr, Xte = _standardize(X_train, X_test)
    return float((_predict(_ridge_fit(Xtr, Y, best_lam), Xte) == y_test).mean())
