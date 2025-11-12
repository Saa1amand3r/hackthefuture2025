# src/pipeline/weights.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal

Scheme = Literal["inv_freq", "tail_focus"]

@dataclass
class WeightingConfig:
    scheme: Scheme = "inv_freq"
    n_bins: int = 10           # number of bins for y
    power: float = 1.0         # amplify weights (e.g., 1.0 normal, 1.5 stronger)
    tails: float = 0.1         # for tail_focus: fraction per tail (0.1 => 10% low + 10% high)
    normalize: bool = True     # normalize sum(weights) == len(y)

    def as_kwargs(self) -> dict:
        return dict(
            scheme=self.scheme,
            n_bins=self.n_bins,
            power=self.power,
            tails=self.tails,
            normalize=self.normalize,
        )

def _bin_edges_quantiles(y: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y, qs))
    # ensure at least 3 edges (>= 2 bins)
    if len(edges) < 3:
        # fallback to min/max
        edges = np.array([y.min(), y.max()])
        if np.allclose(edges[0], edges[1]):
            # degenerate y
            edges = np.array([y.min() - 1e-9, y.max() + 1e-9, y.max() + 2e-9])
        else:
            edges = np.array([edges[0], (edges[0] + edges[1]) / 2, edges[1]])
    return edges

def inverse_freq_weights(y: np.ndarray, n_bins: int = 10, power: float = 1.0, normalize: bool = True) -> np.ndarray:
    y = np.asarray(y)
    edges = _bin_edges_quantiles(y, n_bins)
    # bin indices in [0 .. n_bins-1]
    b = np.digitize(y, edges[1:-1], right=True)
    # frequencies per bin
    counts = np.bincount(b, minlength=len(edges) - 1).astype(float)
    freq = counts[b]
    w = 1.0 / np.clip(freq, 1.0, None)
    if power != 1.0:
        w = np.power(w, power)
    if normalize:
        w *= len(w) / w.sum()
    return w

def tail_focus_weights(y: np.ndarray, tails: float = 0.1, power: float = 1.0, normalize: bool = True) -> np.ndarray:
    """Upweight lower and upper tails; center stays near 1."""
    y = np.asarray(y)
    lo = np.quantile(y, tails)
    hi = np.quantile(y, 1.0 - tails)
    w = np.ones_like(y, dtype=float)
    w[y <= lo] = 1.0 / max(y[y <= lo].size, 1)   # inverse count in lower tail
    w[y >= hi] = 1.0 / max(y[y >= hi].size, 1)   # inverse count in upper tail
    if power != 1.0:
        w = np.power(w, power)
    if normalize and w.sum() > 0:
        w *= len(w) / w.sum()
    return w

def make_sample_weights(y: np.ndarray, scheme: Scheme = "inv_freq", n_bins: int = 10,
                        power: float = 1.0, tails: float = 0.1, normalize: bool = True) -> np.ndarray:
    if scheme == "inv_freq":
        return inverse_freq_weights(y, n_bins=n_bins, power=power, normalize=normalize)
    elif scheme == "tail_focus":
        return tail_focus_weights(y, tails=tails, power=power, normalize=normalize)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
