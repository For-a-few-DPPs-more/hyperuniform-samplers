"""
qmc_geom.py
-----------
Geometric utilities for Fourier mode selection on the unit torus.

Provides:
  - classical L1-ball construction (half-space: one vector of each pair {k, -k})
  - get_wave_vectors interface (L1 only; the "wd" mode of qmc_sampler does not
    use Fourier modes, it differentiates the exact WD discrepancy)
  - wd_discrepancy_sq: float64 numpy reference of the exact WD^2, for checks

Only one vector from each pair {k, -k} is retained: the first nonzero
coordinate must be positive.
"""
from __future__ import annotations

from math import comb
from typing import Optional

import numpy as np


CHI = 0.5
EPSILON = 1e-12


def _n_modes_target(N: int, D: int, chi: float = CHI) -> int:
    return max(0, int(round(chi * D * (N - 1))))


# ------------------------------------------------------------------
# L1 helpers (unchanged)
# ------------------------------------------------------------------
def _count_l1_ball(D, radius):
    return sum((2 ** j) * comb(D, j) * comb(radius, j)
               for j in range(min(D, radius) + 1))

def _full_l1_ball(D, radius):
    if D == 1:
        return np.arange(-radius, radius + 1, dtype=np.int32)[:, None]
    prev = _full_l1_ball(D - 1, radius)
    prev_norm = np.abs(prev.astype(np.int64)).sum(1)
    remaining = radius - prev_norm
    counts = 2 * remaining + 1
    total = int(counts.sum())
    wdts = np.cumsum(counts, dtype=np.int64) - counts
    row_ids = np.repeat(np.arange(len(prev), dtype=np.int64), counts)
    z = np.arange(total, dtype=np.int64) - np.repeat(wdts, counts)
    z -= np.repeat(remaining, counts)
    return np.concatenate([prev[row_ids], z.astype(np.int32)[:, None]], 1)

def _half_l1_ball(D, radius):
    if radius <= 0:
        return np.empty((0, D), dtype=np.int32)
    if D == 1:
        return np.arange(1, radius + 1, dtype=np.int32)[:, None]
    prev = _full_l1_ball(D - 1, radius)
    prev_norm = np.abs(prev.astype(np.int64)).sum(1)
    max_z = radius - prev_norm
    counts = max_z
    total = int(counts.sum())
    if total:
        wdts = np.cumsum(counts, dtype=np.int64) - counts
        row_ids = np.repeat(np.arange(len(prev), dtype=np.int64), counts)
        z = np.arange(total, dtype=np.int64) - np.repeat(wdts, counts) + 1
        positive_part = np.concatenate(
            [prev[row_ids], z.astype(np.int32)[:, None]], 1)
    else:
        positive_part = np.empty((0, D), dtype=np.int32)
    previous_half = _half_l1_ball(D - 1, radius)
    zero_part = np.concatenate(
        [previous_half, np.zeros((len(previous_half), 1), dtype=np.int32)], 1)
    return np.concatenate([positive_part, zero_part], 0)


def get_wave_vectors(
    N: int,
    D: int,
    mode: str = "l1",
    chi: float = CHI,
    rng: Optional[np.random.Generator] = None,
    epsilon: float = EPSILON,
):
    """Half-space L1 wave vectors. Returns (K, |K|_1, weights, M)."""
    rng = np.random.default_rng() if rng is None else rng
    M_target = _n_modes_target(N, D, chi)
    if M_target == 0:
        empty = np.empty((0, D), dtype=np.int32)
        return empty, np.empty(0, dtype=np.int64), np.empty(0), 0
    if mode == "l1":
        radius = 1
        while (_count_l1_ball(D, radius) - 1) // 2 < M_target:
            radius *= 2
        low, high = max(1, radius // 2), radius
        while low < high:
            mid = (low + high) // 2
            if (_count_l1_ball(D, mid) - 1) // 2 >= M_target:
                high = mid
            else:
                low = mid + 1
        radius = low
        K = _half_l1_ball(D, radius)
        score = np.abs(K).sum(1)
        order = np.lexsort((rng.random(len(score)), score))[:M_target]
        K = K[order]
        raw = 1.0 / (score[order].astype(np.float64) ** 2 + epsilon)
        weights = raw / raw.max()
    else:
        raise ValueError(f"Unknown mode '{mode}'. Only 'l1' is available here.")
    K = np.ascontiguousarray(K.astype(np.int32))
    kl1 = np.abs(K).sum(1).astype(np.int64)
    return K, kl1, weights.astype(np.float64), len(K)


# ------------------------------------------------------------------
# Exact WD discrepancy (reference implementation, float64, O(N^2 D))
# ------------------------------------------------------------------
def wd_discrepancy_sq(x: np.ndarray, block: int = 256) -> float:
    """
    Squared wrap-around L2 discrepancy of x in [0,1)^D (same quantity as
    scipy.stats.qmc.discrepancy(x, method="WD") up to the sqrt convention):

        WD^2 = -(4/3)^D + 1/N^2 sum_{i,j} prod_d (3/2 - |t| (1 - |t|)),
        t = x_id - x_jd.
    """
    x = np.asarray(x, dtype=np.float64)
    N, D = x.shape
    c = 4.0 / 3.0
    total = 0.0
    for i0 in range(0, N, block):
        t = np.abs(x[i0:i0 + block, None, :] - x[None, :, :])
        total += float(np.sum(np.prod(1.5 - t * (1.0 - t), axis=2) - c ** D))
    return total / N ** 2