"""
qmc_geom.py
-----------
Geometric utilities for Fourier mode selection on the unit torus.

Provides:
  - exact product weights of the wrap-around kernel (star discrepancy)
  - half-space star-ball construction via signed-permutation orbits
  - classical L1-ball construction
  - unified get_wave_vectors interface
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations, product
from math import comb, factorial, pi
from typing import Optional, Tuple

import numpy as np

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CHI     = 0.5
EPSILON = 1e-12

TWO_PI2     = 2.0 * pi * pi
INV_TWO_PI2 = 1.0 / TWO_PI2


def _n_modes_target(N: int, D: int, chi: float = CHI) -> int:
    """Number of independent Fourier modes to keep."""
    return max(0, int(round(chi * D * (N - 1))))


# ------------------------------------------------------------------
# 1-D Fourier coefficient of the wrap-around kernel
# ------------------------------------------------------------------
def hatK(m: int) -> float:
    m = abs(int(m))
    return 1.0 if m == 0 else INV_TWO_PI2 / (m * m)


def star_weight(K: np.ndarray) -> np.ndarray:
    """Exact product weight r(k) = ∏_d hatK(k_d) for a batch of multi-indices."""
    absK = np.abs(K).astype(np.float64)
    w = np.ones(len(K), dtype=np.float64)
    nz = absK > 0
    if nz.any():
        factors = np.where(nz, INV_TWO_PI2 / (absK ** 2), 1.0)
        w = factors.prod(axis=1)
    return w


# ------------------------------------------------------------------
# Combinatorial helpers for signed-permutation orbits
# ------------------------------------------------------------------
def _orbit_card(k: Tuple[int, ...]) -> int:
    """Full signed-permutation orbit size of a non-negative multi-index."""
    cnt = Counter(k)
    denom = 1
    for v in cnt.values():
        denom *= factorial(v)
    n_nz = sum(1 for v in k if v != 0)
    return (factorial(len(k)) // denom) * (1 << n_nz)


def _half_orbit_card(k: Tuple[int, ...]) -> int:
    """Number of independent modes (half-space) produced by the orbit."""
    return _orbit_card(k) // 2


# ------------------------------------------------------------------
# Generation of non-increasing non-negative multi-indices
# ------------------------------------------------------------------
def _fundamentals(D: int, max_freq: int):
    """
    Return all non-increasing multi-indices k1 ≥ k2 ≥ … ≥ kD ≥ 0
    (origin excluded), ordered by decreasing product weight.
    Each item is (weight, k_tuple, half_orbit_card).
    """
    items = []

    def rec(prefix, last, remaining):
        if remaining == 0:
            k = tuple(prefix)
            w = 1.0
            for v in k:
                w *= hatK(v)
            items.append((w, k, _half_orbit_card(k)))
            return
        for v in range(last, -1, -1):
            rec(prefix + [v], v, remaining - 1)

    for first in range(max_freq, 0, -1):
        rec([first], first, D - 1)

    items.sort(key=lambda t: -t[0])
    return items


# ------------------------------------------------------------------
# Deterministic expansion of an orbit (lazy)
# ------------------------------------------------------------------
def _expand_half_orbit(k: Tuple[int, ...], max_needed: Optional[int] = None):
    """
    Yield the half-space representatives of the signed-permutation orbit of k
    in a deterministic order. Stops after `max_needed` elements if given.
    """
    vals = list(k)
    unique_perms = set(permutations(vals))
    count = 0

    for p in unique_perms:
        nonzero_idx = [i for i, v in enumerate(p) if v != 0]
        n_nz = len(nonzero_idx)

        for signs in product([-1, 1], repeat=n_nz):
            q = list(p)
            for i, s in zip(nonzero_idx, signs):
                q[i] *= s

            # half-space convention: first non-zero coordinate > 0
            first = next((i for i, v in enumerate(q) if v != 0), None)
            if first is not None and q[first] > 0:
                yield tuple(q)
                count += 1
                if max_needed is not None and count >= max_needed:
                    return


# ------------------------------------------------------------------
# Star-ball construction
# ------------------------------------------------------------------
def half_star_ball(D: int, M: int, max_freq: int = 64):
    """
    Exact top-M independent modes according to the product weight r(k).

    1. Enumerate fundamental representatives ordered by decreasing weight.
    2. Accumulate orbit cardinalities until the cumulative size exceeds M.
    3. Expand every complete orbit fully.
    4. Expand only the first (M - previous) representatives of the last orbital
       (deterministic order, no full materialisation).
    """
    if M <= 0:
        return np.empty((0, D), dtype=np.int32), np.empty(0, dtype=np.float64)

    fund = _fundamentals(D, max_freq)

    # find the last orbital
    selected = []
    cum = 0
    last_orbital = None
    for w, k, hsize in fund:
        if cum + hsize >= M:
            last_orbital = (w, k, hsize)
            break
        selected.append((w, k, hsize))
        cum += hsize
    else:
        last_orbital = None

    # expand complete layers
    K_list = []
    r_list = []

    for w, k, _ in selected:
        for q in _expand_half_orbit(k):
            K_list.append(q)
            r_list.append(w)

    # expand (possibly partial) last orbital
    if last_orbital is not None:
        w, k, hsize = last_orbital
        needed = M - len(K_list)
        for q in _expand_half_orbit(k, max_needed=needed):
            K_list.append(q)
            r_list.append(w)

    K = np.asarray(K_list, dtype=np.int32)
    r = np.asarray(r_list, dtype=np.float64)

    if len(K) > M:
        K = K[:M]
        r = r[:M]

    return K, r


# ------------------------------------------------------------------
# Classical L1-ball helpers
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
    starts = np.cumsum(counts, dtype=np.int64) - counts
    row_ids = np.repeat(np.arange(len(prev), dtype=np.int64), counts)
    z = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
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
        starts = np.cumsum(counts, dtype=np.int64) - counts
        row_ids = np.repeat(np.arange(len(prev), dtype=np.int64), counts)
        z = np.arange(total, dtype=np.int64) - np.repeat(starts, counts) + 1
        positive_part = np.concatenate(
            [prev[row_ids], z.astype(np.int32)[:, None]], 1)
    else:
        positive_part = np.empty((0, D), dtype=np.int32)
    previous_half = _half_l1_ball(D - 1, radius)
    zero_part = np.concatenate(
        [previous_half, np.zeros((len(previous_half), 1), dtype=np.int32)], 1)
    return np.concatenate([positive_part, zero_part], 0)


# ------------------------------------------------------------------
# Unified interface
# ------------------------------------------------------------------
def get_wave_vectors(
    N: int,
    D: int,
    mode: str = "star",
    chi: float = CHI,
    rng: Optional[np.random.Generator] = None,
    epsilon: float = EPSILON,
):
    """
    Select the M most important Fourier modes.

    Parameters
    ----------
    mode : {"star", "l1"}
        "star" – exact WD product weights (recommended)
        "l1"   – classical L1 ball
    chi : float
        Fraction of degrees of freedom (M ≈ chi * D * (N-1))

    Returns
    -------
    K       : (M, D) int32
    kl1     : (M,)   int64
    weights : (M,)   float64  (normalised to max = 1)
    M       : int
    """
    rng = np.random.default_rng() if rng is None else rng
    M_target = _n_modes_target(N, D, chi)

    if M_target == 0:
        empty = np.empty((0, D), dtype=np.int32)
        return empty, np.empty(0, dtype=np.int64), np.empty(0), 0

    if mode == "star":
        K, r = half_star_ball(D, M_target)
        weights = r / (r.max() + epsilon)

    elif mode == "l1":
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
        raise ValueError(f"Unknown mode '{mode}'. Choose 'star' or 'l1'.")

    K = np.ascontiguousarray(K.astype(np.int32))
    kl1 = np.abs(K).sum(1).astype(np.int64)
    return K, kl1, weights.astype(np.float64), len(K)