"""
qmc_geom.py
-----------
Geometric utilities for Fourier mode selection on the unit torus.

Provides:
  - product weights of the wrap-around discrepancy kernel
  - half-space star-ball construction via signed-permutation orbits
  - classical L1-ball construction
  - unified get_wave_vectors interface
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations, permutations, product
from math import comb, factorial, pi
from typing import Optional, Tuple, List, Iterator
import heapq
import numpy as np

CHI     = 0.5
EPSILON = 1e-12
TWO_PI2     = 2.0 * pi * pi
INV_TWO_PI2 = 1.0 / TWO_PI2

_HATK = [1.0] + [INV_TWO_PI2 / (m * m) for m in range(1, 16384)]

def _hatK(m: int) -> float:
    m = abs(int(m))
    if m < len(_HATK):
        return _HATK[m]
    return INV_TWO_PI2 / (m * m)

def _n_modes_target(N: int, D: int, chi: float = CHI) -> int:
    return max(0, int(round(chi * D * (N - 1))))

def star_weight(K: np.ndarray) -> np.ndarray:
    absK = np.abs(K).astype(np.float64)
    w = np.ones(len(K), dtype=np.float64)
    nz = absK > 0
    if nz.any():
        factors = np.where(nz, INV_TWO_PI2 / (absK ** 2), 1.0)
        w = factors.prod(axis=1)
    return w

def _half_orbit_card(k_pos: Tuple[int, ...], D: int) -> int:
    s = len(k_pos)
    if s == 0 or s > D:
        return 0
    cnt = Counter(k_pos)
    denom = factorial(D - s)
    for v in cnt.values():
        denom *= factorial(v)
    return (factorial(D) // denom * (1 << s)) // 2

def _positive_fundamentals(s: int, max_freq: int) -> List[Tuple[float, Tuple[int, ...]]]:
    items = []
    def rec(prefix, last, remaining):
        if remaining == 0:
            k = tuple(prefix)
            w = 1.0
            for v in k:
                w *= _hatK(v)
            items.append((w, k))
            return
        for v in range(min(last, max_freq), 0, -1):
            rec(prefix + [v], v, remaining - 1)
    for first in range(max_freq, 0, -1):
        rec([first], first, s - 1)
    return items

# decreasing max_freq with support – keeps generation cheap
_S_MAX_FREQ = {1: 256, 2: 64, 3: 32, 4: 16, 5: 10, 6: 6, 7: 4, 8: 3}

def _select_orbits(D: int, M: int):
    s_max = min(8, D)
    heap: List[Tuple[float, int, Tuple[int, ...]]] = []
    counter = 0
    for s in range(1, s_max + 1):
        mf = _S_MAX_FREQ.get(s, 3)
        for w, k in _positive_fundamentals(s, mf):
            heapq.heappush(heap, (-w, counter, k))
            counter += 1

    selected = []
    cum = 0
    last = None
    while heap and cum < M:
        neg_w, _, k = heapq.heappop(heap)
        w = -neg_w
        hsize = _half_orbit_card(k, D)
        if hsize == 0:
            continue
        if cum + hsize >= M:
            last = (w, k, hsize)
            break
        selected.append((w, k, hsize))
        cum += hsize
    return selected, last

def _expand_half_orbit_pos(k_pos: Tuple[int, ...], D: int,
                           max_needed: Optional[int] = None) -> Iterator[Tuple[int, ...]]:
    s = len(k_pos)
    if s == 0:
        return
    count = 0
    unique_vals = list(set(permutations(k_pos)))
    # combinations in lex order → deterministic
    for pos in combinations(range(D), s):
        for vals in unique_vals:
            for signs in product((-1, 1), repeat=s):
                q = [0] * D
                for p, v, sg in zip(pos, vals, signs):
                    q[p] = v * sg
                first = next((i for i, v in enumerate(q) if v != 0), None)
                if first is not None and q[first] > 0:
                    yield tuple(q)
                    count += 1
                    if max_needed is not None and count >= max_needed:
                        return

def half_star_ball(D: int, M: int, max_freq: int = 128):
    """max_freq is accepted for API compatibility but ignored (internal table used)."""
    if M <= 0:
        return np.empty((0, D), dtype=np.int32), np.empty(0, dtype=np.float64)
    if D == 1:
        K = np.arange(1, M + 1, dtype=np.int32)[:, None]
        r = np.array([_hatK(m) for m in range(1, M + 1)], dtype=np.float64)
        return K, r

    selected, last = _select_orbits(D, M)

    K_list: List[Tuple[int, ...]] = []
    r_list: List[float] = []

    for w, k, _ in selected:
        for q in _expand_half_orbit_pos(k, D):
            K_list.append(q)
            r_list.append(w)

    if last is not None:
        w, k, _ = last
        needed = M - len(K_list)
        for q in _expand_half_orbit_pos(k, D, max_needed=needed):
            K_list.append(q)
            r_list.append(w)

    K = np.asarray(K_list, dtype=np.int32)
    r = np.asarray(r_list, dtype=np.float64)
    if len(K) > M:
        K = K[:M]
        r = r[:M]
    return K, r

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

def get_wave_vectors(
    N: int,
    D: int,
    mode: str = "star",
    chi: float = CHI,
    rng: Optional[np.random.Generator] = None,
    epsilon: float = EPSILON,
):
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