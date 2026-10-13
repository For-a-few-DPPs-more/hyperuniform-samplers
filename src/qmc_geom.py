"""
qmc_geom.py
-----------
Geometric utilities for Fourier mode selection on the unit torus.
select a L1 half-ball with given number of modes
"""
from __future__ import annotations

from math import comb, pi
from typing import Optional

import numpy as np

# ------------------------------------------------------------------
# Global defaults
# ------------------------------------------------------------------
EPSILON   = 1e-12
PRECISION = "float32"

CHI     = 0.5

real_dtype    = np.float32 if PRECISION == "float32" else np.float64
complex_dtype = np.complex64 if PRECISION == "float32" else np.complex128
two_pi        = 2.0 * pi

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
    chi: float = CHI,
    rng: Optional[np.random.Generator] = None,
    epsilon: float = EPSILON,
):
    rng = np.random.default_rng() if rng is None else rng
    M_target = _n_modes_target(N, D, chi)
    if M_target == 0:
        empty = np.empty((0, D), dtype=np.int32)
        return empty, np.empty(0, dtype=np.int64), np.empty(0), 0

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
    K = np.ascontiguousarray(K.astype(np.int32))
    kl1 = np.abs(K).sum(1).astype(np.int64)
    return K, kl1, weights.astype(np.float64), len(K)

def loss_and_grad_l1(N, D, chi, rng, target):
    from pykeops.numpy import LazyTensor

    K, kl1, w, M = get_wave_vectors(N, D, chi=chi, rng=rng)
    k_coords = np.ascontiguousarray(K.astype(real_dtype))
    norm = float(max(M, 1))

    # optional target spectrum
    if target is not None:
        tx = LazyTensor(target[:, None, :])
        tk = LazyTensor(k_coords[None, :, :])
        phase = two_pi * (tx * tk).sum(-1)
        tr = np.asarray(phase.cos().sum_reduction(0)).ravel()
        ti = -np.asarray(phase.sin().sum_reduction(0)).ravel()
        fk_target = ((tr + 1j * ti) / target.shape[0]).astype(complex_dtype)
    else:
        fk_target = np.zeros(M, dtype=complex_dtype)

    fk_tr = np.ascontiguousarray(fk_target.real.astype(real_dtype))
    fk_ti = np.ascontiguousarray(fk_target.imag.astype(real_dtype))
    k_j   = LazyTensor(k_coords[None, :, :])

    def fourier_modes(pts):
        xi = LazyTensor(pts[:, None, :])
        phase = two_pi * (xi * k_j).sum(-1)
        fr = np.asarray(phase.cos().sum_reduction(0)).ravel() / N
        fi = -np.asarray(phase.sin().sum_reduction(0)).ravel() / N
        return fr.astype(real_dtype), fi.astype(real_dtype)

    def loss_and_grad(pts):
        fr, fi = fourier_modes(pts)
        dr = fr - fk_tr
        di = fi - fk_ti
        loss = float(np.sum(w * (dr ** 2 + di ** 2)) / norm)

        pref = 2.0 / norm * (-two_pi) / N
        cr = np.ascontiguousarray((pref * w * dr).astype(real_dtype))
        ci = np.ascontiguousarray((pref * w * di).astype(real_dtype))

        xi  = LazyTensor(pts[:, None, :])
        crj = LazyTensor(cr[None, :, None])
        cij = LazyTensor(ci[None, :, None])
        phase = two_pi * (xi * k_j).sum(-1)
        scalar = crj * phase.sin() + cij * phase.cos()
        grad = (scalar * k_j).sum_reduction(1)
        return loss, np.asarray(grad).astype(real_dtype)
    return loss_and_grad

def loss_and_grad_wd():
    import torch
    from pykeops.torch import LazyTensor 

    def loss_and_grad(x):
        xj = torch.as_tensor(
            x, dtype=torch.float32
        ).detach()

        xi = torch.as_tensor(
            x, dtype=torch.float32
        ).detach().requires_grad_(True)
        

        xi = LazyTensor(xi[:, None, :])
        xj = LazyTensor(xj[None, :, :])

        delta = (xi - xj).abs()
        loss_val = (1.5 - delta * (1.0 - delta)).prod(-1).sum()

        loss = loss_val.sum()
        loss.backward()

        return loss.item(), xi.grad.detach().cpu().numpy()

    return loss_and_grad