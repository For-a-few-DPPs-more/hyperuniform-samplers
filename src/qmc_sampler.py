"""
see https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft.py:
same but:
  - L1 ball instead of L2 ball
  - keops backend instead of finufft
"""
import time
from math import comb, pi
import numpy as np
try:
    from pykeops.numpy import LazyTensor
except ImportError:
    class LazyTensor:
        # placeholder
        def __init__(self, *args, **kwargs):
            raise ImportError("keops not imported")

CHI = 0.5
epsilon = 1e-3
precision = "float32"

NITER = 150
LR = 1.0

real_dtype = np.float32 if precision == "float32" else np.float64
complex_dtype = np.complex64 if precision == "float32" else np.complex128

two_pi = 2.0 * pi

def _n_modes_target(N, D):
    return max(0, int(round(CHI * D * (N - 1))))


def _count_l1_ball(D, radius):
    return sum((2 ** j) * comb(D, j) * comb(radius, j) for j in range(min(D, radius) + 1))


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
        positive_part = np.concatenate([prev[row_ids], z.astype(np.int32)[:, None]], 1)
    else:
        positive_part = np.empty((0, D), dtype=np.int32)
    previous_half = _half_l1_ball(D - 1, radius)
    zero_part = np.concatenate([previous_half, np.zeros((len(previous_half), 1), dtype=np.int32)], 1)
    return np.concatenate([positive_part, zero_part], 0)


def get_l1_wave_vectors(N, D, rng=None, epsilon=1e-3):
    rng = np.random.default_rng() if rng is None else rng
    M_target = _n_modes_target(N, D)
    if M_target == 0:
        e = np.empty((0, D), dtype=np.int32)
        return e, np.empty(0, dtype=np.int64), np.empty(0), 0, 0.0
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
    kl1 = np.abs(K.astype(np.int64)).sum(1)
    order = np.lexsort((rng.random(len(kl1)), kl1))[:M_target]
    K = np.ascontiguousarray(K[order])
    kl1 = kl1[order].astype(np.int64)
    raw = 1.0 / (kl1.astype(np.float64) ** 2 + epsilon)
    weights = raw / raw.max()
    return K, kl1, weights, len(K)


def _nufft_pipeline_keops_l1(
    N=1_000, D=2, target=None, seed=None, verbose=1
    ):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(N, D)).astype(real_dtype)
    K, kl1, w, M = get_l1_wave_vectors(N, D, rng, epsilon)
    k_coords = np.ascontiguousarray(K.astype(real_dtype))
    norm = float(max(M, 1))
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
    k_j = LazyTensor(k_coords[None, :, :])

    def fourier_modes(pts):
        xi = LazyTensor(pts[:, None, :])
        phase = two_pi * (xi * k_j).sum(-1)
        fr = np.asarray(phase.cos().sum_reduction(0)).ravel() / N
        fi = -np.asarray(phase.sin().sum_reduction(0)).ravel() / N
        return fr.astype(real_dtype), fi.astype(real_dtype)

    def loss_and_grad(pts):
        fr, fi = fourier_modes(pts)
        dr, di = fr - fk_tr, fi - fk_ti
        loss = float(np.sum(w * (dr**2 + di**2)) / norm)
        pref = 2.0 / norm * (-two_pi) / N
        cr = np.ascontiguousarray((pref * w * dr).astype(real_dtype))
        ci = np.ascontiguousarray((pref * w * di).astype(real_dtype))
        xi = LazyTensor(pts[:, None, :])
        crj = LazyTensor(cr[None, :, None])
        cij = LazyTensor(ci[None, :, None])
        phase = two_pi * (xi * k_j).sum(-1)
        scalar = crj * phase.sin() + cij * phase.cos()
        grad = (scalar * k_j).sum_reduction(1)
        return loss, np.asarray(grad).astype(real_dtype)

    adaptive = LR * 0.1 * N ** (-1.0 / D)
    prev_loss = np.inf
    if verbose:
        print(f"[keops-L1] N={N} D={D}")
    t0 = time.time()
    for it in range(NITER):
        loss, grad = loss_and_grad(x)
        rms = float(np.sqrt(np.mean(grad * grad) + 1e-30))
        x = np.ascontiguousarray(np.mod(x - (grad / rms) * adaptive, 1.0).astype(real_dtype))
        adaptive *= 1.05 if loss < prev_loss else 0.8
        if loss < prev_loss:
            prev_loss = loss
        if verbose:
            print(f"iter {it:4d} | loss {loss:.1e}")
    if verbose:
        print(f"done in {time.time() - t0:.1f}s")
    return np.asarray(x)
