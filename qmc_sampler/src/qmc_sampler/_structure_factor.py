"""
Structure factor estimation. Adapted from blue_sampler/structure_factor.py (copied so that
qmc_sampler does not depend on blue_sampler).

Wave vectors: exhaustive grid for D <= 3, lattice + random sampling for D >= 4. Fourier sums
with the backend selected for dimension D (see qmc_sampler.select_backend), like the samplers.
"""

from __future__ import annotations


import numpy as np

from ._backend import nudft1
from ._lattice import integer_grid, isotropic_wave_vectors

K_MAX = 2.0                     # in units of the inter-particle wave number
LOW_MODE_FRACTION = 0.25        # D >= 4: share of exhaustive low modes
RADIAL_BINS = {1: (10, 50), 2: (5, 25), 3: (2, 6)}   # (width, start) in |n|^2
SMOOTHING = 0.01                # gaussian smoothing width, fraction of the log-k range
RESOLUTION_DIVISOR = 10         # D >= 4: wave vectors evaluated = resolution / divisor

__all__ = [
    "structure_factor",
    "structure_factor_and_average",
]

# ---------------------------------------------------------------------------
# Fourier sums F(n) = sum_j c_j exp(2i pi n.x_j)
# ---------------------------------------------------------------------------

def _grid(n_modes: int, D: int) -> np.ndarray:
    """Wave vectors of a centred cube of n_modes^D modes."""
    return integer_grid(np.arange(-(n_modes // 2), n_modes - (n_modes // 2)), D)


def _fourier_sums(pts, c, kint, precision):
    """F(n) = sum_j c_j exp(2i pi n.x_j) for every row n of kint (backend of dimension D)."""
    real = np.float64 if precision == "float64" else np.float32
    return nudft1(np.ascontiguousarray(pts, dtype=real), c, kint, isign=1)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _structure_factor_core(points, resolution, precision):
    """Return (kint, S, n_exp): wave vectors of the analysed window, S, expected count rho|W|."""
    pts = np.asarray(points, dtype=float).reshape(-1, np.asarray(points).shape[-1])
    N, D = pts.shape

    n_exp = float(N)
    weights = np.ones(len(pts))

    kunit = float(n_exp ** (1.0 / D))
    kmax = K_MAX * kunit
    precision = "float64" if precision in ("float64", "double") else "float32"

    if D <= 3:
        kint = _grid(int(np.ceil(kmax)) + 1, D)
    else:
        k_low = max(int((resolution * LOW_MODE_FRACTION) ** (1.0 / D)), 1)
        if kmax <= k_low:
            kmax = k_low + 1
        kint = isotropic_wave_vectors(k_low, kmax, D, int(resolution * (1.0 - LOW_MODE_FRACTION)))
    F = _fourier_sums(pts, weights, kint, precision)

    S = np.abs(F) ** 2 / n_exp

    isnt0 = ~np.all(kint == 0, axis=1)
    kint, S = kint[isnt0], S[isnt0]
    order = np.argsort((kint ** 2).sum(axis=1), kind="stable")
    return kint[order], S[order], n_exp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def structure_factor(
    points: np.ndarray,
    resolution: int = 2000,
    precision="float32",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate S(k) via scattering intensity.

    Parameters
    ----------
    points         : (N, D) array of point coordinates in [0, 1)^D.
    resolution     : number of sampled wave vectors (only used when D >= 4).
    precision      : "float32" | "float64". Use float64 to resolve S below ~1e-6
                     With the (cu)finufft backend, the relative accuracy of the Fourier
                     sums is FINUFFT_EPS[precision] (1e-4 / 1e-8), which sets a floor
                     on the smallest S that can be resolved.

    Returns
    -------
    kint : (M, D) int array of wave vectors of the analysed window (k = 2 pi kint),
           sorted by norm. With trust_boundary=False they refer to the rescaled
           truncated window.
    S    : (M,) array.

    Note
    ----
    Points must lie in the unit hypercube [0, 1)^D. This is not checked.
    """
    kint, S, _ = _structure_factor_core(points, resolution, precision)
    return kint, S


def structure_factor_and_average(
    points,
    resolution: int = 20000,
    min_val: float = 1e-20,
    precision="float32",
):
    """
    Compute the structure factor of a point set and return raw and radially-averaged curves.

    ``trust_boundary`` behave as in
    ``structure_factor``.

    Returns
    -------
    kraw, Sraw     : normalised wave numbers and raw S values (subsampled for display).
    kgroup, Sgroup : radially averaged then log-k smoothed curve.

    Wave numbers are normalised so that k = 1 corresponds to |n| = (rho |W|)^(1/D),
    i.e. to the inter-particle distance.
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    D = pts.shape[1]
    kint, Sraw, n_exp = _structure_factor_core(pts, resolution // RESOLUTION_DIVISOR, precision)
    kunit = n_exp ** (1.0 / D)
    k2_ = (kint ** 2).sum(axis=1)

    if D in RADIAL_BINS:
        groupbin, groupstart = RADIAL_BINS[D]
        mask = k2_ >= groupstart
        k2_[mask] = groupbin * (k2_[mask] // groupbin)

    kraw = np.sqrt(k2_) / kunit
    k2group_, inverse = np.unique(k2_, return_inverse=True)
    counts = np.bincount(inverse)
    Sgroup = (np.bincount(inverse, weights=Sraw) / counts).clip(min=min_val)
    kgroup = np.sqrt(k2group_) / kunit
    Sraw = Sraw.clip(min=min_val)

    if len(kraw) >= resolution:
        p = kraw ** (-D)
        p *= resolution / p.sum()
        keep = np.random.random(len(kraw)) < p
        kraw, Sraw = kraw[keep], Sraw[keep]

    logk = np.log(kgroup)
    logk_u = np.linspace(logk[0], logk[-1], 1000)
    S_u = np.interp(logk_u, logk, Sgroup)
    sigma_pixels = (logk[-1] - logk[0]) * SMOOTHING / (logk_u[1] - logk_u[0])
    Sgroup = np.interp(logk, logk_u, gaussian_filter1d(S_u, sigma_pixels, truncate=4.0)).clip(min=min_val)
    Sraw = Sraw.clip(min=Sgroup.min())
    return kraw, Sraw, kgroup, Sgroup


def gaussian_filter1d(x, sigma, truncate=4.0):
    """Gaussian smoothing of a 1D array using a truncated kernel."""
    radius = int(np.ceil(truncate * sigma))
    if radius == 0:
        return x.copy()
    kernel_x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= kernel.sum()
    xp = np.pad(x, radius, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(xp, 2 * radius + 1)
    return windows @ kernel
