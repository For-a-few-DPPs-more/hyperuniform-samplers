from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import convolve

from .math import clean_grad


# ---------------------------------------------------------------------
# Spatial field
# ---------------------------------------------------------------------

def _make_truncated_grad_kernels(shape, sigma2_kernel, D, n_sigma=4):
    shape = np.asarray(shape)
    sigma_pix = [np.sqrt(sigma2_kernel) * shape[d] for d in range(D)]
    radius = [int(np.ceil(n_sigma * s)) for s in sigma_pix]

    axes = [np.arange(-r, r + 1) for r in radius]
    mesh = np.meshgrid(*axes, indexing="ij")

    r2 = sum((mesh[d] / shape[d]) ** 2 for d in range(D))
    V = np.exp(-r2 / sigma2_kernel)

    return [
        -(2.0 / sigma2_kernel) * (mesh[d] / shape[d]) * V
        for d in range(D)
    ]


def _build_field(target, shape, sigma2_kernel):
    D = target.shape[1]
    shape = np.asarray(shape)

    idx = np.floor(target * shape).astype(np.int64) % shape

    rho = np.zeros(tuple(shape), np.float32)
    np.add.at(rho, tuple(idx[:, d] for d in range(D)), 1.0 / len(target))

    anti_rho = rho.max() - rho

    field = np.empty(tuple(shape) + (D,), np.float32)
    kernels = _make_truncated_grad_kernels(shape, sigma2_kernel, D)

    for d in range(D):
        field[..., d] = convolve(anti_rho, kernels[d], mode="wrap")

    return field


def make_multi_scales_field_fun(target):
    target = np.asarray(target)
    D = target.shape[1]
    cache = {}

    if D == 2:
        shape = (512, 512)
    if D == 3:
        shape = (64, 64, 64)

    def get_field_fn(sigma2_kernel):
        key = (tuple(shape), float(sigma2_kernel))

        if key not in cache:
            cache[key] = jnp.asarray(
                _build_field(target, shape, sigma2_kernel)
            )

        field = cache[key]

        def spatial_grad(x):
            p = x.reshape(-1, D)
            idx = tuple(
                jnp.floor(p[:, d] * shape[d]).astype(jnp.int32) % shape[d]
                for d in range(D)
            )
            return clean_grad(field[idx].reshape(x.shape))

        return spatial_grad

    return get_field_fn


# ---------------------------------------------------------------------
# Spectral gradient (memory-safe)
# ---------------------------------------------------------------------

def prepare_spectral_grad(target, k, k_, chunk_size=8192):
    target = jnp.asarray(target)

    kvec = k[:, 0, : target.shape[1]]
    kval = k_[:, 0]
    D = target.shape[1]
    M = kvec.shape[0]

    phase_tgt = target @ kvec.T
    Sktgt = jnp.sum(jnp.exp(1j * phase_tgt), axis=0)

    def spectral_grad(x):
        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]

        # ---------- Pass 1 : Sk ----------
        def sk_body(i, Sk):
            start = i * chunk_size
            xb = jax.lax.dynamic_slice(
                x_flat,
                (start, 0),
                (min(chunk_size, N - start), D),
            )

            phase = xb @ kvec.T
            return Sk + jnp.sum(jnp.exp(1j * phase), axis=0)

        n_chunks = (N + chunk_size - 1) // chunk_size

        Sk = jax.lax.fori_loop(
            0,
            n_chunks,
            sk_body,
            jnp.zeros(M, dtype=jnp.complex64),
        )

        coeff = Sk - Sktgt

        # ---------- Pass 2 : gradient ----------
        phase = x_flat @ kvec.T
        ek = jnp.exp(-1j * phase)

        grad = jnp.real(
            jnp.sum(coeff[None, :] * kval[None, :] * ek, axis=1)
        )

        return grad.reshape(x.shape)

    return spectral_grad