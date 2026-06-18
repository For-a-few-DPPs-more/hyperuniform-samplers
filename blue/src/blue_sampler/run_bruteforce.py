"""
blue noise sampling solver.
samples random hyperuniform (= sub-poisson density flucation) point clouds (N, D)
hyperuniformity is achieved through standard gradient descent on energy kernels
"""


from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from .math_utils import torus_wrap

from .kernels import gauss_kernel, gauss_sin_kernel

from .progress import  _LevelCtx

# ── Bruteforce (small N) ──────────────────────────────────────────────────────

def _bruteforce_pipeline(N: int, D: int, ctx: _LevelCtx):
    """AOT-compile a gradient-descent sampler for N ≤ ~3 000 points."""
    DX     = 1.0 / N ** (1.0 / D)
    S      = 1.0
    sigma2 = S * 2.0 * DX ** 2
    high_D = sigma2 >= 0.03

    lr_table = {2: 0.4, 3: 0.1, 4: 0.05, 5: 0.01}
    lr    = lr_table.get(D, 0.01) / S
    Niter = 1_000 if high_D else 3_000

    if high_D:
        a = 2.0 * np.pi
        b = 2.0 / (sigma2 * a ** 2)
        c = 1.0 / (2.0 * S * np.pi)
        kernel = lambda x, y: gauss_sin_kernel(x, y, a, b, c)
    else:
        kernel = lambda x, y: gauss_kernel(x, y, sigma2)

    def grad(x):
        return jax.vmap(lambda xi: kernel(xi[None], x).sum(axis=0))(x)

    @jax.jit
    def _run(x):
        def step(_, x):
            return torus_wrap(x - lr * grad(x))
        return jax.lax.fori_loop(0, Niter, step, x)

    ctx.on_compile()
    compiled = _run.lower(jax.ShapeDtypeStruct((N, D), jnp.float32)).compile()

    def sample_fn(init: np.ndarray | None = None) -> jnp.ndarray:
        ctx.on_bruteforce_start()
        if init is None:
            init = np.random.rand(N, D)
        out = compiled(jnp.asarray(init))
        out.block_until_ready()
        ctx.on_bruteforce_done()
        return out

    return sample_fn