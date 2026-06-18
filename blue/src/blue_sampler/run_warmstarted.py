"""
blue noise sampling solver.
samples random hyperuniform (= sub-poisson density flucation) point clouds (N, D)
hyperuniformity is achieved through standard gradient descent on energy kernels
"""

from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from squarenet import SquareNet

from .math_utils import (
    integers_in_half_ball,
    grid_shape,
    torus_wrap,
    clean_points,
    prepare_points,
)

from .kernels import (
    gauss_kernel,
    gauss_sin_kernel,
)

from .progress import ProgressLogger

# ── Core pipeline ─────────────────────────────────────────────────────────────

def _warmstarted_pipeline(
    N: int,
    D: int,
    N_ITER: int,
    logger: ProgressLogger,
    *,
    x: np.ndarray,
    S: float,
    LR_spatial: float,
    spatial_radius: int,
    N_PER_STEP: int,
) -> np.ndarray:
    """
    use sobol as an init for gradient descent
    """
    ctx = logger.enter_level(N, D, N_ITER)

    try:
        IJK, _, Axes = grid_shape(N, D)
        Ncbrt  = N ** (1.0 / D)
        sigma2 = S * 2.0 * (1.0 / Ncbrt) ** 2
        high_D = sigma2 >= 0.03

        SHIFTS = integers_in_half_ball(spatial_radius, D)

        if high_D:
            a = 2.0 * np.pi
            b = 2.0 / (sigma2 * a ** 2)
            c = 1.0 / (2.0 * S * np.pi)
            micro_kernel = lambda x_val, y_val: gauss_sin_kernel(x_val, y_val, a, b, c)
        else:
            micro_kernel = lambda x_val, y_val: gauss_kernel(x_val, y_val, sigma2)

        def micro_grad(x_val):
            def body(acc, shift):
                contrib = micro_kernel(x_val, jnp.roll(x_val, shift, axis=Axes))
                return acc + contrib - jnp.roll(contrib, -shift, axis=Axes), None
            out, _ = jax.lax.scan(body, jnp.zeros_like(x_val), SHIFTS)
            return out

        full_grad = lambda x_val: (LR_spatial / S) * micro_grad(x_val)

        sn = SquareNet(gridshape=IJK, max_iter=50, verbose=0)

        def _gridify_numpy(x_val: np.ndarray) -> np.ndarray:
            ctx.tick()
            flat = torus_wrap(np.random.permutation(x_val.reshape(-1, D + 1)) - 0.5)
            sn.fit(flat[:, :D], method="ultimate")
            return sn.map(flat)

        def gridify(x_val: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(
                _gridify_numpy,
                jax.ShapeDtypeStruct(x_val.shape, x_val.dtype),
                x_val,
            )

        @jax.jit
        def run_iters(x_val: jnp.ndarray) -> jnp.ndarray:
            def step(i, x_val):
                x_val = jax.lax.cond(
                    i % N_PER_STEP == 0,
                    gridify,
                    lambda val: val,
                    x_val,
                )
                return clean_points(torus_wrap(x_val - full_grad(x_val)))
            return jax.lax.fori_loop(0, N_ITER * N_PER_STEP, step, x_val)

        x_init = np.asarray(x)

        x_pts = prepare_points(x_init, N, IJK, D)
        x_pts = run_iters(x_pts)
        ctx.done()

        x_pts = np.array(x_pts.reshape(-1, D + 1))
        x_pts = x_pts[np.isfinite(x_pts[:, -1]), :D]

        return x_pts

    finally:
        logger.exit_level()