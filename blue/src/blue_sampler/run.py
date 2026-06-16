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
    simplex,
    grid_shape,
    torus_wrap,
    clean_points,
    prepare_wave_vectors,
    prepare_points,
    random_rotations,
)
from .kernels import (
    gauss_kernel,
    gauss_sin_kernel,
    spectral_kernel,
)

from .progress import ProgressLogger, _LevelCtx

# ── Bruteforce (small N) ──────────────────────────────────────────────────────

def _build_bruteforce(N: int, D: int, ctx: _LevelCtx):
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


# ── Core pipeline ─────────────────────────────────────────────────────────────

def _run_pipeline(
    N: int,
    D: int,
    N_ITER: int,
    logger: ProgressLogger,
    *,
    x: np.ndarray | None = None,
    S: float,
    expension_factor: float,
    LR_spatial: float,
    LR_spectral: float,
    spatial_radius: int,
    spectral_radius: int,
    N_PER_STEP: int,
    _is_root: bool = False,
    _is_leaf: bool = True,
) -> np.ndarray:
    """Recursive stealthy-sampling pipeline. Spawns child pipelines when N is large."""
    ctx = logger.enter_level(N, D, N_ITER)

    try:
        Dsimp      = min(D, 3)
        IJK, _, Axes = grid_shape(N, D)
        Nsqrt      = N ** 0.5
        Ncbrt      = N ** (1.0 / D)
        is_root    = _is_root or (N <= 2_000) or (x is not None)
        sigma2     = S * 2.0 * (1.0 / Ncbrt) ** 2
        high_D     = sigma2 >= 0.03

        SHIFTS        = integers_in_half_ball(spatial_radius, D)
        Ks            = integers_in_half_ball(spectral_radius, D)
        K_w, K_       = prepare_wave_vectors(Ks)
        Clone_simplex = simplex(Dsimp)


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

        def macro_grad(x_val):
            x_flat = x_val.reshape(-1, D + 1)
            def body(acc, args):
                k, k_ = args
                return acc + spectral_kernel(x_flat, k, k_), None
            out, _ = jax.lax.scan(body, jnp.zeros_like(x_flat), (K_w[:, 0], K_[:, 0]))
            return out.reshape(*IJK, D + 1)

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

        def clone(x_val: np.ndarray) -> np.ndarray:
            """Expand N//(Dsimp+1) parents into N children via simplex offsets."""
            x_val     = x_val.reshape(-1, D + 1)
            x_val     = x_val[np.isfinite(x_val[:, -1]), :D]
            x_val     = np.random.permutation(x_val)
            N_parents = N // (Dsimp + 1)
            N_keep    = N - (Dsimp + 1) * N_parents
            offsets   = random_rotations(Clone_simplex, N_parents, D, Dsimp) * (expension_factor / Ncbrt)
            children  = (x_val[:N_parents, None, :] + offsets).reshape(-1, D)
            if N_keep > 0:
                children = np.concatenate([x_val[N_parents:], children], axis=0)
            return np.asarray(torus_wrap(jnp.array(children)))
        
        def full_grad(x):
            return (LR_spatial / S) * micro_grad(x) +(LR_spectral / (Nsqrt * Ncbrt)) * macro_grad(x)

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

        if is_root:
            xparent = _build_bruteforce(N, D, ctx)(x)
            x_pts   = prepare_points(np.asarray(xparent), N, IJK, D)
        else:
            N_child = N // (Dsimp + 1) + N % (Dsimp + 1)
            xparent = clone(
                _run_pipeline(
                    N=N_child,
                    D=D,
                    logger=logger,
                    S=S,
                    expension_factor=expension_factor,
                    LR_spatial=LR_spatial,
                    LR_spectral=LR_spectral,
                    spatial_radius=spatial_radius,
                    spectral_radius=spectral_radius,
                    N_ITER=N_ITER,
                    N_PER_STEP=N_PER_STEP,
                    _is_root=False,
                    _is_leaf=False,
                )
            )
            x_pts = prepare_points(xparent, N, IJK, D)
            x_pts = run_iters(x_pts)
            ctx.done()

        if _is_leaf:
            x_pts = np.array(x_pts.reshape(-1, D + 1))
            x_pts = x_pts[np.isfinite(x_pts[:, -1]), :D]

        return x_pts

    finally:
        logger.exit_level()