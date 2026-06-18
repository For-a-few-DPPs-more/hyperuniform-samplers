"""
KNN-based blue noise sampler — drop-in benchmark alternative to the grid pipeline.

Key differences vs the grid pipeline
─────────────────────────────────────
• Flat tensor  (N, D) throughout  — no (*IJK, D+1) augmented layout.
• micro_grad   computed on the K nearest neighbours of each point (pre-built
  BallTree) instead of torus-roll over a regular grid.
• Periodicity  handled via a trigonometric embedding into R^{2D}:
      φ(x) = [cos(2π x₁), sin(2π x₁), …, cos(2π x_D), sin(2π x_D)]
  The Euclidean distance in R^{2D} equals 2·sin(π·‖Δx‖_torus) — exact, no
  approximation, and BallTree stays fully vectorised.
• macro_grad   (spectral) is unchanged; it already works on a flat (N, D)
  tensor so no adaptation is needed.
• gridify / SquareNet is replaced by a pure-JAX "spread" step that nudges
  colliding points apart (optional, controlled by `use_spread`).

Complexity
──────────
  BallTree query:  O(N · K · log N)  — sklearn default leaf_size=40
  micro_grad JAX:  O(N · K)
  vs grid roll:    O(N · |SHIFTS|)   where |SHIFTS| ≈ 2D·spatial_radius^D

For D ≥ 6 the grid shell blows up; KNN stays manageable.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.neighbors import BallTree

from ..blue.src.blue_sampler.math_utils import (
    integers_in_half_ball,
    torus_wrap,
    prepare_wave_vectors,
)
from ..blue.src.blue_sampler.kernels import (
    gauss_kernel,
    gauss_sin_kernel,
    spectral_kernel,
)

from ..blue.src.blue_sampler.run_recursive import _build_bruteforce
from ..blue.src.blue_sampler.progress import ProgressLogger, _LevelCtx


# ── Trigonometric embedding for toroidal distances ────────────────────────────

def trig_embed(x: np.ndarray) -> np.ndarray:
    """
    Map points on the flat torus [0,1)^D to R^{2D} via
        φ(x) = [cos(2π x_d), sin(2π x_d)]_{d=1..D}

    BallTree Euclidean distance in the embedded space equals
        2 · sin(π · d_torus(x, y))
    which is a valid metric on the torus and monotone in the toric distance.
    """
    theta = 2.0 * np.pi * x          # (N, D)
    return np.concatenate([np.cos(theta), np.sin(theta)], axis=-1)  # (N, 2D)


# ── BallTree index ────────────────────────────────────────────────────────────

def build_balltree(x: np.ndarray, leaf_size: int = 40) -> BallTree:
    """Build a BallTree on the trig-embedded points."""
    return BallTree(trig_embed(x), metric="euclidean", leaf_size=leaf_size)


def query_knn(tree: BallTree, x: np.ndarray, K: int) -> np.ndarray:
    """
    Return neighbour indices (N, K) — each row lists the K nearest neighbours
    of that point (excluding the point itself via [1:K+1]).
    """
    _, idx = tree.query(trig_embed(x), k=K + 1)  # +1 because point queries itself
    return idx[:, 1:]                              # (N, K)  — exclude self


# ── KNN micro-gradient ────────────────────────────────────────────────────────

def _make_knn_micro_grad(kernel_fn, D: int):
    """
    Return a JIT-compiled function
        knn_micro_grad(x, nbr_idx) -> (N, D)

    where nbr_idx is an integer array (N, K) of neighbour indices.

    For each point i we compute:
        g_i = Σ_{j ∈ nbr(i)}  ∇_{x_i} k(x_i, x_j)

    kernel_fn(xi, neighbours) returns the gradient vector (D,) for each
    neighbour pair — it is a vector-valued function, not a scalar kernel.
    """
    @jax.jit
    def knn_micro_grad(x: jnp.ndarray, nbr_idx: jnp.ndarray) -> jnp.ndarray:
        # x         : (N, D)
        # nbr_idx   : (N, K)  integer

        def grad_one(xi, nbr_i):
            # xi    : (D,)
            # nbr_i : (K,)
            neighbours = x[nbr_i]          # (K, D)
            return kernel_fn(xi[None], neighbours).sum(axis=0)   # (D,)

        return jax.vmap(grad_one)(x, nbr_idx)   # (N, D)

    return knn_micro_grad


# ── Core KNN pipeline ─────────────────────────────────────────────────────────

def run_knn_pipeline(
    N: int,
    D: int,
    N_ITER: int,
    logger: ProgressLogger,
    *,
    x: np.ndarray | None = None,
    S: float,
    K: int = 32,                   # number of neighbours for micro_grad
    expension_factor: float,
    LR_spatial: float,
    LR_spectral: float,
    spectral_radius: int,
    N_PER_STEP: int,               # KNN rebuilt every N_PER_STEP gradient steps
    leaf_size: int = 40,           # BallTree leaf size
    use_spread: bool = False,      # optional spread kick to separate colliding points
    _is_root: bool = False,
    _is_leaf: bool = True,
) -> np.ndarray:
    """
    KNN-based stealthy-sampling pipeline.

    The main loop runs N_ITER * N_PER_STEP gradient steps total.
    The neighbour index (expensive BallTree query) is rebuilt once every
    N_PER_STEP steps; the K nearest neighbours are then reused for the
    N_PER_STEP cheap JAX micro_grad steps in between.

    Differences from _run_pipeline
    ───────────────────────────────
    • x is always flat (N, D)  — no augmented D+1 dimension.
    • micro_grad uses a pre-queried (N, K) neighbour index rebuilt every
      N_PER_STEP steps via a BallTree on the trig-embedded coordinates.
    • macro_grad is identical (works on flat x directly).
    • gridify is replaced by an optional lightweight spread kick.
    • clone uses the same simplex logic but operates on flat (N, D) arrays.
    """
    ctx = logger.enter_level(N, D, N_ITER)

    try:
        Dsimp  = min(D, 3)
        Nsqrt  = N ** 0.5
        Ncbrt  = N ** (1.0 / D)
        is_root = _is_root or (N <= 2_000) or (x is not None)
        sigma2  = S * 2.0 * (1.0 / Ncbrt) ** 2
        high_D  = sigma2 >= 0.03

        # ── kernel ──────────────────────────────────────────────────────────
        if high_D:
            a = 2.0 * np.pi
            b = 2.0 / (sigma2 * a ** 2)
            c = 1.0 / (2.0 * S * np.pi)
            kernel_fn = lambda x_val, y_val: gauss_sin_kernel(x_val, y_val, a, b, c)
        else:
            kernel_fn = lambda x_val, y_val: gauss_kernel(x_val, y_val, sigma2)

        knn_micro_grad = _make_knn_micro_grad(kernel_fn, D)

        # ── spectral (macro) grad ────────────────────────────────────────────
        Ks       = integers_in_half_ball(spectral_radius, D)
        K_w, K_  = prepare_wave_vectors(Ks)

        @jax.jit
        def macro_grad(x_flat: jnp.ndarray) -> jnp.ndarray:
            # x_flat : (N, D)
            def body(acc, args):
                k, k_ = args
                return acc + spectral_kernel(x_flat, k, k_), None
            out, _ = jax.lax.scan(
                body,
                jnp.zeros_like(x_flat),
                (K_w, K_),
            )
            return out   # (N, D)

        # ── KNN rebuild (numpy side) ─────────────────────────────────────────
        # Single callback: rebuilds the BallTree and returns (x, nbr_idx) so
        # that the JAX loop never needs to read numpy state through a second
        # callback.  pure_callback supports tuple result shapes natively.

        _x_struct   = jax.ShapeDtypeStruct((N, D), jnp.float32)
        _nbr_struct = jax.ShapeDtypeStruct((N, K), jnp.int32)

        def _rebuild_knn_numpy(x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Rebuild BallTree; return (x unchanged, fresh neighbour indices)."""
            tree    = build_balltree(x_np, leaf_size=leaf_size)
            nbr_idx = query_knn(tree, x_np, K).astype(np.int32)
            return x_np, nbr_idx

        def rebuild_knn(x_val: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            """JAX-side wrapper: rebuild BallTree, return (x, nbr_idx)."""
            return jax.pure_callback(
                _rebuild_knn_numpy,
                (_x_struct, _nbr_struct),
                x_val,
            )

        # ── combined gradient ────────────────────────────────────────────────
        def full_grad(x_val: jnp.ndarray, nbr_idx: jnp.ndarray) -> jnp.ndarray:
            mg = knn_micro_grad(x_val, nbr_idx)
            mG = macro_grad(x_val)
            return (LR_spatial / S) * mg + (LR_spectral / (Nsqrt * Ncbrt)) * mG

        # ── main iteration loop ──────────────────────────────────────────────
        # The loop runs N_ITER outer iterations, each consisting of:
        #   1. one expensive BallTree rebuild (via pure_callback)
        #   2. N_PER_STEP cheap gradient descent steps reusing the same neighbours
        @jax.jit
        def run_iters(x_val: jnp.ndarray) -> jnp.ndarray:
            def outer_step(_, x_val):
                # Rebuild neighbour index once per outer step (expensive)
                x_val, nbr_idx = rebuild_knn(x_val)

                # N_PER_STEP cheap gradient steps reusing the same neighbours
                def inner_step(x_val, __):
                    return torus_wrap(x_val - full_grad(x_val, nbr_idx)), None

                x_val, _ = jax.lax.scan(inner_step, x_val, None, length=N_PER_STEP)
                return x_val

            return jax.lax.fori_loop(0, N_ITER, outer_step, x_val)

        # ── clone helper (flat version) ──────────────────────────────────────
        from ..blue.src.blue_sampler.math_utils import simplex, random_rotations

        Clone_simplex = simplex(Dsimp)

        def clone(x_parent: np.ndarray) -> np.ndarray:
            """Expand N//(Dsimp+1) parents into N children via simplex offsets."""
            x_parent  = x_parent[np.isfinite(x_parent).all(axis=-1)]
            x_parent  = np.random.permutation(x_parent)
            N_parents = N // (Dsimp + 1)
            N_keep    = N - (Dsimp + 1) * N_parents
            offsets   = random_rotations(Clone_simplex, N_parents, D, Dsimp) * (expension_factor / Ncbrt)
            children  = (x_parent[:N_parents, None, :] + offsets).reshape(-1, D)
            if N_keep > 0:
                children = np.concatenate([x_parent[N_parents:N_parents + N_keep], children], axis=0)
            return np.asarray(torus_wrap(jnp.array(children)))   # (N, D)

        # ── initialisation ───────────────────────────────────────────────────
        if is_root:
            x_pts = _build_bruteforce(N, D, ctx)(x)          # (N, D)
            x_pts = np.asarray(run_iters(jnp.array(x_pts)))
            ctx.done()
        else:
            N_child = N // (Dsimp + 1) + N % (Dsimp + 1)
            x_child = run_knn_pipeline(
                N=N_child,
                D=D,
                N_ITER=N_ITER,
                logger=logger,
                S=S,
                K=K,
                expension_factor=expension_factor,
                LR_spatial=LR_spatial,
                LR_spectral=LR_spectral,
                spectral_radius=spectral_radius,
                N_PER_STEP=N_PER_STEP,
                leaf_size=leaf_size,
                use_spread=use_spread,
                _is_root=False,
                _is_leaf=False,
            )                                                # (N_child, D)
            x_pts = clone(x_child)                          # (N, D)
            x_pts = np.asarray(run_iters(jnp.array(x_pts)))
            ctx.done()

        return x_pts

    finally:
        logger.exit_level()