"""
Fair random splitting of quadrilaterals into equal-area quadrilaterals.

This is a FAIR STIT implementation inspired from following article:
https://arxiv.org/pdf/2605.22803
Persistence of asymptotic variance under transport:
from hyperfluctuation to stealthy hyperuniformity
Luca Lotz and Michael A. Klat 

Physical idea
-------------
A random direction is selected and used to define a cutting line.
The position of the cut is chosen so that it divides the quadrilateral
into two regions of equal area. This reduces to solving a deg 2 polynom
equation since the area can be written analytically from the coordinates
of the four vertices with a determinant.

The goal is for both children to remain quadrilaterals. Depending on the
cut direction, the split may instead produce a triangle and a pentagon.
To avoid this, several candidate directions are sampled and invalid
splits are discarded.
"""

import numpy as np
import matplotlib.pyplot as plt
from .viz import plot_tessel

# ------------------------------------------------------------
# Formules d'aire (optimisées avec np.cross)
# ------------------------------------------------------------

def cross2d(A, B):
    """determinant of 2D vectors (cross product)"""
    return A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]

def area4(P0, P1, P2, P3):
    """area of a quadrilateral"""
    return 0.5 * (cross2d(P0, P1) + cross2d(P1, P2) + cross2d(P2, P3) + cross2d(P3, P0))

def quad_signed_area(q):
    """area of  a batch of quadrilaterals."""
    return area4(q[..., 0, :], q[..., 1, :], q[..., 2, :], q[..., 3, :])

# ------------------------------------------------------------
# Fair cut
# ------------------------------------------------------------

def _attempt(quads, pool, rng, eps=1e-9):
    B = quads.shape[0]
    A, Bv, C, D = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    target = (quad_signed_area(quads) * 0.5)[:, None]

    # Directions de coupe aléatoires
    theta = rng.uniform(0.0, np.pi, size=(B, pool))
    nx, ny = -np.sin(theta), np.cos(theta)

    lvl = lambda P: P[:, None, 0] * nx + P[:, None, 1] * ny
    hA, hB, hC, hD = lvl(A), lvl(Bv), lvl(C), lvl(D)
    h = np.stack((hA, hB, hC, hD), axis=-1)

    h_sorted = np.sort(h, axis=-1)
    l1, l2 = h_sorted[..., 1], h_sorted[..., 2]
    lm = 0.5 * (l1 + l2)

    is_bottom2 = h <= l1[..., None]
    cross_DA_BC = (is_bottom2[..., 0] & is_bottom2[..., 1]) | (is_bottom2[..., 2] & is_bottom2[..., 3])


    def pt(V1, V2, h1, h2, t_val, batch_mode=True):
        denom = np.where(np.abs(h2 - h1) < eps, eps, h2 - h1)
        alpha = (t_val - h1) / denom
        if batch_mode:
            return V1[:, None, :] + alpha[..., None] * (V2 - V1)[:, None, :]
        return V1 + alpha[:, None] * (V2 - V1)

    def get_area(t):
        return np.where(
            cross_DA_BC,
            area4(A[:, None], Bv[:, None], pt(Bv, C, hB, hC, t), pt(D, A, hD, hA, t)),
            area4(pt(A, Bv, hA, hB, t), Bv[:, None], C[:, None], pt(C, D, hC, hD, t))
        )

    f_l1 = get_area(l1) - target
    f_lm = get_area(lm) - target
    f_l2 = get_area(l2) - target

    pool_ok = (f_l1 * f_l2) <= 0
    ok = np.any(pool_ok, axis=1)

    idx_sel = np.argmax(pool_ok, axis=1)[:, None]
    pick = lambda arr: np.take_along_axis(arr, idx_sel, axis=1)[:, 0]

    l1, l2, lm = pick(l1), pick(l2), pick(lm)
    f_l1, f_lm, f_l2 = pick(f_l1), pick(f_lm), pick(f_l2)
    cross_DA_BC = pick(cross_DA_BC)
    hA, hB, hC, hD = pick(hA), pick(hB), pick(hC), pick(hD)

    dx = 0.5 * (l2 - l1)
    denom = np.where(dx < eps, eps, dx)
    c = f_lm
    b = (f_l2 - f_l1) / (2.0 * denom)
    a = (f_l2 + f_l1 - 2.0 * f_lm) / (2.0 * denom ** 2)

    discriminant = np.maximum(b ** 2 - 4 * a * c, 0.0)
    sqrt_disc = np.sqrt(discriminant)

    denom_a = np.where(np.abs(a) < eps, eps, 2 * a)
    sol1 = (-b + sqrt_disc) / denom_a
    sol2 = (-b - sqrt_disc) / denom_a
    sol_linear = -c / np.where(np.abs(b) < eps, eps, b)

    x_star = np.where(np.abs(a) < eps, sol_linear, sol1)
    x_star = np.where(((x_star < -dx) | (x_star > dx)) & (np.abs(a) >= eps), sol2, x_star)

    t_star = np.clip(lm + x_star, l1, l2)

    # New quads
    P_AB = pt(A, Bv, hA, hB, t_star, batch_mode=False)
    P_BC = pt(Bv, C, hB, hC, t_star, batch_mode=False)
    P_CD = pt(C, D, hC, hD, t_star, batch_mode=False)
    P_DA = pt(D, A, hD, hA, t_star, batch_mode=False)

    mask = cross_DA_BC[:, None, None]
    piece1 = np.where(mask, np.stack((A, Bv, P_BC, P_DA), axis=1), np.stack((P_AB, Bv, C, P_CD), axis=1))
    piece2 = np.where(mask, np.stack((P_BC, C, D, P_DA), axis=1), np.stack((A, P_AB, P_CD, D), axis=1))

    return piece1, piece2, ok

def fair_random_split(quads, pool=5, rng=None, max_retries=8, growth=2):
    """
    Split each quadrilateral into two equal-area quadrilaterals.

    Parameters
    ----------
    quads : (B, 4, 2) array
        Batch of quadrilaterals.

    Returns
    -------
    quads : (2*B, 4, 2) array
        Batch containing the two children of each input quadrilateral.

    Notes
    -----
    Not every equal-area cut produces two quadrilaterals. Some cut
    directions generate a triangle and a pentagon instead. To avoid
    such cases, several candidate directions are sampled and invalid
    splits are rejected. The parameters `pool`, `max_retries`, and
    `growth` control this rejection strategy.
    """
    rng = np.random.default_rng() if rng is None else rng
    B = quads.shape[0]

    piece1 = np.empty((B, 4, 2), dtype=quads.dtype)
    piece2 = np.empty((B, 4, 2), dtype=quads.dtype)
    resolved = np.zeros(B, dtype=bool)
    remaining_idx = np.arange(B)
    cur_pool = pool

    for _ in range(max_retries + 1):
        if len(remaining_idx) == 0:
            break

        p1, p2, ok = _attempt(quads[remaining_idx], cur_pool, rng)
        good = remaining_idx[ok]

        piece1[good] = p1[ok]
        piece2[good] = p2[ok]
        resolved[good] = True
        remaining_idx = remaining_idx[~ok]
        cur_pool *= growth

    if len(remaining_idx):
        raise RuntimeError(f"no fair quad/quad split found for {len(remaining_idx)} polygon(s)")

    return np.concatenate((piece1, piece2), axis=0)