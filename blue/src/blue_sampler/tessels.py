import numpy as np

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
    """area of a batch of quadrilaterals."""
    return area4(q[..., 0, :], q[..., 1, :], q[..., 2, :], q[..., 3, :])


# ------------------------------------------------------------
# Core Attempt function (Bimodal with safeguards)
# ------------------------------------------------------------

def _attempt(quads, targets, pool, rng, eps=1e-9):
    A, B, C, D = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]

    # Directions de coupe aléatoires
    theta = rng.uniform(0.0, np.pi, size=(quads.shape[0], pool))
    nx, ny = -np.sin(theta), np.cos(theta)

    lvl = lambda P: P[:, None, 0] * nx + P[:, None, 1] * ny
    hA, hB, hC, hD = lvl(A), lvl(B), lvl(C), lvl(D)
    h = np.stack((hA, hB, hC, hD), axis=-1)

    h_sorted = np.sort(h, axis=-1)
    l1, l2 = h_sorted[..., 1], h_sorted[..., 2]

    is_bottom2 = h <= l1[..., None]
    cross_DA_BC = (is_bottom2[..., 0] & is_bottom2[..., 1]) | (is_bottom2[..., 2] & is_bottom2[..., 3])

    def pt(V1, V2, h1, h2, t_val, batch_mode=True):
        denom = np.where(np.abs(h2 - h1) < eps, eps, h2 - h1)
        alpha = (t_val - h1) / denom
        if batch_mode:
            return V1[:, None, :] + alpha[..., None] * (V2 - V1)[:, None, :]
        return V1 + alpha[:, None] * (V2 - V1)

    if targets is None:
        # --- MODE 1: Équidistribution des aires ---
        target_area = (quad_signed_area(quads) * 0.5)[:, None]
        lm = 0.5 * (l1 + l2)

        def get_area(t):
            return np.where(
                cross_DA_BC,
                area4(A[:, None], B[:, None], pt(B, C, hB, hC, t), pt(D, A, hD, hA, t)),
                area4(pt(A, B, hA, hB, t), B[:, None], C[:, None], pt(C, D, hC, hD, t))
            )

        f_l1 = get_area(l1) - target_area
        f_lm = get_area(lm) - target_area
        f_l2 = get_area(l2) - target_area

        pool_ok = (f_l1 * f_l2) <= 0
        sort_idx = None
    else:
        # --- MODE 2: Split Médian des Atomes ---
        h_targets = targets[:, None, :, 0] * nx[..., None] + targets[:, None, :, 1] * ny[..., None]
        sort_idx = np.argsort(h_targets, axis=-1)
        h_targets_sorted = np.take_along_axis(h_targets, sort_idx, axis=-1)
        
        K2 = targets.shape[1] // 2
        t_star_pool = 0.5 * (h_targets_sorted[..., K2 - 1] + h_targets_sorted[..., K2])
        
        # Validité topologique (la coupe médiane doit tomber entre l1 et l2)
        pool_ok = (l1 <= t_star_pool) & (t_star_pool <= l2)

    # --- Sélection de la meilleure direction valide ---
    ok = np.any(pool_ok, axis=1)
    idx_sel = np.argmax(pool_ok, axis=1)[:, None]
    pick = lambda arr: np.take_along_axis(arr, idx_sel, axis=1)[:, 0]

    l1, l2 = pick(l1), pick(l2)
    cross_DA_BC = pick(cross_DA_BC)
    hA, hB, hC, hD = pick(hA), pick(hB), pick(hC), pick(hD)

    # --- Calcul final de t_star (uniquement sur la direction sélectionnée) ---
    if targets is None:
        lm = pick(lm)
        f_l1, f_lm, f_l2 = pick(f_l1), pick(f_lm), pick(f_l2)

        dx = 0.5 * (l2 - l1)
        denom_dx = np.where(dx < eps, eps, dx)
        c = f_lm
        b = (f_l2 - f_l1) / (2.0 * denom_dx)
        a = (f_l2 + f_l1 - 2.0 * f_lm) / (2.0 * denom_dx ** 2)

        discriminant = np.maximum(b ** 2 - 4 * a * c, 0.0)
        sqrt_disc = np.sqrt(discriminant)

        denom_a = np.where(np.abs(a) < eps, eps, 2 * a)
        sol1 = (-b + sqrt_disc) / denom_a
        sol2 = (-b - sqrt_disc) / denom_a
        sol_linear = -c / np.where(np.abs(b) < eps, eps, b)

        x_star = np.where(np.abs(a) < eps, sol_linear, sol1)
        x_star = np.where(((x_star < -dx) | (x_star > dx)) & (np.abs(a) >= eps), sol2, x_star)

        t_star = np.clip(lm + x_star, l1, l2)
    else:
        t_star = pick(t_star_pool)

    # --- Reconstruction des morceaux enfants ---
    P_AB = pt(A, B, hA, hB, t_star, batch_mode=False)
    P_BC = pt(B, C, hB, hC, t_star, batch_mode=False)
    P_CD = pt(C, D, hC, hD, t_star, batch_mode=False)
    P_DA = pt(D, A, hD, hA, t_star, batch_mode=False)

    mask = cross_DA_BC[:, None, None]
    piece1 = np.where(mask, np.stack((A, B, P_BC, P_DA), axis=1), np.stack((P_AB, B, C, P_CD), axis=1))
    piece2 = np.where(mask, np.stack((P_BC, C, D, P_DA), axis=1), np.stack((A, P_AB, P_CD, D), axis=1))

    # --- Distribution des atomes (Mode 2) ---
    if targets is not None:
        nx_sel = pick(nx)
        ny_sel = pick(ny)
        
        pick_k = lambda arr: np.take_along_axis(arr, idx_sel[..., None], axis=1)[:, 0, :]
        best_sort_idx = pick_k(sort_idx)

        targets_sorted = np.take_along_axis(targets, best_sort_idx[..., None], axis=1)
        targets_bottom = targets_sorted[:, :K2, :]
        targets_top = targets_sorted[:, K2:, :]

        c1 = np.mean(piece1, axis=1)
        hc1 = c1[:, 0] * nx_sel + c1[:, 1] * ny_sel
        is_p1_bottom = hc1 < t_star

        targets1 = np.where(is_p1_bottom[:, None, None], targets_bottom, targets_top)
        targets2 = np.where(is_p1_bottom[:, None, None], targets_top, targets_bottom)

        return piece1, piece2, targets1, targets2, ok

    return piece1, piece2, None, None, ok


# ------------------------------------------------------------
# Wrapper principal recursif
# ------------------------------------------------------------

def fair_random_split(quads, targets=None, pool=5, rng=None, max_retries=8, growth=2, eps=1e-9):
    """
    Split each quadrilateral into two children. 
    If `targets` is None, splits by equal area.
    If `targets` is provided, splits by median separation of targets.
    """
    rng = np.random.default_rng() if rng is None else rng
    B = quads.shape[0]
    has_targets = targets is not None

    piece1 = np.empty((B, 4, 2), dtype=quads.dtype)
    piece2 = np.empty((B, 4, 2), dtype=quads.dtype)
    
    if has_targets:
        K2 = targets.shape[1] // 2
        t1 = np.empty((B, K2, 2), dtype=targets.dtype)
        t2 = np.empty((B, K2, 2), dtype=targets.dtype)

    resolved = np.zeros(B, dtype=bool)
    remaining_idx = np.arange(B)
    cur_pool = pool

    for _ in range(max_retries + 1):
        if len(remaining_idx) == 0:
            break

        curr_quads = quads[remaining_idx]
        curr_targets = targets[remaining_idx] if has_targets else None

        p1, p2, out_t1, out_t2, ok = _attempt(curr_quads, curr_targets, cur_pool, rng, eps=eps)
        good = remaining_idx[ok]

        piece1[good] = p1[ok]
        piece2[good] = p2[ok]
        
        if has_targets:
            t1[good] = out_t1[ok]
            t2[good] = out_t2[ok]

        resolved[good] = True
        remaining_idx = remaining_idx[~ok]
        cur_pool *= growth

    if len(remaining_idx):
        raise RuntimeError(f"no valid quad/quad split found for {len(remaining_idx)} polygon(s)")

    if has_targets:
        return np.concatenate((piece1, piece2), axis=0), np.concatenate((t1, t2), axis=0)

    return np.concatenate((piece1, piece2), axis=0)

def quad_diameters(quads):
    """
    Computes the diameter of each quad (max distance between any two vertices).
    quads: np.array of shape (Nquads, 4, 2)
    Returns: np.array of shape (Nquads,)
    """
    # Vectorized pairwise differences using broadcasting -> (N, 4, 4, 2)
    deltas = quads[:, None, :, :] - quads[:, :, None, :]
    
    # Euclidean distances -> (N, 4, 4)
    dists = np.linalg.norm(deltas, axis=-1)
    
    # Max distance per quad -> (N,)
    return dists.max(axis=(1, 2))

def debias_quads(quads, targets, criterion=10.0):
    """
    Replaces overly large quads with the centroid of their respective targets.
    Modifies 'quads' in-place.
    """
    qA = quad_diameters(quads)
    too_big = qA >= criterion * qA.mean()
    
    # Broadcasts the (M, 1, 2) centroids into the (M, 4, 2) quads
    quads[too_big] = targets[too_big].mean(axis=1, keepdims=True)
    
    return quads
