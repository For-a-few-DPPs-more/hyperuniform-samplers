#!/usr/bin/env python
"""Build a new randomised (N, D) hyperuniform point cloud.

Usage
-----
    python new_data.py --dim 2 --npoints 1024 --ball l2 --seed 42
    python new_data.py --dim 20 --npoints 2**20 --ball wd --seed 0

Balls (spectral exclusion regions, chi = 0.43, S(k) <= 1e-10)
-------------------------------------------------------------
    l2 : Euclidean ball (isotropic, blue noise).
         data/l2/D{j}/N{n}_D{j}.h5
    wd : exclusion set of the wrap-around discrepancy (axis-oriented,
         natural for QMC, see scipy.stats.qmc.discrepancy).
         data/wd/D{j}/N{n}_D{j}.h5

Pipeline
--------
1. Base cloud: N_base = min(N, 2**17).
2. Projection: if D is not stored, project from the smallest stored
   dimension >= D by keeping D randomly chosen coordinates.
3. Tiling: if N > 2**17, multiply the number of points by 2**K.
4. Randomisation: random shift, coordinate flips and axis permutation.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

BALLS = ("l2", "wd")
AVAILABLE_DIMS = list(range(2, 17)) + [20, 24, 28, 32]
MIN_LOG2 = 10
MAX_BASE_LOG2 = 17


def read_base(data_dir, ball, log2n, D_src):
    """Read the stored cloud data/<ball>/D<D_src>/N<log2n>_D<D_src>.h5."""
    path = Path(data_dir) / ball / f"D{D_src}" / f"N{log2n}_D{D_src}.h5"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: "
            f"python load_data.py --dim {D_src} --ball {ball}"
        )

    with h5py.File(path, "r") as f:
        return f["points"][()]


def pure_tile(X, k):
    """Create 2**k periodic copies per axis."""
    n, D = X.shape
    r = 2 ** k
    grid = np.stack(
        np.meshgrid(*[np.arange(r)] * D, indexing="ij"), -1
    ).reshape(-1, D)

    out = np.empty((len(grid) * n, D), dtype=np.float32)
    for i, t in enumerate(grid):
        out[i * n:(i + 1) * n] = (X.astype(np.float64) + t) / r
    return out


def rotate_pair(X, a, b):
    """Double the cloud using a 45-degree slice on axes (a, b)."""
    n = len(X)
    Y = np.concatenate([X, X])

    u = Y[:, a].astype(np.float64)
    v = Y[:, b].astype(np.float64)
    u[n:] += 1.0

    Y[:, a] = np.mod((u + v) / 2, 1.0)
    Y[:, b] = np.mod((u - v) / 2, 1.0)
    return Y


def tile(X, K, rng=None):
    """Multiply the number of points by 2**K using periodic tiling."""
    if rng is None:
        rng = np.random.default_rng(42)

    D = X.shape[1]
    P = D // 2
    k = K // D if K > 2 * P else 0

    if k:
        X = pure_tile(X, k)

    rem = K - k * D
    counts = (min(rem, P), rem - min(rem, P))
    order = rng.permutation(D)

    for count, ordr in zip(counts, (order, np.roll(order, -1))):
        pairs = ordr[:2 * P].reshape(P, 2)
        for j in rng.choice(P, size=count, replace=False):
            X = rotate_pair(X, *pairs[j])

    return X


def new_data(N, D, ball, seed, data_dir="data"):
    """Return a randomised (N, D) float32 point cloud on [0,1)^D."""
    if ball not in BALLS:
        raise ValueError(f"Unknown ball {ball!r}. Available: {list(BALLS)}")

    if N < 2 ** MIN_LOG2 or N & (N - 1):
        raise ValueError(
            f"N must be a power of 2 >= {2 ** MIN_LOG2}, got {N}"
        )

    if not 2 <= D <= AVAILABLE_DIMS[-1]:
        raise ValueError(f"D must be in [2, {AVAILABLE_DIMS[-1]}], got {D}")

    rng = np.random.default_rng(seed)

    log2n = N.bit_length() - 1
    base_log2 = min(log2n, MAX_BASE_LOG2)
    D_src = next(d for d in AVAILABLE_DIMS if d >= D)

    X = read_base(data_dir, ball, base_log2, D_src)

    if D_src != D:
        X = X[:, rng.permutation(D_src)[:D]]

    X = tile(X, log2n - base_log2, rng)

    shift = rng.random(D)
    flip = rng.random(D) < 0.5
    axis = rng.permutation(D)

    X = np.mod(X.astype(np.float64) - shift, 1.0)
    X[:, flip] = 1.0 - X[:, flip]
    X = X[:, axis]

    return np.mod(X.astype(np.float32), 1.0)


def parse_int(s):
    """Accept '1024', '2**10' or '1<<10'."""
    s = s.replace(" ", "")

    if "**" in s:
        a, b = s.split("**")
        return int(a) ** int(b)

    if "<<" in s:
        a, b = s.split("<<")
        return int(a) << int(b)

    return int(s)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dim", type=int, required=True,
                   help="dimension D (2..32)")
    p.add_argument("--npoints", type=parse_int, required=True,
                   help="number of points N (power of 2, >= 1024)")
    p.add_argument("--ball", choices=BALLS, required=True,
                   help="exclusion region: l2 (isotropic) or wd (QMC)")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed (default: 0)")
    p.add_argument("--data", default="data",
                   help="directory used by load_data.py (default: data)")
    p.add_argument("--out", default=None,
                   help="output .npy file (default: generated automatically)")
    args = p.parse_args()

    try:
        X = new_data(args.npoints, args.dim, args.ball, args.seed, args.data)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")

    out = Path(
        args.out
        or f"new_data/N{args.npoints}_D{args.dim}_{args.ball}_seed{args.seed}.npy"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, X)

    print(f"Wrote {out}: shape {X.shape}, ball = {args.ball}.")


if __name__ == "__main__":
    main()