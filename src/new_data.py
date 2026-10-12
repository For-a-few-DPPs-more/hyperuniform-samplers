
#!/usr/bin/env python
"""Build a new randomised (N, D) hyperuniform point cloud.

Usage
-----
    python new_data.py --dim 2 --npoints 1024 --chi 43 --seed 42
    python new_data.py --dim 20 --npoints 2**20 --chi 30 --seed 0
    python new_data.py --dim 2 --npoints 1024 --chi L1-50 --seed 42

Dataset families
----------------
    30     -> data/chi30/D{D_src}/N{i}_D{D_src}.h5
    43     -> data/chi43/D{D_src}/N{i}_D{D_src}.h5
    L1-50  -> data/L1chi50/D{D_src}/N{i}_D{D_src}.h5

L1-50 denotes chi = 0.50 with the L1 norm. It is a distinct dataset
family and must not be confused with chi50.

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


AVAILABLE_DIMS = list(range(2, 17)) + [23, 32]

# Explicit mapping protects the distinction between dataset families.
DATASET_FAMILIES = {
    "30": {
        "directory": "chi30",
        "label": "chi30",
    },
    "43": {
        "directory": "chi43",
        "label": "chi43",
    },
    "L1-50": {
        "directory": "L1chi50",
        "label": "L1chi50",
    },
}

MIN_LOG2 = 10
MAX_BASE_LOG2 = 17


def read_base(data_dir, chi, log2n, D_src):
    """Read the stored cloud for the selected dataset family."""
    chi = str(chi)

    if chi not in DATASET_FAMILIES:
        raise ValueError(
            f"Unknown dataset family {chi!r}. "
            f"Available: {list(DATASET_FAMILIES)}"
        )

    family_dir = DATASET_FAMILIES[chi]["directory"]
    path = (
        Path(data_dir)
        / family_dir
        / f"D{D_src}"
        / f"N{log2n}_D{D_src}.h5"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: "
            f"python load_data.py --dim {D_src} --chi {chi}"
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
        out[i * n:(i + 1) * n] = (
            X.astype(np.float64) + t
        ) / r

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

    for count, ordr in zip(
        counts, (order, np.roll(order, -1))
    ):
        pairs = ordr[:2 * P].reshape(P, 2)

        for j in rng.choice(P, size=count, replace=False):
            X = rotate_pair(X, *pairs[j])

    return X


def new_data(N, D, chi, seed, data_dir="data"):
    """Return a randomised (N, D) float32 point cloud on [0,1)^D."""
    chi = str(chi)

    if N < 2 ** MIN_LOG2 or N & (N - 1):
        raise ValueError(
            f"N must be a power of 2 >= {2 ** MIN_LOG2}, got {N}"
        )

    if D not in range(2, AVAILABLE_DIMS[-1] + 1):
        raise ValueError(
            f"D must be in [2, {AVAILABLE_DIMS[-1]}], got {D}"
        )

    if chi not in DATASET_FAMILIES:
        raise ValueError(
            f"Unknown dataset family {chi!r}. "
            f"Available: {list(DATASET_FAMILIES)}"
        )

    rng = np.random.default_rng(seed)

    log2n = N.bit_length() - 1
    base_log2 = min(log2n, MAX_BASE_LOG2)
    D_src = next(d for d in AVAILABLE_DIMS if d >= D)

    X = read_base(data_dir, chi, base_log2, D_src)

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

    p.add_argument(
        "--dim",
        type=int,
        required=True,
        help="dimension D (2..32)",
    )
    p.add_argument(
        "--npoints",
        type=parse_int,
        required=True,
        help="number of points N (power of 2, >= 1024)",
    )
    p.add_argument(
        "--chi",
        type=str,
        required=True,
        choices=list(DATASET_FAMILIES),
        help="dataset family: 30, 43 or L1-50",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: 0)",
    )
    p.add_argument(
        "--data",
        default="data",
        help="directory used by load_data.py",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output .npy file (default: generated automatically)",
    )

    args = p.parse_args()

    family_label = DATASET_FAMILIES[args.chi]["label"]

    try:
        X = new_data(
            args.npoints,
            args.dim,
            args.chi,
            args.seed,
            args.data,
        )
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")

    out = Path(
        args.out
        or (
            f"new_data/N{args.npoints}_D{args.dim}_"
            f"{family_label}_seed{args.seed}.npy"
        )
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, X)

    print(f"Dataset family: {args.chi} ({family_label})")
    print(f"Wrote {out}: shape {X.shape}.")


if __name__ == "__main__":
    main()
