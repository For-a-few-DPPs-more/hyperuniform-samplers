"""Benchmark hyperuniform point clouds against standard QMC / random samplers.

For each (N, D) configuration, every sampler is scored with several scipy
discrepancies (lower = better) and ranked (1 = best). Per-configuration prints
only show the rankings. At the end, two summary tables are printed:

1. Global ranking: mean rank over all configurations, sorted best to worst.
2. Geometric mean of the discrepancy ratio relative to Sobol, per discrepancy
   (< 1 means better than Sobol).

Usage
-----
    python benchmark_data.py
"""

import time

import numpy as np
from scipy.stats import qmc

from new_data import new_data

DISCREPANCIES = ("CD", "WD", "MD")  # optional: add "L2-star"
REFERENCE = "Sobol"  # sampler used as the baseline for the ratios


def make_generators(N, D, seed):
    """Return the samplers to compare, as {name: callable returning (N, D)}."""
    rng = np.random.default_rng(seed)
    return {
        "Sobol": lambda: qmc.Sobol(D, seed=seed).random(N),
        "Halton": lambda: qmc.Halton(D, seed=seed).random(N),
        "l2": lambda: new_data(N=N, D=D, ball="l2", seed=seed),
        "wd": lambda: new_data(N=N, D=D, ball="wd", seed=seed),
        "LHS": lambda: qmc.LatinHypercube(D, seed=seed).random(N),
        "LHS random-cd": lambda: qmc.LatinHypercube(
            D, optimization="random-cd", seed=seed
        ).random(N),
        "i.i.d. uniform": lambda: rng.random((N, D)),
    }


def scipy_benchmark(shape, seed=42, verbose=True):
    """Compare samplers for a given shape (N, D).

    Prints the ranking per discrepancy (1 = best) if `verbose`.

    Returns
    -------
    values : dict {name: list of discrepancy values, one per DISCREPANCIES}
    ranks  : dict {name: list of ranks, one per DISCREPANCIES}
    """
    N, D = shape

    # ---- generate the point clouds ----
    clouds = {}
    for name, gen in make_generators(N, D, seed).items():
        try:
            clouds[name] = np.asarray(gen(), dtype=float)
        except Exception as e:  # e.g. missing data files for l2 / wd
            print(f"[skip] {name}: {type(e).__name__}: {e}")

    # ---- discrepancies (lower = better) ----
    values = {
        n: [qmc.discrepancy(P, method=d) for d in DISCREPANCIES]
        for n, P in clouds.items()
    }
    names = list(values)

    # ---- rank per discrepancy (1 = best) ----
    ranks = {n: [] for n in names}
    for j in range(len(DISCREPANCIES)):
        for r, n in enumerate(sorted(names, key=lambda n: values[n][j]), start=1):
            ranks[n].append(r)

    if verbose:
        w = max(len(n) for n in names) + 2
        header = f"{'':<{w}}" + "".join(f"{d:>12}" for d in DISCREPANCIES) + f"{'mean':>9}"
        print(f"\n(N = {N}, D = {D}) Discrepancy ranking (1 = best)")
        print(header)
        print("-" * len(header))
        for n in sorted(names, key=lambda n: np.mean(ranks[n])):
            print(
                f"{n:<{w}}"
                + "".join(f"{r:>12d}" for r in ranks[n])
                + f"{np.mean(ranks[n]):>9.2f}"
            )

    return values, ranks


def summarize(all_values, all_ranks):
    """Print the final tables aggregated over all (N, D) configurations.

    all_values, all_ranks : lists of the dicts returned by scipy_benchmark.
    """
    names = sorted({n for r in all_ranks for n in r})
    w = max(len(n) for n in names) + 2
    nd = len(DISCREPANCIES)

    # ---- global ranking: mean rank over configurations where the sampler ran ----
    mean_rank = {
        n: np.mean([r[n] for r in all_ranks if n in r], axis=0) for n in names
    }
    global_rank = {n: mean_rank[n].mean() for n in names}
    order = sorted(names, key=lambda n: global_rank[n])

    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY over {len(all_ranks)} configurations")
    print("=" * 60)

    header = f"{'':<{w}}" + "".join(f"{d:>12}" for d in DISCREPANCIES) + f"{'global':>9}"
    print("\nGlobal ranking (mean rank, 1 = best), sorted by global rank")
    print(header)
    print("-" * len(header))
    for n in order:
        print(
            f"{n:<{w}}"
            + "".join(f"{x:>12.2f}" for x in mean_rank[n])
            + f"{global_rank[n]:>9.2f}"
        )

    # ---- geometric mean of discrepancy ratios relative to the reference ----
    if not any(REFERENCE in v for v in all_values):
        return

    print(f"\nGeometric mean of discrepancy relative to {REFERENCE} (< 1 = better than {REFERENCE})")
    header = f"{'':<{w}}" + "".join(f"{d:>12}" for d in DISCREPANCIES)
    print(header)
    print("-" * len(header))
    for n in order:
        row = []
        for j in range(nd):
            # only configurations where both the sampler and the reference ran
            logs = [
                np.log(v[n][j] / v[REFERENCE][j])
                for v in all_values
                if n in v and REFERENCE in v
            ]
            row.append(np.exp(np.mean(logs)) if logs else np.nan)
        print(f"{n:<{w}}" + "".join(f"{x:>12.3f}" for x in row))


def main(Ns=(2**10, 2**11, 2**12), Ds=range(2, 9), seed=42):
    all_values, all_ranks = [], []
    for N in Ns:
        print("=" * 30)
        print(f"=== benchmark for N = {N} ===")
        print("=" * 30)
        for D in Ds:
            values, ranks = scipy_benchmark((N, D), seed=seed)
            all_values.append(values)
            all_ranks.append(ranks)

    summarize(all_values, all_ranks)


if __name__ == "__main__":
    main()