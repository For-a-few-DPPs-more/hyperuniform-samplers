import time
import numpy as np
from scipy.stats import qmc
from new_data import new_data

DISCREPANCIES = ("CD", "WD", "MD",) #optionnal: include "L2-star"

def scipy_benchmark(shape, seed=42, values = False):
    """Compare scipy QMC engines and blue-sampler point clouds for a given shape (N, D).

    Prints a ranking per discrepancy table (1 = best).
    """
    N, D = shape

    gens = {
        "Sobol": lambda: qmc.Sobol(D, seed=seed).random(N),
        "Halton": lambda: qmc.Halton(D, seed=seed).random(N),
        "Chi30": lambda: new_data(N=N, D=D, chi=30, seed=seed),
        "Chi43": lambda: new_data(N=N, D=D, chi=43, seed=seed),
        "L1Chi43": lambda: new_data(N=N, D=D, chi="L1-43", seed=seed),
        "WDChi43": lambda: new_data(N=N, D=D, chi="WD-43", seed=seed),
        "LHS": lambda: qmc.LatinHypercube(D, seed=seed).random(N),
        "LHS random-cd": lambda: qmc.LatinHypercube(D, optimization="random-cd", seed=seed).random(N),
        "poisson i.i.d": lambda: np.random.rand(N, D)
    }

    np.random.seed(seed)
    clouds, times = {}, {}
    for name, g in gens.items():
        try:
            t0 = time.perf_counter()
            clouds[name] = np.asarray(g(), dtype=float)
            times[name] = time.perf_counter() - t0
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {e}")

    res = {n: [qmc.discrepancy(P, method=d) for d in DISCREPANCIES] for n, P in clouds.items()}
    names = list(res)
    w = max(len(n) for n in names) + 2

    if values:

        # ---- table 1: discrepance values (lower = better) ----
        header = f"{'':<{w}}" + "".join(f"{d:>14}" for d in DISCREPANCIES)
        print(f"\nDiscrepancies (lower = better)  —  N={N}, D={D}\n{header}\n{'-' * len(header)}")
        for n in names:
            vals = []
            for v in res[n]:
                vals.append(f"{v:>14.2e}")
            print(f"{n:<{w}}" + "".join(vals))

    # ---- table 2: rank per discrepancy (1 = best), sorted by mean rank ----
    ranks = {n: [] for n in names}
    for j in range(len(DISCREPANCIES)):
        for r, n in enumerate(sorted(names, key=lambda n: res[n][j]), start=1):
            ranks[n].append(r)
    header = f"{'':<{w}}" + "".join(f"{d:>12}" for d in DISCREPANCIES) + f"{'mean':>9}"
    print(f"\n(N = {N}, D = {D}) Discrepancy Ranking (1 = best)\n{header}\n{'-' * len(header)}")
    for n in sorted(names, key=lambda n: np.mean(ranks[n])):
        print(f"{n:<{w}}" + "".join(f"{r:>12d}" for r in ranks[n]) + f"{np.mean(ranks[n]):>9.2f}")

    return

import numpy as np
import matplotlib.pyplot as plt


def plot2D_structure_factor(pts, wv, ax=None, cmap="gray", figsize=(7, 6)):
    """
    Plot the 2D structure factor S(k) at selected wavevectors.

    Parameters
    ----------
    pts : array-like, shape (N, 2)
        Positions of the points in real space.
    wv : array-like, shape (M, 2)
        Wavevectors at which to evaluate S(k).
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    cmap : str
        Colormap used for S(k).
    figsize : tuple
        Figure size if ax is None.

    Returns
    -------
    fig, ax, sf : figure, axes, ndarray
        Figure, axes and computed structure factor.
    """
    pts = np.asarray(pts, dtype=float)
    wv = np.asarray(wv, dtype=float)
    wv = wv[~(wv == 0).all(axis = -1)]
    # S(k) = |sum_j exp(2i pi k.r_j)|^2 / N
    phase = 2*np.pi*wv @ pts.T
    sf = np.abs(np.exp(1j * phase).sum(axis=1))**2 / len(pts)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    wv = np.concatenate((wv, -wv), axis = 0)
    sf = np.concatenate((sf, sf), axis = 0)
    sf = np.log(sf).clip(min = -10, max = 3)

    sc = ax.scatter(
        wv[:, 0], wv[:, 1],
        c=sf,
        cmap=cmap,
        s = 300*(1000/len(wv)),
        linewidths=0,
        edgecolors="none",
    )

    cbar = fig.colorbar(sc, ax=ax)
    ax.set_title("2D Structure Factor (log)")

    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax, sf
    
if __name__ == "__main__":
    for n in (2**10, 2**11, 2**12):
        print("="*30)
        print(f"=== benchmark for N = {n} ===")
        print("="*30)
        for d in range(2, 9):
            scipy_benchmark((n, d))