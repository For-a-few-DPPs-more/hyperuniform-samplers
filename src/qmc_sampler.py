"""
see https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft.py:
same but:
  - L1 ball instead of L2 ball
  - keops backend instead of finufft

  -for wd (wrap arround discrepancy): exact brute-force computation of the O(N^2*D) kernel
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from qmc_geom import loss_and_grad_l1, loss_and_grad_wd, real_dtype


# ------------------------------------------------------------------
# Global defaults
# ------------------------------------------------------------------
NITER     = 150
LR        = 1.0
CHI = 0.3

# ------------------------------------------------------------------
# Main KeOps optimisation pipeline
# ------------------------------------------------------------------

def sample(
    N,
    D,
    mode = "l1",
    chi: float = CHI,
    target: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    n_iter: int = NITER,
    lr: float = LR,
    verbose: int = 1,
):
    """
    Generate a low-discrepancy point set on [0,1)^D by spectral optimisation.

    Parameters
    ----------
    N        : number of points
    D        : dimension
    chi      : fraction of degrees of freedom kept as Fourier modes
    target   : optional target point cloud (for non-uniform densities)
    seed     : random seed
    n_iter   : number of gradient steps
    lr       : base learning-rate scale
    verbose  : print progress

    Returns
    -------
    x : (N, D) array of points in [0,1)^D
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(N, D)).astype(real_dtype)

    if mode == "l1":
        loss_and_grad = loss_and_grad_l1(N, D, chi, rng, target)
    else:
        loss_and_grad = loss_and_grad_wd()

    adaptive = lr * 0.1 * N ** (-1.0 / D)
    prev_loss = np.inf

    if verbose:
        print(f"[keops] N={N}  D={D}")

    t0 = time.time()
    for it in range(n_iter):
        loss, grad = loss_and_grad(x)
        rms = float(np.sqrt(np.mean(grad * grad) + 1e-30))
        step = (grad / rms) * adaptive
        x_new = np.mod(x - step, 1.0).astype(real_dtype)

        if loss < prev_loss:
            x = x_new
            adaptive *= 1.05
            prev_loss = loss
        else:
            adaptive *= 0.8
            x = x_new   

        if verbose and (it % 20 == 0 or it == n_iter - 1):
            print(f"  iter {it:4d} | loss {loss:.3e}")

    if verbose:
        print(f"done in {time.time() - t0:.1f}s")

    return np.asarray(x)