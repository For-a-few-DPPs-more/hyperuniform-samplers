"""
Public Entry Point :
blue.sample_points is the main fonction of blue, calling the internal solver under the hook.
blue can also sample tessels, but only in 2D.
"""
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from .run_bruteforce import _bruteforce_pipeline
from .run_recursive import _recursive_pipeline
from .run_warmstarted import _warmstarted_pipeline
from .warm_start import sobol_init
from .progress import ProgressLogger

from .viz import plot_tessel
from .tessels import fair_random_split

_PRESETS: dict[int, dict] = {
    2: dict(spatial_radius=7, spectral_radius=7, LR_spatial=0.1,  LR_spectral=0.1, expension_factor=0.3, S=1.0),
    3: dict(spatial_radius=5, spectral_radius=5, LR_spatial=0.1,  LR_spectral=0.1, expension_factor=0.3, S=1.0),
    4: dict(spatial_radius=3, spectral_radius=3, LR_spatial=0.01, LR_spectral=0.1, expension_factor=1.0, S=0.5),
}


def sample_points(
    N: int,
    D: int,
    bruteforce: bool = False,
    warmstart: None | np.ndarray | Literal["Sobol"] = None,
    n_iter: int = 6,
    verbose: int = 1,
) -> np.ndarray:
    """
    Generate N stealthy points in [0, 1)^D.

    Parameters
    ----------
    N          : number of output points.
    D          : spatial dimension.
    bruteforce : whether to use a bruteforce algorithm with O(N^2) complexity.
    warmstart  : None, custom initial configuration (as a numpy array of points in [0, 1]) 
                or "Sobol". "Sobol" replaces the default random/recursive initialisation 
                of the points with a Sobol sequence.
    n_iter     : pipeline iterations — more is better but slower.
    verbose    : 0 = silent, 1 = live progress.

    Note
    ----
    Bruteforce is automatically used for small N (N <= 2_000).
    It gives a much better sample, but is intractable for N >= 50_000.
    If N lies between 2_000 and 50_000, it is recommended to give it a try,
    especially with a GPU and when quality matters more than speed.

    n_iter is the number of big iterations performed by the solver.
    Each iteration consists of 10 gradient steps and a structural
    gridification step (for neighbor lookup).

    warmstart="Sobol" requires scipy (scipy.stats.qmc.Sobol).
    and allow initialising gradient descent with a very good 
    starting point.
    """
    if isinstance(warmstart, np.ndarray):
        if warmstart.shape != (N, D):
            raise ValueError(f"warmstart must have shape {(N, D)}, got {warmstart.shape}")
        x = warmstart 
    elif warmstart is None:
        x = None
    elif warmstart == "Sobol":
        x = sobol_init(N, D)
    else:
        raise ValueError(f"unsupported warmstart={warmstart!r}, expected None, a custom np.array or 'Sobol'")
    
    if n_iter == 0:
        return x
    
    logger = ProgressLogger(verbose)

    if bruteforce or N <= 2_000:
        reason = f"N={N} ≤ 2 000" if N <= 2_000 else "bruteforce asked"
        ctx  = logger.enter_level(N, D, 0, reason)
        blue = _bruteforce_pipeline(N, D, ctx)
        out  = np.array(blue(x))
        logger.exit_level()
        return out

    preset = _PRESETS[min(D, 4)]

    if warmstart is not None:
        return _warmstarted_pipeline(
            N=N,
            D=D,
            N_ITER=n_iter,
            logger=logger,
            x=x,
            S=preset["S"],
            LR_spatial=preset["LR_spatial"],
            spatial_radius=preset["spatial_radius"],
            N_PER_STEP=10,
        )

    return _recursive_pipeline(
        N=N,
        D=D,
        N_ITER=n_iter,
        logger=logger,
        x=None,
        S=preset["S"],
        expension_factor=preset["expension_factor"],
        LR_spatial=preset["LR_spatial"],
        LR_spectral=preset["LR_spectral"],
        spatial_radius=preset["spatial_radius"],
        spectral_radius=preset["spectral_radius"],
        N_PER_STEP=10,
        _is_root=False,
        _is_leaf=True,
    )


def sample_tessels(N = 2**10, display = True):
    """
    Starting from the unit square, recursively split it in random quadrilaterals.
    If display is true, display the 9 first steps of the recursive spliting.

    Returns
    -------------
        quad np.array (2**depth, 4, 2)
        a tesselation composed of 2**depth quadrilaterals  with same area.

    Notes
    -------------
        support only for 2D geometry and powers of 2 number of tessels
        steps after the 9th split are NOT displayed as quads would be to small 
        to plot theim.
    """
    depth = int(np.log2(N))
    assert 2**depth == N, "N must be a power of 2"
    if display:
        _, axes = plt.subplots(3, 3, figsize=(10,10))
    
    quad = np.array([[[0,0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]]) #initialisation as the unit square
    for k in range(depth):
        if (k < 9) & display:
            ki, kj = k//3, k%3
            plot_tessel(ax = axes[ki, kj], tessels = quad)
            axes[ki, kj].axis("off")
        quad = fair_random_split(quad)
    
    if display:
        plt.tight_layout()
        plt.show()
    return quad