"""
Public Entry Point :
blue.sample is the main fonction of blue, calling the internal solver under the hook
"""
import numpy as np
from .run import _build_bruteforce, _run_pipeline
from .progress import ProgressLogger

_PRESETS: dict[int, dict] = {
    2: dict(spatial_radius=7, spectral_radius=7, LR_spatial=0.1,  LR_spectral=0.1, expension_factor=0.3, S=1.0),
    3: dict(spatial_radius=5, spectral_radius=5, LR_spatial=0.1,  LR_spectral=0.1, expension_factor=0.3, S=1.0),
    4: dict(spatial_radius=3, spectral_radius=3, LR_spatial=0.01, LR_spectral=0.1, expension_factor=1.0, S=0.5),
}

def sample(
    N: int,
    D: int,
    bruteforce: bool = False,
    n_iter: int = 6,
    verbose: int = 1,
) -> np.ndarray:
    """
    Generate N stealthy points in [0, 1)^D.

    Parameters
    ----------
    N        : number of output points.
    D        : spatial dimension.
    bruteforce: wether to use a bruteforce algorithm with O(N2) complexity.
    n_iter   : pipeline iterations — more is better but slower.
    verbose  : 0 = silent, 1 = live progress.

    Note
    ----------
    Bruteforce is automatically set to True for small N (N<=2_000)
    It gives a much better sample, but is intractable for N >= 50_000.
    If N lies between 2000 and 50_000, it is recommanded to give it a try.
    Typicall usecase is when N <~=10_000 and quality matters more than speed 
    especially if one dispose of a powerfull computer or even a GPU

    n_iter is the number of big iteration performed by the solver.
    Each iteration consists of 10 gradient steps and a structural 
    gridification step (for neighbors lookup)
    """
    logger = ProgressLogger(verbose)

    if bruteforce or N <= 2_000:
        reason = (
            f"N={N} ≤ 2 000" if N <= 2_000
            else "bruteforce asked"
        )
        ctx  = logger.enter_level(N, D, 0, reason)
        blue = _build_bruteforce(N, D, ctx)
        out  = np.array(blue())
        logger.exit_level()
        return out

    preset = _PRESETS[min(D, 4)]

    return _run_pipeline(
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