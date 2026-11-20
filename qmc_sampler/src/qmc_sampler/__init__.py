MESSAGE = """
qmc_sampler: QMC point sets by spectral optimisation, see
     https://github.com/For-a-few-DPPs-more/hyperuniform-samplers 

Available methods:
- sample_l2: uses a classical isotropic L2-ball as the spectral exclusion region.
- sample_l1: uses an L1-ball, which is anisotropic and can be useful for QMC
  in settings with low effective dimensionality.
- sample_wd: uses the wrap-around discrepancy, further exploiting the
  low-effective-dimensionality assumption.
"""

print(MESSAGE)


from ._backend import (
    available_backends,
    get_backend,
    get_nufft_tolerance,
    get_precision,
    select_backend,
    select_nufft_tolerance,
    select_precision,
)
from ._viz import plot, plot_structure_factor
from .qmc_sampler import sample_l1, sample_l2, sample_wd

__all__ = [
    "sample_l1",
    "sample_l2",
    "sample_wd",
    "plot",
    "plot_structure_factor",
    "select_backend",
    "get_backend",
    "available_backends",
    "select_precision",
    "get_precision",
    "select_nufft_tolerance",
    "get_nufft_tolerance",
]
__version__ = "3.0.2"
