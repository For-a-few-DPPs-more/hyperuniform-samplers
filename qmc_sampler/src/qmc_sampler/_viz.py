"""
Plots, borrowed from structure_factor_v2 (pip install structure-factor-v2).

plot                   Scatter plot of a point set (structure_factor_v2.plot, as is).
plot_structure_factor  Log-log plot of the radial structure factor S(k): structure_factor_v2's,
                       measured with the module settings (backend, precision, NUFFT tolerance).
"""

from __future__ import annotations

import structure_factor_v2 as _sf
from structure_factor_v2 import plot

import numpy as np

from ._backend import get_backend, get_nufft_tolerance, get_precision

__all__ = ["plot", "plot_structure_factor"]


def plot_structure_factor(points, resolution: int = 20_000, ax=None, return_fig: bool = False,
                          title: str = None, **kwargs):
    """
    Log-log plot of the radial structure factor S(k) of points in [0, 1)^D, see
    `structure_factor_v2.plot_structure_factor` for every option (trust_boundary, margin,
    beta, and keywords forwarded to ``ax.loglog``).

    The Fourier sums use the module settings, like the samplers, so that points are measured
    the way they were sampled: backend of the dimension of the points (select_backend),
    precision (select_precision) and (cu)finufft tolerance (select_nufft_tolerance).
    ``backend``, ``precision`` or ``nufft_tolerance`` passed here override them for this plot.
    """
    D = np.asarray(points).shape[-1]
    kwargs.setdefault("backend", get_backend(D))
    kwargs.setdefault("precision", get_precision())
    kwargs.setdefault("nufft_tolerance", get_nufft_tolerance())
    return _sf.plot_structure_factor(points, resolution=resolution, ax=ax, return_fig=return_fig,
                                     title=title, **kwargs)
