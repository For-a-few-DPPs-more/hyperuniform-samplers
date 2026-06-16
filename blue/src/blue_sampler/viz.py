"""
Visualisation helpers:
plot display the points, 
plot_structure_factor display the structure factor
estimated through scattering intensity 
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .math_utils import structure_factor as _structure_factor


def plot(
    points: np.ndarray,
    max_scatter: int = 30_000,
    ax: plt.Axes | None = None,
    **scatter_kw,
) -> plt.Figure:
    """
    Scatter plot of a 2-D or 3-D point set.

    For large point sets the view is automatically zoomed so that at most
    *max_scatter* points are displayed.

    Parameters
    ----------
    points : array-like
        Point coordinates, shape (N, D) with D ∈ {2, 3}.
        Higher-dimensional arrays are silently projected onto the first 3 axes.
    max_scatter : int
        Maximum number of points to draw.  Excess points are cropped by
        zooming into the lower-left corner of the domain.
    ax : matplotlib Axes | None
        Existing axes to draw into.  When *None* a new figure is created.
    **scatter_kw
        Extra keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    D   = min(pts.shape[-1], 3)
    pts = pts[:, :D]

    if len(pts) > max_scatter:
        zoom = (max_scatter / len(pts)) ** (1.0 / D)
        pts  = pts[(pts <= zoom).all(axis=1)]

    kw = dict(s=0.4, color="black")
    kw.update(scatter_kw)

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        if D == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if D == 2:
        ax.scatter(pts[:, 0], pts[:, 1], **kw)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kw)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return fig


def plot_structure_factor(
    points: np.ndarray,
    bins: int = 100,
    resolution: float = 30.0,
    ax: plt.Axes | None = None,
    **plot_kw,
) -> plt.Figure:
    """
    Log-log plot of the radial structure factor S(k).
    S(k) is estimated using standard scattering intensity

    Parameters
    ----------
    points : (N, D) array
        Point coordinates in [0, 1)^D.
    bins : int
        Number of radial bins for the structure-factor estimate.
    resolution : float
        Random wave-vector density (vectors per bin) for the estimate.
    ax : matplotlib Axes | None
        Existing axes to draw into.  When *None* a new figure is created.
    **plot_kw
        Extra keyword arguments forwarded to ``ax.loglog``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    k, S = _structure_factor(pts, nbins=bins, resolution=resolution)

    kw = dict(marker="o", markersize=2, linewidth=1)
    kw.update(plot_kw)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    ax.loglog(k, S, **kw)
    ax.set_xlabel("k")
    ax.set_ylabel("S(k)")
    ax.set_title("Structure factor  (log-log)")
    ax.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.show()
    return fig
