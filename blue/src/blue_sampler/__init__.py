"""
blue_sampler
============

Generate stealthy point patterns — low-discrepancy, spectrally isotropic
samples on the unit torus [0, 1)^D.

Quick start
-----------
>>> import blue_sampler as blue
>>> x = blue.sample(N=10_000, D=2)          # (10000, 2) array
>>> blue.plot(x)
>>> blue.plot_structure_factor(x)
"""
from .entry_point import sample
from .viz import plot, plot_structure_factor
from .math_utils import structure_factor

__all__ = [
    "sample",
    "plot",
    "plot_structure_factor",
    "structure_factor",
]
