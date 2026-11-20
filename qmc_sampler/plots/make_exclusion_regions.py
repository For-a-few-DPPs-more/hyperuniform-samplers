"""Plot the 2D spectral exclusion regions of sample_l2, sample_l1 and sample_wd (README figure).

    python plots/make_exclusion_regions.py      # from the qmc_sampler folder
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np

from qmc_sampler.qmc_geom import CHI, get_wave_vectors

N, D = 1000, 2
PANELS = [("l2", "L2", "#026204"), ("l1", "L1", "#2a7ab9"), ("wd", "WD", "#d01c1c")]
CROP = 31        # WD arms go much further along the axes: cropped at CROP, then "..."
PAD = 1.5        # margin around each region, in k units
SCALE = 0.075    # inches per k unit (same scale in every panel)
regions, xlims, ylims = [], [], []
for mode, _, _ in PANELS:
    K, _ = get_wave_vectors(N, D, mode=mode, chi=CHI, rng=np.random.default_rng(0))
    K = np.concatenate([K, -K])                 # half space -> full region (k and -k)
    if mode == "wd":
        K = K[np.abs(K).max(axis=1) <= CROP]
        dots_end = CROP + 9.5 + PAD              # room for the "..." (right, top, bottom)
        xlims.append((-np.abs(K[:, 0]).max() - PAD, dots_end))   # no "..." on the left
        ylims.append((-dots_end, dots_end))
    else:
        lim = np.abs(K).max() + PAD
        xlims.append((-lim, lim))
        ylims.append((-lim, lim))
    regions.append(K)

# each panel is as wide as its own region -> no wasted white between panels
widths = np.array([hi - lo for lo, hi in xlims])
heights = np.array([hi - lo for lo, hi in ylims])
fig, axes = plt.subplots(
    1, 3, figsize=(SCALE * widths.sum(), SCALE * heights.max()),
    gridspec_kw=dict(width_ratios=widths, wspace=0.02),
)
for ax, K, xlim, ylim, (mode, name, color) in zip(axes, regions, xlims, ylims, PANELS):
    if mode == "wd":
        t = CROP + np.array([3.5, 6.5, 9.5])     # "..." at the end of each arm (not the left one,
        z = np.zeros_like(t)                     # which would sit between the L1 and WD panels)
        dots = np.concatenate([np.c_[t, z], np.c_[z, t], np.c_[z, -t]])
        ax.scatter(dots[:, 0], dots[:, 1], s=28, color="black", linewidths=0)
    ax.scatter(K[:, 0], K[:, 1], s=20, color=color, linewidths=0, alpha=0.7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    # bottom-left corner: x relative to the panel, y relative to the figure (same baseline)
    ax.text(0.03, 0.03, name, ha="left", va="bottom", fontsize=14,
            transform=blended_transform_factory(ax.transAxes, fig.transFigure))

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
out = Path(__file__).with_name("exclusion_regions.png")
fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.05)
print(out)