# Projection Effects

While the Zenodo database natively contains point clouds for $D = 2$ to $16$, plus $D = 20, 24, 28, 32$, other dimensions are not materialised in the database. Instead, they are obtained by projecting a materialised point cloud to a lower dimension.

Projecting a $D$-dimensional point cloud to a lower dimension $D_{\text{proj}}$ degrades its quality: the exclusion region shrinks, which results in $\chi \to \chi_{\text{eff}} < \chi$. This quality loss is moderate as long as the difference $D - D_{\text{proj}}$ is not too large.

In practice, the tables below show the impact of projection when a point cloud with dimension $17 \le D_{\text{proj}} \le 31$ is requested, for each of the two dataset families (`l2` and `wd`).

The nominal $\chi$ parameter, which controls the number of targeted wave vectors, is set to $0.43$. We report the relative ratio

$$
\frac{\chi_{\text{eff}}}{0.43} \times 100\%,
$$

which **should be as close to 100% as possible**.

In addition to projection, the phenomenon of incomplete orbits (see [ORBITS.md](ORBITS.md)) also affects the effective $\chi$ coefficient. The following tables include **both effects: incomplete orbits and projection**.

## 1. Projection from $D = 20$

The materialised point clouds have dimension $D = 20$. The projected dimensions considered here are $D_{\text{proj}} = 17, 18, 19$.

### `l2`

| $N$ | $D_{\text{proj}} = 20$ | $D_{\text{proj}} = 19$ | $D_{\text{proj}} = 18$ | $D_{\text{proj}} = 17$ |
| ---: | ---: | ---: | ---: | ---: |
| $2^{10}$ | 56.38% | 48.16% | 40.78% | 34.20% |
| $2^{11}$ | 28.18% | 24.07% | 20.38% | 17.09% |
| $2^{12}$ | 14.08% | 12.03% | 10.19% | 8.54% |
| $2^{13}$ | 62.09% | 50.06% | 39.87% | 31.32% |
| $2^{14}$ | 31.04% | 25.03% | 19.93% | 15.66% |
| $2^{15}$ | 15.52% | 12.51% | 9.97% | 7.83% |
| $2^{16}$ | 51.91% | 39.39% | 29.42% | 21.58% |
| $2^{17}$ | 25.95% | 19.69% | 14.71% | 10.79% |

### `wd`

| $N$ | $D_{\text{proj}} = 20$ | $D_{\text{proj}} = 19$ | $D_{\text{proj}} = 18$ | $D_{\text{proj}} = 17$ |
| ---: | ---: | ---: | ---: | ---: |
| $2^{10}$ | 49.10% | 44.49% | 40.10% | 35.94% |
| $2^{11}$ | 90.09% | 80.08% | 70.79% | 62.19% |
| $2^{12}$ | 95.75% | 83.79% | 72.85% | 62.90% |
| $2^{13}$ | 83.90% | 73.39% | 63.79% | 55.05% |
| $2^{14}$ | 78.24% | 67.93% | 58.55% | 50.09% |
| $2^{15}$ | 99.77% | 85.42% | 72.60% | 61.20% |
| $2^{16}$ | 98.54% | 83.04% | 69.43% | 57.56% |
| $2^{17}$ | 99.88% | 84.19% | 70.40% | 58.34% |

---

## 2. Projection from $D = 24$

The materialised point clouds have dimension $D = 24$. The projected dimensions considered here are $D_{\text{proj}} = 21, 22, 23$.

### `l2`

| $N$ | $D_{\text{proj}} = 24$ | $D_{\text{proj}} = 23$ | $D_{\text{proj}} = 22$ | $D_{\text{proj}} = 21$ |
| ---: | ---: | ---: | ---: | ---: |
| $2^{10}$ | 82.14% | 72.11% | 62.93% | 54.57% |
| $2^{11}$ | 41.05% | 36.04% | 31.45% | 27.27% |
| $2^{12}$ | 20.52% | 18.01% | 15.72% | 13.63% |
| $2^{13}$ | 10.26% | 9.01% | 7.86% | 6.82% |
| $2^{14}$ | 55.42% | 46.42% | 38.56% | 31.74% |
| $2^{15}$ | 27.71% | 23.21% | 19.28% | 15.87% |
| $2^{16}$ | 13.85% | 11.60% | 9.64% | 7.93% |
| $2^{17}$ | 57.29% | 45.68% | 36.04% | 28.10% |

### `wd`

| $N$ | $D_{\text{proj}} = 24$ | $D_{\text{proj}} = 23$ | $D_{\text{proj}} = 22$ | $D_{\text{proj}} = 21$ |
| ---: | ---: | ---: | ---: | ---: |
| $2^{10}$ | 58.20% | 53.59% | 49.18% | 44.95% |
| $2^{11}$ | 95.81% | 86.45% | 77.69% | 69.52% |
| $2^{12}$ | 57.38% | 51.92% | 46.80% | 42.01% |
| $2^{13}$ | 98.76% | 87.83% | 77.74% | 68.45% |
| $2^{14}$ | 92.51% | 82.29% | 72.84% | 64.14% |
| $2^{15}$ | 82.68% | 72.14% | 62.64% | 54.09% |
| $2^{16}$ | 94.68% | 83.14% | 72.63% | 63.08% |
| $2^{17}$ | 97.63% | 84.76% | 73.19% | 62.82% |

---

## 3. Projection from $D = 28$

The materialised point clouds have dimension $D = 28$. The projected dimensions considered here are $D_{\text{proj}} = 25, 26, 27$.

### `l2`

| $N$ | $D_{\text{proj}} = 28$ | $D_{\text{proj}} = 27$ | $D_{\text{proj}} = 26$ | $D_{\text{proj}} = 25$ |
| ---: | ---: | ---: | ---: | ---: |
| $2^{10}$ | 6.37% | 5.92% | 5.49% | 5.07% |
| $2^{11}$ | 56.35% | 50.43% | 44.94% | 39.86% |
| $2^{12}$ | 28.17% | 25.21% | 22.46% | 19.93% |
| $2^{13}$ | 14.08% | 12.60% | 11.23% | 9.96% |
| $2^{14}$ | 90.10% | 77.49% | 66.26% | 56.30% |
| $2^{15}$ | 45.05% | 38.75% | 33.13% | 28.15% |
| $2^{16}$ | 22.52% | 19.37% | 16.56% | 14.07% |
| $2^{17}$ | 11.26% | 9.69% | 8.28% | 7.04% |

### `wd`

| $N$ | $D_{\text{proj}} = 28$ | $D_{\text{proj}} = 27$ | $D_{\text{proj}} = 26$ | $D_{\text{proj}} = 25$ |
| ---: | ---: | ---: | ---: | ---: |
| $2^{10}$ | 67.29% | 62.69% | 58.26% | 53.99% |
| $2^{11}$ | 99.98% | 91.07% | 82.71% | 74.86% |
| $2^{12}$ | 70.93% | 65.02% | 59.43% | 54.15% |
| $2^{13}$ | 89.86% | 81.62% | 73.90% | 66.67% |
| $2^{14}$ | 73.29% | 66.45% | 60.04% | 54.06% |
| $2^{15}$ | 71.54% | 64.57% | 58.06% | 52.01% |
| $2^{16}$ | 99.96% | 89.11% | 79.13% | 69.98% |
| $2^{17}$ | 65.26% | 58.31% | 51.89% | 45.99% |

---

## 4. Projection from $D = 32$

The materialised point clouds have dimension $D = 32$. The projected dimensions considered here are $D_{\text{proj}} = 29, 30, 31$.

### `l2`

| $N$ | $D_{\text{proj}} = 32$ | $D_{\text{proj}} = 31$ | $D_{\text{proj}} = 30$ | $D_{\text{proj}} = 29$ |
| ---: | ---: | ---: | ---: | ---: |
| $2^{10}$ | 7.27% | 6.83% | 6.39% | 5.97% |
| $2^{11}$ | 74.07% | 67.25% | 60.85% | 54.88% |
| $2^{12}$ | 37.03% | 33.61% | 30.42% | 27.43% |
| $2^{13}$ | 18.51% | 16.81% | 15.21% | 13.71% |
| $2^{14}$ | 9.26% | 8.40% | 7.60% | 6.86% |
| $2^{15}$ | 68.44% | 60.04% | 52.43% | 45.58% |
| $2^{16}$ | 34.22% | 30.02% | 26.22% | 22.79% |
| $2^{17}$ | 17.11% | 15.01% | 13.11% | 11.39% |

### `wd`

| $N$ | $D_{\text{proj}} = 32$ | $D_{\text{proj}} = 31$ | $D_{\text{proj}} = 30$ | $D_{\text{proj}} = 29$ |
| ---: | ---: | ---: | ---: | ---: |
| $2^{10}$ | 76.38% | 71.79% | 67.35% | 63.04% |
| $2^{11}$ | 38.17% | 35.88% | 33.66% | 31.51% |
| $2^{12}$ | 85.70% | 79.33% | 73.28% | 67.52% |
| $2^{13}$ | 97.67% | 89.42% | 81.64% | 74.33% |
| $2^{14}$ | 92.14% | 84.50% | 77.30% | 70.51% |
| $2^{15}$ | 91.39% | 83.50% | 76.09% | 69.12% |
| $2^{16}$ | 99.92% | 89.98% | 80.78% | 72.29% |
| $2^{17}$ | 86.67% | 78.45% | 70.80% | 63.69% |

---

## 5. Summary

The tables show that projection generally reduces the effective spectral exclusion budget. The magnitude of the reduction depends on the source dimension, the projected dimension, the point-cloud size $N$, and the dataset family.

- **Effect of projection:** For both `l2` and `wd`, the effective ratio generally decreases as the projected dimension moves farther below the source dimension.
- **Difference between dataset families:** The `wd` family generally retains a larger fraction of the nominal exclusion budget than `l2`, although the amount of degradation depends on the particular point cloud.
- **Effect of incomplete orbits:** The ratio can vary substantially with $N$, even at a fixed source dimension. This reflects the incomplete-orbit effect described in [ORBITS.md](ORBITS.md), in addition to the loss caused by projection.
- **Choosing a projection:** When possible, use a source dimension close to the requested dimension to limit the loss of spectral quality. Compare the tabulated ratios for the relevant $N$ and dataset family when selecting a point cloud.

The reported values are the ratios $\chi_{\text{eff}} / 0.43$, expressed as percentages. Values closer to $100\%$ indicate better preservation of the nominal spectral exclusion budget.