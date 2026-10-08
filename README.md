# Hyperuniform Point Clouds Database

This is the GitHub companion repository of the `Hyperuniform samples` database stored on Zenodo. The database contains hyperuniform point clouds (e.g. blue noise) on the unit hypercube `[0,1)^D`, sampled with the [`blue-sampler`](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise) Python package.

Hyperuniformity means the points are not independent, as in usual random point clouds, but carefully correlated to cover the space more evenly, while remaining disordered.

**2D example** (the `N = 1024`, `D = 2` point clouds of this database):

Chi 43:
![2D example, N = 1024, chi43](https://zenodo.org/records/23244510/files/plot_2D_sample_chi43.png)

## Installation

Install the required dependencies:

```bash
pip install numpy h5py  # Required to load the datasets from Zenodo
```

For visualization, you can optionally install `blue-sampler`:

```bash
pip install blue-sampler  # Optional: only needed for visualization
```

> **Note:** `blue-sampler` is not required if you do not need the visualization features.

## Quick Start

1. Clone this GitHub repository.
2. Open `quick_start.ipynb`.
3. Follow the notebook to:

   * download the datasets from Zenodo;
   * build new randomized datasets.

## Data characteristics

### Format

* Hosted on Zenodo [DOI 10.5281/zenodo.23237510](https://doi.org/10.5281/zenodo.23244510).
* Stored in **HDF5** for interoperability (Python, C++, MATLAB, Julia, R, ...).

### Coverage

* **Number of points:** `N = 2^10 = 1024` to `2^17 = 131072` (powers of 2). Larger `N` is obtained by tiling.
* **Dimension:** `D = 2` to `16`, plus `D = 23` and `D = 32`. Other dimensions up to 32 are obtained by projection.

## Sampling method

**Spectral optimisation** (non-uniform Fourier transform): select all wave vectors `k` within a ball `‖k‖ ≤ Kmax(N, D, χ)`, where `χ` is chosen slightly below 0.5 to ensure near-maximal spectral coverage, initialise with a random point cloud, then run a gradient descent on the Fourier loss.

We mostly follow [Morse et al.](https://doi.org/10.1103/PhysRevResearch.5.033190) (DOI: 10.1103/PhysRevResearch.5.033190). Main differences:

* **Slightly higher χ:** While Morse et al. initially recommended fixing `χ = 0.4`, the `χ` parameter plays a crucial role in the resulting sample. We provide two versions of the samples, `χ = 0.30` and `χ = 0.43`. This allows spectral optimisation up to the particle scale while staying isotropic and unordered (crystallisation empirically appears to start around `χ ≈ 0.45`).

* **Weaker stealthy criterion:** `S(k)` is the structure factor. Morse et al. reach `S(k) ≤ 10⁻⁵¹` using double-double (float128) precision. We use `float32`, much faster on GPU, and stop the descent when, for all targeted frequencies:

  | χ    | Stopping criterion |
  | ---- | ------------------ |
  | 0.30 | `S(k) ≤ 10⁻¹⁰`     |
  | 0.43 | `S(k) ≤ 10⁻⁴`      |

  We retain the value `S(k) ≤ 10⁻⁴` as stealthy since it is enough for our application and probably for most use cases. It would be difficult to go beyond it at large `χ`. Regarding the `χ = 0.30` samples, going beyond `10⁻¹⁰` would be possible, but requires reimplementing the whole sampler with at least `float64` precision.

* **Gradient descent:** Morse et al. use FIRE optimisation. We use a simple adaptive learning-rate scheme on the normalised gradient (only its direction matters): good steps boost the learning rate, bad steps slow it down. It works on any `(N, D)` out of the box, with no hyperparameter tuning. See the ![source code](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft_keops.py) of the sampler for details.

## Guarantees

By construction, the provided samples guarantee:

### Randomisability

Spectral optimisation enforces periodic boundary conditions on the hypercube, so a sample can be randomised by a simple shift:

```text
X_new = (X - s) mod 1,    s drawn uniformly in [0,1]^D
```

The full randomisation procedure exploits all possible symmetries (shift, axis flip, axis swap) of the distribution.

### Tilability

Thanks to the periodic boundary conditions, a sample can be tiled to generate millions of points. Tiling consists of combining several copies of the same periodic point cloud into a larger periodic cloud. Because the copies are exact translations of the original cloud, the resulting structure factor is directly related to the original one.

The price is a loss of stealthiness proportional to the increase in the number of points:

```text
sup_{‖k_t‖ ≤ Kmax_t(N_t, D, χ)} S_t(k_t)
=
[ sup_{‖k‖ ≤ Kmax(N, D, χ)} S(k) ] × N_t / N
```

where `N` is the number of points in the original cloud, `N_t` the number of points after tiling, and the subscript `t` denotes the tiled dataset.

In particular, if the original cloud has `N` points and satisfies `S(k) ≤ ε` in its stealthy region, a tiled cloud with `N_t` points satisfies the corresponding bound `S_t(k_t) ≤ ε × N_t/N`.

The tiling procedure below is designed to increase the number of points by factors of two while preserving periodicity and avoiding a systematic alignment of the copies with the coordinate axes.

### Tiling procedure

Let `N_t = 2^K · N`, with `N = 2^17` the largest stored cloud, and let `P = ⌊D/2⌋` be the number of pairs of dimensions.

Each elementary tiling step doubles the number of points. For a pair of axes `(a, b)`, the periodic pattern is transformed using the lattice generated by `(1,1)` and `(1,-1)`, corresponding to a 45° rotation and a rescaling by `1/√2`:

```text
(u, v) ↦ ((u + v)/2, (u - v)/2) mod 1,
```

and both points

```text
(u, v) = (x_a, x_b)
```

and

```text
(u, v) = (x_a + 1, x_b)
```

are included.

Thus, each original point generates two points in the new periodic cell, so the total number of points is multiplied by two. In Fourier space, the structure factor is preserved up to the expected factor of two:

```text
S_t(k) = 2 S(k')
```

with

```text
k' = ((k_a + k_b)/2, (k_a - k_b)/2).
```

The transformation is applied to one pair of axes at a time.

**Procedure.**

| Tiling `2^K` | What is done                                                                                                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `K ≤ P`      | Perform `K` elementary steps on `K` randomly chosen pairs of axes.                                                                                                                         |
| `P < K ≤ 2P` | Perform one elementary step on each of the `P` pairs, then perform the remaining `K - P` steps using a second shifted pairing of the axes.                                                 |
| `K > 2P`     | First perform ordinary periodic tiling: `2^k` copies along each axis, with `k = ⌊K/D⌋`, multiplying `N` by `2^(kD)`. Then apply the previous procedure to the remaining factor `2^(K-kD)`. |

The pairings are drawn at random from the seed. The procedure slightly breaks isotropy, with no practical consequence.

## Stealthiness

`S(k) ≤ 10⁻⁴` for every wave vector with `‖k‖ ≤ Kmax(N, D, χ)`. For `χ = 0.43`, the ball of radius `Kmax` covers `2χ = 86%` of the spectral domain reachable with `N` points (this is what `χ = 0.43` means). The `χ = 0.30` dataset guarantees `S(k) ≤ 10⁻¹⁰`, with a coverage of `2χ = 60%`.

Here, for a wave vector `k = (k₁, ..., k_D)`,

```text
S(k) = |Σⱼ exp(2iπ ⟨k, xⱼ⟩)|² / N
```

so the Monte Carlo integration error of the Fourier test function `x ↦ exp(2iπ ⟨k, x⟩)` is `|Σⱼ exp(2iπ ⟨k, xⱼ⟩)| / N = sqrt(S(k) / N) ≤ 10⁻² / √N`.

The figure below shows the curve `k ↦ S(k)` for every dimension `D = 2` to `16`, at `N = 1024`. The norm `‖k‖` is normalised by the inverse interparticle distance `1/δ`, with `δ = N^(-1/D)`, so that the abscissa `k = 1` corresponds to wave vectors satisfying `k₁² + ... + k_D² ≈ 1/δ²`. The leftmost point of each curve corresponds to the smallest nonzero frequencies of the unit hypercube: vectors with a single nonzero component in `{-1, 1}`, such as `(±1, 0, ..., 0)`.

![Structure factor S(k) for D = 2 to 16, N = 1024](https://zenodo.org/records/23244510/files/fouriererror_all_samples_from_paper_chi30.png)
![Structure factor S(k) for D = 2 to 16, N = 1024](https://zenodo.org/records/23244510/files/fouriererror_all_samples_from_paper_chi43.png)

## Use cases

* **Monte Carlo / randomised QMC integration**, especially for periodic or compactly supported functions with smooth, low-frequency spectral content. Random shifts give independent replicas for reliable error bars. For other function spaces (e.g. discontinuous integrands, low-rank decomposition), other QMC methods will probably give lower discrepancy.
* **Rendering and image synthesis:** pixel/sub-pixel anti-aliasing, path-tracing sample dimensions, dithering and halftoning, blue-noise textures for denoising.
* **Spatial sampling:** homogeneous, isotropic particle initialisation for simulations (SPH, molecular dynamics), Poisson-disk-like placement, point sampling of geometry.
* **Machine learning and design of experiments:** initial points for Bayesian or black-box optimisation, hyperparameter search, training-set sampling for surrogates and PINNs, uncertainty propagation.
* **Hyperuniformity research:** a reproducible reference set to study structure-factor behaviour, the χ-dependent transition toward crystallisation, and dimension scaling (`D = 2` to `32`), or to benchmark new samplers.
* **Large-scale sampling by tiling:** replicate a cloud to get millions of points, at the cost of stealthiness degrading proportionally to `N_t / N`.

## Acknowledgements

The GPU acceleration of the kernel reductions used in the sampler relies on the [PyKeOps](https://www.kernel-operations.io) library:

> Charlier, Feydy, Glaunès, Collin, Durif, "Kernel Operations on the GPU, with Autodiff, without Memory Overflows", *Journal of Machine Learning Research* 22(74), 2021, pp. 1-6.

## Licence

The database and the corresponding source code are free to use under the **MIT licence**.
