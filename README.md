# Hyperuniform Point Clouds Database

This is the GitHub companion repository of the `Hyperuniform samples` database stored on Zenodo. The database contains hyperuniform point clouds (e.g. blue noise) on the unit hypercube `[0,1)^D`, sampled with the [`blue-sampler`](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise) Python package.

Hyperuniformity means the points are not independent, as in usual random point clouds, but carefully correlated to cover the space more evenly, while remaining disordered.

**2D example** (the N = 1024, D = 2 point clouds of this database):

![2D example, N = 1024, chi43](https://zenodo.org/records/23244510/files/plot_2D_sample_chi43.png)

## Installation

Install the required dependencies:
- numpy (array library for python)
- h5py (required to load datasets from the Zenodo database)

```bash
pip install numpy h5py
```

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

See following (N, D) coverage table:

![ND_coverage](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/blob/5b6cd8a40a866b7ba1324e94577ce814b5fecdc0/src/coverage_table_ND.png)

## Sampling method

**Spectral optimisation** (non-uniform Fourier transform): select all wave vectors `k` within a ball `‖k‖ ≤ Kmax(N, D, χ)`, where `χ` is chosen slightly below 0.5 to ensure near-maximal spectral coverage, initialise with a random point cloud, then run a gradient descent on the Fourier loss. we sample two version of each point cloud: χ = 0.30 resp χ = 0.43, reaching a stealthy structure factor S(k) <= 10-10 resp S(k) <= 10-4 within the full spectral exclusion region.

See all details in Sampling.md for reproducibility. 

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

The exact tiling procedure is described in TILING.md. It allows generating new datasets with up to N = 2^24 (~16 million) points while keeping the loss in stealthiness controlled. Exemple of a ×2 tiling step from number of points N = 8192 to new number of points N_t = 16384:

![Tiling effect](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/blob/02c688c83d98b8b691ac16b9481dbcdd8c528472/src/tiling_effect.png)


## Stealthiness

`S(k) ≤ 10⁻⁴` for every wave vector with `‖k‖ ≤ Kmax(N, D, χ)`. For `χ = 0.43`, the ball of radius Kmax covers `2χ = 86%` of the spectral domain reachable with `N` points (this is what χ = 0.43 means). The `χ = 0.30` dataset guarantees `S(k) ≤ 10⁻¹⁰`, with a coverage of `2χ = 60%`.

Here, for a wave vector `k = (k₁, ..., k_D)`,

```text
S(k) = |Σⱼ exp(2iπ ⟨k, xⱼ⟩)|² / N
```

so the Monte Carlo integration error of the Fourier test function x ↦ exp(2iπ ⟨k, x⟩) is `|Σⱼ exp(2iπ ⟨k, xⱼ⟩)| / N = sqrt(S(k) / N) ≤ 10⁻² / √N`.

The figure below shows the curve `k ↦ S(k)` for every dimension `D = 2` to `16`, at N = 1024. The norm `‖k‖` is normalised by the inverse interparticle distance 1/δ, with `δ = N^(-1/D)`, so that the abscissa `k = 1` corresponds to wave vectors satisfying `k₁² + ... + k_D² ≈ 1/δ²`. The leftmost point of each curve corresponds to the smallest nonzero frequencies of the unit hypercube: vectors with a single nonzero component in `{-1, 1}`, such as `(±1, 0, ..., 0)`.

- Chi = 0.30, D = 2, 3, ..., 16 :
  
![Structure factor S(k) for D = 2 to 16, N = 1024](https://zenodo.org/records/23244510/files/fouriererror_all_samples_from_paper_chi30.png)

- Chi = 0.43, D = 2, 3, ..., 16 :

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
