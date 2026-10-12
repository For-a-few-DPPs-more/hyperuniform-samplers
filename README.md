# Hyperuniform Point Clouds Database

This is the GitHub companion repository of the `Hyperuniform samples` database stored on Zenodo. The database contains hyperuniform point clouds (e.g. blue noise) on the unit hypercube [0,1)^D, sampled with the [`blue-sampler`](https://pypi.org/project/blue-sampler/) Python package.

**2D example** (the N = 1024, D = 2, Chi43 point clouds of this database):

![2D example, N = 1024, chi43](https://zenodo.org/records/23327095/files/plot_2D_sample_chi43.png)

## Installation

Install the required dependencies:

- numpy (array library for python)
- h5py (required to read hdf5 files format)

```
pip install numpy h5py
```

## Quick Start

- First clone this GitHub repository:

```
git clone https://github.com/For-a-few-DPPs-more/hyperuniform-samplers.git
cd hyperuniform-samplers
```

- Then, from the `terminal` :
This script loads the full database and generates a new randomized point cloud

```
python src/load_data.py --dim all --chi all
python src/new_data.py --dim 2 --npoints 1024 --chi 30 --seed 42
# other families: --chi 43, --chi L1-50 or --chi WD-50 (for QMC integration)
```

- or from a notebook :
Open `quick_start.ipynb` and follow the instructions to load and use the database.

## Data characteristics

### Format

- Hosted on Zenodo [DOI 10.5281/zenodo.23327095](https://doi.org/10.5281/zenodo.23327095).
- Stored in **HDF5** for interoperability (Python, C++, MATLAB, Julia, R, ...).
- Size : default ≈ 712 MB (≈ 178 MB per chi family), full (see Note: Large storage) ≈ 3.2 GB

### Coverage

- **χ parameter:** four dataset families:
  Stealthy blue noise
  - χ = 0.30 (`--chi 30`)
  - χ = 0.43 (`--chi 43`) 
  QMC integration
  - χ = L1-0.50 (`--chi L1-50`)
  - χ = WD-0.50 (`--chi WD-50`)
- **Number of points:** Native: N = 2^10 to 2^17 (powers of 2). Including tiling: up to N = 2^24.
- **Dimension:** Native: D = 2 to 16, plus D = 23 and D = 32. Including projection: any 2 ≤ D ≤ 32.


| `--chi` | Exclusion region                          | χ    | Coverage `2χ` | Stealthiness `S(k) ≤` | Typical use                                      |
| ------- | ----------------------------------------- | ---- | ------------- | --------------------- | ------------------------------------------------ |
| `30`    | L2 ball (Euclidean)                       | 0.30 | 60%           | 10⁻¹⁰                 | blue noise, very strong stealthiness             |
| `43`    | L2 ball (Euclidean)                       | 0.43 | 86%           | 10⁻⁴                  | blue noise, larger exclusion region              |
| `L1-50` | L1 ball (`‖k‖₁ = Σ\|kᵢ\|`)                | 0.50 | 100%          | 10⁻²                  | QMC integration                                  |
| `WD-50` | star ball (`r(k) = ∏ 1/kᵢ²`)              | 0.50 | 100%          | minimal WD                  | QMC (worst-case irregular integrands)            |


All four families are available for the same (N, D) values.
See following (N, D) coverage table:

[![ND_coverage](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/raw/5b6cd8a40a866b7ba1324e94577ce814b5fecdc0/src/coverage_table_ND.png)](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/blob/5b6cd8a40a866b7ba1324e94577ce814b5fecdc0/src/coverage_table_ND.png)

> **Note: Large storage**
>
> In low dimensions (D ≤ 4), the Zenodo database provides additional, much larger point clouds. These must be downloaded manually (see [`larges.zip`](https://zenodo.org/records/23327095?preview_file=larges.zip), 2.7 GB). The archive contains χ = 0.30 samples up to:
>
> - **2D**: N = 2²⁷ ≈ 134 million points
> - **3D**: N = 2²⁵ ≈ 34 million points
> - **4D**: N = 2²¹ ≈ 2 million points

## Sampling method

**Spectral optimisation** (non-uniform Fourier transform): select an exclusion region (see figure below) composed of the most important wave vectors k according to a weight function (Euclidean ball, L1 ball, or star-ball product weight), where `χ` is chosen slightly below or equal to 0.5 to ensure near-maximal spectral coverage, initialise with a random point cloud, then run a gradient descent on the Fourier loss. We sample four versions of each point cloud:

- χ = 0.30 and χ = 0.43, reaching a stealthy structure factor S(k) ≤ 10⁻¹⁰ and S(k) ≤ 10⁻⁴ respectively within a Euclidean (L2) exclusion region;
- χ = L1-0.50, reaching S(k) ≤ 10⁻² inside an L1 ball (`‖k‖₁ = Σ|kᵢ|`);
- χ = WD-0.50, minimise the WD (wrap around) discrepancy as a fourier serie (no stealthiness consideration).

Exclusion region for all four types of ball, with N = 1024, D = 2:

[!exclusion ball](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/blob/fa0f202e2dbb3f7f8e25f8f9f0ee3b9aeab1efb0/src/exclusion_balls.png)

See all details in [SAMPLING.md](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/blob/main/SAMPLING.md) for reproducibility.

## Guarantees

By construction, the provided samples guarantee:

### Stealthiness

- For `χ = WD-0.50`, the star ball of product weight covers all the spectral domain reachable with N points, `2χ = 100%`, with a minimised WD disrepancy truncated on the wave vector k belonging to the top-M modes ordered by \( r(k) = \prod_i \hat{K}(k_i) \). This exclusion region is the one that appears in the exact worst-case error bound for the wrap-around kernel (star discrepancy).

- For `χ = L1-0.50`, the ball of radius Kmax covers all the spectral domain reachable with N points, `2χ = 100%`, with a controlled `S(k) ≤ 10⁻²` for every wave vector k with ‖k‖₁ ≤ Kmax(N, D, χ). Note that the norm ‖ ‖₁ here is the L1 norm (see [SAMPLING.md](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/blob/main/SAMPLING.md)).

- For `χ = 0.43`, the ball of radius Kmax covers `2χ = 86%` with a controlled `S(k) ≤ 10⁻⁴` for every wave vector k with ‖k‖ ≤ Kmax(N, D, χ), this time for a usual L2-ball region.

- For `χ = 0.30`, coverage is slightly weaker `2χ = 60%` but the S(k) control much finer: `S(k) ≤ 10⁻¹⁰`.

Recall that, for a wave vector k = (k₁, ..., k_D) with integer components in Z^D,

```
S(k) = |Σⱼ exp(2iπ ⟨k, xⱼ⟩)|² / N
```

So having a near-zero S(k) essentially means that one controls the Monte Carlo integration error of the low frequencies Fourier test function x ↦ exp(2iπ ⟨k, x⟩).

The figure below shows the curve `k ↦ S(k)` for every dimension D = 2 to 16, at N = 1024. To make the plots comparable across dimensions, the norm ‖k‖ is normalised by the inverse interparticle distance 1/δ, where δ = N^(-1/D).

- Chi = 0.30, D = 2, 3, ..., 16 :

![Structure factor S(k) for D = 2 to 16, N = 1024, chi 0.30](https://zenodo.org/records/23327095/files/fouriererror_all_samples_from_paper_chi30.png)

- Chi = 0.43, D = 2, 3, ..., 16 :

![Structure factor S(k) for D = 2 to 16, N = 1024, chi 0.43](https://zenodo.org/records/23327095/files/fouriererror_all_samples_from_paper_chi43.png)

- Chi = 0.50 (L1-norm), D = 2, 3, ..., 16 :

![Structure factor S(k) for D = 2 to 16, N = 1024, chi L1-0.50](https://zenodo.org/records/23327095/files/fouriererror_all_samples_from_paper_L1chi50.png)

### Randomisability

Spectral optimisation enforces periodic boundary conditions on the hypercube, so a sample can be randomised by a simple shift:

```
X_new = (X - s) mod 1, s drawn uniformly in [0,1]^D
```

The full randomisation procedure exploits all possible symmetries (shift, axis flip, axis swap) of the distribution.

### Extensibility

Large point clouds can be found directly in the Zenodo archive (see note: large storage) but are not included in the download script.
The database which is downloaded by the `load_data` command contains only samples with 2^10 ≤ N ≤ 2^17 and 2 ≤ D ≤ 16, plus D = 23 and D = 32. Other shapes (see the Coverage section) are produced on the fly when a point cloud is requested with `new_data`. This is possible because stealthy point clouds can be extended in two ways, at the cost of a controlled loss of quality:

- **Tiling**: more points.
- **Projection**: lower dimension.

**Tiling**:

Thanks to the periodic boundary conditions, a sample can be tiled: several copies of the same periodic point cloud are combined into a larger periodic one. This makes it possible to generate millions of points on the fly. The cost is a loss of stealthiness proportional to the increase in the number of points:

```
sup_{‖k_t‖ ≤ Kmax_t(N_t, D, χ)} S_t(k_t)
=
[ sup_{‖k‖ ≤ Kmax(N, D, χ)} S(k) ] × N_t / N
```

where N is the number of points of the original cloud, N_t the number of points after tiling, and the subscript t denotes the tiled dataset.

The procedure is described in [TILING.md](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/blob/main/TILING.md).
Example of a ×2 tiling, from N = 8192 to N_t = 16384:
![Tiling effect](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/raw/02c688c83d98b8b691ac16b9481dbcdd8c528472/src/tiling_effect.png)

**Projection**:

Dimensions that are not materialised are obtained by projecting a
higher-dimensional point cloud:

- 17 ≤ D_proj ≤ 22 is projected from D = 23
- 24 ≤ D_proj ≤ 31 is projected from D = 32

The procedure and its
impact are described in
[PROJECTION.md](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/blob/main/PROJECTION.md).

## Use cases

- **Rendering and image synthesis:** pixel/sub-pixel anti-aliasing, path-tracing sample dimensions, dithering and halftoning, blue-noise textures for denoising.
- **Spatial sampling:** homogeneous, isotropic particle initialisation for simulations (SPH, molecular dynamics), Poisson-disk-like placement, point sampling of geometry.
- **Monte Carlo / QMC integration:** Stealthy hyperuniform samples are by nature pertinent for periodic or compactly supported functions with smooth, low-frequency spectral content. The `L1-50` dataset is suitable for non-smooth integrands (discrepancy comparable to Sobol sequences). The `WD-50` dataset is specifically designed for the worst-case error bound induced by the wrap-around kernel (star discrepancy) and is therefore recommended when the integrand can be irregular. Random shifts give independent replicas for unbiased error bars.
- **Machine learning and design of experiments:** initial points for Bayesian or black-box optimisation, hyperparameter search, training-set sampling for surrogates and PINNs, uncertainty propagation.
- **Hyperuniformity research:** a reproducible reference set to study structure-factor behaviour, the χ-dependent transition toward crystallisation, and dimension scaling (`D = 2` to `32`), or to benchmark new samplers.
- **Large-scale sampling by tiling:** replicate a cloud to get millions of points, at the cost of stealthiness degrading proportionally to `N_t / N`.

## Citation

If you use this database, please cite:

```bibtex
@dataset{bardenet_lachiezerey_decacqueray_2026_hyperuniform,
author = {Bardenet, Rémi and Lachièze-Rey, Raphaël and de Cacqueray, Armand},
title = {Hyperuniform samples (stealthy point clouds)},
year = {2026},
publisher = {Zenodo},
version = {v3},
doi = {10.5281/zenodo.23327095},
url = {https://doi.org/10.5281/zenodo.23327095}
}
```

## Acknowledgements

The GPU acceleration of the kernel reductions used in the sampler relies on the [PyKeOps](https://www.kernel-operations.io) library:
> Charlier, Feydy, Glaunès, Collin, Durif, "Kernel Operations on the GPU, with Autodiff, without Memory Overflows", *Journal of Machine Learning Research* 22(74), 2021, pp. 1-6.

For the large 2D and 3D point clouds available on Zenodo in `larges.zip`, spectral optimisation is made possible by the [FINUFFT](https://github.com/flatironinstitute/finufft) library:
> Barnett, Magland, af Klinteberg, "A parallel non-uniform fast Fourier transform library based on an “exponential of semicircle” kernel", *SIAM Journal on Scientific Computing* 41(5), C479–C504, 2019.

## Licence

The database and the corresponding source code are free to use under the **MIT licence**.
