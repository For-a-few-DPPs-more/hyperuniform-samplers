# Parameters

`sample_l2`, `sample_l1` and `sample_wd` share the same signature:

```python
sample_l2(N, D, Chi=0.43, n_iter=150, lr=1.0, quasi_newton_finition=True,
          seed=None, warmstart=None, target=None, verbose=0)
```

## Main params

| parameter | default | description |
|-----------|---------|-------------|
| `N` | (required) | Number of points. |
| `D` | (required) | Dimension. The points live on the torus [0, 1)^D. |
| `Chi` | `0.43` | Fraction of the D (N - 1) degrees of freedom that become constrained Fourier modes: about `Chi * D * (N - 1)` wave vectors are kept in the exclusion region. A larger `Chi` means a larger exclusion region, hence a harder optimisation. |
| `target` | `None` | Optional target point cloud. The sampler then matches its Fourier coefficients instead of driving them to zero, which gives point sets with a prescribed (non-uniform) density. |
| `verbose` | `0` | Prints the backend, the number of modes, the loss every 20 iterations and the L-BFGS summary. |

## Advanced params

| parameter | default | description |
|-----------|---------|-------------|
| `n_iter` | `150` | Number of gradient steps. It also caps the L-BFGS phase (at most `n_iter` iterations and `10 * n_iter` loss evaluations). |
| `lr` | `1.0` | Scale of the step size. The initial step is `0.1 * lr * N^(-1/D)`; it then adapts on its own (x1.05 when the loss decreases, x0.8 otherwise). |
| `quasi_newton_finition` | `True` | Refines the result with L-BFGS after the gradient steps. Turn it off for a faster, coarser result. |
| `warmstart` | `None` | Initial point cloud (e.g. a Sobol sequence). By default: uniform random points, except for `sample_wd`, which starts from a quick `sample_l1` run (`Chi=0.5`, no L-BFGS). |
| `seed` | `None` | Random seed (initial points, tie-breaking between wave vectors of the same norm). Set it for reproducible results. |

## Module settings

Backend, precision and NUFFT tolerance are not sampler arguments: they are set once at the module
level, and apply both to the samplers and to the structure factor (`plot_structure_factor`), so
that points are always measured the way they were sampled.

| setting | default | set / read | description |
|---------|---------|------------|-------------|
| backend | first available: keops >> torch >> cufinufft >> finufft | `select_backend(dimensions, backend)` / `get_backend(D)` | Chosen per dimension. (cu)finufft only for D <= 3 and needs extra installations (see the README). |
| precision | `"float32"` | `select_precision(p)` / `get_precision()` | `"float32"` or `"float64"`, for all dimensions. float32 is much faster on GPU and uses half the memory, but limits how small the structure factor can get: S(k) ≤ α N + β \|k\|², with α = 2·10⁻¹⁴ and β = 3·10⁻¹² (see the README). Use float64 to optimise further. |
| nufft_tolerance | `"auto"` | `select_nufft_tolerance(tol)` / `get_nufft_tolerance()` | Relative tolerance of the `"finufft"` and `"cufinufft"` backends. `"auto"` means `1e-4` in float32 and `1e-8` in float64. Ignored by `"keops"` and `"torch"`, which are exact. Lower it (e.g. `1e-12`, with float64) to reach machine precision. |

```python
import qmc_sampler

qmc_sampler.select_backend(dimensions={2, 3}, backend="finufft")
qmc_sampler.select_precision("float64")
qmc_sampler.select_nufft_tolerance(1e-12)
```

The samplers return an `(N, D)` array of points in [0, 1)^D, with dtype `get_precision()`.
