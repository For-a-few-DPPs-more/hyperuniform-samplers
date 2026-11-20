# Parameters

`sample_l2`, `sample_l1` and `sample_wd` share the same signature:

```python
sample_l2(N, D, Chi=0.43, n_iter=150, lr=1.0, quasi_newton_finition=True,
          precision="float32", eps=None, seed=None, warmstart=None, target=None, verbose=0)
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

## Accuracy

| parameter | default | description |
|-----------|---------|-------------|
| `precision` | `"float32"` | `"float32"` or `"float64"`. float32 is much faster on GPU and uses half the memory, but limits how small the structure factor can get: S(k) ≤ α N + β \|k\|², with α = 2·10⁻¹⁴ and β = 3·10⁻¹² (see the README). Use float64 to optimise further. |
| `eps` | `None` | Relative tolerance of the `"finufft"` and `"cufinufft"` backends. By default, `1e-4` in float32 and `1e-8` in float64. Ignored by `"keops"` and `"torch"`, which are exact. Lower it (e.g. `1e-12`, with float64) to reach machine precision. |

## Backend

The backend is selected at the module level, per dimension, and require extra installations (see the Readme): 

```python
qmc_sampler.select_backend(dimensions={2, 3}, backend="finufft") 
```

The samplers return an `(N, D)` array of points in [0, 1)^D, with dtype `precision`.
