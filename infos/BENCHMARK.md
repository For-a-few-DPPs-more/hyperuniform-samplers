# Benchmark for QMC Methods

Comparison of the `l2` and `wd` point clouds with standard QMC and random samplers, based on `scipy.stats.qmc.discrepancy` (lower = better).

Run the benchmark with:

```bash
pip install scipy
python src/benchmark_data.py
```

## Geometric mean of the discrepancy relative to Sobol

For each sampler and each discrepancy, we compute the geometric mean over all configurations of

```
discrepancy(sampler) / discrepancy(Sobol)
```

A value below 1 means better than Sobol. Sobol is 1 by construction.

| Method | CD | WD | MD |
|---|---:|---:|---:|
| Sobol (SciPy QMC) | 1.00 | 1.00 | 1.00 |
| Halton (SciPy QMC) | 1.72 | 1.47 | 1.62 |
| **wd (ours)** | 2.72 | 1.28 | 1.88 |
| LHS random-cd (SciPy QMC) | 3.45 | 3.51 | 3.70 |
| **l2 (ours)** | 17.03 | 9.17 | 15.92 |
| LHS (SciPy QMC) | 25.66 | 12.45 | 18.72 |
| i.i.d. uniform (baseline) | 87.64 | 53.94 | 87.46 |

**Setup.** Averages are taken over three sample sizes (N ∈ {1024, 2048, 4096}) and seven dimensions (D ∈ {2, ..., 8}), i.e. 21 configurations per discrepancy (CD: centered, WD: wrap-around, MD: mixture).

## Reading the results

- **`wd`** is the closest to the QMC sequences: it is better than Halton on the wrap-around discrepancy (1.28 vs 1.47) and better than `LHS random-cd` on all three measures. It remains behind Sobol, which is the best method on this benchmark.
- **`l2`** is not designed to minimise discrepancy: its exclusion region is an isotropic ball, which targets blue-noise quality rather than axis-aligned uniformity. It still improves over plain `LHS` and by a factor of 3 to 6 over i.i.d. sampling.
- These samples are therefore not drop-in replacements for Sobol or Halton. They are complementary: use `wd` for QMC integration when the integrand has favourable low-frequency spectral properties or low effective dimension, and `l2` when isotropy matters.