# blue-sampler

Generate **stealthy point patterns** — low-discrepancy, spectrally isotropic
samples on the unit torus [0, 1)^D.

Stealthy patterns suppress long-range density fluctuations while remaining
aperiodic.  They are useful in rendering, quadrature, and computational
physics wherever quasi-random, isotropic spatial coverage is needed.

## Installation

```bash
pip install blue_sampler
```

## Quick start

```python
import blue_sampler as blue

# 10 000 points in 2-D
x = blue.sample_points(10_000)
blue.plot(x)
blue.plot_structure_factor(x)

# 3-D
x = blue.sample_points(5_000, D=3)

# Tessels (only supports 2D and  N has to be a power of 2)
quad = blue.sample_tessels(N=2**10)
```
## Supported dimensions

| D | Notes |
|---|-------|
| 2 | Fast, recommended for exploration |
| 3 | ~3× slower than 2-D |
|4/5| Requires more iterations |
|>= 6| Only with bruteforce method and thus for small sample size (<10_000 points)

## Algorithm overview

The pipeline alternates between:

1. **Spatial gradient** — short-range Gaussian repulsion via
   neighbour convolution on the torus.
2. **Spectral gradient** — minimises the structure factor S(k) for k below
   a chosen cut-off, using a set of all the wave vectors within an integer
   half-ball.
3. **Grid assignment** (SquareNet) — periodic re-assignment to a regular
   grid for efficient sparse local operations.

For N ≤ Z 000 a direct O(N²) bootstrap is used.  For larger N a
hierarchical strategy clones and refines a coarser solution.

## License

MIT