# blue-sampler

Generate large **stealthy point patterns** on the unit torus [0, 1)^D.

Stealthy point patterns have vanishing density fluctuations at low ("blue")
frequencies, making them useful for Monte Carlo integration, image
stippling, and any application that needs well-spread, low-discrepancy
points.

The blue noise sampler implemented here have linear complexity in the number of
points and the dimension (per gradient step), and run in under 10 minutes for 
1 million points 2D points.

## Installation

```bash
pip install blue_sampler
```

## Quick start

```python
import blue_sampler as blue

# 10 000 points in 2-D
x = blue.sample_points(N=10_000)
blue.plot(x)
blue.plot_structure_factor(x)

# arbitrary dimension D
x = blue.sample_points(N=2_000, D=5)

# image stippling
x = blue.im2points(image="zebra.jpg")
```

## Supported dimensions

| D    | Notes                                 |
|------|----------------------------------------|
| 2    | Fast, recommended for exploration      |
| 3    | ~2x slower than 2-D                    |
| 4–5  | Requires more iterations               |
| ≥ 6  | Experimental, for small sample sizes   |

## License

MIT