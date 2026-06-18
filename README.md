In this repository, we implement a few point process samplers from the statistical physics literature, run hyperuniformity diagnostics, and compare the complexity of the samplers.

📦 This repository contains the blue_sampler package (see blue.src).
blue_sampler is available on PyPi:

```python
#pip install blue_sampler

import blue_sampler as blue

x = blue.sample_points(N = 10_000, D = 3)
blue.plot(x)
blue.plot_structure_factor(x) #estimated using scattering intensity

quad = blue.sample_tessels(N = 2**10) #will sample 2D quadrilaterals following a blue pattern
```
