## Sampling method

Spectral optimisation (non-uniform Fourier transform): select all wave vectors "k" within a ball "‖k‖ ≤ Kmax(N, D, χ)", where "χ" is chosen slightly below 0.5 to ensure near-maximal spectral coverage. Starting from a random point cloud, we then minimise the Fourier loss by gradient descent.

We mostly follow "Morse et al." (https://doi.org/10.1103/PhysRevResearch.5.033190). The main differences are:

* Slightly higher χ: Morse et al. initially recommended fixing "χ = 0.4", but "χ" has a strong influence on the resulting samples. We provide two sets of samples, "χ = 0.30" and "χ = 0.43". This allows spectral optimisation up to the particle scale while preserving isotropy and avoiding ordering. Crystallisation empirically starts to appear around "χ ≈ 0.45".

* Weaker stealthy criterion: "S(k)" denotes the structure factor. Morse et al. reach "S(k) ≤ 10⁻⁵¹" using double-double (float128) precision. We use float32, which is much faster on GPUs, and stop the optimisation when all targeted frequencies satisfy:
  
  χ| Stopping criterion
  0.30| "S(k) ≤ 10⁻¹⁰"
  0.43| "S(k) ≤ 10⁻⁴"
  
  We consider "S(k) ≤ 10⁻⁴" sufficiently stealthy for our application. Achieving lower values becomes increasingly difficult at large "χ". For "χ = 0.30", lower values are attainable in principle, but would require reimplementing the sampler with at least float64 precision.

* Gradient descent: Morse et al. use FIRE optimisation. We instead use a simple adaptive learning-rate scheme based on the normalised gradient: only its direction matters, while good steps increase the learning rate and bad steps decrease it. The scheme works directly for arbitrary "(N, D)" without hyperparameter tuning. See the "sampler source code" (https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft_keops.py) for details.
