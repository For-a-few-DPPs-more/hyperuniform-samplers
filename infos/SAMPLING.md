## Sampling method

Here we describe how the samples were generated, for reproducibility.

The method is **spectral optimisation** (non-uniform Fourier transform). We select the
$M = \mathrm{round}(\chi \, D \, (N-1))$ most important wave vectors $k$ of an exclusion set $\Omega$
(L2 ball, L1 ball or star ball), keeping one representative per pair $(k,-k)$. We initialise with a
random $(N, D)$ point cloud, then run a gradient descent on

```math
\text{loss} = \sum_{k \in \Omega} \frac{1}{|k|^2} \left| \sum_{j=1}^{N} e^{2i\pi \langle k, x_j \rangle} \right|^2 .
```

We take $\chi \le 0.5$ because each $k$ gives two real constraints (real and imaginary parts), so we
want slightly fewer independent equations than free parameters.

We mostly follow [Morse et al. (2023)](https://doi.org/10.1103/PhysRevResearch.5.033190). Main differences:

* **Higher $\chi$.** Morse et al. recommend $\chi = 0.4$. We provide $\chi = 0.30$ and $\chi = 0.43$,
  which allows optimisation up to the particle scale while staying isotropic and unordered
  (crystallisation empirically starts around $\chi \approx 0.45$).

* **QMC versions (not isotropic).** They target integrands with a low effective dimension, where an L2 ball
  is not the optimal exclusion set. The L2 versions are recommended for blue noise.
  * **L1 ball, $\chi = 0.43$** (`--chi L1-43`): modes with the smallest $\|k\|_1 = \sum_i |k_i|$
    (cross-polytope geometry). Intermediate between WD and L2, recommended for usual smooth integrands.
  * **Star ball (WD), $\chi = 0.43$** (`--chi WD-43`): modes with the largest exact product weight of the
    wrap-around discrepancy kernel,

```math
    r(k) = \prod_{i=1}^{D} \hat{K}(k_i),
    \qquad
    \hat{K}(m) =
    \begin{cases}
    \dfrac{4}{3} & \text{if } m = 0, \\[2mm]
    \dfrac{1}{2\pi^2 m^2} & \text{if } m \neq 0.
    \end{cases}
```

    This is the exclusion set of the exact worst-case QMC error bound for the wrap-around
    discrepancy, recommended for highly low-dimensional integrands.

* **Exclusion sets in high dimension.** An important subtlety, see [Exclusionsets.md](Exclusionsets.md)
  (technical, can be read on a second reading).

* **Weaker stealthy criterion.** Morse et al. reach $S(k) \le 10^{-51}$ with `float128`. We use `float32`
  (much faster on GPU) and stop after 150 iterations. For all targeted frequencies:

  | $\chi$ | Exclusion set | Criterion |
  | :--- | :--- | :--- |
  | 0.30 | L2 ball | $S(k) \le 10^{-10}$ |
  | 0.43 | L2 ball | $S(k) \le 10^{-3}$ |
  | 0.43 (L1) | L1 ball | $S(k) \le 10^{-3}$ |
  | 0.43 (WD) | Star ball, $r(k) \ge \text{cst}$ | $S(k) \le 10^{-2}$ |

  $S(k) \le 10^{-3}$ is sufficient for most uses and hard to beat at large $\chi$. On the other hand, going beyond
  $10^{-10}$ for $\chi = 0.30$ is conceptually not difficult but would require reimplementing the sampler in at least `float64`.

* **Gradient descent.** We apply 150 steps of a gradient descent scheme. Instead of the FIRE method recommanded by Morse et.al, we use a `bold driver` scheme on the normalised gradient:
  good steps boost the learning rate, bad steps slow it down. This method has the strong advantage of working out of the box (no hyperparameter tuning) on the full (N, D) range that we considered.

The algorithm cost is $O(N^2 D^2)$ per sampled point cloud, so a GPU is nearly mandatory for the sampling. We ran the notebook `src/generate_all_samples_from_paper.ipynb`,which internally rely on following [source code](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/tree/main/src/qmc_sampler),
on a Google Colab T4 GPU for all
four ball versions, with $N = 2^{10}, \dots, 2^{17}$ and $D = 2$ to $16$, plus $D = 23$ and $D = 32$
(about 2 hours for D = 2 to 16, plus 2 extra hours for D = (23,32), per ball version). 

*Note:* Using the pykeops LazyTensor framework makes the spectral loss computation significantly faster. PyKeOps is a high-performance library for efficient kernel reduction.

## Large point clouds

In 2d / 3d, were the `FINUFFT` transform  is available (replaces the $O(N^2)$ gradient evaluation by
$O(N \log N)$ via a fast fourier transform), we additionaly sampled massive point clouds (up to 134 millions in 2d and 32 millions in 3d). They can be downloaded manually from the Zenodo database, along with up to 1 million sized 4D point clouds.
[Source code](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft.py) for the finufft, large point clouds version.
