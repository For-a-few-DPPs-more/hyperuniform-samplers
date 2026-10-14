## Sampling method

This file describes how the samples were generated, for reproducibility.

The method is **spectral optimisation** (non-uniform Fourier transform): select the most important wave vectors `k` according to a chosen exclusion region (Euclidean ball, L1 ball, or star ball), where `χ` is chosen slightly below or equal to 0.5 to ensure near-maximal spectral coverage, initialise with a random point cloud, then run a gradient descent on the Fourier loss.

We mostly follow [Morse et al.](https://doi.org/10.1103/PhysRevResearch.5.033190) (DOI: 10.1103/PhysRevResearch.5.033190). Main differences:

* **Slightly higher χ:** While Morse et al. initially recommended fixing χ = 0.4, the χ parameter plays a crucial role in the resulting sample. We provide two Euclidean-ball versions of the samples, `χ = 0.30` and `χ = 0.43`. This allows spectral optimisation up to the particle scale while staying isotropic and unordered (crystallisation empirically appears to start around χ ≈ 0.45).

Two additional versions, built with an L1 ball (`χ = 0.50`) and a star ball (`χ = 0.50`), serve a different purpose. They are neither stealthy in the strict sense (S(k) ≤ 10⁻⁴) nor isotropic, but they are better suited to QMC methods. The WD (star-ball) version is specifically designed for worst-case QMC error bounds induced by the wrap-around kernel and is recommended for irregular / non-smooth integrands, and reaches a discrepancy similar to that of Sobol sequences. The L1 ball version is the intermediate, between WD and L2, recommanded for usual smooth integrands.

* **Weaker stealthy criterion:** `S(k)` is the structure factor. Morse et al. reach S(k) ≤ 10⁻⁵¹ using double-double (float128) precision. We use float32, much faster on GPU, and stop the descent after 150 iterations. For all targeted frequencies:

| χ                | Exclusion region     | goal                    |
| ---------------- | -------------------- | ----------------------- |
| 0.30             | L2 ball (Euclidean)  | `S(k) ≤ 10⁻¹⁰`          |
| 0.43             | L2 ball (Euclidean)  | `S(k) ≤ 10⁻⁴`           |
| 0.50 (L1 ball)   | L1 ball              | `S(k) ≤ 10⁻²`           |
| 0.50 (WD star)   | star ball (product)  | minimal WD discrepancy  |


We retain the value S(k) ≤ 10⁻⁴ as stealthy since it is enough for our application and probably for most use cases. It would be difficult to go beyond it at large χ. Regarding the χ = 0.30 samples, going beyond 10⁻¹⁰ would be possible, but requires reimplementing the whole sampler with at least float64 precision.

* **L1 ball and χ = 0.50 (`--chi L1-50`):** this version selects the wave vectors with the L1 norm `‖k‖₁ = Σ|kᵢ|` instead of the Euclidean norm, and pushes χ up to 0.50, i.e. `2χ = 100%` of the spectral domain reachable with N points. It is aimed at QMC integration (an L1 ball has a cross-polytope geometry, a diamond in 2D, suited to integrands with a low effective dimension or a strong coordinate-wise structure), whereas the L2 versions are recommended for blue noise, where strict stealthiness and isotropy matters more than discrepancy. [Source code](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/tree/main/src/qmc_sampler) for all the code.

* **Star ball (WD) and χ = 0.50 (`--chi WD-50`):** this version selects the wave vectors according to the exact product weight of the wrap-around kernel
  \[
  r(k) = \prod_{i=1}^D \hat{K}(k_i),\qquad
  \hat{K}(m) = \begin{cases}
  1 & \text{if } m = 0,\\
  \dfrac{1}{2\pi^2 m^2} & \text{if } m \neq 0.
  \end{cases}
  \]
  The top-M modes (M ≈ χ · D · (N−1)) ordered by decreasing r(k) form the *star ball*. This is the natural exclusion region that appears in the exact worst-case error bound for the star discrepancy. Consequently the WD-50 samples are particularly well suited to QMC integration of irregular or non-smooth integrands.

* **Gradient descent:** Morse et al. use FIRE optimisation. We use a simple  `bold driver`scheme on the normalised gradient (only its direction matters): good steps boost the learning rate, bad steps slow it down. We perform 150 iterations of this auto-scaled gradient step. It works on any `(N, D)` out of the box, with no hyperparameter tuning.

The samples were collected by running the notebook in `src/generate_all_samples_from_paper.ipynb` on a Google Colab T4 GPU. The optimisation algorithm has O(N²·D²) complexity per single (N,D) point cloud and thus requires intensive computation, making a GPU nearly mandatory for large point clouds. We ran χ = 0.30, χ = 0.43, χ = L1-0.50 and χ = WD-0.50, looping over all numbers of points N = 2¹⁰, …, 2¹⁷ and over dimensions D = 2 to 16, plus D = 23 and D = 32, which took roughly 2 hours in total per χ version.

Note: using the `pykeops` Lazytensor framework make the spectral loss computation significantly faster. Pykeops is a high performance library for efficient kernel reduction.

## Large point clouds

We used the `finufft` python library to sample additional massive point clouds in 2d / 3d. They can be downloaded manually from the Zenodo database. finufft essentially allow to replace the O(N^2) complexity of the gradient descent to O(N log N) using a fast fourier transform. Here is the [source code](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft.py) to run the finufft based pipeline.
