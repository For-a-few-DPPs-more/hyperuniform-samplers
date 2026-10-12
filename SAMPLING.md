## Sampling method

This file describes how the samples were generated, for reproducibility.

The method is **spectral optimisation** (non-uniform Fourier transform): select all wave vectors `k` within a ball `‖k‖ ≤ Kmax(N, D, χ)`, where `χ` is chosen slightly below or equal to 0.5 to ensure near-maximal spectral coverage, initialise with a random point cloud, then run a gradient descent on the Fourier loss.

We mostly follow [Morse et al.](https://doi.org/10.1103/PhysRevResearch.5.033190) (DOI: 10.1103/PhysRevResearch.5.033190). Main differences:

* **Slightly higher χ:** While Morse et al. initially recommended fixing χ = 0.4, the χ parameter plays a crucial role in the resulting sample. We provide two such versions of the samples, `χ = 0.30` and `χ = 0.43`. This allows spectral optimisation up to the particle scale while staying isotropic and unordered (crystallisation empirically appears to start around χ ≈ 0.45).

  A third version, built with an L1 ball and `χ = 0.50` (see below), serves a different purpose. It is neither stealthy (in the strict sense S(k) ≤ 10⁻⁴) nor isotropic, but it is better suited to QMC methods, as it reaches a discrepancy similar to that of Sobol sequences.

* **Weaker stealthy criterion:** `S(k)` is the structure factor. Morse et al. reach S(k) ≤ 10⁻⁵¹ using double-double (float128) precision. We use float32, much faster on GPU, and stop the descent when, for all targeted frequencies:

  | χ    | Stopping criterion |
  | ---- | ------------------ |
  | 0.30 | `S(k) ≤ 10⁻¹⁰`     |
  | 0.43 | `S(k) ≤ 10⁻⁴`      |
  | 0.50 (L1 ball) | `S(k) ≤ 10⁻²` |

  We retain the value S(k) ≤ 10⁻⁴ as stealthy since it is enough for our application and probably for most use cases. It would be difficult to go beyond it at large χ. Regarding the χ = 0.30 samples, going beyond 10⁻¹⁰ would be possible, but requires reimplementing the whole sampler with at least float64 precision.

* **L1 ball and χ = 0.50 (`--chi L1-50`):** the third version of the samples selects the wave vectors with the L1 norm `‖k‖₁ = Σ|kᵢ|` instead of the Euclidean norm, and pushes χ up to 0.50, i.e. `2χ = 100%` of the spectral domain reachable with N points. It is aimed at QMC integration (an L1 ball has a cross-polytope geometry, a diamond in 2D, suited to integrands with a low effective dimension or a strong coordinate-wise structure), whereas the L2 versions are recommended for blue noise, where strict stealthiness and isotropy matters more than discrepancy. Source code for the special L1 version is in [`src/qmc_sampler.py`](https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/blob/main/src/qmc_sampler.py).

* **Gradient descent:** Morse et al. use FIRE optimisation. We use a simple adaptive learning-rate scheme on the normalised gradient (only its direction matters): good steps boost the learning rate, bad steps slow it down. We perform 150 iterations of this auto-scaled gradient step. It works on any `(N, D)` out of the box, with no hyperparameter tuning. See the [source code](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft_keops.py) of the sampler for details.

The samples were collected by running the notebook in `src/generate_all_samples_from_paper.ipynb` on a Google Colab T4 GPU. The optimisation algorithm has O(N²·D²) complexity per single (N,D) point cloud and thus requires intensive computation, making a GPU nearly mandatory for large point clouds. We ran χ = 0.30, χ = 0.43 and χ = L1-0.50, looping over all numbers of points N = 2¹⁰, …, 2¹⁷ and over dimensions D = 2 to 16, plus D = 23 and D = 32, which took 2 hours in total per χ version.

Note: using the `pykeops` Lazytensor framework make the spectral loss computation significantly faster. Pykeops is a high performance library for efficient kernel reduction.

## Large point clouds

We used the `finufft` python library to sample additional massive point clouds in 2d / 3d. They can be downloaded manually from the Zenodo database. finufft essentially allow to replace the O(N^2) complexity of the gradient descent to O(N log N) using a fast fourier transform. Here is the [source code](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft.py) to run the finufft based pipeline.
