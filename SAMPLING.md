## Sampling method

This file describes how the samples were generated, for reproducibility.

The method is **spectral optimisation** (non-uniform Fourier transform): select all wave vectors `k` within a ball `‖k‖ ≤ Kmax(N, D, χ)`, where `χ` is chosen slightly below 0.5 to ensure near-maximal spectral coverage, initialise with a random point cloud, then run a gradient descent on the Fourier loss.

We mostly follow [Morse et al.](https://doi.org/10.1103/PhysRevResearch.5.033190) (DOI: 10.1103/PhysRevResearch.5.033190). Main differences:

* **Slightly higher χ:** While Morse et al. initially recommended fixing χ = 0.4, the χ parameter plays a crucial role in the resulting sample. We provide two versions of the samples, `χ = 0.30` and `χ = 0.43`. This allows spectral optimisation up to the particle scale while staying isotropic and unordered (crystallisation empirically appears to start around χ ≈ 0.45).

* **Weaker stealthy criterion:** `S(k)` is the structure factor. Morse et al. reach S(k) ≤ 10⁻⁵¹ using double-double (float128) precision. We use float32, much faster on GPU, and stop the descent when, for all targeted frequencies:

  | χ    | Stopping criterion |
  | ---- | ------------------ |
  | 0.30 | `S(k) ≤ 10⁻¹⁰`     |
  | 0.43 | `S(k) ≤ 10⁻⁴`      |

  We retain the value S(k) ≤ 10⁻⁴ as stealthy since it is enough for our application and probably for most use cases. It would be difficult to go beyond it at large χ. Regarding the χ = 0.30 samples, going beyond 10⁻¹⁰ would be possible, but requires reimplementing the whole sampler with at least float64 precision.

* **Gradient descent:** Morse et al. use FIRE optimisation. We use a simple adaptive learning-rate scheme on the normalised gradient (only its direction matters): good steps boost the learning rate, bad steps slow it down. It works on any `(N, D)` out of the box, with no hyperparameter tuning. See the [source code](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft_keops.py) of the sampler for details.

The samples were collected by running the notebook in `src/generate_all_samples_from_the_paper.ipynb` on a Google Colab T4 GPU. The optimisation algorithm has O(N²·D²) complexity per single (N,D) point cloud and thus requires intensive computation, making a GPU nearly mandatory for large point clouds. We ran χ = 0.30 and χ = 0.43, looping over all numbers of points N = 2¹⁰, …, 2¹⁷ and over dimensions D = 2 to 16, plus D = 23 and D = 32, which took 5 hours in total.

Note: using the `pykeops` Lazytensor framework make the spectral loss computation significantly faster. Pykeops is a high performance library for efficient kernel reduction.
