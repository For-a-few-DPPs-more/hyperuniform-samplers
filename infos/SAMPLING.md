## Sampling method

This file describes how the samples were generated, for reproducibility.

The method is **spectral optimisation** (non-uniform Fourier transform): select the $(N-1) \cdot D \cdot \chi$ most important wave vectors `k` according to a chosen exclusion set $\Omega$ (Euclidean ball, L1 ball, or star ball), were we take only one representant among (k, -k) per symetry and $\chi$ chosen slightly below or equal to 0.5 to ensure near-maximal spectral coverage, initialise with a random (N, D) point cloud, then run a gradient descent on the Fourier loss:

$$\text{loss} = \sum_{k \in \Omega} 1/\left\vert{}k\right\vert{}^2 \left\vert{} \sum_{\text{point} \in \text{points}} e^{2i\pi \langle k, \text{point}\rangle} \right\vert{}^2$$

The heuristic to determine that $\chi$ should be <= 0.5 is: "a bit less indepandant equations than free parameters of the point clouds", were each k constraint internaly counts two constraints, for the real and imaginary part.

We mostly follow [Morse et al. (2023)](https://doi.org/10.1103/PhysRevResearch.5.033190). Main differences:

* **Slightly higher $\chi$:** While Morse et al. initially recommended fixing $\chi = 0.4$, the parameter plays a crucial role in the resulting sample. We provide two versions of the samples, $\chi = 0.30$ and $\chi = 0.43$. This allows spectral optimisation up to the particle scale while staying isotropic and unordered (crystallisation empirically appears to start around $\chi \approx 0.45$).

  Two additional versions, built with an L1 ball and a star ball, serve a different purpose. They are not isotropic, because they are intended for QMC methods with low effective dimensional integrands where the optimal shape of an exclusion set is not a usual L2 ball. The WD (star-ball) version is specifically designed for worst-case QMC error bounds induced by the wrap-around discrepancy kernel and is recommended for highly low-dimensional integrands. The L1 version (rhombus-ball) is the intermediate option between WD and L2, recommended for usual smooth integrands.


* an **important subtlety** about what the exclusion set precisely is, which comes up in high dimensions: see [Exclusionsets.md](Exclusionsets.md). This note is important but technical, and can be read on a second reading.

* **Weaker stealthy criterion:** $S(k)$ is the structure factor. Morse et al. reach $S(k) \le 10^{-51}$ using double-double (`float128`) precision. We use `float32`—which is much faster on GPU—and stop the descent after 150 iterations. For all targeted frequencies:

| $\chi$ | Exclusion set | Criterion |
| :--- | :--- | :--- |
| 0.30 | L2 ball (Euclidean) | $S(k) \le 10^{-10}$ |
| 0.43 | L2 ball (Euclidean) | $S(k) \le 10^{-3}$ |
| 0.43 (L1 ball) | L1 ball | $S(k) \le 10^{-3}$ |
| 0.43 (WD star) | Star ball $r(k) \le \text{cst}$ | $S(k) \le 10^{-2}$ |

  Here, $r(k)$ is the Fourier coefficient of the wrap-around discrepancy.

  We retain the value $S(k) \le 10^{-3}$ as stealthy since it is sufficient for our application and most use cases. It would be difficult to go beyond it at large $\chi$. Regarding the $\chi = 0.30$ samples, going beyond $10^{-10}$ would be possible, but requires reimplementing the whole sampler with at least `float64` precision.

* **L1 ball and $\chi = 0.43$ (`--chi L1-43`):** This version selects wave vectors with the L1 norm $\Vert{}k\Vert{}_1 = \sum \vert{}k_i\vert{}$ instead of the Euclidean norm, pushing $\chi$ up to 0.43 (i.e., $2\chi = 86\%$ of the spectral domain reachable with $N$ points). It is aimed at QMC integration (an L1 ball has a cross-polytope geometry, like a diamond in 2D, suited to integrands with a low effective dimension or a strong coordinate-wise structure), whereas the L2 versions are recommended for blue noise, where strict stealthiness and isotropy matter more than discrepancy. Check the [source code](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/tree/main/src/qmc_sampler) for implementation details.

* **Star ball (WD) and $\chi = 0.43$ (`--chi WD-43`):** This version selects wave vectors according to the exact product weight of the wrap-around discrepancy kernel:
  $$r(k) = \prod_{i=1}^D \hat{K}(k_i),\qquad \hat{K}(m) = \begin{cases} 1 & \text{if } m = 0,\\ \dfrac{1}{2\pi^2 m^2} & \text{if } m \neq 0. \end{cases}$$
  The top-$M$ modes ($M \approx \chi \cdot D \cdot (N-1)$) ordered by decreasing $r(k)$ form the **star ball**. This is the natural exclusion set that appears in the exact worst-case error bound for the star discrepancy. Consequently, the WD-43 samples are particularly well suited to QMC integration of irregular or non-smooth integrands.

* **Gradient descent:** Morse et al. use FIRE optimisation. We use a simple `bold driver` scheme on the normalised gradient (only its direction matters): good steps boost the learning rate, bad steps slow it down. We perform 150 iterations of this auto-scaled gradient step. It works on any $(N, D)$ out of the box, with no hyperparameter tuning.

The samples were collected by running the notebook in `src/generate_all_samples_from_paper.ipynb` on a Google Colab T4 GPU. The optimisation algorithm has $O(N^2 \cdot D^2)$ complexity per single $(N, D)$ point cloud and thus requires intensive computation, making a GPU nearly mandatory for large point clouds. We ran $\chi = 0.30$, $\chi = 0.43$, $\chi = \text{L1-0.43}$, and $\chi = \text{WD-0.43}$, looping over all numbers of points $N = 2^{10}, \dots, 2^{17}$ and over dimensions $D = 2$ to 16, plus $D = 23$ and $D = 32$, which took roughly 2 hours in total per $\chi$ version.

*Note:* Using the `pykeops` LazyTensor framework makes the spectral loss computation significantly faster. PyKeOps is a high-performance library for efficient kernel reduction.

## Large point clouds

We used the `FINUFFT` Python library to sample additional massive point clouds in 2D / 3D. They can be downloaded manually from the Zenodo database. `FINUFFT` essentially allows replacing the $O(N^2)$ complexity of the gradient descent with $O(N \log N)$ using a Fast Fourier Transform (FFT). 

Here is the [source code](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft.py) to run the FINUFFT-based pipeline.