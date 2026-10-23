## Sampling method

Here we describe how the samples were generated, for reproducibility.

The method is **spectral optimisation** (non-uniform Fourier transform), also called collective coordinate descent in the literature,  mostly following [Morse et al. (2023)](https://doi.org/10.1103/PhysRevResearch.5.033190). We select the M most important wave vectors $k$ of an exclusion set $\Omega$ (L2 ball, L1 ball or WD ball), keeping one representative per pair $(k,-k)$. We initialise the $(N, D)$ point cloud with random i.i.d. points, then run a gradient descent on

```math
\text{loss} = \sum_{k \in \Omega} \frac{1}{|k|^2} \left| \sum_{j=1}^{N} e^{2i\pi \langle k, x_j \rangle} \right|^2 .
```

The exclusion set $\Omega$ is chosen with a given shape (l2, l1 or wd) and such that the number of wave vectors M within it is fixed as $M = \mathrm{round}(\chi \cdot D \cdot (N-1))$, where $\chi$ is a hyperparameter of the sampler. One needs $\chi \le 0.5$, because each wave vector of the exclusion set gives two real constraints (real and imaginary parts). The goal is to obtain slightly fewer independent equations on the spectrum than there are free parameters in the point cloud.

The main differences between the implemented version and the method of Morse et al. are:

* **$\chi$ parameter.** They recommend $\chi <= 0.4$, but we pushed a bit further, to $\chi = 0.43$.

* **Exclusion set shape.** Beyond the classical ball-shaped exclusion sets, well suited to blue-noise applications, we also explore other geometries tailored to QMC integration of functions with low effective dimension, for which the Euclidean ball is not the natural choice of exclusion region.
  * **L1 ball, $\chi = 0.43$** (`--chi L1-43`): modes with the smallest $\|k\|_1 = \sum_i |k_i|$
    (cross-polytope geometry). Intermediate between WD and L2, recommended for usual smooth integrands.
  * **Star ball (WD), $\chi = 0.43$** (`--chi WD-43`): modes with the largest spectral weights of the
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
  All optimisations start from random i.i.d. point clouds, except the WD version, which starts from an L1 warm start (spectral exclusion set is an L1 ball) so as to converge to a better minimum by inheriting part of the spectral properties of the L1 sample.

* **Exclusion sets in high dimension.** An important subtlety lead to an incomplete orbits phenomenom, see [ORBITS.md](ORBITS.md)
  (technical, can be read on a second reading), which arise from the fact that we sample point up to rather "high" dimension (up to dimension 32).

* **Weaker stealthy criterion.** Morse et al. reach $S(k) \le 10^{-51}$ with `float128`. We use `float32`,
  which makes it impossible to go beyond $S(k) \le 10^{-12}$ due to numerical precision. In practice, we stop after 150 iterations of gradient descent followed by 150 iterations of quasi-Newton descent, which yields $S(k) \le 10^{-10}$.

* **Gradient descent.** Instead of the FIRE method recommended by Morse et al., we use a `bold driver` scheme on the normalised gradient: good steps boost the learning rate, bad steps slow it down. This method has the strong advantage of working out of the box (no hyperparameter tuning) on the full $(N, D)$ range that we considered.
  The bold driver alone would be sufficient to converge for low $\chi$ ($\chi \le 0.3$, or even a bit more). However, our goal was to push $\chi$ as high as possible ($\chi = 0.43$ in the samples of the database), which makes the optimisation significantly harder. Gradient descent is then not enough, so we complete it with 150 additional quasi-Newton iterations.

The algorithm cost is $O(N^2 D^2)$ per sampled point cloud, so a GPU is nearly mandatory. We ran the notebook `src/generate_all_samples_from_paper.ipynb`, which internally relies on the [qmc-sampler](https://pypi.org/project/qmc-sampler/) python package,
on a Google Colab T4 GPU for all four ball versions, with $N = 2^{10}, \dots, 2^{17}$ and $D = 2$ to $16$, plus $D = 20$, $24$, $28$ and $32$
(about 2 hours for $D = 2$ to $16$, plus 2 extra hours for $D = 20, 24, 28, 32$, per ball version).

*Note:* The PyKeOps `LazyTensor` framework makes the spectral loss computation significantly faster. PyKeOps is a high-performance library for efficient kernel reductions.

## Large point clouds

In 2d / 3d, where the `FINUFFT` transform is available (it replaces the $O(N^2)$ gradient evaluation by
$O(N \log N)$ via a fast Fourier transform), we additionally sampled massive point clouds (up to 134 million points in 2d and 32 million in 3d). They can be downloaded manually from the Zenodo database, along with 4D point clouds of up to 1 million points. The finufft based 2D/3D point clouds were generated using following [python file](https://github.com/For-a-few-DPPs-more/Recursive-Gaussian-Blue-Noise/blob/main/src/blue_sampler/run/run_nufft.py).