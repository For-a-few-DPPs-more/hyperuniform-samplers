## Exclusion Sets

The sampling process to build the hyperuniform database proceeded as follows:

* Initialize with random i.i.d. points.
* Select a spectral exclusion set $\Omega$ based on a chosen geometry (`L2` ball or `WD` ball), with a budget of $M = \lfloor \chi \cdot D \cdot (N-1) \rfloor$ wave vectors (note that we count the pair (k, -k) as a single wave vector due to spectral symmetry), where $\chi = 0.43$, to compute the Fourier loss:

$$\text{loss} = \sum_{k \in \Omega} 1/\left\vert{}k\right\vert{}^2 \left\vert{} \sum_{\text{point} \in \text{points}} e^{2i\pi \langle k, \text{point}\rangle} \right\vert{}^2$$

* This is what a typical exclusion set looks like, for 2D wave vectors (the geometrical constructions also generalise to $D > 2$) and balls `L2` and `WD`:

[![exclusion_regions](../plots/exclusion_balls.png)](../plots/exclusion_balls.png)

* The set $\Omega$ is built by accumulating complete orbits ($weight(k) = cst$, where weight is the l2 norm of the wave vector k for l2 or the wrap around discrepancy coefficient for wd) until the budget $M$ is reached. **The last orbit, which would be incomplete, is truncated (discarded)**, so that $\Omega$ only contains complete orbits.

The spectral bound $S(k) \le \varepsilon$ is only guaranteed on **complete orbits**, which is why the incomplete one is discarded. As a consequence, the number of wave vectors actually used is $M_{\text{eff}} \le M$, which results in an effective

$$\chi_{\text{eff}} = \frac{M_{\text{eff}}}{D\,(N-1)} \le \chi = 0.43 .$$

This is especially critical in high dimensions, where a single orbit can contain thousands of wave vectors: discarding the last one can leave $\chi_{\text{eff}}$ at only a small fraction of the nominal $\chi$ (7% in the worst case for the `L2` ball).

---

## 1. Orbit definitions

| Ball | Exclusion Region | Weight $w(k)$ | Orbit Definition (Modes of Equal Weight) |
| --- | --- | --- | --- |
| **`L2`** | $L_2$ ball | $\Vert{}k\Vert{}_2^2$ | Shells where $\Vert{}k\Vert{}_2^2 = \text{const}$ |
| **`WD`** | Star ball | $r(k) = \prod \hat{K}(k_i)$ (Fourier coefficient of the wrap-around discrepancy) | Shells where $r(k) = \text{const}$ |

---

## 2. Worst-Case $\chi_{\text{eff}} / \chi$ by Dimension

Analysis across the whole database (272 point clouds, $N = 2^{10}$ to $2^{17}$, $D = 2$ to 16 plus $D = 23$, 32).
Note that the analysis for $17 \le D \le 22$ point clouds, which are obtained by projecting $D = 23$, and for $24 \le D \le 31$, obtained from $D = 32$, is not presented but can be deduced from the unprojected point clouds.

$\chi_{\text{eff}} / \chi$ table, in percent (the higher the better, as $\chi - \chi_{\text{eff}}$ corresponds to the wave vectors lost with the truncated last orbit):

| Dimension ($D$) | `L2` | `WD` |
| --- | --- | --- |
| **2 – 3** | 98% | 94% |
| **4 – 6** | 79% | 96% |
| **7 – 9** | 67% | 87–96% |
| **10 – 12** | 35% | 76% |
| **13 – 16** | 18% | 63% |
| **23** | 19% | 54% |
| **32** | 7% | 38% |

## 3. Black, gray and light gray lists

These are the point clouds with the lowest $\chi_{\text{eff}} / \chi$, and thus not as good as the rest of the database. All of them are `L2` point clouds: `WD` never falls below 38%.

**Black: 5 – 10%** (2 point clouds, both with $D = 32$)

| Ball | D | N |
| :--- | :--- | :--- |
| `L2` | 32 | 2^10 (7.3%), 2^14 (9.3%) |

**Gray: 10 – 20%** (4 point clouds)

| Ball | D | N |
| :--- | :--- | :--- |
| `L2` | 16 | 2^11 (17.7%) |
| `L2` | 23 | 2^12 (18.8%) |
| `L2` | 32 | 2^13 (18.5%), 2^17 (17.1%) |

**Light gray: 20 – 30%** (5 point clouds)

| Ball | D | N |
| :--- | :--- | :--- |
| `L2` | 13 | 2^10 (23.0%) |
| `L2` | 14 | 2^10 (26.8%) |
| `L2` | 15 | 2^13 (24.6%), 2^15 (29.1%) |
| `L2` | 23 | 2^15 (24.2%) |

---

## Conclusion

* **$D \le 3$ Regime:** $\chi_{\text{eff}} \approx \chi$ for both balls (negligible loss).
* **High-Dimension Regime ($D \ge 23$):** Orbits become very wide, causing $\chi_{\text{eff}} / \chi$ to drop to about 7% for the `L2` ball.
* **Robustness of the `WD` ball:** Thanks to a much larger number of distinct orbits, truncating its last shell has a minor impact: **it never drops below 38%**.