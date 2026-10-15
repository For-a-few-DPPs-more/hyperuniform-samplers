## Exclusion Sets

The sampling process to build the hyperuniform database proceeded as follows:

* Initialize with random i.i.d. points.
* Select a spectral exclusion set $\Omega$ containing exactly $M = \lfloor \chi \cdot D \cdot (N-1) \rfloor$ wave vectors based on a chosen geometry ($L_2$ ball, $L_1$ ball, or WD weights) to compute the Fourier loss:

$$\text{loss} = \sum_{k \in \Omega} \text{weight}(k) \left\vert{} \sum_{\text{point} \in \text{points}} e^{2i\pi \langle k, \text{point}\rangle} \right\vert{}^2$$


* The set $\Omega$ is built by accumulating complete orbitals ($\Vert{}k\Vert{} = \text{const}$) until reaching budget $M$, which means **the last orbital is typically truncated**.

Because the spectral bound $S(k) \le \varepsilon$ is only guaranteed on **complete orbits**, the nominal $\chi$ must be replaced by an effective **$\chi_{\text{eff}} \le \chi$**. This is especially critical in high dimensions, where a single orbit can represent thousands of wave vectors, and the fact that the last orbit is incomplete result in effective **$\chi_{\text{eff}} \le \chi$** beying only a fraction (5 percent in the worst case) of the nominal $\chi$.

---

## 1. Orbit definitions

| Chi version | Exclusion Region | Weight $w(k)$ | Orbit Definition (Modes of Equal Weight) |
| --- | --- | --- | --- |
| **`30`, `43**` | $L_2$ ball | Decreasing in $\Vert{}k\Vert{}_2^2$ | Shells where $\Vert{}k\Vert{}_2^2 = \text{const}$ |
| **`L1-43`** | $L_1$ ball | Decreasing in $\Vert{}k\Vert{}_1$ | Shells where $\Vert{}k\Vert{}_1 = \text{const}$ |
| **`WD-43`** | Star ball | $r(k) = \prod \hat{K}(k_i)$ | Orbits by permutations and sign changes |

---

## 2. Worst-Case $\chi_{\text{eff}} / \chi$ by Dimension

Analysis across the whole database (544 point clouds, $N = 2^{10}$ to $2^{17}$, D = 2 to 16 plus D = 23, 32).

| Dimension ($D$) | `30` ($L_2$) | `43` ($L_2$) | `L1-43` ($L_1$) | `WD-43` (Star) |
| --- | --- | --- | --- | --- |
| **2 – 3** | 97–100% | 98–100% | 93–97% | 94% |
| **4 – 6** | 78–93% | 79–97% | 63–67% | 96–98% |
| **7 – 9** | 52–63% | 67–77% | 35–46% | 87–96% |
| **10 – 12** | 23–37% | 35–52% | 20–24% | 76–87% |
| **13 – 16** | 19–33% | 18–27% | 17–25% | 63–85% |
| **23** | 7% | 19% | 11% | 54% |
| **32** | 5% | 7% | 8% | 38% |

### 3. Black, gray and light gray lists

these are the points cloud with lowest $\chi_{\text{eff}} \le \chi$

**Black: 5 – 10%** (5 point clouds, all with D ≥ 23)

| Version | D | N |
| :--- | :--- | :--- |
| `30` | 23 | 2^10 (7.5%) |
| `30` | 32 | 2^11 (5.2%) |
| `43` | 32 | 2^10 (7.3%), 2^14 (9.3%) |
| `L1-43` | 32 | 2^10 (7.5%) |

**Gray: 10 – 20%** (17 point clouds)

| Version | D | N |
| :--- | :--- | :--- |
| `30` | 14 | 2^11 (19.2%) |
| `30` | 23 | 2^13 (13.5%), 2^16 (17.4%) |
| `30` | 32 | 2^10 (10.4%), 2^14 (13.3%) |
| `43` | 16 | 2^11 (17.7%) |
| `43` | 23 | 2^12 (18.8%) |
| `43` | 32 | 2^13 (18.5%), 2^17 (17.1%) |
| `L1-43` | 12 | 2^13 (19.7%) |
| `L1-43` | 14 | 2^11 (16.6%) |
| `L1-43` | 15 | 2^11 (18.9%), 2^14 (18.5%) |
| `L1-43` | 16 | 2^17 (18.6%) |
| `L1-43` | 23 | 2^13 (10.7%), 2^16 (15.8%) |
| `L1-43` | 32 | 2^14 (10.2%) |

**Light gray: 20 – 30%** (29 point clouds)

| Version | D | N |
| :--- | :--- | :--- |
| `30` | 11 | 2^10 (23.1%) |
| `30` | 12 | 2^10 (27.8%) |
| `30` | 14 | 2^13 (28.1%) |
| `30` | 15 | 2^11 (22.2%) |
| `30` | 16 | 2^11 (25.4%), 2^14 (21.7%), 2^16 (27.8%) |
| `30` | 23 | 2^12 (26.9%) |
| `30` | 32 | 2^13 (26.5%), 2^17 (24.5%) |
| `43` | 13 | 2^10 (23.0%) |
| `43` | 14 | 2^10 (26.8%) |
| `43` | 15 | 2^13 (24.6%), 2^15 (29.1%) |
| `43` | 23 | 2^15 (24.2%) |
| `L1-43` | 10 | 2^12 (23.7%) |
| `L1-43` | 11 | 2^10 (21.1%) |
| `L1-43` | 12 | 2^10 (24.9%), 2^15 (25.2%), 2^17 (27.3%) |
| `L1-43` | 13 | 2^10 (28.9%), 2^13 (24.6%) |
| `L1-43` | 14 | 2^16 (22.4%) |
| `L1-43` | 15 | 2^16 (29.1%) |
| `L1-43` | 16 | 2^11 (21.4%), 2^14 (22.2%) |
| `L1-43` | 23 | 2^12 (21.4%) |
| `L1-43` | 32 | 2^13 (20.3%), 2^17 (20.7%) |

---

## Conclusion

* **$D \le 3$ Regime:** $\chi_{\text{eff}} \approx \chi$ for all versions (negligible loss).
* **High-Dimension Regime ($D \ge 23$):** Orbits become very wide, causing $\chi_{\text{eff}} / \chi$ to drop down to 5% for $L_1$ and $L_2$ balls.
* **Robustness of the Star Version (`WD-43`):** Thanks to a much larger number of distinct orbits, its last incomplete shell carries a minor impact: **it never drops below 38%**.