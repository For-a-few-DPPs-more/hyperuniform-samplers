## Exclusion Sets and incomplete orbits:

The sampling process to build the hyperuniform database proceeded as follows:

- Initialize with random i.i.d. points.
- Select a spectral exclusion set \(\Omega\) based on a chosen geometry (`L2` ball or `WD` ball), with a budget of \(M = \lfloor \chi \cdot D \cdot (N-1) \rfloor\) wave vectors (note that we count the pair \((k, -k)\) as a single wave vector due to spectral symmetry), where \(\chi = 0.43\), to compute the Fourier loss:

$$
\text{loss} = \sum_{k \in \Omega} \frac{1}{|k|^2}
\left|\sum_{\text{point} \in \text{points}} e^{2i\pi \langle k, \text{point}\rangle}\right|^2
$$

- This is what a typical exclusion set looks like for 2D wave vectors (the geometrical constructions also generalise to \(D > 2\)) and balls `L2` and `WD`:

[![exclusion_regions](../plots/exclusion_balls.png)](../plots/exclusion_balls.png)

- The set \(\Omega\) is built by accumulating complete orbits (\(weight(k) = cst\), where weight is the squared L2 norm of the wave vector \(k\) for `L2`, or the wrap-around discrepancy coefficient for `WD`) until the budget \(M\) is reached. **The last orbit, which would be incomplete, is truncated (discarded)**, so that \(\Omega\) only contains complete orbits.

The spectral bound \(S(k) \le \varepsilon\) is only guaranteed on **complete orbits**, which is why the incomplete one is discarded. As a consequence, the number of wave vectors actually used is \(M_{\text{eff}} \le M\), which results in an effective

$$
\chi_{\text{eff}} = \frac{M_{\text{eff}}}{D\,(N-1)} \le \chi = 0.43.
$$

This is especially critical in high dimensions, where a single orbit can contain thousands of wave vectors: discarding the last one can leave \(\chi_{\text{eff}}\) at only a small fraction of the nominal \(\chi\).

---

## 1. Orbit Definitions

| Ball | Exclusion Region | Weight \(w(k)\) | Orbit Definition (Modes of Equal Weight) |
| --- | --- | --- | --- |
| **`L2`** | \(L_2\) ball | \(\Vert k\Vert_2^2\) | Shells where \(\Vert k\Vert_2^2 = \text{const}\) |
| **`WD`** | Star ball | \(r(k) = \prod_i \hat{K}(k_i)\) (Fourier coefficient of the wrap-around discrepancy) | Shells where \(r(k) = \text{const}\) |

---

## 2. Worst-Case \(\chi_{\text{eff}} / \chi\) by Dimension

Analysis for \(N = 2^{10}\) to \(2^{17}\), with \(D = 2\) to 16, and \(D = 20, 24, 28, 32\).

For the remaining dimensions (\(17 \le D \le 19\), \(21 \le D \le 23\), \(25 \le D \le 27\), and \(29 \le D \le 31\)), the incomplete-orbit effect combines with an additional projection effect. The exact \(\chi_{\text{eff}} / \chi\) table, including both effects, is provided in [PROJECTIONS.md](PROJECTIONS.md).

The following table reports the **worst-case** \(\chi_{\text{eff}} / \chi\), in percent (higher is better). The worst case is the minimum across \(N = 2^{10}, \ldots, 2^{17}\).

| Dimension (\(D\)) | `L2` | `WD` |
| --- | ---: | ---: |
| **2–3** | 98% | 94% |
| **4–6** | 79% | 96% |
| **7–9** | 67% | 87–96% |
| **10–12** | 35% | 76% |
| **13–16** | 18% | 63% |
| **20** | 14.08% | 49.10% |
| **24** | **10.26%** | 57.38% |
| **28** | **6.37%** | 65.26% |
| **32** | **7.27%** | 38.17% |

---

## 3. Black, Gray and Light Gray Lists

These are the point clouds with the lowest \(\chi_{\text{eff}} / \chi\), and thus the least effective use of the spectral exclusion budget. All of them are `L2` point clouds: `WD` never falls below 38% in the dimensions considered.

**Black: 5–10%** (3 point clouds)

| Ball | D | N |
| :--- | ---: | --- |
| `L2` | 24 | \(2^{13}\) (10.26%) |
| `L2` | 28 | \(2^{10}\) (6.37%) |
| `L2` | 32 | \(2^{10}\) (7.27%) |

**Gray: 10–20%** (9 point clouds)

| Ball | D | N |
| :--- | ---: | --- |
| `L2` | 16 | \(2^{11}\) (17.7%) |
| `L2` | 20 | \(2^{12}\) (14.08%) |
| `L2` | 20 | \(2^{15}\) (15.52%) |
| `L2` | 24 | \(2^{12}\) (20.52%) |
| `L2` | 24 | \(2^{15}\) (13.85%) |
| `L2` | 24 | \(2^{16}\) (13.85%) |
| `L2` | 28 | \(2^{12}\) (14.08%) |
| `L2` | 28 | \(2^{17}\) (11.26%) |
| `L2` | 32 | \(2^{13}\) (18.51%), \(2^{17}\) (17.11%) |

**Light Gray: 20–30%** (7 point clouds)

| Ball | D | N |
| :--- | ---: | --- |
| `L2` | 13 | \(2^{10}\) (23.0%) |
| `L2` | 14 | \(2^{10}\) (26.8%) |
| `L2` | 15 | \(2^{13}\) (24.6%), \(2^{15}\) (29.1%) |
| `L2` | 20 | \(2^{11}\) (28.18%) |
| `L2` | 20 | \(2^{17}\) (25.95%) |
| `L2` | 24 | \(2^{14}\) (27.71%) |
| `L2` | 28 | \(2^{14}\) (28.17%) |

---

## Conclusion

- **Low-dimensional regime (\(D \le 3\)):** \(\chi_{\text{eff}} \approx \chi\) for both balls, with negligible loss.
- **High-dimensional regime (\(D \ge 20\)):** Orbits become very wide, causing \(\chi_{\text{eff}} / \chi\) to drop to as low as 6.37% for the `L2` ball at \(D = 28\).
- **Robustness of the `WD` ball:** Thanks to a much larger number of distinct orbits, truncating its last shell has a smaller impact. For the supplied dimensions \(D = 20, 24, 28, 32\), the worst-case `WD` ratios are 49.10%, 57.38%, 65.26%, and 38.17%, respectively.