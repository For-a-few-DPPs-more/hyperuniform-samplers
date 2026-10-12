While the Zenodo database natively contains the point clouds for D = 2 to 16, \
plus D = 23 and D = 32, other dimensions are not materialised in the database. \
Instead, they are obtained by projecting a materialised point cloud. \
Projecting a D-dimensional point cloud to a lower dimension D_proj degrades its quality: \
the exclusion region shrinks, which results in χ → χ_eff < χ. \
This quality loss is moderate as long as the difference D − D_proj is not too large.

In practice, the tables below show the impact of the projection applied when a point cloud with dimension 17 ≤ D_proj ≤ 22 or 24 ≤ D_proj ≤ 31 is requested, for each of the three dataset families (`30`, `43` and `L1-50`). 

## χ = 0.3 (initial coverage 2·χ = 60%)

Effective coverage = 2·χ_eff (in %):

| D | D_proj | N = 1 000 | N = 10 000 | N = 100 000 | N = 1 000 000 |
|---|---|---|---|---|---|
| 23 | 23 | 60 | 60 | 60 | 60 |
| | 22 | 55 | 52 | 49 | 46 |
| | 21 | 50 | 45 | 40 | 34 |
| | 20 | 45 | 39 | 32 | 25 |
| | 19 | 41 | 33 | 25 | 18 |
| | 18 | 36 | 28 | 20 | 13 |
| | 17 | 32 | 23 | 15 | 9.2 |
| | | | | | |
| 32 | 32 | 60 | 60 | 60 | 60 |
| | 31 | 56 | 55 | 53 | 51 |
| | 30 | 53 | 50 | 47 | 43 |
| | 29 | 49 | 45 | 41 | 36 |
| | 28 | 46 | 40 | 35 | 31 |
| | 27 | 43 | 36 | 31 | 25 |
| | 26 | 40 | 33 | 26 | 21 |
| | 25 | 37 | 29 | 23 | 17 |
| | 24 | 34 | 26 | 19 | 14 |

## χ = 0.43 (initial coverage 2·χ = 86%)

Effective coverage = 2·χ_eff (in %):

| D | D_proj | N = 1 000 | N = 10 000 | N = 100 000 | N = 1 000 000 |
|---|---|---|---|---|---|
| 23 | 23 | 86 | 86 | 86 | 86 |
| | 22 | 78 | 74 | 69 | 65 |
| | 21 | 70 | 63 | 55 | 48 |
| | 20 | 62 | 53 | 44 | 35 |
| | 19 | 55 | 45 | 34 | 25 |
| | 18 | 49 | 37 | 26 | 17 |
| | 17 | 43 | 30 | 20 | 12 |
| | | | | | |
| 32 | 32 | 86 | 86 | 86 | 86 |
| | 31 | 81 | 78 | 76 | 73 |
| | 30 | 76 | 71 | 66 | 61 |
| | 29 | 70 | 64 | 58 | 52 |
| | 28 | 66 | 57 | 50 | 43 |
| | 27 | 61 | 51 | 43 | 35 |
| | 26 | 57 | 46 | 37 | 29 |
| | 25 | 52 | 41 | 31 | 24 |
| | 24 | 48 | 36 | 27 | 19 |

## χ = L1-0.50 (initial coverage 2·χ = 100%, L1 ball)

Effective coverage = 2·χ_eff (in %):

| D | D_proj | N = 1 000 | N = 10 000 | N = 100 000 | N = 1 000 000 |
|---|---|---|---|---|---|
| 23 | 23 | 100 | 100 | 100 | 100 |
| | 22 | 91 | 87 | 84 | 79 |
| | 21 | 82 | 76 | 69 | 62 |
| | 20 | 74 | 66 | 57 | 49 |
| | 19 | 66 | 56 | 46 | 37 |
| | 18 | 59 | 48 | 37 | 28 |
| | 17 | 52 | 41 | 30 | 21 |
| | | | | | |
| 32 | 32 | 100 | 100 | 100 | 100 |
| | 31 | 94 | 91 | 89 | 86 |
| | 30 | 88 | 83 | 79 | 74 |
| | 29 | 83 | 76 | 69 | 63 |
| | 28 | 77 | 68 | 61 | 54 |
| | 27 | 72 | 62 | 53 | 46 |
| | 26 | 67 | 56 | 46 | 38 |
| | 25 | 62 | 50 | 40 | 32 |
| | 24 | 58 | 44 | 34 | 27 |

## How the effective coverage is computed

Projecting a D-dimensional cloud to D_proj dimensions keeps D_proj of its coordinates (chosen at random). The structure factor of the projected cloud at `k' ∈ Z^D_proj` is the structure factor of the original cloud at `k = (k', 0, …, 0)`. Hence only the wave vectors of the exclusion region whose components on the removed coordinates are zero remain controlled.

Let `M = round(χ · D · (N − 1))` be the number of wave vectors (counted up to `k ↦ -k`) of the original exclusion region, and `M_proj` the number of those that survive the projection. Then

```text
χ_eff = M_proj / (D_proj · (N − 1)),        effective coverage = 2 · χ_eff.
```

* **L2 versions (`30`, `43`):** the exclusion region is made of the `M` wave vectors of smallest Euclidean norm. The surviving ones are those lying in the D_proj-dimensional Euclidean ball of the same radius.
* **L1 version (`L1-50`):** the exclusion region is made of all wave vectors with `‖k‖₁ < r`, plus a random subset of the shell `‖k‖₁ = r`, where `r` is the smallest integer such that the ball contains at least `M` wave vectors. The slice of an L1 ball by a coordinate subspace is again an L1 ball, of dimension D_proj and the same radius `r`, so the surviving wave vectors are counted in this smaller ball (the shell `r` being counted with the same random fraction as in dimension D). The number of integer points in an L1 ball of dimension `d` and radius `r` is `Σⱼ 2ʲ · C(d, j) · C(r, j)`.

Values are rounded to the nearest percent; because the last shell is chosen at random, small variations of ±1 are expected between two runs.
