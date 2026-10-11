While the Zenodo database natively contains the point clouds for D = 2 to 16, \
plus D = 23 and D = 32, other dimensions are not materialised in the database. \
Instead, they are obtained by projecting a materialised point cloud. \
Projecting a D-dimensional point cloud to a lower dimension D_proj degrades its quality: \
the exclusion region shrinks, which results in χ → χ_eff < χ. \
This quality loss is moderate as long as the difference D − D_proj is not too large.

In practice, the table below shows the impact of the projection applied when a point cloud with dimension 17 ≤ D_proj ≤ 22 or 24 ≤ D_proj ≤ 31 is requested. \



## χ = 0.3 (initial coverage 2·χ = 30%)

Effective coverage = 2·χ_eff (in %):

| D | D_proj | N = 1 000 | N = 10 000 | N = 100 000 | N = 1 000 000 |
|---|---|---|---|---|---|
| 23 | 17 | 32 | 23 | 15 | 9.2|
| 23 | 18 | 36 | 28 | 20 | 13 |
| 23 | 19 | 41 | 33 | 25 | 18 |
| 23 | 20 | 45 | 39 | 32 | 25 |
| 23 | 21 | 50 | 45 | 40 | 34 |
| 23 | 22 | 55 | 52 | 49 | 46 |
| 23 | 23 | 60 | 60 | 60 | 60 |
|    |    |    |    |    |    |
| 32 | 24 | 34 | 26 | 19 | 14 |
| 32 | 25 | 37 | 29 | 23 | 17 |
| 32 | 26 | 40 | 33 | 26 | 21 |
| 32 | 27 | 43 | 36 | 31 | 25 |
| 32 | 28 | 46 | 40 | 35 | 31 |
| 32 | 29 | 49 | 45 | 41 | 36 |
| 32 | 30 | 53 | 50 | 47 | 43 |
| 32 | 31 | 56 | 55 | 53 | 51 |
| 32 | 32 | 60 | 60 | 60 | 60 |

## χ = 0.43 (initial coverage 2·χ = 86%)

Effective coverage = 2·χ_eff (in %):


| D | D_proj | N = 1 000 | N = 10 000 | N = 100 000 | N = 1 000 000 |
|---|---|---|---|---|---|
| 23 | 17 | 43 | 30 | 20 | 12 |
| 23 | 18 | 49 | 37 | 26 | 17 |
| 23 | 19 | 55 | 45 | 34 | 25 |
| 23 | 20 | 62 | 53 | 44 | 35 |
| 23 | 21 | 70 | 63 | 55 | 48 |
| 23 | 22 | 78 | 74 | 69 | 65 |
| 23 | 23 | 86 | 86 | 86 | 86 |
|    |    |    |    |    |    |
| 32 | 24 | 48 | 36 | 27 | 19 |
| 32 | 25 | 52 | 41 | 31 | 24 |
| 32 | 26 | 57 | 46 | 37 | 29 |
| 32 | 27 | 61 | 51 | 43 | 35 |
| 32 | 28 | 66 | 57 | 50 | 43 |
| 32 | 29 | 70 | 64 | 58 | 52 |
| 32 | 30 | 76 | 71 | 66 | 61 |
| 32 | 31 | 81 | 78 | 76 | 73 |
| 32 | 32 | 86 | 86 | 86 | 86 |
