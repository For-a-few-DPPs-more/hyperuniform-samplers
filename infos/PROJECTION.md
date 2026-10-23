While the Zenodo database natively contains the point clouds for D = 2 to 16, \
plus D = 20, 24, 28 and D = 32, other dimensions are not materialised in the database. \
Instead, they are obtained by projecting a materialised point cloud. \
Projecting a D-dimensional point cloud to a lower dimension D_proj degrades its quality: \
the exclusion region shrinks, which results in χ → χ_eff < χ. \
This quality loss is moderate as long as the difference D − D_proj is not too large.

In practice, the tables below show the impact of the projection applied when a point cloud with dimension 17 ≤ D_proj ≤ 31 is requested, for each of the two dataset families (`l2` and `wd`). The nominal χ parameter (which control the number of targeted wave vectors) is set to 0.43, we display the relative ratio χ_eff / 0.43 which thus, **should be as close to 100% as possible**.
Recall that additionaly to projection, the phenomenom of incomplete orbits (see [ORBITS.md](ORBITS.md)) impact the X_eff coefficient. The following χ_eff / χ ratio table include both effects (incomplete orbits and projection).

