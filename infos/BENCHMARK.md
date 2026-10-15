## Benchmark for QMC Methods

Average ranking based on `scipy.stats.qmc.discrepancy` computation.

Run the benchmark with:

```bash
pip install scipy
python src/benchmark_data.py
```

### Global Mean Rank by Method

| Global Rank | Method | Mean CD Rank | Mean WD Rank | Mean MD Rank | Overall Mean Rank |
|---:|---|---:|---:|---:|---:|
| 1 | Sobol (SciPy QMC) | 1.00 | 1.14 | 1.00 | 1.05 |
| 2 | Halton (SciPy QMC) | 2.10 | 2.86 | 2.14 | 2.36 |
| 3 | WDChi43 (ours) | 3.33 | 2.00 | 2.86 | 2.73 |
| 4 | LHS random-cd (SciPy QMC) | 3.57 | 4.00 | 4.00 | 3.86 |
| 5 | L1Chi43 (ours) | 5.14 | 5.00 | 5.05 | 5.06 |
| 6 | Chi30 (ours) | 6.52 | 6.76 | 6.86 | 6.71 |
| 6 | Chi43 (ours) | 6.71 | 6.48 | 6.95 | 6.71 |
| 8 | LHS (SciPy QMC) | 7.62 | 7.76 | 7.14 | 7.51 |
| 9 | poisson i.i.d (baseline) | 9.00 | 9.00 | 9.00 | 9.00 |


**Note:** Ranks are averaged over three sample sizes (\(N \in \{1024, 2048, 4096\}\)), seven dimensions (\(D \in \{2,\ldots,8\}\)), and three discrepancy measures (CD, WD, and MD). Lower ranks indicate better relative performance. Chi30 and Chi43 have identical overall mean ranks.

