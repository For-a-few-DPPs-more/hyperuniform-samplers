This `data/` folder is empty by default. It serves as a placeholder for the sample datasets.

The datasets can be downloaded manually from [Zenodo](https://doi.org/10.5281/zenodo.23244510), or automatically from the root of the repository:

```bash
python src/load_data.py --dim all       # all datasets: D = 2 to 16, plus 23 and 32
python src/load_data.py --dim 2         # a single dimension (faster)
python src/load_data.py --dim 2 3 4     # several dimensions
```
