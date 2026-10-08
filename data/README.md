This `data/` folder is empty by default. It serves as a placeholder for the sample datasets.

The datasets can be downloaded manually from [Zenodo](https://doi.org/10.5281/zenodo.23244510), or automatically from the root of the repository:

```bash
# all datasets: D = 2 to 16, plus 23 and 32, with chi = 0.30 and chi = 0.43
python src/load_data.py --dim all --chi all    
# a single dimension (faster)
python src/load_data.py --dim 2 --chi all     
# several dimensions, single chi  
python src/load_data.py --dim 2 3 4 --chi 43    
```
