This `data/` folder is empty by default. It serves as a placeholder for the sample datasets.

The datasets can be downloaded manually from [Zenodo](https://doi.org/10.5281/zenodo.23244509), or automatically from the root of the repository (open a new terminal and run this):

```bash
# all datasets: D = 2 to 16, plus 23 and 32, with ball = l2 and ball = wd
python src/load_data.py --dim all --ball all    
# a single dimension (faster)
python src/load_data.py --dim 2 --ball all     
# several dimensions, single ball  
python src/load_data.py --dim 2 3 4 --ball l2    
```
