This `new_data/` folder is empty by default. It serves as a placeholder for the new generated datasets.

Once the `data/` folder is filled with hyperuniform datasets from zenodo, one can create new ones
using randomisation, projection and tiling. 
Supported: any (N, D) with N a power of 2 and 2^10 <= N <= 2^24, 2 <= D <= 32.


```bash
python src/new_data.py --dim 2 --npoints 1024 --seed 42
```
