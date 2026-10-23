This `new_data/` folder is empty by default. It is where newly generated point clouds are saved.

## Generating a new point cloud

Once the `data/` folder contains the datasets downloaded from Zenodo, new point clouds can be built from them by randomisation, projection and tiling.

**Supported:** any `(N, D)` with `N` a power of 2, `2^10 <= N <= 2^24`, and `2 <= D <= 32`.

The matching samples must be present in `data/`. 

## Example

```bash
python src/new_data.py --dim 2 --npoints 1024 --ball l2 --seed 42
```

This writes `new_data/N1024_D2_l2_seed42.npy`, an array of shape `(1024, 2)` in float32.

Useful options: `--npoints 2**20` (powers of 2 are accepted), `--data <dir>` (where the downloaded datasets are), `--out <file.npy>` (custom output path).

The -- seed parametter is passed to the random generator for the randomisation procedure, so that e.g. -- seed 42 will always output the same sample, while -- seed 73 will give a different sample.
