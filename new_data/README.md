This `new_data/` folder is empty by default. It is where newly generated point clouds are saved.

## Generating a new point cloud

Once the `data/` folder contains the datasets downloaded from Zenodo, new point clouds can be built from them by randomisation, projection and tiling.

**Supported:** any `(N, D)` with `N` a power of 2, `2^10 <= N <= 2^24`, and `2 <= D <= 32`.

**The `--chi` parameter** (required) selects the dataset the cloud is built from. Two values are available, with different spectral properties:

| `--chi` | χ    | Stealthiness within the spectral exclusion region |
|---------|------|---------------------------------------------------|
| `30`    | 0.30 | `S(k) <= 1e-10` (smaller exclusion region)        |
| `43`    | 0.43 | `S(k) <= 1e-4` (larger exclusion region)          |

The matching samples must be present in `data/`. 

## Example

```bash
python src/new_data.py --dim 2 --npoints 1024 --chi 43 --seed 42
```

This writes `new_data/N1024_D2_chi43_seed42.npy`, an array of shape `(1024, 2)` in float32.

Useful options: `--npoints 2**20` (powers of 2 are accepted), `--data <dir>` (where the downloaded datasets are), `--out <file.npy>` (custom output path).

The -- seed parametter is passed to the random generator for the randomisation procedure, so that e.g. -- seed 42 will always output the same sample, while -- seed 73 will give a different sample.
