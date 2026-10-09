#!/usr/bin/env python
"""Download the hyperuniform point-cloud database from Zenodo.

Usage
-----
    python load_data.py --dim all --chi all     # D = 2..16, 23, 32 ; chi = 0.30 and 0.43

    python load_data.py --dim 2 3 4 --chi 30    # several dimensions

Each `chi{X}_D{j}.zip` is extracted to `<out>/chi{X}/D{j}/`, which gives files
`<out>/chi{X}/D{j}/N{i}_D{j}.h5` (N = 2**i points, shape (N, j), float32,
dataset "points"). Already-downloaded (chi, D) pairs are skipped (use --force).
"""
import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Zenodo record holding the files (DOI 10.5281/zenodo.23244510).
BASE_URL = "https://zenodo.org/records/23244510/files"

AVAILABLE_DIMS = list(range(2, 17)) + [23, 32]
AVAILABLE_CHIS = [30, 43]          # chi = 0.30 and chi = 0.43


def parse_choices(values, available, name):
    """['all'] -> available ; ['2', '5'] -> [2, 5]."""
    if values == ["all"]:
        return available
    chosen = sorted({int(v) for v in values})
    if not set(chosen) <= set(available):
        sys.exit(f"error: invalid {name} {chosen}. Available: {available}")
    return chosen


def download(url, dest, retries=3):
    """Download `url` to `dest` (written to .part, then renamed)."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, part)
            part.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  attempt {attempt}/{retries} failed: {e}")
    part.unlink(missing_ok=True)
    sys.exit(f"error: could not download {url}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", nargs="+", required=True,
                   help="dimension(s) to download, or 'all' (2..16, 23, 32)")
    p.add_argument("--chi", nargs="+", required=True,
                   help="chi value(s): 30 (chi=0.30), 43 (chi=0.43) or 'all'")
    p.add_argument("--out", default="data", help="output directory (default: ./data)")
    p.add_argument("--base-url", default=BASE_URL, help="Zenodo files URL")
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args()

    dims = parse_choices(args.dim, AVAILABLE_DIMS, "--dim")
    chis = parse_choices(args.chi, AVAILABLE_CHIS, "--chi")

    for chi in chis:
        for d in dims:
            target = Path(args.out) / f"chi{chi}" / f"D{d}"
            if any(target.glob("*.h5")) and not args.force:
                print(f"chi={chi}, D={d}: already present in {target}, skipping")
                continue

            target.mkdir(parents=True, exist_ok=True)
            zpath = target.with_suffix(".zip")
            url = f"{args.base_url}/chi{chi}_D{d}.zip?download=1"
            print(f"chi={chi}, D={d}: downloading {url}")
            download(url, zpath)

            with zipfile.ZipFile(zpath) as z:
                for m in z.infolist():
                    if m.filename.endswith(".h5"):   # flatten, whatever the zip layout
                        (target / Path(m.filename).name).write_bytes(z.read(m))
            zpath.unlink()
            print(f"chi={chi}, D={d}: {len(list(target.glob('*.h5')))} files in {target}")


if __name__ == "__main__":
    main()