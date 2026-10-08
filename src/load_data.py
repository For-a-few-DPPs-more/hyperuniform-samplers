#!/usr/bin/env python
"""Download the hyperuniform point-cloud database from Zenodo.

Usage
-----
    python load_data.py --dim all          # D = 2..16, 23, 32
    python load_data.py --dim 2            # only D = 2
    python load_data.py --dim 2 3 4        # several dimensions

Each `D{j}.zip` is downloaded then extracted to `<out>/D{j}/`, which gives
files `<out>/D{j}/N{i}_D{j}.h5` (N = 2**i points, shape (N, j), float32).
Already-downloaded dimensions are skipped (use --force to redo them).
"""
import argparse
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Zenodo record holding the files (DOI 10.5281/zenodo.23237510).
BASE_URL = "https://zenodo.org/records/23237510/files"

AVAILABLE_DIMS = list(range(2, 17)) + [23, 32]


def parse_dims(values):
    """['all'] -> all dims ; ['2', '5'] -> [2, 5]."""
    if len(values) == 1 and values[0].lower() == "all":
        return AVAILABLE_DIMS
    dims = []
    for v in values:
        try:
            d = int(v)
        except ValueError:
            sys.exit(f"error: invalid --dim value {v!r} (use an integer or 'all')")
        if d not in AVAILABLE_DIMS:
            sys.exit(f"error: no dataset for D={d}. Available: {AVAILABLE_DIMS}")
        dims.append(d)
    return sorted(set(dims))


def download(url, dest, retries=3):
    """Download `url` to `dest` (atomic: written to .part then renamed)."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, t0 = 0, time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB "
                              f"({100 * done / total:5.1f}%)", end="", flush=True)
                print(f"\r  {done / 1e6:.1f} MB in {time.time() - t0:.0f}s" + " " * 20)
            part.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"\n  attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                part.unlink(missing_ok=True)
                raise


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", nargs="+", required=True,
                   help="dimension(s) to download, or 'all' (2..16, 23, 32)")
    p.add_argument("--out", default="data", help="output directory (default: ./data)")
    p.add_argument("--base-url", default=BASE_URL, help="Zenodo files URL")
    p.add_argument("--force", action="store_true", help="re-download existing dims")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for d in parse_dims(args.dim):
        target = out / f"D{d}"
        if target.is_dir() and any(target.glob("*.h5")) and not args.force:
            print(f"D={d}: already present in {target}, skipping")
            continue
        zpath = out / f"D{d}.zip"
        url = f"{args.base_url}/D{d}.zip?download=1"
        print(f"D={d}: downloading {url}")
        try:
            download(url, zpath)
        except Exception as e:
            sys.exit(f"error: could not download D={d}: {e}")
        with zipfile.ZipFile(zpath) as z:
            # flatten: keep only the basename of each member, whatever the zip layout
            target.mkdir(exist_ok=True)
            for m in z.infolist():
                if m.is_dir() or not m.filename.endswith(".h5"):
                    continue
                (target / Path(m.filename).name).write_bytes(z.read(m))
        zpath.unlink()
        n = len(list(target.glob("*.h5")))
        print(f"D={d}: {n} files extracted to {target}")


if __name__ == "__main__":
    main()
