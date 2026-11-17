
#!/usr/bin/env python
"""Download the hyperuniform point-cloud database.

Usage
-----
    python load_data.py --dim all --ball all
    python load_data.py --dim 2 3 4 --ball l2
    python load_data.py --dim all --ball wd --url zenodo

Dataset families
----------------
    l2  -> l2D{j}.zip  -> <out>/l2/D{j}/   Euclidean-ball exclusion set (blue noise)
    wd  -> wdD{j}.zip  -> <out>/wd/D{j}/   star-shaped exclusion set built from the
                                           Fourier coefficients of the wrap-around
                                           discrepancy (see scipy.stats.qmc.discrepancy)

Archives are downloaded from GitHub Releases by default, with
Zenodo as an alternative source.

Already-downloaded dimensions are skipped unless --force is used.
"""

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


# Dataset mirrors
GITHUB_URL = (
    "https://github.com/For-a-few-DPPs-more/"
    "hyperuniform-samplers/releases/download/dataset-v5.0.0" #v5 include quasi newton finition
)

ZENODO_URL = "https://zenodo.org/records/23405562/files"

AVAILABLE_DIMS = list(range(2, 17)) + [20, 24, 28, 32]

# Public labels map explicitly to archive prefixes and output folders.
# Never derive archive names directly from the public label.
DATASET_FAMILIES = {
    "l2": {
        "archive_prefix": "l2",
        "output_dir": "l2",
    },
    "wd": {
        "archive_prefix": "wd",
        "output_dir": "wd",
    },
}


def parse_dims(values):
    """Parse dimensions, accepting 'all' or a list of integers."""
    if values == ["all"]:
        return AVAILABLE_DIMS

    try:
        chosen = sorted({int(v) for v in values})
    except ValueError:
        sys.exit(
            f"error: invalid dimension(s) {values}. "
            f"Available: {AVAILABLE_DIMS}"
        )

    if not set(chosen) <= set(AVAILABLE_DIMS):
        sys.exit(
            f"error: invalid dimension(s) {chosen}. "
            f"Available: {AVAILABLE_DIMS}"
        )

    return chosen


def parse_balls(values):
    """Parse dataset family label."""
    if values == ["all"]:
        return list(DATASET_FAMILIES)

    chosen = list(dict.fromkeys(values))
    invalid = [v for v in chosen if v not in DATASET_FAMILIES]

    if invalid:
        sys.exit(
            f"error: invalid dataset family {invalid}. "
            f"Available: {list(DATASET_FAMILIES)} or 'all'"
        )

    return chosen


def download(url, dest, retries=3):
    """Download to a temporary .part file, then rename on success."""
    part = dest.with_name(dest.name + ".part")

    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, part)
            part.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  attempt {attempt}/{retries} failed: {e}")

    part.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {url}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dim",
        nargs="+",
        required=True,
        help="dimension(s), or 'all' (2..16, 20, 24, 28, 32)",
    )
    p.add_argument(
        "--ball",
        nargs="+",
        required=True,
        help="dataset family: l2, wd or all",
    )
    p.add_argument(
        "--out",
        default="data",
        help="output directory (default: ./data)",
    )
    p.add_argument(
        "--url",
        choices=["github", "zenodo"],
        default="github",
        help="download source (default: github)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-download existing files",
    )
    args = p.parse_args()

    print("Hyperuniform point-cloud database downloader")
    print("Recommended source: GitHub Releases.")
    print("Alternative source: Zenodo (--url zenodo).")
    print(f"Selected source: {args.url}")
    print()

    base_url = GITHUB_URL if args.url == "github" else ZENODO_URL
    dims = parse_dims(args.dim)
    balls = parse_balls(args.ball)

    for ball in balls:
        family = DATASET_FAMILIES[ball]
        archive_prefix = family["archive_prefix"]
        output_dir = family["output_dir"]

        for d in dims:
            target = Path(args.out) / output_dir / f"D{d}"

            if any(target.glob("*.h5")) and not args.force:
                print(
                    f"{ball}, D={d}: already present in {target}, skipping"
                )
                continue

            target.mkdir(parents=True, exist_ok=True)
            zpath = target.with_suffix(".zip")

            # Examples: l2D2.zip, wdD2.zip
            filename = f"{archive_prefix}D{d}.zip"

            if args.url == "github":
                url = f"{base_url}/{filename}"
            else:
                url = f"{base_url}/{filename}?download=1"

            print(f"{ball}, D={d}: downloading {url}")

            try:
                download(url, zpath)

                with zipfile.ZipFile(zpath) as z:
                    bad_file = z.testzip()
                    if bad_file is not None:
                        raise zipfile.BadZipFile(
                            f"Corrupted member in archive: {bad_file}"
                        )

                    for member in z.infolist():
                        if member.filename.endswith(".h5"):
                            output = target / Path(member.filename).name
                            output.write_bytes(z.read(member))

                zpath.unlink()
                count = len(list(target.glob("*.h5")))
                print(f"{ball}, D={d}: {count} files in {target}")

            except (
                RuntimeError,
                urllib.error.URLError,
                zipfile.BadZipFile,
                OSError,
            ) as e:
                zpath.unlink(missing_ok=True)
                print(
                    f"ERROR: {ball}, D={d}: {e}",
                    file=sys.stderr,
                )
                print(
                    "If using GitHub, retry with --url zenodo.",
                    file=sys.stderr,
                )
                sys.exit(1)


if __name__ == "__main__":
    main()
