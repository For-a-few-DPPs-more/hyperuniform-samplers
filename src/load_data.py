#!/usr/bin/env python
"""Download the hyperuniform point-cloud database.

Usage
-----
    python load_data.py --dim all --chi all
    python load_data.py --dim 2 3 4 --chi 30
    python load_data.py --dim all --chi all --url zenodo

Downloads ZIP archives from GitHub Releases by default, with Zenodo as a
fallback. Each archive is extracted to <out>/chi{X}/D{j}/, giving files
N{i}_D{j}.h5 (N = 2**i points, shape (N, j), float32, dataset "points").

Already-downloaded (chi, D) pairs are skipped unless --force is specified.
"""

print(
    "Downloading may take a few minutes. If anything goes wrong, "
    "you can download the ZIP files manually from: \n"
    "- the github release https://github.com/For-a-few-DPPs-more/hyperuniform-samplers/releases/tag/dataset-v1.0.0 \n"
    "- the Zenodo page https://zenodo.org/records/23244510 \n"
)


import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Dataset mirrors
GITHUB_URL = (
    "https://github.com/For-a-few-DPPs-more/"
    "hyperuniform-samplers/releases/download/dataset-v1.0.0"
)
ZENODO_URL = "https://zenodo.org/records/23244510/files"

AVAILABLE_DIMS = list(range(2, 17)) + [23, 32]
AVAILABLE_CHIS = [30, 43]  # chi = 0.30 and chi = 0.43


def parse_choices(values, available, name):
    """['all'] -> available ; ['2', '5'] -> [2, 5]."""
    if values == ["all"]:
        return available
    chosen = sorted({int(v) for v in values})
    if not set(chosen) <= set(available):
        sys.exit(f"error: invalid {name} {chosen}. Available: {available}")
    return chosen


def download(url, dest, retries=3):
    """Download `url` to `dest` (.part first, then rename)."""
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
        "--dim", nargs="+", required=True,
        help="dimension(s) to download, or 'all' (2..16, 23, 32)",
    )
    p.add_argument(
        "--chi", nargs="+", required=True,
        help="chi value(s): 30, 43, or 'all'",
    )
    p.add_argument(
        "--out", default="data",
        help="output directory (default: ./data)",
    )
    p.add_argument(
        "--url", choices=["github", "zenodo"], default="github",
        help="download source (default: github, recommended; if it fails, try zenodo)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="re-download existing files",
    )
    args = p.parse_args()

    # Print the recommended source and how to switch.
    print("Hyperuniform point-cloud database downloader")
    print("Recommended: download from GitHub Releases.")
    print("If downloading from GitHub fails, try --url zenodo.")
    print(f"Selected source: {args.url}")
    print()

    base_url = GITHUB_URL if args.url == "github" else ZENODO_URL

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
            filename = f"chi{chi}_D{d}.zip"

            if args.url == "github":
                url = f"{base_url}/{filename}"
            else:
                url = f"{base_url}/{filename}?download=1"

            print(f"chi={chi}, D={d}: downloading {url}")

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
                print(f"chi={chi}, D={d}: {count} files in {target}")

            except (
                RuntimeError,
                urllib.error.URLError,
                zipfile.BadZipFile,
                OSError,
            ) as e:
                zpath.unlink(missing_ok=True)
                print(f"ERROR: chi={chi}, D={d}: {e}", file=sys.stderr)
                print(
                    "If using GitHub, retry with --url zenodo.",
                    file=sys.stderr,
                )
                sys.exit(1)


if __name__ == "__main__":
    main()
