
#!/usr/bin/env python3
"""Rebuild Zenodo-style ZIP files from the extracted HDF5 database."""

import hashlib
import zipfile
from pathlib import Path

DATA_DIR = Path("data")
MIRROR_DIR = Path("mirror_data")

DIMS = list(range(2, 17)) + [23, 32]
CHIS = [30, 43]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    checksums = []
    missing = []

    for chi in CHIS:
        for dim in DIMS:
            folder = DATA_DIR / f"chi{chi}" / f"D{dim}"
            files = sorted(folder.glob(f"N*_D{dim}.h5"))

            if not files:
                missing.append(f"chi{chi}_D{dim}")
                continue

            archive = MIRROR_DIR / f"chi{chi}_D{dim}.zip"

            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_DEFLATED
            ) as z:
                for file in files:
                    z.write(file, arcname=file.name)

            with zipfile.ZipFile(archive, "r") as z:
                assert z.testzip() is None, f"Corrupted archive: {archive}"

            checksums.append(f"{sha256(archive)}  {archive.name}")
            print(f"Created {archive} ({len(files)} HDF5 files)")

    (MIRROR_DIR / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + ("\n" if checksums else ""),
        encoding="utf-8",
    )

    print(f"\nArchives created: {len(checksums)}")
    print(f"Missing datasets: {len(missing)}")

    if missing:
        print("Missing:", ", ".join(missing))

    print(f"Output directory: {MIRROR_DIR.resolve()}")


if __name__ == "__main__":
    main()