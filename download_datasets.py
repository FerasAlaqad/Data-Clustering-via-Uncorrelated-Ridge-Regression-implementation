"""
Utility script to fetch the benchmark datasets referenced in
“Data Clustering via Uncorrelated Ridge Regression” (Zhang et al., 2021).

The paper evaluates the algorithms on the following data sources:

    1. AT&T Faces
    2. 1COLON (Colon cancer gene expression)
    3. ETHZ-53 (ETH-80 object images)
    4. FEI Face Database
    5. GLIOMA gene expression
    6. GT database (Georgia Tech face database)
    7. FLOWER17
    8. IMM Face Database

Not all datasets expose direct download links (some require a form or manual
acceptance of terms).  The script below automates those with stable public
URLs and prints instructions for the ones that still require manual steps.

Usage
-----
    python download_datasets.py            # download everything available
    python download_datasets.py att_faces  # download a single dataset

Downloaded archives are stored in ./datasets/ and extracted next to the
archive if automatic extraction is supported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests


# Known direct-download URLs
DATASETS: Dict[str, Dict[str, object]] = {
    "att_faces": {
        "url": "http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip",
        "filename": "att_faces.zip",
        "extract": True,
    },
    "gt_db": {
        "url": "http://www.anefian.com/research/gt_db.zip",
        "filename": "gt_db.zip",
        "extract": True,
    },
    "flower17": {
        "url": "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz",
        "filename": "17flowers.tgz",
        "extract": True,
    },
    "imm": {
        # Several mirrors exist; we'll try each until one works.
        "urls": [
            "https://www2.compute.dtu.dk/pubdb/edoc/imm3219.zip",
            "http://www2.compute.dtu.dk/pubdb/edoc/imm3219.zip",
            "https://www2.imm.dtu.dk/pubdb/edoc/imm3219.zip",
            "http://www2.imm.dtu.dk/pubdb/edoc/imm3219.zip",
        ],
        "filename": "imm_face_db.zip",
        "extract": True,
    },
}

# Datasets that require manual download (form fill, captcha, etc.)
MANUAL_DATASETS: Dict[str, str] = {
    "1colon": "http://penglab.janelia.org/proj/mRMR/index.htm#data",
    "ethz53": "https://www.vision.ee.ethz.ch/en/datasets/",
    "fei": "https://fei.edu.br/~cet/facedatabase.html",
    "glioma": "https://featureselection.asu.edu/datasets.php",
}

CHUNK_SIZE = 1 << 15  # 32 KiB


def ensure_dataset_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, destination: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with destination.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 // total
                sys.stdout.write(f"\r  downloading {destination.name}: {percent:3d}%")
                sys.stdout.flush()
    if total:
        sys.stdout.write("\n")


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    if archive_path.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
    elif archive_path.suffix in {".tgz", ".gz"} or archive_path.suffixes[-2:] == [".tar", ".gz"]:
        import tarfile

        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(target_dir)
    else:
        print(f"  [skip] Unknown archive format for {archive_path.name}; not extracted.")


def download_dataset(name: str, root: Path) -> None:
    if name not in DATASETS:
        print(f"[warn] '{name}' is not available for automated download.")
        hint = MANUAL_DATASETS.get(name)
        if hint:
            print(f"       Please download manually from: {hint}")
        return

    meta = DATASETS[name]
    urls = meta.get("urls")
    if urls:
        candidates = list(urls)  # type: ignore[arg-type]
    else:
        candidates = [meta["url"]]  # type: ignore[assignment]

    filename: str = meta["filename"]  # type: ignore[assignment]
    extract: bool = meta.get("extract", False)  # type: ignore[assignment]

    archive_path = root / filename
    if archive_path.exists():
        print(f"[skip] {filename} already exists.")
    else:
        last_error: Optional[Exception] = None
        for url in candidates:
            try:
                print(f"[info] Fetching {name} from {url}")
                download_file(url, archive_path)
                print(f"[done] Saved to {archive_path}")
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[warn] Failed to download from {url}: {exc}")
        if last_error:
            print(
                f"[error] Unable to fetch '{name}'. "
                "Please download manually if the mirror is unavailable."
            )
            return

    if extract:
        target_dir = root / name
        if target_dir.exists():
            print(f"[skip] Extraction target {target_dir} already exists.")
        else:
            print(f"[info] Extracting {filename} to {target_dir}")
            extract_archive(archive_path, target_dir)
            print(f"[done] Extracted {name}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Subset of dataset keys to download "
        "(default: download all available automatic datasets).",
    )
    parser.add_argument(
        "--root",
        default="datasets",
        help="Directory to store downloaded archives (default: ./datasets)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    root = ensure_dataset_dir(Path(args.root))

    targets = args.datasets or sorted(DATASETS.keys())
    for name in targets:
        download_dataset(name.lower(), root)

    if MANUAL_DATASETS:
        print("\nDatasets requiring manual download:")
        for key, url in MANUAL_DATASETS.items():
            print(f"  - {key}: {url}")


if __name__ == "__main__":
    main()

