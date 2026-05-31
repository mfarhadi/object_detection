#!/usr/bin/env python3
"""Download COCO 2017 dataset in YOLO detection (bbox) format."""

from pathlib import Path

from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download
from ultralytics.data.utils import check_det_dataset

DATASETS_DIR = Path("/data/yolo/datasets")
COCO_DIR = DATASETS_DIR / "coco"


def main() -> None:
    print("Step 1/2: Downloading YOLO bbox labels (coco2017labels.zip)...")
    download([f"{ASSETS_URL}/coco2017labels.zip"], dir=DATASETS_DIR, unzip=True, delete=True)

    print("Step 2/2: Downloading train2017 + val2017 images (~20 GB)...")
    download(
        [
            "http://images.cocodataset.org/zips/train2017.zip",
            "http://images.cocodataset.org/zips/val2017.zip",
        ],
        dir=COCO_DIR / "images",
        unzip=True,
        delete=True,
        threads=2,
    )

    print("\nVerifying dataset...")
    data = check_det_dataset("/data/yolo/coco.yaml", autodownload=False)
    print("\nDownload complete!")
    print(f"  Path:    {data['path']}")
    print(f"  Train:   {data['train']}")
    print(f"  Val:     {data['val']}")
    print(f"  Classes: {data['nc']}")


if __name__ == "__main__":
    main()
