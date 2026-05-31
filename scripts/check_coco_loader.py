#!/usr/bin/env python3
"""Smoke-check the COCO YOLO loader."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from yolo_tests.data import CocoYoloDetection, create_coco_yolo_dataloader, load_coco_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/yolo/datasets/coco"))
    parser.add_argument("--split", default="train2017")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--names-yaml", type=Path, default=Path("/data/yolo/coco.yaml"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--no-letterbox", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--verify-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = load_coco_class_names(args.names_yaml) if args.names_yaml.exists() else {}
    dataset = CocoYoloDetection(
        root=args.root,
        split=args.split,
        split_file=args.split_file,
        image_size=args.image_size,
        letterbox=not args.no_letterbox,
        verify_files=args.verify_files,
    )
    loader = create_coco_yolo_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
    )

    batch = next(iter(loader))
    images = batch["images"]
    targets = batch["targets"]
    yolo_targets = batch["yolo_targets"]

    image_shape = tuple(images.shape) if isinstance(images, torch.Tensor) else [tuple(image.shape) for image in images]
    first_target = targets[0]
    first_labels = first_target["labels"][:5].tolist()
    first_names = [names.get(int(label), str(int(label))) for label in first_labels]

    print(f"dataset_root: {dataset.root}")
    print(f"split: {dataset.split}")
    print(f"samples: {len(dataset)}")
    print(f"image_batch_shape: {image_shape}")
    print(f"flat_yolo_targets_shape: {tuple(yolo_targets.shape)}")
    print(f"first_image: {first_target['path']}")
    print(f"first_original_size_hw: {first_target['orig_size'].tolist()}")
    print(f"first_current_size_hw: {first_target['size'].tolist()}")
    print(f"first_object_count: {int(first_target['labels'].numel())}")
    print(f"first_labels: {first_labels}")
    print(f"first_label_names: {first_names}")
    print(f"first_boxes_yolo: {first_target['boxes_yolo'][:5].tolist()}")


if __name__ == "__main__":
    main()
