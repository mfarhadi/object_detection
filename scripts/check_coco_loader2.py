#!/usr/bin/env python3
"""Smoke-check the COCO YOLO loader and save a plotted sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("coco_yolo_sample.png"),
        help="Path to save the plotted image.",
    )
    return parser.parse_args()


def _to_numpy_image(image: torch.Tensor):
    image = image.detach().cpu()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.float()
    if image.max() > 1.5:
        image = image / 255.0
    return image.clamp(0, 1).numpy()


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

    # Plot and save the first image in the batch.
    if isinstance(images, torch.Tensor):
        img = images[0]
    else:
        img = images[0]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(_to_numpy_image(img))
    ax.set_title(f"COCO sample: {Path(first_target['path']).name}")
    ax.axis("off")

    boxes = first_target.get("boxes_yolo", None)
    labels = first_target.get("labels", None)

    if boxes is not None and labels is not None:
        boxes = boxes.detach().cpu()
        labels = labels.detach().cpu()

        # YOLO boxes are expected as [cx, cy, w, h] normalized to image size.
        h, w = img.shape[-2], img.shape[-1] if img.ndim == 3 else (img.shape[0], img.shape[1])
        if img.ndim == 3 and img.shape[0] in (1, 3):
            height, width = img.shape[1], img.shape[2]
        else:
            height, width = img.shape[0], img.shape[1]

        for box, label in zip(boxes[:20], labels[:20]):
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            rect_w = bw * width
            rect_h = bh * height

            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    rect_w,
                    rect_h,
                    fill=False,
                    linewidth=2,
                )
            )
            class_name = names.get(int(label), str(int(label)))
            ax.text(
                x1,
                max(0, y1 - 3),
                class_name,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"saved_plot: {args.output}")


if __name__ == "__main__":
    main()