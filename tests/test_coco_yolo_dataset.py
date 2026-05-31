from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from yolo_tests.data.coco_yolo import CocoYoloDetection, yolo_collate_fn


def _write_sample(root: Path) -> None:
    image_dir = root / "images" / "train2017"
    label_dir = root / "labels" / "train2017"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[..., 1] = 255
    cv2.imwrite(str(image_dir / "000000000001.jpg"), image)
    (label_dir / "000000000001.txt").write_text("2 0.5 0.5 0.5 0.5\n")
    (root / "train2017.txt").write_text("./images/train2017/000000000001.jpg\n")


def test_loader_reads_yolo_labels_and_converts_to_xyxy(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    dataset = CocoYoloDetection(tmp_path, split="train2017", image_size=None)

    image, target = dataset[0]

    assert tuple(image.shape) == (3, 20, 40)
    assert target["labels"].tolist() == [2]
    assert torch.allclose(target["boxes_yolo"], torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    assert torch.allclose(target["boxes"], torch.tensor([[10.0, 5.0, 30.0, 15.0]]))


def test_letterbox_updates_boxes_for_padded_canvas(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    dataset = CocoYoloDetection(tmp_path, split="train2017", image_size=32, letterbox=True)

    image, target = dataset[0]

    assert tuple(image.shape) == (3, 32, 32)
    assert torch.allclose(target["boxes"], torch.tensor([[8.0, 12.0, 24.0, 20.0]]))
    assert torch.allclose(target["boxes_yolo"], torch.tensor([[0.5, 0.5, 0.5, 0.25]]))


def test_collate_stacks_images_and_flattens_yolo_targets(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    dataset = CocoYoloDetection(tmp_path, split="train2017", image_size=32, letterbox=False)
    batch = yolo_collate_fn([dataset[0], dataset[0]])

    assert tuple(batch["images"].shape) == (2, 3, 32, 32)
    assert torch.allclose(
        batch["yolo_targets"],
        torch.tensor(
            [
                [0.0, 2.0, 0.5, 0.5, 0.5, 0.5],
                [1.0, 2.0, 0.5, 0.5, 0.5, 0.5],
            ]
        ),
    )
