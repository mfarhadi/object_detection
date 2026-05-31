"""COCO image loader for YOLO-format detection labels.

The dataset reads images with OpenCV and uses torchvision utilities for image
tensor conversion and box format conversion. Labels are expected to be one row
per object: ``class_id x_center y_center width height`` with normalized box
coordinates in the range [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.ops import box_convert
from torchvision.transforms import functional as tvF

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    yaml = None


ImageTargetTransform = Callable[[Tensor, dict[str, Any]], tuple[Tensor, dict[str, Any]]]


@dataclass(frozen=True)
class ImageRecord:
    """Resolved image path and matching label path."""

    image_path: Path
    label_path: Path


def _as_hw(size: int | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"image_size must be an int or (height, width), got {size!r}")
    return (int(size[0]), int(size[1]))


def _resolve_image_path(root: Path, entry: str) -> Path:
    path = Path(entry.strip())
    if path.is_absolute():
        return path
    entry_without_dot = entry[2:] if entry.startswith("./") else entry
    return root / entry_without_dot


def _label_path_for_image(root: Path, image_path: Path) -> Path:
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        parts = image_path.parts
        if "images" not in parts:
            raise ValueError(f"Cannot infer label path for image outside root: {image_path}")
        index = parts.index("images")
        return Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")

    parts = list(relative.parts)
    if not parts or parts[0] != "images":
        raise ValueError(f"Expected image path under {root / 'images'}, got {image_path}")
    return (root / "labels" / Path(*parts[1:])).with_suffix(".txt")


def _read_split_file(root: Path, split_file: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for line_number, line in enumerate(split_file.read_text().splitlines(), start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        image_path = _resolve_image_path(root, entry)
        try:
            label_path = _label_path_for_image(root, image_path)
        except ValueError as exc:
            raise ValueError(f"{split_file}:{line_number}: {exc}") from exc
        records.append(ImageRecord(image_path=image_path, label_path=label_path))
    return records


def _discover_images(root: Path, split: str) -> list[ImageRecord]:
    image_dir = root / "images" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image split directory: {image_dir}")
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records = [
        ImageRecord(image_path=path, label_path=_label_path_for_image(root, path))
        for path in sorted(image_dir.iterdir())
        if path.suffix.lower() in suffixes
    ]
    if not records:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return records


def read_yolo_labels(label_path: str | Path) -> tuple[Tensor, Tensor]:
    """Read a YOLO txt label file.

    Returns:
        A tuple ``(labels, boxes_yolo)`` where labels are zero-based class ids
        with shape ``[N]`` and boxes are normalized ``cxcywh`` with shape
        ``[N, 4]``. Missing label files are treated as images with no objects.
    """

    path = Path(label_path)
    if not path.exists():
        return torch.empty((0,), dtype=torch.long), torch.empty((0, 4), dtype=torch.float32)

    labels: list[int] = []
    boxes: list[list[float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{path}:{line_number}: expected 5 YOLO fields "
                f"'class x_center y_center width height', got {len(parts)}"
            )
        labels.append(int(float(parts[0])))
        boxes.append([float(value) for value in parts[1:5]])

    if not labels:
        return torch.empty((0,), dtype=torch.long), torch.empty((0, 4), dtype=torch.float32)

    return torch.tensor(labels, dtype=torch.long), torch.tensor(boxes, dtype=torch.float32)


def yolo_to_xyxy(boxes_yolo: Tensor, image_width: int, image_height: int) -> Tensor:
    """Convert normalized YOLO ``cxcywh`` boxes into pixel ``xyxy`` boxes."""

    if boxes_yolo.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)

    scale = boxes_yolo.new_tensor([image_width, image_height, image_width, image_height])
    boxes_xyxy = box_convert(boxes_yolo * scale, in_fmt="cxcywh", out_fmt="xyxy")
    clamp = boxes_xyxy.new_tensor([image_width, image_height, image_width, image_height])
    return torch.minimum(torch.clamp(boxes_xyxy, min=0), clamp)


def xyxy_to_yolo(boxes_xyxy: Tensor, image_width: int, image_height: int) -> Tensor:
    """Convert pixel ``xyxy`` boxes into normalized YOLO ``cxcywh`` boxes."""

    if boxes_xyxy.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)

    scale = boxes_xyxy.new_tensor([image_width, image_height, image_width, image_height])
    return box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh") / scale


def read_image_cv2(image_path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8 HWC using OpenCV."""

    path = Path(image_path)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"OpenCV could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def letterbox_image_and_boxes(
    image: np.ndarray,
    boxes_xyxy: Tensor,
    output_size: tuple[int, int],
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, Tensor]:
    """Resize with unchanged aspect ratio and pad to ``output_size``."""

    input_height, input_width = image.shape[:2]
    output_height, output_width = output_size
    scale = min(output_width / input_width, output_height / input_height)
    resized_width = int(round(input_width * scale))
    resized_height = int(round(input_height * scale))

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((output_height, output_width, 3), fill, dtype=np.uint8)

    pad_left = (output_width - resized_width) // 2
    pad_top = (output_height - resized_height) // 2
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized

    if boxes_xyxy.numel() == 0:
        return canvas, boxes_xyxy

    offset = boxes_xyxy.new_tensor([pad_left, pad_top, pad_left, pad_top])
    boxes = boxes_xyxy * scale + offset
    clamp = boxes.new_tensor([output_width, output_height, output_width, output_height])
    boxes = torch.minimum(torch.clamp(boxes, min=0), clamp)
    return canvas, boxes


class CocoYoloDetection(Dataset[tuple[Tensor, dict[str, Any]]]):
    """COCO detection dataset backed by YOLO-format label txt files."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train2017",
        split_file: str | Path | None = None,
        image_size: int | tuple[int, int] | list[int] | None = 640,
        letterbox: bool = True,
        transforms: ImageTargetTransform | None = None,
        verify_files: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.image_size = _as_hw(image_size)
        self.letterbox = letterbox
        self.transforms = transforms

        if split_file is None:
            candidate = self.root / f"{split}.txt"
            self.records = _read_split_file(self.root, candidate) if candidate.exists() else _discover_images(self.root, split)
        else:
            split_path = Path(split_file)
            if not split_path.is_absolute():
                split_path = self.root / split_path
            self.records = _read_split_file(self.root, split_path)

        if verify_files:
            self._verify_records()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Any]]:
        record = self.records[index]
        image = read_image_cv2(record.image_path)
        original_height, original_width = image.shape[:2]
        labels, boxes_yolo_original = read_yolo_labels(record.label_path)
        boxes_xyxy = yolo_to_xyxy(boxes_yolo_original, original_width, original_height)

        if self.image_size is not None:
            output_height, output_width = self.image_size
            if self.letterbox:
                image, boxes_xyxy = letterbox_image_and_boxes(image, boxes_xyxy, self.image_size)
                boxes_yolo = xyxy_to_yolo(boxes_xyxy, output_width, output_height)
            else:
                image = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                boxes_xyxy = yolo_to_xyxy(boxes_yolo_original, output_width, output_height)
                boxes_yolo = boxes_yolo_original.clone()
            current_height, current_width = output_height, output_width
        else:
            boxes_yolo = boxes_yolo_original.clone()
            current_height, current_width = original_height, original_width

        image_tensor = tvF.to_tensor(image)
        target: dict[str, Any] = {
            "boxes": boxes_xyxy,
            "boxes_yolo": boxes_yolo,
            "boxes_yolo_original": boxes_yolo_original,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "orig_size": torch.tensor([original_height, original_width], dtype=torch.int64),
            "size": torch.tensor([current_height, current_width], dtype=torch.int64),
            "path": str(record.image_path),
            "label_path": str(record.label_path),
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target

    def _verify_records(self) -> None:
        missing_images = [str(record.image_path) for record in self.records if not record.image_path.exists()]
        if missing_images:
            preview = "\n".join(missing_images[:5])
            raise FileNotFoundError(f"{len(missing_images)} images are missing. First examples:\n{preview}")


def yolo_collate_fn(batch: list[tuple[Tensor, dict[str, Any]]]) -> dict[str, Any]:
    """Collate detection samples and flatten YOLO labels for detector losses."""

    images, targets = zip(*batch, strict=True)
    shapes = {tuple(image.shape) for image in images}
    image_batch: Tensor | list[Tensor]
    image_batch = torch.stack(list(images), dim=0) if len(shapes) == 1 else list(images)

    flat_targets: list[Tensor] = []
    for batch_index, target in enumerate(targets):
        labels = target["labels"]
        boxes = target["boxes_yolo"]
        if labels.numel() == 0:
            continue
        batch_column = boxes.new_full((boxes.shape[0], 1), float(batch_index))
        flat_targets.append(torch.cat([batch_column, labels.to(boxes.dtype).unsqueeze(1), boxes], dim=1))

    if flat_targets:
        yolo_targets = torch.cat(flat_targets, dim=0)
    else:
        yolo_targets = torch.empty((0, 6), dtype=torch.float32)

    return {
        "images": image_batch,
        "targets": list(targets),
        "yolo_targets": yolo_targets,
    }


def _distributed_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def create_coco_yolo_dataloader(
    dataset: CocoYoloDetection,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = False,
    distributed: bool | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create a DataLoader that switches to DistributedSampler under DDP/FSDP."""

    use_distributed = _distributed_is_ready() if distributed is None else distributed
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last) if use_distributed else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        collate_fn=yolo_collate_fn,
    )


def load_coco_class_names(yaml_path: str | Path) -> dict[int, str]:
    """Load the COCO class-name mapping from an Ultralytics-style YAML file."""

    if yaml is None:
        raise RuntimeError("PyYAML is required to load class names from YAML.")

    path = Path(yaml_path)
    data = yaml.safe_load(path.read_text())
    names = data.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}
