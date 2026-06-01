from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_train_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_resnet_yolo.py"
    spec = importlib.util.spec_from_file_location("train_resnet_yolo", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_decode_batch_detections_casts_amp_outputs_for_nms() -> None:
    train = _load_train_module()
    args = SimpleNamespace(
        val_score_threshold=0.1,
        val_pre_nms_topk=1000,
        val_nms_threshold=0.5,
        val_max_detections=100,
    )
    predictions = {
        "class_probs": torch.full((1, 2, 2, 2), 0.05, dtype=torch.float32),
        "boxes_yolo": torch.full((1, 2, 2, 4), 0.25, dtype=torch.bfloat16),
    }
    predictions["class_probs"][0, 1, 0, 0] = 0.9

    detections = train.decode_batch_detections(predictions, args)

    assert len(detections) == 1
    assert detections[0]["boxes"].dtype == torch.float32
    assert detections[0]["scores"].dtype == torch.float32
    assert detections[0]["labels"].tolist() == [1]


def test_filter_detection_for_plot_uses_display_threshold_and_limit() -> None:
    train = _load_train_module()
    args = SimpleNamespace(val_plot_score_threshold=0.5, val_plot_max_boxes=2)
    detection = {
        "boxes": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        "scores": torch.tensor([0.8, 0.2, 0.95, 0.7], dtype=torch.float16),
        "labels": torch.tensor([1, 2, 3, 4]),
    }

    filtered = train.filter_detection_for_plot(detection, args)

    assert filtered["boxes"].dtype == torch.float32
    assert filtered["scores"].dtype == torch.float32
    assert torch.allclose(filtered["scores"], torch.tensor([0.9502, 0.7998]), atol=1e-4)
    assert filtered["labels"].tolist() == [3, 1]


def test_filter_detection_for_plot_honors_zero_limit() -> None:
    train = _load_train_module()
    args = SimpleNamespace(val_plot_score_threshold=0.0, val_plot_max_boxes=0)
    detection = {
        "boxes": torch.ones((2, 4), dtype=torch.float32),
        "scores": torch.ones((2,), dtype=torch.float32),
        "labels": torch.ones((2,), dtype=torch.long),
    }

    filtered = train.filter_detection_for_plot(detection, args)

    assert filtered["boxes"].shape == (0, 4)
    assert filtered["scores"].shape == (0,)
    assert filtered["labels"].shape == (0,)


def test_validation_plot_image_id_cycles_by_epoch() -> None:
    train = _load_train_module()
    loader = SimpleNamespace(dataset=range(5))

    assert train.validation_plot_image_id(loader, 0, SimpleNamespace(val_plot_sample_index=None)) == 0
    assert train.validation_plot_image_id(loader, 7, SimpleNamespace(val_plot_sample_index=None)) == 2
    assert train.validation_plot_image_id(loader, 7, SimpleNamespace(val_plot_sample_index=4)) == 4
