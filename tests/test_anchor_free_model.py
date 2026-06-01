from __future__ import annotations

import torch

from yolo_tests.models import ResNetAnchorFreeYolo, ResNetOneAnchorYolo


def test_anchor_free_model_export_and_output_contract() -> None:
    assert ResNetOneAnchorYolo is ResNetAnchorFreeYolo

    model = ResNetAnchorFreeYolo(
        num_classes=5,
        backbone_name="resnet18",
        weights="none",
        head_channels=32,
    )
    model.eval()

    with torch.no_grad():
        predictions = model(torch.rand(2, 3, 128, 128))

    assert predictions["class_logits"].shape[:2] == (2, 5)
    assert predictions["box_raw"].shape[1] == 6
    assert predictions["boxes_yolo"].shape[-1] == 4
    assert predictions["boxes_yolo"].amin() >= 0.0
    assert predictions["boxes_yolo"].amax() <= 1.0
