from __future__ import annotations

import torch

from yolo_tests.losses import (
    OneAnchorYoloLoss,
    build_multi_positive_targets,
    build_one_anchor_targets,
)


def test_loss_package_exports_current_and_legacy_target_builders() -> None:
    assert OneAnchorYoloLoss is not None
    assert build_multi_positive_targets is not None
    assert build_one_anchor_targets is not None


def test_multi_positive_targets_assign_more_than_one_cell_for_large_gt() -> None:
    yolo_targets = torch.tensor([[0.0, 2.0, 0.5, 0.5, 0.6, 0.6]], dtype=torch.float32)

    targets = build_multi_positive_targets(
        yolo_targets=yolo_targets,
        batch_size=1,
        grid_h=4,
        grid_w=4,
        num_classes=5,
        device=torch.device("cpu"),
    )

    assert int(targets.positive_mask.sum().item()) == 4
    assert targets.class_targets[targets.positive_mask, 2].eq(1.0).all()


def test_legacy_target_builder_uses_multi_positive_assignment() -> None:
    yolo_targets = torch.tensor([[0.0, 1.0, 0.5, 0.5, 0.6, 0.6]], dtype=torch.float32)

    targets = build_one_anchor_targets(
        yolo_targets=yolo_targets,
        batch_size=1,
        grid_h=4,
        grid_w=4,
        num_classes=5,
        device=torch.device("cpu"),
    )

    assert int(targets.positive_mask.sum().item()) == 4
