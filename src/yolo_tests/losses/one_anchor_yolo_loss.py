"""Clear educational loss for the one-anchor YOLO-style model.

This file favors readability over speed. It shows the key steps that a YOLO loss
needs:

1. Assign each ground-truth object to one feature-grid cell.
2. Build class targets for every grid point.
3. Build box targets only for grid points that own an object.
4. Compute separately normalized positive-class, background-class, and box
   losses.

There is no objectness loss here because the requested model head predicts only
80 class channels plus x/y/w/h. Background grid points are represented by all
class targets being zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class OneAnchorTargets:
    """Dense training targets after assigning objects to grid cells."""

    class_targets: Tensor  # [B, H, W, C], one-hot at positive cells, zero elsewhere
    box_targets: Tensor  # [B, H, W, 4], normalized YOLO cxcywh at positive cells
    positive_mask: Tensor  # [B, H, W], True where one object is assigned


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert normalized cxcywh boxes to clipped xyxy boxes."""

    centers = boxes[..., 0:2]
    sizes = boxes[..., 2:4].clamp_min(1e-6)
    top_left = centers - sizes / 2
    bottom_right = centers + sizes / 2
    return torch.cat([top_left, bottom_right], dim=-1).clamp(0.0, 1.0)


def aligned_box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """IoU for aligned xyxy box pairs with shape [N, 4]."""

    top_left = torch.maximum(boxes_a[:, 0:2], boxes_b[:, 0:2])
    bottom_right = torch.minimum(boxes_a[:, 2:4], boxes_b[:, 2:4])
    intersection_wh = (bottom_right - top_left).clamp_min(0.0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

    area_a_wh = (boxes_a[:, 2:4] - boxes_a[:, 0:2]).clamp_min(0.0)
    area_b_wh = (boxes_b[:, 2:4] - boxes_b[:, 0:2]).clamp_min(0.0)
    area_a = area_a_wh[:, 0] * area_a_wh[:, 1]
    area_b = area_b_wh[:, 0] * area_b_wh[:, 1]
    union = (area_a + area_b - intersection).clamp_min(1e-6)
    return intersection / union


def build_one_anchor_targets(
    yolo_targets: Tensor,
    batch_size: int,
    grid_h: int,
    grid_w: int,
    num_classes: int,
    device: torch.device,
) -> OneAnchorTargets:
    """Assign flat YOLO targets to a single grid cell each.

    Args:
        yolo_targets: Flat tensor from the dataloader with rows
            [batch_index, class_id, cx, cy, width, height]. All box values are
            normalized to the current training image.
        batch_size: Number of images in the local batch.
        grid_h: Feature-grid height from the model output.
        grid_w: Feature-grid width from the model output.
        num_classes: Number of class channels.
        device: Device where target tensors should live.

    If two objects land in the same cell, this educational implementation keeps
    the larger one. A full YOLO implementation would use multiple anchors, better
    assignment rules, or a matching strategy.
    """

    class_targets = torch.zeros((batch_size, grid_h, grid_w, num_classes), device=device)
    box_targets = torch.zeros((batch_size, grid_h, grid_w, 4), device=device)
    positive_mask = torch.zeros((batch_size, grid_h, grid_w), dtype=torch.bool, device=device)
    assigned_area = torch.full((batch_size, grid_h, grid_w), -1.0, device=device)

    if yolo_targets.numel() == 0:
        return OneAnchorTargets(class_targets, box_targets, positive_mask)

    yolo_targets = yolo_targets.to(device=device, dtype=torch.float32)
    for target in yolo_targets:
        batch_index = int(target[0].item())
        class_id = int(target[1].item())
        cx, cy, width, height = target[2:6].tolist()

        if not (0 <= batch_index < batch_size and 0 <= class_id < num_classes):
            continue
        if width <= 0 or height <= 0:
            continue

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)

        grid_x = min(int(cx * grid_w), grid_w - 1)
        grid_y = min(int(cy * grid_h), grid_h - 1)
        area = width * height

        if area < float(assigned_area[batch_index, grid_y, grid_x].item()):
            continue

        class_targets[batch_index, grid_y, grid_x].zero_()
        class_targets[batch_index, grid_y, grid_x, class_id] = 1.0
        box_targets[batch_index, grid_y, grid_x] = torch.tensor(
            [cx, cy, width, height], device=device, dtype=torch.float32
        )
        positive_mask[batch_index, grid_y, grid_x] = True
        assigned_area[batch_index, grid_y, grid_x] = area

    return OneAnchorTargets(class_targets, box_targets, positive_mask)


class OneAnchorYoloLoss(nn.Module):
    """Loss for a one-anchor, class-plus-box YOLO head.

    The model should return:
        predictions['class_logits']: [B, C, H, W]
        predictions['boxes_yolo']: [B, H, W, 4]

    The dataloader should return:
        yolo_targets: [N, 6] with rows [batch, class, cx, cy, w, h]
    """

    def __init__(
        self,
        num_classes: int = 80,
        class_weight: float = 1.0,
        box_weight: float = 5.0,
        negative_class_weight: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.box_weight = box_weight
        self.negative_class_weight = negative_class_weight

    def forward(self, predictions: dict[str, Tensor], yolo_targets: Tensor) -> dict[str, Tensor]:
        class_logits = predictions["class_logits"].permute(0, 2, 3, 1).contiguous()
        pred_boxes = predictions["boxes_yolo"]
        batch_size, grid_h, grid_w, _ = class_logits.shape

        targets = build_one_anchor_targets(
            yolo_targets=yolo_targets,
            batch_size=batch_size,
            grid_h=grid_h,
            grid_w=grid_w,
            num_classes=self.num_classes,
            device=class_logits.device,
        )

        class_loss_raw = F.binary_cross_entropy_with_logits(
            class_logits,
            targets.class_targets,
            reduction="none",
        )
        positive_count = targets.positive_mask.sum()
        negative_mask = ~targets.positive_mask

        if positive_count > 0:
            positive_logits = class_logits[targets.positive_mask]
            positive_targets = targets.class_targets[targets.positive_mask]

            target_class_logits = positive_logits[positive_targets.bool()]
            target_class_loss = F.binary_cross_entropy_with_logits(
                target_class_logits,
                torch.ones_like(target_class_logits),
                reduction="mean",
            )
            other_class_logits = positive_logits[~positive_targets.bool()]
            other_class_loss = F.binary_cross_entropy_with_logits(
                other_class_logits,
                torch.zeros_like(other_class_logits),
                reduction="mean",
            )
            positive_class_loss = target_class_loss + 0.25 * other_class_loss
            target_class_prob = torch.sigmoid(target_class_logits).mean()
            best_class_prob = torch.sigmoid(positive_logits).max(dim=-1).values.mean()
        else:
            positive_class_loss = class_logits.sum() * 0.0
            target_class_loss = class_logits.sum() * 0.0
            other_class_loss = class_logits.sum() * 0.0
            target_class_prob = class_logits.sum() * 0.0
            best_class_prob = class_logits.sum() * 0.0

        if negative_mask.any():
            negative_class_loss = class_loss_raw[negative_mask].mean()
        else:
            negative_class_loss = class_logits.sum() * 0.0

        class_loss = positive_class_loss + self.negative_class_weight * negative_class_loss

        if positive_count > 0:
            positive_pred_boxes = pred_boxes[targets.positive_mask]
            positive_target_boxes = targets.box_targets[targets.positive_mask]
            pred_xyxy = cxcywh_to_xyxy(positive_pred_boxes)
            target_xyxy = cxcywh_to_xyxy(positive_target_boxes)
            iou = aligned_box_iou(pred_xyxy, target_xyxy)
            iou_loss = 1.0 - iou.mean()
            l1_box_loss = F.l1_loss(positive_pred_boxes, positive_target_boxes, reduction="mean")
            box_loss = iou_loss + 0.5 * l1_box_loss
            mean_iou = iou.mean()
        else:
            box_loss = pred_boxes.sum() * 0.0
            iou_loss = pred_boxes.sum() * 0.0
            l1_box_loss = pred_boxes.sum() * 0.0
            mean_iou = pred_boxes.sum() * 0.0

        total_loss = self.class_weight * class_loss + self.box_weight * box_loss
        return {
            "loss": total_loss,
            "class_loss": class_loss.detach(),
            "positive_class_loss": positive_class_loss.detach(),
            "negative_class_loss": negative_class_loss.detach(),
            "target_class_loss": target_class_loss.detach(),
            "other_class_loss": other_class_loss.detach(),
            "target_class_prob": target_class_prob.detach(),
            "best_class_prob": best_class_prob.detach(),
            "box_loss": box_loss.detach(),
            "iou_loss": iou_loss.detach(),
            "l1_box_loss": l1_box_loss.detach(),
            "mean_iou": mean_iou.detach(),
            "num_positive": positive_count.detach().to(torch.float32),
        }
