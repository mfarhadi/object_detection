"""Clear educational loss for the anchor-free YOLO-style model.

This file favors readability over speed. It shows the key steps that a YOLO loss
needs:

1. Assign each ground-truth object to one or more feature-grid cells.
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
class AnchorFreeTargets:
    """Dense training targets after assigning objects to grid cells."""

    class_targets: Tensor  # [B, H, W, C], one-hot at positive cells, zero elsewhere
    box_targets: Tensor    # [B, H, W, 4], normalized YOLO cxcywh at positive cells
    positive_mask: Tensor  # [B, H, W], True where at least one object is assigned


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


def build_multi_positive_targets(
    yolo_targets: Tensor,
    batch_size: int,
    grid_h: int,
    grid_w: int,
    num_classes: int,
    device: torch.device,
) -> AnchorFreeTargets:
    """
    Multi-positive assignment.

    For each GT box, every grid cell whose center lies inside the GT box becomes positive.
    If multiple GT boxes try to claim the same cell, the one whose center is closest to that
    cell wins.

    yolo_targets rows:
        [batch_index, class_id, cx, cy, width, height]
    """
    class_targets = torch.zeros((batch_size, grid_h, grid_w, num_classes), device=device)
    box_targets = torch.zeros((batch_size, grid_h, grid_w, 4), device=device)
    positive_mask = torch.zeros((batch_size, grid_h, grid_w), dtype=torch.bool, device=device)

    # Smaller score is better.
    # We use normalized squared distance from the cell center to the GT center.
    best_score = torch.full((batch_size, grid_h, grid_w), float("inf"), device=device)

    if yolo_targets.numel() == 0:
        return AnchorFreeTargets(class_targets, box_targets, positive_mask)

    yolo_targets = yolo_targets.to(device=device, dtype=torch.float32)

    # Grid cell centers in normalized coordinates.
    cell_x = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) / grid_w
    cell_y = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) / grid_h
    cell_x_grid = cell_x[None, :].expand(grid_h, grid_w)
    cell_y_grid = cell_y[:, None].expand(grid_h, grid_w)

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

        # GT corners.
        x1 = max(cx - width / 2, 0.0)
        y1 = max(cy - height / 2, 0.0)
        x2 = min(cx + width / 2, 1.0)
        y2 = min(cy + height / 2, 1.0)

        # Cells whose centers fall inside the box.
        inside_x = (cell_x_grid >= x1) & (cell_x_grid <= x2)
        inside_y = (cell_y_grid >= y1) & (cell_y_grid <= y2)
        mask = inside_x & inside_y

        # Fallback: if the box is too tiny to cover any cell center, assign the center cell.
        if not mask.any():
            grid_x = min(int(cx * grid_w), grid_w - 1)
            grid_y = min(int(cy * grid_h), grid_h - 1)
            mask = torch.zeros((grid_h, grid_w), dtype=torch.bool, device=device)
            mask[grid_y, grid_x] = True

        # Conflict score: closer GT center wins for that cell.
        # Normalized by box size so large boxes do not dominate too easily.
        score = ((cell_x_grid - cx) / max(width, 1e-6)) ** 2 + ((cell_y_grid - cy) / max(height, 1e-6)) ** 2

        replace = mask & (score < best_score[batch_index])
        if not replace.any():
            continue

        ys, xs = replace.nonzero(as_tuple=True)

        class_targets[batch_index, ys, xs] = 0.0
        class_targets[batch_index, ys, xs, class_id] = 1.0
        box_targets[batch_index, ys, xs] = torch.tensor(
            [cx, cy, width, height], device=device, dtype=torch.float32
        )
        positive_mask[batch_index, ys, xs] = True
        best_score[batch_index, ys, xs] = score[ys, xs]

    return AnchorFreeTargets(class_targets, box_targets, positive_mask)


def build_anchor_free_targets(
    yolo_targets: Tensor,
    batch_size: int,
    grid_h: int,
    grid_w: int,
    num_classes: int,
    device: torch.device,
) -> AnchorFreeTargets:
    """Canonical name for the multi-positive anchor-free assignment."""

    return build_multi_positive_targets(
        yolo_targets=yolo_targets,
        batch_size=batch_size,
        grid_h=grid_h,
        grid_w=grid_w,
        num_classes=num_classes,
        device=device,
    )


def build_one_anchor_targets(
    yolo_targets: Tensor,
    batch_size: int,
    grid_h: int,
    grid_w: int,
    num_classes: int,
    device: torch.device,
) -> AnchorFreeTargets:
    """Backward-compatible name for the current multi-positive assignment."""

    return build_anchor_free_targets(
        yolo_targets=yolo_targets,
        batch_size=batch_size,
        grid_h=grid_h,
        grid_w=grid_w,
        num_classes=num_classes,
        device=device,
    )


class AnchorFreeYoloLoss(nn.Module):
    """Loss for an anchor-free, class-plus-box YOLO head.

    The model should return:
        predictions["class_logits"]: [B, C, H, W]
        predictions["boxes_yolo"]: [B, H, W, 4]

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

        targets = build_multi_positive_targets(
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

        positive_mask = targets.positive_mask
        negative_mask = ~positive_mask
        positive_count = positive_mask.sum()

        # Positive cells: supervise the target class.
        if positive_mask.any():
            positive_logits = class_logits[positive_mask]          # [P, C]
            positive_targets = targets.class_targets[positive_mask] # [P, C]
            target_class_ids = positive_targets.argmax(dim=-1)     # [P]

            target_class_loss = F.cross_entropy(
                positive_logits,
                target_class_ids,
                reduction="mean",
            )

            # Small auxiliary term: encourage the other classes to stay low at positives.
            other_class_loss = F.binary_cross_entropy_with_logits(
                positive_logits,
                positive_targets,
                reduction="mean",
            )

            positive_class_loss = target_class_loss + 0.1 * other_class_loss
            target_class_prob = torch.sigmoid(
                positive_logits.gather(1, target_class_ids[:, None])
            ).mean()
            best_class_prob = torch.sigmoid(positive_logits).max(dim=-1).values.mean()
        else:
            positive_class_loss = class_logits.sum() * 0.0
            target_class_loss = class_logits.sum() * 0.0
            other_class_loss = class_logits.sum() * 0.0
            target_class_prob = class_logits.sum() * 0.0
            best_class_prob = class_logits.sum() * 0.0

        # Negative cells: background suppression.
        if negative_mask.any():
            negative_class_loss = class_loss_raw[negative_mask].mean()
        else:
            negative_class_loss = class_logits.sum() * 0.0

        class_loss = positive_class_loss + self.negative_class_weight * negative_class_loss

        # Box regression on all positive cells.
        if positive_mask.any():
            positive_pred_boxes = pred_boxes[positive_mask]
            positive_target_boxes = targets.box_targets[positive_mask]

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


OneAnchorTargets = AnchorFreeTargets
OneAnchorYoloLoss = AnchorFreeYoloLoss
