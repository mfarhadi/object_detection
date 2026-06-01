"""Educational anchor-free YOLO-style model built on a ResNet backbone.

This is intentionally simple so the tensor shapes are easy to follow:

1. Read three ResNet feature levels: layer2, layer3, and layer4.
2. Upsample layer3 and layer4 to the layer2 spatial size.
3. Concatenate the three feature maps along the channel dimension.
4. Apply separate 1x1 heads for class logits and anchor-free box parameters.
5. Decode one anchor-free box at every feature-grid point into normalized YOLO boxes.

The model predicts one box per grid point. There is no separate objectness head
in this educational version; background is learned through the class logits being
trained toward zero at locations that do not own an object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)


@dataclass(frozen=True)
class ResNetSpec:
    builder: Any
    default_weights: Any
    feature_channels: tuple[int, int, int]


RESNET_SPECS: dict[str, ResNetSpec] = {
    "resnet18": ResNetSpec(resnet18, ResNet18_Weights.DEFAULT, (128, 256, 512)),
    "resnet34": ResNetSpec(resnet34, ResNet34_Weights.DEFAULT, (128, 256, 512)),
    "resnet50": ResNetSpec(resnet50, ResNet50_Weights.DEFAULT, (512, 1024, 2048)),
    "resnet101": ResNetSpec(resnet101, ResNet101_Weights.DEFAULT, (512, 1024, 2048)),
}


def _resolve_weights(backbone_name: str, weights: str | None) -> Any:
    """Map a friendly string to torchvision's weights object."""
    if weights is None or weights.lower() in {"none", "false", "0"}:
        return None
    if weights.lower() in {"default", "pretrained", "imagenet"}:
        return RESNET_SPECS[backbone_name].default_weights
    raise ValueError(
        f"Unsupported weights={weights!r}. Use one of: default, pretrained, imagenet, none."
    )


def ensure_resnet_weights_available(backbone_name: str, weights: str | None) -> None:
    if backbone_name not in RESNET_SPECS:
        choices = ", ".join(sorted(RESNET_SPECS))
        raise ValueError(f"Unsupported backbone {backbone_name!r}. Choices: {choices}")
    resolved_weights = _resolve_weights(backbone_name, weights)
    if resolved_weights is not None:
        resolved_weights.get_state_dict(progress=True)


class ConvBNAct(nn.Module):
    """Small conv block used in the detection head."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResNetFeatureExtractor(nn.Module):
    """Return layer2/layer3/layer4 feature maps from a torchvision ResNet."""

    def __init__(self, backbone_name: str = "resnet101", weights: str | None = "default") -> None:
        super().__init__()
        if backbone_name not in RESNET_SPECS:
            choices = ", ".join(sorted(RESNET_SPECS))
            raise ValueError(f"Unsupported backbone {backbone_name!r}. Choices: {choices}")

        spec = RESNET_SPECS[backbone_name]
        model = spec.builder(weights=_resolve_weights(backbone_name, weights))
        self.out_channels = spec.feature_channels

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        level1 = self.layer2(x)  # stride 8
        level2 = self.layer3(level1)  # stride 16
        level3 = self.layer4(level2)  # stride 32
        return level1, level2, level3


class ResNetAnchorFreeYolo(nn.Module):
    """
    Anchor-free detector built on a ResNet backbone.

    Output dictionary:
        class_logits: raw class scores, shape [B, C, H, W]
        class_probs: sigmoid(class_logits), shape [B, C, H, W]
        box_raw: raw box parameters, shape [B, 6, H, W]
            channels = [dx, dy, l, t, r, b]
        boxes_yolo: decoded normalized cxcywh boxes, shape [B, H, W, 4]
        raw: concatenated raw head output, shape [B, C + 6, H, W]

    Box decoding:
        - dx, dy are center offsets inside the cell, passed through sigmoid
        - l, t, r, b are positive distances from the predicted center to edges
          passed through softplus
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone_name: str = "resnet101",
        weights: str | None = "default",
        freeze_backbone: bool = False,
        normalize_inputs: bool = True,
        head_channels: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNetFeatureExtractor(backbone_name=backbone_name, weights=weights)
        self.normalize_inputs = normalize_inputs

        in_channels = sum(self.backbone.out_channels)

        self.fuse = nn.Sequential(
            ConvBNAct(in_channels, head_channels, kernel_size=1),
            ConvBNAct(head_channels, head_channels, kernel_size=3),
            ConvBNAct(head_channels, head_channels, kernel_size=3),
        )

        self.class_head = nn.Conv2d(head_channels, num_classes, kernel_size=1)
        self.box_head = nn.Conv2d(head_channels, 6, kernel_size=1)

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self._init_heads()

    def _init_heads(self) -> None:
        nn.init.normal_(self.class_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.class_head.bias)
        with torch.no_grad():
            self.class_head.bias.fill_(-6.0)

        nn.init.normal_(self.box_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.box_head.bias)
        with torch.no_grad():
            # Keep initial boxes small and stable.
            # dx, dy near 0.5 after sigmoid; ltrb near small positive values after softplus.
            self.box_head.bias[0:2].fill_(0.0)
            self.box_head.bias[2:6].fill_(-2.0)

    def forward(self, images: Tensor) -> dict[str, Tensor | tuple[tuple[int, int], ...]]:
        if self.normalize_inputs:
            images = (images - self.image_mean) / self.image_std

        level1, level2, level3 = self.backbone(images)
        target_size = level1.shape[-2:]

        level2_up = F.interpolate(level2, size=target_size, mode="bilinear", align_corners=False)
        level3_up = F.interpolate(level3, size=target_size, mode="bilinear", align_corners=False)

        fused = torch.cat([level1, level2_up, level3_up], dim=1)
        fused = self.fuse(fused)

        class_logits = self.class_head(fused)  # [B, C, H, W]
        box_raw = self.box_head(fused)         # [B, 6, H, W]
        boxes_yolo = self.decode_boxes(box_raw)

        raw = torch.cat([class_logits, box_raw], dim=1)

        return {
            "raw": raw,
            "class_logits": class_logits,
            "class_probs": torch.sigmoid(class_logits),
            "box_raw": box_raw,
            "boxes_yolo": boxes_yolo,
            "feature_shapes": (level1.shape[-2:], level2.shape[-2:], level3.shape[-2:]),
        }

    def decode_boxes(self, box_raw: Tensor) -> Tensor:
        """
        Decode raw box params into normalized YOLO cxcywh boxes.

        box_raw channels:
            0: dx   center x offset within cell, sigmoid to [0, 1]
            1: dy   center y offset within cell, sigmoid to [0, 1]
            2: l    left distance from center to box edge, softplus
            3: t    top distance from center to box edge, softplus
            4: r    right distance from center to box edge, softplus
            5: b    bottom distance from center to box edge, softplus
        """
        batch_size, _, grid_h, grid_w = box_raw.shape
        raw = box_raw.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 6]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=box_raw.device, dtype=box_raw.dtype),
            torch.arange(grid_w, device=box_raw.device, dtype=box_raw.dtype),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, grid_h, grid_w, 2)
        grid_size = box_raw.new_tensor([grid_w, grid_h]).view(1, 1, 1, 2)

        # Predicted center point for each cell.
        center_offset = torch.sigmoid(raw[..., 0:2])
        center_xy = (grid + center_offset) / grid_size
        center_xy = center_xy.expand(batch_size, -1, -1, -1)

        # Positive distances from center to edges.
        ltrb = F.softplus(raw[..., 2:6])

        x1 = (center_xy[..., 0:1] - ltrb[..., 0:1]).clamp(0.0, 1.0)
        y1 = (center_xy[..., 1:2] - ltrb[..., 1:2]).clamp(0.0, 1.0)
        x2 = (center_xy[..., 0:1] + ltrb[..., 2:3]).clamp(0.0, 1.0)
        y2 = (center_xy[..., 1:2] + ltrb[..., 3:4]).clamp(0.0, 1.0)

        x1y1 = torch.cat([x1, y1], dim=-1)
        x2y2 = torch.cat([x2, y2], dim=-1)

        # Convert xyxy -> cxcywh
        cxcy = (x1y1 + x2y2) / 2.0
        wh = (x2y2 - x1y1).clamp_min(1e-6)

        return torch.cat([cxcy, wh], dim=-1)


# Backward-compatible alias if old code still imports the old name.
ResNetOneAnchorYolo = ResNetAnchorFreeYolo
