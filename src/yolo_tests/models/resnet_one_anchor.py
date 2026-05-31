"""Educational one-anchor YOLO-style model built on a ResNet backbone.

This is intentionally simple so the tensor shapes are easy to follow:

1. Read three ResNet feature levels: layer2, layer3, and layer4.
2. Upsample layer3 and layer4 to the layer2 spatial size.
3. Concatenate the three feature maps along the channel dimension.
4. Apply one 1x1 convolution that predicts 80 class logits and 4 box deltas.
5. Decode one anchor at every feature-grid point into normalized YOLO boxes.

The model predicts one box per grid point. There is no separate objectness head in
this educational version; background is learned through the 80 class logits being
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
        level1 = self.layer2(x)
        level2 = self.layer3(level1)
        level3 = self.layer4(level2)
        return level1, level2, level3


class ResNetOneAnchorYolo(nn.Module):
    """A compact one-anchor detector for COCO-style YOLO targets.

    Output dictionary:
        class_logits: raw class scores, shape [B, 80, H, W]
        class_probs: sigmoid(class_logits), shape [B, 80, H, W]
        box_deltas: raw anchor adjustments, shape [B, 4, H, W]
        boxes_yolo: decoded normalized cxcywh boxes, shape [B, H, W, 4]
        raw: concatenated raw head output, shape [B, 84, H, W]
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone_name: str = "resnet101",
        weights: str | None = "default",
        anchor_size: float | tuple[float, float] = 0.10,
        freeze_backbone: bool = False,
        normalize_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNetFeatureExtractor(backbone_name=backbone_name, weights=weights)
        in_channels = sum(self.backbone.out_channels)
        self.pre_prediction=nn.Sequential(nn.Conv2d(in_channels, 1024, kernel_size=3, stride=2,padding=1),nn.Conv2d(1024, 512, kernel_size=3, stride=2,padding=1))
        self.prediction = nn.Conv2d(512, num_classes + 4, kernel_size=1)
        self.normalize_inputs = normalize_inputs

        if isinstance(anchor_size, (int, float)):
            anchor_wh = (float(anchor_size), float(anchor_size))
        else:
            if len(anchor_size) != 2:
                raise ValueError("anchor_size must be a float or (width, height)")
            anchor_wh = (float(anchor_size[0]), float(anchor_size[1]))
        self.register_buffer("anchor_wh", torch.tensor(anchor_wh, dtype=torch.float32))

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self._init_prediction_head()

    def _init_prediction_head(self) -> None:
        nn.init.normal_(self.prediction.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.prediction.bias)
        with torch.no_grad():
            # Start class probabilities low because most grid points are background.
            self.prediction.bias[: self.num_classes].fill_(-6.0)

    def forward(self, images: Tensor) -> dict[str, Tensor | tuple[tuple[int, int], ...]]:
        if self.normalize_inputs:
            images = (images - self.image_mean) / self.image_std

        level1, level2, level3 = self.backbone(images)
        target_size = level1.shape[-2:]
        level2_up = F.interpolate(level2, size=target_size, mode="bilinear", align_corners=False)
        level3_up = F.interpolate(level3, size=target_size, mode="bilinear", align_corners=False)
        fused = torch.cat([level1, level2_up, level3_up], dim=1)
        fused= self.pre_prediction(fused)
        raw = self.prediction(fused)
        class_logits, box_deltas = raw.split([self.num_classes, 4], dim=1)
        boxes_yolo = self.decode_boxes(box_deltas)

        return {
            "raw": raw,
            "class_logits": class_logits,
            "class_probs": torch.sigmoid(class_logits),
            "box_deltas": box_deltas,
            "boxes_yolo": boxes_yolo,
            "feature_shapes": (level1.shape[-2:], level2.shape[-2:], level3.shape[-2:]),
        }

    def decode_boxes(self, box_deltas: Tensor) -> Tensor:
        """Decode raw deltas into normalized YOLO cxcywh boxes.

        One anchor lives at every grid point. The first two deltas move the box
        center inside that grid cell. The last two deltas multiply the shared
        anchor width and height.
        """

        batch_size, _, grid_h, grid_w = box_deltas.shape
        deltas = box_deltas.permute(0, 2, 3, 1).contiguous()

        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=box_deltas.device, dtype=box_deltas.dtype),
            torch.arange(grid_w, device=box_deltas.device, dtype=box_deltas.dtype),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, grid_h, grid_w, 2)
        grid_size = box_deltas.new_tensor([grid_w, grid_h]).view(1, 1, 1, 2)

        center_xy = (grid + torch.sigmoid(deltas[..., 0:2])) / grid_size
        center_xy = center_xy.expand(batch_size, -1, -1, -1)
        anchor_wh = self.anchor_wh.to(device=box_deltas.device, dtype=box_deltas.dtype).view(1, 1, 1, 2)
        size_wh = anchor_wh * torch.exp(deltas[..., 2:4].clamp(min=-4.0, max=4.0))
        size_wh = size_wh.clamp(min=1.0 / max(grid_h, grid_w), max=1.0)

        return torch.cat([center_xy, size_wh], dim=-1)
