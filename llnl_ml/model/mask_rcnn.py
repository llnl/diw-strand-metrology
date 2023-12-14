from typing import Optional

import numpy as np
import torch
import torchmetrics

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import ResNet50_Weights


class MaskRCNNResNet50(torch.nn.Module):
    """
    Wrapper around Torch Visions reference of Mask-RCNN ResNet50 FPN V2 model:
    https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        score_thresh: float = 0.5,
        **kwargs,
    ):
        assert input_channels == 3, ValueError(
            f"MaskRCNN only works with RGB input images or 3 channel images. "
            f"Tried training with {input_channels = }"
        )
        super().__init__()
        backbone_weights = ResNet50_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(num_classes=output_channels + 1, weights_backbone=backbone_weights)
        self.score_thresh = score_thresh

    @property
    def calculates_loss(self):
        return True

    @property
    def needs_boxes(self):
        return True

    def forward(self, images, targets: Optional[dict] = None):
        # If training mode, send both images and targets and return losses
        if self.training:
            # Convert targets to lists as well
            losses = self.model(images, targets)
            # Only return the Mask loss for doing back prop as we have temp bounding boxes.
            return losses
        # Otherwise, we are in eval mode
        outputs = self.model(images)
        # Outputs is a list for each image, combine back along the batch dimension for val/test/inference
        masks = []
        for output in outputs:
            # Output mask is of shape [N_Regions, Label, H, W]
            # Flatten the N_Regions into a single mask dimension
            keep = output["scores"] >= self.score_thresh
            if torch.any(keep):
                mask = output["masks"][keep].max(dim=0)[0]
            else:
                mask = torch.zeros(
                    output["masks"].shape[1:], dtype=output["masks"].dtype, device=output["scores"].device
                )
            masks.append(mask)
        return torch.concatenate(masks, dim=0)
