from typing import Optional

import numpy as np
import torch
import torchmetrics

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

def eager_outputs_patch(self, losses, detections):
    # Patches the default eager_outputs that returns losses during training
    # and detections/masks on inference
    # This outputs both always to keep behavior consistent during val and train
    return detections, losses


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

        # Patch the eager outputs method in the mask-rcnn model to always output both losses and masks
        self.model.eager_outputs = eager_outputs_patch.__get__(self.model, GeneralizedRCNN)

    @property
    def calculates_loss(self):
        """This model calculates its own loss during training."""
        return True
    
    @staticmethod
    def get_dataset_requirements():
        """Returns the dataset wrapper class and collate function needed for this model."""
        from ..data.wrappers import BoundingBoxDatasetWrapper, list_collate_fn
        return BoundingBoxDatasetWrapper, list_collate_fn

    def forward(self, images, targets: Optional[dict] = None):
        # If training mode, send both images and targets and return losses
        outputs, losses = self.model(images, targets)

        # Outputs is a list for each image, combine back along the batch dimension for val/test/inference
        masks = []
        for output in outputs:
            # Output mask is of shape [N_Regions, Label, H, W]
            # Flatten the N_Regions into a single mask dimension
            keep = output["scores"] >= self.score_thresh
            if torch.any(keep):
                mask = output["masks"][keep].max(dim=0, keepdim=True)[0]
            else:
                mask = torch.zeros(
                    output["masks"].shape[1:], dtype=output["masks"].dtype, device=output["scores"].device
                )
            masks.append(mask)
        
        combined_masks = torch.concatenate(masks, dim=0)
        
        # Convert probabilities back to logits for consistency with UNet
        # Mask-RCNN outputs probabilities [0, 1], convert to logits [-inf, inf]
        # Use inverse sigmoid: logit(p) = log(p / (1 - p))
        eps = 1e-7  # Avoid log(0)
        combined_masks = torch.clamp(combined_masks, eps, 1 - eps)
        logits = torch.log(combined_masks / (1 - combined_masks))
        
        return logits, losses

