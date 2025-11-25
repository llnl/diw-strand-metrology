from typing import Optional, Dict, Union, Callable
import torch
import torch.nn as nn


class SegmentationModelWithLoss(nn.Module):
    """
    Wrapper around segmentation models that computes loss internally during training.
    Makes models behave consistently for simplified training loops.
    
    Supports single or multiple loss functions.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        loss_fns: Union[Callable, Dict[str, Callable]] = None
    ):
        """
        Args:
            model: An instance of UNet, FCN, or similar segmentation model
            loss_fns: Either:
                - Single loss function (e.g., nn.BCEWithLogitsLoss())
                - Dict of loss functions (e.g., {"bce": nn.BCEWithLogitsLoss(), "dice": DiceLoss()})
                - None (defaults to BCEWithLogitsLoss)
        """
        super().__init__()
        self.model = model
        
        # Handle loss function configuration
        if loss_fns is None:
            # Default to BCE with logits
            self.loss_fns = {"loss_mask": nn.BCEWithLogitsLoss()}
        elif isinstance(loss_fns, dict):
            # Multiple losses provided
            self.loss_fns = loss_fns
        else:
            # Single loss function provided
            self.loss_fns = {"loss_mask": loss_fns}
    
    @property
    def calculates_loss(self) -> bool:
        """Indicates this model computes loss internally"""
        return True
    
    @property
    def needs_boxes(self) -> bool:
        """Indicates this model doesn't need bounding box format"""
        return False
    
    def forward(self, images: torch.Tensor, targets: Optional[Dict] = None):
        """
        Forward pass that computes loss during training, returns logits during eval.
        
        Args:
            images: Input images tensor [B, C, H, W]
            targets: Dict containing 'masks' key with ground truth masks [B, H, W]
        
        Returns:
            If training: Dict with loss values (e.g., {"loss_mask": 0.5, "loss_dice": 0.3})
            If eval: Logits tensor [B, H, W]
        """
        # Get predictions from model (already squeezed if output_channels=1)
        pred_logits = self.model(images)
        
        if self.training:
            # Training mode: compute and return loss dict
            if targets is None:
                raise ValueError("targets must be provided during training")
            
        loss_dict = {}
        if targets is not None:
            mask = targets["masks"].float()
        
            # Compute all losses
            for loss_name, loss_fn in self.loss_fns.items():
                loss_dict[loss_name] = loss_fn(pred_logits, mask)
        
        # Return logits and loss dict (loss dict will be empty if targets is None)
        return pred_logits, loss_dict
