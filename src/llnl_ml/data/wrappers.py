"""
Dataset wrapper classes for transforming dataset outputs to meet specific model requirements.

This module provides a flexible system for adapting the standard SegmentationDataset
output format to meet the specific needs of different model architectures.
"""

import cv2
import torch
import numpy as np
from typing import Any, Callable, Optional, Tuple, Dict, List
from torch.utils.data import Dataset, default_collate


class DatasetWrapper(Dataset):
    """
    Base wrapper class that delegates most operations to the wrapped dataset
    while allowing transformation of the __getitem__ output.
    """
    
    def __init__(self, dataset: Dataset, transform_fn: Optional[Callable] = None):
        """
        Initialize the wrapper.
        
        Args:
            dataset: The dataset to wrap
            transform_fn: Optional function to transform the output of dataset.__getitem__
        """
        self.dataset = dataset
        self.transform_fn = transform_fn
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped dataset."""
        return getattr(self.dataset, name)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get item from wrapped dataset and optionally transform it."""
        item = self.dataset[idx]
        
        if self.transform_fn is not None:
            return self.transform_fn(item, idx)
        
        return item
    
    @classmethod
    def wrap(cls, dataset: Dataset, **kwargs) -> 'DatasetWrapper':
        """
        Factory method to create a wrapper instance.
        
        Args:
            dataset: The dataset to wrap
            **kwargs: Additional arguments for the wrapper
            
        Returns:
            Wrapped dataset instance
        """
        return cls(dataset, **kwargs)


class BoundingBoxDatasetWrapper(DatasetWrapper):
    """
    Wrapper that adds bounding box information to segmentation masks.
    
    This wrapper processes the mask output from SegmentationDataset and adds:
    - Individual instance masks for each connected component
    - Bounding boxes in XYXY format
    - Labels for each instance
    """
    
    def __init__(self, dataset: Dataset):
        """Initialize the bounding box wrapper."""
        super().__init__(dataset, self._add_bounding_boxes)
    
    def _add_bounding_boxes(self, item: Tuple[torch.Tensor, Dict], idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Transform the dataset item to include bounding box information.
        
        Args:
            item: Original (image, targets) tuple from dataset
            idx: Index of the item (unused but required for interface)
            
        Returns:
            Transformed (image, targets) tuple with bounding box information
        """
        image, targets = item
        mask = targets["masks"]
        
        # Convert mask to numpy for processing
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().numpy().astype(np.uint8)
        else:
            mask_np = np.array(mask, dtype=np.uint8)
        
        # Find connected components
        _, mask_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask_np.astype(np.int8))
        
        # cc_stats is a [N+1, 5] matrix where each row is a component 
        # and columns are [X, Y, Width, Height, Size]
        # First component is always the "background" and will be ignored
        
        if cc_stats.shape[0] <= 1:  # Only background, no objects
            # Return empty tensors
            targets.update({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, mask_np.shape[0], mask_np.shape[1]), dtype=torch.uint8)
            })
            return image, targets
        
        # Process each connected component (skip background at index 0)
        instance_masks = []
        boxes = torch.tensor(cc_stats[1:, :-1], dtype=torch.float32)  # [X, Y, Width, Height]
        
        # Convert to x1, y1, x2, y2 format from xywh
        boxes[:, 2:] += boxes[:, :2]
        
        # Create individual instance masks
        for component_idx in range(1, cc_stats.shape[0]):
            instance_mask = (mask_labels == component_idx).astype(np.uint8)
            instance_masks.append(torch.tensor(instance_mask, dtype=torch.uint8))
        
        if instance_masks:
            masks_tensor = torch.stack(instance_masks, dim=0)
        else:
            masks_tensor = torch.zeros((0, mask_np.shape[0], mask_np.shape[1]), dtype=torch.uint8)
        
        # Remove degenerate boxes (no height or width)
        if len(boxes) > 0:
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            masks_tensor = masks_tensor[keep]
        
        # Create labels (all instances are class 1 for binary segmentation)
        labels = torch.ones(len(boxes), dtype=torch.int64)
        
        # Update targets with bounding box information
        targets.update({
            "boxes": boxes,
            "labels": labels,
            "masks": masks_tensor
        })
        
        return image, targets
    
    @classmethod
    def wrap(cls, dataset: Dataset) -> 'BoundingBoxDatasetWrapper':
        """Factory method to create a bounding box wrapper."""
        return cls(dataset)


def list_collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Custom collate function for datasets that return variable-sized targets.
    
    This is necessary for models that require bounding boxes, as these
    cannot be handled by the default_collate function due to variable
    numbers of objects per image.
    
    Args:
        batch: List of (image, targets) tuples
        
    Returns:
        Tuple of (images_list, targets_list)
    """
    images, targets = list(zip(*batch))
    return list(images), list(targets)


# Registry of available wrappers
DATASET_WRAPPERS = {
    'bounding_box': BoundingBoxDatasetWrapper,
    'default': DatasetWrapper,
}


def get_dataset_wrapper(wrapper_name: str) -> type:
    """
    Get a dataset wrapper class by name.
    
    Args:
        wrapper_name: Name of the wrapper to retrieve
        
    Returns:
        Dataset wrapper class
        
    Raises:
        KeyError: If wrapper_name is not found
    """
    if wrapper_name not in DATASET_WRAPPERS:
        available = list(DATASET_WRAPPERS.keys())
        raise KeyError(f"Unknown dataset wrapper '{wrapper_name}'. Available: {available}")
    
    return DATASET_WRAPPERS[wrapper_name]