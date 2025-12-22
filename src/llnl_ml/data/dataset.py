import cv2
import logging
import json
import os
import math
import numpy as np
import random
import torch
import albumentations as A

from torch.utils.data import Dataset, DataLoader, default_collate
from PIL import Image
from torch import nn

from sklearn.model_selection import train_test_split

from typing import List, Optional, Tuple, Dict, Sequence

from .utils import load_and_convert_image

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[A.Compose] = None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Use pillow to open the image
        image = load_and_convert_image(self.image_paths[idx])

        # Make sure image has 3 dimensions
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        # Load the Mask
        mask = np.asarray(Image.open(self.mask_paths[idx]))

        # "Green" areas in the mask are positive or "1" and "White" areas are negative or "0"
        # We will use the Red channel to determine if the pixel is "White" (has value) or "Green" (no value)
        if mask.ndim == 3:
            mask = mask[:, :, 1] > 112
      
        # Check for an odd scaling case
        if mask.max() == 255 and mask.min() == 112:
            mask[mask==255] = 1
            mask[mask==112] = 0
        

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Always ensure the output is of torch tensor types
            image = torch.as_tensor(image)
            mask = torch.as_tensor(mask)

        if mask.ndim == 2:
            mask = torch.unsqueeze(mask, 0)

        targets = {"image_name": os.path.basename(self.image_paths[idx]).strip(), "masks": mask}

        return image, targets

