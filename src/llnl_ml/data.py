import cv2
import logging
import json
import os
import math
import numpy as np
import random
import torch

from torch.utils.data import Dataset, DataLoader, default_collate
from PIL import Image
from torch import nn
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2.functional import to_pil_image
import torchvision

if int(torchvision.__version__.split(".")[1]) < 16:
    from torchvision.datapoints import Mask as TVMask, BoundingBox as TVBBox
    from torchvision.transforms.v2.functional import to_image_tensor
    from torchvision.transforms.v2 import SanitizeBoundingBox

    CANVAS_SHAPE = "spatial_size"
else:
    from torchvision.tv_tensors import Mask as TVMask, BoundingBoxes as TVBBox
    from torchvision.transforms.v2.functional import to_image as to_image_tensor
    from torchvision.transforms.v2 import SanitizeBoundingBoxes as SanitizeBoundingBox

    CANVAS_SHAPE = "canvas_size"

from sklearn.model_selection import train_test_split

from typing import List, Optional, Tuple, Dict, Sequence

logger = logging.getLogger(__name__)


class ToPILTransform(nn.Module):
    """
    Calls torch's built in to_pill function to ensure the data is in PIL format prior
    to passing to Lightly's transforms
    """

    def __init__(self):
        super().__init__()

    def forward(self, image):
        if isinstance(image, Image.Image):
            return image
        return to_pil_image(image)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[transforms.Transform] = None,
        image_mode: str = "RGB",
        center_crop: bool = False,
        crop_size: int = 1200,
        crop_offset: Tuple[int] = (-60, -50),
        include_bbox: bool = False,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_mode = image_mode
        self.center_crop = center_crop
        self.crop_size = crop_size
        self.crop_offset = crop_offset
        self.include_bbox = include_bbox

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # img_mode = "RGB" if self.image_mode == "RGB" else "L"
        img_mode = cv2.IMREAD_COLOR if self.image_mode == "RGB" else cv2.IMREAD_GRAYSCALE
        # Use opencv to read in the images as it handles int16 images natively PIL does not
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_ANYDEPTH | img_mode)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

        # If we read image in as COLOR, opencv reads as BGR, convert to RGB
        if img_mode == cv2.IMREAD_COLOR:
            img = img[:, :, ::-1]
        # Else, if grayscale, add an explicit channel dimension such that shape = [H, W, 1]
        else:
            img = img[:, :, np.newaxis]

        # We expect the images to either be
        # 1) int16 greyscale images or
        # 2) uint8 RGB images
        # Normalize and convert the images as appropriate for the given image_mode
        # Check that image is loaded as an expected type
        if img.dtype not in (np.uint16, np.int16, np.uint8):
            raise TypeError(
                f"Loaded image {self.image_paths[idx]} not of expected type. "
                f"Expected uint8, uint16 or int16. loaded as {img.dtype} instead"
            )
        # Normalize the images to be between [0, 1] based on dtype max
        img = img.astype(np.float32) / np.iinfo(img.dtype).max

        # "Green" areas in the mask are positive or "1" and "White" areas are negative or "0"
        # We will use the Red channel to determine if the pixel is "White" (has value) or "Green" (no value)
        if mask.ndim == 3:
            mask = mask[:, :, 0] < 112
        mask = mask.astype(np.int16)

        # Crop out just the center with an offset if required
        if self.center_crop:
            top = (img.shape[0] // 2 + self.crop_offset[0]) - math.floor(self.crop_size / 2)
            bottom = (img.shape[0] // 2 + self.crop_offset[0]) + math.ceil(self.crop_size / 2)
            left = (img.shape[1] // 2 + self.crop_offset[1]) - math.floor(self.crop_size / 2)
            right = (img.shape[1] // 2 + self.crop_offset[1]) + math.ceil(self.crop_size / 2)
            img = img[top:bottom, left:right]
            mask = mask[top:bottom, left:right]

        # Return a tuple of Image and Targets where targets is a dict containing masks, boxes, and labels
        # This conforms to TorchVisions expected format for new v2 transforms to enable geometric transforms
        # on both the image and targets (masks and boxes).
        # TVImage and TVMask are both subclasses of torch.Tensor and convert from numpy to torch tensor
        image = to_image_tensor(img)
        targets = {"image_name": os.path.basename(self.image_paths[idx]).strip(), "masks": TVMask(mask)}

        if self.include_bbox:
            # To support TorchVision and Timm/Transformer based models, also create a bounding box and label targets
            # These will be ignored by models that do not need them.
            _, mask_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.int8))
            # cc_stats is a [N+1, 5] matrix where each row is a component and columns are [X, Y, Width, Height, Size]
            # First component is always the "background" and will be ignored
            masks = []
            boxes = torch.asarray(cc_stats[1:, :-1])
            # Convert to x1, y1, x2, y2 format from xywh
            boxes[:, 2:] += boxes[:, :2]
            for idx in range(1, cc_stats.shape[0]):
                masks.append(torch.asarray(mask_labels == idx, dtype=torch.uint8).unsqueeze(0))
            if masks:
                masks = torch.concatenate(masks, dim=0)
            # Remove degenerate boxes, no height or width
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            masks = masks[keep]

            # Create BoundingBox Object, use a dictionary and pass as key-value pairs
            # as the parameter name changed from "spatial_size" to "canvas_size" from torchvision 0.15->0.16
            boxes = TVBBox(**{"data": boxes, "format": "XYXY", CANVAS_SHAPE: mask.shape})

            targets["boxes"] = boxes
            targets["labels"] = torch.asarray([1] * boxes.shape[0], dtype=torch.int64)
            targets["masks"] = TVMask(masks)

        sample = (image, targets)

        if self.transform:
            sample = self.transform(sample)
        return sample


# From TorchVisision training example. Used to collate the batches into a list of images
# and list of targets. This is necessary for models that require bounding boxes as these
# can not be handled by the default_collate function.
def list_collate_fn(batch):
    images, targets = list(zip(*batch))
    return images, targets


def get_data_loaders(
    image_folder: str,
    mask_folder: str,
    batch_size: int,
    split_file: Optional[str] = None,
    train_count: Optional[int] = -1,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    num_workers: int = 8,
    image_mode: str = "RGB",
    image_size: int = 512,
    val_image_size: Optional[int] = None,
    test_image_size: Optional[int] = None,
    use_random_resize: bool = False,
    use_random_crop: bool = False,
    use_random_rotate: bool = False,
    color_jitter: float = 0.0,
    center_crop: bool = False,
    center_crop_size: int = 1200,
    center_crop_offset: Tuple[int] = (-60, -50),
    needs_boxes: bool = False,
):
    if split_file is not None:

        def prepend_directory(filenames, directory):
            return [os.path.join(directory, fn) for fn in filenames]

        split_dict = _load_json_file(split_file)

        img_train = prepend_directory(split_dict["train"]["image"], image_folder)
        mask_train = prepend_directory(split_dict["train"]["mask"], mask_folder)

        img_val = prepend_directory(split_dict["val"]["image"], image_folder)
        mask_val = prepend_directory(split_dict["val"]["mask"], mask_folder)

        img_test = prepend_directory(split_dict["test"]["image"], image_folder)
        mask_test = prepend_directory(split_dict["test"]["mask"], mask_folder)
    else:
        image_paths = sorted(
            [
                os.path.join(image_folder, fname)
                for fname in os.listdir(image_folder)
                if fname.endswith(("jpeg", "png", "jpg"))
            ]
        )
        mask_paths = sorted(
            [
                os.path.join(mask_folder, fname)
                for fname in os.listdir(mask_folder)
                if fname.endswith(("jpeg", "png", "jpg"))
            ]
        )

        # Split data into train, validation, and test sets
        img_train, img_temp, mask_train, mask_temp = train_test_split(
            image_paths, mask_paths, test_size=0.2, random_state=42
        )
        img_val, img_test, mask_val, mask_test = train_test_split(img_temp, mask_temp, test_size=0.5, random_state=42)

    # Sub-sample training set if percent_train is less than 1.0
    if train_count > 1:
        init_train_count = len(img_train)
        train_count = min(init_train_count, train_count)
        sampled_ind = sorted(random.sample(range(init_train_count), k=train_count))
        img_train = [img_train[ind] for ind in sampled_ind]
        mask_train = [mask_train[ind] for ind in sampled_ind]
        logger.info(f"Training with {len(sampled_ind)}/{init_train_count} training images.")
    # Set up Train and Val transformations
    # For example see:
    # https://pytorch.org/vision/0.15/auto_examples/plot_transforms_v2_e2e.html#sphx-glr-auto-examples-plot-transforms-v2-e2e-py
    train_transforms = []
    val_transforms = []
    test_transforms = []

    val_image_size = val_image_size if val_image_size is not None else image_size
    test_image_size = test_image_size if test_image_size is not None else val_image_size

    if use_random_crop and use_random_resize:
        logger.warning(
            "Both use_random_resize and use_random_crop set to True. Only applying random_resize augmentation."
        )
    if use_random_resize:
        train_transforms.append(
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0), antialias=True),
        )
    elif use_random_crop:
        train_transforms.append(transforms.RandomCrop(image_size))
    else:
        train_transforms.append(transforms.Resize(image_size, antialias=True))
    val_transforms.append(transforms.Resize(val_image_size, antialias=True))
    test_transforms.append(transforms.Resize(test_image_size, antialias=True))

    if use_random_rotate:
        train_transforms.append(transforms.RandomRotation(degrees=90))

    if color_jitter > 0.0:
        train_transforms.append(transforms.ColorJitter(brightness=color_jitter, hue=color_jitter))

    train_transforms.append(transforms.ConvertImageDtype(torch.float32))
    val_transforms.append(transforms.ConvertImageDtype(torch.float32))
    test_transforms.append(transforms.ConvertImageDtype(torch.float32))

    if needs_boxes:
        train_transforms.append(SanitizeBoundingBox())

    # Combine the lists of transforms using Combine to create callable transform function
    train_transforms = transforms.Compose(train_transforms)
    val_transforms = transforms.Compose(val_transforms)
    test_transforms = transforms.Compose(test_transforms)

    shared_kwargs = dict(
        image_mode=image_mode,
        center_crop=center_crop,
        crop_size=center_crop_size,
        crop_offset=center_crop_offset,
        include_bbox=needs_boxes,
    )
    # Create datasets
    train_dataset = SegmentationDataset(img_train, mask_train, transform=train_transforms, **shared_kwargs)
    val_dataset = SegmentationDataset(img_val, mask_val, transform=val_transforms, **shared_kwargs)
    test_dataset = SegmentationDataset(img_test, mask_test, transform=test_transforms, **shared_kwargs)

    # Create data loaders
    collate_fn = list_collate_fn if needs_boxes else default_collate
    val_batch_size = val_batch_size if val_batch_size is not None else batch_size
    test_batch_size = test_batch_size if test_batch_size is not None else val_batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


def _load_json_file(filename: str) -> Dict[str, Dict[str, Sequence]]:
    if filename.startswith("s3://"):
        import boto3

        s3 = boto3.resource("s3")
        bucket, key = filename.removeprefix("s3://").split("/", 1)
        data = s3.Object(bucket, key).get()["Body"].read().decode()
        split_file = json.loads(data)
    else:
        with open(filename, "r") as fp:
            split_file = json.load(fp)
    return split_file
