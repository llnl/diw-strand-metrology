import argparse
import logging
import os
from tqdm import tqdm
from typing import Union

import cv2
import numpy as np

from collections import defaultdict
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create Stream handler
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

VALID_IMAGE_TYPES = ("png", "jpg", "jpeg")

TAB = "\t"
NEWLINE = "\n"
NLTAB = "\n\t"


def validate_data(image_dir: str, mask_dir: str, delete_invalid: bool = False) -> None:
    """
    Crawls through the image and mask directories and validates the data.
    If delete_invalid is set to True, then invalid data will be deleted.

    Checks the following criteria:
    * Every image has a mask and vice-versa
    * Images are
        * of shape [2048, 2048]
        * Contain int16 ranged values
    * Masks are
        * of shape [2048, 2048]
        * contain positive label area

    Any mask or image which does not meet above criteria are deleted if delete_invalid is true.

    validate_data_with_stats(image_dir, mask_dir, delete_invalid)
    :param image_dir: Path to images
    :param mask_dir: Path to masks
    :param delete_invalid: If True, deletes all invalid images and masks
    """
    # Get all images in image_dir
    image_filenames = [f for f in os.listdir(image_dir) if f.split(".")[-1] in VALID_IMAGE_TYPES]

    # Get all masks in mask_dir
    mask_filenames = [f for f in os.listdir(mask_dir) if f.split(".")[-1] in VALID_IMAGE_TYPES]

    # 1) Ensure there is a one to one mapping between masks and images
    image_basenames = {f.split("-")[0] for f in image_filenames}
    mask_basenames = {f.split("-")[0] for f in mask_filenames}

    logger.info(f"Found {len(image_basenames)} images and {len(mask_basenames)} masks.")

    logger.info(f"Images without a mask: \n\t{NLTAB.join(image_basenames - mask_basenames)}")
    logger.info(f"Masks without an image: \n\t{NLTAB.join(mask_basenames - image_basenames)}")

    # 2) Validate images have expected value ranges
    invalid_images = defaultdict(list)
    image_stats = {}
    for image_file in tqdm(image_filenames, desc="Validating Images"):
        image_path = os.path.join(image_dir, image_file)
        image = np.array(Image.open(image_path))
        stats = {
            "shape": image.shape,
            "min": image.min(),
            "max": image.max(),
            "dtype": image.dtype,
        }
        image_base = image_file.split("-")[0]
        image_stats[image_base] = stats
        # Expect int16 imagery, therefore values should be in the range [0, 2**16]
        # We also expect the max value to be above 255 as this will indicate the image
        # values are smashed and no longer viable
        if stats["min"] < 0 or stats["max"] < 255 or stats["max"] > 2**16:
            invalid_images[image_base].append("Range")

        # Ensure shape is 2048 x 2048
        if stats["shape"] != (2048, 2048):
            invalid_images[image_base].append("Shape")

    logger.info(f"The following images were invalid: \n\t{_dumps_invalid(invalid_images)}")

    # 3) Validate masks
    invalid_masks = defaultdict(list)
    for mask_file in tqdm(mask_filenames, desc="Validating Masks"):
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        mask_base = mask_file.split("-")[0]
        if mask.ndim == 3:
            mask = mask[:, :, 0] < 112
        # 3.1) Ensure masks have some "positive" area
        if mask.sum() < 1:
            invalid_masks[mask_base].append("No positive area")

        # 3.2) Ensure masks have same shape as images
        if mask.shape != (2048, 2048):
            invalid_masks[mask_base].append("Shape")

    logger.info(f"The following images were invalid: \n\t{_dumps_invalid(invalid_masks)}")

    if delete_invalid:
        to_delete = (
            set(invalid_images.keys())
            | set(invalid_masks.keys())
            | set(image_basenames ^ mask_basenames)  # symmetric_difference
        )

        logger.info(f"Deleting the following images and masks: \n\t{NLTAB.join(list(to_delete))}")
        for basename in to_delete:
            mask_file = os.path.join(mask_dir, f"{basename}-mask.png")
            image_file = os.path.join(image_dir, f"{basename}-raw.png")
            if os.path.exists(mask_file):
                os.remove(mask_file)
            if os.path.exists(image_file):
                os.remove(image_file)

    return None


def _dumps_invalid(invalid: dict, div: str = NLTAB) -> str:
    lines = [f"{key}: {items}" for key, items in invalid.items()]
    return div.join(lines)


def str2bool(v: Union[bool, str]) -> bool:
    # When using bool, empty strings evaluate to false, and everything else evaluates to True.
    # Better to use this function for parsing purposes.
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--delete-invalid", type=str2bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_data(**vars(args))
