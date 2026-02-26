import logging
import os
import random
from typing import Optional

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, default_collate

from .dataset import SegmentationDataset
from .transforms import build_transforms
from .utils import load_json_file
from .wrappers import get_dataset_wrapper

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_FILE = "default_transforms.yaml"


def get_dataloaders(
    image_folder: str,
    mask_folder: str,
    batch_size: int,
    transform_config: str,
    split_file: Optional[str] = None,
    train_count: int = -1,
    val_batch_size: Optional[int] = None,
    test_batch_size: int = 1,
    num_workers: int = 8,
    model_class: Optional[type] = None,
):
    # Get the image and mask file paths for each split
    train_paths, val_paths, test_paths = get_paths_for_splits(image_folder, mask_folder, split_file, train_count)

    if not transform_config:
        transform_config = DEFAULT_CONFIG_FILE

    train_transform, val_transform = build_transforms(transform_config)

    # Build the base datasets
    train_dataset = SegmentationDataset(**train_paths, transform=train_transform)
    val_dataset = SegmentationDataset(**val_paths, transform=val_transform)
    test_dataset = SegmentationDataset(**test_paths, transform=val_transform)

    # Apply dataset wrapper and get collate function based on model requirements
    collate_fn = default_collate
    if model_class is not None and hasattr(model_class, 'get_dataset_requirements'):
        wrapper_class, collate_fn = model_class.get_dataset_requirements()
        
        if wrapper_class is not None:
            logger.info(f"Applying dataset wrapper: {wrapper_class.__name__}")
            train_dataset = wrapper_class.wrap(train_dataset)
            val_dataset = wrapper_class.wrap(val_dataset)
            test_dataset = wrapper_class.wrap(test_dataset)
        else:
            logger.info("No dataset wrapper needed for this model")
    else:
        logger.info("No model class provided, using default dataset format")

    # Build the dataloaders
    val_batch_size = val_batch_size if val_batch_size is not None else batch_size

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


def get_paths_for_splits(
    image_folder: str,
    mask_folder: str,
    split_file: Optional[str] = None,
    train_count: int = -1,
):
    if split_file is not None:

        def prepend_directory(filenames, directory):
            return [os.path.join(directory, fn) for fn in filenames]

        split_dict = load_json_file(split_file)

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

    return {"image_paths": img_train, "mask_paths": mask_train}, {"image_paths": img_val, "mask_paths": mask_val}, {"image_paths": img_test, "mask_paths": mask_test}


