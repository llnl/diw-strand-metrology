"""
Dataset tracking module for MLFlow integration.

This module provides functions to track datasets used in training runs,
including dataset name extraction, split information loading, and MLFlow logging.
"""

import json
import logging
import hashlib
import os
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Optional

import boto3
import pandas as pd
import mlflow
from mlflow.data import Dataset, DatasetInput
from mlflow.entities import InputTag
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource

logger = logging.getLogger(__name__)


class S3DataSource(FileSystemDatasetSource):
    """
    Implements an S3 dataset source for MLFlow dataset tracking.
    """
    def __init__(self, split_uri: Optional[str] = None, image_uri: Optional[str] = None, mask_uri: Optional[str] = None):
        # Don't call super().__init__() as FileSystemDatasetSource expects a path parameter
        self.split_uri = split_uri
        self.image_uri = image_uri
        self.mask_uri = mask_uri
        # Use split_uri as primary, fall back to image_uri
        self._uri = split_uri if split_uri else image_uri

    @property
    def uri(self):
        """Return the primary URI for this dataset source."""
        return self._uri

    @staticmethod
    def _get_source_type() -> str:
        """Return the source type identifier."""
        return "s3"

    def load(self, dst_path=None) -> str:
        """
        Load method required by FileSystemDatasetSource.
        For S3 sources, we don't actually download the data here.
        """
        return dst_path if dst_path else self._uri

    @classmethod
    def _resolve(cls, raw_source) -> "S3DataSource":
        """
        Resolve a raw source into an S3DataSource instance.
        """
        if isinstance(raw_source, dict):
            return cls.from_dict(raw_source)
        elif isinstance(raw_source, str):
            # Assume it's an image URI if just a string
            return cls(image_uri=raw_source)
        return raw_source

    def to_dict(self) -> dict:
        """Convert the data source to a dictionary for serialization."""
        return {
            "split_uri": self.split_uri,
            "image_uri": self.image_uri,
            "mask_uri": self.mask_uri,
        }

    @classmethod
    def from_dict(cls, source_dict: dict) -> "S3DataSource":
        """Create an S3DataSource from a dictionary."""
        return cls(
            split_uri=source_dict.get("split_uri"),
            image_uri=source_dict.get("image_uri"),
            mask_uri=source_dict.get("mask_uri")
        )


def extract_dataset_name(split_file: Optional[str], train_image_s3_uri: Optional[str]) -> Optional[str]:
    """
    Extract dataset name from split file or S3 URI.
    
    :param split_file: Path or S3 URI to split file
    :param train_image_s3_uri: S3 URI for training images
    :return: Dataset name or None if unable to determine
    """
    # Try split file first
    if split_file:
        return Path(split_file).stem
    
    # Try S3 URI
    if train_image_s3_uri:
        # Parse S3 URI and extract dataset name
        # Remove trailing slashes and split by '/'
        parts = train_image_s3_uri.rstrip('/').split('/')
        
        # Generic directory names to skip
        generic_names = {'images', 'masks', 'data', 'train', 'val', 'test'}
        
        # Walk backwards through path parts to find non-generic name
        for part in reversed(parts):
            if part and part.lower() not in generic_names:
                return part
    
    return None


def load_split_information(split_file: Optional[str]) -> Optional[Dict]:
    """
    Load split information from JSON file.
    
    :param split_file: Path or S3 URI to split file
    :return: Dictionary with split information or None if not available
    """
    if not split_file:
        return None
    
    try:
        # Handle S3 URIs by downloading first
        local_split_file = split_file
        if split_file.startswith('s3://'):
            local_split_file = _download_s3_uri(split_file)
        
        # Load JSON file
        with open(local_split_file, 'r') as f:
            raw_split = json.load(f)
        
        # Normalize split data structure
        split_info = {}
        for split_name in ['train', 'val', 'test']:
            if split_name in raw_split:
                split_data = raw_split[split_name]
                
                # Handle both 'image' and 'images' keys
                images = split_data.get('images', split_data.get('image', []))
                masks = split_data.get('masks', split_data.get('mask', []))
                
                split_info[split_name] = {
                    'images': images,
                    'masks': masks,
                    'count': len(images)
                }
        
        return split_info
        
    except FileNotFoundError:
        logger.warning(f"Split file not found: {split_file}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse split file {split_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading split information: {e}")
        return None


def create_mlflow_dataset(
    dataset_name: str,
    s3_uris: Dict[str, Optional[str]],
    split_info: Optional[Dict],
    split_file: Optional[str] = None
) -> tuple[Dataset, Dict]:
    """
    Create MLFlow dataset object and metadata dictionary.
    
    :param dataset_name: Name of the dataset
    :param s3_uris: Dictionary with 'train_image', 'train_mask', and optionally 'split_file' S3 URIs
    :param split_info: Optional split information dictionary
    :param split_file: Optional path or S3 URI to split file
    :return: Tuple of (MLFlow Dataset object, metadata dictionary)
    """
    # Prefer split_file as source if available, otherwise use train_image S3 URI
    if split_file:
        source_uri = split_file
    else:
        source_uri = s3_uris.get('train_image', 'unknown')
    
    # Build metadata dictionary to log separately
    metadata = {}
    # Add split counts if available
    if split_info:
        for split_name, split_data in split_info.items():
            metadata[f'dataset_{split_name}_count'] = split_data['count']

    # Create a digest by hashing the split metadata information
    digest_data = json.dumps(s3_uris | metadata, sort_keys=True)
    digest = hashlib.sha256(digest_data.encode('utf-8')).hexdigest()[:8]
    
    # Create Dataset with source (split_file if available, otherwise train_image URI) and name
    # Metadata will be logged separately as MLFlow parameters
    dataset = Dataset(
        source=S3DataSource(
            split_uri=split_file,  # Will be None if not provided
            image_uri=s3_uris.get('train_image'),
            mask_uri=s3_uris.get('train_mask')
        ),
        name=dataset_name,
        digest=digest,
    )
    
    return dataset, metadata


def log_split_artifact(split_info: Dict, mlflow_logger) -> None:
    """
    Log split information as CSV artifact to MLFlow.
    
    :param split_info: Dictionary with split information
    :param mlflow_logger: MLFlow logger instance
    """
    # Convert split info to list of rows
    rows = []
    for split_name, split_data in split_info.items():
        images = split_data.get('images', [])
        masks = split_data.get('masks', [])
        
        for img, mask in zip(images, masks):
            rows.append({
                'split': split_name,
                'image_file': img,
                'mask_file': mask
            })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to temporary CSV file and log as artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        df.to_csv(temp_path, index=False)
    
    try:
        mlflow_logger.experiment.log_artifact(local_path=temp_path, artifact_path='dataset', run_id=mlflow_logger.run_id)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def extract_split_from_loaders(
    train_loader,
    val_loader,
    test_loader,
    image_folder: str,
    mask_folder: str
) -> Optional[Dict]:
    """
    Extract split information from data loaders as a fallback when split_file is not available.
    
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param test_loader: Test data loader
    :param image_folder: Base path for image folder
    :param mask_folder: Base path for mask folder
    :return: Dictionary with split information or None if extraction fails
    """
    try:
        split_info = {}
        
        for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            dataset = loader.dataset
            
            # Extract relative paths from absolute paths
            image_files = [os.path.relpath(path, image_folder) for path in dataset.image_paths]
            mask_files = [os.path.relpath(path, mask_folder) for path in dataset.mask_paths]
            
            split_info[split_name] = {
                'images': image_files,
                'masks': mask_files,
                'count': len(image_files)
            }
        
        logger.info("Successfully extracted split information from data loaders")
        return split_info
        
    except Exception as e:
        logger.warning(f"Failed to extract split information from loaders: {e}")
        return None


def log_dataset_to_mlflow(
    training_data_info: Dict,
    split_file: Optional[str],
    mlflow_logger,
    train_loader=None,
    val_loader=None,
    test_loader=None,
    image_folder: Optional[str] = None,
    mask_folder: Optional[str] = None
) -> bool:
    """
    Main orchestration function to log dataset information to MLFlow.
    
    :param training_data_info: Dictionary with training data source information
    :param split_file: Path or S3 URI to split file
    :param mlflow_logger: MLFlow logger instance
    :param train_loader: Optional training data loader for fallback split extraction
    :param val_loader: Optional validation data loader for fallback split extraction
    :param test_loader: Optional test data loader for fallback split extraction
    :param image_folder: Optional image folder path for fallback split extraction
    :param mask_folder: Optional mask folder path for fallback split extraction
    :return: True if successful, False otherwise
    """
    try:
        # Extract S3 URIs from training_data_info
        input_data_config = training_data_info.get('input_data_config', {})
        train_image_s3_uri = input_data_config.get('train_image', {}).get('S3Uri')
        train_mask_s3_uri = input_data_config.get('train_mask', {}).get('S3Uri')
        
        # Extract dataset name
        dataset_name = extract_dataset_name(split_file, train_image_s3_uri)
        
        if not dataset_name:
            logger.warning("Could not determine dataset name. Skipping dataset tracking.")
            return False
        
        logger.info(f"Dataset name: {dataset_name}")
        
        # Load split information from split_file if available
        split_info = load_split_information(split_file)
        
        # Fallback: Extract split information from data loaders if split_file not available
        if not split_info and train_loader and val_loader and test_loader and image_folder and mask_folder:
            logger.info("Split file not available, extracting split information from data loaders")
            split_info = extract_split_from_loaders(
                train_loader, val_loader, test_loader, image_folder, mask_folder
            )
        
        # Prepare S3 URIs dictionary
        s3_uris = {
            'train_image': train_image_s3_uri,
            'train_mask': train_mask_s3_uri,
        }
        
        # Create MLFlow dataset and get metadata
        # Pass split_file so it can be used as the primary source URI if available
        dataset, metadata = create_mlflow_dataset(dataset_name, s3_uris, split_info, split_file)
        
        # Log dataset to current run using mlflow module-level function
        dataset_input = DatasetInput(
            dataset._to_mlflow_entity(),
            tags=[InputTag("context", "training")]
        )
        mlflow_logger.experiment.log_inputs(
            run_id=mlflow_logger.run_id,
            datasets=[dataset_input]
        )
        # mlflow.log_input(dataset=dataset, context="training")
        
        # Log dataset metadata as MLFlow parameters
        mlflow_logger.log_hyperparams(metadata)
        
        logger.info(f"Successfully logged dataset '{dataset_name}' to MLFlow")
        
        # Log split information as artifact if available
        if split_info:
            log_split_artifact(split_info, mlflow_logger)
            logger.info("Logged split information as artifact")
        
        return True
        
    except Exception as e:
        logger.warning(f"Failed to log dataset to MLFlow: {e}")
        logger.warning(traceback.format_exc())
        logger.warning("Training will continue without dataset tracking")
        return False


def _download_s3_uri(s3_uri: str) -> str:
    """
    Download file from S3 URI to local temporary location.
    
    :param s3_uri: S3 URI in format s3://bucket/key
    :return: Local file pathß
    """
    if not s3_uri.startswith("s3://"):
        return s3_uri
    
    bucket, key = s3_uri.removeprefix("s3://").split("/", 1)
    
    s3 = boto3.client("s3")
    local_file = f"/tmp/{Path(s3_uri).name}"
    s3.download_file(bucket, key, local_file)
    
    return local_file
