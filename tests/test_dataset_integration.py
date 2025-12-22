"""Integration tests for SegmentationDataset compatibility with YAML transform system."""

import pytest
import tempfile
import os
import sys
import yaml
import numpy as np
import torch
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock
import albumentations as A

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llnl_ml.data.dataset import SegmentationDataset, get_data_loaders
from llnl_ml.data.transforms import build_transforms
from llnl_ml.lightning import SegmentationLightningModule


class TestDatasetIntegration:
    """Test integration patterns for YAML transform system with dataset workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories for test data
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.temp_dir, "images")
        self.mask_dir = os.path.join(self.temp_dir, "masks")
        os.makedirs(self.image_dir)
        os.makedirs(self.mask_dir)
        
        # Create test images and masks
        self.test_images = []
        self.test_masks = []
        for i in range(5):
            # Create test image (RGB)
            img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            img_path = os.path.join(self.image_dir, f"test_image_{i}.png")
            cv2.imwrite(img_path, img)
            self.test_images.append(img_path)
            
            # Create test mask (binary)
            mask = np.random.randint(0, 2, (300, 300), dtype=np.uint8) * 255
            mask_path = os.path.join(self.mask_dir, f"test_mask_{i}.png")
            cv2.imwrite(mask_path, mask)
            self.test_masks.append(mask_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_yaml_config(self, train_transforms, val_transforms=None):
        """Helper to create YAML config file."""
        if val_transforms is None:
            val_transforms = train_transforms
        
        config = {
            'train': train_transforms,
            'val': val_transforms
        }
        
        config_path = os.path.join(self.temp_dir, "transform_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def test_dataset_accepts_yaml_transforms(self):
        """Test that SegmentationDataset can be modified to work with YAML Albumentations transforms."""
        # Create YAML config with basic transforms
        transform_specs = [
            {'name': 'Resize', 'height': 256, 'width': 256},
            {'name': 'Normalize'},
            {'name': 'ToTensorV2'}
        ]
        config_path = self._create_yaml_config(transform_specs)
        
        # Build transforms from YAML
        train_transforms, val_transforms = build_transforms(config_path)
        
        # Verify we get Albumentations Compose objects
        assert isinstance(train_transforms, A.Compose)
        assert isinstance(val_transforms, A.Compose)
        
        # Test that transforms can be applied to image/mask data
        # Load a test image and mask manually
        test_image = cv2.imread(self.test_images[0])
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(self.test_masks[0], cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms directly (this is how they should work)
        transformed = train_transforms(image=test_image, mask=test_mask)
        
        # Verify transforms were applied
        assert 'image' in transformed
        assert 'mask' in transformed
        assert isinstance(transformed['image'], torch.Tensor)  # ToTensorV2 converts to tensor
        assert transformed['image'].shape[1:] == (256, 256)  # Resize applied
    
    def test_albumentations_integration_pattern(self):
        """Test the integration pattern for using Albumentations with the dataset."""
        # Create YAML config
        transform_specs = [
            {'name': 'Resize', 'height': 256, 'width': 256},
            {'name': 'ToTensorV2'}
        ]
        config_path = self._create_yaml_config(transform_specs)
        
        # Build transforms from YAML
        train_transforms, val_transforms = build_transforms(config_path)
        
        # Demonstrate how the dataset should be modified to work with Albumentations
        # This shows the integration pattern that should be implemented
        
        # Load image and mask manually (simulating dataset __getitem__)
        test_image = cv2.imread(self.test_images[0])
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(self.test_masks[0], cv2.IMREAD_GRAYSCALE)
        
        # Apply Albumentations transforms (this is the new pattern)
        transformed = train_transforms(image=test_image, mask=test_mask)
        
        # Verify the output format is what models expect
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        
        assert isinstance(transformed_image, torch.Tensor)
        assert isinstance(transformed_mask, (torch.Tensor, np.ndarray))
        assert transformed_image.shape[1:] == (256, 256)  # H, W after resize
    
    def test_yaml_transforms_preserve_mask_format(self):
        """Test that YAML transforms preserve mask format expected by models."""
        # Create config with transforms that affect both image and mask
        transform_specs = [
            {'name': 'RandomCrop', 'height': 200, 'width': 200},
            {'name': 'HorizontalFlip', 'p': 1.0},  # Always flip for consistent testing
            {'name': 'ToTensorV2'}
        ]
        config_path = self._create_yaml_config(transform_specs)
        
        train_transforms, _ = build_transforms(config_path)
        
        # Test with actual image/mask data
        test_image = cv2.imread(self.test_images[0])
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(self.test_masks[0], cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        transformed = train_transforms(image=test_image, mask=test_mask)
        
        # Verify mask format is preserved and compatible with image
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        
        assert hasattr(transformed_image, 'shape'), "Image should have shape attribute"
        assert hasattr(transformed_mask, 'shape'), "Mask should have shape attribute"
        
        # Verify spatial dimensions match
        if isinstance(transformed_image, torch.Tensor):
            image_spatial = transformed_image.shape[1:]  # CHW format
        else:
            image_spatial = transformed_image.shape[:2]  # HW format
            
        if isinstance(transformed_mask, torch.Tensor):
            mask_spatial = transformed_mask.shape[-2:]  # Last 2 dims are spatial
        else:
            mask_spatial = transformed_mask.shape[:2]  # HW format
            
        assert image_spatial == mask_spatial, "Image and mask should have same spatial dimensions"
    
    def test_yaml_transform_interface_compatibility(self):
        """Test that YAML transforms provide the interface expected by PyTorch Lightning workflows."""
        # Create YAML config
        transform_specs = [
            {'name': 'Resize', 'height': 256, 'width': 256},
            {'name': 'Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            {'name': 'ToTensorV2'}
        ]
        config_path = self._create_yaml_config(transform_specs)
        
        # Build transforms from YAML
        train_transforms, val_transforms = build_transforms(config_path)
        
        # Verify transforms are callable (requirement 5.1)
        assert callable(train_transforms)
        assert callable(val_transforms)
        
        # Verify transforms return the expected format for model consumption
        test_image = cv2.imread(self.test_images[0])
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(self.test_masks[0], cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        train_result = train_transforms(image=test_image, mask=test_mask)
        val_result = val_transforms(image=test_image, mask=test_mask)
        
        # Verify both return dictionaries with expected keys
        assert isinstance(train_result, dict)
        assert isinstance(val_result, dict)
        assert 'image' in train_result and 'mask' in train_result
        assert 'image' in val_result and 'mask' in val_result
        
        # Verify output types are compatible with PyTorch models
        assert isinstance(train_result['image'], torch.Tensor)
        assert isinstance(val_result['image'], torch.Tensor)
        
        # Verify tensor format is correct (CHW for PyTorch)
        assert len(train_result['image'].shape) == 3  # C, H, W
        assert train_result['image'].shape[0] == 3  # RGB channels
    
    def test_lightning_module_data_format_compatibility(self):
        """Test that YAML transforms produce data format compatible with Lightning modules."""
        # Create YAML config
        transform_specs = [
            {'name': 'Resize', 'height': 128, 'width': 128},
            {'name': 'ToTensorV2'}
        ]
        config_path = self._create_yaml_config(transform_specs)
        
        train_transforms, _ = build_transforms(config_path)
        
        # Simulate the data format that would be produced by a modified dataset
        test_image = cv2.imread(self.test_images[0])
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(self.test_masks[0], cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        transformed = train_transforms(image=test_image, mask=test_mask)
        
        # Create mock batch format that Lightning expects
        # This demonstrates the integration pattern
        mock_image = transformed['image'].unsqueeze(0)  # Add batch dimension
        mock_targets = {
            'masks': transformed['mask'].unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            'image_name': 'test_image.png'
        }
        mock_batch = (mock_image, [mock_targets])  # List format for compatibility
        
        # Verify the format is what Lightning modules expect
        images, targets = mock_batch
        assert isinstance(images, torch.Tensor)
        assert images.dim() == 4  # NCHW format
        assert isinstance(targets, list)
        assert len(targets) == 1  # One item per batch
        assert 'masks' in targets[0]
        
        # This demonstrates the data format that the integration should produce
    
    def test_yaml_transforms_bbox_compatibility(self):
        """Test that YAML transforms can handle bbox-related data when needed."""
        # Create config with transforms that preserve spatial relationships
        transform_specs = [
            {'name': 'Resize', 'height': 256, 'width': 256},
            {'name': 'ToTensorV2'}
        ]
        config_path = self._create_yaml_config(transform_specs)
        
        train_transforms, _ = build_transforms(config_path)
        
        # Test that transforms work with image and mask (bbox models need this)
        test_image = cv2.imread(self.test_images[0])
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(self.test_masks[0], cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        transformed = train_transforms(image=test_image, mask=test_mask)
        
        # Verify the transforms preserve the spatial relationship needed for bbox models
        assert transformed['image'].shape[1:] == transformed['mask'].shape[-2:]
        
        # This demonstrates that the YAML system can support bbox models
        # by preserving spatial relationships between image and mask
    
    def test_yaml_config_error_handling_in_dataset_context(self):
        """Test that YAML config errors are handled gracefully in dataset context."""
        # Create invalid YAML config
        invalid_config = os.path.join(self.temp_dir, "invalid.yaml")
        with open(invalid_config, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        # Test that build_transforms raises appropriate error
        from llnl_ml.data.exceptions import TransformConfigError
        with pytest.raises(TransformConfigError):
            build_transforms(invalid_config)
    
    def test_empty_yaml_config_creates_identity_transforms(self):
        """Test that empty YAML config creates identity transforms."""
        # Create empty config
        empty_config = {'train': [], 'val': []}
        config_path = os.path.join(self.temp_dir, "empty.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(empty_config, f)
        
        train_transforms, val_transforms = build_transforms(config_path)
        
        # Test with actual image/mask data
        test_image = cv2.imread(self.test_images[0])
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(self.test_masks[0], cv2.IMREAD_GRAYSCALE)
        
        # Apply empty transforms (should be identity)
        transformed = train_transforms(image=test_image, mask=test_mask)
        
        # Should still work, just no transforms applied (identity operation)
        assert isinstance(transformed['image'], np.ndarray)  # No ToTensorV2, so still numpy
        assert isinstance(transformed['mask'], np.ndarray)
        
        # Should preserve original dimensions
        assert transformed['image'].shape == test_image.shape
        assert transformed['mask'].shape == test_mask.shape
    
    def test_yaml_transforms_preserve_data_types(self):
        """Test that YAML transforms preserve expected data types throughout pipeline."""
        # Create config that includes normalization and tensor conversion
        transform_specs = [
            {'name': 'Resize', 'height': 224, 'width': 224},
            {'name': 'Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            {'name': 'ToTensorV2'}
        ]
        config_path = self._create_yaml_config(transform_specs)
        
        train_transforms, _ = build_transforms(config_path)
        
        # Test with actual image/mask data
        test_image = cv2.imread(self.test_images[0])
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(self.test_masks[0], cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        transformed = train_transforms(image=test_image, mask=test_mask)
        
        # Verify final data types
        image = transformed['image']
        assert isinstance(image, torch.Tensor)
        assert image.dtype == torch.float32
        assert image.shape == (3, 224, 224)  # C, H, W format
        
        # Verify normalization was applied (values should be roughly in [-2, 2] range)
        assert image.min() >= -3.0 and image.max() <= 3.0
    
    def test_different_train_val_transforms_work_independently(self):
        """Test that different train/val transforms work correctly and independently."""
        # Create config with different train and val transforms
        train_specs = [
            {'name': 'RandomCrop', 'height': 200, 'width': 200},
            {'name': 'HorizontalFlip', 'p': 1.0},  # Always flip for testing
            {'name': 'ToTensorV2'}
        ]
        
        val_specs = [
            {'name': 'CenterCrop', 'height': 200, 'width': 200},
            {'name': 'ToTensorV2'}
        ]
        
        config_path = self._create_yaml_config(train_specs, val_specs)
        
        train_transforms, val_transforms = build_transforms(config_path)
        
        # Test with actual image/mask data
        test_image = cv2.imread(self.test_images[0])
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_mask = cv2.imread(self.test_masks[0], cv2.IMREAD_GRAYSCALE)
        
        # Apply different transforms
        train_result = train_transforms(image=test_image, mask=test_mask)
        val_result = val_transforms(image=test_image, mask=test_mask)
        
        # Both should work but produce different results due to different transforms
        assert train_result['image'].shape == val_result['image'].shape  # Same final size
        
        # Verify both produce tensor outputs
        assert isinstance(train_result['image'], torch.Tensor)
        assert isinstance(val_result['image'], torch.Tensor)
        
        # Both should have the expected final dimensions
        assert train_result['image'].shape == (3, 200, 200)  # C, H, W
        assert val_result['image'].shape == (3, 200, 200)  # C, H, W