"""
Unit tests for backward compatibility of the transform system.

Tests that the new YAML-based transform system maintains compatibility
while properly deprecating old CLI arguments.
"""

import pytest
import sys
import os
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import parse_args


class TestBackwardCompatibility:
    """Test backward compatibility of the transform system."""
    
    def setup_method(self):
        """Set up test environment."""
        # Set required environment variables
        os.environ['SM_CHANNEL_TRAIN_IMAGE'] = 'data'
        os.environ['SM_CHANNEL_TRAIN_MASK'] = 'data'
    
    def test_deprecated_arguments_raise_error(self):
        """Test that deprecated transform arguments raise ValueError."""
        deprecated_args = [
            ['--use_random_resize', 'True'],
            ['--use_random_crop', 'True'],
            ['--use_random_rotation', 'True'],
            ['--color_jitter', '0.5'],
            ['--image_size', '256'],
            ['--val_image_size', '512'],
            ['--test_image_size', '512'],
            ['--center_crop', 'True'],
            ['--center_crop_size', '800'],
            ['--center_crop_offset', '[-30,-40]']
        ]
        
        for deprecated_arg in deprecated_args:
            with patch.object(sys, 'argv', ['train.py'] + deprecated_arg):
                with pytest.raises(ValueError) as exc_info:
                    parse_args()
                
                # Check that the error message mentions the deprecated argument
                error_msg = str(exc_info.value)
                assert "deprecated" in error_msg.lower()
                assert deprecated_arg[0] in error_msg
                assert "transform_config" in error_msg
    
    def test_new_transform_config_argument_works(self):
        """Test that the new transform_config argument works correctly."""
        with patch.object(sys, 'argv', ['train.py', '--transform_config', 'test_config.yaml']):
            args, model_params = parse_args()
            assert args.transform_config == 'test_config.yaml'
    
    def test_empty_transform_config_works(self):
        """Test that empty transform_config (default) works."""
        with patch.object(sys, 'argv', ['train.py']):
            args, model_params = parse_args()
            assert args.transform_config == ''
    
    def test_multiple_deprecated_args_in_error(self):
        """Test that multiple deprecated arguments are all listed in error."""
        with patch.object(sys, 'argv', ['train.py', '--use_random_resize', 'True', '--image_size', '256']):
            with pytest.raises(ValueError) as exc_info:
                parse_args()
            
            error_msg = str(exc_info.value)
            assert '--use_random_resize' in error_msg
            assert '--image_size' in error_msg
    
    def test_non_deprecated_args_still_work(self):
        """Test that non-deprecated arguments still work correctly."""
        with patch.object(sys, 'argv', [
            'train.py', 
            '--batch_size', '4',
            '--epochs', '10',
            '--learning_rate', '0.01',
            '--transform_config', 'my_config.yaml'
        ]):
            args, model_params = parse_args()
            assert args.batch_size == 4
            assert args.epochs == 10
            assert args.learning_rate == 0.01
            assert args.transform_config == 'my_config.yaml'


if __name__ == '__main__':
    pytest.main([__file__])