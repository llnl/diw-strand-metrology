import json
import io
import numpy as np
from PIL import Image
from typing import Dict, Sequence, Union


def convert_image_to_normalized_array(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to normalized numpy array in range [0, 1].
    
    Handles:
    - RGB mode (uint8): Standard 8-bit RGB images
    - I;16 mode (uint16): 16-bit grayscale images
    - Other modes: Converts to RGB first
    
    Args:
        image: PIL Image object
        
    Returns:
        numpy array with values in range [0, 1], shape (H, W, C) for RGB or (H, W) for grayscale
    """
    if image.mode == 'RGB':
        # Standard 8-bit RGB
        array = np.array(image, dtype=np.float32)
        return array / 255.0
    
    elif image.mode == 'I;16':
        # 16-bit unsigned integer
        array = np.array(image, dtype=np.float32)
        return array / 65535.0  # 2^16 - 1
    
    elif image.mode in ['L', 'I']:
        # 8-bit or 32-bit grayscale
        array = np.array(image, dtype=np.float32)
        max_val = 255.0 if image.mode == 'L' else array.max()
        return array / max_val if max_val > 0 else array
    
    else:
        # Convert other modes to RGB first, then normalize
        rgb_image = image.convert('RGB')
        array = np.array(rgb_image, dtype=np.float32)
        return array / 255.0


def load_and_convert_image(image_input: Union[str, bytes]) -> np.ndarray:
    """
    Load image from path or bytes and convert to normalized array.
    
    Args:
        image_input: Either a file path (str) or image bytes (bytes)
        
    Returns:
        Normalized numpy array with values in range [0, 1]
    """
    if isinstance(image_input, bytes):
        # Handle image bytes (e.g., from REST API payload)
        with Image.open(io.BytesIO(image_input)) as img:
            return convert_image_to_normalized_array(img)
    else:
        # Handle file path (existing behavior)
        with Image.open(image_input) as img:
            return convert_image_to_normalized_array(img)


def load_json_file(filename: str) -> Dict[str, Dict[str, Sequence]]:
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