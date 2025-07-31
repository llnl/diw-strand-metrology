import argparse
import base64
import io
import json
import numpy as np
import zlib

from typing import Union, Callable, Any, List
from PIL import Image


def numpy_to_image_bytes(numpy_array: np.ndarray, format="PNG") -> bytes:
    """Convert a numpy image to encoded image format

    :param numpy_array: Numpy array of shape [H, W] or [H, W, 3]
    :param format: Image format to save bytes as. Default PNG

    :return: Encoded bytes of the image in the desired format
    """
    # Convert numpy array to PIL image
    image = Image.fromarray(numpy_array)

    # Create an in-memory bytes buffer
    buffer = io.BytesIO()

    # Save the image to the buffer
    image.save(buffer, format=format)
    return buffer.getvalue()


def image_bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Loads in memory encoded image to a numpy array

    :param image_bytes: The JPEG/PNG encoded image bytes
    :return: Numpy array of uncompressed image
    """
    # Create in-memory bytes buffer
    buffer = io.BytesIO(image_bytes)

    # Open the image from the buffer
    image = Image.open(buffer)

    # Convert to numpy array
    numpy_array = np.array(image)
    return numpy_array


def encode_image(image_raw: bytes, compress: bool = False) -> str:
    """Generates an encoded and potentially compressed string from an image

    :param image_raw: the raw bytes of the image file
    :param compress: whether or not to gzip the encoded string

    :return: The encoded and compressed string representing the image
    """
    if compress:
        image_compressed = zlib.compress(image_raw)
    else:
        image_compressed = image_raw

    image_b64 = base64.encodebytes(image_compressed).decode("ascii")

    return image_b64


def decode_image(image_encoded: str) -> bytes:
    """
    Decodes an encoded and potentially compressed string into the raw bytes of an image

    :param image_encoded: the encoded and compressed string representing the image

    :return: the original bytes of the raw image
    """
    image_raw_compressed = base64.b64decode(image_encoded)

    try:
        image_raw = zlib.decompress(image_raw_compressed)
    except zlib.error:
        image_raw = image_raw_compressed

    return image_raw


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


def str2list(string: str, dtype: Union[type, Callable[[str], Any]] = float) -> List[Any]:
    """
    Converts a json formatted list ([1, 2, 3]) or
    comma separated list (1.0,2.0,3.0) to a list of the desired data type

    For example
        str2list('[1, 2, 3]', int) -> [1, 2, 3]
        str2list('1, 2, 3', int) -> [1, 2, 3]
        str2list('1,2,3', float) -> [1.0, 2.0, 3.0]
        str2list('1,2,3', str) -> ['1', '2', '3']
        str2list('True, False, False', str2bool) -> [True, False, False]
    Parameters
    ----------
    string: str
        Input string to convert
    dtype: type or Callable
        Data type to convert values to

    Returns
    -------
    data_list: list of dtypes
        list of the input string converted to the desired data type
    """
    if string.startswith("["):
        parsed = json.loads(string)
    else:
        parsed = [v.strip() for v in string.split(",")]
    return [dtype(v) for v in parsed]


# Helper functions for specific types
def str2floatlist(string: str) -> List[float]:
    return str2list(string, float)


def str2intlist(string: str) -> List[int]:
    return str2list(string, int)


def str2strlist(string: str) -> List[str]:
    return str2list(string, str)


def str2boollist(string: str) -> List[bool]:
    return str2list(string, str2bool)
