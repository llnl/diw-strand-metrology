import boto3
import json
import logging
import math
import sys
from contextlib import contextmanager
from time import time

import cv2
import torch
import os
import numpy as np
import yaml

from llnl_ml.lightning import SegmentationLightningModule
from llnl_ml.util import decode_image, encode_image
from llnl_ml.data.transforms import build_transforms
from llnl_ml.data.utils import load_and_convert_image


JSON_CONTENT_TYPE = "application/json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}") 
    logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")


def get_s3_object(s3_resource, s3_uri: str):
    try:  # Add this try
        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        try:
            obj = s3_resource.Object(bucket, key).get()
            return obj["Body"].read()
        except s3_resource.meta.client.exceptions.NoSuchKey:
            return None
    except Exception as e:
        logger.exception(e)  # log the exception with a traceback for debugging
        raise (e)  # Reraise the error so the function fails


@contextmanager
def log_timing(operation_name: str):
    start_time = time()
    try:
        yield
    finally:
        elapsed_time = time() - start_time
        logger.info(f"{operation_name} completed in {elapsed_time:.4f} seconds")


def model_fn(model_dir, context):
    try:  # Add this try
        logger.info(f"Loading model from directory: {model_dir}")
        model_name = "best.ckpt" if os.path.exists(os.path.join(model_dir, "best.ckpt")) else "last.ckpt"
        logger.info(f"Using model checkpoint: {model_name}")

        model = SegmentationLightningModule.load_from_checkpoint(os.path.join(model_dir, model_name))
        model.to(device).eval()

        logger.info(f"Model loaded successfully on to device {device}")

        # Load the config file if present
        if os.path.exists(os.path.join(model_dir, "config.yaml")):
            logger.info("Loading configuration file")
            with open(os.path.join(model_dir, "config.yaml"), "r") as fp:
                data_params = yaml.safe_load(fp)
            data_params["READ_MODE"] = cv2.IMREAD_COLOR if data_params["image_mode"] == "RGB" else cv2.IMREAD_GRAYSCALE
            data_params["THRESHOLD"] = 0.5 if data_params["model_name"] == "MaskRCNN" else 0.0
            logger.info(f"Configuration loaded: {data_params}")
            if "transform_config" in data_params:
                _, data_params["transforms"] = build_transforms(data_params["transform_config"])
                logger.info(f"Loaded the transform_config file: {data_params['transform_config']}")
        # If config doesn't exist, get expected input shape from model and use default threshold
        else:
            logger.warning("No config file found, using default parameters")
            # Check what the expected input size is
            first_param = next(model.parameters())
            input_channels = first_param.size()[1]
            data_params = {
                "READ_MODE": cv2.IMREAD_GRAYSCALE if input_channels == 1 else cv2.IMREAD_COLOR,
                "THRESHOLD": 0.5,
            }
            logger.info(f"Configuration loaded: {data_params}")
        # Use Environment Variables to load any parameters to be used later on.
        # For example, thresholds for applying  to masking, metric calculation styles, etc.

        return model, data_params
    except Exception as e:
        logger.exception(f"Error in model_fn: {e}")  # log the exception with a traceback for debugging
        raise e  # Reraise the error so the function fails


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    try:
        if content_type == JSON_CONTENT_TYPE:
            body = json.loads(serialized_input_data)
            if isinstance(body, dict):
                if "input" in body:
                    image_bytes = decode_image(body["input"])
                elif "image" in body:
                    s3_resource = boto3.resource("s3")
                    image_bytes = get_s3_object(s3_resource, body["image"])
                    if image_bytes is None:
                        raise ValueError(f"Image not found in S3: {body['image']}")
                else:
                    raise ValueError(f"Invalid input format. {body.keys()} given.")
            else:
                image_bytes = decode_image(body)
        elif content_type == "application/x-image":
            image_bytes = serialized_input_data
        else:
            raise ValueError(f"Unsupported input content type. {content_type} given.")

        return image_bytes
    except Exception as e:
        logger.exception(f"Error in input_fn: {e}")
        raise e


def predict_fn(input_object, model, context):
    try:
        with log_timing("Inference"):
            model, data_params = model
            # Finish loading the image
            if "transforms" in data_params:
                img = load_and_convert_image(input_object)
                image = data_params["transforms"](image=img)["image"]
            else:
                image = default_image_load(input_object, data_params)

            # Add a batch dimension and move to device
            image = image.unsqueeze(0).to(device)
            # pass through model
            with torch.no_grad():
                mask, _ = model(image)
            mask = mask.detach().squeeze().cpu().numpy()
            mask = mask > data_params["THRESHOLD"]
            # return mask
            return mask
    except Exception as e:
        logger.exception(f"Error in predict_fn: {e}")
        raise e


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    try:
        # Mask is a numpy array in float32
        mask = prediction_output.astype(np.uint8) * 255
        _, mask_bytes = cv2.imencode(".png", mask)
        mask_bytes = mask_bytes.tobytes()
        if accept == JSON_CONTENT_TYPE:
            # Convert to base64 encoded image string in json format
            result = json.dumps(encode_image(mask_bytes))
        elif accept == "application/x-image":
            # Return raw byte array
            result = mask_bytes
        else:
            raise ValueError(f"Unsupported return content type. {accept} given.")
        return result
    except Exception as e:
        logger.exception(f"Error in output_fn: {e}")
        raise e


def default_image_load(image_bytes, data_params):
    image_byte_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_byte_array, cv2.IMREAD_ANYDEPTH | data_params["READ_MODE"])
    
    if image is None:
        raise ValueError("Failed to decode image bytes. Invalid image format or corrupted data.")
    
    if data_params["READ_MODE"] == cv2.IMREAD_COLOR:
        # Convert from BGR to RGB
        image = image[:, :, ::-1]
    else:
        # Add a Channel dim to front
        image = image[:, :, np.newaxis]

    if data_params.get("center_crop", False):
        crop_size = data_params["center_crop_size"]
        crop_offset = data_params["center_crop_offset"]
        top = (image.shape[0] // 2 + crop_offset[0]) - math.floor(crop_size / 2)
        bottom = (image.shape[0] // 2 + crop_offset[0]) + math.ceil(crop_size / 2)
        left = (image.shape[1] // 2 + crop_offset[1]) - math.floor(crop_size / 2)
        right = (image.shape[1] // 2 + crop_offset[1]) + math.ceil(crop_size / 2)
        image = image[top:bottom, left:right]

    # Convert to torch tensor of correct shape and normalize
    # Handle normalization based on image data type
    if image.dtype == np.uint8:
        max_val = 255.0
    elif image.dtype == np.uint16:
        max_val = 65535.0
    elif np.issubdtype(image.dtype, np.integer):
        # For other integer types, use a safe approach
        max_val = float(image.max()) if image.size > 0 else 1.0
    else:
        max_val = 1.0
    
    image = image.astype(np.float32) / max_val

    # Convert from HWC to CHW
    return torch.asarray(image).permute(2, 0, 1)