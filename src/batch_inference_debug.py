import json
import logging
import math
import sys
from contextlib import contextmanager
from time import time

import boto3
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import os
import yaml

from torchmetrics.classification import BinaryMatthewsCorrCoef
from torchmetrics.segmentation import DiceScore

from llnl_ml.lightning import SegmentationLightningModule
from llnl_ml.util import encode_image, image_bytes_to_numpy
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
    """
    Modify this function as needed to parse the incoming data.
    By default, we assume the image data is being passed in as a stream of bytes
    via the "application/x-image" content type, or a base64 encoded string with the "application/json"
    content_type.

    If the mask is also desired, modify the "application/json" content type to read in a dict with 2 keys
    pointing to s3 uris. For example, given the following input:
    {
        "mask": "s3://bucket-name/path/to/mask.png",
        "image": "s3://bucket-name/path/to/image.png"
    }

    if content_type == JSON_CONTENT_TYPE:
        body = json.loads(serialized_input_data)
        image_uri = body["image"]
        mask_uri = body["mask"]
        s3 = boto3.resource("s3")
        # Get image byte data
        bucket, key = image_uri.replace("s3://", "").split("/", 1)
        image_obj = s3.Object(bucket, key).get()
        image_bytes = image_obj["Body"].read()
        # repeat for mask

        mask_bytes = ...

        return (image_bytes, mask_bytes)

    Update the predict_fn to handle a tuple of inputs instead of a single input.

    :param serialized_input_data:
    :param content_type:
    :return:
    """
    try:  # Add this try
        if content_type == JSON_CONTENT_TYPE:
            content = json.loads(serialized_input_data)
            s3_resource = boto3.resource("s3")
            image = get_s3_object(s3_resource, content["image"])
            mask = get_s3_object(s3_resource, content["mask"])
            
            if image is None or mask is None:
                raise ValueError(f"Unable to load either mask or image in set: \n  {json.dumps(content, indent=2)}")
            
            return (image, mask)
        else:
            raise ValueError(f"Unexpected content type given: {content_type}")
    except Exception as e:
        logger.exception(f"Error in input_fn: {e}")  # log the exception with a traceback for debugging
        raise e  # Reraise the error so the function fails


def predict_fn(input_object, model, context):
    try:  # Add this try
        with log_timing("Inference"):
            model, data_params = model
            # Finish loading the image
            image_bytes, mask_bytes = input_object  # this is a tuple (image, mask) from input_fn

            # Load mask
            mask = image_bytes_to_numpy(mask_bytes)

            if np.max(mask) == 255 and np.min(mask) == 112:
                mask[mask == 112] = 0
                mask[mask == 255] = 1

            # Load image using transforms if available, otherwise use default_image_load
            if "transforms" in data_params:
                img = load_and_convert_image(image_bytes)
                transformed = data_params["transforms"](image=img, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
                image, mask = default_image_load(image_bytes, mask, data_params)

            # Add a batch dimension and move to device
            image = image.unsqueeze(0).to(device)
            mask_tensor = torch.tensor(mask, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
            # pass through model with targets so loss is computed
            with torch.no_grad():
                prediction, loss_dict = model(image, {"masks": mask_tensor})
    
        """
        Add your custom logic here. For example, calculating metrics, plotting images etc.
        It is recommended that any images are directly uploaded to S3 inside this function.
        Aggregate any numerical outputs in a dictionary that can be saved as a json file.
        """
        sig = nn.Sigmoid()
        prediction_display = sig(prediction).detach().squeeze().cpu().numpy()
        prediction = prediction.detach().squeeze().cpu().numpy()

        acc_metric = torchmetrics.Accuracy(task="binary", threshold=0.5, multidim_average="global")
        jaccard_metric = torchmetrics.JaccardIndex(task="binary", threshold=0.5, ignore_index=0)
        dice_metric = DiceScore(2, include_background=False, input_format="one-hot", average="macro")
        mathew_coer_metric = BinaryMatthewsCorrCoef(ignore_index=0, threshold=0.5)

        loss = sum(value.cpu().item() for value in loss_dict.values())
        pred_tensor = torch.tensor(prediction)
        mask_metric_tensor = torch.tensor(mask_tensor, dtype=torch.float)
        pred_mask_tensor = (torch.sigmoid(pred_tensor) > 0.5).long()
        mask_long_tensor = mask_metric_tensor.long()
        acc = float(acc_metric(pred_tensor, mask_metric_tensor))
        jac = float(jaccard_metric(pred_tensor, mask_metric_tensor))
        dice = float(dice_metric(pred_mask_tensor, mask_long_tensor))
        mathews_cc = float(mathew_coer_metric(pred_tensor, mask_metric_tensor))
    
        prediction_mask = ( prediction_display > 0.5 ).astype('uint8')
        #mask = mask.astype(np.uint8) * 255
        prediction_mask = (255 * prediction_mask).astype(np.uint8)
        _, prediction_mask_bytes = cv2.imencode(".png", prediction_mask)
        prediction_mask_bytes = prediction_mask_bytes.tobytes()
        prediction_mask_json_ready = encode_image(prediction_mask_bytes)
        
        _, mask_bytes = cv2.imencode(".png", (255 * mask).astype(np.uint8))
        mask_bytes = mask_bytes.tobytes()
        mask_json_ready = encode_image(mask_bytes)
            
        _, image_bytes = cv2.imencode(".png", (255 * image.detach().squeeze().cpu().numpy()).astype('uint8'))
        image_bytes = image_bytes.tobytes()
        image_json_ready = encode_image(image_bytes)
        
        _, prediction_bytes = cv2.imencode(".png", (255 * prediction_display).astype('uint8'))
        prediction_bytes = prediction_bytes.tobytes()
        prediction_json_ready = encode_image(prediction_bytes)        
        
        # next step, create dict.
        output_dict = dict(
            center_crop_size=data_params.get("center_crop_size"),
            center_crop_offset=data_params.get("center_crop_offset"),
            acc=acc,
            jac=jac,
            dice=dice,
            mathews_cc=mathews_cc,
            loss=loss,
            thresh=data_params["THRESHOLD"],
            pred_mask=prediction_mask_json_ready,
            image=image_json_ready,
            mask=mask_json_ready,
            pred=prediction_json_ready,
        )
        
        # Example, upload the mask to s3 rather than
        # s3_resource
        # return mask
        return output_dict
    except Exception as e:
        logger.exception(f"Error in predict_fn: {e}")  # log the exception with a traceback for debugging
        raise e  # Reraise the error so the function fails


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    try:  # Add this try
        # Mask is a numpy array in float32

        if accept == JSON_CONTENT_TYPE:
            # Convert to base64 encoded image string in json format
            result = json.dumps(prediction_output)
        elif accept == "application/x-image":
            # Return raw byte array
            result = prediction_output
        else:
            raise ValueError(f"Unsupported return content type. {accept} given.")
        return result
    except Exception as e:
        logger.exception(f"Error in output_fn: {e}")  # log the exception with a traceback for debugging
        raise e  # Reraise the error so the function fails


def default_image_load(image_bytes, mask, data_params):
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
        mask = mask[top:bottom, left:right]

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
    return torch.asarray(image).permute(2, 0, 1), mask