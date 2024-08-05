import json
import math

import cv2
import torch
import os
import numpy as np
import yaml

from llnl_ml.lightning import SegmentationLightningModule
from llnl_ml.util import decode_image, encode_image


JSON_CONTENT_TYPE = "application/json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir, context):
    model_name = "best.ckpt" if os.path.exists(os.path.join(model_dir, "best.ckpt")) else "last.ckpt"
    model = SegmentationLightningModule.load_from_checkpoint(os.path.join(model_dir, model_name))
    model.to(device).eval()

    # Load the config file if present
    if os.path.exists(os.path.join(model_dir, "config.yaml")):
        with open(os.path.join(model_dir, "config.yaml"), "r") as fp:
            data_params = yaml.safe_load(fp)
        data_params["READ_MODE"] = cv2.IMREAD_COLOR if data_params["image_mode"] == "RGB" else cv2.IMREAD_GRAYSCALE
        data_params["THRESHOLD"] = 0.5 if data_params["model_name"] == "MaskRCNN" else 0.0
    # If config doesn't exist, get expected input shape from model and use default threshold
    else:
        # Check what the expected input size is
        first_param = next(model.parameters())
        input_channels = first_param.size()[1]
        data_params = {
            "READ_MODE": cv2.IMREAD_GRAYSCALE if input_channels == 1 else cv2.IMREAD_COLOR,
            "THRESHOLD": 0.5,
        }

    # Use Environment Variables to load any parameters to be used later on.
    # For example, thresholds for applying  to masking, metric calculation styles, etc.

    return model, data_params


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
    if content_type == JSON_CONTENT_TYPE:
        body = json.loads(serialized_input_data)
        if isinstance(body, dict):
            encoded_image_str = body["input"]
        else:
            encoded_image_str = body
        image_bytes = decode_image(encoded_image_str)
    elif content_type == "application/x-image":
        image_bytes = serialized_input_data
    else:
        raise ValueError(f"Unsupported input content type. {content_type} given.")

    return image_bytes


def predict_fn(input_object, model, context):
    model, data_params = model
    # Finish loading the image
    image_byte_array = np.frombuffer(input_object, dtype=np.uint8)
    image = cv2.imdecode(image_byte_array, cv2.IMREAD_ANYDEPTH | data_params["READ_MODE"])
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
    image = image.astype(dtype=np.float32) / np.iinfo(image.dtype).max

    # Convert from HWC to CHW
    image = torch.asarray(image).permute(2, 0, 1)

    # Add a batch dimension
    image = image.unsqueeze(0)
    # pass through model
    mask = model(image)

    """
    Add your custom logic here. For example, calculating metrics, plotting images etc.
    It is recommended that any images are directly uploaded to S3 inside this function.
    Aggregate any numerical outputs in a dictionary that can be saved as a json file.
    """
    mask = mask.detach().squeeze().numpy()
    mask = mask > data_params["THRESHOLD"]

    # Example, upload the mask to s3 rather than
    # s3_resource
    # return mask
    return mask


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
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
