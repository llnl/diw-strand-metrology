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
    model = SegmentationLightningModule.load_from_checkpoint(os.path.join(model_dir, "last.ckpt"))
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
    return model, data_params


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
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
    mask = mask.detach().squeeze().numpy()
    mask = mask > data_params["THRESHOLD"]
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
