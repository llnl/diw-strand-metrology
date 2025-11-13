import json
import math

import cv2
import torch
import os
import numpy as np
import yaml
import torchmetrics
import torch.nn as nn
import boto3

from llnl_ml.lightning import SegmentationLightningModule
from llnl_ml.util import decode_image, encode_image, image_bytes_to_numpy


###### Required for debuging ############
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
###### End Required for debuging ############


JSON_CONTENT_TYPE = "application/json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_s3_object(s3_resource, s3_uri: str):
    try:  # Add this try
        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        try:
            obj = s3_resource.Object(bucket, key).get()
            return obj["Body"].read()
        except s3_resource.meta.client.exceptions.NoSuchKey:
            return None
    except Exception as e:
        logging.exception(e) # log the exception with a traceback for debugging
        raise(e) # Reraise the error so the function fails


def model_fn(model_dir, context):
    try:  # Add this try
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
    except Exception as e:
        logging.exception(e) # log the exception with a traceback for debugging
        raise(e) # Reraise the error so the function fails


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
        
        
        
        '''if content_type == JSON_CONTENT_TYPE:
            body = json.loads(serialized_input_data)
            if isinstance(body, dict):
                encoded_image_str = body["input"]
            else:
                encoded_image_str = body
            image_bytes = decode_image(encoded_image_str)
        elif content_type == "application/x-image":
            image_bytes = serialized_input_data
        else:
            raise ValueError(f"Unsupported input content type. {content_type} given.")'''
    
        #return image_bytes
    except Exception as e:
        logging.exception(e) # log the exception with a traceback for debugging
        raise(e) # Reraise the error so the function fails


def predict_fn(input_object, model, context):
    try:  # Add this try
        model, data_params = model
        # Finish loading the image
        image_bytes, mask_bytes = input_object  # this is a tuple (image, mask) from input_fn
        image_byte_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_byte_array, cv2.IMREAD_ANYDEPTH | data_params["READ_MODE"])
        # Load mask
        #mask_byte_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        #mask = cv2.imdecode(mask_byte_array, cv2.IMREAD_ANYDEPTH | data_params["READ_MODE"])
        mask = image_bytes_to_numpy(mask_bytes)

        if np.max(mask)==255 and np.min(mask)==112:
            mask[mask==112]=0
            mask[mask==255]=1

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
        image = image.astype(dtype=np.float32) / np.iinfo(image.dtype).max
    
        # Convert from HWC to CHW
        image = torch.asarray(image).permute(2, 0, 1)
    
        # Add a batch dimension
        image = image.unsqueeze(0)
        # pass through model
        prediction = model(image)
    
        """
        Add your custom logic here. For example, calculating metrics, plotting images etc.
        It is recommended that any images are directly uploaded to S3 inside this function.
        Aggregate any numerical outputs in a dictionary that can be saved as a json file.
        """
        sig = nn.Sigmoid()
        prediction_display = sig(prediction).detach().squeeze().numpy()
        prediction = prediction.detach().squeeze().numpy()

        loss_fn = torch.nn.BCEWithLogitsLoss()
        acc_metric = torchmetrics.classification.BinaryAccuracy(threshold=0.5, multidim_average="global")
        jaccard_metric = torchmetrics.JaccardIndex(task="binary", threshold=0.5, ignore_index=0)
        dice_metric = torchmetrics.Dice(threshold=0.5)
        
        loss = float(loss_fn(torch.tensor(prediction), torch.tensor(mask, dtype=float)))
        acc = float(acc_metric(torch.tensor(prediction), torch.tensor(mask, dtype=float)))
        jac = float(jaccard_metric(torch.tensor(prediction), torch.tensor(mask, dtype=float)))
        dice = float(dice_metric(torch.tensor(prediction), torch.tensor(mask)))
    
        prediction_mask = ( prediction_display > 0.5 ).astype('uint8')
        #mask = mask.astype(np.uint8) * 255
        prediction_mask = (255 * prediction_mask).astype(np.uint8)
        _, prediction_mask_bytes = cv2.imencode(".png", prediction_mask)
        prediction_mask_bytes = prediction_mask_bytes.tobytes()
        prediction_mask_json_ready = encode_image(prediction_mask_bytes)
        
        _, mask_bytes = cv2.imencode(".png", (255 * mask).astype(np.uint8))
        mask_bytes = mask_bytes.tobytes()
        mask_json_ready = encode_image(mask_bytes)
            
        _, image_bytes = cv2.imencode(".png", (255 * image.detach().squeeze().numpy()).astype('uint8'))
        image_bytes = image_bytes.tobytes()
        image_json_ready = encode_image(image_bytes)
        
        _, prediction_bytes = cv2.imencode(".png", (255 * prediction_display).astype('uint8'))
        prediction_bytes = prediction_bytes.tobytes()
        prediction_json_ready = encode_image(prediction_bytes)        
        
        # next step, create dict.  
        output_dict = dict(center_crop_size = crop_size, 
                           center_crop_offset = crop_offset,
                           acc = acc, jac = jac, dice = dice, loss = loss,
                           thresh = data_params["THRESHOLD"],
                           pred_mask = prediction_mask_json_ready, 
                           image = image_json_ready, mask = mask_json_ready,
                           pred = prediction_json_ready)
        
        # Example, upload the mask to s3 rather than
        # s3_resource
        # return mask
        return output_dict 
    except Exception as e:
        logging.exception(e) # log the exception with a traceback for debugging
        raise(e) # Reraise the error so the function fails


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    try:  # Add this try
        # Mask is a numpy array in float32
    
        if accept == JSON_CONTENT_TYPE:
            # Convert to base64 encoded image string in json format
            result = json.dumps( prediction_output )
        elif accept == "application/x-image":
            # Return raw byte array
            result = mask_bytes
        else:
            raise ValueError(f"Unsupported return content type. {accept} given.")
        return result
    except Exception as e:
        logging.exception(e) # log the exception with a traceback for debugging
        raise(e) # Reraise the error so the function fails