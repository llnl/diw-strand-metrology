# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:28:45 2024

Use this code for applying a trained image segmentation algorithm to do on- 
machine inspection.

@author: zelinski1
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import v2 as transforms
from llnl_ml.lightning import SegmentationLightningModule
from llnl_ml.util import decode_image, encode_image
import torchvision

if int(torchvision.__version__.split(".")[1]) < 16:
    from torchvision.datapoints import Mask as TVMask, BoundingBox as TVBBox
    from torchvision.transforms.v2.functional import to_image_tensor
    from torchvision.transforms.v2 import SanitizeBoundingBox

    CANVAS_SHAPE = "spatial_size"
else:
    from torchvision.tv_tensors import Mask as TVMask, BoundingBoxes as TVBBox
    from torchvision.transforms.v2.functional import to_image as to_image_tensor
    from torchvision.transforms.v2 import SanitizeBoundingBoxes as SanitizeBoundingBox

    CANVAS_SHAPE = "canvas_size"

    

def om_inference(     
    model_dir, 
    model_name, 
    image_dir, 
    output_dir, 
    crop, crop_size, 
    crop_offset, 
    image_size ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Specify transmforms, these are the transforms used in the Jan 26 2024 results
    eval_transforms = []
    eval_transforms.append(transforms.Resize(image_size, antialias=True))
    eval_transforms.append(transforms.ConvertImageDtype(torch.float32))
    eval_transforms = transforms.Compose(eval_transforms)
    
    # Load model
    model_path = os.path.join(model_dir, model_name)
    model = SegmentationLightningModule.load_from_checkpoint(model_path)
    model.to(device).eval()
    
    # Get list of images in image dir
    images_fn = os.listdir( image_dir )
    
    for f in images_fn:
        print( image_dir + f )
    
        #load image and scale to float [0, 1] based on 16-bit integer value.     
        img = np.array( Image.open(image_dir + f) ) / (2**16 - 1)
        
        # Crop if crop is needed.
        if crop: 
            top = int((img.shape[0] // 2 + crop_offset[0]) - np.floor(crop_size / 2))
            bottom = int((img.shape[0] // 2 + crop_offset[0]) + np.ceil(crop_size / 2))
            left = int((img.shape[1] // 2 + crop_offset[1]) - np.floor(crop_size / 2))
            right = int((img.shape[1] // 2 + crop_offset[1]) + np.ceil(crop_size / 2))
            
            img = img[top:bottom, left:right]
        
        # Convert to pytorch tensor and apply transforms
        img = img[:, :, np.newaxis]
        image = to_image_tensor(img).unsqueeze(0)
        image = eval_transforms(image)
            
        # Run image pytorch tensor through model and obtain a mask.
        mask = model(image)
        mask = mask.detach().squeeze().numpy()
        mask = mask > .5
        
        # Convert mask to PIL image and save to output directory
        Image.fromarray(mask).save(output_dir+f)



if __name__ == "__main__":
    
    parser = ArgumentParser()

    parser.add_argument("--model_dir", 
                         default="C:\\Users\\zelinski1\\Desktop\\work\\Thrust2\\repo\\sm-unet-training-pipeline\\models\\17k_dataset\\")

    parser.add_argument("--model_name", default="epoch=23-val_loss=0.091.ckpt")
    parser.add_argument("--image_dir", default=r"C:\Users\zelinski1\Desktop\work\Thrust2\data\t231220\images\\")
    parser.add_argument("--output_dir", default=r"C:\Users\zelinski1\Desktop\work\Thrust2\data\t231220\output\\")
    parser.add_argument("--crop", default=True)
    parser.add_argument("--crop_size", default=1200)
    parser.add_argument("--crop_offset", default=[-60, -50])
    parser.add_argument("--image_size", default=512)

    args = parser.parse_args()

    kwargs = vars(args)

    om_inference(**kwargs)
    
