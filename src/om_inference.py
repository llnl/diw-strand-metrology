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
import torchmetrics
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
    mask_dir, 
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
    print( model_dir )
    model_path = os.path.join(model_dir, model_name)
    model = SegmentationLightningModule.load_from_checkpoint(model_path)
    model.to(device).eval()
    
    # Get list of images in image dir
    images_fn = os.listdir( image_dir )
    masks_fn = os.listdir( mask_dir )
    
    for i, f in enumerate(images_fn):
        print( image_dir + f )
        print( mask_dir + masks_fn[i] )
        #load image and scale to float [0, 1] based on 16-bit integer value.     
        img = np.array( Image.open(image_dir + f) ) / (2**16 - 1)
        mask = np.array( Image.open(mask_dir + masks_fn[i]) ) 

        # Crop if crop is needed.
        if crop: 
            top = int((img.shape[0] // 2 + crop_offset[0]) - np.floor(crop_size / 2))
            bottom = int((img.shape[0] // 2 + crop_offset[0]) + np.ceil(crop_size / 2))
            left = int((img.shape[1] // 2 + crop_offset[1]) - np.floor(crop_size / 2))
            right = int((img.shape[1] // 2 + crop_offset[1]) + np.ceil(crop_size / 2))
            
            img = img[top:bottom, left:right]
            mask = mask[top:bottom, left:right]
        
        # Convert to pytorch tensor and apply transforms
        img = img[:, :, np.newaxis]
        image = to_image_tensor(img).unsqueeze(0)
        image = eval_transforms(image)
            
        # Run image pytorch tensor through model and obtain a mask.
        prediction = model(image)
        prediction = prediction.detach().squeeze().numpy()
        output = prediction > .5
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        acc_metric = torchmetrics.classification.BinaryAccuracy(threshold=0.5, multidim_average="global")
        jaccard_metric = torchmetrics.JaccardIndex(task="binary", threshold=0.5, ignore_index=0)
        dice_metric = torchmetrics.Dice(threshold=0.5)
        
        loss = float( loss_fn( torch.tensor(prediction), torch.tensor(mask, dtype=float) ) )
        acc = float( acc_metric( torch.tensor(prediction), torch.tensor(mask, dtype=float) ) )
        jac = float( jaccard_metric( torch.tensor(prediction), torch.tensor(mask, dtype=float) ) )
        dice = float( dice_metric( torch.tensor(prediction), torch.tensor(mask) ) )
        
        # Convert output to PIL image and save to output directory
        Image.fromarray(output).save(output_dir+f)
        
        output = output.astype('int')
        con_im = np.concatenate(( (255*(np.squeeze(img) - np.min(img))/np.max(img - np.min(img))).astype('uint8'), 
                             (255*(np.squeeze(mask) - np.min(mask))/np.max(mask - np.min(mask))).astype('uint8'), 
                             (255*(np.squeeze(prediction) - np.min(prediction))/np.max(prediction - np.min(prediction))).astype('uint8'), 
                             (255*(np.squeeze(output) - np.min(output))/np.max(output - np.min(output))).astype('uint8')), axis=1)

        label_bottom = True        
        if label_bottom: 
            from PIL import ImageDraw, ImageFont
            con_im_pil = Image.fromarray( np.vstack(( con_im, np.zeros( (300,4800), dtype=np.uint8 ) )) )
            draw = ImageDraw.Draw(con_im_pil)
            font = ImageFont.truetype("arial.ttf", size=150)
            text = r"loss_val: {:.4f}, acc: {:.4f}, jac: {:.4f}, dice: {:.4f}".format(loss,acc,jac,dice)
            position = (300, 1300)  # (x, y) coordinates
            text_color = 255# (255, 255, 255)  # RGB color
            draw.text(position, text, fill=text_color, font=font)
            con_im_pil.show()
            con_im_pil.save(output_dir+masks_fn[i])
        else:
            Image.fromarray(con_im).save(output_dir+masks_fn[i])



if __name__ == "__main__":
    
    parser = ArgumentParser()

    parser.add_argument("--model_dir", 
                         default=r"C:\\Users\zelinski1\Desktop\work\Thrust2\repo\sm-unet-training-pipeline\models\test-count-9-13446-it0-240424-1457\\".replace("\\", "\\"))

    parser.add_argument("--model_name", default="epoch=19-val_loss=0.099.ckpt")
    parser.add_argument("--image_dir", default=r"C:\Users\zelinski1\Desktop\work\Thrust2\data\t240418\images\\")
    parser.add_argument("--mask_dir", default=r"C:\Users\zelinski1\Desktop\work\Thrust2\data\t240418\masks\\")
    parser.add_argument("--output_dir", default=r"C:\Users\zelinski1\Desktop\work\Thrust2\data\t240418\output\13k_epoch19\\")
    parser.add_argument("--crop", default=True)
    parser.add_argument("--crop_size", default=1200)
    parser.add_argument("--crop_offset", default=[-60, -50])
    parser.add_argument("--image_size", default=1200)

    args = parser.parse_args()

    kwargs = vars(args)

    om_inference(**kwargs)
    
