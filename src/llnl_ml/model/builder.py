# import timm

from .unet import UNet, UNetSmall, UNetMedium
from .fcn import FCN
from .mask_rcnn import MaskRCNNResNet50
from .segmentation_wrapper import SegmentationModelWithLoss
from .util import he_initialization

MODEL_NAMES = {
    "UNet": UNet,
    "FCN": FCN,
    "UNetSmall": UNetSmall,
    "UNetMedium": UNetMedium,
    "MaskRCNN": MaskRCNNResNet50,
}

USES_HE_INITIALIZATION = {"UNet", "UNetSmall", "UNetMedium", "FCN"}
SEGMENTATION_MODELS = {"UNet", "UNetSmall", "UNetMedium", "FCN"}  # Models to wrap


def get_model(model_name: str, input_channels: int, output_channels: int):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model name {model_name}")
    
    # Create the base model
    model = MODEL_NAMES[model_name](input_channels, output_channels)
    
    # Apply HE initialization if needed (before wrapping)
    if model_name in USES_HE_INITIALIZATION:
        model.apply(he_initialization)
    
    # Wrap segmentation models with loss computation wrapper
    if model_name in SEGMENTATION_MODELS:
        model = SegmentationModelWithLoss(model)
    
    return model
