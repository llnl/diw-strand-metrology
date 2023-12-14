# import timm

from .unet import UNet, UNetSmall, UNetMedium
from .fcn import FCN
from .mask_rcnn import MaskRCNNResNet50
from .util import he_initialization

MODEL_NAMES = {
    "UNet": UNet,
    "FCN": FCN,
    "UNetSmall": UNetSmall,
    "UNetMedium": UNetMedium,
    "MaskRCNN": MaskRCNNResNet50,
}

USES_HE_INITIALIZATION = {"UNet", "UNetSmall", "UNetMedium", "FCN"}


def get_model(model_name: str, input_channels: int, output_channels: int):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model name {model_name}")
    model = MODEL_NAMES[model_name](input_channels, output_channels)
    if model_name in USES_HE_INITIALIZATION:
        model.apply(he_initialization)
    return model
