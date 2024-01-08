from torch import nn


def he_initialization(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
