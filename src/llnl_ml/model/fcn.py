import torch
from torch import nn


class FCN(nn.Module):
    def __init__(self, input_channels: int = 1, output_channels: int = 1):
        super(FCN, self).__init__()
        self.output_channels = output_channels  # Store for forward pass

        # Encoding layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoding layers
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        )

        self.dec3 = nn.Conv2d(64, output_channels, kernel_size=1)  # Output layer

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.dec1(x2)
        x4 = self.dec2(x3)
        x5 = self.dec3(x4)
        
        # Return logits for consistency with other models
        return x5

    @property
    def calculates_loss(self):
        return False

    @property
    def needs_boxes(self):
        return False
