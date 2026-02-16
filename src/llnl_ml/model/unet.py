from typing import List

import torch
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        depth_channels: List[int] = (64, 128, 256, 512, 1024),
    ):
        """
        Generalized UNet architecture. Define the depth of the UNet via depth_channels.

        :param output_channels: Input data channels
        :param output_channels: Output channels
        :param depth_channels: Creates a down and up conv layer per channel.
        """
        super().__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.depth_channels = depth_channels

        self.conv1 = DoubleConv(input_channels, depth_channels[0])
        self.down_conv = nn.ModuleList()
        for in_ch, out_ch in zip(depth_channels[:-1], depth_channels[1:]):
            self.down_conv.append(DownConv(in_ch, out_ch))

        channels = depth_channels[::-1]
        self.up_conv = nn.ModuleList()
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.up_conv.append(UpConv(in_ch, out_ch))
            self.final_conv = nn.Conv2d(depth_channels[0], output_channels, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    @property
    def calculates_loss(self):
        return False

    @staticmethod
    def get_dataset_requirements():
        """Returns the dataset wrapper class and collate function needed for this model."""
        from torch.utils.data import default_collate

        return None, default_collate  # UNet doesn't need special dataset wrapper

    def encode(self, x):
        """
        Used to generate image encodings from the model. Outputs the GlobalAveragePool of the
        final down conv layer in the network. This will have the same dimension as the final
        depth_channels value for the model.
        """
        x = self.conv1(x)
        for down in self.down_conv:
            x = down(x)

        x = self.pool(x)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x_down = []
        for down in self.down_conv:
            x_down.append(x)
            x = down(x)

        for up, x_prior in zip(self.up_conv, reversed(x_down)):
            x = up(x, x_prior)

        out = self.final_conv(x)

        return out


class UNetSmall(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = 1):
        super(UNetSmall, self).__init__()

        # Contracting Path
        self.enc1 = self.contract_block(input_channels, 64)
        self.enc2 = self.contract_block(64, 128)

        # Expansive Path
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.expand_block(128, 128, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.expand_block(32 + input_channels, 64, 32)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    @property
    def calculates_loss(self):
        return False

    @staticmethod
    def get_dataset_requirements():
        """Returns the dataset wrapper class and collate function needed for this model."""
        from torch.utils.data import default_collate

        return None, default_collate  # UNetSmall doesn't need special dataset wrapper

    def contract_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        return block

    def expand_block(self, in_channels, middle_channels, out_channels):  # Add middle_channels argument
        block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),  # Use middle_channels here
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),  # And here
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        upconv1 = self.upconv1(enc2)

        dec1 = self.dec1(torch.cat([upconv1, enc1], 1))

        upconv2 = self.upconv2(dec1)
        dec2 = self.dec2(torch.cat([upconv2, x], 1))  # connecting back to original input

        return torch.sigmoid(self.final_conv(dec2))


class UNetMedium(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.conv = DoubleConv(in_channels=in_channels, out_channels=64)
        self.down1 = DownConv(in_channels=64, out_channels=128)
        self.down2 = DownConv(in_channels=128, out_channels=256)
        self.down3 = DownConv(in_channels=256, out_channels=512)
        self.up3 = UpConv(in_channels=512, out_channels=256)
        self.up2 = UpConv(in_channels=256, out_channels=128)
        self.up1 = UpConv(in_channels=128, out_channels=64)
        self.out = torch.nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv(x)  # (1, 256, 256) -> (64, 256, 256)
        x2 = self.down1(x1)  # (64, 256, 256) -> (128, 128, 128)
        x3 = self.down2(x2)  # (128, 128, 128) -> (256, 64, 64)
        x = self.down3(x3)  # (256, 64, 64) -> (512, 32, 32)
        x = self.up3(x, x3)  # (512, 32, 32) -> (256, 64, 64)
        x = self.up2(x, x2)  # (256, 64, 64) -> (128, 128, 128)
        x = self.up1(x, x1)  # (128, 128, 128) -> (64, 256, 256)
        return self.out(x)  # (64, 256, 256) -> (1, 256, 256)

    @property
    def calculates_loss(self):
        return False

    @staticmethod
    def get_dataset_requirements():
        """Returns the dataset wrapper class and collate function needed for this model."""
        from torch.utils.data import default_collate

        return None, default_collate  # UNetMedium doesn't need special dataset wrapper


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.double_conv(x)

        return x


class DownConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        x = self.down_conv(x)

        return x


class UpConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x
