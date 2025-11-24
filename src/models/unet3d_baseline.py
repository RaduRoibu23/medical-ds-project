# src/models/unet3d_baseline.py
import torch
import torch.nn as nn


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    Input:  (B,1,D,H,W)  CT patch
    Output: (B,1,D,H,W) logits; folosim doar slice-ul central la loss.
    """

    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()

        self.enc1 = DoubleConv3d(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3d(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3d(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3d(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = DoubleConv3d(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = DoubleConv3d(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = DoubleConv3d(base_channels * 2, base_channels)

        self.out_conv = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        x4 = self.bottleneck(self.pool3(x3))

        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        logits_3d = self.out_conv(x)
        return logits_3d
