# 4
# src/models/unet3d_baseline.py
import torch
import torch.nn as nn


# class DoubleConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             # Conv 1
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             # Conv 2
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class UNet3D(nn.Module):
#     # CHANGE FOR M3: default in_channels set to 4
#     def __init__(self, in_channels=4, out_channels=1, init_features=32):
#         super().__init__()
        
#         features = init_features
        
#         # --- Encoder ---
#         # Layer 1: Accepts 4 channels (Intensity + dx + dy + dz)
#         self.encoder1 = DoubleConv3d(in_channels, features)
#         self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2)) # Pooling only in H, W (keep Depth)

#         # Layer 2
#         self.encoder2 = DoubleConv3d(features, features * 2)
#         self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

#         # Layer 3
#         self.encoder3 = DoubleConv3d(features * 2, features * 4)
#         self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

#         # --- Bottleneck ---
#         self.bottleneck = DoubleConv3d(features * 4, features * 8)

#         # --- Decoder ---
#         # Up 3
#         self.up3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
#         self.decoder3 = DoubleConv3d(features * 8, features * 4) # Input is cat(up3, enc3)

#         # Up 2
#         self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
#         self.decoder2 = DoubleConv3d(features * 4, features * 2)

#         # Up 1
#         self.up1 = nn.ConvTranspose3d(features * 2, features, kernel_size=(1, 2, 2), stride=(1, 2, 2))
#         self.decoder1 = DoubleConv3d(features * 2, features)

#         # --- Final Classifier ---
#         self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         enc1 = self.encoder1(x)
#         p1 = self.pool1(enc1)

#         enc2 = self.encoder2(p1)
#         p2 = self.pool2(enc2)

#         enc3 = self.encoder3(p2)
#         p3 = self.pool3(enc3)

#         # Bottleneck
#         bottleneck = self.bottleneck(p3)

#         # Decoder
#         # Concatenate skip connections along channel axis (dim=1)
#         up3 = self.up3(bottleneck)
#         dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))

#         up2 = self.up2(dec3)
#         dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))

#         up1 = self.up1(dec2)
#         dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))

#         # Output
#         return self.final_conv(dec1)


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Conv 1
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # Conv 2
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    # CHANGE FOR M3: default in_channels set to 4
    def __init__(self, in_channels=4, out_channels=1, init_features=32):
        super().__init__()
        
        features = init_features
        
        # --- Encoder ---
        # Layer 1: Accepts 4 channels (Intensity + dx + dy + dz)
        self.encoder1 = DoubleConv3d(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2)) # Pooling only in H, W (keep Depth)

        # Layer 2
        self.encoder2 = DoubleConv3d(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Layer 3
        self.encoder3 = DoubleConv3d(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # --- Bottleneck ---
        self.bottleneck = DoubleConv3d(features * 4, features * 8)

        # --- Decoder ---
        # Up 3
        self.up3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder3 = DoubleConv3d(features * 8, features * 4) # Input is cat(up3, enc3)

        # Up 2
        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder2 = DoubleConv3d(features * 4, features * 2)

        # Up 1
        self.up1 = nn.ConvTranspose3d(features * 2, features, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder1 = DoubleConv3d(features * 2, features)

        # --- Final Classifier ---
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        p1 = self.pool1(enc1)

        enc2 = self.encoder2(p1)
        p2 = self.pool2(enc2)

        enc3 = self.encoder3(p2)
        p3 = self.pool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(p3)

        # Decoder
        # Concatenate skip connections along channel axis (dim=1)
        up3 = self.up3(bottleneck)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))

        up2 = self.up2(dec3)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))

        # Output
        return self.final_conv(dec1)
