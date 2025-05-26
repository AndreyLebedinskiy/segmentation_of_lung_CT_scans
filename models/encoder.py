import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            nn.Dropout3d(p=0.2)
        )

    def forward(self, x):
        x1 = self.enc1(x)                      # Full resolution
        x2 = self.enc2(self.pool1(x1))         # 1/2 resolution
        x3 = self.enc3(self.pool2(x2))         # 1/4 resolution
        x4 = self.enc4(self.pool3(x3))         # 1/8 resolution
        x5 = self.bottleneck(self.pool4(x4))   # 1/16 resolution

        return x5, [x4, x3, x2, x1]