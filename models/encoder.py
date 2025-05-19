import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(UNetEncoder, self).__init__()
        self.down1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.down2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.down3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.down4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

    def forward(self, x):
        x1 = self.down1(x) # 256×256×128
        x2 = self.down2(self.pool1(x1)) # 128×128×64
        x3 = self.down3(self.pool2(x2)) # 64×64×32
        x4 = self.down4(self.pool3(x3)) # 32×32×16
        x5 = self.bottleneck(self.pool4(x4)) # 16×16×8
        return x5