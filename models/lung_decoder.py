import torch
import torch.nn as nn

class LungDecoder(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(LungDecoder, self).__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 256, kernel_size=2, stride=2),
            nn.InstanceNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm3d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.up4(x)
        x = self.conv4(x)
        return self.final(x)