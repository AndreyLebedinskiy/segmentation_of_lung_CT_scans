import torch.nn as nn

class RotationHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=4):
        super(RotationHead, self).__init__()

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global average pooling to reduce memory
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # Shape becomes [B, 512]
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(x)
        return self.classifier(x)
