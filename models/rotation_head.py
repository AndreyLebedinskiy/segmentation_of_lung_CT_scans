import torch.nn as nn

class RotationHead(nn.Module):
    def __init__(self, in_channels=512, hidden_dim=1024, num_classes=4):
        super(RotationHead, self).__init__()

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.pool(x)
        return self.classifier(x)
