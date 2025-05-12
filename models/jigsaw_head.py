import torch.nn as nn

class JigsawHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=100):
        super(JigsawHead, self).__init__()

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),                # [B, C, 1, 1, 1] → [B, C]
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pool(x)
        return self.classifier(x)
