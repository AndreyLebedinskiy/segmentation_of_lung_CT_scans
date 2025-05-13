import torch.nn as nn

class RestorationHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, patch_size=(32, 64, 64)):
        super(RestorationHead, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.output_shape = patch_size

    def forward(self, x):
        out = self.decoder(x)
        return out
