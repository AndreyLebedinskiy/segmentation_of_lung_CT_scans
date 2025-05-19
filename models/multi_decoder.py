#NOT WORKING AS INTENDED RN

import torch
import torch.nn as nn

class MultiHeadDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels=(256, 128, 64), head_out_channels=None):
        super().__init__()
        if head_out_channels is None:
            head_out_channels = {
                'lung': 1,
                'heart': 6,
                'vessel': 1
            }

        self.lung_head = self._make_decoder_head(in_channels, skip_channels, head_out_channels['lung'])
        self.heart_head = self._make_decoder_head(in_channels, skip_channels, head_out_channels['heart'])
        self.vessel_head = self._make_decoder_head(in_channels, skip_channels, head_out_channels['vessel'])

    def _make_decoder_head(self, in_channels, skip_channels, out_channels):
        return nn.ModuleDict({
            'up1': nn.Sequential(
                nn.ConvTranspose3d(in_channels, 256, kernel_size=2, stride=2),
                nn.InstanceNorm3d(256),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=0.2)
            ),
            'conv1': nn.Sequential(
                nn.Conv3d(256 + skip_channels[0], 256, kernel_size=3, padding=1),
                nn.InstanceNorm3d(256),
                nn.ReLU(inplace=True)
            ),
            'up2': nn.Sequential(
                nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
                nn.InstanceNorm3d(128),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=0.2)
            ),
            'conv2': nn.Sequential(
                nn.Conv3d(128 + skip_channels[1], 128, kernel_size=3, padding=1),
                nn.InstanceNorm3d(128),
                nn.ReLU(inplace=True)
            ),
            'up3': nn.Sequential(
                nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=0.2)
            ),
            'conv3': nn.Sequential(
                nn.Conv3d(64 + skip_channels[2], 64, kernel_size=3, padding=1),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True)
            ),
            'final': nn.Conv3d(64, out_channels, kernel_size=1)
        })

    def forward(self, x, skips):
        return {
            'lung': self._forward_head(self.lung_head, x, skips),
            'heart': self._forward_head(self.heart_head, x, skips),
            'vessel': self._forward_head(self.vessel_head, x, skips)
        }

    def _forward_head(self, head, x, skips):
        x = head['up1'](x)
        if skips[2] is not None:
            x = torch.cat([x, skips[2]], dim=1)
        x = head['conv1'](x)

        x = head['up2'](x)
        if skips[1] is not None:
            x = torch.cat([x, skips[1]], dim=1)
        x = head['conv2'](x)

        x = head['up3'](x)
        if skips[0] is not None:
            x = torch.cat([x, skips[0]], dim=1)
        x = head['conv3'](x)

        return head['final'](x)
