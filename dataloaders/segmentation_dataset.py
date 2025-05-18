import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SegmentationDataset(Dataset):
    def __init__(self, scan_dirs, mask_dirs, task_type, target_shape=(128, 256, 256), num_classes=None):
        self.scan_paths = []
        self.mask_paths = []
        for scan_dir in scan_dirs:
            self.scan_paths.extend([os.path.join(scan_dir, f) for f in os.listdir(scan_dir) if f.endswith('.nii.gz')])
        for mask_dir in mask_dirs:
            self.mask_paths.extend([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        assert len(self.scan_paths) == len(self.mask_paths), "Mismatch between number of scans and masks"

        self.target_shape = target_shape
        self.num_classes = num_classes
        self.task_type = task_type

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        scan_path = self.scan_paths[idx]
        mask_path = self.mask_paths[idx]
        scan = nib.load(scan_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)
        scan = torch.tensor(scan[None, ...], dtype=torch.float32)  # [1, D, H, W]
        mask = torch.tensor(mask[None, ...], dtype=torch.float32)     # [1, D, H, W]
        scan = F.interpolate(scan.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.target_shape, mode='nearest').squeeze(0)
        
        if self.num_classes and self.num_classes > 1:
            mask = F.one_hot(mask.squeeze(0), num_classes=self.num_classes).permute(3, 0, 1, 2).float()  # [C, D, H, W]

        return scan, mask, self.task_type