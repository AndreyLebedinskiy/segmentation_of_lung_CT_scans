import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class RestorationDataset(Dataset):
    def __init__(self, scan_folders, target_shape=(128, 256, 256), mask_size=(32, 64, 64)):
        if isinstance(scan_folders, str):
            scan_folders = [scan_folders]

        self.scan_paths = []
        for folder in scan_folders:
            self.scan_paths.extend([
                os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nii.gz')
            ])

        self.target_shape = target_shape
        self.mask_size = mask_size

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        path = self.scan_paths[idx]
        scan = nib.load(path).get_fdata()
        scan = torch.tensor(scan.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        scan = F.interpolate(scan, size=self.target_shape, mode='trilinear', align_corners=False)
        scan = scan.squeeze(0).squeeze(0)  # [D, H, W]

        D, H, W = self.target_shape
        d_mask, h_mask, w_mask = self.mask_size

        # Random location, avoiding edges
        z = random.randint(0, D - d_mask - 1)
        y = random.randint(0, H - h_mask - 1)
        x = random.randint(0, W - w_mask - 1)

        # Extract the ground truth patch
        target_patch = scan[z:z+d_mask, y:y+h_mask, x:x+w_mask].clone()

        # Zero out the patch in the input scan
        masked_scan = scan.clone()
        masked_scan[z:z+d_mask, y:y+h_mask, x:x+w_mask] = 0.0

        input_tensor = masked_scan.unsqueeze(0)  # [1, D, H, W]
        target_tensor = target_patch.unsqueeze(0)  # [1, d_mask, h_mask, w_mask]

        return input_tensor, target_tensor
