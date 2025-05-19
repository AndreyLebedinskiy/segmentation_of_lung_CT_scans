import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import random
import torch.nn.functional as functional

class RotationDataset(Dataset):
    def __init__(self, scan_folders):
        if isinstance(scan_folders, str):
            scan_folders = [scan_folders]
        self.scan_paths = []
        for folder in scan_folders:
            self.scan_paths.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nii.gz') ])

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        path = self.scan_paths[idx]
        scan = nib.load(path).get_fdata()
        scan = torch.tensor(scan.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scan = functional.interpolate(scan, size=(128, 256, 256), mode='trilinear', align_corners=False)
        scan = scan.squeeze(0).squeeze(0).numpy()
        k = random.randint(0, 3)
        rotated_scan = np.rot90(scan, k=k, axes=(0, 1)).copy()
        rotated_scan = torch.tensor(rotated_scan[None, ...], dtype=torch.float32)
        label = torch.tensor(k, dtype=torch.long)
        return rotated_scan, label
