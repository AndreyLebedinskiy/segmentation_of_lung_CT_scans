import os
import torch
import random
import nibabel as nib
from torch.utils.data import Dataset
import torch.nn.functional as functional

class RotationDataset(Dataset):
    def __init__(self, scan_dirs, output_size=(128, 256, 256), rotation_axes=(2, 3)):
        if isinstance(scan_dirs, str):
            scan_dirs = [scan_dirs]

        self.scan_paths = []
        for folder in scan_dirs:
            self.scan_paths.extend([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith('.nii.gz')
            ])
        self.output_size = output_size
        self.rotation_axes = rotation_axes

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        path = self.scan_paths[idx]
        scan = nib.load(path).get_fdata()
        scan = torch.from_numpy(scan).float().unsqueeze(0)
        scan = functional.interpolate(scan.unsqueeze(0), size=self.output_size, mode='trilinear', align_corners=False)
        scan = scan.squeeze(0)
        rotation_number = random.randint(0, 3)
        rotated = torch.rot90(scan, k=rotation_number, dims=self.rotation_axes)
        return rotated, torch.tensor(rotation_number, dtype=torch.long)
