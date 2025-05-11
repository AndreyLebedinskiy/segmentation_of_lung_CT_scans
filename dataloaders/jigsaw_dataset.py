import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from itertools import permutations

# Generate a list of 100 valid permutations of 8 elements where all indices are used at least once
def generate_permutation_set(n=8, k=100, seed=42):
    random.seed(seed)
    perm_set = set()
    all_perms = list(permutations(range(n)))
    random.shuffle(all_perms)
    for perm in all_perms:
        perm_set.add(perm)
        if len(perm_set) >= k:
            break
    return list(perm_set)

PERMUTATIONS = generate_permutation_set()

class JigsawDataset(Dataset):
    def __init__(self, scan_folders, target_shape=(128, 256, 256), grid_size=(2, 2, 2)):
        if isinstance(scan_folders, str):
            scan_folders = [scan_folders]

        self.scan_paths = []
        for folder in scan_folders:
            self.scan_paths.extend([
                os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nii.gz')
            ])

        self.target_shape = target_shape
        self.grid_size = grid_size
        self.permutations = PERMUTATIONS

    def __len__(self):
        return len(self.scan_paths)

    def _split_volume(self, volume):
        dz, dy, dx = self.target_shape
        gz, gy, gx = self.grid_size
        sz, sy, sx = dz // gz, dy // gy, dx // gx
        patches = []
        for z in range(gz):
            for y in range(gy):
                for x in range(gx):
                    patch = volume[
                        z*sz:(z+1)*sz,
                        y*sy:(y+1)*sy,
                        x*sx:(x+1)*sx
                    ]
                    patches.append(patch)
        return patches

    def __getitem__(self, idx):
        path = self.scan_paths[idx]
        scan = nib.load(path).get_fdata()
        scan = torch.tensor(scan.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        scan = F.interpolate(scan, size=self.target_shape, mode='trilinear', align_corners=False)
        scan = scan.squeeze(0).squeeze(0).numpy()  # [D, H, W]

        # Split into 8 patches
        patches = self._split_volume(scan)  # list of 8 subvolumes

        # Select and apply a permutation
        perm_index = random.randint(0, len(self.permutations) - 1)
        perm = self.permutations[perm_index]
        permuted_patches = [patches[i] for i in perm]

        # Reassemble permuted volume
        gz, gy, gx = self.grid_size
        sz, sy, sx = self.target_shape[0] // gz, self.target_shape[1] // gy, self.target_shape[2] // gx
        volume = np.zeros(self.target_shape, dtype=np.float32)
        i = 0
        for z in range(gz):
            for y in range(gy):
                for x in range(gx):
                    volume[
                        z*sz:(z+1)*sz,
                        y*sy:(y+1)*sy,
                        x*sx:(x+1)*sx
                    ] = permuted_patches[i]
                    i += 1

        volume = torch.tensor(volume[None, ...], dtype=torch.float32)  # [1, D, H, W]
        label = torch.tensor(perm_index, dtype=torch.long)
        return volume, label
