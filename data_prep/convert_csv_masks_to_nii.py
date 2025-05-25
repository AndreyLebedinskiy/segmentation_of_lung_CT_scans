import csv
import os
import numpy as np
import nibabel as nib

def convert_csv_to_mask(csv_path, reference_scan_path, output_mask_path):
    output_dir = os.path.dirname(output_mask_path)
    os.makedirs(output_dir, exist_ok=True)
    ref_img = nib.load(reference_scan_path)
    shape = ref_img.shape
    affine = ref_img.affine
    mask = np.zeros(shape, dtype=np.uint8)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x, y, z, label = map(int, row)
            mask[x, y, z] = 1
    nib.save(nib.Nifti1Image(mask, affine), output_mask_path)
