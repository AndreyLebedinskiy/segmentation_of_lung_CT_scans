import os
import numpy as np
import nibabel as nib
import pandas as pd

def csv_to_mask_nifti(csv_path, reference_scan_path, output_mask_path):
    df = pd.read_csv(csv_path, names=["x", "y", "z", "label"])
    ref_img = nib.load(reference_scan_path)
    shape = ref_img.shape
    affine = ref_img.affine
    mask = np.zeros(shape, dtype=np.uint8)

    for _, row in df.iterrows():
        x, y, z, label = map(int, row)
        mask[x, y, z] = label

    mask_img = nib.Nifti1Image(mask, affine)
    nib.save(mask_img, output_mask_path)
    print(f"Saved mask: {output_mask_path}")


csv_to_mask_nifti(
    csv_path='segmentation_of_lung_CT_scans/data/vessel12_data/ExampleScans/Annotations/VESSEL12_21_Annotations.csv',
    reference_scan_path='segmentation_of_lung_CT_scans/data/vessel12_data/ExampleScans/Scans/VESSEL12_21.nii.gz',
    output_mask_path='segmentation_of_lung_CT_scans/data/vessel12_data/ExampleScans/Vessel_masks/VESSEL12_21.nii.gz'
)

csv_to_mask_nifti(
    csv_path='segmentation_of_lung_CT_scans/data/vessel12_data/ExampleScans/Annotations/VESSEL12_22_Annotations.csv',
    reference_scan_path='segmentation_of_lung_CT_scans/data/vessel12_data/ExampleScans/Scans/VESSEL12_22.nii.gz',
    output_mask_path='segmentation_of_lung_CT_scans/data/vessel12_data/ExampleScans/Vessel_masks/VESSEL12_22.nii.gz'
)

csv_to_mask_nifti(
    csv_path='segmentation_of_lung_CT_scans/data/vessel12_data/ExampleScans/Annotations/VESSEL12_23_Annotations.csv',
    reference_scan_path='segmentation_of_lung_CT_scans/data/vessel12_data/ExampleScans/Scans/VESSEL12_23.nii.gz',
    output_mask_path='segmentation_of_lung_CT_scans/data/vessel12_data/ExampleScans/Vessel_masks/VESSEL12_23.nii.gz'
)