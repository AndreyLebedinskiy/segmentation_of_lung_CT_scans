import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

def resample(image, target_spacing=(1.0, 1.0, 1.0), is_mask=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.   ())
    resampler.SetTransform(sitk.Transform())
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)

def crop_or_pad(volume, target_shape=(128, 128, 128)):
    result = np.zeros(target_shape, dtype=volume.dtype)
    offset = [(t - s) // 2 for s, t in zip(volume.shape, target_shape)]
    crop = [max((s - t) // 2, 0) for s, t in zip(volume.shape, target_shape)]

    cropped = volume[
        crop[0]:volume.shape[0] - crop[0],
        crop[1]:volume.shape[1] - crop[1],
        crop[2]:volume.shape[2] - crop[2]
    ]

    insert_slices = tuple(
        slice(offset[i], offset[i] + cropped.shape[i]) for i in range(3)
    )
    result[insert_slices] = cropped
    return result

def normalize_hu(volume, hu_min=-1000, hu_max=400):
    """Normalize HU values to range [0, 1]"""
    volume = np.clip(volume, hu_min, hu_max)
    return (volume - hu_min) / (hu_max - hu_min)

def process_nii_scan(input_path, output_path, target_spacing=(1.0, 1.0, 1.0), target_shape=(128, 128, 128)):
    image = sitk.ReadImage(input_path)
    resampled_image = resample(image, target_spacing)
    volume = sitk.GetArrayFromImage(resampled_image)  # [z, y, x]
    volume = np.transpose(volume, (2, 1, 0))  # [x, y, z]
    volume = crop_or_pad(volume, target_shape)
    volume = normalize_hu(volume)

    # Convert back to nibabel format for saving
    affine = np.eye(4)
    nib_img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nib_img, output_path)