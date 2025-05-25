import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

DEFAULT_SPACING = (0.7, 0.7, 1.4)
DEFAULT_SHAPE = (512, 512, 256)


def resample_image(image, spacing, is_mask=False):
    """
    Resample image to given voxel spacing using linear (or nearest) interpolation.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(size * old_spacing / new_spacing))
        for size, old_spacing, new_spacing in zip(original_size, original_spacing, spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resampler.Execute(image)


def center_crop_or_pad(volume, target_shape):
    """
    Center-crop or pad a 3D volume to the target shape.
    """
    result = np.zeros(target_shape, dtype=volume.dtype)
    input_shape = volume.shape
    crop_starts = [(s - t) // 2 if s > t else 0 for s, t in zip(input_shape, target_shape)]
    cropped = volume[
        crop_starts[0]:crop_starts[0] + min(target_shape[0], input_shape[0]),
        crop_starts[1]:crop_starts[1] + min(target_shape[1], input_shape[1]),
        crop_starts[2]:crop_starts[2] + min(target_shape[2], input_shape[2])
    ]
    pad_starts = [(t - s) // 2 if s < t else 0 for s, t in zip(cropped.shape, target_shape)]
    slices = tuple(slice(p, p + s) for p, s in zip(pad_starts, cropped.shape))

    result[slices] = cropped
    return result


def normalize_intensity(volume, hu_min=-1000, hu_max=500):
    volume = np.clip(volume, hu_min, hu_max)
    return (volume - hu_min) / (hu_max - hu_min)


def process_single_scan(input_path, output_path, spacing=DEFAULT_SPACING, shape=DEFAULT_SHAPE, is_mask=False):
    """
    Load, resample, crop/pad, normalize and save a single nifti scan.
    """
    image = sitk.ReadImage(input_path)
    resampled = resample_image(image, spacing, is_mask)
    volume = sitk.GetArrayFromImage(resampled)

    volume = np.transpose(volume, (2, 1, 0))
    volume = center_crop_or_pad(volume, shape)
    if not is_mask:
        volume = normalize_intensity(volume)

    nib_img = nib.Nifti1Image(volume.astype(np.float32), np.eye(4))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib_img, output_path)


def preprocess_folder(input_folder, output_folder, is_mask=False):
    """
    Apply preprocessing to all nifti scans in a folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii.gz"):
                print(f"Processing: {file}")
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, file)
                process_single_scan(input_path, output_path, is_mask=is_mask)