import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def resample(image, target_spacing, is_mask):
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
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    if is_mask:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def crop_or_pad(volume, target_shape):
    """Crop or pad a 3D volume to the desired shape (centered)."""
    result = np.zeros(target_shape, dtype=volume.dtype)
    input_shape = volume.shape
    crop = [max((s - t) // 2, 0) for s, t in zip(input_shape, target_shape)]
    cropped = volume[
        crop[0]:crop[0] + min(target_shape[0], input_shape[0]),
        crop[1]:crop[1] + min(target_shape[1], input_shape[1]),
        crop[2]:crop[2] + min(target_shape[2], input_shape[2])
    ]
    insert_offset = [max((t - c) // 2, 0) for t, c in zip(target_shape, cropped.shape)]
    insert_slices = tuple(
        slice(insert_offset[i], insert_offset[i] + cropped.shape[i]) for i in range(3)
    )

    result[insert_slices] = cropped
    return result


def normalize_hu(volume, hu_min=-1000, hu_max=500):
    volume = np.clip(volume, hu_min, hu_max)
    return (volume - hu_min) / (hu_max - hu_min)


def process_nii_scan(input_path, output_path, target_spacing, target_shape, is_mask):
    image = sitk.ReadImage(input_path)
    resampled_image = resample(image, target_spacing, is_mask)
    volume = sitk.GetArrayFromImage(resampled_image)
    volume = np.transpose(volume, (2, 1, 0)) #different axis order in numpy
    volume = crop_or_pad(volume, target_shape)
    volume = normalize_hu(volume)
    affine = np.eye(4)
    nib_img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nib_img, output_path)


def applying_preprocesing(input_folder, output_folder, is_mask):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii.gz"):
                print("preprocesing:", file)
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, file)
                process_nii_scan(
                        input_path,
                        output_path,
                        (0.7, 0.7, 1.4),
                        (512, 512, 256),
                        is_mask
                )

luna_masks_input = "data/luna16_converted/lung_masks/"
luna_data_input = "data/luna16_converted/scans/"
luna_masks_output = "data/preprocesd/luna16/lung_masks/"
luna_data_output = "data/preprocesd/luna16/scans/"

vessel_masks_input = "data/vessel12_converted/lung_masks/"
vessel_data_input = "data/vessel12_converted/scans/"
vessel_masks_output = "data/preprocesd/vessel12/lung_masks/"
vessel_data_output = "data/preprocesd/vessel12/scans/"

mmwhs_masks_input = "data/MMWHS_data/all_masks"
mmwhs_data_input = "data/MMWHS_data/all_scans"
mmwhs_masks_output = "data/preprocesd/MMWHS_data/heart_masks"
mmwhs_data_output = "data/preprocesd/MMWHS_data/scans"

#applying_preprocesing(luna_masks_input, luna_masks_output, True)
applying_preprocesing(luna_data_input, luna_data_output, False)
applying_preprocesing(vessel_masks_input, vessel_masks_output, True)
applying_preprocesing(vessel_data_input, vessel_data_output, False)
applying_preprocesing(mmwhs_masks_input, mmwhs_masks_output, True)
applying_preprocesing(mmwhs_data_input, mmwhs_data_output, False)
