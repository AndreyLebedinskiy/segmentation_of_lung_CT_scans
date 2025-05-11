import os
import SimpleITK as sitk


def convert_mhd_to_nifti(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mhd"):
                mhd_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                output_path = os.path.join(output_folder, base_name + ".nii.gz")
                try:
                    image = sitk.ReadImage(mhd_path)
                    sitk.WriteImage(image, output_path)
                    print("Converted: ", file)
                except Exception:
                    print("Failed to convert")


def dir_itteration(dir, res_dir, subdirs):
    for subdir in subdirs:
        input_dir = dir + subdir
        convert_mhd_to_nifti(input_dir, res_dir)


luna_subdirs = [
    "subset0"
    ,"subset1"
    ,"subset2"
    ,"subset3"
    ,"subset4"
    ,"subset5"
    ,"subset6"
    ,"subset7"
    ,"subset8"
    ,"subset9"
    ,"subset10"
]
luna_masks_subdirs = [
    "seg-lungs-LUNA16"
]
vessel_subdirs = [
    "Scans"
]
vessel_masks_subdirs = [
    "Lungmasks"
]

luna_input_dir = "data/luna16_data/"
luna_output_dir = "data/luna16_converted/scans/"
luna_masks_output_dir = "data/luna16_converted/lung_masks/"
vessel_input_dir = "data/vessel12_data/"
vessel_output_dir = "data/vessel12_converted/scans/"
vessel_masks_output_dir = "data/vessel12_converted/lung_masks/"

dir_itteration(luna_input_dir, luna_output_dir, luna_subdirs)
dir_itteration(luna_input_dir, luna_masks_output_dir, luna_masks_subdirs)
dir_itteration(vessel_input_dir, vessel_output_dir, vessel_subdirs)
dir_itteration(vessel_input_dir, vessel_masks_output_dir, vessel_masks_subdirs)