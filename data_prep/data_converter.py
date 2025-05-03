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
vessel_subdirs = [
    "Scans"
]

luna_input_dir = "segmentation_of_lung_CT_scans/data/luna16_data/"
luna_output_dir = "segmentation_of_lung_CT_scans/data/luna16_converted/"
vessel_input_dir = "segmentation_of_lung_CT_scans/data/vessel12_data/"
vessel_output_dir = "segmentation_of_lung_CT_scans/data/vessel12_converted/"

dir_itteration(luna_input_dir, luna_output_dir, luna_subdirs)
dir_itteration(vessel_input_dir, vessel_output_dir, vessel_subdirs)