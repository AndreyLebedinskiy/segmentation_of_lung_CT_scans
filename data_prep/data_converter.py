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


def dir_conversion(dir, res_dir, subdirs):
    for subdir in subdirs:
        input_dir = dir + subdir
        convert_mhd_to_nifti(input_dir, res_dir)
