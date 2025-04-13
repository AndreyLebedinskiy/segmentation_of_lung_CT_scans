import os
import SimpleITK as sitk

def convert_mhd_folder_to_nifti(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # Scan for .mhd files
    for root, _, files in os.walk(input_folder):
        for fname in files:
            if fname.endswith(".mhd"):
                print('!')
                mhd_path = os.path.join(root, fname)
                base_name = os.path.splitext(fname)[0]
                output_path = os.path.join(output_folder, base_name + ".nii.gz")
                try:
                    image = sitk.ReadImage(mhd_path)
                    sitk.WriteImage(image, output_path)
                    print(f"Converted: {fname} to {base_name}.nii.gz")
                except Exception as e:
                    print(f"Failed to convert {fname}: {e}")

subsets = [
    "subset0"
    ,"subset1"
    ,"subset2"
    ,"subset3"
    ,"subset4"
#    ,"subset5"
#    ,"subset6"
#    ,"subset7"
#    ,"subset8"
#    ,"subset9"
#    ,"subset10"
]

for subset in subsets:
    input_dir = "segmentation_of_lung_CT_scans/luna16_data/" + subset
    output_dir = "segmentation_of_lung_CT_scans/luna16_converted_nifti/" + subset
    convert_mhd_folder_to_nifti(input_dir, output_dir)