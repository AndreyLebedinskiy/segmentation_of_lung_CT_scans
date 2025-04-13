import SimpleITK as sitk
import matplotlib.pyplot as plt

def display_mhd_slice(mhd_path, slice_index=None):
    image = sitk.ReadImage(mhd_path)
    volume = sitk.GetArrayFromImage(image)  # shape: [slices, height, width]

    # Choose slice index to show
    if slice_index is None:
        slice_index = volume.shape[0] // 2  # Show middle slice

    slice_2d = volume[slice_index]
    
    plt.imshow(slice_2d, cmap='gray')
    plt.title(f"Slice {slice_index} of {mhd_path}")
    plt.axis('off')
    plt.show()


display_mhd_slice("luna16_data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd")
