import SimpleITK as sitk
import numpy as np
import cv2


def load_volume(path):
    if path.endswith(".mhd"):
        image = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(image)
    else:
        raise ValueError("Unsupported format")


def normalize_slice(slice_2d, window):
    center, width = window
    min_val = center - width // 2
    max_val = center + width // 2
    slice_2d = np.clip(slice_2d, min_val, max_val)
    slice_2d = (slice_2d - min_val) / (max_val - min_val + 1e-5)
    return (slice_2d * 255).astype(np.uint8)


def view_ct_keyboard(path):
    volume = load_volume(path)
    index = volume.shape[0] // 2
    window = [ -600, 1500 ]
    window_names = {
        (-600, 1500): "Lung",
        (40, 300): "Soft Tissue"
    }


    print("Controls: 'a/d' to scroll, 'w' to toggle window, ESC to quit")
    while True:
        slice_img = normalize_slice(volume[index], window)
        color_img = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)

        label = f"Slice {index + 1}/{volume.shape[0]} | Window: {window_names.get(tuple(window), 'Custom')}"
        cv2.putText(color_img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("CT Viewer", color_img)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:
            break
        elif key == ord('d'):
            index = min(index + 1, volume.shape[0] - 1)
        elif key == ord('a'):
            index = max(index - 1, 0)
        elif key == ord('w'):
            if window == [-600, 1500]:
                window = [40, 300]
            else:
                window = [-600, 1500]
    cv2.destroyAllWindows()

# Main
path = input("Provide reletive path: ")
view_ct_keyboard(path)