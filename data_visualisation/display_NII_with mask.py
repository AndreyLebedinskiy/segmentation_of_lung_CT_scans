import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

class NiftiViewerWithMask:
    def __init__(self, volume, mask):
        self.volume = volume
        self.mask = mask
        self.index = volume.shape[0] // 2

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.volume[self.index], cmap='gray')
        self.overlay = self.ax.imshow(np.ma.masked_where(self.mask[self.index] == 0, self.mask[self.index]),
                                      cmap='Reds', alpha=0.5)
        self.update_title()
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.show()

    def update_title(self):
        self.ax.set_title(f"Slice {self.index + 1}/{self.volume.shape[0]}")
        self.im.set_data(self.volume[self.index])
        self.overlay.set_data(np.ma.masked_where(self.mask[self.index] == 0, self.mask[self.index]))
        self.im.axes.figure.canvas.draw()

    def on_scroll(self, event):
        if event.button == 'up':
            self.index = min(self.index + 1, self.volume.shape[0] - 1)
        elif event.button == 'down':
            self.index = max(self.index - 1, 0)
        self.update_title()

def load_nifti(path):
    nifti = nib.load(path)
    volume = nifti.get_fdata()
    return np.transpose(volume, (2, 1, 0))

def main():
    path_to_scan = input("Provide absolute path to the CT scan: ")
    path_to_mask = input("Provide absolute path to the mask: ")

    scan = load_nifti(path_to_scan)
    mask = load_nifti(path_to_mask)

    if scan.shape != mask.shape:
        raise ValueError("Scan and mask dimensions do not match!")

    NiftiViewerWithMask(scan, mask)

if __name__ == "__main__":
    main()