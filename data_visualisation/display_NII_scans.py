import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

class NiftiViewer:
    def __init__(self, volume):
        self.volume = volume
        self.index = volume.shape[0] // 2
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.volume[self.index], cmap='gray')
        self.update_title()
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.show()


    def update_title(self):
        self.ax.set_title(f"Slice {self.index + 1}/{self.volume.shape[0]}")
        self.im.set_data(self.volume[self.index])
        self.im.axes.figure.canvas.draw()


    def on_scroll(self, event):
        if event.button == 'up':
            self.index = min(self.index + 1, self.volume.shape[0] - 1)
        elif event.button == 'down':
            self.index = max(self.index - 1, 0)
        self.update_title()


def load_and_view(path):
    nifti = nib.load(path)
    volume = nifti.get_fdata()
    volume = np.transpose(volume, (2, 1, 0))
    NiftiViewer(volume)


path_to_file = input("Provide absolute path to the .nii.gz file: ")
load_and_view(path_to_file)