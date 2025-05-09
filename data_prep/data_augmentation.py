import os
import torchio as tio
from glob import glob


transform = tio.Compose([
    tio.RandomElasticDeformation(num_control_points=7, max_displacement=7, locked_borders=2, p=0.7),
    tio.RandomNoise(mean=0, std=0.02, p=0.7),
    tio.RandomGamma(log_gamma=(0.9, 1.01), p=0.7),
])

def augment_scan_only(image_path, output_dir, num_aug=2):
    print("augmenting scan only:", image_path)
    image = tio.ScalarImage(image_path)
    subject = tio.Subject(image=image)

    for i in range(num_aug):
        augmented = transform(subject)
        base = os.path.basename(image_path).replace('.nii.gz', f'_aug{i}.nii.gz')
        augmented.image.save(os.path.join(output_dir, 'scans', base))

def augment_scan_mask_pair(image_path, mask_path, output_dir, num_aug=2):
    print("augmenting scan+mask:", image_path)
    image = tio.ScalarImage(image_path)
    mask = tio.LabelMap(mask_path)
    subject = tio.Subject(image=image, mask=mask)

    for i in range(num_aug):
        augmented = transform(subject)
        base = os.path.basename(image_path).replace('.nii.gz', f'_aug{i}.nii.gz')
        mask_base = os.path.basename(mask_path).replace('.nii.gz', f'_aug{i}.nii.gz')
        augmented.image.save(os.path.join(output_dir, 'scans', base))
        augmented.mask.save(os.path.join(output_dir, 'masks', mask_base))


def augment_dataset(scan_folder, mask_folder, output_folder, num_aug=2):
    os.makedirs(os.path.join(output_folder, 'scans'), exist_ok=True)
    if mask_folder:
        os.makedirs(os.path.join(output_folder, 'masks'), exist_ok=True)
    print("Augmenting folder:", scan_folder)

    scan_paths = sorted(glob(os.path.join(scan_folder, '*.nii.gz')))
    mask_paths = []
    if mask_folder: 
        mask_paths = sorted(glob(os.path.join(mask_folder, '*.nii.gz')))
    mask_dict = {os.path.basename(p): path for path in mask_paths}

    for scan_path in scan_paths:
        scan_fname = os.path.basename(scan_path)
        mask_path = mask_dict.get(scan_fname)

        if mask_path:
            augment_scan_mask_pair(scan_path, mask_path, output_folder, num_aug=num_aug)
        else:
            augment_scan_only(scan_path, output_dir=output_folder, num_aug=num_aug)


augment_dataset(
    scan_folder='segmentation_of_lung_CT_scans/data/preprocesd/luna16/scans',
    mask_folder='segmentation_of_lung_CT_scans/data/preprocesd/luna16/lung_masks',
    output_folder='segmentation_of_lung_CT_scans/data/augmented/luna16',
    num_aug=2
)