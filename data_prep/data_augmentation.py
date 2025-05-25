import os
from glob import glob
import torchio as tio


# Define augmentation pipeline
AUGMENTATION = tio.Compose([
    tio.RandomElasticDeformation(num_control_points=5, max_displacement=10, locked_borders=2, p=0.9),
    tio.RandomNoise(mean=0, std=0.03, p=0.7),
    tio.RandomGamma(log_gamma=(0.9, 1.01), p=0.7),
])


def augment_image_only(image_path, save_dir, num_aug=2):
    print(f"Augmenting scan only: {os.path.basename(image_path)}")
    image = tio.ScalarImage(image_path)
    subject = tio.Subject(image=image)
    for i in range(num_aug):
        augmented = AUGMENTATION(subject)
        output_name = os.path.basename(image_path).replace('.nii.gz', f'_aug{i}.nii.gz')
        output_path = os.path.join(save_dir, 'scans', output_name)
        augmented.image.save(output_path)


def augment_image_and_mask(image_path, mask_path, save_dir, num_aug=2):
    print(f"Augmenting scan and mask: {os.path.basename(image_path)}")
    image = tio.ScalarImage(image_path)
    mask = tio.LabelMap(mask_path)
    subject = tio.Subject(image=image, mask=mask)
    for i in range(num_aug):
        augmented = AUGMENTATION(subject)
        scan_name = os.path.basename(image_path).replace('.nii.gz', f'_aug{i}.nii.gz')
        mask_name = os.path.basename(mask_path).replace('.nii.gz', f'_aug{i}.nii.gz')
        augmented.image.save(os.path.join(save_dir, 'scans', scan_name))
        augmented.mask.save(os.path.join(save_dir, 'masks', mask_name))


def run_augmentation(scan_dir, mask_dir, output_dir, num_aug):
    os.makedirs(os.path.join(output_dir, 'scans'), exist_ok=True)
    if mask_dir:
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    scan_paths = sorted(glob(os.path.join(scan_dir, '*.nii.gz')))
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.nii.gz'))) if mask_dir else []
    mask_lookup = {os.path.basename(m): m for m in mask_paths}

    print(f"Found {len(scan_paths)} scans in: {scan_dir}")
    for scan_path in scan_paths:
        scan_name = os.path.basename(scan_path)
        mask_path = mask_lookup.get(scan_name)
        if mask_path:
            augment_image_and_mask(scan_path, mask_path, output_dir, num_aug)
        else:
            augment_image_only(scan_path, output_dir, num_aug)
