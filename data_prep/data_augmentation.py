import os
import torchio as tio
from glob import glob


transform = tio.Compose([
    tio.RandomElasticDeformation(num_control_points=5, max_displacement=10, locked_borders=2, p=0.9),
    tio.RandomNoise(mean=0, std=0.03, p=0.7),
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
    
    mask_paths = sorted(glob(os.path.join(mask_folder, '*.nii.gz'))) if mask_folder else []
    mask_dict = {os.path.basename(p): p for p in mask_paths}

    for scan_path in scan_paths:
        fname = os.path.basename(scan_path)
        mask_path = mask_dict.get(fname)

        if mask_path:
            augment_scan_mask_pair(scan_path, mask_path, output_folder, num_aug=num_aug)
        else:
            augment_scan_only(scan_path, output_folder, num_aug=num_aug)

# === Run augmentation ===
augment_dataset(
    scan_folder='data/preprocesd/vessel12/ExampleScans/scans',
    mask_folder='data/preprocesd/vessel12/ExampleScans/vessel_masks',
    output_folder='data/augmented/vessel12/ExampleScans',
    num_aug=10
)
