import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.encoder import UNetEncoder
from models.restoration_head import RestorationHead
from dataloaders.restoration_dataset import RestorationDataset
import matplotlib.pyplot as plt

# PSNR metric
def compute_psnr(pred, target, max_val=1.0):
    mse = nn.functional.mse_loss(pred, target, reduction='none')
    mse = mse.mean(dim=[1, 2, 3, 4])  # average over all spatial dims
    psnr = 10 * torch.log10((max_val ** 2) / (mse + 1e-8))
    return psnr.mean().item()

# === Configs ===
DATA_DIRS = [
    #'data/augmented/luna16/scans',
    #'data/augmented/MMWHS/scans',
    #'data/augmented/vessel12/scans',
    'data/preprocesd/luna16/scans',
    'data/preprocesd/MMWHS/scans',
    'data/preprocesd/vessel12/scans'
]
SAVE_PATH = 'pretrained_encoders/best_encoder_restoration.pth'
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Validation Function ===
def validate(encoder, head, dataloader, criterion, device):
    encoder.eval()
    head.eval()
    val_loss = 0
    val_psnr = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            features = encoder(inputs)
            outputs = head(features)
            loss = criterion(outputs, targets)
            psnr = compute_psnr(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            val_psnr += psnr * inputs.size(0)
    n = len(dataloader.dataset)
    return val_loss / n, val_psnr / n

patch_size = (32, 64, 64)
dataset = RestorationDataset(scan_folders=DATA_DIRS, mask_size=patch_size)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

encoder = UNetEncoder().to(DEVICE)
head = RestorationHead(in_channels=512, patch_size=patch_size).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=LEARNING_RATE)

train_losses, val_losses = [], []
train_psnrs, val_psnrs = [], []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    encoder.train()
    head.train()
    total_loss = 0
    total_psnr = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        features = encoder(inputs)
        outputs = head(features)
        loss = criterion(outputs, targets)
        psnr = compute_psnr(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_psnr += psnr * inputs.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    avg_train_psnr = total_psnr / len(train_loader.dataset)
    avg_val_loss, avg_val_psnr = validate(encoder, head, val_loader, criterion, DEVICE)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_psnrs.append(avg_train_psnr)
    val_psnrs.append(avg_val_psnr)

    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Train Loss: {avg_train_loss:.4f} - Train PSNR: {avg_train_psnr:.2f} - "
          f"Val Loss: {avg_val_loss:.4f} - Val PSNR: {avg_val_psnr:.2f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(encoder.state_dict(), SAVE_PATH)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curve')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_psnrs, label='Train PSNR')
plt.plot(val_psnrs, label='Val PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.title('PSNR Curve')
plt.legend()
plt.tight_layout()
plt.show()