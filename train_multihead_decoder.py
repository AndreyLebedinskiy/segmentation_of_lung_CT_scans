#DOES NOT WORK AS INTENDED

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio

from models.encoder import UNetEncoder
from models.restoration_head import RestorationHead
from dataloaders.restoration_dataset import RestorationDataset


DATA_DIRS = [
    'data/preprocesd/luna16/scans',
    'data/preprocesd/MMWHS/scans',
    'data/preprocesd/vessel12/scans'
]
SAVE_PATH = 'pretrained_encoders/best_encoder_restoration.pth'
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
def validate(encoder, head, dataloader, criterion, psnr_metric, device):
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
            val_loss += loss.item() * inputs.size(0)
            val_psnr += psnr_metric(outputs, targets).item() * inputs.size(0)
    return val_loss / len(dataloader.dataset), val_psnr / len(dataloader.dataset)

train_losses, val_losses, val_psnrs = [], [], []
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    encoder.train()
    head.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        features = encoder(inputs)
        outputs = head(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    avg_train_loss = total_loss / len(train_loader.dataset)
    avg_val_loss, avg_val_psnr = validate(encoder, head, val_loader, criterion, psnr_metric, DEVICE)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_psnrs.append(avg_val_psnr)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val PSNR: {avg_val_psnr:.2f}dB")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(encoder.state_dict(), SAVE_PATH)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Restoration Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_psnrs, label='Val PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.title('Peak Signal-to-Noise Ratio')
plt.legend()

plt.tight_layout()
plt.savefig('restoration_metrics.png')
plt.show()
