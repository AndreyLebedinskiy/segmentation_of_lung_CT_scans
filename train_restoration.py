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

# Configs

DATA_DIRS = [
    #'data/augmented/luna16/scans',
    #'data/augmented/MMWHS/scans',
    #'data/augmented/vessel12/scans',
    'data/preprocesd/luna16/scans',
    'data/preprocesd/MMWHS_data/scans',
    'data/preprocesd/vessel12/scans'
]
ENCODER_PATH = 'best_encoder_jigsaw.pth'  # Start from best self-supervised checkpoint
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
patch_size = (32, 64, 64)
dataset = RestorationDataset(scan_folders=DATA_DIRS, mask_size=patch_size)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load encoder pretrained on jigsaw task
encoder = UNetEncoder().to(DEVICE)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))

# Restoration head
head = RestorationHead(in_channels=512, patch_size=patch_size).to(DEVICE)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=LEARNING_RATE)

# Validation
def validate(encoder, head, dataloader, criterion, device):
    encoder.eval()
    head.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            features = encoder(inputs)
            outputs = head(features)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(dataloader.dataset)

# Training
train_losses, val_losses = [], []
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
    avg_val_loss = validate(encoder, head, val_loader, criterion, DEVICE)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(encoder.state_dict(), 'best_encoder_restoration.pth')

# Plot
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Restoration Reconstruction Loss')
plt.legend()
plt.savefig('restoration_loss_curve.png')
plt.show()