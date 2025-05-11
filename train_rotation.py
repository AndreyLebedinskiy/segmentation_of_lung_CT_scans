import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.encoder import UNetEncoder
from models.rotation_head import RotationHead
from dataloaders.rotation_dataset import RotationDataset
import matplotlib.pyplot as plt



DATA_DIRS = [
    #'data/augmented/luna16/scans',
    #'data/augmented/MMWHS/scans',
    #'data/augmented/vessel12/scans',
    'data/preprocesd/luna16/scans',
    'data/preprocesd/MMWHS_data/scans',
    'data/preprocesd/vessel12/scans'
]
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(encoder, head, dataloader, criterion, device):
    encoder.eval()
    head.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = head(encoder(images))
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = val_loss / total
    acc = correct / total
    return avg_loss, acc


dataset = RotationDataset(scan_folders=DATA_DIRS)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Models
encoder = UNetEncoder().to(DEVICE)
head = RotationHead().to(DEVICE)
# Optimizer & Loss
optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

torch.cuda.empty_cache()
train_accs, val_accs = [], []
best_val_acc = 0
for epoch in range(EPOCHS):
    encoder.train()
    head.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        features = encoder(images)
        outputs = head(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_accs.append(train_acc)

    val_loss, val_acc = validate(encoder, head, val_loader, criterion, DEVICE)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(encoder.state_dict(), 'best_encoder_rotation.pth')

# Plot accuracy curves
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Rotation Prediction Accuracy')
plt.legend()
plt.savefig('rotation_accuracy_curve.png')
plt.show()
