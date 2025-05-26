import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.encoder import UNetEncoder
from models.rotation_head import RotationHead
from dataloaders.rotation_dataset import RotationDataset
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIRS = [
    'data/augmented/luna16/scans',
    'data/augmented/MMWHS/scans',
    'data/augmented/vessel12/scans',
    'data/preprocesd/luna16/scans',
    'data/preprocesd/MMWHS/scans',
    'data/preprocesd/vessel12/scans'
]
SAVE_PATH = 'pretrained_encoders/best_encoder_rotation.pth'
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model(encoder, head, dataloader, criterion, device):
    encoder.eval()
    head.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            features, _ = encoder(inputs)
            outputs = head(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


dataset = RotationDataset(scan_dirs=DATA_DIRS)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# --- Model and training setup ---
encoder = UNetEncoder().to(DEVICE)
head = RotationHead(in_channels=512).to(DEVICE)
optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
train_accuracies, val_accuracies = [], []

for epoch in range(EPOCHS):
    encoder.train()
    head.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        features, _ = encoder(inputs)
        outputs = head(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        del loss, outputs, features
        torch.cuda.empty_cache()

    train_acc = correct / total
    val_loss, val_acc = evaluate_model(encoder, head, val_loader, criterion, DEVICE)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} — "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save best encoder
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(encoder.state_dict(), SAVE_PATH)

plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Rotation Prediction Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()