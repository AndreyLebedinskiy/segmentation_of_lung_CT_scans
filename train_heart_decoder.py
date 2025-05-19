import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from models.encoder import UNetEncoder
from models.heart_decoder import HeartDecoder
from dataloaders.segmentation_dataset import SegmentationDataset


SCAN_DIRS = [
    'data/preprocesd/MMWHS/scans',
    'data/augmented/MMWHS/scans'
]
MASK_DIRS = [
    'data/preprocesd/MMWHS/heart_masks',
    'data/augmented/MMWHS/masks'
]
ENCODER_PATH = 'pretrained_encoders/best_encoder_jigsaw.pth'
SAVE_PATH = 'pretrained_decoders/heart_decoder.pth'
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4


def multiclass_dice(preds, targets, num_classes):
    preds = torch.argmax(preds, dim=1)
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets[:, cls] if targets.ndim == 5 else (targets == cls).float())
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        dice_scores.append(dice.item())
    return sum(dice_scores) / num_classes


dataset = SegmentationDataset(
    scan_dirs=SCAN_DIRS,
    mask_dirs=MASK_DIRS,
    task_type='heart',
    target_shape=(128, 256, 256),
    num_classes=NUM_CLASSES
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

encoder = UNetEncoder().to(DEVICE)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False
decoder = HeartDecoder(in_channels=512, num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
train_losses, val_losses, val_dice_scores = [], [], []
best_dice = 0
for epoch in range(EPOCHS):
    decoder.train()
    total_loss = 0
    for scans, masks, _ in train_loader:
        scans, masks = scans.to(DEVICE), masks.to(DEVICE)
        with torch.no_grad():
            features = encoder(scans)
        preds = decoder(features)
        loss = criterion(preds, masks.argmax(1) if masks.ndim == 5 else masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * scans.size(0)
    avg_train_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    decoder.eval()
    val_loss = 0
    all_dice = []
    with torch.no_grad():
        for scans, masks, _ in val_loader:
            scans, masks = scans.to(DEVICE), masks.to(DEVICE)
            features = encoder(scans)
            preds = decoder(features)
            loss = criterion(preds, masks.argmax(1) if masks.ndim == 5 else masks)
            val_loss += loss.item() * scans.size(0)
            all_dice.append(multiclass_dice(preds, masks, NUM_CLASSES))

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_dice = sum(all_dice) / len(all_dice)
    val_losses.append(avg_val_loss)
    val_dice_scores.append(avg_dice)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Dice: {avg_dice:.4f}")
    if avg_dice > best_dice:
        best_dice = avg_dice
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(decoder.state_dict(), SAVE_PATH)

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.plot(val_dice_scores, label='Val Dice')
plt.xlabel('Epoch')
plt.ylabel('Loss / Dice')
plt.title('Heart Decoder Training Metrics')
plt.legend()
plt.savefig('heart_decoder_metrics.png')
plt.show()
