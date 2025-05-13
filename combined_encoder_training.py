import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random
import matplotlib.pyplot as plt

from dataloaders.rotation_dataset import RotationDataset
from dataloaders.jigsaw_dataset import JigsawDataset
from dataloaders.restoration_dataset import RestorationDataset
from models.encoder import UNetEncoder
from models.rotation_head import RotationHead
from models.jigsaw_head import JigsawHead
from models.restoration_head import RestorationHead

# === Configuration ===
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
PATIENCE = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Dataset setup ===
rotation_ds = RotationDataset(scan_folders=DATA_DIRS)
jigsaw_ds = JigsawDataset(scan_folders=DATA_DIRS)
restoration_ds = RestorationDataset(scan_folders=DATA_DIRS)

rotation_train, rotation_val = random_split(rotation_ds, [int(0.8*len(rotation_ds)), len(rotation_ds)-int(0.8*len(rotation_ds))])
jigsaw_train, jigsaw_val = random_split(jigsaw_ds, [int(0.8*len(jigsaw_ds)), len(jigsaw_ds)-int(0.8*len(jigsaw_ds))])
restoration_train, restoration_val = random_split(restoration_ds, [int(0.8*len(restoration_ds)), len(restoration_ds)-int(0.8*len(restoration_ds))])

rotation_loader = DataLoader(rotation_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
jigsaw_loader = DataLoader(jigsaw_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
restoration_loader = DataLoader(restoration_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

rotation_val_loader = DataLoader(rotation_val, batch_size=BATCH_SIZE)
jigsaw_val_loader = DataLoader(jigsaw_val, batch_size=BATCH_SIZE)
restoration_val_loader = DataLoader(restoration_val, batch_size=BATCH_SIZE)

rotation_iter = iter(rotation_loader)
jigsaw_iter = iter(jigsaw_loader)
restoration_iter = iter(restoration_loader)

# === Model definitions ===
encoder = UNetEncoder().to(DEVICE)
rotation_head = RotationHead().to(DEVICE)
jigsaw_head = JigsawHead().to(DEVICE)
restoration_head = RestorationHead().to(DEVICE)

# === Optimizer ===
optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(rotation_head.parameters()) +
    list(jigsaw_head.parameters()) +
    list(restoration_head.parameters()),
    lr=LEARNING_RATE
)

rotation_loss_fn = nn.CrossEntropyLoss()
jigsaw_loss_fn = nn.CrossEntropyLoss()
restoration_loss_fn = nn.MSELoss()

task_losses = {'rotation': [], 'jigsaw': [], 'restoration': []}
val_losses = {'rotation': [], 'jigsaw': [], 'restoration': []}
best_score = float('inf')
patience_counter = 0

# === Training Loop ===
for epoch in range(EPOCHS):
    encoder.train()
    rotation_head.train()
    jigsaw_head.train()
    restoration_head.train()
    print(1)
    for _ in range(len(rotation_loader)):
        task = random.choice(['rotation', 'jigsaw', 'restoration'])

        optimizer.zero_grad()

        if task == 'rotation':
            try:
                inputs, labels = next(rotation_iter)
            except StopIteration:
                rotation_iter = iter(rotation_loader)
                inputs, labels = next(rotation_iter)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = rotation_head(encoder(inputs))
            loss = rotation_loss_fn(outputs, labels)

        elif task == 'jigsaw':
            try:
                inputs, labels = next(jigsaw_iter)
            except StopIteration:
                jigsaw_iter = iter(jigsaw_loader)
                inputs, labels = next(jigsaw_iter)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = jigsaw_head(encoder(inputs))
            loss = jigsaw_loss_fn(outputs, labels)

        elif task == 'restoration':
            try:
                inputs, targets = next(restoration_iter)
            except StopIteration:
                restoration_iter = iter(restoration_loader)
                inputs, targets = next(restoration_iter)
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = restoration_head(encoder(inputs))
            loss = restoration_loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        task_losses[task].append(loss.item())

    # === Validation Phase ===
    encoder.eval()
    with torch.no_grad():
        rot_val_loss = sum(rotation_loss_fn(rotation_head(encoder(x.to(DEVICE))), y.to(DEVICE)).item()
                           for x, y in rotation_val_loader) / len(rotation_val_loader)
        jig_val_loss = sum(jigsaw_loss_fn(jigsaw_head(encoder(x.to(DEVICE))), y.to(DEVICE)).item()
                           for x, y in jigsaw_val_loader) / len(jigsaw_val_loader)
        inp_val_loss = sum(restoration_loss_fn(restoration_head(encoder(x.to(DEVICE))), y.to(DEVICE)).item()
                           for x, y in restoration_val_loader) / len(restoration_val_loader)

    val_losses['rotation'].append(rot_val_loss)
    val_losses['jigsaw'].append(jig_val_loss)
    val_losses['restoration'].append(inp_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Rotation: {rot_val_loss:.4f}, Jigsaw: {jig_val_loss:.4f}, Restoration: {inp_val_loss:.4f}")

    score = inp_val_loss + 0.5 * jig_val_loss + 0.25 * rot_val_loss
    if score < best_score:
        best_score = score
        patience_counter = 0
        torch.save(encoder.state_dict(), 'best_encoder_multitask.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

# === Plot loss trends ===
for task, losses in task_losses.items():
    plt.plot(losses, label=f"Train {task}")
for task, losses in val_losses.items():
    plt.plot(losses, label=f"Val {task}", linestyle='--')

plt.title("Multitask SSL: Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("multitask_ssl_loss_curve.png")
plt.show()
