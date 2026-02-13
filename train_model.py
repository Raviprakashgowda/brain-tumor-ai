import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import os

# ---------------- CONFIG ----------------
TRAIN_DIR = "data/training"
TEST_DIR  = "data/testing"
SAVE_PATH = "models/weights/best_resnet50_brain_tumor.pth"

BATCH_SIZE = 16
EPOCHS = 45
PATIENCE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- TRANSFORMS (MRI SAFE) ----------------
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- DATASETS ----------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Class order:", train_dataset.classes)

# ---------------- CLASS WEIGHTS ----------------
class_counts = np.bincount(train_dataset.targets)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", class_weights)

# ---------------- MODEL (RESNET50) ----------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze deeper layers
for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

# ---------------- LOSS & OPTIMIZER ----------------
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = optim.Adam([
    {"params": model.layer3.parameters(), "lr": 1e-5},
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4}
])

# ---------------- TRAINING LOOP ----------------
best_acc = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ---------------- VALIDATION ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {running_loss:.4f} | "
        f"Test Accuracy: {test_acc*100:.2f}%"
    )

    # ---------------- EARLY STOPPING ----------------
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), SAVE_PATH)
        patience_counter = 0
        print("✅ Best model saved")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹ Early stopping triggered")
            break

print(f"\n🎯 Training complete. Best Accuracy: {best_acc*100:.2f}%")
print("📦 Model saved at:", SAVE_PATH)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import os

# ---------------- CONFIG ----------------
TRAIN_DIR = "data/training"
TEST_DIR  = "data/testing"
SAVE_PATH = "models/weights/best_resnet50_brain_tumor.pth"

BATCH_SIZE = 16
EPOCHS = 45
PATIENCE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- TRANSFORMS (MRI SAFE) ----------------
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- DATASETS ----------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Class order:", train_dataset.classes)

# ---------------- CLASS WEIGHTS ----------------
class_counts = np.bincount(train_dataset.targets)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", class_weights)

# ---------------- MODEL (RESNET50) ----------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze deeper layers
for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

# ---------------- LOSS & OPTIMIZER ----------------
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = optim.Adam([
    {"params": model.layer3.parameters(), "lr": 1e-5},
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4}
])

# ---------------- TRAINING LOOP ----------------
best_acc = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ---------------- VALIDATION ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {running_loss:.4f} | "
        f"Test Accuracy: {test_acc*100:.2f}%"
    )

    # ---------------- EARLY STOPPING ----------------
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), SAVE_PATH)
        patience_counter = 0
        print("✅ Best model saved")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹ Early stopping triggered")
            break

print(f"\n🎯 Training complete. Best Accuracy: {best_acc*100:.2f}%")
print("📦 Model saved at:", SAVE_PATH)
