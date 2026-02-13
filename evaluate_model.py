import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import torch.nn as nn

# ---------------- CONFIG ----------------
TEST_DIR = "data/testing"
MODEL_PATH = "models/weights/best_resnet50_brain_tumor.pth"
BATCH_SIZE = 16
NUM_CLASSES = 4

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Current directory:", os.getcwd())
print("Test folder exists:", os.path.exists(TEST_DIR))
print("Model path exists:", os.path.exists(MODEL_PATH))

# ---------------- TRANSFORMS ----------------
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- DATASET ----------------
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Detected classes:", test_dataset.classes)

# ---------------- LOAD MODEL (RESNET50) ----------------
print("🔍 Loading ResNet50 model...")

model = models.resnet50(weights=None)  # IMPORTANT: weights=None for evaluation
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

print("✅ Model loaded successfully")

# ---------------- EVALUATION ----------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ---------------- METRICS ----------------
accuracy = accuracy_score(y_true, y_pred)

print("\n✅ Overall Test Accuracy: {:.2f}%\n".format(accuracy * 100))

print("📊 Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=test_dataset.classes
))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
