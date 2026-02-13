import os
import torch
import numpy as np
import cv2
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# -------------------------------------------------
# Basic U-Net (same structure used during training)
# -------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TrainedUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        # Encoder
        self.d1 = DoubleConv(in_ch, 64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)

        # Bridge
        self.bridge = DoubleConv(256, 512)

        # Decoder
        self.u1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c1 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c2 = DoubleConv(256, 128)

        self.u3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c3 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        d1 = self.d1(x)
        p1 = self.p1(d1)

        d2 = self.d2(p1)
        p2 = self.p2(d2)

        d3 = self.d3(p2)
        p3 = self.p3(d3)

        bridge = self.bridge(p3)

        u1 = self.u1(bridge)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.c1(u1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.c2(u2)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.c3(u3)

        return self.out(u3)

# -------------------------------------------------
# Load Segmentation Model
# -------------------------------------------------
def load_segmentation_model(weights_path):
    model = TrainedUNet(in_ch=3, out_ch=1)
    sd = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    print("✅ Segmentation model loaded correctly")
    return model



# -------------------------------------------------
# Tumor Segmentation
# -------------------------------------------------
def segment_tumor(model, bgr_img, input_size=256, thresh=0.3):
    H, W = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    x = transform(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = (prob >= thresh).astype(np.uint8) * 255

    return mask


# -------------------------------------------------
# Overlay Mask (clean tumor region)
# -------------------------------------------------
def overlay_mask(bgr_img, mask):
    output = bgr_img.copy()

    # Fill tumor region
    output[mask > 0] = (0, 0, 255)

    # Draw boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 0, 200), 2)

    return output


# -------------------------------------------------
# Tumor Measurements
# -------------------------------------------------
def compute_tumor_stats(mask, pixel_spacing=0.05):
    """
    Computes tumor bounding box, pixel size, and real-world size.
    pixel_spacing: cm per pixel (default ≈0.05 cm = 0.5 mm per pixel)
    """

    coords = np.column_stack(np.where(mask > 0))

    if len(coords) == 0:
        return {
            "location": None,
            "width_px": 0,
            "height_px": 0,
            "area_px": 0,
            "width_cm": 0,
            "height_cm": 0,
            "area_cm2": 0
        }

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    width_px = x_max - x_min
    height_px = y_max - y_min
    area_px = int(np.sum(mask > 0))

    # Convert to cm
    width_cm = width_px * pixel_spacing
    height_cm = height_px * pixel_spacing
    area_cm2 = area_px * (pixel_spacing ** 2)

    return {
        "location": (int(x_min), int(y_min), int(x_max), int(y_max)),
        "width_px": int(width_px),
        "height_px": int(height_px),
        "area_px": area_px,
        "width_cm": round(width_cm, 2),
        "height_cm": round(height_cm, 2),
        "area_cm2": round(area_cm2, 2)
    }
def draw_bounding_box(image, stats, color=(0, 0, 255), thickness=2):
    """
    Draw bounding box around tumor.
    """
    if not stats or not stats.get("location"):
        return image

    x1, y1, x2, y2 = stats["location"]
    boxed = image.copy()
    cv2.rectangle(boxed, (x1, y1), (x2, y2), color, thickness)
    return boxed
