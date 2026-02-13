import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Fallback: Tiny UNet (works out-of-the-box) ==========
# Replace this with your real TransUNet class when you have it.
class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16,16,3,padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32,32,3,padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(64,64,3,padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32,32,3,padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16,16,3,padding=1), nn.ReLU())
        self.outc = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)        # B,16,H,W
        p1 = self.pool1(e1)      # B,16,H/2,W/2
        e2 = self.enc2(p1)       # B,32,H/2,W/2
        p2 = self.pool2(e2)      # B,32,H/4,W/4

        b  = self.bottleneck(p2) # B,64,H/4,W/4

        u2 = self.up2(b)         # B,32,H/2,W/2
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)        # B,16,H,W
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        out = self.outc(d1)      # B,1,H,W
        return out

# ========== Placeholder for real TransUNet ==========
# When you have TransUNet code + weights:
# class TransUNet(nn.Module):
#     def __init__(...): ...
#     def forward(...): ...
