import cv2
import numpy as np

path = "dataset/masks/1.png"   # change this if needed
mask = cv2.imread(path, 0)

if mask is None:
    print("Mask not found. Check path:", path)
else:
    print("Min value:", np.min(mask))
    print("Max value:", np.max(mask))
    print("Unique values:", np.unique(mask))
