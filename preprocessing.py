import cv2
import numpy as np

def enhance_mri_contrast(image_path):
    """
    Stronger Color Contrast Channeling using CLAHE + LAB amplification
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path")

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Strong CLAHE on L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Recombine
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 🔥 Stronger fusion (more enhancement)
    output = cv2.addWeighted(img, 0.3, enhanced, 0.7, 0)

    return output
