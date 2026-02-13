# 🧠 TumorDetect AI – Brain Tumor Detection System

An AI-powered web application for automated brain tumor detection, classification, and segmentation from MRI scans.
Developed as a final-year B.Tech project using Deep Learning, Explainable AI, and Flask.

---

## 🚀 Project Overview

TumorDetect AI assists in early detection of brain tumors by analyzing MRI images.
The system performs:

* Tumor classification
* Pixel-level tumor segmentation
* Tumor measurement (location, size, area)
* Explainable AI visualization
* Automatic medical report generation (PDF)

---

## 🧠 AI Models Used

### 1. Classification Model

* Architecture: **ResNet50 (fine-tuned)**
* Classes:

  * Glioma
  * Meningioma
  * Pituitary
  * No Tumor
* Accuracy: ~80%

### 2. Segmentation Model

* Architecture: **U-Net**
* Output: Pixel-level tumor mask
* Used for:

  * Tumor boundary detection
  * Size and area estimation

### 3. Explainable AI

* Grad-CAM visualization
* Highlights regions influencing prediction

---

## 🛠 Tech Stack

### Backend

* Python
* Flask
* PyTorch
* OpenCV

### Frontend

* HTML
* CSS
* JavaScript

### Report Generation

* ReportLab (PDF)

---

## 🔬 Features

* MRI image upload
* Contrast enhancement (CLAHE)
* Tumor classification
* Tumor segmentation
* Bounding box and measurement
* Grad-CAM heatmap
* Patient details input
* Automatic medical report generation
* Downloadable PDF report

---

## 📁 Project Structure

```
brain_tumor_website/
│
├── app.py
├── segmentation.py
├── classification.py
├── preprocessing.py
├── explainable_ai.py
├── report_generator.py
├── xai_text.py
│
├── models/
│   └── weights/
│
├── static/
├── templates/
├── uploads/
├── reports/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation (Local Setup)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-ai.git
cd brain-tumor-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000
```

---

## ☁️ Deployment

This project can be deployed on:

* Render (recommended)
* Railway
* PythonAnywhere

Deployment flow:

```
Local Code → GitHub → Render → Live Website
```

---

## 📄 Medical Report

The system automatically generates a clinical-style PDF containing:

* Patient details
* Tumor type
* Risk level
* Model confidence
* Tumor measurements
* MRI images
* AI explanation

---

## 📱 Mobile Support

After deployment, the system can be accessed from:

* Mobile phones
* Tablets
* Laptops

Using the public URL.

---

## 🎯 Project Goals

* Improve tumor detection accuracy
* Provide explainable AI predictions
* Generate clinically realistic reports
* Create a deployable medical AI system

---

## ⚠️ Disclaimer

This system is developed for **academic and research purposes only**.
It is not intended for clinical or medical decision-making.

---

## 👨‍💻 Team Members

* Raviprakash B P
* Darshan S K
* Pavan Kumar M
* Ravindra G

Alliance University
B.Tech Computer Science & Engineering
2026
