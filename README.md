# M3: Geometric Feature Injection for Ink Detection on Herculaneum Papyri

**Authors:** Roibu Radu Gheorghe, Silvestru Radu  
**Course:** Medical Data Science  
**Date:** December 19, 2025  

## 📌 Overview
This project (Milestone 3) implements a **Geometric-Aware 3D U-Net** for detecting ink in micro-CT scans of carbonized scrolls. Unlike the baseline (M2), which relies solely on X-ray intensity, our approach injects explicit 3D surface normals (gradients) and uses Test-Time Augmentation (TTA) to separate ink texture from papyrus fibers.

**Key Achievements:**
- **F0.5 Score:** 0.7184 (vs Baseline 0.6565)
- **Precision:** 0.8343 (High reduction in false positives)
- **PSNR:** 32.05 dB

## 📂 Submission Structure
- `M3_Ink_Detection.ipynb`: The complete source code (Data Loading, Model, Training, TTA Evaluation).
- `M3_Report.pdf`: The documentation explaining the methodology and results (IEEE Format).
- `README.md`: This file.

## 🛠️ Prerequisites
The code is written in Python 3.8+ and relies on PyTorch. We recommend running this on a machine with a GPU (NVIDIA T4, RTX 30xx, or better).

### Required Libraries
```bash
pip install torch torchvision opencv-python tqdm scipy matplotlib numpy
DATA_ROOT = '/kaggle/input/vesuvius-challenge-ink-detection'
