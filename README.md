# M3: Geometric Feature Injection for Ink Detection on Herculaneum Papyri

**Authors:** Roibu Radu Gheorghe, Silvestru Radu  
**Course:** Medical Data Science  
**Date:** December 19, 2025  

## 📌 Overview
This project (Milestone 3) implements a **Geometric-Aware 3D U-Net** for detecting ink in micro-CT scans of carbonized scrolls. Unlike the baseline (M2), which relied solely on X-ray intensity, our approach injects explicit 3D surface normals (gradients) and uses Test-Time Augmentation (TTA) to separate ink texture from papyrus fibers.

**Key Achievements:**
- **F0.5 Score:** 0.7184 (vs Baseline 0.6565)
- **Precision:** 0.8343 (Significant reduction in false positives)
- **PSNR:** 32.05 dB

## 📂 Submission Structure
- `M3_Ink_Detection.ipynb`: The complete source code (Data Loading, Model, Training, TTA Evaluation).
- `M3_Report.pdf`: The documentation explaining the methodology and results (IEEE Format).
- `README.md`: This file.

## 🛠️ Prerequisites
The code is written in Python 3.8+ and relies on PyTorch. We recommend running this on a machine with a GPU (NVIDIA T4, RTX 30xx, or better).

### Required Libraries
    pip install torch torchvision opencv-python tqdm scipy matplotlib numpy

## ⚙️ Data Setup
The code expects the **Vesuvius Challenge Ink Detection** dataset structure (Fragment 1).

**Option A: Running on Kaggle (Recommended)**
1. Upload the notebook to Kaggle.
2. Add the "Vesuvius Challenge - Ink Detection" dataset to your notebook.
3. The code will automatically detect the data at `/kaggle/input/vesuvius-challenge-ink-detection`.

**Option B: Running Locally**
1. Download the dataset from Kaggle.
2. Extract it to a folder (e.g., `./data`).
3. **Crucial Step:** Open the notebook and find the **Configuration Cell** (Cell #2). Update the `DATA_ROOT` variable:
   
    # Change this path to your local folder
    DATA_ROOT = './data/vesuvius-challenge-ink-detection' 

## 🚀 How to Run
1. **Open the Notebook:** Launch Jupyter Lab or VS Code and open `M3_Ink_Detection.ipynb`.
2. **Verify Data Path:** Ensure `DATA_ROOT` in the second cell points to the correct dataset location.
3. **Execute All Cells:**
   - In Jupyter Lab: Go to **Run** -> **Run All Cells**.
   - In VS Code: Click **Run All**.
4. **Wait for Training:** The training loop runs for **100 epochs**. On a standard GPU (T4), this takes approximately **1-1.5 hours**.
   - *Note:* Progress bars will show the status of each epoch.

## 🔄 Execution Flow
The notebook proceeds in three main stages automatically:

1.  **Dataset Initialization:**
    - Loads Fragment 1 images and volume slices.
    - Computes 3D gradients on-the-fly during data loading.
    - Pre-calculates balanced sampling indices (50% Ink / 50% Background) to handle class imbalance.

2.  **Training:**
    - Initializes the custom 4-channel 3D U-Net.
    - Trains using `BCEWithLogitsLoss`.
    - Automatically saves the best checkpoint (`best_model_m3.pth`) whenever the Validation F0.5 score improves.

3.  **Final Evaluation (TTA):**
    - Loads the best model checkpoint (`best_model_m3.pth`).
    - Runs the **Test-Time Augmentation (TTA)** loop:
        1. Predicts on original image.
        2. Predicts on horizontally flipped image.
        3. Predicts on vertically flipped image.
        4. Averages the probabilities.
    - Prints the final performance metrics.

## 📊 Expected Output
At the very end of the execution, the notebook will print the final results block:

    ----------------------------------------
    🏆 FINAL M3 RESULTS (TTA Enabled)
    ----------------------------------------
    F0.5 Score:  0.7184
    Precision:   0.8343
    Recall:      0.4618
    pFM Score:   0.5626
    PSNR:        32.05 dB
    ----------------------------------------
