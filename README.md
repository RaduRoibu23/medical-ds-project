# Ink Detection on Herculaneum Papyri – Fragment 1 (Project Readme)

This repository contains the code for our Medical Data Science project on
ink detection in X-ray micro-CT scans of carbonized Herculaneum papyri.

We currently work only with Fragment 1 (PHercParis2Fr47) from the Vesuvius / EduceLab dataset.
The goal is to detect carbon ink on the exposed papyrus surface using a 3D CT volume as input
and a 2D ink mask as target.

---

## 1. Repository structure

```text
project-root/
  data/
    raw/
      fragment1/                     # raw Vesuvius Fragment 1 data
    processed/
      train/                         # .npz patches for training
      val/                           # .npz patches for validation
      test/                          # .npz patches for test
  reports/
    M2/                              # M2 IEEE section (dataset + baseline)
    ...
  src/
    datasets/
      scrolls_patches.py             # PyTorch dataset for .npz patches
    models/
      unet3d_baseline.py             # 3D U-Net baseline model
    preprocess/
      make_patches_frag1.py          # raw -> processed patches for fragment 1
    utils/
      metrics.py                     # evaluation metrics (F0.5, pFM, PSNR)
    train_baseline.py                # training script for baseline model
    eval_baseline.py                 # evaluation on test split
  requirements.txt
  README.md                          # this file
```

---

## 2. Data folders and files

### 2.1 `data/raw/fragment1/`

This folder contains the **original Fragment 1 data** downloaded from the Vesuvius server:

- `surface_volume/*.tif` – 2D CT slices of the papyrus surface.
- `inklabels.png` – 2D binary mask where ink pixels are 1.
- `mask.png` – 2D binary mask of the valid papyrus surface (to ignore air/background).
- `ir.png` – IR image (used only for figures / qualitative visualisation, not in training).

This raw data is never modified by our code.  
All preprocessing outputs go into `data/processed/`.

The fragment we use in the project is:

```text
fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/
```

and is mapped locally into `data/raw/fragment1/`.

---

### 2.2 `data/processed/{train,val,test}/`

After running the preprocessing script, we obtain:

- `data/processed/train/train_patch_XXXX.npz`
- `data/processed/val/val_patch_XXXX.npz`
- `data/processed/test/test_patch_XXXX.npz`

Each `.npz` file contains three arrays:

- `volume` – shape `(D, H, W)` float32  
  A downsampled 3D CT patch (D depth slices, H×W spatial resolution).
- `mask` – shape `(H, W)` float32  
  The corresponding ink mask for the patch (1 = ink, 0 = non‑ink).
- `tag` – int8  
  Indicator of how the patch was sampled:
  - `1` – positive patch (sampled around an ink pixel)
  - `0` – background patch (sampled from non‑ink area but inside mask)

The patch dataset is what the PyTorch `Dataset` reads during training and evaluation.

---

## 3. Source code – files and roles

### 3.1 `src/preprocess/make_patches_frag1.py`

**Role:** Convert raw Fragment 1 data into a compact **patch dataset** suitable for training.

This script is heavily optimised for limited RAM (WSL):

1. **Load and downsample labels**
   - Reads `inklabels.png` and `mask.png` from `data/raw/fragment1/`.
   - Handles both grayscale and colour PNGs (if 3 channels, uses the first).
   - Binarises them into `{0,1}`.
   - Downsamples them by a fixed integer factor (currently `ds_factor = 8`) using subsampling:
     ```python
     ink_ds = ink[::ds_factor, ::ds_factor]
     mask_ds = mask[::ds_factor, ::ds_factor]
     ```
   - Result: smaller label images `ink_ds` and `mask_ds` used for patch sampling.

2. **Compute Region of Interest (ROI)**
   - Finds all pixels where `mask_ds == 1`.
   - Builds a bounding box around them (with a small margin) using:
     ```python
     get_mask_bbox(mask_ds, margin=4)
     ```
   - Crops `ink_ds` and `mask_ds` to this ROI (`ink_roi`, `mask_roi`), discarding large empty borders.

3. **Load and downsample CT volume**
   - Reads a small subset of CT slices (currently `max_slices = 4`) from
     `data/raw/fragment1/surface_volume/`.
   - Each slice is downsampled by the same factor `ds_factor` using subsampling:
     ```python
     img_ds = img[::ds_factor, ::ds_factor]
     ```
   - Slices are stacked into a 3D array `volume_ds` of shape `(D, H_ds, W_ds)`.

4. **Crop CT volume to ROI**
   - Applies the same ROI as for labels:
     ```python
     volume_roi = volume_ds[:, y_min:y_max, x_min:x_max]
     ```
   - Now `volume_roi` and `ink_roi` / `mask_roi` are spatially aligned and much smaller.

5. **Normalise CT volume**
   - Clips intensities to `[1%, 99%]` percentiles and rescales to `[0,1]`:
     ```python
     volume_roi = normalize_volume(volume_roi)
     ```

6. **Sample patch centres**
   - Uses a random number generator with fixed seed (`rng = np.random.default_rng(42)`).
   - Samples:
     - `n_pos` positive centres from pixels where `ink_roi == 1`.
     - `n_bg` background centres where `mask_roi == 1` and `ink_roi == 0`.
   - Returns two shuffled lists of `(y, x)` coordinates.

7. **Extract patches**
   - For each centre `(y, x)`, extracts:
     - `vol_patch` of size `(D, P, P)` from `volume_roi`.
     - `ink_patch` of size `(P, P)` from `ink_roi`.
   - The patch size is currently `P = 128`. If ROI is smaller than this, the script automatically reduces `P`.

8. **Split into train / val / test**
   - Concatenates positive and background patches and shuffles them.
   - Splits into:
     - `train` – 70%
     - `val` – 15%
     - `test` – 15%

9. **Save `.npz` patch files**
   - For each split, saves compressed `.npz` files with arrays `volume`, `mask`, `tag` in:
     ```text
     data/processed/train/
     data/processed/val/
     data/processed/test/
     ```

**Usage:**

From the project root:

```bash
python3 src/preprocess/make_patches_frag1.py
```

---

### 3.2 `src/datasets/scrolls_patches.py`

**Role:** Provide a PyTorch `Dataset` for the patch data created by `make_patches_frag1.py`.

Main behaviour:

- In `__init__`:
  - Takes `root_dir` (usually `"data/processed"`) and a `split` (`"train"`, `"val"`, `"test"`).
  - Builds the path `<root_dir>/<split>/` and gathers all `.npz` files there.

- In `__getitem__(idx)`:
  - Loads one `.npz` file.
  - Reads:
    - `volume` `(D, H, W)` → converts to tensor and adds a channel dimension: `(1, D, H, W)`.
    - `mask` `(H, W)` → converts to tensor and adds a channel dimension: `(1, H, W)`.
  - Ensures the volume is normalised to `[0,1]` (safety step).
  - Returns `(volume, mask)` as tensors ready for the model.

---

### 3.3 `src/models/unet3d_baseline.py`

**Role:** Define the **3D U‑Net baseline model**.

Key points:

- Input: `(B, 1, D, H, W)`  
  (batch size, channels, depth, height, width)
- Output: `(B, 1, D, H, W)` logits.
- Because we have very few depth slices (`D = 4`), the architecture **does not pool along depth**:
  - Uses `MaxPool3d(kernel_size=(1,2,2))` and `ConvTranspose3d(kernel_size=(1,2,2))`.
  - This keeps `D` constant while downsampling / upsampling only in H and W.

Blocks:

- `DoubleConv3d(in_channels, out_channels)` – two `Conv3d + BatchNorm3d + ReLU` layers.
- Encoder:
  - `enc1 -> pool1` → `enc2 -> pool2` → `enc3 -> pool3` → `bottleneck`.
- Decoder:
  - `up3 -> concat with enc3 -> dec3`
  - `up2 -> concat with enc2 -> dec2`
  - `up1 -> concat with enc1 -> dec1`
- Final 1×1×1 convolution `out_conv` to map to 1 output channel.

During training and evaluation we apply the loss and metrics only to the **central depth slice**:
```python
logits_2d = logits_3d[:, :, depth_central, :, :]
```

---

### 3.4 `src/utils/metrics.py`

**Role:** Implement evaluation metrics for ink detection.

Functions:

- `binarize_probs(probs, threshold=0.5)`  
  Converts probability maps in `[0,1]` to binary maps `{0,1}`.

- `f_beta_score(target, prediction, beta=0.5, eps=1e-8)`  
  Computes pixel‑wise F\_beta score (we use `beta=0.5` to weigh precision higher than recall).

- `psnr(prediction, target, max_val=1.0)`  
  Computes Peak Signal‑to‑Noise Ratio between predicted and ground‑truth maps.

- `pseudo_f_measure(target, prediction_probs, threshold=0.5, eps=1e-8)`  
  Approximate pseudo F‑measure (pFM) by computing F1 on the binarised predictions.

These metrics are used in both training and evaluation:

- `train_baseline.py` – validation metrics after each epoch.
- `eval_baseline.py` – final test metrics.

---

### 3.5 `src/train_baseline.py`

**Role:** Train the 3D U‑Net baseline model on Fragment 1 patches.

Main steps:

1. **Setup**
   - Defines hyperparameters:
     - `batch_size` (can be reduced if GPU memory is low),
     - `num_epochs`,
     - learning rate,
     - `base_channels` for the U‑Net.
   - Selects device (`cuda` if available, otherwise `cpu`).

2. **Load datasets**
   - Creates:
     ```python
     train_dataset = ScrollsPatchDataset("data/processed", split="train")
     val_dataset   = ScrollsPatchDataset("data/processed", split="val")
     ```
   - Detects depth `D` from a sample patch and sets:
     ```python
     depth_central = D // 2
     ```

3. **Create data loaders**
   - `DataLoader` for train and val sets (with shuffling for train).

4. **Initialise model, loss and optimiser**
   - Model: `UNet3D(in_channels=1, base_channels=16)`
   - Loss: combination of
     - `BCEWithLogitsLoss`
     - `Dice loss` on the 2D slice
   - Optimiser: `AdamW`.

5. **Training loop**
   - For each epoch:
     - Calls `train_one_epoch(...)`:
       - Sends batches to device.
       - Forwards through model.
       - Extracts central slice `logits_2d`.
       - Computes combined loss and backpropagates.
     - Calls `evaluate(...)` on the validation set:
       - Computes F\_{0.5}, approximate pFM, and PSNR.
     - If validation F\_{0.5} improves, saves a checkpoint:
       ```python
       checkpoints/unet3d_baseline_best.pt
       ```

This script provides the **baseline model and validation metrics** used in **M2**.

**Usage:**

From the `src` folder (or project root with correct module path):

```bash
cd src
python3 train_baseline.py
```

---

### 3.6 `src/eval_baseline.py`

**Role:** Evaluate the best baseline model on the test split.

Steps:

1. Loads the test dataset:
   ```python
   test_dataset = ScrollsPatchDataset("data/processed", split="test")
   ```
2. Detects depth `D` and central slice `depth_central`.
3. Creates the same `UNet3D` architecture and loads weights from:
   ```text
   checkpoints/unet3d_baseline_best.pt
   ```
4. Runs inference on all test patches.
5. Computes and prints final metrics:
   - F\_{0.5}
   - pseudo F‑measure (pFM)
   - PSNR

These numbers are used in the M2 report as **baseline performance** and later for comparison in **M3**.

**Usage:**

```bash
cd src
python3 eval_baseline.py
```

---

## 4. Order of use (what to run and when)

### Step 0 – Install dependencies

From project root:

```bash
pip install -r requirements.txt
```

### Step 1 – Prepare raw data

Make sure the following exists:

```text
data/raw/fragment1/
  surface_volume/*.tif
  inklabels.png
  mask.png
  ir.png
```

If the folder is named `Frag1` instead of `fragment1`, either rename it or adjust `make_patches_frag1.py`.

### Step 2 – Preprocessing (raw -> patches)

From project root:

```bash
python3 src/preprocess/make_patches_frag1.py
```

This will create:

```text
data/processed/train/*.npz
data/processed/val/*.npz
data/processed/test/*.npz
```

### Step 3 – Train baseline model

From `src/`:

```bash
cd src
python3 train_baseline.py
```

This script trains the 3D U‑Net baseline and saves the best model checkpoint:

```text
checkpoints/unet3d_baseline_best.pt
```

### Step 4 – Evaluate baseline model

From `src/`:

```bash
python3 eval_baseline.py
```

This prints the final test metrics (F\_{0.5}, pFM, PSNR).  
These are the numbers to include in the **M2 report** under “Baseline Results” and later to compare with improved models in **M3**.

---

## 5. How this fits into the milestones

- **M2 – Dataset Collection and Baseline Results**
  - Preprocessing: `make_patches_frag1.py`
  - Baseline model: `UNet3D` (`unet3d_baseline.py`)
  - Training: `train_baseline.py`
  - Evaluation: `eval_baseline.py`
  - Metrics: `metrics.py`
  - Dataset description (Fragment 1) and preprocessing details → to be written in `reports/M2/`.

---
