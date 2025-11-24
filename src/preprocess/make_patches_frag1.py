# src/preprocess/make_patches_frag1.py
import os
from pathlib import Path

import numpy as np
from skimage import io
from tqdm import tqdm


def load_surface_volume(surface_dir: Path):
    """
    Citeste toate .tif-urile din surface_volume si le stivuieste intr-un volum 3D.
    Shape final: (D, H, W), cu D = numar de slice-uri.
    """
    tif_files = sorted(surface_dir.glob("*.tif"))
    if not tif_files:
        raise RuntimeError(f"No .tif files found in {surface_dir}")

    slices = []
    for f in tqdm(tif_files, desc="Loading surface_volume slices"):
        img = io.imread(str(f)).astype(np.float32)
        slices.append(img)

    volume = np.stack(slices, axis=0)  # (D, H, W)
    return volume


def load_labels(raw_dir: Path):
    """
    Citeste inklabels.png si mask.png.
    Returneaza:
      ink: (H, W) in {0,1}
      mask: (H, W) in {0,1}
    """
    ink_path = raw_dir / "inklabels.png"
    mask_path = raw_dir / "mask.png"

    ink = io.imread(str(ink_path))
    mask = io.imread(str(mask_path))

    # Du in 0/1
    ink = (ink > 0).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    return ink, mask


def get_mask_bbox(mask: np.ndarray, margin: int = 16):
    """
    Calculeaza bounding box pentru zona unde mask == 1.
    intoarce (y_min, y_max, x_min, x_max), capete incluse/excluse (slice-friendly).
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise RuntimeError("Mask is empty, no valid region found.")

    y_min = max(0, ys.min() - margin)
    y_max = min(mask.shape[0], ys.max() + margin + 1)
    x_min = max(0, xs.min() - margin)
    x_max = min(mask.shape[1], xs.max() + margin + 1)

    return y_min, y_max, x_min, x_max


def sample_patch_centers(ink: np.ndarray, mask: np.ndarray, n_pos: int, n_bg: int, rng: np.random.Generator):
    """
    Alege pozitii centrale pentru patch-uri:
      - n_pos centre in zone cu cerneala (ink == 1)
      - n_bg centre in zone fara cerneala, dar cu mask == 1
    Returneaza doua liste: [(y,x), ...] pentru positive si background.
    """
    # pozitii cu cerneala
    ys_pos, xs_pos = np.where(ink > 0)
    pos_coords = list(zip(ys_pos, xs_pos))
    rng.shuffle(pos_coords)

    if len(pos_coords) > n_pos:
        pos_coords = pos_coords[:n_pos]

    # pozitii fara cerneala, dar in zona valida
    ys_bg, xs_bg = np.where((ink == 0) & (mask == 1))
    bg_coords = list(zip(ys_bg, xs_bg))
    rng.shuffle(bg_coords)

    if len(bg_coords) > n_bg:
        bg_coords = bg_coords[:n_bg]

    return pos_coords, bg_coords


def extract_patch(volume: np.ndarray,
                  ink: np.ndarray,
                  y_center: int,
                  x_center: int,
                  patch_size: int):
    """
    Taie un patch 3D din volum si un patch 2D din ink, centrate in (y_center, x_center).
    volum shape: (D, H, W)
    ink shape: (H, W)
    """
    D, H, W = volume.shape
    ps = patch_size

    y0 = max(0, y_center - ps // 2)
    y1 = y0 + ps
    if y1 > H:
        y1 = H
        y0 = H - ps

    x0 = max(0, x_center - ps // 2)
    x1 = x0 + ps
    if x1 > W:
        x1 = W
        x0 = W - ps

    vol_patch = volume[:, y0:y1, x0:x1]  # (D, ps, ps)
    ink_patch = ink[y0:y1, x0:x1]        # (ps, ps)

    assert vol_patch.shape[1] == ps and vol_patch.shape[2] == ps, vol_patch.shape
    assert ink_patch.shape[0] == ps and ink_patch.shape[1] == ps, ink_patch.shape

    return vol_patch, ink_patch


def normalize_volume(volume: np.ndarray, clip_low=1.0, clip_high=99.0):
    """
    Normalizare simpla: clip la percentila [clip_low, clip_high], apoi scale in [0,1].
    """
    lo = np.percentile(volume, clip_low)
    hi = np.percentile(volume, clip_high)
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo + 1e-8)
    return volume.astype(np.float32)


def main():
    project_root = Path(__file__).resolve().parents[2]  # mergem 2 niveluri in sus din src/preprocess
    raw_dir = project_root / "data" / "raw" / "fragment1"
    surface_dir = raw_dir / "surface_volume"

    processed_root = project_root / "data" / "processed"
    train_dir = processed_root / "train"
    val_dir = processed_root / "val"
    test_dir = processed_root / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", project_root)
    print("Raw dir:", raw_dir)
    print("Surface dir:", surface_dir)

    # 1. Incarcam volum si labels
    volume = load_surface_volume(surface_dir)   # (D, H, W)
    ink, mask = load_labels(raw_dir)           # (H, W), (H, W)

    D, H, W = volume.shape
    print(f"Volume shape: D={D}, H={H}, W={W}")
    print(f"Ink positives: {ink.sum()} pixels")

    # Normalizare volum
    volume = normalize_volume(volume)

    # BBox masca, ca sa nu generam patch-uri in zone complet goale
    y_min, y_max, x_min, x_max = get_mask_bbox(mask)
    print("Mask bbox (y_min, y_max, x_min, x_max):", y_min, y_max, x_min, x_max)

    # Taiem volume / ink / mask la bounding box pentru eficienta
    volume = volume[:, y_min:y_max, x_min:x_max]
    ink = ink[y_min:y_max, x_min:x_max]
    mask = mask[y_min:y_max, x_min:x_max]

    D, H, W = volume.shape
    print(f"After cropping to bbox: D={D}, H={H}, W={W}")

    # 2. Sampling de patch-uri
    rng = np.random.default_rng(42)

    patch_size = 256
    n_pos = 400   # patch-uri cu cerneala
    n_bg = 400    # patch-uri background

    pos_centers, bg_centers = sample_patch_centers(ink, mask, n_pos, n_bg, rng)
    print(f"Sampled {len(pos_centers)} positive centers and {len(bg_centers)} background centers.")

    all_patches = []
    # tag = 1 pentru patch-uri cu cerneala, 0 pentru background (info optionala)
    for (y, x) in pos_centers:
        all_patches.append(((y, x), 1))
    for (y, x) in bg_centers:
        all_patches.append(((y, x), 0))

    rng.shuffle(all_patches)

    # 3. Impartim in train / val / test: 70 / 15 / 15
    n_total = len(all_patches)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    # restul test
    indices_train = all_patches[:n_train]
    indices_val = all_patches[n_train:n_train + n_val]
    indices_test = all_patches[n_train + n_val:]

    print(f"Total patches: {n_total}  -> train={len(indices_train)}, val={len(indices_val)}, test={len(indices_test)}")

    # 4. Salvam patch-urile in .npz
    def save_split(indices, out_dir: Path, split_name: str):
        counter = 0
        for (y, x), tag in tqdm(indices, desc=f"Saving {split_name} patches"):
            try:
                vol_patch, ink_patch = extract_patch(volume, ink, y, x, patch_size)
            except AssertionError as e:
                # in caz rarisim in care patch-ul nu are dimensiune corecta
                print("Skipping patch at", y, x, "due to shape error:", e)
                continue

            # (D, H, W) -> salvam ca atare; ScrollsPatchDataset va adauga channel-ul
            out_path = out_dir / f"{split_name}_patch_{counter:04d}.npz"
            np.savez_compressed(
                out_path,
                volume=vol_patch.astype(np.float32),
                mask=ink_patch.astype(np.float32),
                tag=np.int8(tag),
            )
            counter += 1

        print(f"Saved {counter} {split_name} patches to {out_dir}")

    save_split(indices_train, train_dir, "train")
    save_split(indices_val, val_dir, "val")
    save_split(indices_test, test_dir, "test")


if __name__ == "__main__":
    main()
