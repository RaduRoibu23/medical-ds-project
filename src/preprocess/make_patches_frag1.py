# src/preprocess/make_patches_frag1.py
from pathlib import Path

import numpy as np
from skimage import io
from skimage.transform import resize
from tqdm import tqdm

def load_labels_resized(raw_dir: Path, scale_factor: float = 0.25):
    """
    Citeste inklabels.png si mask.png si le redimensioneaza pentru a reduce memoria.
    Returneaza:
      ink: (H_ds, W_ds) in {0,1}
      mask: (H_ds, W_ds) in {0,1}
    """
    ink_path = raw_dir / "inklabels.png"
    mask_path = raw_dir / "mask.png"

    ink = io.imread(str(ink_path))
    mask = io.imread(str(mask_path))

    # Daca sunt RGB/RGBA (H,W,3/4), luam doar primul canal
    if ink.ndim == 3:
        ink = ink[..., 0]
    if mask.ndim == 3:
        mask = mask[..., 0]

    ink = (ink > 0).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    H, W = ink.shape
    H_ds = int(H * scale_factor)
    W_ds = int(W * scale_factor)
    print(f"Original label size: H={H}, W={W}")
    print(f"Downsampled label size: H_ds={H_ds}, W_ds={W_ds}")

    # folosim resize cu nearest-neighbor (order=0) pentru a pastra binaritatea
    ink_ds = resize(
        ink,
        (H_ds, W_ds),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    mask_ds = resize(
        mask,
        (H_ds, W_ds),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    ink_ds = (ink_ds > 0.5).astype(np.uint8)
    mask_ds = (mask_ds > 0.5).astype(np.uint8)

    return ink_ds, mask_ds



def get_mask_bbox(mask: np.ndarray, margin: int = 8):
    """
    Bounding box pentru zona unde mask == 1.
    intoarce (y_min, y_max, x_min, x_max) ca slice-uri [min, max).
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise RuntimeError("Mask is empty, no valid region found.")

    y_min = max(0, ys.min() - margin)
    y_max = min(mask.shape[0], ys.max() + margin + 1)
    x_min = max(0, xs.min() - margin)
    x_max = min(mask.shape[1], xs.max() + margin + 1)

    return y_min, y_max, x_min, x_max


def load_surface_volume_resized(surface_dir: Path,
                                out_h: int,
                                out_w: int,
                                max_slices: int = 8):
    """
    Citeste .tif-urile din surface_volume unul cate unul,
    le redimensioneaza la (out_h, out_w) si le stocheaza intr-un volum 3D mic.
    """
    all_tifs = sorted(surface_dir.glob("*.tif"))
    if not all_tifs:
        raise RuntimeError(f"No .tif files found in {surface_dir}")

    total = len(all_tifs)
    if total > max_slices:
        # luam un bloc central de max_slices
        mid = total // 2
        half = max_slices // 2
        start = max(0, mid - half)
        end = min(total, start + max_slices)
        tif_files = all_tifs[start:end]
    else:
        tif_files = all_tifs
        start, end = 0, total

    print(
        f"Found {total} slices, using {len(tif_files)} slices "
        f"from index {start} to {end - 1}."
    )

    slices = []
    for f in tqdm(tif_files, desc="Loading + resizing CT slices"):
        img = io.imread(str(f)).astype(np.float32)
        # redimensionam la (out_h, out_w)
        img_ds = resize(
            img,
            (out_h, out_w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
        slices.append(img_ds)

    volume = np.stack(slices, axis=0)  # (D, out_h, out_w)
    return volume


def normalize_volume(volume: np.ndarray, clip_low=1.0, clip_high=99.0):
    """
    Normalizare simpla: clip la percentila [clip_low, clip_high], apoi scale in [0,1].
    """
    lo = np.percentile(volume, clip_low)
    hi = np.percentile(volume, clip_high)
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo + 1e-8)
    return volume.astype(np.float32)


def sample_patch_centers(ink: np.ndarray,
                         mask: np.ndarray,
                         n_pos: int,
                         n_bg: int,
                         rng: np.random.Generator):
    """
    Alege pozitii centrale pentru patch-uri:
      - n_pos centre in zone cu cerneala
      - n_bg centre in zone background (mask==1, ink==0)
    """
    ys_pos, xs_pos = np.where(ink > 0)
    pos_coords = list(zip(ys_pos, xs_pos))
    rng.shuffle(pos_coords)
    if len(pos_coords) > n_pos:
        pos_coords = pos_coords[:n_pos]

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
    Taie un patch 3D din volume si un patch 2D din ink, centrate in (y_center, x_center).
    volume shape: (D, H, W)
    ink shape: (H, W)
    """
    D, H, W = volume.shape
    ps = patch_size

    if H < ps or W < ps:
        raise RuntimeError(f"Patch size {ps} is larger than volume size (H={H}, W={W}).")

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


def main():
    # mergem 2 niveluri in sus din src/preprocess -> project-root
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw" / "fragment1"  # schimba in "Frag1" daca ai redenumit folderul
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

    # 1. Incarcam labels la rezolutie micsorata
    scale_factor = 0.2  # 20% din rezolutie pe H si W
    ink_ds, mask_ds = load_labels_resized(raw_dir, scale_factor=scale_factor)
    H_ds, W_ds = ink_ds.shape
    print(f"Downsampled labels shape: H_ds={H_ds}, W_ds={W_ds}")
    print(f"Ink positives (downsampled): {int(ink_ds.sum())} pixels")

    # 2. Bounding box pe mask
    y_min, y_max, x_min, x_max = get_mask_bbox(mask_ds, margin=8)
    print("Mask bbox on downsampled labels (y_min, y_max, x_min, x_max):",
          y_min, y_max, x_min, x_max)

    ink_roi = ink_ds[y_min:y_max, x_min:x_max]
    mask_roi = mask_ds[y_min:y_max, x_min:x_max]
    H_roi, W_roi = ink_roi.shape
    print(f"ROI label size: H_roi={H_roi}, W_roi={W_roi}")

    # 3. Incarcam si redimensionam volumele CT direct la rezolutia downsampled
    volume_ds = load_surface_volume_resized(
    surface_dir,
    out_h=H_ds,
    out_w=W_ds,
    max_slices=4,   # doar 4 slice-uri pe adancime pentru siguranta la RAM
)

    D, H_v, W_v = volume_ds.shape
    print(f"Downsampled volume shape before ROI crop: D={D}, H={H_v}, W={W_v}")

    # 4. Cropping la acelasi ROI
    volume_roi = volume_ds[:, y_min:y_max, x_min:x_max]
    D, H_roi_v, W_roi_v = volume_roi.shape
    print(f"Volume ROI shape: D={D}, H={H_roi_v}, W={W_roi_v}")

    # 5. Normalizare pe volum ROI
    volume_roi = normalize_volume(volume_roi)

    # 6. Sampling patch-uri pe ROI downsampled
    rng = np.random.default_rng(42)
    patch_size = 256

    # daca ROI e mai mic decat 256, reducem patch_size
    if H_roi < patch_size or W_roi < patch_size:
        patch_size = min(H_roi, W_roi, 192)
        print(f"ROI too small for 256x256, using patch_size={patch_size}")

    n_pos = 200   # patch-uri cu cerneala
    n_bg = 200    # patch-uri background

    pos_centers, bg_centers = sample_patch_centers(ink_roi, mask_roi, n_pos, n_bg, rng)
    print(f"Sampled {len(pos_centers)} positive centers and {len(bg_centers)} background centers.")

    all_patches = []
    for (y, x) in pos_centers:
        all_patches.append(((y, x), 1))
    for (y, x) in bg_centers:
        all_patches.append(((y, x), 0))

    rng.shuffle(all_patches)

    # 7. Split train / val / test: 70 / 15 / 15
    n_total = len(all_patches)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    indices_train = all_patches[:n_train]
    indices_val = all_patches[n_train:n_train + n_val]
    indices_test = all_patches[n_train + n_val:]

    print(
        f"Total patches: {n_total} "
        f"-> train={len(indices_train)}, val={len(indices_val)}, test={len(indices_test)}"
    )

    # 8. Salvare patch-uri .npz
    def save_split(indices, out_dir: Path, split_name: str):
        counter = 0
        for (y, x), tag in tqdm(indices, desc=f"Saving {split_name} patches"):
            try:
                vol_patch, ink_patch = extract_patch(volume_roi, ink_roi, y, x, patch_size)
            except AssertionError as e:
                print("Skipping patch at", y, x, "due to shape error:", e)
                continue

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
