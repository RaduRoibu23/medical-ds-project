# src/datasets/scrolls_patches.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ScrollsPatchDataset(Dataset):
    """
    Citeste patch-uri salvate ca .npz:
      - volume: (D,H,W) float32
      - mask:   (H,W)   float32 / uint8
    """

    def __init__(self, root_dir: str, split: str = "train"):
        super().__init__()
        assert split in ("train", "val", "test")
        self.split = split
        self.data_dir = os.path.join(root_dir, split)

        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)

        volume = data["volume"].astype(np.float32)  # (D,H,W)
        mask = data["mask"].astype(np.float32)      # (H,W)

        # asigura [0,1]
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)

        volume = torch.from_numpy(volume).unsqueeze(0)  # (1,D,H,W)
        mask = torch.from_numpy(mask).unsqueeze(0)      # (1,H,W)

        return volume, mask
