# 3
# src/datasets/scrolls_patches.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


# class ScrollsPatchDataset(Dataset):
#     """
#     Citeste patch-uri salvate ca .npz:
#       - volume: (D,H,W) float32
#       - mask:   (H,W)   float32 / uint8
#     """

#     def __init__(self, root_dir: str, split: str = "train"):
#         super().__init__()
#         assert split in ("train", "val", "test")
#         self.split = split
#         self.data_dir = os.path.join(root_dir, split)

#         self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
#         if not self.files:
#             raise RuntimeError(f"No .npz files found in {self.data_dir}")

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path = self.files[idx]
#         data = np.load(path)

#         volume = data["volume"].astype(np.float32)  # (D,H,W)
#         mask = data["mask"].astype(np.float32)      # (H,W)

#         # asigura [0,1]
#         v_min, v_max = volume.min(), volume.max()
#         if v_max > v_min:
#             volume = (volume - v_min) / (v_max - v_min)

#         volume = torch.from_numpy(volume).unsqueeze(0)  # (1,D,H,W)
#         mask = torch.from_numpy(mask).unsqueeze(0)      # (1,H,W)

#         return volume, mask

class GeometricInkDataset(Dataset):
    def __init__(self, root_dir, split='train', slice_depth=4, patch_size=128):
        self.root_dir = root_dir
        self.split = split
        self.slice_depth = slice_depth
        self.patch_size = patch_size
        self.fragment_id = '1'
        self.base_path = os.path.join(root_dir, 'train', self.fragment_id)
        
        # Load Images
        self.ink_labels = self._load_image('inklabels.png')
        self.mask = self._load_image('mask.png')
        
        # Load Volume
        print(f"Loading volume for {split}...")
        self.volume = self._load_volume_slices()
        
        # Use Balanced Sampling (Keep this! It works great)
        self.valid_indices = self._generate_balanced_indices() 

    def _load_image(self, filename):
        path = os.path.join(self.base_path, filename)
        img = cv2.imread(path, 0)
        return img // 255 

    def _load_volume_slices(self):
        start_slice = 28
        end_slice = start_slice + self.slice_depth
        slices = []
        for i in range(start_slice, end_slice):
            path = os.path.join(self.base_path, 'surface_volume', f'{i:02d}.tif')
            if not os.path.exists(path): path = os.path.join(self.base_path, 'surface_volume', f'{i}.tif')
            slices.append(cv2.imread(path, 0))
        return np.stack(slices, axis=0)

    def _generate_balanced_indices(self):
        margin = self.patch_size // 2
        valid_mask = self.mask[margin:-margin, margin:-margin]
        valid_ink = self.ink_labels[margin:-margin, margin:-margin]
        
        y_pos, x_pos = np.where((valid_mask == 1) & (valid_ink == 1))
        y_neg, x_neg = np.where((valid_mask == 1) & (valid_ink == 0))
        
        y_pos, x_pos = y_pos + margin, x_pos + margin
        y_neg, x_neg = y_neg + margin, x_neg + margin
        
        # 600 patches for Train, 100 for Test
        num_patches = 600 if self.split == 'train' else 100
        half = num_patches // 2
        
        indices = []
        if len(y_pos) > 0:
            idxs = np.random.choice(len(y_pos), half, replace=True)
            for i in idxs: indices.append((y_pos[i] - margin, x_pos[i] - margin))
        
        if len(y_neg) > 0:
            idxs = np.random.choice(len(y_neg), half, replace=True)
            for i in idxs: indices.append((y_neg[i] - margin, x_neg[i] - margin))
                
        np.random.shuffle(indices)
        return indices

    def _compute_gradients(self, volume_patch):
        # Compute gradients on the already-rotated volume
        grads = np.gradient(volume_patch, axis=(0, 1, 2))
        return np.stack(grads, axis=0)

    def __getitem__(self, idx):
        y, x = self.valid_indices[idx]
        
        # 1. Extract Raw Patch
        vol_patch = self.volume[:, y:y+self.patch_size, x:x+self.patch_size].astype(np.float32) / 255.0
        label_patch = self.ink_labels[y:y+self.patch_size, x:x+self.patch_size].astype(np.float32)
        
        # --- AUGMENTATION (Train Only) ---
        if self.split == 'train':
            # Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            vol_patch = np.rot90(vol_patch, k, axes=(1, 2))      
            label_patch = np.rot90(label_patch, k, axes=(0, 1))  
            
            # Random Horizontal Flip
            if np.random.random() > 0.5:
                vol_patch = np.flip(vol_patch, axis=2)           
                label_patch = np.flip(label_patch, axis=1)
            
            # Random Vertical Flip
            if np.random.random() > 0.5:
                vol_patch = np.flip(vol_patch, axis=1)           
                label_patch = np.flip(label_patch, axis=0)
        # ---------------------------------

        # 2. Compute Geometric Features (Gradients)
        # MUST happen after rotation to get correct normal directions
        # .copy() prevents memory stride errors in PyTorch
        gradients = self._compute_gradients(vol_patch.copy()) 
        
        # 3. Stack (4 Channels)
        input_tensor = np.concatenate([vol_patch[np.newaxis, ...], gradients], axis=0)
        
        return torch.from_numpy(np.ascontiguousarray(input_tensor)).float(), torch.from_numpy(np.ascontiguousarray(label_patch)).float().unsqueeze(0)

    def __len__(self):
        return len(self.valid_indices)
