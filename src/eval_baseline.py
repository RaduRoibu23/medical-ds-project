# src/eval_baseline.py
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from datasets.scrolls_patches import ScrollsPatchDataset
from models.unet3d_baseline import UNet3D
from utils.metrics import binarize_probs, f_beta_score, pseudo_f_measure, psnr


@torch.no_grad()
def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "processed"
    ckpt_path = project_root / "checkpoints" / "unet3d_baseline_best.pt"

    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = ScrollsPatchDataset(str(data_root), split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    sample_volume, _ = test_dataset[0]
    _, D, _, _ = sample_volume.shape
    depth_central = D // 2
    print(f"Detected depth size D={D}, using central slice index {depth_central}")

    model = UNet3D(in_channels=1, base_channels=16).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}, best val F0.5={ckpt['best_val_f05']:.4f}")
    model.eval()

    all_f05, all_pfm, all_psnr = [], [], []

    for volume, mask in test_loader:
        volume = volume.to(device)
        mask = mask.to(device)

        logits_3d = model(volume)
        logits_2d = logits_3d[:, :, depth_central, :, :]
        probs_2d = torch.sigmoid(logits_2d)

        pred_bin = binarize_probs(probs_2d, threshold=0.5)

        f05 = f_beta_score(mask, pred_bin, beta=0.5)
        pfm = pseudo_f_measure(mask, probs_2d, threshold=0.5)
        p = psnr(probs_2d, mask)

        all_f05.append(f05)
        all_pfm.append(pfm)
        all_psnr.append(p)

    mean_f05 = sum(all_f05) / len(all_f05)
    mean_pfm = sum(all_pfm) / len(all_pfm)
    mean_psnr = sum(all_psnr) / len(all_psnr)

    print("\n=== Test results (fragment 1) ===")
    print(f"F0.5: {mean_f05:.4f}")
    print(f"pFM (approx): {mean_pfm:.4f}")
    print(f"PSNR: {mean_psnr:.4f}")


if __name__ == "__main__":
    main()
