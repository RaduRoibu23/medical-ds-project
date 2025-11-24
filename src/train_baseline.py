# src/train_baseline.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from datasets.scrolls_patches import ScrollsPatchDataset
from models.unet3d_baseline import UNet3D
from utils.metrics import binarize_probs, f_beta_score, pseudo_f_measure, psnr


def train_one_epoch(model, loader, optimizer, device, criterion, depth_central):
    model.train()
    running_loss = 0.0

    for volume, mask in tqdm(loader, desc="Train", leave=False):
        volume = volume.to(device)  # (B,1,D,H,W)
        mask = mask.to(device)      # (B,1,H,W)

        optimizer.zero_grad()
        logits_3d = model(volume)                 # (B,1,D,H,W)
        logits_2d = logits_3d[:, :, depth_central, :, :]  # (B,1,H,W)

        loss = criterion(logits_2d, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * volume.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, depth_central):
    model.eval()
    all_f05, all_pfm, all_psnr = [], [], []

    for volume, mask in tqdm(loader, desc="Eval", leave=False):
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

    return mean_f05, mean_pfm, mean_psnr


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "processed"

    batch_size = 4
    num_epochs = 10
    lr = 1e-3
    base_channels = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = ScrollsPatchDataset(str(data_root), split="train")
    val_dataset = ScrollsPatchDataset(str(data_root), split="val")

    sample_volume, _ = train_dataset[0]
    _, D, _, _ = sample_volume.shape
    depth_central = D // 2
    print(f"Detected depth size D={D}, using central slice index {depth_central}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = UNet3D(in_channels=1, base_channels=base_channels).to(device)

    bce = nn.BCEWithLogitsLoss()

    def dice_loss(logits, targets, eps=1e-8):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice.mean()

    def combined_loss(logits, targets):
        return bce(logits, targets) + dice_loss(logits, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_f05 = -1.0
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     device, combined_loss, depth_central)
        print(f"Train loss: {train_loss:.4f}")

        val_f05, val_pfm, val_psnr = evaluate(model, val_loader,
                                              device, depth_central)
        print(f"Val F0.5={val_f05:.4f}, pFM={val_pfm:.4f}, PSNR={val_psnr:.4f}")

        if val_f05 > best_val_f05:
            best_val_f05 = val_f05
            ckpt_path = ckpt_dir / "unet3d_baseline_best.pt"
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "best_val_f05": best_val_f05},
                ckpt_path,
            )
            print(f"Saved new best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
