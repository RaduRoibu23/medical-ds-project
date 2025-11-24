# src/utils/metrics.py
import torch
import torch.nn.functional as F


def binarize_probs(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """probs in [0,1], shape (B,1,H,W) -> 0/1 tensor"""
    return (probs >= threshold).float()


def f_beta_score(target: torch.Tensor,
                 prediction: torch.Tensor,
                 beta: float = 0.5,
                 eps: float = 1e-8) -> float:
    """
    F_beta la nivel de pixeli. target & prediction binare (0/1), shape (B,1,H,W).
    Returneaza media pe batch (float Python).
    """
    target = target.float()
    prediction = prediction.float()

    tp = (prediction * target).sum(dim=(1, 2, 3))
    fp = (prediction * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - prediction) * target).sum(dim=(1, 2, 3))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    beta2 = beta ** 2
    f_beta = (1 + beta2) * (precision * recall) / (beta2 * precision + recall + eps)
    return f_beta.mean().item()


def psnr(prediction: torch.Tensor,
         target: torch.Tensor,
         max_val: float = 1.0) -> float:
    """
    PSNR intre doua harti in [0,1]. prediction & target shape (B,1,H,W) sau (B,H,W).
    """
    if prediction.dim() == 4:
        prediction = prediction.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    mse = F.mse_loss(prediction, target, reduction="none").mean(dim=(1, 2))
    mse = torch.clamp(mse, min=1e-8)
    psnr_batch = 20.0 * torch.log10(torch.tensor(max_val)) - 10.0 * torch.log10(mse)
    return psnr_batch.mean().item()


def pseudo_f_measure(target: torch.Tensor,
                     prediction_probs: torch.Tensor,
                     threshold: float = 0.5,
                     eps: float = 1e-8) -> float:
    """
    Aproximare pFM: F1 pe harti binare. Pentru M2 e suficient.
    """
    pred_bin = binarize_probs(prediction_probs, threshold=threshold)

    target = target.float()
    pred_bin = pred_bin.float()

    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1.mean().item()
