# 1
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm  # Use tqdm.notebook for nice progress bars in Kaggle

# CONFIGURATION
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4  # Keep small for 3D
LR = 1e-3
EPOCHS = 10
DATA_ROOT = '/kaggle/input/vesuvius-challenge-ink-detection'
FRAGMENT_ID = '1'
