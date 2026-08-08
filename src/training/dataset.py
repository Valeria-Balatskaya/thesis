# src/training/dataset.py
# Dataset loader for HiDDeN training.
# Uses a folder of images + random 48-bit messages per sample.

import os
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class WatermarkDataset(Dataset):
    """
    Loads images from one or more folders and pairs each with a random binary message.
    Returns (image_tensor [3,H,W] in [0,1], message_tensor [msg_len] in {0,1}).
    """

    def __init__(self, image_dirs: List[str], msg_len: int = 48,
                 image_size: int = 128, extensions=(".png", ".jpg", ".jpeg")):
        self.msg_len = msg_len
        self.image_size = image_size

        # Gather all image paths from all folders
        self.paths: List[Path] = []
        for d in image_dirs:
            root = Path(d)
            if not root.exists():
                raise FileNotFoundError(f"Dataset folder not found: {d}")
            for ext in extensions:
                self.paths.extend(root.rglob(f"*{ext}"))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {image_dirs}")

        # HiDDeN paper uses random crop + horizontal flip
        self.transform = transforms.Compose([
            transforms.Resize(image_size + 16),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # [0,1] float, [3,H,W]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            # Some COCO images occasionally fail; return a neighbour
            return self.__getitem__((idx + 1) % len(self.paths))
        img_t = self.transform(img)
        msg = torch.randint(0, 2, (self.msg_len,), dtype=torch.float32)
        return img_t, msg