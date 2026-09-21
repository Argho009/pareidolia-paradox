import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# shared CLAHE instance
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ImageNet stats for 3-channel input (grayscale replicated)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class PareidoliaDataset(Dataset):
    def __init__(self, df, image_dir, is_train=True, dataset_mean=0.4185, dataset_std=0.2684):
        self.df    = df.reset_index(drop=True)
        self.dir   = image_dir
        self.train = is_train
        self.mean  = dataset_mean
        self.std   = dataset_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        fname = str(row["image_id"])
        if not fname.endswith(".png"):
            fname += ".png"

        img = cv2.imread(os.path.join(self.dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return empty tensor with label -1 on failure (safe fallback)
            x = torch.zeros(3, 256, 256)
            return x, torch.tensor(-1, dtype=torch.long)

        # ── Rotate by -azimuth to standardise sun direction ──
        # CRITICAL: This normalizes the lighting direction so shadows are consistent.
        # DO NOT vertically or horizontally flip — flipping inverts shadow orientation
        # which turns craters into mounds and mounds into craters (the Pareidolia Paradox).
        angle = -float(row.get("sun_azimuth_angle", 0))
        img   = rotate(img.astype(np.float32), angle, reshape=False,
                       order=1, mode="constant", cval=float(img.mean()))
        img   = img.clip(0, 255).astype(np.uint8)

        # ── CLAHE — sharpens crater rims ──
        img = _CLAHE.apply(img)

        # ── Normalise and replicate to 3 channels for backbone ──
        x = img.astype(np.float32) / 255.0
        x = np.stack([x, x, x], axis=0)
        x = torch.from_numpy(x)

        if self.train:
            x = self._aug(x)

        # ── Standardise with ImageNet stats ──
        x = (x - _MEAN) / _STD

        # ── Random Erasing (Cutout) after standardization ──
        if self.train and torch.rand(1) > 0.5:
            x = T.RandomErasing(p=1.0, scale=(0.02, 0.1), value=0.0)(x)

        label = (torch.tensor(int(row["label"]), dtype=torch.long)
                 if "label" in row.index
                 else torch.tensor(-1, dtype=torch.long))
        return x, label

    def _aug(self, x):
        """Orientation-preserving augmentations.

        Only brightness, contrast, translation, and noise are allowed.
        Flipping and rotation are strictly forbidden as they corrupt the shadow-depth cues.
        """
        if torch.rand(1) > 0.5:
            x = TF.adjust_brightness(x, float(torch.empty(1).uniform_(0.8, 1.2)))
        if torch.rand(1) > 0.5:
            x = TF.adjust_contrast(x, float(torch.empty(1).uniform_(0.8, 1.2)))
        if torch.rand(1) > 0.5:
            max_dx = int(x.shape[2] * 0.1)
            max_dy = int(x.shape[1] * 0.1)
            dx = int(torch.randint(-max_dx, max_dx + 1, (1,)).item())
            dy = int(torch.randint(-max_dy, max_dy + 1, (1,)).item())
            x = TF.affine(x, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])
        if torch.rand(1) > 0.5:
            x = x + torch.randn_like(x) * 0.02
        return torch.clamp(x, 0.0, 1.0)
