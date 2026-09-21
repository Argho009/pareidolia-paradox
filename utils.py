import os
import json
import logging
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass
class Config:
    # paths
    train_images_dir: str = "pareidolia/data/train_images/train_images"
    test_images_dir:  str = "pareidolia/data/eval_images/eval_images"
    train_csv:        str = "pareidolia/data/train_metadata.csv"
    test_csv:         str = "pareidolia/data/test_metadata.csv"
    output_dir:       str = "pareidolia/data/output"

    # model — ensemble of two ConvNeXt backbones (both proven on this dataset)
    model_name:  str = "convnext_tiny"
    model_name2: str = "convnext_small"

    # training
    n_splits:        int   = 5
    epochs:          int   = 60
    batch_size:      int   = 32      # smaller = stable on 6GB VRAM with bigger models
    num_workers:     int   = 2
    lr:              float = 3e-4
    min_lr:          float = 1e-6
    warmup_epochs:   int   = 3
    patience:        int   = 10
    val_every:       int   = 1
    max_norm:        float = 1.0
    label_smoothing: float = 0.05
    mixup_alpha:     float = 0.0     # disabled - destroys lunar depth cues

    # dataset stats (computed from train set after CLAHE)
    dataset_mean: float = 0.4185
    dataset_std:  float = 0.2684

    # misc
    seed:      int = 42
    device:    str = "cuda" if torch.cuda.is_available() else "cpu"
    tta_steps: int = 8

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            return cls(**json.load(f))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log = logging.getLogger("pareidolia")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        # Force UTF-8 on Windows so Unicode chars in log messages don't crash
        import sys
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except Exception:
                pass
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(ch)
        fh = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding='utf-8')
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def check_dependencies():
    import importlib.util, sys, subprocess
    pkgs = {
        "torch": "torch", "torchvision": "torchvision", "timm": "timm",
        "customtkinter": "customtkinter", "PIL": "Pillow", "numpy": "numpy",
        "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
        "matplotlib": "matplotlib", "cv2": "opencv-python",
        "colorama": "colorama", "tqdm": "tqdm",
    }
    missing = [pip for mod, pip in pkgs.items() if not importlib.util.find_spec(mod)]
    if missing:
        print(f"Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
