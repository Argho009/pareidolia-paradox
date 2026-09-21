"""Official Inference Script for The Pareidolia Paradox competition.

Reproduces the final `submission.csv` with exactly 2,000 predictions
using the trained multi-model ConvNeXt ensemble.

Run in 1 command:
    python inference.py
"""

import os
import sys
import glob
import argparse
import logging

# Ensure UTF-8 console output (for Windows CP1252 compatibility)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from pareidolia.utils import Config
    from pareidolia.dataset import PareidoliaDataset
    from pareidolia.model import PareidoliaModel
except ImportError:
    from utils import Config
    from dataset import PareidoliaDataset
    from model import PareidoliaModel

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("inference")

# Test-Time Augmentation (TTA): Original + Horizontal Flip
# Solar azimuth rotation aligns illumination to canonical North (vertical axis),
# so horizontal flip preserves shadow polarity while improving generalization.
TTA = [
    lambda x: x,
    lambda x: TF.hflip(x),
]


def resolve_first_existing(candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def run_inference(
    test_csv=None,
    test_images_dir=None,
    weights_dir=None,
    output_path="submission.csv",
    batch_size=32,
    num_workers=2,
    device=None,
    use_tta=True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("  🌕 THE PAREIDOLIA PARADOX — FINAL INFERENCE REPRODUCTION")
    print(f"  Device : {device}")
    if device == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    # 1. Resolve test_metadata.csv
    csv_candidates = [
        test_csv,
        os.path.join(CURRENT_DIR, "data", "test_metadata.csv"),
        os.path.join(PARENT_DIR, "pareidolia", "data", "test_metadata.csv"),
        os.path.join(PARENT_DIR, "Test", "test_metadata.csv"),
        os.path.join(CURRENT_DIR, "test_metadata.csv"),
    ]
    resolved_csv = resolve_first_existing(csv_candidates)
    if not resolved_csv:
        raise FileNotFoundError(f"Could not locate test_metadata.csv in any of: {csv_candidates}")
    df = pd.read_csv(resolved_csv)
    log.info(f"Loaded test metadata: {len(df)} rows from {resolved_csv}")

    # 2. Resolve test images directory
    img_candidates = [
        test_images_dir,
        os.path.join(CURRENT_DIR, "data", "eval_images", "eval_images"),
        os.path.join(PARENT_DIR, "pareidolia", "data", "eval_images", "eval_images"),
        os.path.join(PARENT_DIR, "Test", "test_images", "test_images"),
        os.path.join(CURRENT_DIR, "eval_images"),
    ]
    resolved_img_dir = resolve_first_existing(img_candidates)
    if not resolved_img_dir:
        raise FileNotFoundError(f"Could not locate test images directory in any of: {img_candidates}")
    log.info(f"Using evaluation images from: {resolved_img_dir}")

    # 3. Resolve weights directory and find all *_best.pth checkpoints
    weights_search_dirs = [
        weights_dir,
        os.path.join(CURRENT_DIR, "data", "output"),
        os.path.join(CURRENT_DIR, "weights"),
        os.path.join(PARENT_DIR, "pareidolia", "data", "output"),
        os.path.join(PARENT_DIR, "weights"),
        CURRENT_DIR,
    ]
    ckpt_paths = []
    for d in weights_search_dirs:
        if d and os.path.exists(d):
            found = glob.glob(os.path.join(d, "*_best.pth"))
            if found:
                ckpt_paths.extend(found)
    ckpt_paths = sorted(list(set(ckpt_paths)))

    if not ckpt_paths:
        raise FileNotFoundError(
            f"No '*_best.pth' model weights found in {weights_search_dirs}.\n"
            "Please download model weights and place in 'data/output/' or 'weights/'."
        )

    log.info(f"Discovered {len(ckpt_paths)} checkpoint(s):")
    models = []
    for cp in ckpt_paths:
        basename = os.path.basename(cp)
        model_name = basename.split("_fold_")[0]
        try:
            m = PareidoliaModel(model_name)
            m.load_state_dict(torch.load(cp, map_location=device))
            m = m.to(device).eval()
            models.append(m)
            log.info(f"  ✔ Loaded {model_name} from {basename}")
        except Exception as e:
            log.warning(f"  ✖ Could not load {basename}: {e}")

    if not models:
        raise RuntimeError("No models could be successfully loaded.")

    # 4. Dataset & DataLoader
    ds = PareidoliaDataset(
        df=df,
        image_dir=resolved_img_dir,
        is_train=False,
        dataset_mean=0.4185,
        dataset_std=0.2684,
    )
    ld = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # 5. Soft ensemble inference
    tfms = TTA if use_tta else [lambda x: x]
    all_preds = []

    log.info(f"Executing ensemble inference on {len(df)} images with {len(models)} model(s)...")
    with torch.no_grad():
        for x, _ in tqdm(ld, desc="Inference", unit="batch"):
            x = x.to(device)
            B = x.size(0)
            fold_probs = torch.zeros(len(models), B, 2, device=device)

            for mi, model in enumerate(models):
                acc = torch.zeros(B, 2, device=device)
                for t in tfms:
                    acc += F.softmax(model(t(x)), dim=1)
                fold_probs[mi] = acc / len(tfms)

            batch_preds = fold_probs.mean(dim=0).argmax(dim=1)
            all_preds.extend(batch_preds.cpu().numpy().tolist())

    # 6. Save submission.csv
    sub = pd.DataFrame({"image_id": df["image_id"], "label": all_preds})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    sub.to_csv(output_path, index=False)
    log.info(f"Successfully saved -> {output_path}")

    # Also sync to default output directory if different
    alt_out = os.path.join(CURRENT_DIR, "data", "output", "submission.csv")
    if os.path.abspath(output_path) != os.path.abspath(alt_out):
        os.makedirs(os.path.dirname(alt_out), exist_ok=True)
        sub.to_csv(alt_out, index=False)

    # 7. Verification Summary
    print("\n" + "=" * 65)
    print("  📋 FINAL SUBMISSION VERIFICATION CHECKLIST")
    print("=" * 65)
    print(f"  File Path            : {os.path.abspath(output_path)}")
    print(f"  Total Predictions    : {len(sub)}")
    print(f"  Expected Row Count   : 2000")
    print(f"  Missing / Null Count : {sub.isna().sum().sum()}")
    print(f"  Format Columns       : {list(sub.columns)}")
    counts = sub["label"].value_counts().to_dict()
    print(f"  Crater / Depth (0)   : {counts.get(0, 0)} ({counts.get(0, 0)/len(sub)*100:.1f}%)")
    print(f"  Mound / Rise (1)     : {counts.get(1, 0)} ({counts.get(1, 0)/len(sub)*100:.1f}%)")
    print("=" * 65)
    if len(sub) == 2000 and sub.isna().sum().sum() == 0 and list(sub.columns) == ["image_id", "label"]:
        print("  ✅ VERIFICATION PASSED: File is 100% ready for official portal submission!")
    else:
        print("  ⚠️ WARNING: Check format requirements.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for The Pareidolia Paradox")
    parser.add_argument("--test_csv", type=str, default=None, help="Path to test metadata CSV")
    parser.add_argument("--test_images_dir", type=str, default=None, help="Path to evaluation images directory")
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory containing trained *_best.pth models")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output path for submission.csv")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    args = parser.parse_args()

    run_inference(
        test_csv=args.test_csv,
        test_images_dir=args.test_images_dir,
        weights_dir=args.weights_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        use_tta=not args.no_tta,
    )
