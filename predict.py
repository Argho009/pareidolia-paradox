import os, logging, queue
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

from pareidolia.utils import Config
from pareidolia.dataset import PareidoliaDataset, _CLAHE
from pareidolia.model import PareidoliaModel

log = logging.getLogger("pareidolia")

# TTA: original + horizontal flip only (preserves azimuth-corrected sun direction)
TTA = [
    lambda x: x,
    lambda x: TF.hflip(x),
]


def _load_all_models(config: Config):
    """Load every *_best.pth from output_dir."""
    models = []
    for ckpt in glob.glob(os.path.join(config.output_dir, "*_best.pth")):
        basename   = os.path.basename(ckpt)
        model_name = basename.split("_fold_")[0]
        try:
            m = PareidoliaModel(model_name)
            m.load_state_dict(torch.load(ckpt, map_location=config.device))
            models.append(m.to(config.device).eval())
            log.info(f"Loaded {model_name} from {basename}")
        except Exception as e:
            log.warning(f"Failed to load {basename}: {e}")
    return models


def predict_pipeline(config: Config, use_tta=True, ensemble="soft",
                     update_queue: queue.Queue = None):
    os.makedirs(config.output_dir, exist_ok=True)
    log.info("Running inference...")

    df  = pd.read_csv(config.test_csv)
    kw  = dict(dataset_mean=config.dataset_mean, dataset_std=config.dataset_std)
    ds  = PareidoliaDataset(df, config.test_images_dir, is_train=False, **kw)
    ld  = DataLoader(ds, config.batch_size, shuffle=False,
                     num_workers=config.num_workers, pin_memory=True,
                     persistent_workers=(config.num_workers > 0))

    models = _load_all_models(config)
    if not models:
        msg = "No trained models found. Train first."
        log.error(msg)
        if update_queue:
            update_queue.put({"type": "error", "message": msg})
        return

    log.info(f"Ensemble of {len(models)} models")
    tfms      = TTA if use_tta else [lambda x: x]
    all_preds = []

    for step, (x, _) in enumerate(tqdm(ld, desc="Predict")):
        x          = x.to(config.device)
        B          = x.size(0)
        fold_probs = torch.zeros(len(models), B, 2, device=config.device)

        with torch.no_grad():
            for mi, model in enumerate(models):
                acc = torch.zeros(B, 2, device=config.device)
                for t in tfms:
                    acc += F.softmax(model(t(x)), dim=1)
                fold_probs[mi] = acc / len(tfms)

        if ensemble == "hard":
            batch_preds, _ = torch.mode(fold_probs.argmax(2), dim=0)
        else:
            batch_preds = fold_probs.mean(0).argmax(1)

        all_preds.extend(batch_preds.cpu().numpy().tolist())

        if update_queue:
            update_queue.put({"type": "predict_progress",
                              "value": (step + 1) / len(ld), "text": "Predicting..."})
            if step % max(1, len(ld) // 5) == 0:
                # Denormalize channel 0 back to [0, 1] for preview
                preview_img = (x[0, 0] * 0.229 + 0.485).clamp(0, 1).cpu().numpy()
                update_queue.put({"type": "preview", "image": preview_img,
                                  "prediction": int(batch_preds[0].item())})

    sub = pd.DataFrame({"image_id": df["image_id"], "label": all_preds})
    out = os.path.join(config.output_dir, "submission.csv")
    sub.to_csv(out, index=False)
    log.info(f"Saved -> {out}")
    if update_queue:
        update_queue.put({"type": "predict_complete", "path": out})


def predict_single_image(config: Config, image_path: str,
                         use_tta=True, ensemble="soft"):
    """Predict on a single image file without needing a CSV."""
    import cv2
    from scipy.ndimage import rotate as ndrotate

    if not os.path.exists(image_path):
        log.error(f"Image not found: {image_path}")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.error(f"Could not read image: {image_path}")
        return None

    img = _CLAHE.apply(img)

    x = img.astype(np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)
    x = torch.from_numpy(x)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x    = (x - mean) / std
    x    = x.unsqueeze(0).to(config.device)

    models = _load_all_models(config)
    if not models:
        log.error("No trained models found. Train first.")
        return None

    tfms       = TTA if use_tta else [lambda _x: _x]
    fold_probs = torch.zeros(len(models), 1, 2, device=config.device)

    with torch.no_grad():
        for mi, model in enumerate(models):
            acc = torch.zeros(1, 2, device=config.device)
            for t in tfms:
                acc += F.softmax(model(t(x)), dim=1)
            fold_probs[mi] = acc / len(tfms)

    avg_probs = fold_probs.mean(0)[0]
    label     = int(avg_probs.argmax().item())
    confidence = float(avg_probs[label].item()) * 100
    result    = "Mound / Rise (1)" if label == 1 else "Crater / Depth (0)"

    print(f"\n{'='*45}")
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {result}")
    print(f"  Confidence : {confidence:.1f}%")
    print(f"{'='*45}\n")
    return label
