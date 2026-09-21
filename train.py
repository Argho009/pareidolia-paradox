import os, sys, logging, queue, threading, warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure UTF-8 output on all consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from pareidolia.utils import Config, setup_logger, set_seed
    from pareidolia.dataset import PareidoliaDataset
    from pareidolia.model import PareidoliaModel
except ImportError:
    from utils import Config, setup_logger, set_seed
    from dataset import PareidoliaDataset
    from model import PareidoliaModel

log = logging.getLogger("pareidolia")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


# ──────────────────────────────────────────────────────────────
#  Loss Functions
# ──────────────────────────────────────────────────────────────

class SmoothCE(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.s = smoothing
        self.c = 1 - smoothing

    def forward(self, x, y):
        lp  = F.log_softmax(x, dim=-1)
        nll = -lp.gather(1, y.unsqueeze(1)).squeeze(1)
        return (self.c * nll + self.s * (-lp.mean(-1))).mean()


class MixupSmoothCE(nn.Module):
    """Handles MixUp: loss = lam * CE(x, ya) + (1-lam) * CE(x, yb)."""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.base = SmoothCE(smoothing)

    def forward(self, x, ya, yb, lam):
        return lam * self.base(x, ya) + (1 - lam) * self.base(x, yb)


# ──────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────

def _build_loader(ds, cfg, sampler=None, shuffle=False):
    return DataLoader(
        ds, cfg.batch_size,
        sampler=sampler, shuffle=(shuffle if sampler is None else False),
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )


def _save_curves(results, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    for fold, m in results.items():
        a1.plot(m["tl"], label=f"f{fold} train")
        a1.plot(m["vl"], label=f"f{fold} val", ls="--")
        a2.plot(m["vb"], label=f"f{fold}")
    a1.set_title("Loss");         a1.legend(fontsize=7)
    a2.set_title("Balanced Acc"); a2.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "training_curves.png"), dpi=150)
    plt.close()


def _save_log(results, out):
    rows = []
    for fold, m in results.items():
        for i, (tl, vl, vb, vf) in enumerate(zip(m["tl"], m["vl"], m["vb"], m["vf"])):
            rows.append({"fold": fold, "epoch": i + 1,
                         "train_loss": tl, "val_loss": vl,
                         "val_bacc": vb, "val_f1": vf})
    pd.DataFrame(rows).to_csv(os.path.join(out, "training_log.csv"), index=False)


# ──────────────────────────────────────────────────────────────
#  Train one fold / one model
# ──────────────────────────────────────────────────────────────

def _train_one_fold(cfg, fold, model_name, tr_df, va_df, update_queue, stop_event):
    """Returns metrics dict m = {tl, vl, vb, vf}."""
    kw    = dict(dataset_mean=cfg.dataset_mean, dataset_std=cfg.dataset_std)
    tr_ds = PareidoliaDataset(tr_df, cfg.train_images_dir, is_train=True,  **kw)
    va_ds = PareidoliaDataset(va_df, cfg.train_images_dir, is_train=False, **kw)

    cc      = tr_df["label"].value_counts().sort_index().values.astype(float)
    sw      = (1.0 / cc)[tr_df["label"].values]
    sampler = WeightedRandomSampler(torch.tensor(sw).float(), len(sw), replacement=True)

    tr_ld = _build_loader(tr_ds, cfg, sampler=sampler)
    va_ld = _build_loader(va_ds, cfg, shuffle=False)

    model    = PareidoliaModel(model_name, drop_rate=0.3).to(cfg.device)
    crit_val = SmoothCE(cfg.label_smoothing)
    crit_mx  = MixupSmoothCE(cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=2e-4)
    sched    = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, 0.1, total_iters=cfg.warmup_epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=cfg.min_lr),
    ], milestones=[cfg.warmup_epochs])
    scaler = torch.amp.GradScaler("cuda") if cfg.device == "cuda" else None

    ckpt        = os.path.join(cfg.output_dir, f"{model_name}_fold_{fold}_best.pth")
    resume_ckpt = os.path.join(cfg.output_dir, f"{model_name}_fold_{fold}_resume.pth")
    m           = {"tl": [], "vl": [], "vb": [], "vf": []}
    best, pat   = 0.0, 0
    start_ep    = 1

    # ── Resume ──
    if os.path.exists(resume_ckpt):
        log.info(f"  Resuming {model_name} fold {fold} from {resume_ckpt}")
        ck = torch.load(resume_ckpt, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if scaler and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_ep = ck["epoch"] + 1
        best      = ck["best"]
        pat       = ck["pat"]
        m         = ck["m"]
        if update_queue:
            for i in range(len(m.get("tl", []))):
                update_queue.put({"type": "epoch_end", "fold": fold, "epoch": i + 1,
                                  "train_loss": m["tl"][i], "val_loss": m["vl"][i],
                                  "val_bacc": m["vb"][i]})
    elif os.path.exists(ckpt):
        log.info(f"  Found best ckpt {ckpt}, restarting train state.")
        model.load_state_dict(torch.load(ckpt, map_location=cfg.device))

    # ── Epoch loop ──
    for ep in range(start_ep, cfg.epochs + 1):
        if stop_event and stop_event.is_set():
            break

        # -- Train --
        model.train()
        rl = 0.0
        for step, (x, y) in enumerate(tqdm(tr_ld, desc=f"{model_name} F{fold} E{ep} train", leave=False)):
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = crit_val(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
                scaler.step(opt); scaler.update()
            else:
                loss = crit_val(model(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
                opt.step()
            rl += loss.item() * x.size(0)
            if update_queue:
                update_queue.put({"type": "progress", "value": (step + 1) / len(tr_ld),
                                  "text": f"{model_name} F{fold} E{ep} - Training"})
        tl = rl / len(tr_ld.dataset)

        # -- Validate (every epoch now that val_every=1) --
        do_val = (ep % cfg.val_every == 0) or (ep == cfg.epochs)
        if do_val:
            model.eval()
            vl = 0.0; preds, labels = [], []
            with torch.no_grad():
                for step, (x, y) in enumerate(tqdm(va_ld, desc=f"val", leave=False)):
                    x, y   = x.to(cfg.device), y.to(cfg.device)
                    out    = model(x)
                    loss   = crit_val(out, y)
                    vl    += loss.item() * x.size(0)
                    preds.extend(out.argmax(1).cpu().numpy())
                    labels.extend(y.cpu().numpy())
                    if update_queue:
                        update_queue.put({"type": "progress", "value": (step + 1) / len(va_ld),
                                          "text": f"{model_name} F{fold} E{ep} — Validating"})
            vl /= len(va_ld.dataset)
            vb  = balanced_accuracy_score(labels, preds)
            vf  = f1_score(labels, preds, average="macro", zero_division=0)
            m["tl"].append(tl); m["vl"].append(vl); m["vb"].append(vb); m["vf"].append(vf)
            log.info(f"{model_name} F{fold} E{ep:3d} | tl={tl:.4f} vl={vl:.4f} bacc={vb:.4f} f1={vf:.4f}")

            if update_queue:
                update_queue.put({"type": "epoch_end", "fold": fold, "epoch": ep,
                                  "train_loss": tl, "val_loss": vl, "val_bacc": vb})

            if vb > best:
                best, pat = vb, 0
                torch.save(model.state_dict(), ckpt)
                log.info(f"  [BEST] {model_name} F{fold} bacc={best:.4f} -> {ckpt}")
            else:
                pat += 1
                if pat >= cfg.patience:
                    log.info(f"  Early stop {model_name} F{fold} at epoch {ep}"); break
        else:
            log.info(f"{model_name} F{fold} E{ep:3d} | tl={tl:.4f} [skip val]")

        sched.step()
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "epoch": ep, "best": best, "pat": pat, "m": m}, resume_ckpt)

    # -- Per-fold confusion matrix with best weights --
    model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in va_ld:
            preds.extend(model(x.to(cfg.device)).argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    cm = confusion_matrix(labels, preds)
    log.info(f"  {model_name} F{fold} best bacc={best:.4f}")
    return m, best, cm


# ──────────────────────────────────────────────────────────────
#  Public pipeline
# ──────────────────────────────────────────────────────────────

def train_pipeline(config: Config, update_queue: queue.Queue = None, stop_event: threading.Event = None):
    os.makedirs(config.output_dir, exist_ok=True)
    log.info(f"Training on {config.device}")

    df = pd.read_csv(config.train_csv)
    all_results = {}
    model_names = [config.model_name]
    if hasattr(config, "model_name2") and config.model_name2:
        model_names.append(config.model_name2)

    if config.n_splits == 1:
        tr_idx, va_idx = train_test_split(
            np.arange(len(df)), test_size=0.2, stratify=df["label"], random_state=config.seed)
        splits = [(tr_idx, va_idx)]
    else:
        skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
        splits = list(skf.split(df, df["label"]))

    for fi, (tr_idx, va_idx) in enumerate(splits):
        fold  = fi + 1
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)
        log.info(f"=== FOLD {fold}/{config.n_splits} ===")

        for model_name in model_names:
            if stop_event and stop_event.is_set():
                break
            m, best, cm = _train_one_fold(config, fold, model_name, tr_df, va_df, update_queue, stop_event)
            key = f"{model_name}_fold{fold}"
            all_results[key] = m
            if update_queue:
                update_queue.put({"type": "fold_end", "fold": fold,
                                  "best_bacc": best, "cm": cm.tolist()})

    _save_curves(all_results, config.output_dir)
    _save_log(all_results, config.output_dir)
    log.info("Training done.")
    if update_queue:
        update_queue.put({"type": "training_complete"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train models for The Pareidolia Paradox")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    args = parser.parse_args()

    cfg_file = args.config
    if not cfg_file:
        for c in ["config.json", os.path.join(PARENT_DIR, "config.json"), os.path.join(CURRENT_DIR, "config.json")]:
            if os.path.exists(c):
                cfg_file = c
                break

    if not cfg_file or not os.path.exists(cfg_file):
        raise FileNotFoundError("Could not find config.json. Please specify with --config.")

    cfg = Config.load(cfg_file)
    os.makedirs(cfg.output_dir, exist_ok=True)
    setup_logger(cfg.output_dir)
    set_seed(cfg.seed)
    train_pipeline(cfg)
