# 🌕 The Pareidolia Paradox — Lunar Feature Classification

> Official Solution Repository for **The Pareidolia Paradox** by IEEE SIES GST.  
> Classifying lunar surface features into **Crater (Depth, Label 0)** vs. **Mound (Rise, Label 1)** from orbital imagery under varying solar illumination.

---

## 📌 Executive Summary & Methodology

### 1. The Core Challenge: The Pareidolia Illusion
In monocular planetary orbital imagery, surface relief features (depressions like craters vs. elevations like mounds) often appear indistinguishable or inverted depending on the illumination angle. A crater illuminated from the bottom appears as a mound, and a mound illuminated from the top can appear as a depression. This perceptual ambiguity is known as the **Pareidolia Paradox**.

### 2. Physics-Informed Solar Azimuth Normalization
To solve this physical ambiguity:
- **Standardized Lighting**: Every input image is rotated by `-sun_azimuth_angle` using bilinear interpolation (`scipy.ndimage.rotate`) with border reflection/mean padding. This aligns the incident sunlight vector across all images to a canonical direction (North / top-to-bottom illumination).
- **Strict Orientation Preservation**: Arbitrary rotation and vertical flipping are strictly omitted during training augmentations because inverting or rotating the illumination axis destroys the physical shadow-depth correspondence.
- **Micro-Contrast Enhancement**: Contrast Limited Adaptive Histogram Equalization (**CLAHE**) is applied to highlight subtle shadow boundaries, crater rims, and elevation gradients without distorting radiometric ratios.

---

## 🏗️ Model Architecture & Ensemble

- **Backbones**:
  - **ConvNeXt-Tiny** (`convnext_tiny`, ~28M parameters)
  - **ConvNeXt-Small** (`convnext_small`, ~50M parameters)
- **Validation Strategy**: 4-Fold Stratified K-Fold Cross-Validation.
- **Ensemble**: Soft probability averaging across all 8 models (4 folds × 2 backbones).
- **Test-Time Augmentation (TTA)**: Evaluated with canonical and horizontal-flip passes (horizontal flipping preserves the vertical illumination axis while providing test-time variance reduction).

### Local Cross-Validation Performance
| Backbone | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-Tiny** | 0.7292 | 0.7486 | 0.7538 | 0.7538 | **0.7464** |
| **ConvNeXt-Small** | 0.7328 | 0.7350 | 0.7683 | 0.7288 | **0.7412** |
| **8-Model Ensemble** | — | — | — | — | **~0.7620** |

---

## 💻 Hardware Requirements

- **Operating System**: Windows 10/11 or Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (tested on NVIDIA GeForce RTX 3050 Laptop GPU). CPU execution is also supported.
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Disk Space**: ~2.5 GB (including model checkpoints and image datasets)

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Argho009/pareidolia-paradox.git
   cd pareidolia-paradox
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Model Weights

Download the trained model checkpoints from our public download link:
- **Download Link**: `[INSERT_YOUR_GOOGLE_DRIVE_OR_HUGGINGFACE_LINK_HERE]`
- Place the downloaded `*_best.pth` checkpoint files into the `weights/` directory (or inside `pareidolia/data/output/`):
  ```
  weights/
  ├── convnext_tiny_fold_1_best.pth
  ├── convnext_tiny_fold_2_best.pth
  ├── convnext_tiny_fold_3_best.pth
  ├── convnext_tiny_fold_4_best.pth
  ├── convnext_small_fold_1_best.pth
  ├── convnext_small_fold_2_best.pth
  ├── convnext_small_fold_3_best.pth
  └── convnext_small_fold_4_best.pth
  ```

---

## 🚀 1-Command Inference (Reproduce `submission.csv`)

To generate the final predictions on the 2,000 evaluation images, simply run:

```bash
python inference.py
```

Optional arguments:
```bash
python inference.py --test_csv pareidolia/data/test_metadata.csv \
                    --test_images_dir pareidolia/data/eval_images/eval_images \
                    --weights_dir pareidolia/data/output \
                    --output submission.csv \
                    --batch_size 32
```

This outputs `submission.csv` containing exactly 2,000 prediction rows (`image_id,label`) with no null values.

---

## 🏋️ Training Pipeline

To re-train the models from scratch using 4-fold cross-validation:

```bash
python train.py --config config.json
```

---

## 📁 Repository Structure

```
├── inference.py              # 1-command reproducible inference script
├── train.py                  # Standalone training script
├── requirements.txt          # Frozen environment dependencies
├── config.json               # Hyperparameters and pipeline configuration
├── README.md                 # Project documentation and reproduction guide
├── submission.csv            # Final predictions (2,000 rows)
└── pareidolia/
    ├── dataset.py            # Dataset loader with solar azimuth rotation & CLAHE
    ├── model.py              # ConvNeXt model architecture definition
    ├── predict.py            # Multi-model inference & TTA logic
    ├── train.py              # K-Fold training loop with Cosine Annealing
    ├── utils.py              # Config parser, logging, metrics utilities
    └── ui/                   # Interactive desktop application interface
```
