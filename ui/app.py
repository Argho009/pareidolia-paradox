"""Pareidolia Paradox — Professional Machine Learning Pipeline GUI.

Modern, responsive CustomTkinter interface for:
- Dataset & Hyperparameter Configuration
- Multi-Model Stratified K-Fold Training & Live Metrics
- Full Evaluation Inference with TTA & Soft Ensembling
- Interactive Single-Image Crater vs. Mound Testing
- Confusion Matrix & Performance Visualization
- Live Log Streaming & Export
"""

import os
import sys
import queue
import threading
import time
import subprocess
from datetime import datetime

import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Ensure root & parent are on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PKG_DIR)
for p in [CURRENT_DIR, PKG_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from pareidolia.utils import Config
    from pareidolia.train import train_pipeline
    from pareidolia.predict import predict_pipeline
    from pareidolia.dataset import _CLAHE
    from pareidolia.model import PareidoliaModel
except ImportError:
    from utils import Config
    from train import train_pipeline
    from predict import predict_pipeline
    from dataset import _CLAHE
    from model import PareidoliaModel

import torch
import torch.nn.functional as F
import cv2
from scipy.ndimage import rotate as ndrotate

# ── Color Palette (Modern Dark / Space Theme) ──
THEME = {
    "bg_dark": "#11111b",
    "sidebar_bg": "#181825",
    "card_bg": "#1e1e2e",
    "card_border": "#313244",
    "accent_blue": "#3b82f6",
    "accent_cyan": "#06b6d4",
    "accent_green": "#10b981",
    "accent_amber": "#f59e0b",
    "accent_red": "#ef4444",
    "text_main": "#f8fafc",
    "text_muted": "#94a3b8",
    "btn_inactive": "#1e1e2e",
    "btn_hover": "#2a2b3d",
}


class _LogRedirect:
    """Redirects stdout writes to the CTkTextbox log widget safely."""

    def __init__(self, widget: ctk.CTkTextbox) -> None:
        self.widget = widget

    def write(self, message: str) -> None:
        try:
            self.widget.insert(ctk.END, message)
            self.widget.see(ctk.END)
        except Exception:
            pass

    def flush(self) -> None:
        pass


class PareidoliaApp(ctk.CTk):
    """Next-Generation CustomTkinter GUI for The Pareidolia Paradox."""

    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.title("🌕 The Pareidolia Paradox — Deep Learning Pipeline")
        self.geometry("1240x820")
        self.minsize(1120, 740)
        self.configure(fg_color=THEME["bg_dark"])

        # Load config
        cfg_paths = [
            os.path.join(PKG_DIR, "config.json"),
            os.path.join(ROOT_DIR, "config.json"),
            "config.json",
        ]
        self.config_file = "config.json"
        for cp in cfg_paths:
            if os.path.exists(cp):
                self.config_file = cp
                break
        self.config_obj = Config.load(self.config_file)

        # Threading & Queues
        self._queue = queue.Queue()
        self.stop_event = threading.Event()
        self.train_thread = None
        self.pred_thread = None
        self._start_time = 0.0

        # Metrics history
        self._train_losses = []
        self._val_losses = []
        self._val_baccs = []
        self._best_bacc = 0.0

        # Setup Views & Sidebar buttons tracking
        self._sidebar_buttons = {}
        self._active_tab = "Setup"

        self._build_layout()
        self._set_view("Setup")
        self._tick()

    # ------------------------------------------------------------------
    # Main Window Grid & Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ── Sidebar ──
        self.sidebar = ctk.CTkFrame(
            self, width=220, corner_radius=0, fg_color=THEME["sidebar_bg"], border_width=0
        )
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        # App Brand Header
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.grid(row=0, column=0, padx=16, pady=(24, 28), sticky="w")

        ctk.CTkLabel(
            brand_frame,
            text="🌕 PAREIDOLIA",
            font=ctk.CTkFont(family="Inter", size=18, weight="bold"),
            text_color=THEME["text_main"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            brand_frame,
            text="LUNAR RELIEF CLASSIFIER",
            font=ctk.CTkFont(family="Inter", size=10, weight="bold"),
            text_color=THEME["accent_cyan"],
        ).pack(anchor="w", pady=(2, 0))

        # Navigation Items
        nav_items = [
            ("Setup", "⚙️", "Pipeline Config"),
            ("Train", "🚂", "Model Training"),
            ("Predict", "🔍", "Inference & Test"),
            ("Results", "📊", "Metrics & Matrix"),
            ("Logs", "📄", "Console Output"),
        ]

        for i, (key, icon, subtitle) in enumerate(nav_items, start=1):
            btn = ctk.CTkButton(
                self.sidebar,
                text=f"  {icon}  {key}",
                font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
                anchor="w",
                height=42,
                corner_radius=8,
                fg_color="transparent",
                text_color=THEME["text_muted"],
                hover_color=THEME["btn_hover"],
                command=lambda k=key: self._set_view(k),
            )
            btn.grid(row=i, column=0, padx=12, pady=5, sticky="ew")
            self._sidebar_buttons[key] = btn

        # Sidebar Hardware Pill at bottom
        device_box = ctk.CTkFrame(
            self.sidebar,
            corner_radius=10,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        device_box.grid(row=11, column=0, padx=12, pady=16, sticky="ew")

        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Only"
        if len(gpu_name) > 22:
            gpu_name = gpu_name[:20] + "…"

        status_color = THEME["accent_green"] if has_cuda else THEME["accent_amber"]
        ctk.CTkLabel(
            device_box,
            text=f"● {'CUDA ACCELERATED' if has_cuda else 'CPU MODE'}",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=status_color,
        ).pack(anchor="w", padx=12, pady=(10, 2))

        ctk.CTkLabel(
            device_box,
            text=gpu_name,
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_muted"],
        ).pack(anchor="w", padx=12, pady=(0, 10))

        # ── Main Content Container ──
        self.main_container = ctk.CTkFrame(
            self, corner_radius=0, fg_color="transparent"
        )
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=16)
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        # Header Bar
        self.header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent", height=50)
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.title_lbl = ctk.CTkLabel(
            self.header_frame,
            text="Pipeline Setup",
            font=ctk.CTkFont(family="Inter", size=22, weight="bold"),
            text_color=THEME["text_main"],
        )
        self.title_lbl.grid(row=0, column=0, sticky="w")

        self.subtitle_lbl = ctk.CTkLabel(
            self.header_frame,
            text="Configure datasets, solar azimuth normalization, and training parameters",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_muted"],
        )
        self.subtitle_lbl.grid(row=1, column=0, sticky="w")

        # Top Action Button
        self.header_btn = ctk.CTkButton(
            self.header_frame,
            text="📁 Open Project Folder",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=32,
            fg_color=THEME["card_bg"],
            hover_color=THEME["btn_hover"],
            border_width=1,
            border_color=THEME["card_border"],
            command=self._open_project_dir,
        )
        self.header_btn.grid(row=0, column=1, rowspan=2, sticky="e", padx=4)

        # Views Frame
        self.views_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views_frame.grid(row=1, column=0, sticky="nsew")
        self.views_frame.grid_rowconfigure(0, weight=1)
        self.views_frame.grid_columnconfigure(0, weight=1)

        # Build all View Frames
        self.views = {}
        self.views["Setup"] = self._create_setup_view()
        self.views["Train"] = self._create_train_view()
        self.views["Predict"] = self._create_predict_view()
        self.views["Results"] = self._create_results_view()
        self.views["Logs"] = self._create_logs_view()

        # ── Bottom Status Bar ──
        self.status_bar = ctk.CTkFrame(
            self, fg_color=THEME["sidebar_bg"], height=32, corner_radius=0
        )
        self.status_bar.grid(row=1, column=1, sticky="ew")
        self.status_bar.grid_columnconfigure(0, weight=1)

        self.status_text = ctk.StringVar(value="System Ready • Initialized")
        ctk.CTkLabel(
            self.status_bar,
            textvariable=self.status_text,
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_muted"],
        ).grid(row=0, column=0, padx=16, pady=4, sticky="w")

        self.status_badge = ctk.CTkLabel(
            self.status_bar,
            text="IEEE SIES GST Finalist Edition",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["accent_cyan"],
        )
        self.status_badge.grid(row=0, column=1, padx=16, pady=4, sticky="e")

    # ------------------------------------------------------------------
    # View Navigation Switcher
    # ------------------------------------------------------------------

    def _set_view(self, view_name: str) -> None:
        self._active_tab = view_name

        # Update sidebar buttons
        for name, btn in self._sidebar_buttons.items():
            if name == view_name:
                btn.configure(
                    fg_color=THEME["accent_blue"],
                    text_color="#ffffff",
                    hover_color="#2563eb",
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=THEME["text_muted"],
                    hover_color=THEME["btn_hover"],
                )

        # Update titles
        titles = {
            "Setup": ("Pipeline Configuration", "Configure datasets, solar azimuth normalization, and training parameters"),
            "Train": ("Model Training & Cross-Validation", "Train 4-Fold Stratified ConvNeXt models with real-time learning curves"),
            "Predict": ("Batch & Single Image Inference", "Run 2,000 evaluation predictions with TTA or test custom images"),
            "Results": ("Model Evaluation & Confusion Matrix", "Review per-fold validation balanced accuracy and error analysis"),
            "Logs": ("Live Console & Process Output", "Real-time execution log stream with export capabilities"),
        }
        t, st = titles.get(view_name, (view_name, ""))
        self.title_lbl.configure(text=t)
        self.subtitle_lbl.configure(text=st)

        # Switch Frame
        for name, frame in self.views.items():
            if name == view_name:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

    # ------------------------------------------------------------------
    # VIEW: Setup
    # ------------------------------------------------------------------

    def _create_setup_view(self) -> ctk.CTkFrame:
        scroll = ctk.CTkScrollableFrame(self.views_frame, fg_color="transparent")

        # ── Card 1: Data & Directory Paths ──
        card1 = ctk.CTkFrame(
            scroll,
            corner_radius=12,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        card1.pack(fill="x", pady=(0, 14), padx=2)
        card1.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            card1,
            text="📂 Dataset & Output Directories",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_main"],
        ).grid(row=0, column=0, columnspan=3, padx=16, pady=(14, 10), sticky="w")

        fields = [
            ("Train Images Dir", "train_images_dir", "dir"),
            ("Test Images Dir", "test_images_dir", "dir"),
            ("Train CSV", "train_csv", "file"),
            ("Test CSV", "test_csv", "file"),
            ("Output Directory", "output_dir", "dir"),
        ]

        self._path_vars = {}
        for row, (label, attr, kind) in enumerate(fields, start=1):
            val = getattr(self.config_obj, attr, "")
            var = ctk.StringVar(value=val)
            self._path_vars[attr] = var

            ctk.CTkLabel(
                card1,
                text=label + ":",
                font=ctk.CTkFont(size=12),
                text_color=THEME["text_muted"],
            ).grid(row=row, column=0, padx=(16, 10), pady=7, sticky="w")

            entry = ctk.CTkEntry(
                card1,
                textvariable=var,
                height=34,
                corner_radius=6,
                fg_color=THEME["sidebar_bg"],
                border_color=THEME["card_border"],
            )
            entry.grid(row=row, column=1, padx=6, pady=7, sticky="ew")

            btn_cmd = (
                (lambda v: lambda: v.set(filedialog.askdirectory() or v.get()))(var)
                if kind == "dir"
                else (
                    lambda v: lambda: v.set(
                        filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")]) or v.get()
                    )
                )(var)
            )

            ctk.CTkButton(
                card1,
                text="Browse",
                width=80,
                height=34,
                corner_radius=6,
                fg_color=THEME["btn_hover"],
                hover_color="#374151",
                command=btn_cmd,
            ).grid(row=row, column=2, padx=(6, 16), pady=7)

        # Spacer in card 1
        ctk.CTkLabel(card1, text="").grid(row=len(fields) + 1, column=0, pady=4)

        # ── Card 2: Architecture & Hyperparameters ──
        card2 = ctk.CTkFrame(
            scroll,
            corner_radius=12,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        card2.pack(fill="x", pady=4, padx=2)
        card2.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(
            card2,
            text="⚡ Model Backbones & Hyperparameters",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_main"],
        ).grid(row=0, column=0, columnspan=4, padx=16, pady=(14, 12), sticky="w")

        # Row 1: Backbones
        ctk.CTkLabel(card2, text="Primary Backbone:", font=ctk.CTkFont(size=12), text_color=THEME["text_muted"]).grid(
            row=1, column=0, padx=16, pady=6, sticky="w"
        )
        self.m1_var = ctk.StringVar(value=self.config_obj.model_name)
        ctk.CTkOptionMenu(
            card2,
            variable=self.m1_var,
            values=["convnext_tiny", "convnext_small", "convnext_base", "resnet50"],
            height=32,
            fg_color=THEME["sidebar_bg"],
            button_color=THEME["accent_blue"],
        ).grid(row=1, column=1, padx=10, pady=6, sticky="ew")

        ctk.CTkLabel(card2, text="Secondary Backbone:", font=ctk.CTkFont(size=12), text_color=THEME["text_muted"]).grid(
            row=1, column=2, padx=16, pady=6, sticky="w"
        )
        self.m2_var = ctk.StringVar(value=getattr(self.config_obj, "model_name2", "convnext_small"))
        ctk.CTkOptionMenu(
            card2,
            variable=self.m2_var,
            values=["convnext_small", "convnext_tiny", "tf_efficientnetv2_s", "None"],
            height=32,
            fg_color=THEME["sidebar_bg"],
            button_color=THEME["accent_blue"],
        ).grid(row=1, column=3, padx=(10, 16), pady=6, sticky="ew")

        # Row 2: Epochs & Batch Size
        ctk.CTkLabel(card2, text="Training Epochs:", font=ctk.CTkFont(size=12), text_color=THEME["text_muted"]).grid(
            row=2, column=0, padx=16, pady=6, sticky="w"
        )
        self.epochs_var = ctk.StringVar(value=str(self.config_obj.epochs))
        ctk.CTkEntry(card2, textvariable=self.epochs_var, height=32, fg_color=THEME["sidebar_bg"]).grid(
            row=2, column=1, padx=10, pady=6, sticky="ew"
        )

        ctk.CTkLabel(card2, text="Batch Size:", font=ctk.CTkFont(size=12), text_color=THEME["text_muted"]).grid(
            row=2, column=2, padx=16, pady=6, sticky="w"
        )
        self.batch_var = ctk.StringVar(value=str(self.config_obj.batch_size))
        ctk.CTkEntry(card2, textvariable=self.batch_var, height=32, fg_color=THEME["sidebar_bg"]).grid(
            row=2, column=3, padx=(10, 16), pady=6, sticky="ew"
        )

        # Row 3: Splits & Device
        ctk.CTkLabel(card2, text="K-Fold Splits:", font=ctk.CTkFont(size=12), text_color=THEME["text_muted"]).grid(
            row=3, column=0, padx=16, pady=6, sticky="w"
        )
        self.splits_var = ctk.StringVar(value=str(self.config_obj.n_splits))
        ctk.CTkOptionMenu(
            card2,
            variable=self.splits_var,
            values=["4", "5", "10", "1"],
            height=32,
            fg_color=THEME["sidebar_bg"],
            button_color=THEME["accent_blue"],
        ).grid(row=3, column=1, padx=10, pady=6, sticky="ew")

        ctk.CTkLabel(card2, text="Compute Device:", font=ctk.CTkFont(size=12), text_color=THEME["text_muted"]).grid(
            row=3, column=2, padx=16, pady=6, sticky="w"
        )
        self.dev_var = ctk.StringVar(value=self.config_obj.device)
        ctk.CTkOptionMenu(
            card2,
            variable=self.dev_var,
            values=["cuda", "cpu"],
            height=32,
            fg_color=THEME["sidebar_bg"],
            button_color=THEME["accent_blue"],
        ).grid(row=3, column=3, padx=(10, 16), pady=6, sticky="ew")

        ctk.CTkLabel(card2, text="").grid(row=4, column=0, pady=4)

        # ── Save Button ──
        btn_bar = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_bar.pack(fill="x", pady=(14, 20))

        ctk.CTkButton(
            btn_bar,
            text="💾  Save Configuration Changes",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=42,
            fg_color=THEME["accent_blue"],
            hover_color="#2563eb",
            command=self._save_settings,
        ).pack(side="right", padx=4)

        return scroll

    def _save_settings(self) -> None:
        try:
            for attr, var in self._path_vars.items():
                setattr(self.config_obj, attr, var.get().strip())
            self.config_obj.model_name = self.m1_var.get()
            self.config_obj.model_name2 = self.m2_var.get() if self.m2_var.get() != "None" else ""
            self.config_obj.epochs = int(self.epochs_var.get())
            self.config_obj.batch_size = int(self.batch_var.get())
            self.config_obj.n_splits = int(self.splits_var.get())
            self.config_obj.device = self.dev_var.get()

            self.config_obj.save(self.config_file)
            messagebox.showinfo("Saved", f"Configuration successfully saved to {self.config_file}!")
            self._set_status("Configuration saved successfully ✓")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save settings: {e}")

    # ------------------------------------------------------------------
    # VIEW: Train
    # ------------------------------------------------------------------

    def _create_train_view(self) -> ctk.CTkFrame:
        view = ctk.CTkFrame(self.views_frame, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(2, weight=1)

        # ── Metrics KPI Cards ──
        kpi_row = ctk.CTkFrame(view, fg_color="transparent")
        kpi_row.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        kpi_row.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.kpi_fold = self._create_kpi_card(kpi_row, 0, "CURRENT FOLD", "Fold: —")
        self.kpi_epoch = self._create_kpi_card(kpi_row, 1, "EPOCH PROGRESS", "0 / 60")
        self.kpi_val_bacc = self._create_kpi_card(kpi_row, 2, "BALANCED ACCURACY", "0.00%", THEME["accent_green"])
        self.kpi_best = self._create_kpi_card(kpi_row, 3, "PEAK LOCAL CV", "0.7620 (Ensemble)", THEME["accent_cyan"])

        # ── Control & Progress Bar ──
        ctrl_card = ctk.CTkFrame(
            view,
            corner_radius=12,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        ctrl_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        ctrl_card.grid_columnconfigure(1, weight=1)

        self._train_btn = ctk.CTkButton(
            ctrl_card,
            text="▶  Start Training Pipeline",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            fg_color=THEME["accent_green"],
            hover_color="#059669",
            command=self._toggle_training,
        )
        self._train_btn.grid(row=0, column=0, padx=16, pady=12)

        pbar_box = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        pbar_box.grid(row=0, column=1, sticky="ew", padx=16, pady=12)
        pbar_box.grid_columnconfigure(0, weight=1)

        self.train_status_lbl = ctk.CTkLabel(
            pbar_box, text="Pipeline ready to train 4-Fold ConvNeXt models", font=ctk.CTkFont(size=11), text_color=THEME["text_muted"]
        )
        self.train_status_lbl.grid(row=0, column=0, sticky="w", pady=(0, 4))

        self._train_pbar = ctk.CTkProgressBar(pbar_box, height=10, corner_radius=5)
        self._train_pbar.grid(row=1, column=0, sticky="ew")
        self._train_pbar.set(0)

        # ── Matplotlib Plots ──
        plot_card = ctk.CTkFrame(
            view,
            corner_radius=12,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        plot_card.grid(row=2, column=0, sticky="nsew")
        plot_card.grid_columnconfigure(0, weight=1)
        plot_card.grid_rowconfigure(0, weight=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor(THEME["card_bg"])
        for ax in (ax1, ax2):
            ax.set_facecolor(THEME["sidebar_bg"])
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.title.set_color("#f8fafc")
            ax.xaxis.label.set_color("#94a3b8")
            ax.yaxis.label.set_color("#94a3b8")
            for spine in ax.spines.values():
                spine.set_color(THEME["card_border"])
            ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")

        ax1.set_title("Training vs Validation Loss", fontsize=11, fontweight="bold", pad=10)
        ax2.set_title("Validation Balanced Accuracy", fontsize=11, fontweight="bold", pad=10)
        plt.tight_layout()

        self._train_fig = fig
        self._train_ax1, self._train_ax2 = ax1, ax2
        self._train_canvas = FigureCanvasTkAgg(fig, master=plot_card)
        self._train_canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)

        return view

    def _create_kpi_card(self, parent, col, title, value, val_color=THEME["text_main"]):
        card = ctk.CTkFrame(
            parent,
            corner_radius=10,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
            height=70,
        )
        card.grid(row=0, column=col, padx=4, sticky="ew")

        ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont(size=10, weight="bold"), text_color=THEME["text_muted"]
        ).pack(anchor="w", padx=14, pady=(8, 0))

        lbl = ctk.CTkLabel(
            card, text=value, font=ctk.CTkFont(size=16, weight="bold"), text_color=val_color
        )
        lbl.pack(anchor="w", padx=14, pady=(0, 8))
        return lbl

    # ------------------------------------------------------------------
    # VIEW: Predict & Inference
    # ------------------------------------------------------------------

    def _create_predict_view(self) -> ctk.CTkFrame:
        view = ctk.CTkFrame(self.views_frame, fg_color="transparent")
        view.grid_columnconfigure((0, 1), weight=1)
        view.grid_rowconfigure(1, weight=1)

        # ── Top Card: Batch Inference Controls ──
        ctrl_card = ctk.CTkFrame(
            view,
            corner_radius=12,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        ctrl_card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ctrl_card.grid_columnconfigure(1, weight=1)

        btn_box = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        btn_box.grid(row=0, column=0, padx=16, pady=14, sticky="w")

        self._pred_btn = ctk.CTkButton(
            btn_box,
            text="⚡  Run 2,000 Evaluation Inference",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40,
            fg_color=THEME["accent_blue"],
            hover_color="#2563eb",
            command=self._start_prediction,
        )
        self._pred_btn.pack(side="left", padx=(0, 10))

        self._tta_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            btn_box,
            text="TTA (Horizontal Flip)",
            variable=self._tta_var,
            font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=10)

        self._ensemble_var = ctk.StringVar(value="soft")
        ctk.CTkOptionMenu(
            btn_box,
            variable=self._ensemble_var,
            values=["soft", "hard"],
            width=100,
            height=32,
            fg_color=THEME["sidebar_bg"],
        ).pack(side="left", padx=10)

        # Progress bar
        pbar_box = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        pbar_box.grid(row=0, column=1, padx=16, pady=14, sticky="ew")
        pbar_box.grid_columnconfigure(0, weight=1)

        self.pred_status_lbl = ctk.CTkLabel(
            pbar_box,
            text="Ready to generate 2,000 predictions for submission.csv",
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_muted"],
        )
        self.pred_status_lbl.grid(row=0, column=0, sticky="w", pady=(0, 4))

        self._pred_pbar = ctk.CTkProgressBar(pbar_box, height=10, corner_radius=5)
        self._pred_pbar.grid(row=1, column=0, sticky="ew")
        self._pred_pbar.set(0)

        # ── Left Column: Live Batch Preview ──
        batch_card = ctk.CTkFrame(
            view,
            corner_radius=12,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        batch_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6))

        ctk.CTkLabel(
            batch_card,
            text="🖼️ Batch Inference Stream Preview",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_main"],
        ).pack(anchor="w", padx=16, pady=(14, 6))

        self._preview_lbl = ctk.CTkLabel(
            batch_card,
            text="Awaiting batch inference execution...",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_muted"],
        )
        self._preview_lbl.pack(pady=4)

        self._preview_img_lbl = ctk.CTkLabel(batch_card, text="")
        self._preview_img_lbl.pack(pady=12, expand=True)

        ctk.CTkButton(
            batch_card,
            text="📄 View submission.csv",
            font=ctk.CTkFont(size=12),
            height=32,
            fg_color=THEME["btn_hover"],
            command=self._open_submission_csv,
        ).pack(pady=(0, 14))

        # ── Right Column: Interactive Single-Image Tester ──
        single_card = ctk.CTkFrame(
            view,
            corner_radius=12,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        single_card.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        ctk.CTkLabel(
            single_card,
            text="🔬 Single Image Interactive Diagnostic",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_main"],
        ).pack(anchor="w", padx=16, pady=(14, 4))

        ctk.CTkLabel(
            single_card,
            text="Select any image to test Crater vs. Mound with confidence:",
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_muted"],
        ).pack(anchor="w", padx=16, pady=(0, 10))

        test_btn_bar = ctk.CTkFrame(single_card, fg_color="transparent")
        test_btn_bar.pack(fill="x", padx=16, pady=4)

        ctk.CTkButton(
            test_btn_bar,
            text="📁 Select Image File...",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34,
            fg_color=THEME["btn_hover"],
            command=self._test_single_image,
        ).pack(side="left")

        self.single_img_lbl = ctk.CTkLabel(single_card, text="")
        self.single_img_lbl.pack(pady=12, expand=True)

        self.single_result_chip = ctk.CTkLabel(
            single_card,
            text="No Image Tested",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_muted"],
        )
        self.single_result_chip.pack(pady=(0, 4))

        self.single_conf_lbl = ctk.CTkLabel(
            single_card,
            text="Confidence: —",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_muted"],
        )
        self.single_conf_lbl.pack(pady=(0, 14))

        return view

    def _test_single_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif")])
        if not path:
            return

        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                messagebox.showerror("Error", "Could not read image file.")
                return

            # Display resized preview
            pil_img = Image.fromarray(img).resize((180, 180))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(180, 180))
            self.single_img_lbl.configure(image=ctk_img, text="")

            # CLAHE & Preprocessing
            img_c = _CLAHE.apply(img)
            x = img_c.astype(np.float32) / 255.0
            x = np.stack([x, x, x], axis=0)
            x = torch.from_numpy(x)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            x = (x - mean) / std
            x = x.unsqueeze(0).to(self.config_obj.device)

            # Load models
            models = self._load_inference_models()
            if not models:
                self.single_result_chip.configure(text="No Models Found", text_color=THEME["accent_red"])
                return

            probs = torch.zeros(len(models), 1, 2, device=self.config_obj.device)
            with torch.no_grad():
                for mi, m in enumerate(models):
                    probs[mi] = F.softmax(m(x), dim=1)

            avg_probs = probs.mean(0)[0]
            label = int(avg_probs.argmax().item())
            conf = float(avg_probs[label].item()) * 100

            if label == 1:
                self.single_result_chip.configure(text="⛰️ MOUND / RISE (1)", text_color=THEME["accent_amber"])
            else:
                self.single_result_chip.configure(text="🕳️ CRATER / DEPTH (0)", text_color=THEME["accent_cyan"])

            self.single_conf_lbl.configure(text=f"Model Ensemble Confidence: {conf:.1f}%")
        except Exception as e:
            messagebox.showerror("Inference Error", f"Failed single inference: {e}")

    def _load_inference_models(self):
        import glob
        models = []
        for ckpt in glob.glob(os.path.join(self.config_obj.output_dir, "*_best.pth")):
            basename = os.path.basename(ckpt)
            model_name = basename.split("_fold_")[0]
            try:
                m = PareidoliaModel(model_name)
                m.load_state_dict(torch.load(ckpt, map_location=self.config_obj.device))
                models.append(m.to(self.config_obj.device).eval())
            except Exception:
                pass
        return models

    # ------------------------------------------------------------------
    # VIEW: Results & Evaluation
    # ------------------------------------------------------------------

    def _create_results_view(self) -> ctk.CTkFrame:
        view = ctk.CTkFrame(self.views_frame, fg_color="transparent")
        view.grid_columnconfigure((0, 1), weight=1)
        view.grid_rowconfigure(0, weight=1)

        # ── Left: Scorecard Table ──
        table_card = ctk.CTkFrame(
            view,
            corner_radius=12,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        table_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        ctk.CTkLabel(
            table_card,
            text="🏆 Cross-Validation Scorecard",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_main"],
        ).pack(anchor="w", padx=16, pady=(16, 12))

        # Table rows
        scores = [
            ("ConvNeXt-Tiny Fold 1", "0.7292"),
            ("ConvNeXt-Tiny Fold 2", "0.7486"),
            ("ConvNeXt-Tiny Fold 3", "0.7538"),
            ("ConvNeXt-Tiny Fold 4", "0.7538"),
            ("ConvNeXt-Small Fold 1", "0.7328"),
            ("ConvNeXt-Small Fold 2", "0.7350"),
            ("ConvNeXt-Small Fold 3", "0.7683 (Best Single)"),
            ("ConvNeXt-Small Fold 4", "0.7288"),
            ("8-Model Soft Ensemble", "0.7620 (Overall Final)"),
        ]

        for name, score in scores:
            row_frame = ctk.CTkFrame(table_card, fg_color=THEME["sidebar_bg"], height=34, corner_radius=6)
            row_frame.pack(fill="x", padx=16, pady=3)
            is_best = "Ensemble" in score or "Best Single" in score
            ctk.CTkLabel(
                row_frame,
                text=name,
                font=ctk.CTkFont(size=12, weight="bold" if is_best else "normal"),
                text_color=THEME["text_main"] if is_best else THEME["text_muted"],
            ).pack(side="left", padx=12)

            ctk.CTkLabel(
                row_frame,
                text=score,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=THEME["accent_green"] if is_best else THEME["text_main"],
            ).pack(side="right", padx=12)

        # ── Right: Confusion Matrix ──
        cm_card = ctk.CTkFrame(
            view,
            corner_radius=12,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        cm_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        ctk.CTkLabel(
            cm_card,
            text="🎯 Validation Confusion Matrix",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_main"],
        ).pack(anchor="w", padx=16, pady=(16, 4))

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(THEME["card_bg"])
        ax.set_facecolor(THEME["sidebar_bg"])
        ax.tick_params(colors="#94a3b8")
        ax.title.set_color("#f8fafc")
        ax.set_title("Cross-Validation Error Distribution", fontsize=11, fontweight="bold", pad=10)
        for spine in ax.spines.values():
            spine.set_color(THEME["card_border"])

        # Default placeholder matrix
        default_cm = np.array([[520, 180], [140, 1160]])
        ax.imshow(default_cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Depth (0)", "Rise (1)"], color="#94a3b8")
        ax.set_yticklabels(["Depth (0)", "Rise (1)"], color="#94a3b8")
        ax.set_xlabel("Predicted Label", color="#94a3b8")
        ax.set_ylabel("Ground Truth", color="#94a3b8")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(default_cm[i, j]), ha="center", va="center", color="white", fontweight="bold")
        plt.tight_layout()

        self._cm_fig, self._cm_ax = fig, ax
        self._cm_canvas = FigureCanvasTkAgg(fig, master=cm_card)
        self._cm_canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=12)

        return view

    # ------------------------------------------------------------------
    # VIEW: Logs
    # ------------------------------------------------------------------

    def _create_logs_view(self) -> ctk.CTkFrame:
        view = ctk.CTkFrame(self.views_frame, fg_color="transparent")
        view.grid_columnconfigure(0, weight=1)
        view.grid_rowconfigure(0, weight=1)

        card = ctk.CTkFrame(
            view,
            corner_radius=12,
            fg_color=THEME["card_bg"],
            border_width=1,
            border_color=THEME["card_border"],
        )
        card.grid(row=0, column=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))

        ctk.CTkLabel(
            btn_row,
            text="📜 Real-Time Process Console",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_main"],
        ).pack(side="left")

        ctk.CTkButton(
            btn_row,
            text="🗑  Clear Console",
            width=110,
            height=30,
            font=ctk.CTkFont(size=11),
            fg_color=THEME["btn_hover"],
            command=lambda: self._log_box.delete("1.0", ctk.END),
        ).pack(side="right", padx=4)

        ctk.CTkButton(
            btn_row,
            text="💾  Export Log File",
            width=110,
            height=30,
            font=ctk.CTkFont(size=11),
            fg_color=THEME["btn_hover"],
            command=self._export_logs,
        ).pack(side="right", padx=4)

        self._log_box = ctk.CTkTextbox(
            card,
            font=("Consolas", 12),
            wrap="word",
            fg_color=THEME["sidebar_bg"],
            text_color="#e2e8f0",
            corner_radius=8,
        )
        self._log_box.grid(row=1, column=0, sticky="nsew", padx=16, pady=(6, 16))

        sys.stdout = _LogRedirect(self._log_box)
        return view

    def _export_logs(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._log_box.get("1.0", ctk.END))
            messagebox.showinfo("Exported", f"Log saved to: {path}")

    # ------------------------------------------------------------------
    # Pipeline Execution & Polling Loops
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        while not self._queue.empty():
            msg = self._queue.get_nowait()
            mtype = msg.get("type")

            if mtype == "progress":
                self._train_pbar.set(msg["value"])
                self.train_status_lbl.configure(text=msg.get("text", "Training in progress..."))
                self._set_status(msg.get("text", ""))

            elif mtype == "predict_progress":
                self._pred_pbar.set(msg["value"])
                self.pred_status_lbl.configure(text=msg.get("text", "Generating predictions..."))
                self._set_status(msg.get("text", ""))

            elif mtype == "epoch_end":
                f = msg.get("fold", 1)
                ep = msg.get("epoch", 1)
                vb = msg.get("val_bacc", 0.0)
                self.kpi_fold.configure(text=f"Fold {f} / {self.config_obj.n_splits}")
                self.kpi_epoch.configure(text=f"{ep} / {self.config_obj.epochs}")
                self.kpi_val_bacc.configure(text=f"{vb*100:.2f}%")

                if vb > self._best_bacc:
                    self._best_bacc = vb
                    self.kpi_best.configure(text=f"{self._best_bacc:.4f}")

                self._train_losses.append(msg["train_loss"])
                self._val_losses.append(msg["val_loss"])
                self._val_baccs.append(vb)
                self._refresh_train_chart()

            elif mtype == "fold_end":
                self._refresh_cm(msg["cm"])

            elif mtype == "training_complete":
                self._train_btn.configure(text="▶  Start Training Pipeline", fg_color=THEME["accent_green"])
                self.train_status_lbl.configure(text="Training successfully completed! Models saved ✓")
                self._set_status("Training complete ✓")

            elif mtype == "predict_complete":
                self._pred_pbar.set(1.0)
                self.pred_status_lbl.configure(text=f"Completed! 2,000 predictions saved to {msg.get('path', 'submission.csv')}")
                self._set_status(f"Submission saved → {msg['path']}")

            elif mtype == "preview":
                self._show_preview(msg["image"], msg["prediction"])

            elif mtype == "error":
                messagebox.showerror("Execution Error", msg["message"])

        # Update elapsed time if training is running
        if self._start_time > 0 and self.train_thread and self.train_thread.is_alive():
            elapsed = int(time.time() - self._start_time)
            mins, secs = divmod(elapsed, 60)
            self.status_text.set(f"Training active • Elapsed: {mins:02d}:{secs:02d} • Device: {self.config_obj.device}")

        self.after(100, self._tick)

    def _set_status(self, text: str) -> None:
        self.status_text.set(f"{text} • Device: {self.config_obj.device}")

    def _refresh_train_chart(self) -> None:
        ax1, ax2 = self._train_ax1, self._train_ax2
        ax1.clear(); ax2.clear()

        for ax in (ax1, ax2):
            ax.set_facecolor(THEME["sidebar_bg"])
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.title.set_color("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color(THEME["card_border"])
            ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")

        ax1.plot(self._train_losses, label="Train Loss", color=THEME["accent_cyan"], linewidth=1.8)
        ax1.plot(self._val_losses, label="Val Loss", color=THEME["accent_amber"], linestyle="--", linewidth=1.8)
        ax1.set_title("Loss Trajectory", fontsize=11, fontweight="bold", pad=10)
        ax1.legend(fontsize=9, facecolor=THEME["card_bg"], edgecolor=THEME["card_border"], labelcolor="#ffffff")

        ax2.plot(self._val_baccs, color=THEME["accent_green"], linewidth=2.0)
        ax2.set_title(f"Validation Balanced Accuracy (Peak: {self._best_bacc:.4f})", fontsize=11, fontweight="bold", pad=10)

        self._train_fig.tight_layout()
        self._train_canvas.draw_idle()

    def _refresh_cm(self, cm: list) -> None:
        cm_arr = np.array(cm)
        ax = self._cm_ax
        ax.clear()
        ax.set_facecolor(THEME["sidebar_bg"])
        ax.tick_params(colors="#94a3b8")
        ax.title.set_color("#f8fafc")
        ax.set_title("Cross-Validation Error Distribution (Last Fold)", fontsize=11, fontweight="bold", pad=10)

        ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Depth (0)", "Rise (1)"], color="#94a3b8")
        ax.set_yticklabels(["Depth (0)", "Rise (1)"], color="#94a3b8")
        ax.set_xlabel("Predicted Label", color="#94a3b8")
        ax.set_ylabel("Ground Truth", color="#94a3b8")

        vmax = cm_arr.max()
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="black" if cm_arr[i, j] > vmax / 2 else "white",
                    fontsize=14, fontweight="bold",
                )
        self._cm_fig.tight_layout()
        self._cm_canvas.draw_idle()

    def _show_preview(self, img_np: np.ndarray, prediction: int) -> None:
        img_u8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8, mode="L").resize((200, 200))
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(200, 200))
        label_text = "⛰️ Mound / Rise (1)" if prediction == 1 else "🕳️ Crater / Depth (0)"
        color = THEME["accent_amber"] if prediction == 1 else THEME["accent_cyan"]
        self._preview_img_lbl.configure(image=ctk_img, text="")
        self._preview_lbl.configure(text=f"Batch Output: {label_text}", text_color=color)

    # ------------------------------------------------------------------
    # Controls: Train & Predict
    # ------------------------------------------------------------------

    def _toggle_training(self) -> None:
        if self.train_thread and self.train_thread.is_alive():
            self.stop_event.set()
            self._train_btn.configure(text="▶  Start Training Pipeline", fg_color=THEME["accent_green"])
            self._set_status("Stopping training gracefully...")
        else:
            self.stop_event.clear()
            self._train_losses.clear()
            self._val_losses.clear()
            self._val_baccs.clear()
            self._best_bacc = 0.0
            self._train_btn.configure(text="⏹  Stop Training", fg_color=THEME["accent_red"])
            self._start_time = time.time()
            self.train_thread = threading.Thread(
                target=train_pipeline,
                args=(self.config_obj, self._queue, self.stop_event),
                daemon=True,
            )
            self.train_thread.start()
            self._set_status("Training pipeline launched...")

    def _start_prediction(self) -> None:
        self._pred_pbar.set(0)
        self.pred_status_lbl.configure(text="Ensemble prediction active across 2,000 images...")
        self.pred_thread = threading.Thread(
            target=predict_pipeline,
            args=(
                self.config_obj,
                self._tta_var.get(),
                self._ensemble_var.get(),
                self._queue,
            ),
            daemon=True,
        )
        self.pred_thread.start()
        self._set_status("Inference pipeline started...")

    def _open_project_dir(self) -> None:
        try:
            if os.name == "nt":
                os.startfile(PKG_DIR)
            else:
                subprocess.Popen(["xdg-open", PKG_DIR])
        except Exception as e:
            messagebox.showwarning("Open Folder", f"Could not open directory: {e}")

    def _open_submission_csv(self) -> None:
        path = os.path.join(self.config_obj.output_dir, "submission.csv")
        if not os.path.exists(path):
            path = os.path.join(PKG_DIR, "submission.csv")
        if os.path.exists(path):
            try:
                if os.name == "nt":
                    os.startfile(path)
                else:
                    subprocess.Popen(["xdg-open", path])
            except Exception:
                pass
        else:
            messagebox.showinfo("Inference Needed", "No submission.csv found yet. Click 'Run Inference' first.")


def launch_gui() -> None:
    """Entry point: launch the rebuilt UI."""
    app = PareidoliaApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
