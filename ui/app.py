import os
import queue
import threading
import time

import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image

from pareidolia.utils import Config
from pareidolia.train import train_pipeline
from pareidolia.predict import predict_pipeline


class _LogRedirect:
    """Redirects stdout writes to a CTkTextbox widget."""

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
    """Dark-themed CustomTkinter GUI for the Pareidolia Paradox pipeline."""

    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.title("🌕 Pareidolia Paradox — ML Pipeline")
        self.geometry("1200x780")
        self.minsize(1100, 750)

        self.config_obj   = Config.load("config.json")
        self._queue       = queue.Queue()
        self.stop_event   = threading.Event()
        self.train_thread = None
        self.pred_thread  = None
        self._start_time  = 0.0

        # Chart data accumulators
        self._train_losses: list = []
        self._val_losses:   list = []
        self._val_baccs:    list = []

        self._build_ui()
        self._tick()  # start queue polling loop

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ---- Sidebar ----
        sidebar = ctk.CTkFrame(self, width=180, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(sidebar, text="🌕  Pareidolia",
                     font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, padx=16, pady=(20, 30))

        for i, (emoji, name) in enumerate([
            ("⚙️", "Setup"), ("🚂", "Train"), ("🔍", "Predict"),
            ("📊", "Results"), ("📄", "Logs"),
        ], start=1):
            ctk.CTkButton(
                sidebar, text=f"{emoji}  {name}", anchor="w",
                command=lambda n=name: self.tabs.set(n),
            ).grid(row=i, column=0, padx=12, pady=6, sticky="ew")

        # ---- Main content ----
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(main)
        self.tabs.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")

        for name in ("Setup", "Train", "Predict", "Results", "Logs"):
            self.tabs.add(name)

        self._build_setup_tab()
        self._build_train_tab()
        self._build_predict_tab()
        self._build_results_tab()
        self._build_logs_tab()

        # ---- Status bar ----
        self.status_var = ctk.StringVar(value="Ready")
        ctk.CTkLabel(self, textvariable=self.status_var, anchor="w",
                     fg_color=("gray75", "gray20"), height=28).grid(
            row=1, column=1, sticky="ew", padx=0, pady=0)

    # ---- Tab: Setup ----
    def _build_setup_tab(self) -> None:
        tab = self.tabs.tab("Setup")
        tab.grid_columnconfigure(1, weight=1)

        fields = [
            ("Train Images Dir",  "train_images_dir",  "dir"),
            ("Test Images Dir",   "test_images_dir",   "dir"),
            ("Train CSV",         "train_csv",         "file"),
            ("Test CSV",          "test_csv",          "file"),
            ("Output Directory",  "output_dir",        "dir"),
        ]

        self._path_vars: dict = {}
        for row, (label, attr, kind) in enumerate(fields):
            var = ctk.StringVar(value=getattr(self.config_obj, attr))
            self._path_vars[attr] = var
            ctk.CTkLabel(tab, text=label + ":").grid(
                row=row, column=0, padx=10, pady=8, sticky="w")
            ctk.CTkEntry(tab, textvariable=var).grid(
                row=row, column=1, padx=10, pady=8, sticky="ew")
            btn_cmd = (
                (lambda v: lambda: v.set(filedialog.askdirectory() or v.get()))(var)
                if kind == "dir"
                else (lambda v: lambda: v.set(
                    filedialog.askopenfilename(filetypes=[("CSV", "*.csv")]) or v.get()
                ))(var)
            )
            ctk.CTkButton(tab, text="Browse", width=80, command=btn_cmd).grid(
                row=row, column=2, padx=10, pady=8)

        row = len(fields)
        self._device_var = ctk.StringVar(value=self.config_obj.device)
        ctk.CTkLabel(tab, text="Device:").grid(row=row, column=0, padx=10, pady=8, sticky="w")
        ctk.CTkOptionMenu(tab, variable=self._device_var,
                          values=["cuda", "cpu"]).grid(
            row=row, column=1, padx=10, pady=8, sticky="w")

        ctk.CTkButton(tab, text="💾  Save Settings",
                      command=self._save_settings).grid(
            row=row + 1, column=0, columnspan=3, pady=24)

    def _save_settings(self) -> None:
        for attr, var in self._path_vars.items():
            setattr(self.config_obj, attr, var.get())
        self.config_obj.device = self._device_var.get()
        self.config_obj.save("config.json")
        messagebox.showinfo("Saved", "Settings saved to config.json")
        self._set_status("Settings saved.")

    # ---- Tab: Train ----
    def _build_train_tab(self) -> None:
        tab = self.tabs.tab("Train")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        self._train_btn = ctk.CTkButton(
            ctrl, text="▶  Start Training", fg_color="green",
            command=self._toggle_training)
        self._train_btn.pack(side="left", padx=10, pady=8)

        self._fold_lbl = ctk.CTkLabel(ctrl, text="Fold: —  Epoch: —")
        self._fold_lbl.pack(side="left", padx=20)

        self._train_pbar = ctk.CTkProgressBar(ctrl, width=280)
        self._train_pbar.pack(side="right", padx=10)
        self._train_pbar.set(0)

        # Embedded matplotlib chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
        fig.patch.set_facecolor("#1e1e2e")
        for ax in (ax1, ax2):
            ax.set_facecolor("#2a2a3e")
            ax.tick_params(colors="white")
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
        ax1.set_title("Loss"); ax2.set_title("Balanced Accuracy")
        plt.tight_layout()

        self._train_fig = fig
        self._train_ax1, self._train_ax2 = ax1, ax2
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self._train_canvas = canvas

    # ---- Tab: Predict ----
    def _build_predict_tab(self) -> None:
        tab = self.tabs.tab("Predict")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        ctrl = ctk.CTkFrame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        self._pred_btn = ctk.CTkButton(
            ctrl, text="🔍  Run Inference", command=self._start_prediction)
        self._pred_btn.pack(side="left", padx=10, pady=8)

        self._tta_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(ctrl, text="Use TTA (8×)", variable=self._tta_var).pack(
            side="left", padx=16)

        self._ensemble_var = ctk.StringVar(value="soft")
        ctk.CTkOptionMenu(ctrl, variable=self._ensemble_var,
                          values=["soft", "hard"]).pack(side="left", padx=8)

        self._pred_pbar = ctk.CTkProgressBar(ctrl, width=240)
        self._pred_pbar.pack(side="right", padx=10)
        self._pred_pbar.set(0)

        self._preview_lbl = ctk.CTkLabel(
            tab, text="Prediction preview will appear here.",
            font=ctk.CTkFont(size=13))
        self._preview_lbl.grid(row=1, column=0, pady=12)

        self._preview_img_lbl = ctk.CTkLabel(tab, text="")
        self._preview_img_lbl.grid(row=2, column=0)

    # ---- Tab: Results ----
    def _build_results_tab(self) -> None:
        tab = self.tabs.tab("Results")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        self._results_summary = ctk.CTkLabel(
            tab, text="Run training to see per-fold results.",
            font=ctk.CTkFont(size=13))
        self._results_summary.grid(row=0, column=0, padx=16, pady=12)

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        self._cm_fig, self._cm_ax = fig, ax
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=16, pady=8)
        self._cm_canvas = canvas

    # ---- Tab: Logs ----
    def _build_logs_tab(self) -> None:
        import sys
        tab = self.tabs.tab("Logs")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        self._log_box = ctk.CTkTextbox(tab, font=("Consolas", 11), wrap="word")
        self._log_box.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        ctk.CTkButton(btn_row, text="🗑  Clear", width=100,
                      command=lambda: self._log_box.delete("1.0", ctk.END)).pack(
            side="left", padx=4)
        ctk.CTkButton(btn_row, text="💾  Export", width=100,
                      command=self._export_logs).pack(side="left", padx=4)

        sys.stdout = _LogRedirect(self._log_box)

    def _export_logs(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._log_box.get("1.0", ctk.END))

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Poll the work queue and dispatch GUI updates (runs on main thread)."""
        while not self._queue.empty():
            msg = self._queue.get_nowait()
            t   = msg.get("type")

            if t == "progress":
                self._train_pbar.set(msg["value"])
                self._set_status(msg["text"])

            elif t == "predict_progress":
                self._pred_pbar.set(msg["value"])
                self._set_status(msg["text"])

            elif t == "epoch_end":
                self._fold_lbl.configure(
                    text=f"Fold: {msg['fold']}  Epoch: {msg['epoch']}")
                self._train_losses.append(msg["train_loss"])
                self._val_losses.append(msg["val_loss"])
                self._val_baccs.append(msg["val_bacc"])
                self._refresh_train_chart()

            elif t == "fold_end":
                self._refresh_cm(msg["cm"])
                self._results_summary.configure(
                    text=f"Fold {msg['fold']} done  |  best BAcc = {msg['best_bacc']:.4f}")

            elif t == "training_complete":
                self._train_btn.configure(text="▶  Start Training", fg_color="green")
                self._set_status("Training complete ✓")

            elif t == "predict_complete":
                self._pred_pbar.set(1.0)
                self._set_status(f"Submission saved → {msg['path']}")

            elif t == "preview":
                self._show_preview(msg["image"], msg["prediction"])

            elif t == "error":
                messagebox.showerror("Error", msg["message"])

        # Update elapsed time if training is running
        if self._start_time > 0 and self.train_thread and self.train_thread.is_alive():
            elapsed = int(time.time() - self._start_time)
            self.status_var.set(
                self.status_var.get().split("  |  elapsed")[0]
                + f"  |  elapsed {elapsed}s"
            )

        self.after(120, self._tick)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self.status_var.set(
            f"{text}  |  device={self.config_obj.device}"
        )

    def _refresh_train_chart(self) -> None:
        ax1, ax2 = self._train_ax1, self._train_ax2
        ax1.clear(); ax2.clear()
        for ax in (ax1, ax2):
            ax.set_facecolor("#2a2a3e")
            ax.tick_params(colors="white")
            ax.title.set_color("white")

        ax1.plot(self._train_losses, label="train", color="#6eb5ff")
        ax1.plot(self._val_losses,   label="val",   color="#ff9e6e", linestyle="--")
        ax1.set_title("Loss"); ax1.legend(fontsize=8)

        ax2.plot(self._val_baccs, color="#6effa8")
        ax2.set_title("Balanced Accuracy")

        self._train_fig.tight_layout()
        self._train_canvas.draw_idle()

    def _refresh_cm(self, cm: list) -> None:
        cm_arr = np.array(cm)
        ax = self._cm_ax
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.imshow(cm_arr, cmap="Blues")
        ax.set_title("Confusion Matrix (last fold)")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Depth", "Rise"], color="white")
        ax.set_yticklabels(["Depth", "Rise"], color="white")
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual",    color="white")
        vmax = cm_arr.max()
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                        color="black" if cm_arr[i, j] > vmax / 2 else "white",
                        fontsize=14, fontweight="bold")
        self._cm_fig.tight_layout()
        self._cm_canvas.draw_idle()

    def _show_preview(self, img_np: np.ndarray, prediction: int) -> None:
        img_u8  = (img_np * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8, mode="L").resize((200, 200))
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(200, 200))
        label   = "🪨 Mound / Rise" if prediction == 1 else "🕳 Crater / Depth"
        self._preview_img_lbl.configure(image=ctk_img, text="")
        self._preview_lbl.configure(text=f"Prediction: {label}")

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def _toggle_training(self) -> None:
        if self.train_thread and self.train_thread.is_alive():
            self.stop_event.set()
            self._train_btn.configure(text="▶  Start Training", fg_color="green")
            self._set_status("Stopping…")
        else:
            self.stop_event.clear()
            self._train_losses.clear()
            self._val_losses.clear()
            self._val_baccs.clear()
            self._train_btn.configure(text="⏹  Stop Training", fg_color="#c0392b")
            self._start_time = time.time()
            self.train_thread = threading.Thread(
                target=train_pipeline,
                args=(self.config_obj, self._queue, self.stop_event),
                daemon=True,
            )
            self.train_thread.start()
            self._set_status("Training started…")

    # ------------------------------------------------------------------
    # Prediction control
    # ------------------------------------------------------------------

    def _start_prediction(self) -> None:
        self._pred_pbar.set(0)
        self.pred_thread = threading.Thread(
            target=predict_pipeline,
            args=(self.config_obj, self._tta_var.get(),
                  self._ensemble_var.get(), self._queue),
            daemon=True,
        )
        self.pred_thread.start()
        self._set_status("Inference started…")


def launch_gui() -> None:
    """Entry point: create and run the GUI event loop."""
    app = PareidoliaApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
