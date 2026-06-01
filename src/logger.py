"""
Logger — TensorBoard SummaryWriter wrapper.

Log directory convention:
    runs/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/

Logged per epoch:
    Scalars  : train/loss, train/accuracy, val/loss, val/<target_metric>,
               val/accuracy, val/f1_macro, val/f1_weighted,
               val/precision_macro, val/recall_macro,
               val/specificity_macro, val/mcc, val/auc_macro, val/auc_weighted,
               val/top_k_accuracy, lr
    Images   : val/confusion_matrix  (colour-mapped heatmap)
    Per-class: val/per_class/auc_<classname>,
               val/per_class/accuracy_<classname>,
               val/per_class/specificity_<classname>

Logged once at the end of training:
    HParams  : all hyperparams + best val target metric value

Usage:
    logger = ExperimentLogger("runs", experiment_name="mnist_baseline")
    # inside on_epoch_end callback:
    logger.log_epoch(info, class_names)
    # after training finishes:
    logger.log_hparams(hyperparams, best_metrics)
    logger.close()
"""

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


# Scalar keys from val_metrics to log directly (excludes list/array values)
_VAL_SCALAR_KEYS = [
    "avg_loss", "accuracy", "top_k_accuracy",
    "precision_macro", "precision_weighted",
    "recall_macro",    "recall_weighted",
    "f1_macro",        "f1_weighted",
    "specificity_macro", "specificity_weighted",
    "mcc", "auc_macro", "auc_weighted",
]


def _confusion_matrix_image(cm: np.ndarray, class_names: list[str]) -> torch.Tensor:
    """Render a confusion matrix as a (1, 3, H, W) float32 tensor for TensorBoard."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)

    from PIL import Image as PILImage
    img = np.array(PILImage.open(buf).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


class ExperimentLogger:
    """Thin wrapper around SummaryWriter with helpers for this project's metrics."""

    def __init__(self, log_root: str | Path, experiment_name: str = "experiment"):
        timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = Path(log_root) / experiment_name / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        print(f"TensorBoard log dir: {self.log_dir}")

    # ------------------------------------------------------------------
    def log_epoch(self, info: dict, class_names: list[str]):
        """Log all metrics for one epoch.

        Parameters
        ----------
        info        : dict emitted by Trainer.fit via on_epoch_end
        class_names : list of class name strings (from dataset.classes)
        """
        epoch        = info["epoch"]
        train_m      = info["train_metrics"]
        val_m        = info["val_metrics"]
        w            = self._writer

        # ---- train scalars ----
        w.add_scalar("train/loss",     train_m["avg_loss"], epoch)
        w.add_scalar("train/accuracy", train_m["accuracy"], epoch)
        w.add_scalar("train/f1_macro", train_m["f1_macro"], epoch)

        # ---- val scalars ----
        for key in _VAL_SCALAR_KEYS:
            val = val_m.get(key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                w.add_scalar(f"val/{key}", val, epoch)

        # ---- learning rate ----
        w.add_scalar("train/lr", info["lr"], epoch)

        # ---- per-class scalars ----
        for c, name in enumerate(class_names):
            auc_c  = val_m["per_class_auc"][c]
            acc_c  = val_m["per_class_accuracy"][c]
            spec_c = val_m["per_class_specificity"][c]
            if not np.isnan(auc_c):
                w.add_scalar(f"val/per_class/auc_{name}",         auc_c,  epoch)
            w.add_scalar(f"val/per_class/accuracy_{name}",     acc_c,  epoch)
            w.add_scalar(f"val/per_class/specificity_{name}",  spec_c, epoch)

        # ---- per-class probability thresholds at target recall(s) ----
        for r_key, entries in val_m.get("per_class_thresholds", {}).items():
            for c, name in enumerate(class_names):
                e = entries[c]
                if np.isnan(e["threshold"]):
                    continue
                w.add_scalar(f"val/threshold@r{r_key}/{name}", e["threshold"], epoch)
                w.add_scalar(f"val/precision@r{r_key}/{name}", e["precision"], epoch)

        # ---- confusion matrix image ----
        cm_tensor = _confusion_matrix_image(val_m["confusion_matrix"], class_names)
        if cm_tensor is not None:
            w.add_images("val/confusion_matrix", cm_tensor, epoch)

    # ------------------------------------------------------------------
    def log_hparams(self, hyperparams: dict, best_metrics: dict):
        """Log hyperparameters + final best metric values (called once after training).

        Only scalar (int/float/str/bool) hparam values are included — TensorBoard
        does not accept nested dicts or lists.
        """
        flat_hparams = {
            k: v for k, v in hyperparams.items()
            if isinstance(v, (int, float, str, bool))
        }
        flat_metrics = {
            f"hparam/{k}": v
            for k, v in best_metrics.items()
            if isinstance(v, (int, float)) and not np.isnan(v)
        }
        self._writer.add_hparams(flat_hparams, flat_metrics)

    # ------------------------------------------------------------------
    def close(self):
        self._writer.flush()
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
