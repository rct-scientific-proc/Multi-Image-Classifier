"""
Metrics — loss and classification metrics tracking per epoch.

Available metrics (keys in the dict returned by compute()):
    avg_loss              — mean cross-entropy loss over all batches
    accuracy              — overall top-1 accuracy
    top_k_accuracy        — top-k accuracy (k = min(5, num_classes))
    precision_macro       — macro-averaged precision
    precision_weighted    — sample-weighted precision
    recall_macro          — macro-averaged recall
    recall_weighted       — sample-weighted recall
    f1_macro              — macro-averaged F1
    f1_weighted           — sample-weighted F1
    specificity_macro     — macro-averaged specificity (TN / (TN+FP), one-vs-rest)
    specificity_weighted  — sample-weighted specificity
    mcc                   — Matthews Correlation Coefficient
    auc_macro             — macro-averaged one-vs-rest ROC AUC
    auc_weighted          — sample-weighted one-vs-rest ROC AUC
    per_class_accuracy    — list[float], one value per class
    per_class_specificity — list[float], one value per class
    per_class_auc         — list[float], one value per class (nan if class absent in epoch)
    confusion_matrix      — np.ndarray (num_classes, num_classes)

Usage:
    tracker = MetricTracker(num_classes=11)
    for logits, labels in batches:
        tracker.update(logits, labels, loss)
    summary = tracker.compute()
    tracker.reset()
"""

import numpy as np
import torch

# Target metric choices exposed to the GUI / CLI
TARGET_METRICS = [
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "specificity_macro",
    "specificity_weighted",
    "mcc",
    "auc_macro",
    "auc_weighted",
    "top_k_accuracy",
    "avg_loss",          # minimise — lower is better
]
DEFAULT_TARGET_METRIC = "f1_macro"


def _roc_auc_binary(labels_bin: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC for a single one-vs-rest binary problem (no sklearn dependency).

    Returns nan if only one class is present in labels_bin.
    """
    n_pos = int(labels_bin.sum())
    n_neg = len(labels_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order        = np.argsort(-scores)
    labels_sorted = labels_bin[order]
    tpr = np.concatenate([[0.0], np.cumsum(labels_sorted) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - labels_sorted) / n_neg])
    return float(np.trapezoid(tpr, fpr))


class MetricTracker:
    """Accumulates per-batch stats and computes epoch-level metrics."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self._top_k      = min(5, num_classes)
        self.reset()

    def reset(self):
        self._loss_sum    = 0.0
        self._loss_count  = 0
        self._confusion   = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self._top_k_hits  = 0
        self._total       = 0
        self._probs_list  = []   # list of (B, C) float32 numpy arrays
        self._labels_list = []   # list of (B,) int64 numpy arrays

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float):
        """Call after each batch.

        Parameters
        ----------
        logits : (B, num_classes) — raw model output (before softmax)
        labels : (B,)            — ground-truth class indices
        loss   : scalar loss value for this batch
        """
        with torch.no_grad():
            probs_np  = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        labels_np = labels.cpu().numpy()
        preds_np  = logits.argmax(dim=1).cpu().numpy()

        self._loss_sum   += loss
        self._loss_count += 1
        self._total      += len(labels_np)

        self._probs_list.append(probs_np)
        self._labels_list.append(labels_np)

        for t, p in zip(labels_np, preds_np):
            self._confusion[t, p] += 1

        # Top-k hits
        top_k_preds = logits.topk(self._top_k, dim=1).indices.cpu().numpy()  # (B, k)
        for i, t in enumerate(labels_np):
            if t in top_k_preds[i]:
                self._top_k_hits += 1

    def compute(self) -> dict:
        """Return a dict of all metrics for the epoch."""
        C        = self._confusion
        avg_loss = self._loss_sum / max(self._loss_count, 1)

        # ---- top-1 accuracy ----
        correct  = np.diag(C).sum()
        total    = C.sum()
        accuracy = float(correct) / float(max(total, 1))

        # ---- top-k accuracy ----
        top_k_accuracy = float(self._top_k_hits) / float(max(self._total, 1))

        # ---- per-class support ----
        support = C.sum(axis=1)                    # actual positives per class (row sums)
        tp      = np.diag(C).astype(float)
        fp      = C.sum(axis=0).astype(float) - tp
        fn      = support.astype(float) - tp
        tn      = float(total) - tp - fp - fn

        # ---- precision / recall / F1 per class ----
        precision_c = tp / np.maximum(tp + fp, 1)
        recall_c    = tp / np.maximum(tp + fn, 1)
        f1_c        = (2 * precision_c * recall_c) / np.maximum(precision_c + recall_c, 1e-9)

        # ---- specificity per class (one-vs-rest: TN / (TN + FP)) ----
        specificity_c = tn / np.maximum(tn + fp, 1)

        # ---- macro averages ----
        precision_macro    = float(precision_c.mean())
        recall_macro       = float(recall_c.mean())
        f1_macro           = float(f1_c.mean())
        specificity_macro  = float(specificity_c.mean())

        # ---- weighted averages ----
        w = support.astype(float) / float(max(support.sum(), 1))
        precision_weighted   = float((precision_c   * w).sum())
        recall_weighted      = float((recall_c      * w).sum())
        f1_weighted          = float((f1_c          * w).sum())
        specificity_weighted = float((specificity_c * w).sum())

        # ---- Matthews Correlation Coefficient (multiclass) ----
        N   = float(total)
        num = N * np.trace(C) - np.sum(C.sum(axis=1) * C.sum(axis=0))
        d1  = np.sqrt(max(N * N - np.sum(C.sum(axis=1) ** 2), 0.0))
        d2  = np.sqrt(max(N * N - np.sum(C.sum(axis=0) ** 2), 0.0))
        mcc = float(num / max(d1 * d2, 1e-9))

        # ---- per-class ROC AUC (one-vs-rest) ----
        all_probs  = np.concatenate(self._probs_list,  axis=0)   # (N, C)
        all_labels = np.concatenate(self._labels_list, axis=0)   # (N,)

        per_class_auc = []
        for c in range(self.num_classes):
            bin_labels = (all_labels == c).astype(np.int32)
            auc_c      = _roc_auc_binary(bin_labels, all_probs[:, c])
            per_class_auc.append(auc_c)

        valid_aucs  = [v for v in per_class_auc if not np.isnan(v)]
        auc_macro   = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
        w_sum       = sum(float(support[c]) for c in range(self.num_classes) if not np.isnan(per_class_auc[c]))
        auc_weighted = (
            float(sum(per_class_auc[c] * float(support[c])
                      for c in range(self.num_classes)
                      if not np.isnan(per_class_auc[c])) / max(w_sum, 1))
            if valid_aucs else float("nan")
        )

        # ---- per-class accuracy ----
        per_class_accuracy = [
            float(C[i, i]) / float(max(support[i], 1))
            for i in range(self.num_classes)
        ]

        return {
            "avg_loss":             avg_loss,
            "accuracy":             accuracy,
            "top_k_accuracy":       top_k_accuracy,
            "precision_macro":      precision_macro,
            "precision_weighted":   precision_weighted,
            "recall_macro":         recall_macro,
            "recall_weighted":      recall_weighted,
            "f1_macro":             f1_macro,
            "f1_weighted":          f1_weighted,
            "specificity_macro":    specificity_macro,
            "specificity_weighted": specificity_weighted,
            "mcc":                  mcc,
            "auc_macro":            auc_macro,
            "auc_weighted":         auc_weighted,
            "per_class_accuracy":   per_class_accuracy,
            "per_class_specificity": list(specificity_c.tolist()),
            "per_class_auc":        per_class_auc,
            "confusion_matrix":     C.copy(),
        }

