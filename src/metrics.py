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
    per_class_thresholds  — dict[str, list[dict]]: keyed by target recall (e.g. "0.99"),
                            each entry per class has {threshold, precision, recall}.
                            Only present when recall_targets is provided.
    confusion_matrix      — np.ndarray (num_classes, num_classes)

Score-based metrics (AUC and the recall thresholds) are accumulated into
fixed-width per-class probability histograms rather than by retaining every
sample's probability vector, so tracker memory is O(num_classes * SCORE_BINS)
regardless of dataset size. See SCORE_BINS for the accuracy trade-off.

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


# Histogram resolution for score-based metrics. Probabilities live in [0, 1],
# so uniform bins are well conditioned; 4096 bins keeps AUC error around 1e-4
# and quantises reported thresholds to ~2.4e-4. Raising this costs
# num_classes * bins * 16 bytes of tracker memory and nothing else.
SCORE_BINS = 4096


def _roc_auc_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float:
    """ROC AUC for a single one-vs-rest binary problem (no sklearn dependency).

    *pos_hist* / *neg_hist* are score histograms in ascending-score bin order,
    so they are swept in reverse to walk the ROC curve from the highest-scoring
    samples downwards. The trapezoid rule across a bin gives samples sharing
    that bin half credit — the same handling exact ties get.

    Returns nan if only one class is present.
    """
    n_pos = int(pos_hist.sum())
    n_neg = int(neg_hist.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tpr = np.concatenate([[0.0], np.cumsum(pos_hist[::-1]) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(neg_hist[::-1]) / n_neg])
    return float(np.trapezoid(tpr, fpr))


def _threshold_at_recall_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray,
                                   target_recall: float) -> dict:
    """Smallest probability threshold whose one-vs-rest recall ≥ target_recall.

    Returns {threshold, precision, recall}, where *threshold* is the lower edge
    of the lowest-scoring bin still admitted — i.e. the cutoff to apply as
    ``score >= threshold``. All NaN when the class has no positive samples in
    this epoch.
    """
    n_pos = int(pos_hist.sum())
    if n_pos == 0:
        return {"threshold": float("nan"),
                "precision": float("nan"),
                "recall":    float("nan")}

    bins   = len(pos_hist)
    cum_tp = np.cumsum(pos_hist[::-1])
    cum_fp = np.cumsum(neg_hist[::-1])

    needed = int(np.ceil(target_recall * n_pos))
    needed = max(1, min(needed, n_pos))
    # First reversed-bin index where cum_tp reaches the needed positive count
    k = int(np.searchsorted(cum_tp, needed, side="left"))
    k = min(k, bins - 1)

    tp        = float(cum_tp[k])
    fp        = float(cum_fp[k])
    threshold = float(bins - 1 - k) / float(bins)
    precision = tp / max(tp + fp, 1.0)
    recall    = tp / float(n_pos)
    return {"threshold": threshold, "precision": precision, "recall": recall}


class MetricTracker:
    """Accumulates per-batch stats and computes epoch-level metrics."""

    def __init__(self, num_classes: int, recall_targets: list[float] | None = None,
                 score_bins: int = SCORE_BINS):
        self.num_classes    = num_classes
        self._top_k         = min(5, num_classes)
        self._bins          = max(2, int(score_bins))
        self.recall_targets = sorted({float(r) for r in (recall_targets or [])
                                      if 0.0 < float(r) <= 1.0})
        self.reset()

    def reset(self):
        self._loss_sum    = 0.0
        self._loss_count  = 0
        self._confusion   = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self._top_k_hits  = 0
        self._total       = 0
        # One-vs-rest score histograms, ascending bin order. Fixed size, so the
        # tracker's footprint does not grow with the number of samples seen.
        self._pos_hist    = np.zeros((self.num_classes, self._bins), dtype=np.int64)
        self._neg_hist    = np.zeros((self.num_classes, self._bins), dtype=np.int64)

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

        self._accumulate_scores(probs_np, labels_np)

        for t, p in zip(labels_np, preds_np):
            self._confusion[t, p] += 1

        # Top-k hits
        top_k_preds = logits.topk(self._top_k, dim=1).indices.cpu().numpy()  # (B, k)
        for i, t in enumerate(labels_np):
            if t in top_k_preds[i]:
                self._top_k_hits += 1

    def _accumulate_scores(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Bin a batch's (B, num_classes) probabilities into the score histograms.

        Each column c is a one-vs-rest problem: a sample lands in pos_hist[c]
        when its true label is c and in neg_hist[c] otherwise. Both updates are
        done as a single bincount over a class-offset flat index.
        """
        bins    = self._bins
        n_cls   = self.num_classes
        classes = np.arange(n_cls, dtype=np.int64)

        bin_idx = np.clip((probs * bins).astype(np.int64), 0, bins - 1)  # (B, C)
        flat    = classes * bins + bin_idx                               # (B, C)
        is_pos  = labels[:, None].astype(np.int64) == classes[None, :]   # (B, C)

        size = n_cls * bins
        self._pos_hist += np.bincount(flat[is_pos],
                                      minlength=size).reshape(n_cls, bins)
        self._neg_hist += np.bincount(flat[~is_pos],
                                      minlength=size).reshape(n_cls, bins)

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
        per_class_auc = [
            _roc_auc_from_hist(self._pos_hist[c], self._neg_hist[c])
            for c in range(self.num_classes)
        ]

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

        # ---- per-class probability thresholds at target recall(s) ----
        per_class_thresholds: dict[str, list[dict]] = {}
        for r in self.recall_targets:
            key = f"{r:.2f}"
            per_class_thresholds[key] = [
                _threshold_at_recall_from_hist(self._pos_hist[c],
                                               self._neg_hist[c], r)
                for c in range(self.num_classes)
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
            "per_class_thresholds": per_class_thresholds,
            "confusion_matrix":     C.copy(),
        }

