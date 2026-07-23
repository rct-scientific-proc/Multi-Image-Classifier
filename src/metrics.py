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
    per_class_support     — list[int], ground-truth sample count per class
    per_class_thresholds  — dict[str, list[dict]]: keyed by target recall (e.g. "0.99"),
                            each entry per class has
                            {threshold, precision, recall, specificity}.
                            Populated only when recall_targets is provided.
    per_class_thresholds_specificity
                          — same shape, keyed by target specificity. Populated
                            only when specificity_targets is provided.
    confusion_matrix      — np.ndarray (num_classes, num_classes)

Thresholds are operating points, and are only meaningful when chosen on data the
model was not fitted to — the Trainer computes them on the validation split.
Applying them to fresh data is the intended use; re-deriving them on the same
data you then report precision for is self-fulfilling.

build_threshold_table() / write_threshold_table() flatten them into the CSV or
JSON you would ship alongside a model.

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

import csv
import json
import math
from pathlib import Path

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
    n_neg     = float(neg_hist.sum())
    threshold = float(bins - 1 - k) / float(bins)
    precision = tp / max(tp + fp, 1.0)
    recall    = tp / float(n_pos)
    spec      = 1.0 - (fp / n_neg) if n_neg > 0 else float("nan")
    return {"threshold": threshold, "precision": precision,
            "recall": recall, "specificity": spec}


def _threshold_at_specificity_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray,
                                        target_specificity: float) -> dict:
    """Lowest probability threshold whose one-vs-rest specificity ≥ target.

    The mirror of _threshold_at_recall_from_hist. Specificity is
    ``TN / (TN + FP)`` and ``TN + FP`` is just the negative count, so a
    specificity floor is a false-positive budget: admit as many samples as
    possible while keeping ``cum_fp`` within it. Taking the *largest* such set
    (lowest threshold) maximises recall at the requested specificity.

    Returns {threshold, precision, recall, specificity}. All NaN when the class
    has no negatives; a target no operating point can meet yields a threshold of
    1.0, i.e. admit nothing.
    """
    n_pos = float(pos_hist.sum())
    n_neg = float(neg_hist.sum())
    if n_neg == 0:
        return {"threshold": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "specificity": float("nan")}

    bins   = len(pos_hist)
    cum_tp = np.cumsum(pos_hist[::-1])
    cum_fp = np.cumsum(neg_hist[::-1])

    # How many false positives the specificity floor permits.
    allowed_fp = np.floor(n_neg * (1.0 - float(target_specificity)))
    k = int(np.searchsorted(cum_fp, allowed_fp, side="right")) - 1

    if k < 0:
        # Even the top-scoring bin blows the budget: predict nothing positive.
        return {"threshold": 1.0, "precision": float("nan"),
                "recall": 0.0, "specificity": 1.0}

    tp, fp = float(cum_tp[k]), float(cum_fp[k])
    return {
        "threshold":   float(bins - 1 - k) / float(bins),
        "precision":   tp / max(tp + fp, 1.0),
        "recall":      tp / n_pos if n_pos > 0 else float("nan"),
        "specificity": 1.0 - (fp / n_neg),
    }


class MetricTracker:
    """Accumulates per-batch stats and computes epoch-level metrics."""

    def __init__(self, num_classes: int, recall_targets: list[float] | None = None,
                 score_bins: int = SCORE_BINS,
                 specificity_targets: list[float] | None = None):
        self.num_classes    = num_classes
        self._top_k         = min(5, num_classes)
        self._bins          = max(2, int(score_bins))
        self.recall_targets = sorted({float(r) for r in (recall_targets or [])
                                      if 0.0 < float(r) <= 1.0})
        self.specificity_targets = sorted({float(s) for s in (specificity_targets or [])
                                           if 0.0 < float(s) <= 1.0})
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

    def score_histograms(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-class (positive, negative) score histograms, ascending-score bins.

        These are the raw material every threshold-swept curve is drawn from.
        Exposed as copies so a caller can hold them after the tracker is gone —
        the alternative, retaining every probability vector, is what the
        histogram design exists to avoid.
        """
        return self._pos_hist.copy(), self._neg_hist.copy()

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

        # ---- and at target specificity, from the same histograms ----
        per_class_thresholds_spec: dict[str, list[dict]] = {}
        for sp in self.specificity_targets:
            key = f"{sp:.2f}"
            per_class_thresholds_spec[key] = [
                _threshold_at_specificity_from_hist(self._pos_hist[c],
                                                    self._neg_hist[c], sp)
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
            # Ground-truth count per class. A threshold read off 3 samples means
            # something very different from one read off 3000, and the confusion
            # matrix it could otherwise be derived from is stripped from
            # checkpoints, so it is surfaced explicitly.
            "per_class_support":    [int(v) for v in support],
            "per_class_thresholds": per_class_thresholds,
            "per_class_thresholds_specificity": per_class_thresholds_spec,
            "confusion_matrix":     C.copy(),
        }



# ---------------------------------------------------------------------------
# Shared helpers — parsing, JSON coercion, threshold tables
# ---------------------------------------------------------------------------

def parse_target_list(text: str) -> list[float]:
    """Parse "0.95, 0.99" into [0.95, 0.99], dropping anything outside (0, 1].

    Shared by the recall-target and specificity-target fields, and by the
    inference worker when it replays a checkpoint's targets.
    """
    out: list[float] = []
    for tok in (text or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 < v <= 1.0:
            out.append(v)
    return sorted(set(out))


def to_jsonable(obj):
    """Recursively convert numpy / torch / non-finite values to JSON-safe ones.

    NaN is mapped to None rather than left alone: json.dump would emit a bare
    ``NaN`` token, which is not valid JSON and is rejected by most parsers.
    Absent classes legitimately produce NaN thresholds, so this is the common
    case, not an edge one.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# Column order for the exported table — class identity first, then the
# operating point, then what it buys you.
THRESHOLD_TABLE_COLUMNS = (
    "class_index", "class_name", "support",
    "criterion", "target", "threshold",
    "precision", "recall", "specificity",
)


def build_threshold_table(metrics: dict, class_names: list[str] | None = None) -> list[dict]:
    """Flatten per-class thresholds into one row per (class, criterion, target).

    *metrics* is a MetricTracker.compute() result — or the ``metrics`` dict from
    a checkpoint, which carries the same keys. Names are optional; without them
    the rows fall back to ``class_<i>`` so the table is still usable.

    Rows are emitted for both criteria, so a reader can compare "the cutoff that
    gets me 99% recall" against "the cutoff that gets me 99% specificity" for
    the same class side by side.
    """
    names = list(class_names or [])
    support = metrics.get("per_class_support") or []
    rows: list[dict] = []

    for criterion, key in (("recall", "per_class_thresholds"),
                           ("specificity", "per_class_thresholds_specificity")):
        table = metrics.get(key) or {}
        for target in sorted(table, key=lambda t: float(t)):
            for idx, entry in enumerate(table[target]):
                rows.append({
                    "class_index": idx,
                    "class_name":  names[idx] if idx < len(names) else f"class_{idx}",
                    "support":     int(support[idx]) if idx < len(support) else None,
                    "criterion":   criterion,
                    "target":      float(target),
                    "threshold":   entry.get("threshold"),
                    "precision":   entry.get("precision"),
                    "recall":      entry.get("recall"),
                    "specificity": entry.get("specificity"),
                })

    rows.sort(key=lambda r: (r["class_index"], r["criterion"], r["target"]))
    return [to_jsonable(r) for r in rows]


def write_threshold_table(
    path: str | Path,
    rows: list[dict],
    meta: dict | None = None,
) -> Path:
    """Write *rows* as CSV or JSON, chosen by the file extension.

    CSV is the default because this table is meant to be read by a human
    deciding operating points, and opened in a spreadsheet. JSON keeps the
    *meta* block (source checkpoint, epoch, split) that CSV has nowhere to put.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        payload = {"meta": to_jsonable(meta or {}), "thresholds": rows}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
        return path

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(THRESHOLD_TABLE_COLUMNS))
        writer.writeheader()
        for r in rows:
            # None -> "" so a missing value reads as blank rather than "None".
            writer.writerow({k: ("" if r.get(k) is None else r.get(k))
                             for k in THRESHOLD_TABLE_COLUMNS})
    return path


# ---------------------------------------------------------------------------
# Threshold sweeps — the curves every operating-point figure is drawn from
# ---------------------------------------------------------------------------

def sweep_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> dict:
    """Every one-vs-rest rate as a function of threshold, in one pass.

    ROC, precision-recall and the recall/specificity trade-off are three views
    of this same sweep, so they are computed once rather than three times.
    Arrays run from the highest threshold (admit nothing) to the lowest.
    """
    bins   = len(pos_hist)
    n_pos  = float(pos_hist.sum())
    n_neg  = float(neg_hist.sum())
    cum_tp = np.cumsum(pos_hist[::-1]).astype(float)
    cum_fp = np.cumsum(neg_hist[::-1]).astype(float)

    nan = np.full(bins, np.nan)
    recall = cum_tp / n_pos if n_pos > 0 else nan
    fpr    = cum_fp / n_neg if n_neg > 0 else nan
    return {
        "threshold":   (bins - 1 - np.arange(bins)) / float(bins),
        "recall":      recall,
        "specificity": 1.0 - fpr,
        "fpr":         fpr,
        "precision":   cum_tp / np.maximum(cum_tp + cum_fp, 1.0),
        "n_pos":       n_pos,
        "n_neg":       n_neg,
    }


def calibration_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray,
                          n_bins: int = 20) -> dict:
    """Observed positive rate vs predicted probability, coarsely binned.

    Answers "when this model says 0.8, is it right 80% of the time?" — which is
    what decides whether a threshold chosen on validation still means the same
    thing on new data. Bins with no samples are dropped rather than plotted as
    zero, and each bin's weight is returned so a reader can discount the sparse
    high-confidence end.
    """
    bins    = len(pos_hist)
    edges   = np.linspace(0, bins, n_bins + 1).astype(int)
    centres = (np.arange(bins) + 0.5) / bins

    pred, obs, count = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        w = pos_hist[a:b] + neg_hist[a:b]
        total = float(w.sum())
        if total == 0:
            continue
        pred.append(float((centres[a:b] * w).sum() / total))
        obs.append(float(pos_hist[a:b].sum() / total))
        count.append(int(total))
    return {"predicted": np.array(pred), "observed": np.array(obs),
            "count": np.array(count)}
