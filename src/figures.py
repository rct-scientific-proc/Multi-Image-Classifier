"""
Static figures for an inference run — the report you keep, not the live view.

matplotlib rather than pyqtgraph: these are exported to PNG/PDF/SVG and read
later, often by someone who was not at the machine. pyqtgraph owns the live
Metrics tab; this owns the artifact.

Design notes that matter with 61 classes:

  * Curve figures use EMPHASIS, not 61 colours. Every class is drawn in a
    recessive grey, then the few best and worst by AUC are drawn over the top in
    two hues and directly labelled. Cycling hues past ~8 makes series
    indistinguishable under colour-vision deficiency and turns the legend into
    the chart.
  * Two categorical hues only: blue #2a78d6 and orange #eb6834. Validated as a
    pair on a white surface — worst-case CVD ΔE 24.7, normal-vision ΔE 33.6,
    both well clear of the 8 / 15 floors, and both above 3:1 contrast.
  * The confusion matrix is a magnitude, so it gets a single-hue sequential
    ramp, never a rainbow.
  * Nothing is dual-axis. Where two measures share a figure they share a scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.metrics import calibration_from_hist, sweep_from_hist

# ── palette ──────────────────────────────────────────────────────────────────
BLUE   = "#2a78d6"     # categorical slot 1 — "best", and single-series marks
ORANGE = "#eb6834"     # categorical slot 2 — "worst"
CROWD  = "#b8b8b8"     # the other 50-odd classes: context, deliberately quiet
INK    = "#52514e"     # all text and axes — never a series colour
GRID   = "#e6e6e3"
SURFACE = "#ffffff"

_LINE_W = 2.0
_CROWD_W = 0.8
_CROWD_ALPHA = 0.35
_HIGHLIGHT_N = 3       # best and worst singled out per curve figure


def _plt():
    """Import pyplot with a headless backend, or explain why we cannot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:                       # pragma: no cover
        raise RuntimeError(
            "Figure export needs matplotlib:  pip install matplotlib"
        ) from exc


def _style(ax, title: str, xlabel: str, ylabel: str, *, legend: bool = False):
    ax.set_title(title, color=INK, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if legend:
        leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        if leg is not None:
            leg.set_zorder(5)


def _rank_by_auc(per_class_auc, n: int) -> tuple[list[int], list[int]]:
    """(best, worst) class indices by AUC, ignoring classes with no AUC."""
    scored = [(i, v) for i, v in enumerate(per_class_auc)
              if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not scored:
        return [], []
    scored.sort(key=lambda t: t[1])
    worst = [i for i, _ in scored[:n]]
    best = [i for i, _ in reversed(scored[-n:])]
    return best, worst


def _name(names, i: int) -> str:
    return names[i] if i < len(names) else f"class_{i}"


def _curve_figure(plt, hists, names, per_class_auc, xkey, ykey,
                  title, xlabel, ylabel, *, invert_x=False, diagonal=False):
    """Shared body for the three threshold-swept curve figures."""
    pos_h, neg_h = hists
    n_classes = pos_h.shape[0]
    best, worst = _rank_by_auc(per_class_auc, _HIGHLIGHT_N)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    plotted = 0
    for c in range(n_classes):
        if c in best or c in worst:
            continue
        s = sweep_from_hist(pos_h[c], neg_h[c])
        if np.all(np.isnan(s[ykey])) or np.all(np.isnan(s[xkey])):
            continue
        ax.plot(s[xkey], s[ykey], color=CROWD, linewidth=_CROWD_W,
                alpha=_CROWD_ALPHA, zorder=1)
        plotted += 1

    # Highlighted classes are drawn last so they sit above the crowd, and are
    # named in the legend — identity is never left to colour alone.
    for group, colour, tag in ((best, BLUE, "best"), (worst, ORANGE, "worst")):
        for rank, c in enumerate(group):
            s = sweep_from_hist(pos_h[c], neg_h[c])
            auc = per_class_auc[c]
            ax.plot(s[xkey], s[ykey], color=colour, linewidth=_LINE_W,
                    alpha=1.0 - 0.22 * rank, zorder=3,
                    label=f"{tag}: {_name(names, c)} (AUC {auc:.3f})")

    if diagonal:
        ax.plot([0, 1], [0, 1], color=INK, linewidth=1.0, linestyle=":",
                alpha=0.5, zorder=2, label="chance")

    if invert_x:
        ax.set_xlim(1.02, -0.02)
    else:
        ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _style(ax, title, xlabel, ylabel, legend=True)
    ax.text(0.5, -0.13, f"{plotted} further classes shown in grey",
            transform=ax.transAxes, ha="center", color=INK, fontsize=8,
            alpha=0.8)
    fig.tight_layout()
    return fig


# ── individual figures ───────────────────────────────────────────────────────

def fig_recall_specificity(hists, names, metrics, targets=None):
    """Recall against specificity, swept over the decision threshold.

    The operating-point chooser: every point on a curve is a threshold, and the
    curve shows exactly what each class costs you in false alarms to reach a
    given recall. Specificity descends left-to-right so "better" is up-and-left,
    matching the ROC convention readers already carry.
    """
    plt = _plt()
    fig = _curve_figure(
        plt, hists, names, metrics.get("per_class_auc", []),
        xkey="specificity", ykey="recall",
        title="Recall vs specificity trade-off (one-vs-rest, per class)",
        xlabel="Specificity  (TN / (TN + FP))  →  fewer false alarms",
        ylabel="Recall  (TP / (TP + FN))",
        invert_x=True,
    )
    ax = fig.axes[0]
    for t in (targets or []):
        ax.axvline(t, color=INK, linewidth=1.0, linestyle="--", alpha=0.45,
                   zorder=2)
        ax.text(t, 1.015, f"spec {t:g}", color=INK, fontsize=7.5,
                ha="center", va="bottom", alpha=0.8)
    return fig


def fig_pr_curves(hists, names, metrics, targets=None):
    """Precision against recall.

    More honest than ROC at this class balance: with a class at ~1.6% of the
    data, a ROC curve can look excellent while precision is near zero, because
    the false-positive rate is diluted by a huge negative pool.
    """
    plt = _plt()
    fig = _curve_figure(
        plt, hists, names, metrics.get("per_class_auc", []),
        xkey="recall", ykey="precision",
        title="Precision vs recall (one-vs-rest, per class)",
        xlabel="Recall", ylabel="Precision",
    )
    ax = fig.axes[0]
    for t in (targets or []):
        ax.axvline(t, color=INK, linewidth=1.0, linestyle="--", alpha=0.45)
        ax.text(t, 1.015, f"recall {t:g}", color=INK, fontsize=7.5,
                ha="center", va="bottom", alpha=0.8)
    return fig


def fig_roc_curves(hists, names, metrics, targets=None):
    """True-positive against false-positive rate. Familiar, and AUC's basis."""
    plt = _plt()
    return _curve_figure(
        plt, hists, names, metrics.get("per_class_auc", []),
        xkey="fpr", ykey="recall",
        title="ROC (one-vs-rest, per class)",
        xlabel="False positive rate", ylabel="True positive rate",
        diagonal=True,
    )


def fig_confusion_matrix(hists, names, metrics, targets=None, normalize=True):
    """Confusion heatmap. Row-normalised by default.

    Raw counts are unreadable when one class holds 40% of the data — the
    hard_negative row swamps every other. Normalising by row turns each cell
    into "of the true X, what fraction went to Y", which is comparable across
    rows of wildly different size.
    """
    plt = _plt()
    cm = np.asarray(metrics.get("confusion_matrix"))
    if cm is None or cm.ndim != 2:
        raise ValueError("metrics has no confusion_matrix "
                         "(it is stripped from checkpoints; run inference)")
    n = cm.shape[0]
    shown = cm.astype(float)
    if normalize:
        shown = shown / np.maximum(shown.sum(axis=1, keepdims=True), 1)

    side = float(np.clip(0.20 * n + 3.0, 6.0, 18.0))
    # constrained layout, not tight_layout: tight_layout cannot place a
    # colorbar against a square image axes and strands it far to the right.
    fig, ax = plt.subplots(figsize=(side, side * 0.92), facecolor=SURFACE,
                           layout="constrained")
    im = ax.imshow(shown, cmap="Blues", interpolation="nearest",
                   vmin=0, vmax=1 if normalize else None)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.82)
    cbar.set_label("fraction of true class" if normalize else "count",
                   color=INK, fontsize=9)
    cbar.ax.tick_params(colors=INK, labelsize=8)

    # Per-cell numbers stop being legible well before 61 classes.
    if n <= 25:
        thresh = shown.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if shown[i, j] > thresh else INK)

    step = 1 if n <= 40 else max(1, n // 40)
    ticks = list(range(0, n, step))
    labels = [_name(names, i) for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted", color=INK, fontsize=9)
    ax.set_ylabel("True", color=INK, fontsize=9)
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else " (counts)"),
                 color=INK, fontsize=11, pad=10)
    ax.tick_params(colors=INK)
    return fig


def fig_per_class_bars(hists, names, metrics, targets=None, metric_key="per_class_auc"):
    """Per-class score, sorted worst-first, with support annotated.

    One series, so one colour — a darker-where-bigger ramp would double-encode
    bar length as hue and burn the only free channel on information the bar
    already carries. Sorted so the classes that need work are adjacent, and
    annotated with support because a bad score on 15 samples is a different
    problem from a bad score on 600.
    """
    plt = _plt()
    vals = list(metrics.get(metric_key) or [])
    support = list(metrics.get("per_class_support") or [])
    if not vals:
        raise ValueError(f"metrics has no {metric_key}")

    order = sorted(range(len(vals)),
                   key=lambda i: (np.inf if vals[i] is None or np.isnan(vals[i])
                                  else vals[i]))
    y = np.arange(len(order))
    v = [0.0 if vals[i] is None or np.isnan(vals[i]) else vals[i] for i in order]

    height = float(np.clip(0.22 * len(order) + 1.5, 4.0, 24.0))
    fig, ax = plt.subplots(figsize=(8.0, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(y, v, color=BLUE, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{_name(names, i)}" + (f"  (n={support[i]})" if i < len(support) else "")
         for i in order], fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.invert_yaxis()
    label = metric_key.replace("per_class_", "").replace("_", " ")
    label = "AUC" if label == "auc" else label.capitalize()
    _style(ax, f"Per-class {label} — worst first", label, "")
    macro = np.nanmean([x for x in vals if x is not None])
    ax.axvline(macro, color=ORANGE, linewidth=1.5, linestyle="--", zorder=3)
    ax.text(macro, -0.8, f"macro mean {macro:.3f}", color=ORANGE, fontsize=8,
            ha="center", va="bottom")
    fig.tight_layout()
    return fig


def fig_calibration(hists, names, metrics, targets=None):
    """Predicted probability against observed frequency, pooled over classes.

    The figure that says whether a threshold means what it claims. A curve below
    the diagonal is overconfident — a 0.9 score that is right 70% of the time —
    and a threshold chosen on validation will not transfer honestly to new data.
    """
    plt = _plt()
    pos_h, neg_h = hists
    cal = calibration_from_hist(pos_h.sum(axis=0), neg_h.sum(axis=0), n_bins=20)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.5, 6.5), facecolor=SURFACE, sharex=True,
        layout="constrained", gridspec_kw={"height_ratios": [3, 1]})
    ax.set_facecolor(SURFACE); ax2.set_facecolor(SURFACE)

    ax.plot([0, 1], [0, 1], color=INK, linestyle=":", linewidth=1.0, alpha=0.5,
            label="perfectly calibrated")
    ax.plot(cal["predicted"], cal["observed"], color=BLUE, linewidth=_LINE_W,
            marker="o", markersize=5, label="observed")
    ax.set_ylim(-0.02, 1.02)
    _style(ax, "Calibration — does a score of 0.8 mean 80%?", "",
           "Observed positive rate", legend=True)

    # Sample count per bin, so the sparse high-confidence tail is not read as
    # confidently as the dense low end.
    ax2.bar(cal["predicted"], cal["count"], width=0.04, color=CROWD)
    ax2.set_yscale("log")
    _style(ax2, "", "Predicted probability", "samples")
    return fig


# ── export ───────────────────────────────────────────────────────────────────

# name -> (renderer, human label). Order is the order they are written.
FIGURES: dict[str, tuple] = {
    "recall_specificity": (fig_recall_specificity, "Recall vs specificity trade-off"),
    "pr_curves":          (fig_pr_curves,           "Precision-recall curves"),
    "roc_curves":         (fig_roc_curves,          "ROC curves"),
    "confusion_matrix":   (fig_confusion_matrix,    "Confusion matrix"),
    "per_class_bars":     (fig_per_class_bars,      "Per-class AUC bars"),
    "calibration":        (fig_calibration,         "Calibration curve"),
}

DEFAULT_FIGURES = ("recall_specificity", "pr_curves", "confusion_matrix",
                   "per_class_bars")
FORMATS = ("png", "pdf", "svg")


def export_figures(
    out_dir: str | Path,
    hists: tuple[np.ndarray, np.ndarray],
    class_names: list[str],
    metrics: dict,
    which=DEFAULT_FIGURES,
    fmt: str = "png",
    dpi: int = 150,
    prefix: str = "",
    targets: dict | None = None,
    on_progress=None,
) -> list[Path]:
    """Render *which* figures into *out_dir*. Returns the paths written.

    A renderer that fails takes its own figure down, not the whole export — a
    missing confusion matrix should not cost you the curves.
    """
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = targets or {}
    written: list[Path] = []
    errors: list[str] = []

    for i, key in enumerate(which):
        entry = FIGURES.get(key)
        if entry is None:
            errors.append(f"{key}: unknown figure")
            continue
        render, label = entry
        if on_progress is not None:
            on_progress(i + 1, len(which), label)
        try:
            t = targets.get("specificity" if key == "recall_specificity" else "recall")
            fig = render(hists, class_names, metrics, t)
            path = out_dir / f"{prefix}{key}.{fmt}"
            fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")

    if errors and not written:
        raise RuntimeError("; ".join(errors))
    if errors:
        (out_dir / f"{prefix}figures_errors.txt").write_text(
            "\n".join(errors), encoding="utf-8")
    return written
