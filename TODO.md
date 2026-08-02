# Implementation TODO

Work through these phases in order. Each phase produces something runnable before the GUI
is involved, so you can validate the core logic independently.

---

## Phase 1 — Data layer
- [x] `src/dataset.py` — PyTorch `Dataset` wrapping the H5 file; respects `split` and `gt` filters
- [x] `src/dataset.py` — `DataLoader` factory (num_workers, pin_memory, shuffle per split)
- [x] Verify a batch loads correctly and channel dim is right (1 vs 3)

## Phase 2 — Model
- [x] `src/model.py` — define a baseline CNN (or thin wrapper around a torchvision backbone)
- [x] Model accepts configurable `in_channels` (1 or 3) and `num_classes`
- [x] Verify a forward pass runs on a sample batch

## Phase 3 — Training engine (no GUI)
- [x] `src/trainer.py` — `Trainer` class with `train_one_epoch()` and `validate()` methods
- [x] Emit metrics (loss, accuracy) via a callback / signal so the GUI can consume them later
- [x] Support clean stop via a `threading.Event` cancel token
- [x] `src/metrics.py` — track loss, per-class accuracy, confusion matrix per epoch
- [x] `src/checkpoints.py` — save/load checkpoint (model weights, optimizer state, epoch, metrics)

## Phase 4 — Logging
- [x] `src/logger.py` — TensorBoard `SummaryWriter` wrapper; log scalars, confusion matrix image
- [x] Decide log directory convention (`runs/<experiment_name>/<timestamp>`)
- [x] Log hyperparameters with `writer.add_hparams()`

## Phase 5 — PyQt5 app skeleton
- [x] `app/main.py` — entry point; `QApplication` + show main window
- [x] `app/main_window.py` — `QMainWindow` with dockable panels and a central widget placeholder
- [x] Decide layout: left panel = settings, center = training log/plots, right = controls

## Phase 6 — Settings panel
- [x] `app/panels/settings_panel.py` — widgets for:
  - H5 file path (QLineEdit + browse button), in_channels
  - Model backbone selector (QComboBox), pretrained checkbox (disabled for simple_cnn)
  - Learning rate, batch size, epochs, num_workers (QDoubleSpinBox / QSpinBox)
  - Optimizer choice (Adam, AdamW, SGD, RMSprop)
  - Target metric dropdown (all TARGET_METRICS), device combo (auto-detects CUDA GPUs)
  - Checkpoint output directory, TensorBoard log directory, experiment name, TB port
- [x] `get_settings() -> dict` public API for Phase 7
- [x] Serialize/deserialize settings to JSON (app-data folder, auto-save on close)

## Phase 7 — Training controls
- [x] `app/panels/control_panel.py` — Start / Pause / Stop buttons with progress bar + epoch/metric labels
- [x] `TrainingWorker(QThread)` — builds dataset, model, optimizer, scheduler, logger and runs `Trainer.fit()` in background
- [x] Pause via `threading.Event` polled in `on_batch_end`; Stop via cancel token
- [x] Worker emits: `sig_log`, `sig_epoch_end`, `sig_batch_end`, `sig_finished`, `sig_error`, `sig_checkpoint`
- [x] `ControlPanel` bridges worker signals → panel-level `sig_*` signals consumed by `MainWindow`
- [x] Settings auto-saved to JSON on window close (`MainWindow.closeEvent`)

## Phase 8 — Console output panel
- [x] `app/panels/console_panel.py` — read-only `QPlainTextEdit` for training output, monospace font, auto-scroll
- [x] Append a formatted line per epoch: `Epoch 012  train_loss=0.0412  val_loss=0.0398  f1_macro=0.9231  lr=1.00e-03`
- [x] Progress bar for current epoch (batch-level granularity) — in `ControlPanel`

## Phase 9 — Checkpoint management
- [x] `app/panels/checkpoint_panel.py` — lists `epoch_*.pt` files (★ marks best), auto-refreshes after each epoch
- [x] Double-click to inspect epoch, backbone, and all scalar metrics
- [x] "Resume from selected" emits `sig_resume_requested(path)` → logged in console (full resume in Phase 10)
- [x] "Export best…" copies `best.pt` to a user-chosen path via `QFileDialog`
- [x] Manual refresh button (↻)

## Phase 10 — TensorBoard integration
- [x] Port setting already in settings panel (default `6006`)
- [x] `app/panels/tensorboard_panel.py` — ▶ Start TB / ⏹ Stop TB / 🌐 Open in browser buttons
- [x] Launches `tensorboard --logdir <dir> --port <port>` as a managed `subprocess.Popen`
- [x] Polls every 2 s to detect unexpected exits; updates status label
- [x] `configure(log_dir, port)` called at startup and on each training log message (stays in sync)
- [x] `cleanup()` called from `MainWindow.closeEvent` — terminates subprocess cleanly on app exit

## Phase 11 — Polish
- [x] Status bar: shows epoch + metric + training-started / training-complete state
- [x] Error handling: invalid H5 path, missing resume checkpoint, empty experiment name → `QMessageBox.warning`
- [x] Training exceptions show concise `QMessageBox.critical` (full traceback still in console)
- [x] Logger resource leak fixed (try/finally in worker)
- [x] TensorBoard exe path resolution fixed (`os.path.dirname` instead of brittle string replace)
- [x] TensorBoard stderr captured and shown in status label on unexpected exit
- [x] Best checkpoint marker (★) identified by reading `best.pt` epoch field, not filesize coincidence
- [x] TensorBoard panel configured once per training run, not on every log message
- [x] Console scrollbar reference cached
- [x] Improved tooltips (focal γ); proper type hint on `Trainer.scheduler`
- [x] Entry point already packaged in `pyproject.toml` (`image-classifier`)
- [x] Dark/light theme toggle — `app/theme.py` (QPalette + QSS), View ▸ Theme menu,
      persisted in settings.json (kept out of `get_settings()` so it never reaches
      `add_hparams`)

---

## Phase 12 — UI restructure

- [x] High-DPI opt-in (`AA_EnableHighDpiScaling`, `AA_UseHighDpiPixmaps`,
      `PassThrough` rounding) — must run before `QApplication` exists
- [x] Fusion style, so QSS applies consistently
- [x] Right dock tabbed: Train / Inference / Checkpoints / TensorBoard
      (was ~685px stacked, now ~220px — the tallest single page)
- [x] `InferencePanel` split out of `control_panel.py`
- [x] Settings panel tabbed: Data / Model / Optimizer / Hardware / Output
      (was one 24-row 1025px form, now ≤7 rows per page)
- [x] Per-tab scroll areas so any page can grow without clipping
- [x] Menu bar: File / View (dock toggles + theme) / Help; docks now closable
- [x] Live metric charts in the centre panel (pyqtgraph) — Log / Metrics tabs
      `app/panels/metrics_panel.py`: 2×2 linked plots (loss, accuracy, target
      metric, log-scaled LR), theme-aware, shared crosshair readout. Four separate
      plots rather than twin axes. Degrades to an install hint without pyqtgraph.

---

## Phase 13 — GPU data augmentation

- [x] Spike: torchvision v2 randomises PER BATCH on batched input (one draw for
      every image) — unusable here. kornia is per-sample but raises on
      `.to(device)`. Hand-rolled batched ops are per-sample and ~35x cheaper
      than looping v2 per image. See `test/bench_augment.py`. No new dependency.
- [x] `src/augment.py` — `GpuAugment` (train only) + `Normalizer` (everywhere),
      9 per-sample ops, 16 flat scalar config keys, 4 presets
- [x] `compute_dataset_stats()` — streamed mean/std, agrees with the published
      CIFAR-100 values to ~3 decimal places
- [x] Trainer: `_prepare_batch(images, training)` holds the one asymmetry —
      augment on train only, normalise on both. Runs outside autocast.
- [x] Normalisation stats written into the checkpoint; `InferenceWorker` replays
      them via `Normalizer.from_checkpoint`. Pre-existing checkpoints -> identity.
- [x] `log_hparams` renders list-valued stats instead of silently dropping them
- [x] Settings: Augment tab (preset + 15 controls), normalisation on the Data
      tab, random seed on Optimizer
- [x] `app/panels/preview_panel.py` — before/after thumbnails as a centre tab
- [ ] MixUp / CutMix — deferred: soft targets break `MetricTracker`'s confusion
      matrix and `FocalLoss`'s `F.nll_loss`

---

## Phase 14 — Operating points (thresholds) for downstream inference

- [x] Checkpoints store `classes`, so every per-class array in the payload
      (thresholds, specificity, support) is self-describing instead of an
      anonymous index-ordered list
- [x] `InferenceWorker` replays the checkpoint's own `recall_targets` /
      `specificity_targets` — it previously built `MetricTracker(num_classes)`
      with none, so the test split computed no thresholds at all
- [x] Fixed `_metrics_to_jsonable` recursing with itself, which injected a
      `class_names` key into every nested dict and left `per_class_thresholds`
      holding the class list instead of its tables. Now delegates to
      `src.metrics.to_jsonable`, which also maps NaN -> null (a bare `NaN`
      token is not valid JSON, and absent classes produce them routinely).
- [x] `_threshold_at_specificity_from_hist()` — the mirror of the recall
      version, off the same histograms. Recall entries now also report the
      specificity you get there, and vice versa.
- [x] `per_class_support` exposed — a threshold read off 3 samples is not the
      same as one read off 3000, and the confusion matrix it could be derived
      from is stripped from checkpoints
- [x] `build_threshold_table()` / `write_threshold_table()` + "Export
      thresholds…" on the Checkpoints tab — one row per (class, criterion,
      target) as CSV or JSON, the artifact to ship with a model
- [x] Shared `parse_target_list()` / `to_jsonable()` in `src/metrics.py`,
      replacing three near-duplicate copies

---

## Phase 15 — Pretrained weights on offline machines (Models menu)

- [x] `src/model.py` — `download_pretrained_weights()` fetches a backbone's
      ImageNet weights into a chosen folder (torch.hub download: temp file +
      sha256 check, so an interrupted transfer never leaves a plausible file);
      `build_model(..., weights_dir=...)` loads from that folder instead of the
      network when the file is present
- [x] Models menu — Download Weights (current / all backbones) on a QThread
      worker, Set Weights Folder (verifies the copy, lists what it found),
      Clear. Folder persisted in settings; shown read-only on the Model page
- [x] Training log states the weight source (local file vs download) before
      the model is built
- [x] `InferenceWorker` no longer honours the checkpoint's `pretrained` flag —
      the state dict overwrites every weight anyway, and the flag only bought
      an ImageNet download that fails offline
- [x] Fixed `mobilenet_v3_small` head config: `in_features` was read from
      `classifier[0]` (576) but the replacement swaps the last linear, which
      takes 1024 — every forward pass crashed. Found by the first real
      end-to-end test; all six backbones now forward-pass tested at 1 and 3
      channels

---

## Suggested file structure

```
image_classifier/
  src/
    dataset.py
    model.py
    trainer.py
    metrics.py
    checkpoints.py
    logger.py
  app/
    main.py
    main_window.py
    panels/
      settings_panel.py
      control_panel.py
      metrics_panel.py
      checkpoint_panel.py
  test/
    download_and_prep_mnist.py
  docs/
    h5_format.md
  TODO.md
```
