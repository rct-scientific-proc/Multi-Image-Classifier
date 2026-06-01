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
- [ ] Status bar: current epoch, ETA, last checkpoint saved
- [ ] Error handling: invalid H5 path, CUDA OOM, missing deps — show `QMessageBox`
- [ ] Dark/light theme toggle
- [ ] Package entry point in `pyproject.toml` / `setup.cfg`

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
