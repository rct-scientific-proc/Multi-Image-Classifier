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
- [ ] `src/logger.py` — TensorBoard `SummaryWriter` wrapper; log scalars, confusion matrix image
- [ ] Decide log directory convention (`runs/<experiment_name>/<timestamp>`)
- [ ] Log hyperparameters with `writer.add_hparams()`

## Phase 5 — PyQt5 app skeleton
- [ ] `app/main.py` — entry point; `QApplication` + show main window
- [ ] `app/main_window.py` — `QMainWindow` with dockable panels and a central widget placeholder
- [ ] Decide layout: left panel = settings, center = training log/plots, right = controls

## Phase 6 — Settings panel
- [ ] `app/panels/settings_panel.py` — widgets for:
  - H5 file path (QLineEdit + browse button)
  - Model backbone selector (QComboBox)
  - Learning rate, batch size, epochs (QDoubleSpinBox / QSpinBox)
  - Optimizer choice (Adam, SGD, etc.)
  - Checkpoint output directory
  - TensorBoard log directory
- [ ] Serialize/deserialize settings to JSON so they persist between sessions

## Phase 7 — Training controls
- [ ] `app/panels/control_panel.py` — Start / Pause / Stop buttons
- [ ] Run `Trainer` in a `QThread` worker so the GUI stays responsive
- [ ] Worker emits Qt signals: `epoch_complete(dict)`, `batch_complete(dict)`, `finished()`, `error(str)`

## Phase 8 — Console output panel
- [ ] `app/panels/console_panel.py` — read-only `QPlainTextEdit` for training output
- [ ] Append a formatted line per epoch: `Epoch 012 | Train loss 0.0412 | Val loss 0.0398 | Val acc 92.31%`
- [ ] Progress bar for current epoch (batch-level granularity)

## Phase 9 — Checkpoint management
- [ ] `app/panels/checkpoint_panel.py` — list saved checkpoints, allow resume from selected one
- [ ] "Export best model" button (copies best val-accuracy checkpoint to a chosen path)

## Phase 10 — TensorBoard integration
- [ ] Add a configurable port setting (default `6006`) to the settings panel
- [ ] Launch `tensorboard --logdir <dir> --port <port>` as a managed subprocess from the GUI
- [ ] "Open TensorBoard" button calls `webbrowser.open(f"http://localhost:{port}")`)
- [ ] Kill the subprocess cleanly on app exit

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
