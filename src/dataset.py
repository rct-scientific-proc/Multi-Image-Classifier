"""
Dataset — wraps the H5 file and exposes PyTorch Dataset / DataLoader.

Split values: 0 = train, 1 = validate, 2 = test
gt flag:      True = genuine sample, False = hard negative

Usage:
    from src.dataset import H5Dataset, make_dataloader

    train_ds = H5Dataset("dataset.h5", split=0)
    train_dl = make_dataloader(train_ds, batch_size=32, shuffle=True)

    for images, labels, gt in train_dl:
        # images: uint8 (B, C, H, W) when no transform is set — convert on GPU
        #         float32 (B, C, H, W) normalised to [0, 1] when a transform is set
        # labels: int64   (B,)
        # gt:     bool    (B,)
        ...
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms


SPLIT_TRAIN    = 0
SPLIT_VALIDATE = 1
SPLIT_TEST     = 2


def peek_h5_meta(h5_path: str) -> dict:
    """Read an H5 file's shape/class metadata without loading any image data.

    Cheap enough to call from GUI validation. Returns keys: num_samples,
    height, width, channels, num_classes, classes.
    Raises OSError if the file cannot be opened, KeyError if a required
    dataset is missing.
    """
    with h5py.File(h5_path, "r") as f:
        shape   = f["images"].shape          # (N, H, W, C)
        classes = list(f["classes"].asstr()[:])

    return {
        "num_samples": int(shape[0]),
        "height":      int(shape[1]),
        "width":       int(shape[2]),
        # A trailing channel axis is optional; its absence means grayscale.
        "channels":    int(shape[3]) if len(shape) > 3 else 1,
        "num_classes": len(classes),
        "classes":     classes,
    }


def count_labels(h5_path: str, split: int | None, num_classes: int) -> np.ndarray:
    """Ground-truth sample count per class for *split* (None = all splits).

    Reads only the tiny ``labels`` / ``split`` arrays, so it is cheap enough to
    call from the GUI when previewing class weights. Hard negatives count under
    their assigned label like any other sample.
    """
    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][:]
        if split is not None:
            labels = labels[f["split"][:] == split]
    return np.bincount(labels.astype(np.int64), minlength=num_classes)


class H5Dataset(Dataset):
    """PyTorch Dataset backed by a single HDF5 file.

    Parameters
    ----------
    h5_path:
        Path to the .h5 file.
    split:
        0 = train, 1 = validate, 2 = test.  Pass None to load all samples.
    include_hard_negatives:
        When False (default) hard negatives (gt == False) are excluded.
        Set to True to include them during training.
    transform:
        Optional torchvision transform applied to each image tensor after
        it has been converted to float32 and normalised to [0, 1].
    """

    def __init__(
        self,
        h5_path: str,
        split: int | None = SPLIT_TRAIN,
        include_hard_negatives: bool = True,
        transform=None,
    ):
        self.h5_path   = h5_path
        self.transform = transform
        self._file     = None  # opened lazily per worker

        with h5py.File(h5_path, "r") as f:
            splits = f["split"][:]   # (N,) uint8
            gt     = f["gt"][:]      # (N,) bool
            labels = f["labels"][:]  # (N,) int
            self.classes = list(f["classes"].asstr()[:])

            mask = np.ones(len(splits), dtype=bool)
            if split is not None:
                mask &= splits == split
            if not include_hard_negatives:
                mask &= gt

            # Store indices so we only read what we need from the h5 file
            self.indices = np.where(mask)[0].astype(np.int64)
            # Labels aligned with self.indices — the class-cap sampler needs
            # them, and they are a few KB against the images' GBs.
            self.labels = labels[self.indices].astype(np.int64)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def _get_file(self) -> h5py.File:
        """Return the open HDF5 file handle, opening it lazily.

        Each DataLoader worker calls this on its first access, giving one
        file handle per worker rather than one per sample.
        """
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")  # 64 MiB chunk cache
        return self._file

    def __getitem__(self, idx: int):
        h5_idx = int(self.indices[idx])
        f      = self._get_file()

        image = f["images"][h5_idx]          # uint8 (H, W, C)
        label = int(f["labels"][h5_idx])
        gt    = bool(f["gt"][h5_idx])

        if self.transform is None:
            # Fast path: keep as uint8 and let the trainer convert + scale on
            # the GPU in one fused op per batch. 4× less PCIe traffic, no
            # per-sample float work on the CPU.
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        else:
            # Transforms expect float in [0, 1]; do the conversion here.
            image = torch.from_numpy(image.astype(np.float32) * (1.0 / 255.0))
            image = image.permute(2, 0, 1)       # (C, H, W)
            image = self.transform(image)

        return image, label, gt

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Class-imbalance caps
# ---------------------------------------------------------------------------

def class_caps(counts: np.ndarray, max_class_ratio: float) -> np.ndarray:
    """Per-class per-epoch sample budgets. ``inf`` means uncapped.

    Each class's cap is ``ratio × median(counts of the OTHER non-empty
    classes)`` — a class does not get to set its own budget. A plain global
    median fails exactly where capping matters most: with one giant class and
    one small one (120k negatives vs 1k positives), the median sits at ~60k
    and the cap never triggers. Referencing each class against the others
    makes the two-class case work and stays robust to a single tiny class in
    the multi-class case.

    ``max_class_ratio <= 0`` disables capping (all-inf).
    """
    counts = np.asarray(counts, dtype=np.int64)
    caps = np.full(len(counts), np.inf)
    if not max_class_ratio or max_class_ratio <= 0:
        return caps
    nonempty = counts > 0
    if int(nonempty.sum()) < 2:          # nothing to be imbalanced against
        return caps
    for c in range(len(counts)):
        others = counts[nonempty & (np.arange(len(counts)) != c)]
        if len(others):
            caps[c] = float(max_class_ratio) * float(np.median(others))
    return caps


def capped_counts(counts: np.ndarray, max_class_ratio: float) -> np.ndarray:
    """The per-epoch sample count each class actually contributes under the cap.

    This is what class weighting should see: weighting from the raw counts
    while sampling from the capped ones would penalise the majority class
    twice. A finite cap is floored at 1 so no class can vanish entirely.
    """
    counts = np.asarray(counts, dtype=np.int64)
    caps = class_caps(counts, max_class_ratio)
    out = counts.copy()
    for c, cap in enumerate(caps):
        if np.isfinite(cap):
            out[c] = min(out[c], max(1, int(round(cap))))
    return out


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class PeriodicShuffleSampler(Sampler):
    """Shuffle training indices every *shuffle_every* epochs, optionally
    capping how many samples each class contributes per epoch.

    Parameters
    ----------
    dataset_size:
        Total number of samples.
    shuffle_every:
        Re-shuffle the index permutation once every this many epochs.
        ``1`` (default) reshuffles before every epoch.
        ``0`` disables shuffling entirely (sequential order).
    labels:
        Per-sample class labels aligned with dataset indices. Only needed
        when *max_class_ratio* is set.
    max_class_ratio:
        See :func:`class_caps`. When a class exceeds its cap, a fresh random
        subset of it is drawn at every reshuffle — over a long run the model
        still sees most of the big class, just not all of it per epoch.
        ``0`` (default) disables capping.
    """

    def __init__(self, dataset_size: int, shuffle_every: int = 1,
                 labels: np.ndarray | None = None,
                 max_class_ratio: float = 0.0):
        self._n     = dataset_size
        self._every = max(0, int(shuffle_every))
        self._epoch = 0

        # Per-class index pools, only for classes that actually exceed their
        # budget — everything else rides along uncapped.
        self._pools: list[tuple[np.ndarray, int]] = []
        self._uncapped: np.ndarray | None = None
        self._cap_info: list[tuple[int, int, int]] = []   # (class, total, cap)
        if labels is not None and max_class_ratio and max_class_ratio > 0:
            labels = np.asarray(labels)
            counts = np.bincount(labels, minlength=int(labels.max()) + 1 if len(labels) else 0)
            caps   = class_caps(counts, max_class_ratio)
            free   = np.ones(len(labels), dtype=bool)
            for c, cap in enumerate(caps):
                take = max(1, int(round(cap))) if np.isfinite(cap) else None
                if take is not None and counts[c] > take:
                    pool = np.where(labels == c)[0]
                    free[pool] = False
                    self._pools.append((pool, take))
                    self._cap_info.append((c, int(counts[c]), take))
            self._uncapped = np.where(free)[0]

        self._indices: list[int] = []
        self._resample()

    # ------------------------------------------------------------------
    @property
    def cap_summary(self) -> list[tuple[int, int, int]]:
        """(class_index, total_samples, per_epoch_cap) for each capped class."""
        return list(self._cap_info)

    def _resample(self) -> None:
        """Draw this cycle's index list: subset capped classes, then shuffle."""
        if not self._pools:
            idx = np.arange(self._n)
        else:
            parts = [self._uncapped]
            for pool, take in self._pools:
                # torch RNG, like _reshuffle always used, so the existing
                # seeding story covers the subset draw too.
                pick = torch.randperm(len(pool))[:take].numpy()
                parts.append(pool[pick])
            idx = np.concatenate(parts)

        if self._every > 0:
            idx = idx[torch.randperm(len(idx)).numpy()]
        else:
            idx = np.sort(idx)          # shuffle disabled: keep storage order
        self._indices = idx.tolist()

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch; resamples when ``epoch % shuffle_every == 0``."""
        self._epoch = epoch
        if self._every > 0 and epoch % self._every == 0:
            self._resample()

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: H5Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_every: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_class_ratio: float = 0.0,
) -> DataLoader:
    """Return a DataLoader with sensible defaults.

    num_workers > 0 opens one HDF5 file handle per worker process.
    pin_memory speeds up CPU→GPU transfers when using CUDA.

    When *shuffle* is True, *shuffle_every* controls how often the index
    permutation is refreshed: 1 = every epoch (default), N = every N epochs,
    0 = never shuffle (sequential order regardless of *shuffle*).

    *max_class_ratio* > 0 caps each class's per-epoch contribution (see
    :func:`class_caps`), redrawing the capped subset at every reshuffle.
    Deliberately honoured only when *shuffle* is True — i.e. the train loader.
    Validation and test must see the real distribution, or every reported
    metric and threshold is computed against a dataset that does not exist.
    """
    capping = bool(shuffle) and max_class_ratio and max_class_ratio > 0
    if capping or (shuffle and shuffle_every != 1):
        sampler = PeriodicShuffleSampler(
            len(dataset), shuffle_every,
            labels=dataset.labels if capping else None,
            max_class_ratio=max_class_ratio if capping else 0.0,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
