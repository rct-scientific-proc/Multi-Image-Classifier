"""
Dataset — wraps the H5 file and exposes PyTorch Dataset / DataLoader.

Split values: 0 = train, 1 = validate, 2 = test
gt flag:      True = genuine sample, False = hard negative

Usage:
    from src.dataset import H5Dataset, make_dataloader

    train_ds = H5Dataset("dataset.h5", split=0)
    train_dl = make_dataloader(train_ds, batch_size=32, shuffle=True)

    for images, labels, gt in train_dl:
        # images: float32 (B, C, H, W)  — normalised to [0, 1]
        # labels: int64   (B,)
        # gt:     bool    (B,)
        ...
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


SPLIT_TRAIN    = 0
SPLIT_VALIDATE = 1
SPLIT_TEST     = 2


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
            self.classes = list(f["classes"].asstr()[:])

            mask = np.ones(len(splits), dtype=bool)
            if split is not None:
                mask &= splits == split
            if not include_hard_negatives:
                mask &= gt

            # Store indices so we only read what we need from the h5 file
            self.indices = np.where(mask)[0].astype(np.int64)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def _get_file(self) -> h5py.File:
        """Return the open HDF5 file handle, opening it lazily.

        Each DataLoader worker calls this on its first access, giving one
        file handle per worker rather than one per sample.
        """
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx: int):
        h5_idx = int(self.indices[idx])
        f      = self._get_file()

        image = f["images"][h5_idx]          # uint8 (H, W, C)
        label = int(f["labels"][h5_idx])
        gt    = bool(f["gt"][h5_idx])

        # HDF5 stores (H, W, C); PyTorch expects (C, H, W)
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        image = image.permute(2, 0, 1)       # (C, H, W)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, gt

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: H5Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Return a DataLoader with sensible defaults.

    num_workers > 0 opens one HDF5 file handle per worker process.
    pin_memory speeds up CPU→GPU transfers when using CUDA.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
