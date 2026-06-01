# HDF5 Dataset Format

All datasets used for training, validation, and testing are stored in a single `.h5` file.
Every array shares the same first axis length `N` (total number of samples), giving a 1-to-1
index mapping across all datasets.

---

## Datasets

| Name      | dtype   | Shape        | Description |
|-----------|---------|--------------|-------------|
| `images`  | uint8   | (N, H, W, C) | Pixel values. `C=1` for grayscale, `C=3` for RGB. |
| `labels`  | uint16  | (N,)         | Integer class index. Look up the name via `classes[labels[i]]`. |
| `gt`      | bool    | (N,)         | `True` = genuine example. `False` = hard negative. |
| `split`   | uint8   | (N,)         | `0` = train, `1` = validate, `2` = test. |
| `classes` | str     | (K,)         | Ordered list of class name strings. Last entry is always `"hard_negative"`. K = number unique labels uint16 values |

---

## Split Values

| Value | Meaning  |
|-------|----------|
| `0`   | Train    |
| `1`   | Validate |
| `2`   | Test     |

---

## Ground Truth Flag (`gt`)

`gt` separates *what class something is* (`labels`) from *whether it is a real example of that class* (`gt`).

| `gt`    | Meaning |
|---------|---------|
| `True`  | Genuine labelled example |
| `False` | Hard negative — looks like a class but is not one |

Hard negatives are assigned to the `"hard_negative"` class (last index in `classes`) and are
distributed across train/val/test splits proportionally to the genuine sample counts.

---

## Class Labels

`labels[i]` is a `uint16` index into the `classes` array:

```python
with h5py.File("dataset.h5", "r") as f:
    classes = f["classes"].asstr()[:]   # numpy array of strings
    label   = int(f["labels"][i])
    name    = classes[label]            # e.g. "3" or "hard_negative"
```

---

## Reading a Split

```python
import h5py
import numpy as np

with h5py.File("dataset.h5", "r") as f:
    split  = f["split"][:]
    images = f["images"][split == 0]   # all train images
    labels = f["labels"][split == 0]
    gt     = f["gt"][split == 0]
```

---

## Storage Notes

- `images` is gzip-compressed and chunked on the first axis (e.g. 256 samples per chunk)
  for efficient sequential reads during batched training.
- Boolean and uint8 arrays are stored uncompressed; their size is negligible.
- The file is self-contained — no external label files or directory structure required.
