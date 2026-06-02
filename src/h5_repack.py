"""
H5 inspection + repack utility.

Inspect an existing dataset H5 file and optionally rewrite it with chunking
optimised for shuffled random-access training (one sample per chunk).

Usage (CLI):
    python -m src.h5_repack inspect path/to/dataset.h5
    python -m src.h5_repack repack  path/to/dataset.h5 path/to/output.h5
    python -m src.h5_repack repack  path/to/dataset.h5 path/to/output.h5 \\
        --compression lz4 --chunk-rows 1

Programmatic:
    from src.h5_repack import inspect_h5, repack_h5
    info = inspect_h5("dataset.h5")
    repack_h5("dataset.h5", "dataset_repacked.h5", chunk_rows=1, compression="gzip")
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np


# Datasets we expect in a training H5 (see docs/h5_format.md).
_EXPECTED = ("images", "labels", "gt", "split", "classes")


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def inspect_h5(path: str | Path) -> dict:
    """Return a dict describing the H5 file's datasets and chunking.

    Keys returned per dataset: shape, dtype, chunks, compression, size_bytes,
    chunk_size_bytes (None if unchunked).
    Top-level keys: path, file_size_bytes, datasets (dict), warnings (list).
    """
    path = Path(path)
    info: dict = {
        "path":            str(path),
        "file_size_bytes": path.stat().st_size,
        "datasets":        {},
        "warnings":        [],
    }
    with h5py.File(path, "r") as f:
        for name in _EXPECTED:
            if name not in f:
                info["warnings"].append(f"missing dataset: {name}")
                continue
            d = f[name]
            chunk_size = None
            if d.chunks is not None:
                chunk_size = int(np.prod(d.chunks)) * d.dtype.itemsize
            info["datasets"][name] = {
                "shape":            tuple(d.shape),
                "dtype":            str(d.dtype),
                "chunks":           tuple(d.chunks) if d.chunks else None,
                "compression":      d.compression,
                "compression_opts": d.compression_opts,
                "size_bytes":       int(np.prod(d.shape)) * d.dtype.itemsize,
                "chunk_size_bytes": chunk_size,
            }

        # Heuristic warnings about the "images" dataset
        img = info["datasets"].get("images")
        if img is not None:
            chunks = img["chunks"]
            if chunks is None:
                info["warnings"].append(
                    "images is unchunked — random access will read the whole "
                    "dataset block. Repack with chunks=(1, H, W, C)."
                )
            elif len(chunks) >= 1 and chunks[0] > 1:
                info["warnings"].append(
                    f"images chunks={chunks} bundles {chunks[0]} samples per "
                    f"chunk. For shuffled training prefer (1, H, W, C) so each "
                    f"random read loads only one sample."
                )
    return info


def format_inspection(info: dict) -> str:
    """Return a human-readable string of the inspection result."""
    lines = []
    lines.append(f"File:   {info['path']}")
    lines.append(f"Size:   {_human_bytes(info['file_size_bytes'])}")
    lines.append("")
    for name, d in info["datasets"].items():
        lines.append(f"  [{name}]")
        lines.append(f"    shape       : {d['shape']}")
        lines.append(f"    dtype       : {d['dtype']}")
        lines.append(f"    chunks      : {d['chunks']}")
        lines.append(
            f"    compression : {d['compression']}"
            f"{'' if d['compression_opts'] is None else f' (opts={d['compression_opts']})'}"
        )
        lines.append(f"    size        : {_human_bytes(d['size_bytes'])}")
        if d["chunk_size_bytes"] is not None:
            lines.append(
                f"    chunk size  : {_human_bytes(d['chunk_size_bytes'])}"
            )
        lines.append("")
    if info["warnings"]:
        lines.append("Warnings:")
        for w in info["warnings"]:
            lines.append(f"  ! {w}")
    else:
        lines.append("No warnings — layout looks good for random-access training.")
    return "\n".join(lines)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Repack
# ---------------------------------------------------------------------------

_COMPRESSION_CHOICES = ("none", "gzip", "lzf", "lz4")


def repack_h5(
    src: str | Path,
    dst: str | Path,
    chunk_rows: int = 1,
    compression: str = "gzip",
    compression_opts: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Rewrite *src* into *dst* with sample-aligned chunking on ``images``.

    The other datasets (labels, gt, split, classes) are copied unchanged —
    they're tiny and not the I/O bottleneck.

    Parameters
    ----------
    chunk_rows:
        Samples per chunk for the ``images`` dataset. ``1`` (default) is best
        for shuffled random-access training.
    compression:
        ``"none"``, ``"gzip"``, ``"lzf"``, or ``"lz4"`` (lz4 requires hdf5plugin).
    compression_opts:
        Level for gzip (0-9). Ignored otherwise.
    overwrite:
        If False (default), refuse to overwrite an existing dst.

    Returns a dict with size_before, size_after, elapsed_seconds.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists (pass overwrite=True)")
    if compression not in _COMPRESSION_CHOICES:
        raise ValueError(
            f"compression={compression!r} not in {_COMPRESSION_CHOICES}"
        )

    comp_kwargs: dict = {}
    if compression == "gzip":
        comp_kwargs["compression"]      = "gzip"
        comp_kwargs["compression_opts"] = 4 if compression_opts is None else int(compression_opts)
    elif compression == "lzf":
        comp_kwargs["compression"] = "lzf"
    elif compression == "lz4":
        # hdf5plugin registers filter id 32004 ("lz4") with h5py
        try:
            import hdf5plugin  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "compression='lz4' requires the 'hdf5plugin' package"
            ) from e
        comp_kwargs["compression"] = 32004  # LZ4 filter id

    t0 = time.perf_counter()
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        if "images" not in fin:
            raise KeyError("source file has no 'images' dataset")

        img = fin["images"]
        n   = img.shape[0]
        trailing = tuple(img.shape[1:])  # (H, W, C)
        chunk_rows = max(1, min(int(chunk_rows), n))
        chunk_shape = (chunk_rows, *trailing)

        out_img = fout.create_dataset(
            "images",
            shape=img.shape,
            dtype=img.dtype,
            chunks=chunk_shape,
            **comp_kwargs,
        )

        # Stream-copy in slabs to bound RAM usage
        slab = max(chunk_rows, 1024)
        for start in range(0, n, slab):
            stop = min(start + slab, n)
            out_img[start:stop] = img[start:stop]

        # Copy auxiliary datasets verbatim (tiny — no chunking needed)
        for name in ("labels", "gt", "split", "classes"):
            if name in fin:
                fin.copy(name, fout)

        # Copy file-level attributes too, just in case
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

    elapsed = time.perf_counter() - t0
    return {
        "size_before":     src.stat().st_size,
        "size_after":      dst.stat().st_size,
        "elapsed_seconds": elapsed,
        "chunk_shape":     chunk_shape,
        "compression":     compression,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="src.h5_repack",
        description="Inspect or repack a training H5 file.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="Print dataset shapes, chunks, and warnings.")
    p_ins.add_argument("path", help="Path to the .h5 file")

    p_rep = sub.add_parser("repack", help="Rewrite with optimal chunking.")
    p_rep.add_argument("src", help="Source .h5 path")
    p_rep.add_argument("dst", help="Destination .h5 path")
    p_rep.add_argument("--chunk-rows", type=int, default=1,
                       help="Samples per chunk for 'images' (default: 1)")
    p_rep.add_argument("--compression", choices=_COMPRESSION_CHOICES, default="gzip",
                       help="Compression filter (default: gzip)")
    p_rep.add_argument("--compression-opts", type=int, default=None,
                       help="Gzip level 0-9 (default: 4)")
    p_rep.add_argument("--overwrite", action="store_true",
                       help="Overwrite dst if it exists")

    args = parser.parse_args(argv)

    if args.cmd == "inspect":
        info = inspect_h5(args.path)
        print(format_inspection(info))
        return 0

    if args.cmd == "repack":
        if os.path.abspath(args.src) == os.path.abspath(args.dst):
            print("ERROR: src and dst must differ.", file=sys.stderr)
            return 2
        result = repack_h5(
            args.src, args.dst,
            chunk_rows=args.chunk_rows,
            compression=args.compression,
            compression_opts=args.compression_opts,
            overwrite=args.overwrite,
        )
        print(
            f"Repacked {args.src} → {args.dst}\n"
            f"  chunk_shape : {result['chunk_shape']}\n"
            f"  compression : {result['compression']}\n"
            f"  size before : {_human_bytes(result['size_before'])}\n"
            f"  size after  : {_human_bytes(result['size_after'])}\n"
            f"  elapsed     : {result['elapsed_seconds']:.2f}s"
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
