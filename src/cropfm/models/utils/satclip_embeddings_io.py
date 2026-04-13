"""Load precomputed SatCLIP (or other) embeddings for EAM alignment.

Supports:

- ``*.npy`` — ``(N, D)`` float array, row ``i`` = ``zarr_sample_idx`` ``i`` (preferred for large files).
- ``*.csv`` — columns ``emb_0`` … ``emb_{D-1}`` as produced by ``analysis/compute_satclip_embeddings.py``,
  optionally ``zarr_sample_idx`` to join rows to global zarr indices.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def load_satclip_embedding_matrix(path: str | Path) -> np.ndarray:
    """
    Load a float32 matrix of shape ``(N, D)`` where row ``i`` is the embedding for
    global zarr sample index ``i`` (after optional ``zarr_sample_idx`` scatter).

    Args:
        path: Path to ``.npy`` or ``.csv``.

    Returns:
        Array suitable for indexing; ``.npy`` may be returned as a read-only memmap.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SatCLIP embeddings file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
        return arr

    if suffix in (".csv", ".txt"):
        return _load_satclip_embeddings_from_csv(path)

    raise ValueError(f"Unsupported embeddings format: {path} (use .npy or .csv)")


def _load_satclip_embeddings_from_csv(path: Path) -> np.ndarray:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Loading SatCLIP embeddings from CSV requires pandas. "
            "Install pandas or convert the file to a .npy matrix."
        ) from e

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 512:
        warnings.warn(
            f"SatCLIP CSV is large ({size_mb:.0f} MB). Prefer a single `.npy` of shape (N, D) "
            f"(``np.save`` / ``np.load(..., mmap_mode='r')``) for faster startup and lower RAM.",
            stacklevel=2,
        )

    header = pd.read_csv(path, nrows=0)
    emb_cols = sorted(
        [c for c in header.columns if c.startswith("emb_")],
        key=lambda c: int(c.split("_", 1)[1]),
    )
    if not emb_cols:
        raise ValueError(f"No emb_* columns found in {path}")

    usecols = emb_cols.copy()
    if "zarr_sample_idx" in header.columns:
        usecols.append("zarr_sample_idx")

    df = pd.read_csv(path, usecols=usecols, dtype=np.float32)
    emb = df[emb_cols].to_numpy(dtype=np.float32, copy=False)

    if "zarr_sample_idx" in df.columns:
        idx = df["zarr_sample_idx"].to_numpy(dtype=np.int64, copy=False)
        n_rows = int(idx.max()) + 1
        out = np.zeros((n_rows, emb.shape[1]), dtype=np.float32)
        out[idx] = emb
        return out

    return emb
