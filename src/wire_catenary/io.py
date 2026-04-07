"""Load LiDAR point clouds from parquet files."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_points(path: Path | str) -> np.ndarray:
    """Read x/y/z columns from a parquet file into an (N, 3) float64 array.

    NaN rows are dropped. Raises ValueError if x/y/z columns are missing.
    """
    path = Path(path)

    df = pd.read_parquet(path)

    required = {"x", "y", "z"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    arr = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return arr[~np.any(np.isnan(arr), axis=1)]