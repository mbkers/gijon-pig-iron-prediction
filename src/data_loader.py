"""Data loading utilities."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio


def load_production_data(base_path: str) -> pd.DataFrame:
    """Load monthly pig iron production data."""
    production_file = (
        Path(base_path) / "PigIronProduction" / "Pig_Iron_Production_Spain.csv.csv"
    )
    df = pd.read_csv(production_file)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def load_thermal_index(
    base_path: str, index_type: str, date: str
) -> Optional[npt.NDArray[np.float32]]:
    """Load thermal index for a specific date (YYYYMMDD format)."""
    index_file = (
        Path(base_path) / "Sentinel2" / index_type / f"SENTINEL2_L1C_{date}.tif"
    )
    if not index_file.exists():
        return None

    with rasterio.open(index_file) as src:
        return src.read(1)


def load_cloud_mask(base_path: str, date: str) -> Optional[npt.NDArray[np.int32]]:
    """Load cloud mask (0=clear, 1=thick cloud, 2=thin cloud, 3=shadow)."""
    cloud_file = (
        Path(base_path)
        / "Sentinel2"
        / "Cloud_Images"
        / f"SENTINEL2_L1C_{date}_cloudmask.tif"
    )
    if not cloud_file.exists():
        return None

    with rasterio.open(cloud_file) as src:
        return src.read(1)


def load_perimeter_mask(base_path: str) -> npt.NDArray[np.bool_]:
    """Load plant perimeter mask."""
    mask_file = Path(base_path) / "aois" / "subasset_location" / "Gijon_Perimeter.tif"
    with rasterio.open(mask_file) as src:
        return src.read(1) > 0


def get_available_dates(base_path: str, index_type: str = "TAI") -> List[str]:
    """Get list of available dates."""
    index_path = Path(base_path) / "Sentinel2" / index_type
    return sorted([f.stem.split("_")[-1] for f in index_path.glob("*.tif")])
