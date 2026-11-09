"""Download and load Copernicus DEM GLO-30 DSM tiles."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import requests
from rasterio.merge import merge
from scipy.interpolate import RegularGridInterpolator

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "dem_cache"


def _tile_name(lat_int: int, lon_int: int) -> str:
    """Build the Copernicus DEM 30m COG tile name for a 1x1 degree cell."""
    lat_letter = "N" if lat_int >= 0 else "S"
    lon_letter = "E" if lon_int >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{lat_letter}{abs(lat_int):02d}_00_"
        f"{lon_letter}{abs(lon_int):03d}_00_DEM"
    )


def download_dem_tile(lat_int: int, lon_int: int, out_dir: str | Path) -> Path | None:
    """Download a single 1x1 degree Copernicus GLO-30 DSM tile (COG GeoTIFF).

    Cache both downloaded tiles and 404 responses. Return None for a missing tile.
    """
    out_dir = Path(out_dir)
    name = _tile_name(lat_int, lon_int)
    url = f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{name}/{name}.tif"
    local_path = out_dir / f"{name}.tif"

    if local_path.exists():
        return local_path
    missing = local_path.with_suffix(".missing")
    if missing.exists():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        if resp.status_code == 404:
            missing.touch()
            return None
        resp.raise_for_status()
        with TemporaryDirectory(dir=out_dir) as tmp:
            partial = Path(tmp) / local_path.name
            with partial.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            partial.replace(local_path)
    return local_path


def dsm_interpolator(
    elev: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> RegularGridInterpolator:
    """Bilinear interpolator over the DSM, queried with (lat, lon). NaN outside the grid."""
    if lats[0] > lats[-1]:  # RegularGridInterpolator needs ascending axes
        elev, lats = elev[::-1, :], lats[::-1]
    return RegularGridInterpolator(
        (lats, lons), elev, method="linear", bounds_error=False, fill_value=np.nan,
    )


def load_dsm(
    lat: float,
    lon: float,
    margin_deg: float = 0.05,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the DSM around (lat, lon) with a margin.

    Returns (elevations, lats, lons) where elevations is a 2-D array and
    lats/lons are 1-D coordinate vectors (descending lat, ascending lon).
    Missing tiles within the mosaic are filled with zero for coastal coverage.
    """
    cache = Path(cache_dir) if cache_dir else _CACHE_DIR

    lat_min = lat - margin_deg
    lat_max = lat + margin_deg
    lon_min = lon - margin_deg
    lon_max = lon + margin_deg

    tile_paths = []
    for la in range(math.floor(lat_min), math.floor(lat_max) + 1):
        for lo in range(math.floor(lon_min), math.floor(lon_max) + 1):
            tile = download_dem_tile(la, lo, cache)
            if tile is not None:
                tile_paths.append(tile)
    if not tile_paths:
        raise ValueError("No DSM tiles cover this area.")
    mosaic, win_transform = merge(
        tile_paths, bounds=(lon_min, lat_min, lon_max, lat_max), nodata=0
    )
    elev = mosaic[0]

    nrows, ncols = elev.shape
    cols = np.arange(ncols)
    rows = np.arange(nrows)
    lons = win_transform.c + (cols + 0.5) * win_transform.a
    lats = win_transform.f + (rows + 0.5) * win_transform.e

    return elev, lats, lons
