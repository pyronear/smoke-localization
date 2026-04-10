"""Download and load Copernicus DEM GLO-30 DSM tiles."""

import math
from pathlib import Path

import numpy as np
import rasterio
import requests

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "dem_cache"


def _tile_name(lat_int: int, lon_int: int) -> str:
    """Build the Copernicus DEM 30m COG tile name for a 1x1 degree cell."""
    lat_letter = "N" if lat_int >= 0 else "S"
    lon_letter = "E" if lon_int >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{lat_letter}{abs(lat_int):02d}_00_"
        f"{lon_letter}{abs(lon_int):03d}_00_DEM"
    )


def download_dem_tile(lat_int: int, lon_int: int, out_dir: str | Path) -> Path:
    """Download a single 1x1 degree Copernicus GLO-30 DSM tile (COG GeoTIFF).

    Tiles are cached on disk so subsequent calls are free.
    """
    out_dir = Path(out_dir)
    name = _tile_name(lat_int, lon_int)
    url = f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{name}/{name}.tif"
    local_path = out_dir / f"{name}.tif"

    if local_path.exists():
        return local_path

    out_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return local_path


def load_dsm(
    lat: float,
    lon: float,
    margin_deg: float = 0.05,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the DSM around (lat, lon) with a margin.

    Returns (elevations, lats, lons) where elevations is a 2-D array and
    lats/lons are 1-D coordinate vectors (descending lat, ascending lon).
    """
    cache = Path(cache_dir) if cache_dir else _CACHE_DIR

    lat_min = lat - margin_deg
    lat_max = lat + margin_deg
    lon_min = lon - margin_deg
    lon_max = lon + margin_deg

    needed_tiles = set()
    for la in range(math.floor(lat_min), math.floor(lat_max) + 1):
        for lo in range(math.floor(lon_min), math.floor(lon_max) + 1):
            needed_tiles.add((la, lo))

    tile_paths = [download_dem_tile(la, lo, cache) for la, lo in sorted(needed_tiles)]

    # Read & merge tiles
    datasets = [rasterio.open(p) for p in tile_paths]
    if len(datasets) == 1:
        ds = datasets[0]
        from rasterio.windows import from_bounds
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, ds.transform)
        elev = ds.read(1, window=window)
        win_transform = ds.window_transform(window)
    else:
        from rasterio.merge import merge
        merged, merged_transform = merge(datasets)
        elev = merged[0]
        res = merged_transform.a
        col_off = int((lon_min - merged_transform.c) / res)
        row_off = int((merged_transform.f - lat_max) / (-merged_transform.e))
        ncols_crop = int((lon_max - lon_min) / res)
        nrows_crop = int((lat_max - lat_min) / (-merged_transform.e))
        elev = elev[row_off:row_off + nrows_crop, col_off:col_off + ncols_crop]
        win_transform = rasterio.transform.from_bounds(
            lon_min, lat_min, lon_max, lat_max, ncols_crop, nrows_crop
        )

    nrows, ncols = elev.shape
    cols = np.arange(ncols)
    rows = np.arange(nrows)
    lons = win_transform.c + (cols + 0.5) * win_transform.a
    lats = win_transform.f + (rows + 0.5) * win_transform.e

    for d in datasets:
        d.close()

    return elev, lats, lons
