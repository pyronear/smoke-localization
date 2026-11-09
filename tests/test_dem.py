from io import BytesIO

import numpy as np
import pytest
import rasterio
import requests
from rasterio.transform import from_bounds

from smoke_localisation.dem import _tile_name, download_dem_tile, load_dsm


def test_coastal_mosaic_fills_missing_tiles_and_caches_404(monkeypatch, tmp_path):
    land = tmp_path / f"{_tile_name(0, 0)}.tif"
    with rasterio.open(
        land,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, 10, 10),
    ) as dataset:
        dataset.write(np.full((10, 10), 123, dtype="float32"), 1)

    requested = []

    def missing_tile(url, **kwargs):
        requested.append(url)
        response = requests.Response()
        response.status_code = 404
        response.raw = BytesIO()
        return response

    monkeypatch.setattr(requests, "get", missing_tile)
    elev, lats, lons = load_dsm(0.5, 0.5, margin_deg=0.7, cache_dir=tmp_path)
    assert elev.shape == (14, 14)
    assert np.all(elev[np.ix_((lats > 0) & (lats < 1), (lons > 0) & (lons < 1))] == 123)
    assert np.all(elev[lats < 0] == 0)
    assert len(requested) == 8
    load_dsm(0.5, 0.5, margin_deg=0.7, cache_dir=tmp_path)
    assert len(requested) == 8


def test_other_http_errors_are_not_cached(monkeypatch, tmp_path):
    response = requests.Response()
    response.status_code = 503
    response.raw = BytesIO()
    monkeypatch.setattr(requests, "get", lambda *a, **kw: response)
    with pytest.raises(requests.HTTPError):
        download_dem_tile(0, 0, tmp_path)
    assert not list(tmp_path.iterdir())
