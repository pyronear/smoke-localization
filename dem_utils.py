from shapely.geometry import box, Point
import geopandas as gpd
import subprocess
from pathlib import Path
import numpy as np
import rioxarray
from rasterio.enums import Resampling

def compute_dem_bbox_from_point(lat, lon, buffer_km=50):
    point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    point_proj = point_gdf.to_crs(epsg=3857)
    x, y = point_proj.geometry[0].x, point_proj.geometry[0].y

    buffered_bounds = box(
        x - buffer_km * 1000, y - buffer_km * 1000,
        x + buffer_km * 1000, y + buffer_km * 1000
    )
    buffered_geo = gpd.GeoSeries([buffered_bounds], crs="EPSG:3857").to_crs(epsg=4326).total_bounds
    return tuple(buffered_geo)

def download_dem_eio(bbox, out_path="srtm_dem.tif", product="SRTM1"):
    cmd = [
        "eio", "--product", product, "clip",
        "--bounds", *map(str, bbox),
        "--output", out_path
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("🚨 Command failed:")
        print(result.stderr)
        raise RuntimeError("DEM download failed")
    elif not Path(out_path).exists():
        raise FileNotFoundError("DEM file not found after eio execution")
    else:
        print(f"✅ DEM downloaded: {out_path}")

def reproject_and_save_dem(input_path, output_path, target_crs="EPSG:2154", nodata_val=-9999):
    dem_data = rioxarray.open_rasterio(input_path, masked=True).squeeze()

    if dem_data.rio.crs is None:
        dem_data.rio.write_crs("EPSG:4326", inplace=True)

    dem_l93 = dem_data.rio.reproject(
        target_crs,
        resampling=Resampling.bilinear,
        nodata=np.nan
    )

    dem_l93_int16 = dem_l93.fillna(nodata_val).round().astype(np.int16)
    dem_l93_int16.rio.write_nodata(nodata_val, inplace=True)
    dem_l93_int16.rio.write_crs(target_crs, inplace=True)
    dem_l93_int16.rio.to_raster(output_path, compress="LZW")

    print(f"✅ Reprojected DEM saved to {output_path}")
