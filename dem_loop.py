import os
import json
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from shapely.geometry import Point
from tqdm import tqdm

from georefcam.dem import RasterioDEM
from dem_utils import get_dem_proj


name = "haguenau"


# === Load Camera Metadata ===
with open(f"site_data/{name}/camera_config.json", "r") as f:
    loaded_config = json.load(f)

# Extract values
lat = loaded_config["lat"]
lon = loaded_config["lon"]

tilt = loaded_config["tilt"]
roll = loaded_config["roll"]
elevation = loaded_config["elevation"]
fov_x = loaded_config["fov_x"]
fov_y = loaded_config["fov_y"]
img_size = tuple(loaded_config["img_size"])

# Use name as login or define separately
cam_name = name

# === Camera Info as GeoDataFrame ===
cam_info = pd.Series({
    "login": cam_name,
    "lat": lat,
    "lon": lon,
    "geometry": Point(lon, lat),
    "azimuth": 0.0,
    "tilt": tilt,
    "elevation": elevation +10,
    "roll": roll,
})
gdf_cams = gpd.GeoDataFrame([cam_info], geometry="geometry", crs="EPSG:4326")

# === Prepare DEM ===
input_dsm_path = f"site_data/{name}/dsm_min.tif"
dem_data = rioxarray.open_rasterio(input_dsm_path, masked=True).squeeze()
if dem_data.rio.crs != "EPSG:2154":
    dem_data = dem_data.rio.reproject("EPSG:2154")
    dem_data.rio.to_raster(input_dsm_path)

dem = RasterioDEM(input_dsm_path, crs="EPSG:2154")
dem.build_pcd(sample_step=2)
dem.build_mesh()

print("[DEBUG] Mesh bounds:", dem.mesh.bounds)
print("[DEBUG] Mesh extents (X/Y):", dem.mesh.extents[:2])
print("[DEBUG] Mesh centroid:", dem.mesh.centroid)

# === Generate Projected DEM Views ===
output_folder = f"site_data/{name}/dem/max"
os.makedirs(output_folder, exist_ok=True)

for azimuth in tqdm(range(50, 180, 10)):
    dem_depth = get_dem_proj(dem, img_size, cam_info, azimuth, fov_x, fov_y)
    arr = dem_depth.copy()
    mask = arr == arr.max()

    arr[mask] = 0

    min_val = np.nanpercentile(arr[~mask], 20)
    max_val = np.nanpercentile(arr[~mask], 95)
    norm = (arr - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)

    result = np.zeros_like(arr, dtype="float32")
    result[~mask] = norm[~mask] * 255

    out_path = os.path.join(output_folder, f"{str(azimuth).zfill(3)}.jpg")
    cv2.imwrite(out_path, result.astype(np.uint8))


print("✅ DEM projections saved.")
