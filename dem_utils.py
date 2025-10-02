from shapely.geometry import box, Point
import geopandas as gpd
import subprocess
from pathlib import Path
import numpy as np
import rioxarray
from rasterio.enums import Resampling
from georefcam.georefcam import GeoRefCam, get_direction_vector
from pyproj import CRS
from georefcam.camera_model import CTCameraModel
import pyvista as pv


def compute_dem_bbox_from_point(lat, lon, buffer_km=50):
    point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    point_proj = point_gdf.to_crs(epsg=3857)
    x, y = point_proj.geometry[0].x, point_proj.geometry[0].y

    buffered_bounds = box(
        x - buffer_km * 1000,
        y - buffer_km * 1000,
        x + buffer_km * 1000,
        y + buffer_km * 1000,
    )
    buffered_geo = (
        gpd.GeoSeries([buffered_bounds], crs="EPSG:3857").to_crs(epsg=4326).total_bounds
    )
    return tuple(buffered_geo)


def download_dem_eio(bbox, out_path="srtm_dem.tif", product="SRTM1"):
    cmd = [
        "eio",
        "--product",
        product,
        "clip",
        "--bounds",
        *map(str, bbox),
        "--output",
        out_path,
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


def reproject_and_save_dem(
    input_path, output_path, target_crs="EPSG:2154", nodata_val=-9999
):
    dem_data = rioxarray.open_rasterio(input_path, masked=True).squeeze()

    if dem_data.rio.crs is None:
        dem_data.rio.write_crs("EPSG:4326", inplace=True)

    dem_l93 = dem_data.rio.reproject(
        target_crs, resampling=Resampling.bilinear, nodata=np.nan
    )

    dem_l93_int16 = dem_l93.fillna(nodata_val).round().astype(np.int16)
    dem_l93_int16.rio.write_nodata(nodata_val, inplace=True)
    dem_l93_int16.rio.write_crs(target_crs, inplace=True)
    dem_l93_int16.rio.to_raster(output_path, compress="LZW")

    print(f"✅ Reprojected DEM saved to {output_path}")


def get_dem_proj(dem, img_size, cam_info, azimuth, fov_x, fov_y):

    cam_crs = CRS.from_user_input("EPSG:2154")

    # === Step 1: Instantiate your camera model ===
    cam_model = CTCameraModel(
        image_res=img_size,
        fov_x_deg=fov_x,
        fov_y_deg=fov_y,
        azimuth_deg=azimuth,
        tilt_deg=cam_info.tilt,
        roll_deg=cam_info.roll,
        lat=cam_info.lat,
        lon=cam_info.lon,
        elevation=cam_info.elevation,
        crs=cam_crs.to_string(),
    )

    # === Step 1: Compute camera location and direction ===
    geocam = GeoRefCam(cam_model, dem)

    cam_dirvec = get_direction_vector(
        geocam.camera_model.azimuth_deg, geocam.camera_model.tilt_deg
    )

    # Reproject camera location if needed
    if geocam.camera_model.cam_loc[3] != geocam.dem.crs:
        cam_loc = geocam.project_points_from_cam_to_dem_crs(
            np.array([geocam.camera_model.cam_loc[:3]])
        )[0]
    else:
        cam_loc = geocam.camera_model.cam_loc[:3]

    # === Step 1: Build GeoRefCam ===
    geocam = GeoRefCam(cam_model, dem)

    # === Step 2: Camera position and direction ===
    cam_dirvec = get_direction_vector(
        geocam.camera_model.azimuth_deg, geocam.camera_model.tilt_deg
    )

    if geocam.camera_model.cam_loc[3] != geocam.dem.crs:
        cam_loc = geocam.project_points_from_cam_to_dem_crs(
            np.array([geocam.camera_model.cam_loc[:3]])
        )[0]
    else:
        cam_loc = geocam.camera_model.cam_loc[:3]

    # === Step 3: Compute vertical FOV ===
    w, h = geocam.camera_model.image_res
    fov_y_deg = 2 * np.degrees(
        np.arctan(np.tan(np.radians(geocam.camera_model.fov_x_deg / 2)) * (h / w))
    )
    # === Step 4: Setup PyVista camera ===
    pv_camera = pv.Camera()
    pv_camera.position = cam_loc
    pv_camera.focal_point = cam_loc + cam_dirvec
    pv_camera.view_angle = fov_y_deg
    pv_camera.clipping_range = (1, 10000)
    pv_camera.up = (0, 0, 1)

    # === Step 5: PyVista structured grid ===
    z = geocam.dem.pcd[:, :, 2]
    z[z < -1000] = np.nan
    geocam.dem.pcd[:, :, 2] = z

    grid = pv.StructuredGrid(*[geocam.dem.pcd[:, :, i] for i in range(3)])
    grid["alt"] = z.ravel(order="F")

    # === Step 6: Remove invalid (water) points
    valid_grid = grid.threshold(value=-999, scalars="alt", invert=False)

    # === Step 7: Render
    plotter = pv.Plotter(off_screen=True, window_size=(w, h))
    plotter.camera = pv_camera
    plotter.add_mesh(valid_grid, scalars="alt", lighting=False)
    plotter.remove_scalar_bar()

    # ✅ Must render manually before accessing depth
    plotter.screenshot()

    # Step 9: Capture depth map
    dem_depth = -1 * plotter.get_image_depth()
    dem_depth = np.nan_to_num(dem_depth, nan=1e6)

    arr = dem_depth.copy()
    mask = arr == arr.max()

    arr[mask] = 0
    min_val = np.nanpercentile(arr[~mask], 0)
    max_val = np.nanpercentile(arr[~mask], 98)

    norm = (arr - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)

    result = np.zeros_like(arr, dtype="float32")
    result[~mask] = norm[~mask] * 255

    return result
