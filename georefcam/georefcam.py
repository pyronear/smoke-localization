import os

os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_COCOA"] = "false"  # 👈 disable macOS GUI backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import signal

from .camera_model import AbstractCameraModel
from .dem import RasterioDEM
from .data_types import (
    Coord3DFloatPoints,
    DfRayInstance,
    ImageArrayRGB,
    RayCoord3DFloatPoints,
)
from .logger import logger
from nptyping import assert_isinstance, Int, NDArray, Shape
from pathlib import Path
from PIL import Image
from pyproj import Transformer

# from transformers import pipeline
from typing import Literal


from pyproj import CRS, Transformer
import numpy as np


def project_points_to_crs(
    points: np.ndarray, from_crs: str | CRS, to_crs: str | CRS
) -> np.ndarray:
    # Convert to string for comparison
    if str(from_crs).lower() == str(to_crs).lower():
        return points  # ✅ Ne rien faire si les CRS sont identiques

    print("FROM CRS:", from_crs)
    print("TO CRS:", to_crs)
    print("Skip reprojection?", str(from_crs).lower() == str(to_crs).lower())

    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    xx, yy, zz = points[:, 0], points[:, 1], points[:, 2]
    pr_xx, pr_yy, pr_zz = transformer.transform(xx, yy, zz)
    result = np.stack((pr_xx, pr_yy, pr_zz), axis=1)

    if not np.all(np.isfinite(result)):
        invalid = np.where(~np.isfinite(result))
        print("[ERROR] Invalid input to transform:")
        print(points[invalid[0]])
        raise ValueError(f"[Reprojection Error] Non-finite coords at indices {invalid}")

    return result


def get_direction_vector(azimuth: float, pitch: float, length: float = 1.0):
    if not np.isfinite(azimuth) or not np.isfinite(pitch):
        raise ValueError(f"Invalid azimuth or pitch: {azimuth}, {pitch}")

    az_rad = np.radians(azimuth)
    pitch_rad = np.radians(pitch)

    dx = length * np.cos(pitch_rad) * np.sin(az_rad)
    dy = length * np.cos(pitch_rad) * np.cos(az_rad)
    dz = length * np.sin(pitch_rad)

    vec = np.array([dx, dy, dz])
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"Invalid direction vector: {vec}")
    return vec


def timeout_handler(signum, frame):
    raise TimeoutError


class GeoRefCam:
    def __init__(self, camera_model: AbstractCameraModel, dem: RasterioDEM):
        self.camera_model = camera_model
        self.dem = dem

    def project_points_from_cam_to_dem_crs(
        self, points: Coord3DFloatPoints
    ) -> Coord3DFloatPoints:
        return project_points_to_crs(points, self.camera_model.crs, self.dem.crs)

    def project_rays_from_cam_to_dem_crs(
        self, rays: RayCoord3DFloatPoints
    ) -> RayCoord3DFloatPoints:
        origins, destinations = rays[:, :, 0], rays[:, :, 1]
        proj_points = self.project_points_from_cam_to_dem_crs(
            np.vstack((origins, destinations))
        )
        return np.dstack((proj_points[: len(origins)], proj_points[len(origins) :]))

    def cast_rays(
        self, rays: RayCoord3DFloatPoints, check_crs: bool = True
    ) -> Coord3DFloatPoints:
        assert_isinstance(rays, RayCoord3DFloatPoints)
        if check_crs and self.camera_model.crs != self.dem.crs:
            rays = self.project_rays_from_cam_to_dem_crs(rays)
        return self.dem.cast_rays_seq(rays)
