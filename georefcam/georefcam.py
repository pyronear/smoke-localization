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
from .data_types import Coord3DFloatPoints, DfRayInstance, ImageArrayRGB, RayCoord3DFloatPoints
from .logger import logger
from nptyping import assert_isinstance, Int, NDArray, Shape
from pathlib import Path
from PIL import Image
from pyproj import Transformer
# from transformers import pipeline
from typing import Literal


def project_points_to_crs(points: Coord3DFloatPoints, from_crs: str, to_crs: str) -> Coord3DFloatPoints:
    transformer = Transformer.from_crs(from_crs, to_crs)
    if points.ndim == 2:
        xx, yy, zz = [points[:, i] for i in range(3)]
    else:
        xx, yy, zz = points
    pr_xx, pr_yy, pr_zz = transformer.transform(xx, yy, zz)
    return np.vstack((pr_xx, pr_yy, pr_zz)).T


def get_direction_vector(azimuth: float, pitch: float, length: float = 1):
    return np.array([
        length * np.sin(np.radians(azimuth)),
        length * np.cos(np.radians(azimuth)),
        -length * np.sin(np.radians(pitch))
    ])


def timeout_handler(signum, frame):
    raise TimeoutError


class GeoRefCam:
    def __init__(self, camera_model: AbstractCameraModel, dem: RasterioDEM):
        self.camera_model = camera_model
        self.dem = dem

    def project_points_from_cam_to_dem_crs(self, points: Coord3DFloatPoints) -> Coord3DFloatPoints:
        return project_points_to_crs(points, self.camera_model.crs, self.dem.crs)

    def project_rays_from_cam_to_dem_crs(self, rays: RayCoord3DFloatPoints) -> RayCoord3DFloatPoints:
        origins, destinations = rays[:, :, 0], rays[:, :, 1]
        proj_points = self.project_points_from_cam_to_dem_crs(np.vstack((origins, destinations)))
        return np.dstack((proj_points[:len(origins)], proj_points[len(origins):]))

    def cast_rays(self, rays: RayCoord3DFloatPoints, check_crs: bool = True) -> Coord3DFloatPoints:
        assert_isinstance(rays, RayCoord3DFloatPoints)
        if check_crs and self.camera_model.crs != self.dem.crs:
            rays = self.project_rays_from_cam_to_dem_crs(rays)
        return self.dem.cast_rays_seq(rays)

    # def evaluate_ypr_correction(
    #     self,
    #     refcam_img_path: Path | str,
    #     model: str = "LiheYoung/depth-anything-small-hf",
    #     debug: bool = False,
    # ) -> np.ndarray:

    #     pipe = pipeline(task="depth-estimation", model=model)
    #     refcam_img = Image.open(refcam_img_path)
    #     img_depth = np.asarray(pipe(refcam_img)["depth"])
    #     img_depth = np.where(img_depth == 255, np.nan, img_depth)

    #     cam_dirvec = get_direction_vector(self.camera_model.yaw_deg, self.camera_model.pitch_deg)
    #     cam_loc = (
    #         self.project_points_from_cam_to_dem_crs(np.array([self.camera_model.cam_loc[:3]]))[0]
    #         if self.camera_model.cam_loc[3] != self.dem.crs else self.camera_model.cam_loc[:3]
    #     )

    #     camera = pv.Camera()
    #     camera.clipping_range = (30, 1e5)
    #     camera.position = cam_loc
    #     camera.focal_point = cam_loc + cam_dirvec
    #     camera.view_angle = self.camera_model.view_y_deg
    #     camera.up = (0, 0, 1)

    #     plot_pv_meshgrid = pv.StructuredGrid(*[self.dem.pcd[:, :, i] for i in range(3)])
    #     plot_pv_meshgrid["alt"] = self.dem.pcd[:, :, 2].ravel(order="F")

    #     plotter = pv.Plotter(window_size=refcam_img.size)
    #     plotter.camera = camera
    #     plotter.add_mesh(plot_pv_meshgrid, lighting=False)
    #     plotter.remove_scalar_bar()
    #     plotter.screenshot()

    #     dem_depth = -1 * plotter.get_image_depth()
    #     dem_depth = np.where(dem_depth <= 0, np.nan, dem_depth)

    #     if debug:
    #         plt.figure(figsize=(12, 5))
    #         plt.subplot(1, 2, 1)
    #         plt.title("Predicted depth")
    #         plt.imshow(img_depth, cmap='gray')
    #         plt.subplot(1, 2, 2)
    #         plt.title("DEM-rendered depth")
    #         plt.imshow(dem_depth, cmap='gray')
    #         plt.tight_layout()
    #         plt.show()

    #     return dem_depth
