import elevation as eio
import numpy as np
import pyvista as pv

from pyproj import CRS

from trimesh.ray.ray_triangle import RayMeshIntersector

import rasterio
import trimesh
from scipy.spatial import Delaunay


class RasterioDEM:
    def __init__(self, filepath, crs=None):
        self.filepath = filepath
        self.dataset = rasterio.open(filepath)
        self.crs = crs or CRS.from_string(self.dataset.crs.to_string())
        self.data = self.dataset.read(1)
        self.transform = self.dataset.transform
        self.pcd = None
        self.pcd_flat = None
        self.mesh = None

    def build_pcd(self, sample_step=1):
        rows, cols = self.data.shape
        x_idx = np.arange(0, cols, sample_step)
        y_idx = np.arange(0, rows, sample_step)

        xx, yy = np.meshgrid(x_idx, y_idx)
        xs, ys = rasterio.transform.xy(self.transform, yy, xx)
        xs = np.array(xs).reshape(xx.shape)
        ys = np.array(ys).reshape(yy.shape)
        zs = self.data[yy, xx]

        self.pcd = np.stack([xs, ys, zs], axis=-1)  # (H, W, 3)
        self.pcd_flat = self.pcd.reshape(-1, 3)
        self.xx, self.yy, self.zz = xs, ys, zs

    def build_mesh(self):
        if self.pcd is None:
            raise ValueError("Call build_pcd() first")

        grid = pv.StructuredGrid(
            self.pcd[:, :, 0],
            self.pcd[:, :, 1],
            self.pcd[:, :, 2]
        )
        self.mesh = grid.cast_to_poly_points().delaunay_2d()



    def cast_rays_seq(self, rays):
        ray_origins = rays[:, :, 0].reshape(-1, 3)
        ray_targets = rays[:, :, 1].reshape(-1, 3)
        ray_dirs = ray_targets - ray_origins
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)

        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError("Ray tracing requires mesh to be a trimesh.Trimesh object. Use build_mesh(method='trimesh').")

        ray_engine = RayMeshIntersector(self.mesh)
        locations, index_ray, _ = ray_engine.intersects_location(ray_origins, ray_dirs, multiple_hits=False)

        inter_points = np.full(ray_origins.shape, np.nan)
        inter_points[index_ray] = locations

        return inter_points.reshape(rays.shape[0], rays.shape[1], 3)
